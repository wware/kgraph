"""Relationship extraction from journal articles.

Extracts relationships from Paper JSON format (from med-lit-schema).
Since the papers already have extracted relationships, we convert those to BaseRelationship objects.
Can also use Ollama LLM for relationship extraction from text.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence, Any

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity
from kgraph.pipeline.interfaces import RelationshipExtractorInterface
from kgraph.relationship import BaseRelationship

from ..relationships import MedicalClaimRelationship
from .llm_client import LLMClientInterface


class MedLitRelationshipExtractor(RelationshipExtractorInterface):
    """Extract relationships from journal articles.

    This extractor works with Paper JSON format from med-lit-schema, which
    already contains extracted relationships. We convert those to BaseRelationship objects.

    Can also use Ollama LLM to extract relationships directly from text if llm_client is provided.
    """

    def __init__(self, llm_client: Optional[LLMClientInterface] = None):
        """Initialize relationship extractor.

        Args:
            llm_client: Optional LLM client for extracting relationships from text.
                        If None, only uses pre-extracted relationships from Paper JSON.
        """
        self._llm = llm_client

    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        """Extract relationships from a journal article.

        If the document metadata contains pre-extracted relationships (from med-lit-schema),
        we convert those to BaseRelationship objects. Otherwise, if llm_client is provided,
        extracts relationships from document text using LLM.

        Args:
            document: The journal article document.
            entities: The resolved entities from this document.

        Returns:
            List of BaseRelationship objects representing relationships in the paper.
        """
        relationships: list[BaseRelationship] = []

        # Check if document has pre-extracted relationships in metadata
        # (from med-lit-schema's Paper format)
        relationships_data = document.metadata.get("relationships", [])

        if relationships_data:
            # Convert pre-extracted relationships to BaseRelationship objects
            entity_by_id: dict[str, BaseEntity] = {e.entity_id: e for e in entities}

            for rel_data in relationships_data:
                # Handle both dict format and AssertedRelationship format
                if isinstance(rel_data, dict):
                    subject_id = rel_data.get("subject_id", "")
                    predicate = rel_data.get("predicate", "")
                    object_id = rel_data.get("object_id", "")
                    confidence = rel_data.get("confidence", 0.5)
                    evidence = rel_data.get("evidence", "")
                    section = rel_data.get("section", "")

                    # Validate that entities exist
                    if subject_id not in entity_by_id or object_id not in entity_by_id:
                        # Skip relationships where entities weren't resolved
                        continue

                    # Normalize predicate (lowercase, handle enum values)
                    if isinstance(predicate, str):
                        predicate = predicate.lower()
                    else:
                        # If it's an enum, get its value
                        predicate = str(predicate).lower()

                    # Create relationship with evidence and provenance in metadata
                    metadata: dict[str, Any] = {
                        "evidence": evidence,
                        "section": section,
                    }

                    # Add any additional metadata from the relationship
                    if "metadata" in rel_data and isinstance(rel_data["metadata"], dict):
                        metadata.update(rel_data["metadata"])

                    relationship = MedicalClaimRelationship(
                        subject_id=subject_id,
                        predicate=predicate,
                        object_id=object_id,
                        confidence=float(confidence),
                        source_documents=(document.document_id,),
                        created_at=datetime.now(timezone.utc),
                        last_updated=None,
                        metadata=metadata,
                    )

                    relationships.append(relationship)
            return relationships

        # No pre-extracted relationships - try LLM extraction if available
        if self._llm and entities and document.content:
            try:
                print(f"  Extracting relationships with LLM from {len(entities)} entities...")
                extracted = await self._extract_with_llm(document, entities)
                relationships.extend(extracted)
                print(f"  LLM extracted {len(extracted)} relationships")
            except Exception as e:
                print(f"Warning: LLM relationship extraction failed: {e}")
                import traceback

                traceback.print_exc()

        return relationships

    async def _extract_with_llm(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        """Extract relationships using LLM."""
        if not self._llm:
            return []

        # Build entity context for LLM
        entity_by_name: dict[str, BaseEntity] = {e.name.lower(): e for e in entities}
        entity_list = "\n".join(f"- {e.name} ({e.get_entity_type()}): {e.entity_id}" for e in entities[:50])  # Limit to avoid huge prompts

        # Use abstract for extraction (shorter, more focused)
        text = document.abstract if hasattr(document, "abstract") and document.abstract else document.content
        text_sample = text[:2000]  # Limit text size for faster processing

        prompt = f"""Extract medical relationships from the following text.

Entities mentioned in the document:
{entity_list}

Text:
{text_sample}

Extract relationships in JSON format:
[
  {{"subject": "entity_name", "predicate": "treats", "object": "entity_name", "confidence": 0.9, "evidence": "sentence text"}},
  ...
]

Valid predicates: treats, causes, increases_risk, prevents, inhibits, associated_with, interacts_with, diagnosed_by, indicates

Return ONLY the JSON array, no explanation."""

        try:
            response = await self._llm.generate_json(prompt)
            relationships: list[BaseRelationship] = []

            if isinstance(response, list):
                for item in response:
                    if isinstance(item, dict):
                        subject_name = item.get("subject", "").strip()
                        predicate = item.get("predicate", "").lower()
                        object_name = item.get("object", "").strip()
                        confidence = float(item.get("confidence", 0.5))
                        evidence = item.get("evidence", "")

                        # Find entities by name
                        subject_entity = entity_by_name.get(subject_name.lower())
                        object_entity = entity_by_name.get(object_name.lower())

                        if subject_entity and object_entity:
                            rel = MedicalClaimRelationship(
                                subject_id=subject_entity.entity_id,
                                predicate=predicate,
                                object_id=object_entity.entity_id,
                                confidence=confidence,
                                source_documents=(document.document_id,),
                                created_at=datetime.now(timezone.utc),
                                last_updated=None,
                                metadata={
                                    "evidence": evidence,
                                    "extraction_method": "llm",
                                },
                            )
                            relationships.append(rel)

            return relationships

        except Exception as e:
            print(f"Warning: LLM relationship extraction failed: {e}")
            return []
