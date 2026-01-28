"""Relationship extraction from journal articles.

Extracts relationships from Paper JSON format (from med-lit-schema).
Since the papers already have extracted relationships, we convert those to BaseRelationship objects.
Can also use Ollama LLM for relationship extraction from text.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence, Any

from kgraph.document import BaseDocument
from kgraph.domain import Evidence, Provenance
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

    def _validate_predicate_semantics(self, predicate: str, evidence: str) -> bool:
        """Validate that predicate semantics match the evidence text.

        Checks for semantic mismatches like:
        - "increases_risk" with positive therapeutic language
        - "treats" with negative/harmful language

        Args:
            predicate: The relationship predicate (e.g., "treats", "increases_risk")
            evidence: The evidence text supporting the relationship

        Returns:
            True if predicate matches evidence semantics, False if there's a mismatch.
        """
        if not evidence:
            return True  # No evidence to validate against

        evidence_lower = evidence.lower()

        # Positive therapeutic language
        positive_words = {
            "therapy",
            "therapeutic",
            "treatment",
            "treats",
            "treating",
            "beneficial",
            "benefit",
            "improves",
            "improvement",
            "reduces",
            "reduction",
            "relieves",
            "relief",
            "effective",
            "efficacy",
            "efficacious",
            "cures",
            "curing",
            "heals",
            "healing",
            "complementary therapy",
            "potential",
            "promising",
        }

        # Negative/harmful language
        negative_words = {
            "risk",
            "risks",
            "increases risk",
            "raises risk",
            "elevates risk",
            "causes",
            "causing",
            "harmful",
            "harm",
            "worsens",
            "worsening",
            "adverse",
            "adversely",
            "damages",
            "damage",
            "detrimental",
            "predisposes",
            "predisposition",
            "leads to",
            "results in",
        }

        has_positive = any(word in evidence_lower for word in positive_words)
        has_negative = any(word in evidence_lower for word in negative_words)

        # Check for semantic mismatches
        if predicate == "increases_risk":
            # "increases_risk" should have negative language, not positive
            if has_positive and not has_negative:
                return False  # Evidence is positive but predicate is negative

        elif predicate == "treats":
            # "treats" should have positive language, not negative
            if has_negative and not has_positive:
                return False  # Evidence is negative but predicate is positive

        # Other predicates are less strict, but we can add more checks if needed
        return True

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
                    evidence_text = rel_data.get("evidence", "")
                    section = rel_data.get("section", "")
                    paragraph = rel_data.get("paragraph")  # Optional paragraph index

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

                    # Create structured provenance
                    provenance = Provenance(
                        document_id=document.document_id,
                        source_uri=document.source_uri,
                        section=section if section else None,
                        paragraph=paragraph if paragraph is not None else None,
                    )

                    # Create structured evidence
                    evidence_obj = Evidence(
                        kind="extracted",
                        source_documents=(document.document_id,),
                        primary=provenance,
                        mentions=(provenance,),
                        notes={"evidence_text": evidence_text} if evidence_text else {},
                    )

                    # Add any additional metadata from the relationship
                    metadata: dict[str, Any] = {}
                    if "metadata" in rel_data and isinstance(rel_data["metadata"], dict):
                        metadata.update(rel_data["metadata"])

                    relationship = MedicalClaimRelationship(
                        subject_id=subject_id,
                        predicate=predicate,
                        object_id=object_id,
                        confidence=float(confidence),
                        source_documents=(document.document_id,),
                        evidence=evidence_obj,
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

Predicate definitions (CRITICAL - use the correct predicate based on evidence meaning):
- treats: Drug/therapy REDUCES, CURES, or IMPROVES disease (positive therapeutic effect)
- increases_risk: Factor CAUSES, WORSENS, or RAISES LIKELIHOOD of disease (negative/harmful effect)
- prevents: Intervention STOPS disease from occurring (protective/preventive)
- causes: Direct causation where one entity leads to another
- inhibits: Entity blocks or reduces activity of another entity
- associated_with: General correlation without clear direction (neutral)
- interacts_with: Two entities affect each other (drug-drug, drug-gene interactions)
- diagnosed_by: Disease is identified using a test/procedure/biomarker
- indicates: Sign/symptom/test result suggests presence of disease

IMPORTANT:
- "treats" and "increases_risk" are OPPOSITE in meaning
- If evidence mentions "therapy", "therapeutic", "beneficial", "reduces", "improves" → use "treats" NOT "increases_risk"
- If evidence mentions "risk", "causes", "harmful", "worsens" → use "increases_risk" NOT "treats"

Entities mentioned in the document:
{entity_list}

Text:
{text_sample}

Extract relationships in JSON format:
[
  {{"subject": "entity_name", "predicate": "treats", "object": "entity_name", "confidence": 0.9, "evidence": "sentence text"}},
  ...
]

Examples:
{{"subject": "aspirin", "predicate": "treats", "object": "headache", "confidence": 0.9, "evidence": "Aspirin effectively relieves headache pain"}}
{{"subject": "smoking", "predicate": "increases_risk", "object": "lung cancer", "confidence": 0.95, "evidence": "Smoking raises the risk of developing lung cancer"}}
{{"subject": "metformin", "predicate": "treats", "object": "diabetes", "confidence": 0.85, "evidence": "Metformin is an effective therapy for type 2 diabetes"}}

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
                            # Validate predicate semantics match evidence
                            if not self._validate_predicate_semantics(predicate, evidence):
                                print(f"  Warning: Semantic mismatch - predicate '{predicate}' " f"does not match evidence: {evidence[:100]}...")
                                continue  # Skip this relationship

                            # Create structured provenance for LLM extraction
                            # Note: LLM extracts from abstract/sample, so we don't have precise offsets
                            provenance = Provenance(
                                document_id=document.document_id,
                                source_uri=document.source_uri,
                                section="abstract" if hasattr(document, "abstract") and document.abstract else None,
                            )

                            # Create structured evidence
                            evidence_obj = Evidence(
                                kind="llm_extracted",
                                source_documents=(document.document_id,),
                                primary=provenance,
                                mentions=(provenance,),
                                notes={"evidence_text": evidence, "extraction_method": "llm"},
                            )

                            rel = MedicalClaimRelationship(
                                subject_id=subject_entity.entity_id,
                                predicate=predicate,
                                object_id=object_entity.entity_id,
                                confidence=confidence,
                                source_documents=(document.document_id,),
                                evidence=evidence_obj,
                                created_at=datetime.now(timezone.utc),
                                last_updated=None,
                                metadata={"extraction_method": "llm"},
                            )
                            relationships.append(rel)

            return relationships

        except Exception as e:
            print(f"Warning: LLM relationship extraction failed: {e}")
            return []
