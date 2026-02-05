"""Relationship extraction from journal articles.

Extracts relationships from Paper JSON format (from med-lit-schema).
Since the papers already have extracted relationships, we convert those to BaseRelationship objects.
Can also use Ollama LLM for relationship extraction from text.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from kgraph.pipeline.interfaces import RelationshipExtractorInterface

from kgschema.document import BaseDocument
from kgschema.domain import Evidence, Provenance
from kgschema.entity import BaseEntity
from kgschema.relationship import BaseRelationship

from ..relationships import MedicalClaimRelationship
from .llm_client import LLMClientInterface

# Default directory for relationship extraction trace files
DEFAULT_TRACE_DIR = Path("/tmp/kgraph-relationship-traces")

if TYPE_CHECKING:
    from ..domain import MedLitDomainSchema


class MedLitRelationshipExtractor(RelationshipExtractorInterface):
    """Extract relationships from journal articles.

    This extractor works with Paper JSON format from med-lit-schema, which
    already contains extracted relationships. We convert those to BaseRelationship objects.

    Can also use Ollama LLM to extract relationships directly from text if llm_client is provided.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClientInterface] = None,
        domain: Optional["MedLitDomainSchema"] = None,
        trace_dir: Optional[Path] = None,
    ):
        """Initialize relationship extractor.

        Args:
            llm_client: Optional LLM client for extracting relationships from text.
                        If None, only uses pre-extracted relationships from Paper JSON.
            domain: Optional domain schema for type validation and predicate constraints.
                    If provided, will attempt to swap subject/object on type mismatches.
            trace_dir: Optional directory for writing trace files. If None, uses default.
        """
        self._llm = llm_client
        self._trace_dir = trace_dir or DEFAULT_TRACE_DIR
        if domain is None:
            # Import at runtime to avoid circular import
            from ..domain import MedLitDomainSchema

            domain = MedLitDomainSchema()
        self._domain = domain

    @property
    def trace_dir(self) -> Path:
        """Get the trace directory."""
        return self._trace_dir

    @trace_dir.setter
    def trace_dir(self, value: Path) -> None:
        """Set the trace directory."""
        self._trace_dir = value

    def _should_swap_subject_object(self, predicate: str, subject_entity: BaseEntity, object_entity: BaseEntity) -> bool:
        """Check if subject and object should be swapped based on type constraints.

        Args:
            predicate: The relationship predicate
            subject_entity: The subject entity
            object_entity: The object entity

        Returns:
            True if swapping subject and object would satisfy type constraints, False otherwise.
        """
        if predicate not in self._domain.predicate_constraints:
            return False  # No constraints to check

        constraints = self._domain.predicate_constraints[predicate]
        subject_type = subject_entity.get_entity_type()
        object_type = object_entity.get_entity_type()

        # Check if current order violates constraints
        subject_valid = subject_type in constraints.subject_types
        object_valid = object_type in constraints.object_types

        if subject_valid and object_valid:
            return False  # Current order is fine

        # Check if swapping would fix it
        swapped_subject_valid = object_type in constraints.subject_types
        swapped_object_valid = subject_type in constraints.object_types

        return swapped_subject_valid and swapped_object_valid

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
        """Extract relationships using LLM.

        Also writes a trace file to /tmp/kgraph-relationship-traces/ for debugging.
        The trace captures: prompt, raw LLM output, parsed JSON, and per-item decisions.
        """
        if not self._llm:
            return []

        # Initialize trace for debugging
        trace: dict[str, Any] = {
            "document_id": document.document_id,
            "source_uri": getattr(document, "source_uri", None),
            "llm_model": getattr(self._llm, "model", None),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "entity_count": len(entities),
            "prompt": None,
            "raw_llm_output": None,
            "parsed_json": None,
            "decisions": [],
            "final_relationships": [],
            "error": None,
        }

        # Build entity context for LLM
        entity_by_name: dict[str, BaseEntity] = {e.name.lower(): e for e in entities}
        entity_list = "\n".join(f"- {e.name} ({e.get_entity_type()}): {e.entity_id}" for e in entities[:50])  # Limit to avoid huge prompts

        # Use abstract for extraction (shorter, more focused)
        text = document.abstract if hasattr(document, "abstract") and document.abstract else document.content
        text_sample = text[:2000]  # Limit text size for faster processing

        prompt = f"""Extract medical relationships from the text below.

PREDICATE TYPE RULES (check entity types carefully):
• treats: ONLY drug/procedure → disease/symptom (drug treats disease, NOT disease treats disease)
• increases_risk: gene/ethnicity → disease (gene increases risk of disease)
• prevents: drug/procedure → disease
• targets: ONLY drug/procedure → gene/protein (drug targets protein, NOT gene targets disease)
• diagnosed_by: disease → procedure/biomarker (disease diagnosed by test, NOT test diagnosed by disease)
• associated_with: use for general correlations, disease subtypes, gene-disease links, or when other predicates don't fit

Entities in document:
{entity_list}

Text:
{text_sample}

Return JSON array of relationships (extract ALL valid relationships you find):
[
  {{"subject": "entity_name", "predicate": "treats", "object": "entity_name", "confidence": 0.9, "evidence": "quote from text"}},
  ...
]

Valid examples:
{{"subject": "paclitaxel", "predicate": "treats", "object": "breast cancer", "confidence": 0.9, "evidence": "Paclitaxel therapy for breast cancer"}}
{{"subject": "trastuzumab", "predicate": "targets", "object": "HER2", "confidence": 0.95, "evidence": "Trastuzumab targets HER2 protein"}}
{{"subject": "BRCA1", "predicate": "increases_risk", "object": "breast cancer", "confidence": 0.9, "evidence": "BRCA1 mutations increase breast cancer risk"}}
{{"subject": "HER2", "predicate": "associated_with", "object": "breast cancer", "confidence": 0.85, "evidence": "HER2 overexpression in breast cancer"}}
{{"subject": "luminal A", "predicate": "associated_with", "object": "breast cancer", "confidence": 0.85, "evidence": "Luminal A subtype of breast cancer"}}

INVALID (skip these):
- {{"subject": "breast cancer", "predicate": "treats", ...}} - disease cannot treat
- {{"subject": "HER2", "predicate": "targets", "object": "breast cancer", ...}} - gene cannot target disease, use "associated_with"
- {{"subject": "therapy", "predicate": "increases_risk", ...}} - therapy doesn't increase risk

Return ONLY the JSON array."""

        trace["prompt"] = prompt

        try:
            # Use generate_json_with_raw to capture both parsed and raw output
            if hasattr(self._llm, "generate_json_with_raw"):
                response, raw_output = await self._llm.generate_json_with_raw(prompt)
                trace["raw_llm_output"] = raw_output
            else:
                response = await self._llm.generate_json(prompt)
                trace["raw_llm_output"] = "<raw unavailable - method not supported>"

            trace["parsed_json"] = response
            relationships: list[BaseRelationship] = []

            if isinstance(response, list):
                for item in response:
                    if isinstance(item, dict):
                        subject_name = item.get("subject", "").strip()
                        predicate = item.get("predicate", "").lower()
                        object_name = item.get("object", "").strip()
                        confidence = float(item.get("confidence", 0.5))
                        evidence = item.get("evidence", "")

                        # Initialize decision record for this item
                        decision: dict[str, Any] = {
                            "item": item,
                            "subject_name": subject_name,
                            "object_name": object_name,
                            "predicate": predicate,
                            "confidence": confidence,
                            "matched_subject": False,
                            "matched_object": False,
                            "semantic_ok": None,
                            "swap_applied": False,
                            "accepted": False,
                            "drop_reason": None,
                            "resolved": {
                                "subject_id": None,
                                "subject_type": None,
                                "object_id": None,
                                "object_type": None,
                            },
                        }

                        # Find entities by name
                        subject_entity = entity_by_name.get(subject_name.lower())
                        object_entity = entity_by_name.get(object_name.lower())

                        decision["matched_subject"] = subject_entity is not None
                        decision["matched_object"] = object_entity is not None

                        if not subject_entity or not object_entity:
                            # Record why we're dropping this relationship
                            if not subject_entity and not object_entity:
                                decision["drop_reason"] = "subject_and_object_unmatched"
                            elif not subject_entity:
                                decision["drop_reason"] = "subject_unmatched"
                            else:
                                decision["drop_reason"] = "object_unmatched"
                            trace["decisions"].append(decision)
                            continue

                        # Validate predicate semantics match evidence
                        semantic_ok = self._validate_predicate_semantics(predicate, evidence)
                        decision["semantic_ok"] = semantic_ok
                        if not semantic_ok:
                            print(f"  Warning: Semantic mismatch - predicate '{predicate}' " f"does not match evidence: {evidence[:100]}...")
                            decision["drop_reason"] = "semantic_mismatch"
                            trace["decisions"].append(decision)
                            continue

                        # Check if we need to swap subject and object based on type constraints
                        if self._should_swap_subject_object(predicate, subject_entity, object_entity):
                            print(
                                f"  Swapping subject/object for predicate '{predicate}': "
                                f"({subject_entity.name} [{subject_entity.get_entity_type()}] ↔ "
                                f"{object_entity.name} [{object_entity.get_entity_type()}])"
                            )
                            # Swap the entities
                            subject_entity, object_entity = object_entity, subject_entity
                            decision["swap_applied"] = True

                        # Record resolved entity info
                        decision["resolved"] = {
                            "subject_id": subject_entity.entity_id,
                            "subject_type": subject_entity.get_entity_type(),
                            "object_id": object_entity.entity_id,
                            "object_type": object_entity.get_entity_type(),
                        }

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

                        decision["accepted"] = True
                        trace["decisions"].append(decision)

            # Record final relationships in trace
            trace["final_relationships"] = [
                {
                    "subject_id": r.subject_id,
                    "predicate": r.predicate,
                    "object_id": r.object_id,
                    "confidence": getattr(r, "confidence", None),
                }
                for r in relationships
            ]

            # Write trace file
            self._write_trace(document.document_id, trace)

            return relationships

        except Exception as e:
            print(f"Warning: LLM relationship extraction failed: {e}")
            trace["error"] = repr(e)
            # Still write trace on error so we can debug
            self._write_trace(document.document_id, trace)
            return []

    def _write_trace(self, document_id: str, trace: dict[str, Any]) -> None:
        """Write trace file for debugging relationship extraction."""
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self._trace_dir / f"{document_id}.relationships.trace.json"
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
            print(f"  Wrote relationship trace: {trace_path}")
        except Exception as e:
            print(f"  Warning: Failed to write trace file: {e}")
