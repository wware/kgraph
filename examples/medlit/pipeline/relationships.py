"""Relationship extraction from journal articles.

Extracts relationships from Paper JSON format (from med-lit-schema).
Since the papers already have extracted relationships, we convert those to BaseRelationship objects.
Can also use Ollama LLM for relationship extraction from text.
"""

import json
from collections import defaultdict
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


def _normalize_entity_name(e: str) -> str:
    return "".join(ch for ch in e.casefold() if ch.isalnum())


def _build_entity_index(entities: Sequence[BaseEntity]) -> dict[str, list[BaseEntity]]:
    idx: dict[str, list[BaseEntity]] = defaultdict(list)
    for e in entities:
        keys = [e.name, *getattr(e, "synonyms", [])]
        for k in keys:
            nk = _normalize_entity_name(k)
            if nk:
                idx[nk].append(e)
    return idx


def _normalize_evidence_for_match(text: str) -> str:
    """Normalize evidence text for substring matching: lowercase, strip, collapse whitespace."""
    return " ".join(text.strip().lower().split()) if text else ""


def _evidence_contains_both_entities(
    evidence: str,
    subject_name: str,
    object_name: str,
    subject_entity: BaseEntity | None,
    object_entity: BaseEntity | None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Check that both subject and object (or synonyms) appear in the evidence text.

    Returns:
        (ok, drop_reason, detail): ok is True only if both entities appear;
        drop_reason is set when ok is False; detail has subject_in_evidence, object_in_evidence.
    """
    detail: dict[str, Any] = {"subject_in_evidence": False, "object_in_evidence": False}
    norm_evidence = _normalize_evidence_for_match(evidence)

    if not norm_evidence:
        return False, "evidence_empty", detail

    def _name_or_synonyms_appear_in_text(name: str, entity: BaseEntity | None) -> bool:
        candidates = [name]
        if entity is not None:
            candidates.append(entity.name)
            candidates.extend(getattr(entity, "synonyms", []))
        for c in candidates:
            if not c or not isinstance(c, str):
                continue
            norm = c.strip().lower()
            if norm and norm in norm_evidence:
                return True
        return False

    subject_ok = _name_or_synonyms_appear_in_text(subject_name, subject_entity)
    object_ok = _name_or_synonyms_appear_in_text(object_name, object_entity)
    detail["subject_in_evidence"] = subject_ok
    detail["object_in_evidence"] = object_ok

    if not subject_ok and not object_ok:
        return False, "evidence_missing_subject_and_object", detail
    if not subject_ok:
        return False, "evidence_missing_subject", detail
    if not object_ok:
        return False, "evidence_missing_object", detail
    return True, None, detail


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

    def _build_llm_prompt(self, text_sample: str, entity_list: str) -> str:
        """
        Build the prompt for the LLM.

        Notes:
        - This prompt is driven by the domain schema:
          - self._domain.relationship_types (vocabulary)
          - self._domain.predicate_constraints (allowed subject/object types)
        - Predicates must be returned in *lowercase* (e.g. "treats"), because downstream
          code currently normalizes and stores predicates as lowercase. :contentReference[oaicite:3]{index=3}
        """
        # 1) Predicate inventory from domain schema
        # Domain keys are like "treats"; we ask LLM for lowercase strings like "treats"
        predicate_keys = sorted(self._domain.relationship_types.keys())
        allowed_predicates = [k.lower() for k in predicate_keys]

        # 2) Build a compact “signature table” from predicate_constraints
        # Example: treats: drug -> disease
        sig_lines: list[str] = []
        for key in predicate_keys:
            lc = key.lower()
            c = self._domain.predicate_constraints.get(key)
            if c:
                subj = ", ".join(sorted(c.subject_types))
                obj = ", ".join(sorted(c.object_types))
                sig_lines.append(f"- {lc}: ({subj}) -> ({obj})")
            else:
                sig_lines.append(f"- {lc}: (no type constraints defined)")

        # Optional: a tiny bit of extra guidance for a few high-frequency predicates
        # (keep this short; the signature table does most of the work)
        extra_guidance = [
            "- Prefer the *most specific* predicate that matches the evidence.",
            '- Use "associated_with" ONLY if no other predicate fits the evidence.',
            "- Each relationship MUST include a short evidence quote where BOTH entities appear.",
            "- Skip speculative relationships not supported by explicit text.",
        ]

        return f"""You extract medical relationships from biomedical text.

You are given:
1) A list of entities (name, type, id) already extracted from this document.
2) A text snippet from the document.

Your job:
- Find relationships between the listed entities that are explicitly supported by the text.
- You MUST choose a predicate whose subject/object types match the signature table. If none apply, output no relationship.
- Output ONLY valid JSON: a JSON array of relationship objects.
- Use ONLY the allowed predicates listed below (lowercase strings).

ALLOWED PREDICATES:
{", ".join(allowed_predicates)}

PREDICATE SIGNATURES (subject_type -> object_type):
{chr(10).join(sig_lines)}

GUIDELINES:
{chr(10).join(extra_guidance)}

ENTITIES IN DOCUMENT (name (type): id):
{entity_list}

TEXT:
{text_sample}

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{
    "subject": "entity_name_exactly_as_listed",
    "predicate": "treats",
    "object": "entity_name_exactly_as_listed",
    "confidence": 0.0,
    "evidence": "short exact quote from text"
  }}
]

IMPORTANT:
- Use entity names from the list when possible; do not invent entities.
- confidence is a float in [0, 1].
- Return ONLY the JSON array, no prose, no markdown fences.
"""

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

        """
        Removing the limit on the size of the entities list......

        This may help recall in some cases, but it can also **tank
        relationship quality** (LLMs get overwhelmed and start producing
        generic or spurious edges). If you notice regressions, the “best of
        both worlds” is:

        - include **all entities** in the *matching index* (which you do)
        - but include a **curated subset** in the prompt list:

            - all canonicals
            - plus entities that appear in the abstract
            - plus top-N by usage_count/confidence

        This will reduce prompt bloat without sacrificing matching coverage.
        """
        # entity_list = "\n".join(f"- {e.name} ({e.get_entity_type()}): {e.entity_id}" for e in entities[:50])  # Limit to avoid huge prompts
        entity_list = "\n".join(f"- {e.name} ({e.get_entity_type()}): {e.entity_id}" for e in entities)

        # Use abstract for extraction (shorter, more focused)
        text = document.abstract if hasattr(document, "abstract") and document.abstract else document.content
        text_sample = text[:2000]  # Limit text size for faster processing

        prompt = self._build_llm_prompt(text_sample, entity_list)

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
                entity_index = _build_entity_index(entities)
                for item in response:
                    relationship, decision = self._process_llm_item(item, entity_index, document)
                    trace["decisions"].append(decision)
                    if relationship:
                        relationships.append(relationship)

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

    def _process_llm_item(self, item: Any, entity_index: dict[str, list[BaseEntity]], document: BaseDocument) -> tuple[BaseRelationship | None, dict[str, Any]]:
        """Process a single item from the LLM response."""
        if not isinstance(item, dict):
            return None, {"item": item, "accepted": False, "drop_reason": "item_not_a_dict"}

        subject_name = item.get("subject", "").strip()
        predicate = item.get("predicate", "").lower()
        object_name = item.get("object", "").strip()
        confidence = float(item.get("confidence", 0.5))
        evidence = item.get("evidence", "")

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

        def _pick_unique(cands: list[BaseEntity]) -> BaseEntity | None:
            if not cands:
                return None
            if len(cands) == 1:
                return cands[0]
            # collision: prefer canonical, else highest usage_count/confidence if you have them
            cands2 = sorted(
                cands,
                key=lambda e: (
                    getattr(e, "status", "") == "canonical",
                    getattr(e, "usage_count", 0),
                    getattr(e, "confidence", 0.0),
                ),
                reverse=True,
            )
            return cands2[0]

        subject_entity = _pick_unique(entity_index.get(_normalize_entity_name(subject_name), []))
        object_entity = _pick_unique(entity_index.get(_normalize_entity_name(object_name), []))

        decision["matched_subject"] = subject_entity is not None
        decision["matched_object"] = object_entity is not None

        if not subject_entity or not object_entity:
            if not subject_entity and not object_entity:
                decision["drop_reason"] = "subject_and_object_unmatched"
            elif not subject_entity:
                decision["drop_reason"] = "subject_unmatched"
            else:
                decision["drop_reason"] = "object_unmatched"
            return None, decision

        evidence_ok, evidence_drop_reason, evidence_detail = _evidence_contains_both_entities(
            evidence, subject_name, object_name, subject_entity, object_entity
        )
        decision["evidence_check"] = evidence_detail
        if not evidence_ok:
            decision["drop_reason"] = evidence_drop_reason
            return None, decision

        semantic_ok = self._validate_predicate_semantics(predicate, evidence)
        decision["semantic_ok"] = semantic_ok
        if not semantic_ok:
            print(f"  Warning: Semantic mismatch - predicate '{predicate}' " f"does not match evidence: {evidence[:100]}...")
            decision["drop_reason"] = "semantic_mismatch"
            return None, decision

        if self._should_swap_subject_object(predicate, subject_entity, object_entity):
            print(
                f"  Swapping subject/object for predicate '{predicate}': "
                f"({subject_entity.name} [{subject_entity.get_entity_type()}] ↔ "
                f"{object_entity.name} [{object_entity.get_entity_type()}])"
            )
            subject_entity, object_entity = object_entity, subject_entity
            decision["swap_applied"] = True

        decision["resolved"] = {
            "subject_id": subject_entity.entity_id,
            "subject_type": subject_entity.get_entity_type(),
            "object_id": object_entity.entity_id,
            "object_type": object_entity.get_entity_type(),
        }


        constraint = self._domain.predicate_constraints.get(predicate)

        if constraint:
            s_type = subject_entity.get_entity_type()
            o_type = object_entity.get_entity_type()

            ok = (s_type in constraint.subject_types) and (o_type in constraint.object_types)
            if not ok:
                decision["accepted"] = False
                decision["drop_reason"] = "type_constraint_mismatch"
                decision["type_check"] = {
                    "predicate_key": predicate,
                    "got_subject_type": s_type,
                    "got_object_type": o_type,
                    "expected_subject_types": sorted(constraint.subject_types),
                    "expected_object_types": sorted(constraint.object_types),
                }
                return None, decision

        provenance = Provenance(
            document_id=document.document_id,
            source_uri=document.source_uri,
            section="abstract" if hasattr(document, "abstract") and document.abstract else None,
        )

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

        decision["accepted"] = True
        return rel, decision

    def _write_trace(self, document_id: str, trace: dict[str, Any]) -> None:
        """Write trace file for debugging relationship extraction."""
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self._trace_dir / f"{document_id}.relationships.trace.json"
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
            print(f"  Wrote relationship trace: {trace_path}")
        except Exception as e:
            print(f"  Warning: Failed to write trace file: {e}")
