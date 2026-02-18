"""Provenance accumulation for bundle export.

Collects entity mentions and relationship evidence during ingestion so the
exporter can write mentions.jsonl and evidence.jsonl and fill EntityRow/
RelationshipRow provenance summary fields.
"""

from typing import Optional

from kgbundle import EvidenceRow, MentionRow


class ProvenanceAccumulator:
    """In-memory collector for entity mentions and relationship evidence.

    The orchestrator calls add_mention when storing/updating entities and
    add_evidence when storing relationships that have evidence. The exporter
    reads the accumulated lists to write mentions.jsonl and evidence.jsonl
    and to compute per-entity and per-relationship summary fields.
    """

    def __init__(self) -> None:
        self._mentions: list[MentionRow] = []
        self._evidence: list[EvidenceRow] = []

    def add_mention(
        self,
        *,
        entity_id: str,
        document_id: str,
        section: Optional[str],
        start_offset: int,
        end_offset: int,
        text_span: str,
        context: Optional[str],
        confidence: float,
        extraction_method: str,
        created_at: str,
    ) -> None:
        """Record one entity mention (one row in mentions.jsonl)."""
        self._mentions.append(
            MentionRow(
                entity_id=entity_id,
                document_id=document_id,
                section=section,
                start_offset=start_offset,
                end_offset=end_offset,
                text_span=text_span,
                context=context,
                confidence=confidence,
                extraction_method=extraction_method,
                created_at=created_at,
            )
        )

    def add_evidence(
        self,
        *,
        relationship_key: str,
        document_id: str,
        section: Optional[str],
        start_offset: int,
        end_offset: int,
        text_span: str,
        confidence: float,
        supports: bool = True,
    ) -> None:
        """Record one evidence span (one row in evidence.jsonl)."""
        self._evidence.append(
            EvidenceRow(
                relationship_key=relationship_key,
                document_id=document_id,
                section=section,
                start_offset=start_offset,
                end_offset=end_offset,
                text_span=text_span,
                confidence=confidence,
                supports=supports,
            )
        )

    @property
    def mentions(self) -> list[MentionRow]:
        return self._mentions

    @property
    def evidence(self) -> list[EvidenceRow]:
        return self._evidence

    def mention_count(self) -> int:
        return len(self._mentions)

    def evidence_count(self) -> int:
        return len(self._evidence)
