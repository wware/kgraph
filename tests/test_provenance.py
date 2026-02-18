"""Tests for provenance accumulation during ingestion.

Covers ProvenanceAccumulator: add_mention, add_evidence, counts, and
that accumulated data is exposed for export.
"""

import pytest

from kgraph.provenance import ProvenanceAccumulator


class TestProvenanceAccumulator:
    """ProvenanceAccumulator records mentions and evidence for bundle export."""

    def test_init_empty(self) -> None:
        """New accumulator has no mentions or evidence."""
        acc = ProvenanceAccumulator()
        assert acc.mention_count() == 0
        assert acc.evidence_count() == 0
        assert acc.mentions == []
        assert acc.evidence == []

    def test_add_mention_appends_and_increments_count(self) -> None:
        """add_mention appends a MentionRow and mention_count increases."""
        acc = ProvenanceAccumulator()
        acc.add_mention(
            entity_id="e1",
            document_id="doc-1",
            section="abstract",
            start_offset=0,
            end_offset=5,
            text_span="Aspirin",
            context="Treatment with Aspirin",
            confidence=0.95,
            extraction_method="llm",
            created_at="2024-01-15T12:00:00Z",
        )
        assert acc.mention_count() == 1
        assert len(acc.mentions) == 1
        m = acc.mentions[0]
        assert m.entity_id == "e1"
        assert m.document_id == "doc-1"
        assert m.section == "abstract"
        assert m.start_offset == 0
        assert m.end_offset == 5
        assert m.text_span == "Aspirin"
        assert m.context == "Treatment with Aspirin"
        assert m.confidence == 0.95
        assert m.extraction_method == "llm"
        assert m.created_at == "2024-01-15T12:00:00Z"

        acc.add_mention(
            entity_id="e2",
            document_id="doc-1",
            section=None,
            start_offset=10,
            end_offset=18,
            text_span="Ibuprofen",
            context=None,
            confidence=0.9,
            extraction_method="rule_based",
            created_at="2024-01-15T12:01:00Z",
        )
        assert acc.mention_count() == 2
        assert len(acc.mentions) == 2
        assert acc.mentions[1].entity_id == "e2"
        assert acc.mentions[1].section is None

    def test_add_evidence_appends_and_increments_count(self) -> None:
        """add_evidence appends an EvidenceRow and evidence_count increases."""
        acc = ProvenanceAccumulator()
        acc.add_evidence(
            relationship_key="e1:treats:e2",
            document_id="doc-1",
            section="results",
            start_offset=100,
            end_offset=150,
            text_span="Aspirin reduces inflammation.",
            confidence=0.88,
            supports=True,
        )
        assert acc.evidence_count() == 1
        assert len(acc.evidence) == 1
        e = acc.evidence[0]
        assert e.relationship_key == "e1:treats:e2"
        assert e.document_id == "doc-1"
        assert e.section == "results"
        assert e.start_offset == 100
        assert e.end_offset == 150
        assert e.text_span == "Aspirin reduces inflammation."
        assert e.confidence == 0.88
        assert e.supports is True

        acc.add_evidence(
            relationship_key="e1:treats:e2",
            document_id="doc-2",
            section=None,
            start_offset=0,
            end_offset=30,
            text_span="Another supporting sentence.",
            confidence=0.7,
            supports=False,
        )
        assert acc.evidence_count() == 2
        assert acc.evidence[1].supports is False

    def test_mentions_and_evidence_independent(self) -> None:
        """Mention and evidence lists are independent; adding one does not affect the other."""
        acc = ProvenanceAccumulator()
        acc.add_mention(
            entity_id="e1",
            document_id="d1",
            section=None,
            start_offset=0,
            end_offset=1,
            text_span="x",
            context=None,
            confidence=1.0,
            extraction_method="test",
            created_at="2024-01-01T00:00:00Z",
        )
        acc.add_evidence(
            relationship_key="e1:rel:e2",
            document_id="d1",
            section=None,
            start_offset=0,
            end_offset=1,
            text_span="y",
            confidence=1.0,
            supports=True,
        )
        assert acc.mention_count() == 1
        assert acc.evidence_count() == 1
        assert len(acc.mentions) == 1
        assert len(acc.evidence) == 1

    def test_mentions_exposed_for_export(self) -> None:
        """Accumulator exposes mentions list for exporter to iterate."""
        acc = ProvenanceAccumulator()
        acc.add_mention(
            entity_id="e1",
            document_id="d1",
            section=None,
            start_offset=0,
            end_offset=1,
            text_span="x",
            context=None,
            confidence=1.0,
            extraction_method="test",
            created_at="2024-01-01T00:00:00Z",
        )
        exported = list(acc.mentions)
        assert len(exported) == 1
        assert exported[0].entity_id == "e1"
        assert exported[0].model_dump_json()  # Serializable for JSONL
