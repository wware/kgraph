"""Tests for kgbundle Pydantic models (bundle contract)."""

from kgbundle import (
    BundleFile,
    BundleManifestV1,
    EntityRow,
    EvidenceRow,
    MentionRow,
    RelationshipRow,
)


class TestEntityRow:
    """Test EntityRow serialization and validation."""

    def test_minimal_entity_row(self):
        row = EntityRow(
            entity_id="char:1",
            entity_type="character",
            status="canonical",
            usage_count=1,
            created_at="2024-01-15T10:00:00Z",
            source="test",
        )
        assert row.entity_id == "char:1"
        assert row.name is None
        assert row.properties == {}

    def test_entity_row_roundtrip_json(self):
        row = EntityRow(
            entity_id="char:2",
            entity_type="character",
            name="Sherlock",
            status="provisional",
            usage_count=2,
            created_at="2024-01-15T10:00:00Z",
            source="test",
            properties={"key": "value"},
        )
        data = row.model_dump()
        restored = EntityRow.model_validate(data)
        assert restored.entity_id == row.entity_id
        assert restored.properties == row.properties

    def test_entity_row_provenance_fields(self):
        row = EntityRow(
            entity_id="char:3",
            entity_type="character",
            status="canonical",
            usage_count=1,
            created_at="2024-01-15T10:00:00Z",
            source="test",
            first_seen_document="doc1",
            first_seen_section="abstract",
            total_mentions=5,
            supporting_documents=["doc1", "doc2"],
        )
        assert row.first_seen_document == "doc1"
        assert row.total_mentions == 5
        assert row.supporting_documents == ["doc1", "doc2"]
        data = row.model_dump()
        restored = EntityRow.model_validate(data)
        assert restored.total_mentions == 5
        assert restored.supporting_documents == row.supporting_documents


class TestRelationshipRow:
    """Test RelationshipRow serialization."""

    def test_minimal_relationship_row(self):
        row = RelationshipRow(
            subject_id="char:1",
            object_id="char:2",
            predicate="knows",
            source_documents=["doc1"],
            created_at="2024-01-15T10:00:00Z",
        )
        assert row.subject_id == "char:1"
        assert row.object_id == "char:2"
        assert row.predicate == "knows"
        assert row.source_documents == ["doc1"]

    def test_relationship_row_evidence_fields(self):
        row = RelationshipRow(
            subject_id="char:1",
            object_id="char:2",
            predicate="treats",
            source_documents=["doc1"],
            created_at="2024-01-15T10:00:00Z",
            evidence_count=2,
            strongest_evidence_quote="Aspirin reduces pain.",
            evidence_confidence_avg=0.85,
        )
        assert row.evidence_count == 2
        assert row.strongest_evidence_quote == "Aspirin reduces pain."
        assert row.evidence_confidence_avg == 0.85


class TestMentionRow:
    """Test MentionRow (mentions.jsonl)."""

    def test_mention_row_roundtrip(self):
        row = MentionRow(
            entity_id="prov:abc",
            document_id="PMC123",
            section="abstract",
            start_offset=0,
            end_offset=10,
            text_span="diabetes",
            context="patients with diabetes",
            confidence=0.9,
            extraction_method="llm",
            created_at="2024-01-15T10:00:00Z",
        )
        data = row.model_dump()
        restored = MentionRow.model_validate(data)
        assert restored.entity_id == row.entity_id
        assert restored.document_id == "PMC123"
        assert restored.text_span == "diabetes"


class TestEvidenceRow:
    """Test EvidenceRow (evidence.jsonl)."""

    def test_evidence_row_roundtrip(self):
        row = EvidenceRow(
            relationship_key="sub:pred:obj",
            document_id="PMC456",
            section=None,
            start_offset=100,
            end_offset=150,
            text_span="Gene X is associated with disease Y.",
            confidence=0.8,
            supports=True,
        )
        data = row.model_dump()
        restored = EvidenceRow.model_validate(data)
        assert restored.relationship_key == "sub:pred:obj"
        assert restored.text_span == row.text_span
        assert restored.supports is True


class TestBundleManifestV1:
    """Test BundleManifestV1."""

    def test_manifest_required_fields(self):
        manifest = BundleManifestV1(
            bundle_id="test-bundle-123",
            domain="test",
            created_at="2024-01-15T10:00:00Z",
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
        )
        assert manifest.bundle_version == "v1"
        assert manifest.bundle_id == "test-bundle-123"
        assert manifest.entities.path == "entities.jsonl"
        assert manifest.doc_assets is None
        assert manifest.mentions is None
        assert manifest.evidence is None

    def test_manifest_with_mentions_and_evidence(self):
        manifest = BundleManifestV1(
            bundle_id="bundle-2",
            domain="medlit",
            created_at="2024-01-15T10:00:00Z",
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
            mentions=BundleFile(path="mentions.jsonl", format="jsonl"),
            evidence=BundleFile(path="evidence.jsonl", format="jsonl"),
        )
        assert manifest.mentions is not None
        assert manifest.mentions.path == "mentions.jsonl"
        assert manifest.evidence is not None
        assert manifest.evidence.path == "evidence.jsonl"
