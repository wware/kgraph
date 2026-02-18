"""Tests for kgbundle Pydantic models (bundle contract)."""

from kgbundle import (
    BundleFile,
    BundleManifestV1,
    EntityRow,
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
