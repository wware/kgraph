"""
Tests for bundle provenance in storage: load_bundle with mentions/evidence,
get_mentions_for_entity, get_evidence_for_relationship, and entity/relationship
properties containing provenance summary.
"""

import json
import pytest
from datetime import datetime

from kgbundle import BundleManifestV1, BundleFile
from storage.backends.sqlite import SQLiteStorage


@pytest.fixture
def bundle_dir_with_provenance(tmp_path):
    """Create a bundle directory with entities, relationships, mentions, and evidence."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    # Entities with provenance summary fields
    (bundle / "entities.jsonl").write_text(
        json.dumps(
            {
                "entity_id": "e1",
                "entity_type": "drug",
                "name": "Aspirin",
                "status": "canonical",
                "confidence": 0.95,
                "usage_count": 5,
                "created_at": "2024-01-15T12:00:00Z",
                "source": "test",
                "first_seen_document": "doc-1",
                "first_seen_section": "abstract",
                "total_mentions": 3,
                "supporting_documents": ["doc-1", "doc-2", "doc-3"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "entity_id": "e2",
                "entity_type": "disease",
                "name": "Headache",
                "status": "canonical",
                "confidence": 0.9,
                "usage_count": 2,
                "created_at": "2024-01-15T12:00:00Z",
                "source": "test",
            }
        )
        + "\n"
    )

    # Relationships with evidence summary
    (bundle / "relationships.jsonl").write_text(
        json.dumps(
            {
                "subject_id": "e1",
                "object_id": "e2",
                "predicate": "treats",
                "confidence": 0.88,
                "source_documents": ["doc-1"],
                "created_at": "2024-01-15T12:00:00Z",
                "evidence_count": 2,
                "strongest_evidence_quote": "Aspirin treats headache.",
                "evidence_confidence_avg": 0.85,
            }
        )
        + "\n"
    )

    # Mentions
    (bundle / "mentions.jsonl").write_text(
        json.dumps(
            {
                "entity_id": "e1",
                "document_id": "doc-1",
                "section": "abstract",
                "start_offset": 0,
                "end_offset": 7,
                "text_span": "Aspirin",
                "context": "Treatment with Aspirin",
                "confidence": 0.95,
                "extraction_method": "llm",
                "created_at": "2024-01-15T12:00:00Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                "entity_id": "e1",
                "document_id": "doc-2",
                "section": None,
                "start_offset": 10,
                "end_offset": 17,
                "text_span": "Aspirin",
                "context": None,
                "confidence": 0.9,
                "extraction_method": "llm",
                "created_at": "2024-01-15T12:01:00Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                "entity_id": "e2",
                "document_id": "doc-1",
                "section": "results",
                "start_offset": 50,
                "end_offset": 58,
                "text_span": "Headache",
                "context": None,
                "confidence": 0.9,
                "extraction_method": "llm",
                "created_at": "2024-01-15T12:00:00Z",
            }
        )
        + "\n"
    )

    # Evidence
    (bundle / "evidence.jsonl").write_text(
        json.dumps(
            {
                "relationship_key": "e1:treats:e2",
                "document_id": "doc-1",
                "section": "results",
                "start_offset": 100,
                "end_offset": 130,
                "text_span": "Aspirin treats headache.",
                "confidence": 0.9,
                "supports": True,
            }
        )
        + "\n"
        + json.dumps(
            {
                "relationship_key": "e1:treats:e2",
                "document_id": "doc-1",
                "section": None,
                "start_offset": 200,
                "end_offset": 235,
                "text_span": "Evidence: Aspirin is used for headache.",
                "confidence": 0.8,
                "supports": True,
            }
        )
        + "\n"
    )

    manifest = BundleManifestV1(
        bundle_id="provenance-test-bundle",
        domain="test",
        created_at=datetime.now().isoformat(),
        bundle_version="v1",
        entities=BundleFile(path="entities.jsonl", format="jsonl"),
        relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
        mentions=BundleFile(path="mentions.jsonl", format="jsonl"),
        evidence=BundleFile(path="evidence.jsonl", format="jsonl"),
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2))

    return bundle


@pytest.fixture
def storage_with_provenance_bundle(tmp_path, bundle_dir_with_provenance):
    """SQLite storage with a bundle loaded that includes mentions and evidence."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(str(db_path))
    manifest = BundleManifestV1.model_validate_json((bundle_dir_with_provenance / "manifest.json").read_text())
    storage.load_bundle(manifest, str(bundle_dir_with_provenance))
    try:
        yield storage
    finally:
        storage.close()


class TestLoadBundleProvenance:
    """Loading a bundle with mentions and evidence populates provenance tables."""

    def test_load_bundle_stores_mentions(self, storage_with_provenance_bundle):
        """After load_bundle, get_mentions_for_entity returns mention rows."""
        storage = storage_with_provenance_bundle
        mentions_e1 = storage.get_mentions_for_entity("e1")
        assert len(mentions_e1) == 2
        by_doc = {m.document_id: m for m in mentions_e1}
        assert "doc-1" in by_doc
        assert by_doc["doc-1"].section == "abstract"
        assert by_doc["doc-1"].text_span == "Aspirin"
        assert "doc-2" in by_doc

        mentions_e2 = storage.get_mentions_for_entity("e2")
        assert len(mentions_e2) == 1
        assert mentions_e2[0].document_id == "doc-1"
        assert mentions_e2[0].text_span == "Headache"

    def test_load_bundle_stores_evidence(self, storage_with_provenance_bundle):
        """After load_bundle, get_evidence_for_relationship returns evidence rows."""
        storage = storage_with_provenance_bundle
        evidence = storage.get_evidence_for_relationship("e1", "treats", "e2")
        assert len(evidence) == 2
        texts = {e.text_span for e in evidence}
        assert "Aspirin treats headache." in texts
        assert "Evidence: Aspirin is used for headache." in texts
        assert all(e.supports for e in evidence)

    def test_get_mentions_for_entity_nonexistent_returns_empty(self, storage_with_provenance_bundle):
        """get_mentions_for_entity for unknown entity returns empty list."""
        storage = storage_with_provenance_bundle
        assert storage.get_mentions_for_entity("nonexistent") == []

    def test_get_evidence_for_relationship_nonexistent_returns_empty(self, storage_with_provenance_bundle):
        """get_evidence_for_relationship for unknown triple returns empty list."""
        storage = storage_with_provenance_bundle
        assert storage.get_evidence_for_relationship("e2", "treats", "e1") == []


class TestEntityRelationshipProvenanceProperties:
    """Entity and relationship provenance summary is stored in properties after load."""

    def test_entity_properties_contain_provenance(self, storage_with_provenance_bundle):
        """Entity loaded from bundle has first_seen_document, total_mentions, etc. in properties."""
        storage = storage_with_provenance_bundle
        entity = storage.get_entity("e1")
        assert entity is not None
        assert entity.properties is not None
        assert entity.properties.get("first_seen_document") == "doc-1"
        assert entity.properties.get("first_seen_section") == "abstract"
        assert entity.properties.get("total_mentions") == 3
        assert set(entity.properties.get("supporting_documents", [])) == {"doc-1", "doc-2", "doc-3"}

    def test_entity_without_provenance_has_empty_or_no_provenance_keys(self, storage_with_provenance_bundle):
        """Entity e2 was written without provenance fields; properties may not have them."""
        storage = storage_with_provenance_bundle
        entity = storage.get_entity("e2")
        assert entity is not None
        # e2 had no first_seen_document etc. in the JSONL row
        assert entity.properties is not None

    def test_relationship_properties_contain_evidence_summary(self, storage_with_provenance_bundle):
        """Relationship loaded from bundle has evidence_count, strongest_evidence_quote in properties."""
        storage = storage_with_provenance_bundle
        rel = storage.get_relationship("e1", "treats", "e2")
        assert rel is not None
        assert rel.properties is not None
        assert rel.properties.get("evidence_count") == 2
        assert rel.properties.get("strongest_evidence_quote") == "Aspirin treats headache."
        assert rel.properties.get("evidence_confidence_avg") == 0.85


class TestLoadBundleWithoutProvenanceFiles:
    """Bundles without mentions/evidence files load successfully."""

    def test_load_bundle_missing_mentions_file_does_not_raise(self, tmp_path):
        """Manifest has mentions but file is missing; load_bundle does not raise."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "entities.jsonl").write_text(
            json.dumps({"entity_id": "e1", "entity_type": "x", "name": "E1", "status": "canonical", "usage_count": 0, "created_at": "2024-01-01T00:00:00Z", "source": "test"}) + "\n"
        )
        (bundle / "relationships.jsonl").write_text(json.dumps({"subject_id": "e1", "object_id": "e1", "predicate": "self", "source_documents": [], "created_at": "2024-01-01T00:00:00Z"}) + "\n")
        manifest = BundleManifestV1(
            bundle_id="no-files-bundle",
            domain="test",
            created_at=datetime.now().isoformat(),
            bundle_version="v1",
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
            mentions=BundleFile(path="mentions.jsonl", format="jsonl"),  # file not created
            evidence=None,
        )
        (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2))

        storage = SQLiteStorage(":memory:")
        storage.load_bundle(manifest, str(bundle))
        assert storage.get_entity("e1") is not None
        assert storage.get_mentions_for_entity("e1") == []
        storage.close()
