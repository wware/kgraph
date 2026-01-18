"""Tests for export functionality."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kgraph.entity import EntityStatus
from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from tests.conftest import (
    MockDocumentParser,
    MockEmbeddingGenerator,
    MockEntityExtractor,
    MockEntityResolver,
    MockRelationshipExtractor,
    TestDocument,
    TestDomainSchema,
    TestEntity,
    TestRelationship,
    make_test_entity,
    make_test_relationship,
)


@pytest.fixture
def orchestrator(
    tmp_path: Path,
) -> IngestionOrchestrator:
    """Create an orchestrator for export tests."""
    return IngestionOrchestrator(
        domain=TestDomainSchema(),
        parser=MockDocumentParser(),
        entity_extractor=MockEntityExtractor(),
        entity_resolver=MockEntityResolver(),
        relationship_extractor=MockRelationshipExtractor(),
        embedding_generator=MockEmbeddingGenerator(),
        entity_storage=InMemoryEntityStorage(),
        relationship_storage=InMemoryRelationshipStorage(),
        document_storage=InMemoryDocumentStorage(),
    )


class TestExportEntities:
    """Tests for entity export."""

    async def test_export_canonical_entities(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should include only canonical entities by default."""
        # Add canonical and provisional entities
        canonical = make_test_entity(
            "Canonical Entity",
            status=EntityStatus.CANONICAL,
            entity_id="canonical-1",
        )
        provisional = make_test_entity(
            "Provisional Entity",
            status=EntityStatus.PROVISIONAL,
            entity_id="provisional-1",
        )
        await orchestrator.entity_storage.add(canonical)
        await orchestrator.entity_storage.add(provisional)

        # Export
        output_file = tmp_path / "entities.json"
        count = await orchestrator.export_entities(output_file)

        assert count == 1
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["domain"] == "test_domain"
        assert data["entity_count"] == 1
        assert len(data["entities"]) == 1
        assert data["entities"][0]["entity_id"] == "canonical-1"
        assert data["entities"][0]["status"] == "canonical"

    async def test_export_includes_provisional_when_requested(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should include provisional entities when requested."""
        canonical = make_test_entity(
            "Canonical Entity",
            status=EntityStatus.CANONICAL,
            entity_id="canonical-1",
        )
        provisional = make_test_entity(
            "Provisional Entity",
            status=EntityStatus.PROVISIONAL,
            entity_id="provisional-1",
        )
        await orchestrator.entity_storage.add(canonical)
        await orchestrator.entity_storage.add(provisional)

        output_file = tmp_path / "all_entities.json"
        count = await orchestrator.export_entities(
            output_file, include_provisional=True
        )

        assert count == 2

        with open(output_file) as f:
            data = json.load(f)

        assert data["entity_count"] == 2
        entity_ids = {e["entity_id"] for e in data["entities"]}
        assert entity_ids == {"canonical-1", "provisional-1"}

    async def test_export_entity_fields(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should serialize all entity fields correctly."""
        entity = TestEntity(
            entity_id="test-123",
            status=EntityStatus.CANONICAL,
            name="Test Entity",
            synonyms=("alias1", "alias2"),
            embedding=(0.1, 0.2, 0.3),
            dbpedia_uri="http://dbpedia.org/resource/Test",
            confidence=0.95,
            usage_count=5,
            created_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            source="test_source",
            metadata={"key": "value"},
        )
        await orchestrator.entity_storage.add(entity)

        output_file = tmp_path / "entities.json"
        await orchestrator.export_entities(output_file)

        with open(output_file) as f:
            data = json.load(f)

        exported = data["entities"][0]
        assert exported["entity_id"] == "test-123"
        assert exported["status"] == "canonical"
        assert exported["name"] == "Test Entity"
        assert exported["synonyms"] == ["alias1", "alias2"]
        assert exported["embedding"] == [0.1, 0.2, 0.3]
        assert exported["dbpedia_uri"] == "http://dbpedia.org/resource/Test"
        assert exported["confidence"] == 0.95
        assert exported["usage_count"] == 5
        assert exported["source"] == "test_source"
        assert exported["metadata"] == {"key": "value"}
        assert exported["entity_type"] == "test_entity"
        assert exported["canonical_id_source"] == "test_authority"

    async def test_export_creates_parent_directories(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should create parent directories if they don't exist."""
        entity = make_test_entity(
            "Entity", status=EntityStatus.CANONICAL, entity_id="e1"
        )
        await orchestrator.entity_storage.add(entity)

        output_file = tmp_path / "subdir" / "deep" / "entities.json"
        await orchestrator.export_entities(output_file)

        assert output_file.exists()


class TestExportDocument:
    """Tests for per-document export."""

    async def test_export_document_relationships(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should include relationships from the document."""
        # Create and store a document
        doc = TestDocument(
            document_id="doc-1",
            content="Test content",
            content_type="text/plain",
            source_uri="file://test.txt",
            created_at=datetime.now(timezone.utc),
        )
        await orchestrator.document_storage.add(doc)

        # Create entities
        e1 = make_test_entity(
            "Entity 1", status=EntityStatus.CANONICAL, entity_id="e1"
        )
        e2 = make_test_entity(
            "Entity 2", status=EntityStatus.CANONICAL, entity_id="e2"
        )
        await orchestrator.entity_storage.add(e1)
        await orchestrator.entity_storage.add(e2)

        # Create relationship from this document
        rel = TestRelationship(
            subject_id="e1",
            predicate="related_to",
            object_id="e2",
            confidence=0.9,
            source_documents=("doc-1",),
            created_at=datetime.now(timezone.utc),
        )
        await orchestrator.relationship_storage.add(rel)

        # Export
        output_file = tmp_path / "paper_doc-1.json"
        stats = await orchestrator.export_document("doc-1", output_file)

        assert stats["relationships"] == 1
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["document_id"] == "doc-1"
        assert data["relationship_count"] == 1
        assert len(data["relationships"]) == 1
        assert data["relationships"][0]["subject_id"] == "e1"
        assert data["relationships"][0]["predicate"] == "related_to"
        assert data["relationships"][0]["object_id"] == "e2"

    async def test_export_document_provisional_entities(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should include provisional entities from the document."""
        doc = TestDocument(
            document_id="doc-1",
            content="Test content",
            content_type="text/plain",
            source_uri="file://test.txt",
            created_at=datetime.now(timezone.utc),
        )
        await orchestrator.document_storage.add(doc)

        # Create provisional entity from this document
        provisional = TestEntity(
            entity_id="prov-1",
            status=EntityStatus.PROVISIONAL,
            name="Provisional",
            confidence=0.8,
            usage_count=1,
            created_at=datetime.now(timezone.utc),
            source="doc-1",  # Source matches document ID
        )
        await orchestrator.entity_storage.add(provisional)

        # Create provisional entity from another document
        other_provisional = TestEntity(
            entity_id="prov-2",
            status=EntityStatus.PROVISIONAL,
            name="Other Provisional",
            confidence=0.8,
            usage_count=1,
            created_at=datetime.now(timezone.utc),
            source="doc-2",  # Different source
        )
        await orchestrator.entity_storage.add(other_provisional)

        output_file = tmp_path / "paper_doc-1.json"
        stats = await orchestrator.export_document("doc-1", output_file)

        assert stats["provisional_entities"] == 1

        with open(output_file) as f:
            data = json.load(f)

        assert data["provisional_entity_count"] == 1
        assert len(data["provisional_entities"]) == 1
        assert data["provisional_entities"][0]["entity_id"] == "prov-1"

    async def test_export_document_includes_title(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should include document title when available."""
        doc = TestDocument(
            document_id="doc-1",
            title="My Document Title",
            content="Test content",
            content_type="text/plain",
            source_uri="file://test.txt",
            created_at=datetime.now(timezone.utc),
        )
        await orchestrator.document_storage.add(doc)

        output_file = tmp_path / "paper_doc-1.json"
        await orchestrator.export_document("doc-1", output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert data["document_title"] == "My Document Title"

    async def test_export_nonexistent_document(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export should handle missing document gracefully."""
        output_file = tmp_path / "paper_missing.json"
        stats = await orchestrator.export_document("nonexistent", output_file)

        assert stats["relationships"] == 0
        assert stats["provisional_entities"] == 0

        with open(output_file) as f:
            data = json.load(f)

        assert data["document_id"] == "nonexistent"
        assert data["document_title"] is None


class TestExportAll:
    """Tests for full export."""

    async def test_export_all_creates_files(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export all should create entities.json and per-document files."""
        # Ingest some documents
        await orchestrator.ingest_document(
            b"Document with [Entity1] and [Entity2]",
            "text/plain",
            "file://doc1.txt",
        )
        await orchestrator.ingest_document(
            b"Another document with [Entity3]",
            "text/plain",
            "file://doc2.txt",
        )

        # Promote some entities to canonical
        entities = await orchestrator.entity_storage.list_all()
        for entity in entities[:2]:
            await orchestrator.entity_storage.promote(
                entity.entity_id, f"canonical-{entity.name}"
            )

        # Export
        output_dir = tmp_path / "export"
        result = await orchestrator.export_all(output_dir)

        # Verify structure
        assert result["documents_exported"] == 2
        assert (output_dir / "entities.json").exists()

        # Verify entities.json
        with open(output_dir / "entities.json") as f:
            entities_data = json.load(f)
        assert entities_data["entity_count"] == 2

    async def test_export_all_returns_statistics(
        self, orchestrator: IngestionOrchestrator, tmp_path: Path
    ) -> None:
        """Export all should return detailed statistics."""
        # Ingest a document
        result = await orchestrator.ingest_document(
            b"[Entity1] relates to [Entity2]",
            "text/plain",
            "file://doc.txt",
        )
        doc_id = result.document_id

        # Promote one entity
        entities = await orchestrator.entity_storage.list_all()
        if entities:
            await orchestrator.entity_storage.promote(
                entities[0].entity_id, "canonical-1"
            )

        output_dir = tmp_path / "export"
        stats = await orchestrator.export_all(output_dir)

        assert stats["output_dir"] == str(output_dir)
        assert stats["canonical_entities"] == 1
        assert stats["documents_exported"] == 1
        assert doc_id in stats["document_stats"]


class TestListAllMethods:
    """Tests for new list_all storage methods."""

    async def test_entity_list_all(
        self, orchestrator: IngestionOrchestrator
    ) -> None:
        """List all should return all entities."""
        e1 = make_test_entity("E1", status=EntityStatus.CANONICAL, entity_id="e1")
        e2 = make_test_entity("E2", status=EntityStatus.PROVISIONAL, entity_id="e2")
        await orchestrator.entity_storage.add(e1)
        await orchestrator.entity_storage.add(e2)

        all_entities = await orchestrator.entity_storage.list_all()
        assert len(all_entities) == 2

    async def test_entity_list_all_with_status_filter(
        self, orchestrator: IngestionOrchestrator
    ) -> None:
        """List all should filter by status."""
        e1 = make_test_entity("E1", status=EntityStatus.CANONICAL, entity_id="e1")
        e2 = make_test_entity("E2", status=EntityStatus.PROVISIONAL, entity_id="e2")
        await orchestrator.entity_storage.add(e1)
        await orchestrator.entity_storage.add(e2)

        canonical = await orchestrator.entity_storage.list_all(status="canonical")
        provisional = await orchestrator.entity_storage.list_all(status="provisional")

        assert len(canonical) == 1
        assert canonical[0].entity_id == "e1"
        assert len(provisional) == 1
        assert provisional[0].entity_id == "e2"

    async def test_entity_list_all_pagination(
        self, orchestrator: IngestionOrchestrator
    ) -> None:
        """List all should support pagination."""
        for i in range(5):
            e = make_test_entity(f"E{i}", entity_id=f"e{i}")
            await orchestrator.entity_storage.add(e)

        page1 = await orchestrator.entity_storage.list_all(limit=2, offset=0)
        page2 = await orchestrator.entity_storage.list_all(limit=2, offset=2)
        page3 = await orchestrator.entity_storage.list_all(limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

    async def test_relationship_get_by_document(
        self, orchestrator: IngestionOrchestrator
    ) -> None:
        """Get by document should return relationships from that document."""
        rel1 = TestRelationship(
            subject_id="e1",
            predicate="related_to",
            object_id="e2",
            source_documents=("doc-1",),
            created_at=datetime.now(timezone.utc),
        )
        rel2 = TestRelationship(
            subject_id="e2",
            predicate="causes",
            object_id="e3",
            source_documents=("doc-2",),
            created_at=datetime.now(timezone.utc),
        )
        rel3 = TestRelationship(
            subject_id="e1",
            predicate="causes",
            object_id="e3",
            source_documents=("doc-1", "doc-2"),  # In both documents
            created_at=datetime.now(timezone.utc),
        )
        await orchestrator.relationship_storage.add(rel1)
        await orchestrator.relationship_storage.add(rel2)
        await orchestrator.relationship_storage.add(rel3)

        doc1_rels = await orchestrator.relationship_storage.get_by_document("doc-1")
        doc2_rels = await orchestrator.relationship_storage.get_by_document("doc-2")

        assert len(doc1_rels) == 2  # rel1 and rel3
        assert len(doc2_rels) == 2  # rel2 and rel3

    async def test_relationship_list_all(
        self, orchestrator: IngestionOrchestrator
    ) -> None:
        """List all should return all relationships."""
        rel1 = make_test_relationship("e1", "e2")
        rel2 = make_test_relationship("e2", "e3")
        await orchestrator.relationship_storage.add(rel1)
        await orchestrator.relationship_storage.add(rel2)

        all_rels = await orchestrator.relationship_storage.list_all()
        assert len(all_rels) == 2
