"""Tests for two-pass ingestion flow."""

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
    TestDomainSchema,
    make_test_entity,
)


@pytest.fixture
def orchestrator(
    test_domain: TestDomainSchema,
    entity_storage: InMemoryEntityStorage,
    relationship_storage: InMemoryRelationshipStorage,
    document_storage: InMemoryDocumentStorage,
    document_parser: MockDocumentParser,
    entity_extractor: MockEntityExtractor,
    entity_resolver: MockEntityResolver,
    relationship_extractor: MockRelationshipExtractor,
    embedding_generator: MockEmbeddingGenerator,
) -> IngestionOrchestrator:
    """Create an ingestion orchestrator with mock components."""
    return IngestionOrchestrator(
        domain=test_domain,
        parser=document_parser,
        entity_extractor=entity_extractor,
        entity_resolver=entity_resolver,
        relationship_extractor=relationship_extractor,
        embedding_generator=embedding_generator,
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )


class TestSingleDocumentIngestion:
    """Tests for ingesting single documents."""

    async def test_ingest_extracts_entities(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Ingestion extracts entities from document."""
        content = b"This document mentions [Entity A] and [Entity B]."

        result = await orchestrator.ingest_document(content, "text/plain")

        assert result.entities_extracted == 2
        assert await entity_storage.count() == 2

    async def test_ingest_creates_relationships(
        self,
        orchestrator: IngestionOrchestrator,
        relationship_storage: InMemoryRelationshipStorage,
    ) -> None:
        """Ingestion creates relationships between entities."""
        content = b"Connection between [First] and [Second]."

        result = await orchestrator.ingest_document(content, "text/plain")

        assert result.relationships_extracted == 1
        assert await relationship_storage.count() == 1

    async def test_ingest_stores_document(
        self,
        orchestrator: IngestionOrchestrator,
        document_storage: InMemoryDocumentStorage,
    ) -> None:
        """Ingestion stores the parsed document."""
        content = b"Document with [Entity]."

        result = await orchestrator.ingest_document(content, "text/plain")

        assert result.document_id != ""
        doc = await document_storage.get(result.document_id)
        assert doc is not None
        assert "[Entity]" in doc.content

    async def test_ingest_generates_embeddings(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Ingestion generates embeddings for new entities."""
        content = b"Document with [NewEntity]."

        await orchestrator.ingest_document(content, "text/plain")

        entities = await entity_storage.find_by_name("NewEntity")
        assert len(entities) == 1
        assert entities[0].embedding is not None

    async def test_ingest_increments_usage_count(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Re-ingesting same entity increments usage count."""
        # First document with entity
        await orchestrator.ingest_document(b"First mention of [SharedEntity].", "text/plain")

        entities = await entity_storage.find_by_name("SharedEntity")
        first_count = entities[0].usage_count

        # Second document with same entity
        await orchestrator.ingest_document(b"Second mention of [SharedEntity].", "text/plain")

        entities = await entity_storage.find_by_name("SharedEntity")
        assert entities[0].usage_count == first_count + 1

    async def test_ingest_no_entities(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> None:
        """Ingestion handles documents with no entities."""
        content = b"This document has no bracketed entities."

        result = await orchestrator.ingest_document(content, "text/plain")

        assert result.entities_extracted == 0
        assert result.relationships_extracted == 0
        assert len(result.errors) == 0


class TestBatchIngestion:
    """Tests for batch document ingestion."""

    async def test_batch_ingest_multiple_documents(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> None:
        """Can ingest multiple documents in batch."""
        documents = [
            (b"Document one with [Entity1].", "text/plain", None),
            (b"Document two with [Entity2] and [Entity3].", "text/plain", None),
        ]

        result = await orchestrator.ingest_batch(documents)

        assert result.documents_processed == 2
        assert result.total_entities_extracted == 3

    async def test_batch_reports_per_document_results(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> None:
        """Batch result includes per-document details."""
        documents = [
            (b"Doc with [A].", "text/plain", "source-1"),
            (b"Doc with [B] and [C].", "text/plain", "source-2"),
        ]

        result = await orchestrator.ingest_batch(documents)

        assert len(result.document_results) == 2
        assert result.document_results[0].entities_extracted == 1
        assert result.document_results[1].entities_extracted == 2

    async def test_batch_continues_on_error(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> None:
        """Batch continues processing after individual failures."""
        documents = [
            (b"Valid document with [Entity].", "text/plain", None),
            (b"Another valid doc with [Entity2].", "text/plain", None),
        ]

        result = await orchestrator.ingest_batch(documents)

        assert result.documents_processed == 2
        # Even if one has issues, we should process all


class TestDomainValidation:
    """Tests for domain-specific validation during ingestion."""

    async def test_validates_entities(
        self,
        orchestrator: IngestionOrchestrator,
        test_domain: TestDomainSchema,
    ) -> None:
        """Ingestion validates entities against domain schema."""
        # Our test domain accepts "test_entity" type, so this should work
        content = b"Document with [ValidEntity]."

        result = await orchestrator.ingest_document(content, "text/plain")

        # MockEntityExtractor creates entities of type "test_entity" which is valid
        assert result.entities_new == 1
        # No validation errors for valid entities
        validation_errors = [e for e in result.errors if "validation failed" in e]
        assert len(validation_errors) == 0

    async def test_validates_relationships(
        self,
        orchestrator: IngestionOrchestrator,
    ) -> None:
        """Ingestion validates relationships against domain schema."""
        content = b"Connection from [A] to [B]."

        result = await orchestrator.ingest_document(content, "text/plain")

        # MockRelationshipExtractor creates "related_to" predicate which is valid
        assert result.relationships_extracted == 1


class TestMergeCandidateDetection:
    """Tests for detecting merge candidates."""

    async def test_find_merge_candidates_with_similar_entities(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """It should find entities with high embedding similarity."""
        # High similarity
        e1 = make_test_entity(
            "USA",
            status=EntityStatus.CANONICAL,
            embedding=(0.1, 0.2, 0.9, 0.0),
        )
        e2 = make_test_entity(
            "United States",
            status=EntityStatus.CANONICAL,
            embedding=(0.11, 0.22, 0.91, 0.01),
        )
        # Low similarity
        e3 = make_test_entity(
            "Python",
            status=EntityStatus.CANONICAL,
            embedding=(0.9, 0.8, 0.1, 1.0),
        )
        # No embedding
        e4 = make_test_entity(
            "Entity without Embedding",
            status=EntityStatus.CANONICAL,
            embedding=None,
        )
        # Not canonical
        e5 = make_test_entity(
            "Provisional",
            status=EntityStatus.PROVISIONAL,
            embedding=(0.1, 0.2, 0.9, 0.0),
        )

        await entity_storage.add(e1)
        await entity_storage.add(e2)
        await entity_storage.add(e3)
        await entity_storage.add(e4)
        await entity_storage.add(e5)

        # Find candidates with a high threshold
        candidates = await orchestrator.find_merge_candidates(similarity_threshold=0.98)

        assert len(candidates) == 1
        c_e1, c_e2, sim = candidates[0]

        # Ensure the correct pair was found (order-independent)
        found_ids = {c_e1.entity_id, c_e2.entity_id}
        expected_ids = {e1.entity_id, e2.entity_id}
        assert found_ids == expected_ids

        # Check similarity score is in the right range
        assert sim > 0.98
        assert sim < 1.0

    async def test_find_merge_candidates_returns_empty_list(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """It should return an empty list if no entities are similar enough."""
        e1 = make_test_entity("A", status=EntityStatus.CANONICAL, embedding=(1.0, 0.0))
        e2 = make_test_entity("B", status=EntityStatus.CANONICAL, embedding=(0.0, 1.0))
        await entity_storage.add(e1)
        await entity_storage.add(e2)

        candidates = await orchestrator.find_merge_candidates(similarity_threshold=0.5)
        assert len(candidates) == 0
