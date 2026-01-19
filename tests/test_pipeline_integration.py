"""Integration test for the full kgraph ingestion pipeline.

This test verifies the complete end-to-end flow:
1. Batch document ingestion
2. Provisional entity creation
3. Entity promotion based on usage thresholds
4. Merge candidate detection via embedding similarity
"""

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


class TestFullPipelineIntegration:
    """End-to-end integration tests for the complete ingestion pipeline.

    These tests verify that batch ingestion, promotion, and merge candidate
    detection work together correctly in a realistic workflow.
    """

    async def test_batch_ingestion_creates_provisional_entities(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Batch ingestion creates provisional entities from document mentions.

        The mock entity extractor finds entities in [brackets]. This test verifies
        that multiple documents are processed and unique entities are stored.
        """
        docs_to_ingest = [
            (b"Paper A cites [Paper B] and [Paper C].", "text/plain", "uri:A"),
            (b"Another paper about [Paper B] which is a cool paper.", "text/plain", "uri:B"),
            (b"[Paper D] is cited by [Paper A].", "text/plain", "uri:D"),
        ]

        result = await orchestrator.ingest_batch(docs_to_ingest)

        assert result.documents_processed == 3
        # Entities extracted: Paper B, Paper C (doc1), Paper B reused (doc2),
        # Paper D, Paper A (doc3) = 4 unique entities
        assert result.total_entities_extracted >= 4

        provisional = await entity_storage.list_all(status="provisional")
        assert len(provisional) >= 4

    async def test_repeated_mentions_increase_usage_count(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Entities mentioned in multiple documents have higher usage counts."""
        docs_to_ingest = [
            (b"First doc about [SharedEntity].", "text/plain", "uri:1"),
            (b"Second doc about [SharedEntity].", "text/plain", "uri:2"),
            (b"Third doc about [SharedEntity].", "text/plain", "uri:3"),
        ]

        await orchestrator.ingest_batch(docs_to_ingest)

        entities = await entity_storage.find_by_name("SharedEntity")
        assert len(entities) == 1
        assert entities[0].usage_count >= 3

    async def test_promotion_converts_high_usage_provisionals_to_canonical(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Entities meeting usage threshold are promoted to canonical status.

        The test domain promotes entities with usage_count >= 2.
        """
        # Ingest docs where Paper B appears twice (meeting promotion threshold)
        docs_to_ingest = [
            (b"Paper A cites [Paper B].", "text/plain", "uri:A"),
            (b"Another paper about [Paper B].", "text/plain", "uri:B"),
        ]
        await orchestrator.ingest_batch(docs_to_ingest)

        # Verify Paper B has usage count >= 2 before promotion
        paper_b_list = await entity_storage.find_by_name("Paper B")
        assert len(paper_b_list) == 1
        assert paper_b_list[0].usage_count >= 2
        assert paper_b_list[0].status == EntityStatus.PROVISIONAL

        # Run promotion
        promoted = await orchestrator.run_promotion()

        # Paper B should be promoted (usage_count >= 2)
        promoted_names = {e.name for e in promoted}
        assert "Paper B" in promoted_names

        # Verify status changed to canonical
        canonical = await entity_storage.list_all(status="canonical")
        canonical_names = {e.name for e in canonical}
        assert "Paper B" in canonical_names

    async def test_merge_candidates_detected_by_embedding_similarity(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Similar canonical entities are identified as merge candidates.

        Entities with high cosine similarity in their embeddings are flagged
        as potential duplicates that may need manual review or merging.
        """
        # Add two similar entities with nearly identical embeddings
        e1 = make_test_entity(
            "USA",
            status=EntityStatus.CANONICAL,
            embedding=(0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        e2 = make_test_entity(
            "United States",
            status=EntityStatus.CANONICAL,
            embedding=(0.11, 0.22, 0.91, 0.01, 0.0, 0.0, 0.0, 0.0),
        )
        await entity_storage.add(e1)
        await entity_storage.add(e2)

        candidates = await orchestrator.find_merge_candidates(similarity_threshold=0.98)

        assert len(candidates) == 1
        found_e1, found_e2, similarity = candidates[0]

        found_names = {found_e1.name, found_e2.name}
        assert found_names == {"USA", "United States"}
        assert similarity > 0.98

    async def test_full_pipeline_ingestion_promotion_merge_detection(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Complete pipeline: ingest documents, promote entities, detect merge candidates.

        This test exercises the full workflow a user would follow:
        1. Ingest a batch of documents
        2. Run promotion to elevate high-usage entities
        3. Add similar canonical entities
        4. Detect merge candidates among canonicals
        """
        # Step 1: Batch ingestion
        docs = [
            (b"Doc 1 mentions [Alpha] and [Beta].", "text/plain", "uri:1"),
            (b"Doc 2 also mentions [Alpha] and [Gamma].", "text/plain", "uri:2"),
            (b"Doc 3 references [Alpha] again.", "text/plain", "uri:3"),
        ]
        result = await orchestrator.ingest_batch(docs)

        assert result.documents_processed == 3
        assert result.total_entities_extracted >= 3

        # Step 2: Verify Alpha has high usage count
        alpha_list = await entity_storage.find_by_name("Alpha")
        assert len(alpha_list) == 1
        assert alpha_list[0].usage_count >= 3

        # Step 3: Run promotion
        promoted = await orchestrator.run_promotion()

        # Alpha should be promoted (usage >= 2)
        promoted_names = {e.name for e in promoted}
        assert "Alpha" in promoted_names

        # Step 4: Add similar canonical entities for merge detection
        # Use 8-dimensional embeddings to match the mock generator's output
        e1 = make_test_entity(
            "NewYork",
            status=EntityStatus.CANONICAL,
            embedding=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        )
        e2 = make_test_entity(
            "New York City",
            status=EntityStatus.CANONICAL,
            embedding=(0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51),
        )
        await entity_storage.add(e1)
        await entity_storage.add(e2)

        # Step 5: Find merge candidates
        candidates = await orchestrator.find_merge_candidates(similarity_threshold=0.99)

        assert len(candidates) >= 1
        # Verify our similar pair is detected
        all_names = set()
        for c1, c2, _ in candidates:
            all_names.add(c1.name)
            all_names.add(c2.name)
        assert "NewYork" in all_names or "New York City" in all_names
