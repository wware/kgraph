"""Tests for entity enrichment interfaces and implementations.

This module verifies:
- EntityEnricherInterface contract and default implementations
- DBPediaEnricher functionality with mocked HTTP responses
- Integration of enrichment into the ingestion pipeline
- Backward compatibility when no enrichers are configured
- Error handling and graceful degradation
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kgraph.entity import EntityStatus
from kgraph.ingest import IngestionOrchestrator
from kgraph.pipeline.enrichment import DBPediaEnricher, EntityEnricherInterface
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
    SimpleEntity,
    SimpleDomainSchema,
    make_test_entity,
)


class TestEntityEnricherInterface:
    """Tests for enricher interface contract."""

    async def test_enrich_batch_default_implementation(self):
        """Test that default enrich_batch calls enrich_entity for each entity."""

        class SimpleEnricher(EntityEnricherInterface):
            def __init__(self):
                self.call_count = 0

            async def enrich_entity(self, entity):
                self.call_count += 1
                # Add a test canonical ID
                ids = dict(entity.canonical_ids)
                ids["test_source"] = f"test_{entity.name}"
                return entity.model_copy(update={"canonical_ids": ids})

        enricher = SimpleEnricher()
        entities = [
            make_test_entity(name="Entity1"),
            make_test_entity(name="Entity2"),
            make_test_entity(name="Entity3"),
        ]

        enriched = await enricher.enrich_batch(entities)

        assert len(enriched) == 3
        assert enricher.call_count == 3
        assert enriched[0].canonical_ids["test_source"] == "test_Entity1"
        assert enriched[1].canonical_ids["test_source"] == "test_Entity2"
        assert enriched[2].canonical_ids["test_source"] == "test_Entity3"


class TestDBPediaEnricher:
    """Tests for DBPedia enricher implementation."""

    @pytest.fixture
    def enricher(self):
        """Create a DBPedia enricher with test configuration."""
        return DBPediaEnricher(
            entity_types_to_enrich=None,  # Enrich all types
            confidence_threshold=0.7,
            min_lookup_score=0.5,
            cache_results=True,
            timeout=5.0,
        )

    async def test_enrich_entity_success(self, enricher):
        """Test successful DBPedia lookup and enrichment."""
        entity = make_test_entity(
            name="Sherlock Holmes",
            confidence=0.9,
        )

        # Mock DBPedia API response
        mock_response = {
            "docs": [
                {
                    "label": ["Sherlock Holmes"],
                    "resource": ["http://dbpedia.org/resource/Sherlock_Holmes"],
                    "score": 0.95,
                }
            ]
        }

        with patch.object(enricher, "_query_dbpedia", return_value=mock_response["docs"]):
            enriched = await enricher.enrich_entity(entity)

        assert "dbpedia" in enriched.canonical_ids
        assert enriched.canonical_ids["dbpedia"] == "http://dbpedia.org/resource/Sherlock_Holmes"
        assert enriched.name == entity.name  # Other fields unchanged

    async def test_enrich_entity_no_match(self, enricher):
        """Test graceful handling when no DBPedia match found."""
        entity = make_test_entity(
            name="NonexistentEntity12345",
            confidence=0.9,
        )

        # Mock empty response
        with patch.object(enricher, "_query_dbpedia", return_value=[]):
            enriched = await enricher.enrich_entity(entity)

        # Should return original entity unchanged
        assert "dbpedia" not in enriched.canonical_ids
        assert enriched.entity_id == entity.entity_id

    async def test_enrich_entity_below_confidence_threshold(self):
        """Test skipping enrichment for low-confidence entities."""
        enricher = DBPediaEnricher(confidence_threshold=0.8)
        entity = make_test_entity(name="Test Entity", confidence=0.5)

        # Should not even query DBPedia
        with patch.object(enricher, "_query_dbpedia") as mock_query:
            enriched = await enricher.enrich_entity(entity)

        mock_query.assert_not_called()
        assert "dbpedia" not in enriched.canonical_ids

    async def test_enrich_entity_filtered_by_type(self):
        """Test entity type filtering."""
        enricher = DBPediaEnricher(
            entity_types_to_enrich={"person", "location"},
            confidence_threshold=0.7,
        )

        entity = SimpleEntity(
            entity_id="test_id",
            status=EntityStatus.PROVISIONAL,
            name="Test Entity",
            entity_type="other_type",
            confidence=0.9,
            usage_count=0,
            created_at=datetime.now(timezone.utc),
            source="test",
        )

        # Should skip enrichment for types not in whitelist
        with patch.object(enricher, "_query_dbpedia") as mock_query:
            enriched = await enricher.enrich_entity(entity)

        mock_query.assert_not_called()
        assert "dbpedia" not in enriched.canonical_ids

    async def test_disambiguation_by_exact_match(self, enricher):
        """Test disambiguation using exact label matching."""
        entity = make_test_entity(name="London", confidence=0.9)

        # Mock multiple results
        mock_results = [
            {
                "label": ["London, Ontario"],
                "resource": ["http://dbpedia.org/resource/London,_Ontario"],
                "score": 0.6,
            },
            {
                "label": ["London"],
                "resource": ["http://dbpedia.org/resource/London"],
                "score": 0.9,
            },
        ]

        with patch.object(enricher, "_query_dbpedia", return_value=mock_results):
            enriched = await enricher.enrich_entity(entity)

        # Should pick exact match
        assert enriched.canonical_ids["dbpedia"] == "http://dbpedia.org/resource/London"

    async def test_disambiguation_by_synonyms(self, enricher):
        """Test using entity synonyms for disambiguation."""
        entity = SimpleEntity(
            entity_id=str(uuid.uuid4()),
            status=EntityStatus.PROVISIONAL,
            name="NYC",
            synonyms=("New York City", "New York"),
            confidence=0.9,
            usage_count=0,
            created_at=datetime.now(timezone.utc),
            source="test",
        )

        mock_results = [
            {
                "label": ["New York City"],
                "resource": ["http://dbpedia.org/resource/New_York_City"],
                "score": 0.85,
            }
        ]

        with patch.object(enricher, "_query_dbpedia", return_value=mock_results):
            enriched = await enricher.enrich_entity(entity)

        # Should match via synonym
        assert "dbpedia" in enriched.canonical_ids
        assert "New_York_City" in enriched.canonical_ids["dbpedia"]

    async def test_caching(self, enricher):
        """Test that repeated lookups use cache."""
        entity = make_test_entity(name="Test Entity", confidence=0.9)

        mock_results = [
            {
                "label": ["Test Entity"],
                "resource": ["http://dbpedia.org/resource/Test_Entity"],
                "score": 0.8,
            }
        ]

        with patch.object(enricher, "_query_dbpedia", return_value=mock_results) as mock_query:
            # First call should query API
            enriched1 = await enricher.enrich_entity(entity)
            assert mock_query.call_count == 1

            # Second call should use cache
            enriched2 = await enricher.enrich_entity(entity)
            assert mock_query.call_count == 1  # Still 1, not called again

        assert enriched1.canonical_ids["dbpedia"] == enriched2.canonical_ids["dbpedia"]

    async def test_caching_negative_results(self, enricher):
        """Test that failed lookups are also cached."""
        entity = make_test_entity(name="Nonexistent", confidence=0.9)

        with patch.object(enricher, "_query_dbpedia", return_value=[]) as mock_query:
            # First call
            await enricher.enrich_entity(entity)
            assert mock_query.call_count == 1

            # Second call should use cached negative result
            await enricher.enrich_entity(entity)
            assert mock_query.call_count == 1

    async def test_http_error_handling(self, enricher):
        """Test graceful handling of API errors."""
        entity = make_test_entity(name="Test", confidence=0.9)

        # Mock API error
        with patch.object(enricher, "_query_dbpedia", side_effect=Exception("HTTP timeout")):
            enriched = await enricher.enrich_entity(entity)

        # Should return original entity
        assert "dbpedia" not in enriched.canonical_ids
        assert enriched.entity_id == entity.entity_id

    async def test_below_min_lookup_score(self, enricher):
        """Test that low-scoring matches are rejected."""
        entity = make_test_entity(name="Test", confidence=0.9)

        # Mock low-quality results
        mock_results = [
            {
                "label": ["Different Entity"],
                "resource": ["http://dbpedia.org/resource/Different"],
                "score": 0.3,  # Below min_lookup_score
            }
        ]

        with patch.object(enricher, "_query_dbpedia", return_value=mock_results):
            enriched = await enricher.enrich_entity(entity)

        # Should not add low-scoring match
        assert "dbpedia" not in enriched.canonical_ids

    async def test_cache_disabled(self):
        """Test enricher works with caching disabled."""
        enricher = DBPediaEnricher(cache_results=False, confidence_threshold=0.7)
        entity = make_test_entity(name="Test", confidence=0.9)

        mock_results = [
            {
                "label": ["Test"],
                "resource": ["http://dbpedia.org/resource/Test"],
                "score": 0.8,
            }
        ]

        with patch.object(enricher, "_query_dbpedia", return_value=mock_results) as mock_query:
            # Multiple calls should all query API
            await enricher.enrich_entity(entity)
            await enricher.enrich_entity(entity)
            assert mock_query.call_count == 2


class TestOrchestrationWithEnrichment:
    """Integration tests for enrichment in ingestion pipeline."""

    @pytest.fixture
    def orchestrator_with_enricher(
        self,
        test_domain: SimpleDomainSchema,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
        document_storage: InMemoryDocumentStorage,
        document_parser: MockDocumentParser,
        entity_extractor: MockEntityExtractor,
        entity_resolver: MockEntityResolver,
        relationship_extractor: MockRelationshipExtractor,
        embedding_generator: MockEmbeddingGenerator,
    ) -> IngestionOrchestrator:
        """Create orchestrator with a mock enricher."""

        # Create a simple mock enricher
        class MockEnricher(EntityEnricherInterface):
            async def enrich_entity(self, entity):
                ids = dict(entity.canonical_ids)
                ids["mock_source"] = f"mock_{entity.name}"
                return entity.model_copy(update={"canonical_ids": ids})

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
            entity_enrichers=[MockEnricher()],
        )

    async def test_orchestrator_with_enricher(
        self,
        orchestrator_with_enricher: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ):
        """Test full ingestion with enrichment enabled."""
        content = b"Document mentioning [Entity A]."

        result = await orchestrator_with_enricher.ingest_document(content, "text/plain")

        assert result.entities_extracted == 1
        assert result.entities_new == 1

        # Verify entity was enriched
        entities = await entity_storage.list_all()
        assert len(entities) == 1
        entity = entities[0]
        assert "mock_source" in entity.canonical_ids
        assert entity.canonical_ids["mock_source"] == "mock_Entity A"

    async def test_orchestrator_without_enricher(
        self,
        test_domain: SimpleDomainSchema,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
        document_storage: InMemoryDocumentStorage,
        document_parser: MockDocumentParser,
        entity_extractor: MockEntityExtractor,
        entity_resolver: MockEntityResolver,
        relationship_extractor: MockRelationshipExtractor,
        embedding_generator: MockEmbeddingGenerator,
    ):
        """Test that enrichment is optional (backward compatibility)."""
        orchestrator = IngestionOrchestrator(
            domain=test_domain,
            parser=document_parser,
            entity_extractor=entity_extractor,
            entity_resolver=entity_resolver,
            relationship_extractor=relationship_extractor,
            embedding_generator=embedding_generator,
            entity_storage=entity_storage,
            relationship_storage=relationship_storage,
            document_storage=document_storage,
            # No enrichers
        )

        content = b"Document with [Entity B]."
        result = await orchestrator.ingest_document(content, "text/plain")

        assert result.entities_extracted == 1
        assert result.entities_new == 1

        # Verify entity exists without enrichment
        entities = await entity_storage.list_all()
        assert len(entities) == 1
        entity = entities[0]
        assert len(entity.canonical_ids) == 0  # No enrichment

    async def test_enrichment_error_doesnt_break_pipeline(
        self,
        test_domain: SimpleDomainSchema,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
        document_storage: InMemoryDocumentStorage,
        document_parser: MockDocumentParser,
        entity_extractor: MockEntityExtractor,
        entity_resolver: MockEntityResolver,
        relationship_extractor: MockRelationshipExtractor,
        embedding_generator: MockEmbeddingGenerator,
    ):
        """Test that enricher errors don't break the ingestion pipeline."""

        class FailingEnricher(EntityEnricherInterface):
            async def enrich_entity(self, entity):
                raise Exception("Enrichment failed!")

        orchestrator = IngestionOrchestrator(
            domain=test_domain,
            parser=document_parser,
            entity_extractor=entity_extractor,
            entity_resolver=entity_resolver,
            relationship_extractor=relationship_extractor,
            embedding_generator=embedding_generator,
            entity_storage=entity_storage,
            relationship_storage=relationship_storage,
            document_storage=document_storage,
            entity_enrichers=[FailingEnricher()],
        )

        content = b"Document with [Entity C]."
        result = await orchestrator.ingest_document(content, "text/plain")

        # Pipeline should continue despite enrichment failure
        assert result.entities_extracted == 1
        assert result.entities_new == 1
        assert len(result.errors) > 0  # Error should be logged
        assert "enrichment failed" in result.errors[0].lower()

    async def test_multiple_enrichers_chained(
        self,
        test_domain: SimpleDomainSchema,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
        document_storage: InMemoryDocumentStorage,
        document_parser: MockDocumentParser,
        entity_extractor: MockEntityExtractor,
        entity_resolver: MockEntityResolver,
        relationship_extractor: MockRelationshipExtractor,
        embedding_generator: MockEmbeddingGenerator,
    ):
        """Test that multiple enrichers can be chained."""

        class Enricher1(EntityEnricherInterface):
            async def enrich_entity(self, entity):
                ids = dict(entity.canonical_ids)
                ids["source1"] = "value1"
                return entity.model_copy(update={"canonical_ids": ids})

        class Enricher2(EntityEnricherInterface):
            async def enrich_entity(self, entity):
                ids = dict(entity.canonical_ids)
                ids["source2"] = "value2"
                return entity.model_copy(update={"canonical_ids": ids})

        orchestrator = IngestionOrchestrator(
            domain=test_domain,
            parser=document_parser,
            entity_extractor=entity_extractor,
            entity_resolver=entity_resolver,
            relationship_extractor=relationship_extractor,
            embedding_generator=embedding_generator,
            entity_storage=entity_storage,
            relationship_storage=relationship_storage,
            document_storage=document_storage,
            entity_enrichers=[Enricher1(), Enricher2()],
        )

        content = b"Document with [Entity D]."
        result = await orchestrator.ingest_document(content, "text/plain")

        # Verify both enrichers ran
        entities = await entity_storage.list_all()
        assert len(entities) == 1
        entity = entities[0]
        assert "source1" in entity.canonical_ids
        assert "source2" in entity.canonical_ids
        assert entity.canonical_ids["source1"] == "value1"
        assert entity.canonical_ids["source2"] == "value2"
