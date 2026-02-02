"""Tests for entity promotion (provisional to canonical) and merging duplicate entities.

This module verifies:
- Promotion: Changing provisional entities to canonical status when they meet
  usage count and confidence thresholds defined by the domain configuration
- Merge: Combining duplicate canonical entities, consolidating their synonyms,
  summing usage counts, and updating relationship references
"""

import pytest

from kgschema.entity import EntityStatus
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
    SimpleDomainSchema,
    make_test_entity,
    make_test_relationship,
)


@pytest.fixture
def orchestrator(
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


class TestEntityPromotion:
    """Tests for promoting provisional entities to canonical status.

    Provisional entities become canonical when they meet domain-specific
    thresholds for usage count and confidence. Promotion assigns a new
    canonical ID and updates the storage reference.
    """

    async def test_promote_updates_status(self, entity_storage: InMemoryEntityStorage) -> None:
        """Promotion changes status from PROVISIONAL to CANONICAL and assigns a new entity ID."""
        entity = make_test_entity("Test", EntityStatus.PROVISIONAL, entity_id="prov-1")
        await entity_storage.add(entity)

        promoted = await entity_storage.promote("prov-1", "canonical-1", canonical_ids={})

        assert promoted is not None
        assert promoted.status == EntityStatus.CANONICAL
        assert promoted.entity_id == "canonical-1"

    async def test_promote_updates_storage_reference(self, entity_storage: InMemoryEntityStorage) -> None:
        """Promotion replaces the old provisional ID with the new canonical ID in storage."""
        entity = make_test_entity("Test", EntityStatus.PROVISIONAL, entity_id="prov-1")
        await entity_storage.add(entity)

        await entity_storage.promote("prov-1", "canonical-1", canonical_ids={})

        # Old ID should not exist
        old = await entity_storage.get("prov-1")
        assert old is None

        # New ID should exist
        new = await entity_storage.get("canonical-1")
        assert new is not None

    async def test_promote_nonexistent_returns_none(self, entity_storage: InMemoryEntityStorage) -> None:
        """Attempting to promote a nonexistent entity ID returns None."""
        result = await entity_storage.promote("nonexistent", "canonical-1", canonical_ids={})
        assert result is None

    async def test_find_provisional_for_promotion(self, entity_storage: InMemoryEntityStorage) -> None:
        """Find provisionals meeting min_usage and min_confidence thresholds for promotion.

        Only provisional entities that exceed both the usage count and confidence
        thresholds are returned as promotion candidates. Already-canonical entities
        and those below either threshold are excluded.
        """
        # Eligible: provisional, high usage, high confidence
        eligible = make_test_entity(
            "Eligible",
            EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.9,
        )
        await entity_storage.add(eligible)

        # Not eligible: low usage
        low_usage = make_test_entity(
            "LowUsage",
            EntityStatus.PROVISIONAL,
            usage_count=1,
            confidence=0.9,
        )
        await entity_storage.add(low_usage)

        # Not eligible: low confidence
        low_conf = make_test_entity(
            "LowConf",
            EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.5,
        )
        await entity_storage.add(low_conf)

        # Not eligible: already canonical
        canonical = make_test_entity(
            "Canonical",
            EntityStatus.CANONICAL,
            usage_count=5,
            confidence=0.9,
        )
        await entity_storage.add(canonical)

        candidates = await entity_storage.find_provisional_for_promotion(min_usage=3, min_confidence=0.8)

        assert len(candidates) == 1
        assert candidates[0].name == "Eligible"

    async def test_run_promotion(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ) -> None:
        """Orchestrator run_promotion() finds and promotes all eligible provisional entities."""
        # Create eligible entity (test domain config: min_usage=2, min_confidence=0.7)
        entity = make_test_entity(
            "Promotable",
            EntityStatus.PROVISIONAL,
            entity_id="prov-promote",
            usage_count=3,
            confidence=0.8,
        )
        await entity_storage.add(entity)

        promoted = await orchestrator.run_promotion()

        assert len(promoted) == 1
        assert promoted[0].status == EntityStatus.CANONICAL


class TestEntityMerging:
    """Tests for merging duplicate canonical entities into a single target entity.

    Merging consolidates synonyms from source entities, sums usage counts,
    removes source entities from storage, and updates all relationship
    references from source IDs to the target ID.
    """

    async def test_merge_combines_synonyms(self, entity_storage: InMemoryEntityStorage) -> None:
        """Merge adds the source entity's name and synonyms to the target's synonym list."""
        target = make_test_entity("Aspirin", entity_id="target")
        target = target.model_copy(update={"synonyms": ("ASA",)})
        await entity_storage.add(target)

        source = make_test_entity("Acetylsalicylic Acid", entity_id="source")
        source = source.model_copy(update={"synonyms": ("acetyl salicylic acid",)})
        await entity_storage.add(source)

        result = await entity_storage.merge(["source"], "target")

        assert result is True
        merged = await entity_storage.get("target")
        assert merged is not None
        # Should have source name and synonyms as new synonyms
        assert "Acetylsalicylic Acid" in merged.synonyms
        assert "acetyl salicylic acid" in merged.synonyms
        assert "ASA" in merged.synonyms

    async def test_merge_combines_usage_counts(self, entity_storage: InMemoryEntityStorage) -> None:
        """Merge sums the usage counts of all source entities into the target."""
        target = make_test_entity("Target", entity_id="target", usage_count=5)
        await entity_storage.add(target)

        source1 = make_test_entity("Source1", entity_id="source1", usage_count=3)
        await entity_storage.add(source1)

        source2 = make_test_entity("Source2", entity_id="source2", usage_count=2)
        await entity_storage.add(source2)

        result = await entity_storage.merge(["source1", "source2"], "target")

        assert result is True
        merged = await entity_storage.get("target")
        assert merged is not None
        assert merged.usage_count == 10  # 5 + 3 + 2

    async def test_merge_removes_source_entities(self, entity_storage: InMemoryEntityStorage) -> None:
        """Merge deletes source entities from storage after consolidation."""
        target = make_test_entity("Target", entity_id="target")
        await entity_storage.add(target)

        source = make_test_entity("Source", entity_id="source")
        await entity_storage.add(source)

        await entity_storage.merge(["source"], "target")

        assert await entity_storage.get("source") is None
        assert await entity_storage.get("target") is not None

    async def test_merge_nonexistent_target_fails(self, entity_storage: InMemoryEntityStorage) -> None:
        """Merge fails (returns False) if the target entity ID does not exist."""
        source = make_test_entity("Source", entity_id="source")
        await entity_storage.add(source)

        result = await entity_storage.merge(["source"], "nonexistent")

        assert result is False

    async def test_merge_updates_relationships(
        self,
        orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
    ) -> None:
        """Merge rewrites all relationship subject_id/object_id references from source to target.

        Relationships that previously referenced a source entity (as subject or object)
        are updated to reference the target entity, maintaining graph connectivity.
        """
        # Create entities
        target = make_test_entity("Target", entity_id="target")
        await entity_storage.add(target)

        source = make_test_entity("Source", entity_id="source")
        await entity_storage.add(source)

        other = make_test_entity("Other", entity_id="other")
        await entity_storage.add(other)

        # Create relationships referencing source
        rel1 = make_test_relationship("source", "other")
        await relationship_storage.add(rel1)

        rel2 = make_test_relationship("other", "source")
        await relationship_storage.add(rel2)

        # Merge
        await orchestrator.merge_entities(["source"], "target")

        # Check relationships were updated
        by_subject = await relationship_storage.get_by_subject("target")
        assert len(by_subject) == 1

        by_object = await relationship_storage.get_by_object("target")
        assert len(by_object) == 1

        # Old references should be gone
        old_subject = await relationship_storage.get_by_subject("source")
        old_object = await relationship_storage.get_by_object("source")
        assert len(old_subject) == 0
        assert len(old_object) == 0
