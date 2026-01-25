"""Tests for entity promotion policy and workflow.

This test module covers:
1. PromotionPolicy base class behavior
2. Domain-specific promotion policies (Sherlock example)
3. Full promotion workflow with relationship updates
4. Entities starting as provisional with canonical_id_hint
"""

import pytest
from datetime import datetime, timezone

from kgraph.entity import BaseEntity, EntityStatus, PromotionConfig
from kgraph.promotion import PromotionPolicy
from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
    InMemoryDocumentStorage,
)
from kgraph.domain import DomainSchema
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from examples.sherlock.domain import SherlockDomainSchema
from examples.sherlock.pipeline import (
    SherlockDocumentParser,
    SherlockEntityExtractor,
    SherlockEntityResolver,
    SherlockRelationshipExtractor,
    SimpleEmbeddingGenerator,
)

# Import test fixtures from conftest
from tests.conftest import (
    make_test_entity,
)

# ============================================================================
# Test PromotionPolicy Base Class
# ============================================================================


class SimplePromotionPolicy(PromotionPolicy):
    """Test implementation with hardcoded mappings."""

    CANONICAL_IDS = {
        "test:entity:Alpha": "canonical:Alpha",
        "test:entity:Beta": "canonical:Beta",
    }

    async def assign_canonical_id(self, entity: BaseEntity) -> str | None:
        return self.CANONICAL_IDS.get(entity.entity_id)


class TestPromotionPolicyBase:
    """Test the base PromotionPolicy class behavior."""

    def test_should_promote_rejects_already_canonical(self):
        """should_promote returns False for entities already canonical."""
        config = PromotionConfig(min_usage_count=1, min_confidence=0.5)
        policy = SimplePromotionPolicy(config)

        entity = make_test_entity(
            "Test",
            entity_id="canonical:Test",
            status=EntityStatus.CANONICAL,
            usage_count=10,
            confidence=0.9,
        )

        assert policy.should_promote(entity) is False

    def test_should_promote_requires_min_usage(self):
        """should_promote checks minimum usage count threshold."""
        config = PromotionConfig(min_usage_count=3, min_confidence=0.5, require_embedding=False)
        policy = SimplePromotionPolicy(config)

        # Below threshold
        entity_low = make_test_entity(
            "Low",
            status=EntityStatus.PROVISIONAL,
            usage_count=2,
            confidence=0.9,
        )
        assert policy.should_promote(entity_low) is False

        # Meets threshold
        entity_ok = make_test_entity(
            "OK",
            status=EntityStatus.PROVISIONAL,
            usage_count=4,
            confidence=0.9,
        )
        assert policy.should_promote(entity_ok) is True

    def test_should_promote_requires_min_confidence(self):
        """should_promote checks minimum confidence threshold."""
        config = PromotionConfig(min_usage_count=1, min_confidence=0.8, require_embedding=False)
        policy = SimplePromotionPolicy(config)

        # Below threshold
        entity_low = make_test_entity(
            "Low",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.7,
        )
        assert policy.should_promote(entity_low) is False

        # Meets threshold
        entity_ok = make_test_entity(
            "OK",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.8,
        )
        assert policy.should_promote(entity_ok) is True

    def test_should_promote_checks_embedding_requirement(self):
        """should_promote respects require_embedding config."""
        config = PromotionConfig(
            min_usage_count=1,
            min_confidence=0.5,
            require_embedding=True,
        )
        policy = SimplePromotionPolicy(config)

        # No embedding
        entity_no_emb = make_test_entity(
            "NoEmb",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.9,
            embedding=None,
        )
        assert policy.should_promote(entity_no_emb) is False

        # Has embedding
        entity_with_emb = make_test_entity(
            "WithEmb",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.9,
            embedding=(0.1, 0.2, 0.3),
        )
        assert policy.should_promote(entity_with_emb) is True

    async def test_assign_canonical_id_returns_mapping(self):
        """assign_canonical_id returns mapped ID or None."""
        config = PromotionConfig(min_usage_count=1, min_confidence=0.5)
        policy = SimplePromotionPolicy(config)

        # Entity with mapping
        entity_mapped = make_test_entity(
            "Alpha",
            entity_id="test:entity:Alpha",
            status=EntityStatus.PROVISIONAL,
        )
        assert await policy.assign_canonical_id(entity_mapped) == "canonical:Alpha"

        # Entity without mapping
        entity_unmapped = make_test_entity(
            "Unknown",
            entity_id="test:entity:Unknown",
            status=EntityStatus.PROVISIONAL,
        )
        assert await policy.assign_canonical_id(entity_unmapped) is None


# ============================================================================
# Test Sherlock Domain Promotion
# ============================================================================


class TestSherlockPromotion:
    """Test Sherlock-specific promotion policy with DBPedia mappings."""

    async def test_sherlock_policy_has_dbpedia_mappings(self):
        """SherlockPromotionPolicy contains DBPedia URI mappings."""
        from examples.sherlock.promotion import SherlockPromotionPolicy

        config = PromotionConfig(min_usage_count=2, min_confidence=0.7)
        policy = SherlockPromotionPolicy(config)

        # Check known mapping
        entity = make_test_entity(
            "Sherlock Holmes",
            entity_id="holmes:char:SherlockHolmes",
            status=EntityStatus.PROVISIONAL,
        )

        canonical_id = await policy.assign_canonical_id(entity)
        assert canonical_id == "http://dbpedia.org/resource/Sherlock_Holmes"

        # Check unmapped entity
        unknown = make_test_entity(
            "Unknown Character",
            entity_id="holmes:char:Unknown",
            status=EntityStatus.PROVISIONAL,
        )
        assert await policy.assign_canonical_id(unknown) is None

    def test_sherlock_promotion_config_has_low_thresholds(self):
        """Sherlock domain uses lower thresholds for small corpus."""
        from examples.sherlock.domain import SherlockDomainSchema

        schema = SherlockDomainSchema()
        config = schema.promotion_config

        # Sherlock uses lenient thresholds since it's a small, curated domain
        assert config.min_usage_count == 2
        assert config.min_confidence == 0.7
        assert config.require_embedding is False

    def test_get_promotion_policy_accepts_lookup_parameter(self):
        """get_promotion_policy accepts lookup parameter for signature compliance."""
        from examples.sherlock.domain import SherlockDomainSchema

        schema = SherlockDomainSchema()
        # Should work with None (Sherlock doesn't use lookup)
        policy1 = schema.get_promotion_policy(lookup=None)
        assert policy1 is not None

        # Should also work with a mock lookup (even if ignored)
        mock_lookup = object()
        policy2 = schema.get_promotion_policy(lookup=mock_lookup)
        assert policy2 is not None


# ============================================================================
# Test Promotion Workflow
# ============================================================================


@pytest.fixture
async def orchestrator(
    test_domain: DomainSchema,
    entity_storage: InMemoryEntityStorage,
    relationship_storage: InMemoryRelationshipStorage,
    document_storage: InMemoryDocumentStorage,
    document_parser: DocumentParserInterface,
    entity_extractor: EntityExtractorInterface,
    entity_resolver: EntityResolverInterface,
    relationship_extractor: RelationshipExtractorInterface,
    embedding_generator: EmbeddingGeneratorInterface,
):
    """Create a generic orchestrator for testing."""

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


@pytest.fixture
async def sherlock_orchestrator(
    entity_storage: InMemoryEntityStorage,
    relationship_storage: InMemoryRelationshipStorage,
    document_storage: InMemoryDocumentStorage,
):
    """Create orchestrator with Sherlock domain for promotion testing."""

    domain = SherlockDomainSchema()
    return IngestionOrchestrator(
        domain=domain,
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(domain=domain),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )


class TestPromotionWorkflow:
    """Test complete promotion workflow with relationship updates."""

    async def test_entities_start_as_provisional_with_canonical_hint(
        self,
        orchestrator: IngestionOrchestrator,  # Use generic test domain
        entity_storage: InMemoryEntityStorage,
    ):
        """Entities created with canonical_id_hint still start as PROVISIONAL."""
        doc_content = b"Test mentions [Entity A] and [Entity B]."

        await orchestrator.ingest_document(
            raw_content=doc_content,
            content_type="text/plain",
            source_uri="test",
        )

        # Generic test entities should be provisional
        entities = await entity_storage.list_all(status="provisional")
        assert len(entities) > 0

    async def test_promotion_changes_id_and_status(
        self,
        sherlock_orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ):
        """Promotion updates entity_id to DBPedia URI and status to CANONICAL."""
        # Create provisional entity meeting promotion criteria
        from examples.sherlock.domain import SherlockCharacter

        holmes = SherlockCharacter(
            entity_id="holmes:char:SherlockHolmes",
            name="Sherlock Holmes",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,  # Exceeds threshold
            confidence=0.9,  # Exceeds threshold
            created_at=datetime.now(timezone.utc),
            source="test",
        )
        await entity_storage.add(holmes)

        # Run promotion
        promoted = await sherlock_orchestrator.run_promotion()

        assert len(promoted) == 1
        promoted_holmes = promoted[0]

        # Check new ID and status
        assert promoted_holmes.entity_id == "http://dbpedia.org/resource/Sherlock_Holmes"
        assert promoted_holmes.status == EntityStatus.CANONICAL
        assert promoted_holmes.name == "Sherlock Holmes"

    async def test_promotion_updates_relationship_references(
        self,
        sherlock_orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
        relationship_storage: InMemoryRelationshipStorage,
    ):
        """Promotion updates relationships to point to new canonical ID."""
        from examples.sherlock.domain import SherlockCharacter, SherlockStory, AppearsInRelationship

        # Create provisional entities
        holmes = SherlockCharacter(
            entity_id="holmes:char:SherlockHolmes",
            name="Sherlock Holmes",
            status=EntityStatus.PROVISIONAL,
            usage_count=5,
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
            source="test",
        )

        story = SherlockStory(
            entity_id="holmes:story:AScandalInBohemia",
            name="A Scandal in Bohemia",
            status=EntityStatus.PROVISIONAL,
            usage_count=4,
            confidence=0.95,
            created_at=datetime.now(timezone.utc),
            source="test",
        )

        await entity_storage.add(holmes)
        await entity_storage.add(story)

        # Create relationship using provisional IDs
        rel = AppearsInRelationship(
            subject_id="holmes:char:SherlockHolmes",
            predicate="appears_in",
            object_id="holmes:story:AScandalInBohemia",
            confidence=0.95,
            source_documents=("test-doc",),
            created_at=datetime.now(timezone.utc),
        )
        await relationship_storage.add(rel)

        # Run promotion
        await sherlock_orchestrator.run_promotion()

        # Check relationship now uses DBPedia URI
        all_rels = await relationship_storage.list_all()
        rels = [r for r in all_rels if r.predicate == "appears_in"]
        assert len(rels) > 0

        updated_rel = rels[0]
        assert updated_rel.subject_id == "http://dbpedia.org/resource/Sherlock_Holmes"
        assert updated_rel.object_id == "http://dbpedia.org/resource/A_Scandal_in_Bohemia"

    async def test_entities_without_mapping_remain_provisional(
        self,
        sherlock_orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ):
        """Entities without canonical ID mapping stay provisional even if eligible."""
        from examples.sherlock.domain import SherlockCharacter

        # Create entity with high usage but no mapping
        unknown = SherlockCharacter(
            entity_id="holmes:char:UnknownPerson",
            name="Unknown Person",
            status=EntityStatus.PROVISIONAL,
            usage_count=10,  # High usage
            confidence=0.95,  # High confidence
            created_at=datetime.now(timezone.utc),
            source="test",
        )
        await entity_storage.add(unknown)

        # Run promotion
        promoted = await sherlock_orchestrator.run_promotion()

        # Unknown not promoted (no mapping in SHERLOCK_CANONICAL_IDS)
        promoted_ids = {e.entity_id for e in promoted}
        assert "holmes:char:UnknownPerson" not in promoted_ids

        # Verify still provisional
        still_provisional = await entity_storage.get("holmes:char:UnknownPerson")
        assert still_provisional is not None
        assert still_provisional.status == EntityStatus.PROVISIONAL

    async def test_low_usage_entities_not_promoted(
        self,
        sherlock_orchestrator: IngestionOrchestrator,
        entity_storage: InMemoryEntityStorage,
    ):
        """Entities below usage threshold aren't promoted even with mapping."""
        from examples.sherlock.domain import SherlockCharacter

        # Create Holmes with low usage
        holmes = SherlockCharacter(
            entity_id="holmes:char:SherlockHolmes",
            name="Sherlock Holmes",
            status=EntityStatus.PROVISIONAL,
            usage_count=1,  # Below threshold of 2
            confidence=0.9,
            created_at=datetime.now(timezone.utc),
            source="test",
        )
        await entity_storage.add(holmes)

        # Run promotion
        promoted = await sherlock_orchestrator.run_promotion()

        # Not promoted due to low usage
        assert len(promoted) == 0

        # Still provisional
        still_provisional = await entity_storage.get("holmes:char:SherlockHolmes")
        assert still_provisional is not None  # Ensure entity is found
        assert still_provisional.status == EntityStatus.PROVISIONAL


# ============================================================================
# Integration Test: Full Pipeline
# ============================================================================


class TestPromotionIntegration:
    """Test promotion in complete ingestion pipeline."""

    async def test_full_sherlock_ingestion_and_promotion(
        self,
        sherlock_orchestrator: IngestionOrchestrator,
    ):
        entity_storage = sherlock_orchestrator.entity_storage
        # relationship_storage = sherlock_orchestrator.relationship_storage
        """Complete workflow: ingest stories → entities accumulate usage → promote."""
        # Ingest multiple documents mentioning Holmes
        stories = [
            b"Sherlock Holmes and Dr. Watson investigated the case.",
            b"Holmes deduced the truth from the evidence.",
            b"The great detective Sherlock Holmes solved another mystery.",
        ]

        for content in stories:
            await sherlock_orchestrator.ingest_document(
                raw_content=content,
                content_type="text/plain",
                source_uri="A Scandal in Bohemia",
            )

        all_entities = await entity_storage.list_all()
        holmes = None
        for e in all_entities:
            if e.name == "Sherlock Holmes":
                holmes = e
                break

        assert holmes is not None
        assert holmes.status == EntityStatus.PROVISIONAL
        assert holmes.usage_count >= 2  # Mentioned in multiple stories
        promoted = await sherlock_orchestrator.run_promotion()

        # Holmes should be promoted
        promoted_names = {e.name for e in promoted}
        assert "Sherlock Holmes" in promoted_names

        # Verify canonical status
        canonical_entities = await entity_storage.list_all(status="canonical")
        canonical_names = {e.name for e in canonical_entities}
        assert "Sherlock Holmes" in canonical_names
