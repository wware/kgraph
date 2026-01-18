"""Tests for entity creation and storage."""

import pytest

from kgraph.entity import EntityStatus
from kgraph.storage.memory import InMemoryEntityStorage

from tests.conftest import make_test_entity


class TestEntityCreation:
    """Tests for creating entities."""

    def test_create_provisional_entity(self) -> None:
        """Provisional entities have correct status and defaults."""
        entity = make_test_entity("Test Entity", EntityStatus.PROVISIONAL)

        assert entity.status == EntityStatus.PROVISIONAL
        assert entity.name == "Test Entity"
        assert entity.get_entity_type() == "test_entity"
        assert entity.canonical_ids == {}

    def test_create_canonical_entity(self) -> None:
        """Canonical entities can store multiple identifiers."""
        c_ids = {"test_authority": "id-123", "wikidata": "Q456"}
        entity = make_test_entity(
            "Test Entity",
            EntityStatus.CANONICAL,
            entity_id="canonical-123",
            canonical_ids=c_ids,
        )

        assert entity.status == EntityStatus.CANONICAL
        assert entity.entity_id == "canonical-123"
        assert entity.canonical_ids["test_authority"] == "id-123"
        assert entity.canonical_ids["wikidata"] == "Q456"

    def test_entity_with_synonyms(self) -> None:
        """Entities can have synonyms."""
        entity = make_test_entity("Aspirin", EntityStatus.CANONICAL)
        entity = entity.model_copy(update={"synonyms": ("ASA", "acetylsalicylic acid")})

        assert "ASA" in entity.synonyms
        assert "acetylsalicylic acid" in entity.synonyms

    def test_entity_with_embedding(self) -> None:
        """Entities can store embeddings."""
        embedding = (0.1, 0.2, 0.3, 0.4)
        entity = make_test_entity("Test", embedding=embedding)

        assert entity.embedding == embedding

    def test_entity_immutability(self) -> None:
        """Entities are immutable (frozen pydantic models)."""
        entity = make_test_entity("Test")

        with pytest.raises(Exception):
            entity.name = "Modified"  # type: ignore


class TestEntityStorage:
    """Tests for in-memory entity storage."""

    async def test_add_and_get(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can add and retrieve entities."""
        entity = make_test_entity("Test Entity")

        await entity_storage.add(entity)
        retrieved = await entity_storage.get(entity.entity_id)

        assert retrieved is not None
        assert retrieved.name == "Test Entity"

    async def test_get_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None:
        """Getting nonexistent entity returns None."""
        result = await entity_storage.get("nonexistent-id")
        assert result is None

    async def test_get_batch(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can retrieve multiple entities at once."""
        e1 = make_test_entity("Entity 1")
        e2 = make_test_entity("Entity 2")
        await entity_storage.add(e1)
        await entity_storage.add(e2)

        results = await entity_storage.get_batch([e1.entity_id, "missing", e2.entity_id])

        assert len(results) == 3
        assert results[0] is not None and results[0].name == "Entity 1"
        assert results[1] is None
        assert results[2] is not None and results[2].name == "Entity 2"

    async def test_find_by_name(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can find entities by name."""
        entity = make_test_entity("Aspirin")
        await entity_storage.add(entity)

        results = await entity_storage.find_by_name("aspirin")  # case insensitive

        assert len(results) == 1
        assert results[0].name == "Aspirin"

    async def test_find_by_name_with_synonyms(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can find entities by synonym."""
        entity = make_test_entity("Aspirin")
        entity = entity.model_copy(update={"synonyms": ("ASA",)})
        await entity_storage.add(entity)

        results = await entity_storage.find_by_name("asa")

        assert len(results) == 1
        assert results[0].name == "Aspirin"

    async def test_find_by_embedding(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can find similar entities by embedding."""
        embedding = (1.0, 0.0, 0.0, 0.0)
        entity = make_test_entity("Test", embedding=embedding)
        await entity_storage.add(entity)

        # Search with same embedding should find it
        results = await entity_storage.find_by_embedding(embedding, threshold=0.9)

        assert len(results) == 1
        assert results[0][0].name == "Test"
        assert results[0][1] == pytest.approx(1.0)

    async def test_find_by_embedding_threshold(self, entity_storage: InMemoryEntityStorage) -> None:
        """Embedding search respects similarity threshold."""
        entity = make_test_entity("Test", embedding=(1.0, 0.0, 0.0, 0.0))
        await entity_storage.add(entity)

        # Orthogonal vector should not match
        results = await entity_storage.find_by_embedding((0.0, 1.0, 0.0, 0.0), threshold=0.5)

        assert len(results) == 0

    async def test_update_entity(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can update existing entity."""
        entity = make_test_entity("Test", usage_count=1)
        await entity_storage.add(entity)

        updated = entity.model_copy(update={"usage_count": 5})
        result = await entity_storage.update(updated)

        assert result is True
        retrieved = await entity_storage.get(entity.entity_id)
        assert retrieved is not None
        assert retrieved.usage_count == 5

    async def test_update_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None:
        """Updating nonexistent entity returns False."""
        entity = make_test_entity("Test")
        result = await entity_storage.update(entity)
        assert result is False

    async def test_delete_entity(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can delete entities."""
        entity = make_test_entity("Test")
        await entity_storage.add(entity)

        result = await entity_storage.delete(entity.entity_id)
        assert result is True

        retrieved = await entity_storage.get(entity.entity_id)
        assert retrieved is None

    async def test_count(self, entity_storage: InMemoryEntityStorage) -> None:
        """Can count stored entities."""
        assert await entity_storage.count() == 0

        await entity_storage.add(make_test_entity("E1"))
        await entity_storage.add(make_test_entity("E2"))

        assert await entity_storage.count() == 2
