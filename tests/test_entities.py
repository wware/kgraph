"""Tests for entity creation, status management, and in-memory storage operations.

This module verifies:
- Entity instantiation with provisional vs canonical status
- Entity attributes: synonyms, embeddings, canonical IDs from authority sources
- Entity immutability (frozen Pydantic models)
- InMemoryEntityStorage CRUD operations (add, get, update, delete)
- Batch retrieval and counting
- Name-based lookups (case-insensitive, synonym-aware)
- Embedding-based similarity search with configurable thresholds
"""

import pytest

from kgraph.entity import EntityStatus
from kgraph.storage.memory import InMemoryEntityStorage

from tests.conftest import make_test_entity


class TestEntityCreation:
    """Tests for creating entities with different statuses and attributes.

    Verifies that entities can be created with provisional or canonical status,
    store multiple canonical IDs from different authority sources (e.g., UMLS,
    Wikidata), maintain synonyms for alternative names, hold semantic embeddings
    for similarity comparisons, and enforce immutability via frozen Pydantic models.
    """

    def test_create_provisional_entity(self) -> None:
        """Provisional entities are created with PROVISIONAL status and empty canonical IDs.

        Provisional entities represent mentions that have been extracted from documents
        but not yet validated against authoritative sources. They start with no
        canonical IDs and can later be promoted to canonical status.
        """
        entity = make_test_entity("Test Entity", EntityStatus.PROVISIONAL)

        assert entity.status == EntityStatus.PROVISIONAL
        assert entity.name == "Test Entity"
        assert entity.get_entity_type() == "test_entity"
        assert entity.canonical_ids == {}

    def test_create_canonical_entity(self) -> None:
        """Canonical entities store validated identifiers from multiple authority sources.

        Canonical entities have been validated and assigned stable IDs from
        authoritative sources (e.g., test_authority, wikidata). Multiple canonical
        IDs allow cross-referencing across different knowledge bases.
        """
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
        """Entities store alternative names as synonyms for improved matching.

        Synonyms enable entity resolution to match different textual mentions
        (e.g., "ASA" and "acetylsalicylic acid") to the same canonical entity.
        """
        entity = make_test_entity("Aspirin", EntityStatus.CANONICAL)
        entity = entity.model_copy(update={"synonyms": ("ASA", "acetylsalicylic acid")})

        assert "ASA" in entity.synonyms
        assert "acetylsalicylic acid" in entity.synonyms

    def test_entity_with_embedding(self) -> None:
        """Entities store semantic vector embeddings for similarity-based operations.

        Embeddings enable semantic similarity comparisons between entities,
        used for detecting potential duplicates during merge candidate detection
        and for semantic search queries.
        """
        embedding = (0.1, 0.2, 0.3, 0.4)
        entity = make_test_entity("Test", embedding=embedding)

        assert entity.embedding == embedding

    def test_entity_immutability(self) -> None:
        """Entities are immutable (frozen Pydantic models) to ensure data integrity.

        Immutability prevents accidental in-place modifications. To modify an
        entity, use model_copy(update={...}) to create a new instance, then
        persist the change through the storage layer.
        """
        entity = make_test_entity("Test")

        with pytest.raises(Exception):
            entity.name = "Modified"  # type: ignore


class TestEntityStorage:
    """Tests for InMemoryEntityStorage CRUD operations and query capabilities.

    Verifies add/get/update/delete operations, batch retrieval, name-based
    lookups (case-insensitive with synonym support), embedding-based similarity
    search with configurable thresholds, and entity counting.
    """

    async def test_add_and_get(self, entity_storage: InMemoryEntityStorage) -> None:
        """Storage supports basic add and retrieve by entity ID."""
        entity = make_test_entity("Test Entity")

        await entity_storage.add(entity)
        retrieved = await entity_storage.get(entity.entity_id)

        assert retrieved is not None
        assert retrieved.name == "Test Entity"

    async def test_get_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None:
        """Retrieving a nonexistent entity ID returns None rather than raising an error."""
        result = await entity_storage.get("nonexistent-id")
        assert result is None

    async def test_get_batch(self, entity_storage: InMemoryEntityStorage) -> None:
        """Batch retrieval returns entities in order, with None for missing IDs."""
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
        """Name-based search is case-insensitive for robust entity matching."""
        entity = make_test_entity("Aspirin")
        await entity_storage.add(entity)

        results = await entity_storage.find_by_name("aspirin")  # case insensitive

        assert len(results) == 1
        assert results[0].name == "Aspirin"

    async def test_find_by_name_with_synonyms(self, entity_storage: InMemoryEntityStorage) -> None:
        """Name-based search includes synonyms, matching alternative entity names."""
        entity = make_test_entity("Aspirin")
        entity = entity.model_copy(update={"synonyms": ("ASA",)})
        await entity_storage.add(entity)

        results = await entity_storage.find_by_name("asa")

        assert len(results) == 1
        assert results[0].name == "Aspirin"

    async def test_find_by_embedding(self, entity_storage: InMemoryEntityStorage) -> None:
        """Embedding search returns entities with cosine similarity above threshold."""
        embedding = (1.0, 0.0, 0.0, 0.0)
        entity = make_test_entity("Test", embedding=embedding)
        await entity_storage.add(entity)

        # Search with same embedding should find it
        results = await entity_storage.find_by_embedding(embedding, threshold=0.9)

        assert len(results) == 1
        assert results[0][0].name == "Test"
        assert results[0][1] == pytest.approx(1.0)

    async def test_find_by_embedding_threshold(self, entity_storage: InMemoryEntityStorage) -> None:
        """Embedding search excludes results below the similarity threshold (orthogonal vectors)."""
        entity = make_test_entity("Test", embedding=(1.0, 0.0, 0.0, 0.0))
        await entity_storage.add(entity)

        # Orthogonal vector should not match
        results = await entity_storage.find_by_embedding((0.0, 1.0, 0.0, 0.0), threshold=0.5)

        assert len(results) == 0

    async def test_update_entity(self, entity_storage: InMemoryEntityStorage) -> None:
        """Update replaces an existing entity with a modified copy (e.g., incremented usage count)."""
        entity = make_test_entity("Test", usage_count=1)
        await entity_storage.add(entity)

        updated = entity.model_copy(update={"usage_count": 5})
        result = await entity_storage.update(updated)

        assert result is True
        retrieved = await entity_storage.get(entity.entity_id)
        assert retrieved is not None
        assert retrieved.usage_count == 5

    async def test_update_nonexistent(self, entity_storage: InMemoryEntityStorage) -> None:
        """Updating a nonexistent entity returns False rather than creating it."""
        entity = make_test_entity("Test")
        result = await entity_storage.update(entity)
        assert result is False

    async def test_delete_entity(self, entity_storage: InMemoryEntityStorage) -> None:
        """Delete removes an entity from storage so subsequent get returns None."""
        entity = make_test_entity("Test")
        await entity_storage.add(entity)

        result = await entity_storage.delete(entity.entity_id)
        assert result is True

        retrieved = await entity_storage.get(entity.entity_id)
        assert retrieved is None

    async def test_count(self, entity_storage: InMemoryEntityStorage) -> None:
        """Count returns the total number of entities in storage."""
        assert await entity_storage.count() == 0

        await entity_storage.add(make_test_entity("E1"))
        await entity_storage.add(make_test_entity("E2"))

        assert await entity_storage.count() == 2
