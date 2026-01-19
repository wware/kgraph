"""Tests for relationship (edge) creation and in-memory storage operations.

This module verifies:
- Relationship instantiation with subject, predicate, and object
- Relationship attributes: metadata, source documents, immutability
- InMemoryRelationshipStorage CRUD operations (add, find, delete)
- Queries by subject, object, and triple (subject-predicate-object)
- Updating entity references when entities are merged
"""

import pytest

from kgraph.storage.memory import InMemoryRelationshipStorage

from tests.conftest import make_test_relationship


class TestRelationshipCreation:
    """Tests for creating relationship (edge) instances.

    Relationships represent directed edges in the knowledge graph, connecting
    a subject entity to an object entity via a predicate (edge type).
    """

    def test_create_relationship(self) -> None:
        """Relationships have a subject_id, predicate, and object_id forming a directed edge."""
        rel = make_test_relationship("entity-1", "entity-2", "related_to")

        assert rel.subject_id == "entity-1"
        assert rel.predicate == "related_to"
        assert rel.object_id == "entity-2"
        assert rel.get_edge_type() == "related_to"

    def test_relationship_with_metadata(self) -> None:
        """Relationships store domain-specific metadata (e.g., evidence_type, section)."""
        rel = make_test_relationship("e1", "e2")
        rel = rel.model_copy(update={"metadata": {"evidence_type": "direct", "section": "results"}})

        assert rel.metadata["evidence_type"] == "direct"
        assert rel.metadata["section"] == "results"

    def test_relationship_with_source_documents(self) -> None:
        """Relationships track which documents they were extracted from via source_documents."""
        rel = make_test_relationship("e1", "e2")
        rel = rel.model_copy(update={"source_documents": ("doc-1", "doc-2")})

        assert "doc-1" in rel.source_documents
        assert "doc-2" in rel.source_documents

    def test_relationship_immutability(self) -> None:
        """Relationships are immutable (frozen Pydantic models) to ensure data integrity."""
        rel = make_test_relationship("e1", "e2")

        with pytest.raises(Exception):
            rel.predicate = "modified"  # type: ignore


class TestRelationshipStorage:
    """Tests for InMemoryRelationshipStorage CRUD operations and queries.

    Verifies add/find/delete operations, queries by subject or object entity,
    optional predicate filtering, entity reference updates for merging, and counting.
    """

    async def test_add_and_find(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Storage supports add and find_by_triple (exact subject-predicate-object lookup)."""
        rel = make_test_relationship("e1", "e2", "related_to")
        await relationship_storage.add(rel)

        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")

        assert found is not None
        assert found.subject_id == "e1"
        assert found.object_id == "e2"

    async def test_find_nonexistent_triple(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """find_by_triple returns None when no matching relationship exists."""
        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")
        assert found is None

    async def test_get_by_subject(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """get_by_subject returns all relationships where the given entity is the subject."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))
        await relationship_storage.add(make_test_relationship("e1", "e3", "causes"))
        await relationship_storage.add(make_test_relationship("e2", "e3", "related_to"))

        results = await relationship_storage.get_by_subject("e1")

        assert len(results) == 2
        assert all(r.subject_id == "e1" for r in results)

    async def test_get_by_subject_with_predicate(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """get_by_subject with predicate filter returns only matching edge types."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))
        await relationship_storage.add(make_test_relationship("e1", "e3", "causes"))

        results = await relationship_storage.get_by_subject("e1", predicate="causes")

        assert len(results) == 1
        assert results[0].predicate == "causes"

    async def test_get_by_object(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """get_by_object returns all relationships where the given entity is the object."""
        await relationship_storage.add(make_test_relationship("e1", "e3", "related_to"))
        await relationship_storage.add(make_test_relationship("e2", "e3", "causes"))

        results = await relationship_storage.get_by_object("e3")

        assert len(results) == 2
        assert all(r.object_id == "e3" for r in results)

    async def test_update_entity_references(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """update_entity_references rewrites subject/object IDs when entities are merged.

        This is called during entity merging to update all relationships that
        reference the source entity to instead reference the target entity.
        """
        await relationship_storage.add(make_test_relationship("old-id", "e2"))
        await relationship_storage.add(make_test_relationship("e1", "old-id"))

        count = await relationship_storage.update_entity_references("old-id", "new-id")

        assert count == 2

        # Check references were updated
        by_subject = await relationship_storage.get_by_subject("new-id")
        by_object = await relationship_storage.get_by_object("new-id")

        assert len(by_subject) == 1
        assert len(by_object) == 1

        # Old references should be gone
        old_subject = await relationship_storage.get_by_subject("old-id")
        old_object = await relationship_storage.get_by_object("old-id")

        assert len(old_subject) == 0
        assert len(old_object) == 0

    async def test_delete_relationship(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """delete removes a relationship by its triple (subject, predicate, object)."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))

        result = await relationship_storage.delete("e1", "related_to", "e2")
        assert result is True

        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")
        assert found is None

    async def test_delete_nonexistent(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """delete returns False when the specified triple does not exist."""
        result = await relationship_storage.delete("e1", "related_to", "e2")
        assert result is False

    async def test_count(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """count returns the total number of relationships in storage."""
        assert await relationship_storage.count() == 0

        await relationship_storage.add(make_test_relationship("e1", "e2"))
        await relationship_storage.add(make_test_relationship("e2", "e3"))

        assert await relationship_storage.count() == 2
