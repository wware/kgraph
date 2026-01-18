"""Tests for relationship creation and storage."""

import pytest

from kgraph.storage.memory import InMemoryRelationshipStorage

from tests.conftest import make_test_relationship


class TestRelationshipCreation:
    """Tests for creating relationships."""

    def test_create_relationship(self) -> None:
        """Can create a basic relationship."""
        rel = make_test_relationship("entity-1", "entity-2", "related_to")

        assert rel.subject_id == "entity-1"
        assert rel.predicate == "related_to"
        assert rel.object_id == "entity-2"
        assert rel.get_edge_type() == "related_to"

    def test_relationship_with_metadata(self) -> None:
        """Relationships can carry domain-specific metadata."""
        rel = make_test_relationship("e1", "e2")
        rel = rel.model_copy(update={"metadata": {"evidence_type": "direct", "section": "results"}})

        assert rel.metadata["evidence_type"] == "direct"
        assert rel.metadata["section"] == "results"

    def test_relationship_with_source_documents(self) -> None:
        """Relationships track source documents."""
        rel = make_test_relationship("e1", "e2")
        rel = rel.model_copy(update={"source_documents": ("doc-1", "doc-2")})

        assert "doc-1" in rel.source_documents
        assert "doc-2" in rel.source_documents

    def test_relationship_immutability(self) -> None:
        """Relationships are immutable."""
        rel = make_test_relationship("e1", "e2")

        with pytest.raises(Exception):
            rel.predicate = "modified"  # type: ignore


class TestRelationshipStorage:
    """Tests for in-memory relationship storage."""

    async def test_add_and_find(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can add and retrieve relationships."""
        rel = make_test_relationship("e1", "e2", "related_to")
        await relationship_storage.add(rel)

        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")

        assert found is not None
        assert found.subject_id == "e1"
        assert found.object_id == "e2"

    async def test_find_nonexistent_triple(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Finding nonexistent triple returns None."""
        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")
        assert found is None

    async def test_get_by_subject(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can retrieve relationships by subject."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))
        await relationship_storage.add(make_test_relationship("e1", "e3", "causes"))
        await relationship_storage.add(make_test_relationship("e2", "e3", "related_to"))

        results = await relationship_storage.get_by_subject("e1")

        assert len(results) == 2
        assert all(r.subject_id == "e1" for r in results)

    async def test_get_by_subject_with_predicate(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can filter by predicate when querying by subject."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))
        await relationship_storage.add(make_test_relationship("e1", "e3", "causes"))

        results = await relationship_storage.get_by_subject("e1", predicate="causes")

        assert len(results) == 1
        assert results[0].predicate == "causes"

    async def test_get_by_object(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can retrieve relationships by object."""
        await relationship_storage.add(make_test_relationship("e1", "e3", "related_to"))
        await relationship_storage.add(make_test_relationship("e2", "e3", "causes"))

        results = await relationship_storage.get_by_object("e3")

        assert len(results) == 2
        assert all(r.object_id == "e3" for r in results)

    async def test_update_entity_references(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can update entity references when entities are merged."""
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
        """Can delete relationships."""
        await relationship_storage.add(make_test_relationship("e1", "e2", "related_to"))

        result = await relationship_storage.delete("e1", "related_to", "e2")
        assert result is True

        found = await relationship_storage.find_by_triple("e1", "related_to", "e2")
        assert found is None

    async def test_delete_nonexistent(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Deleting nonexistent relationship returns False."""
        result = await relationship_storage.delete("e1", "related_to", "e2")
        assert result is False

    async def test_count(self, relationship_storage: InMemoryRelationshipStorage) -> None:
        """Can count stored relationships."""
        assert await relationship_storage.count() == 0

        await relationship_storage.add(make_test_relationship("e1", "e2"))
        await relationship_storage.add(make_test_relationship("e2", "e3"))

        assert await relationship_storage.count() == 2
