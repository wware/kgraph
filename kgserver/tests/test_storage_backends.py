"""
Direct tests for storage backend implementations.

Tests cover:
- SQLiteStorage direct operations
- PostgresStorage direct operations (when available)
- Filter combinations
- Edge cases
"""

# pylint: disable=protected-access
import pytest
from sqlmodel import Session, create_engine, SQLModel
from storage.backends.postgres import PostgresStorage
from storage.models import Entity, Relationship, Bundle
from datetime import datetime


class TestSQLiteStorage:
    """Direct tests for SQLiteStorage."""

    def test_get_entity_existing(self, in_memory_storage, sample_entities):
        """Test retrieving an existing entity."""
        # Add entity
        in_memory_storage._session.add(sample_entities[0])
        in_memory_storage._session.commit()

        entity = in_memory_storage.get_entity("test:entity:1")
        assert entity is not None
        assert entity.entity_id == "test:entity:1"
        assert entity.name == "Test Character 1"

    def test_get_entity_nonexistent(self, in_memory_storage):
        """Test retrieving non-existent entity."""
        entity = in_memory_storage.get_entity("nonexistent")
        assert entity is None

    def test_get_entities_with_filters(self, in_memory_storage, sample_entities):
        """Test get_entities with various filters."""
        # Add all entities
        for entity in sample_entities:
            in_memory_storage._session.add(entity)
        in_memory_storage._session.commit()

        # Filter by type
        entities = in_memory_storage.get_entities(limit=10, entity_type="character")
        assert len(entities) == 2
        for e in entities:
            assert e.entity_type == "character"

        # Filter by name contains
        entities = in_memory_storage.get_entities(limit=10, name_contains="Character")
        assert len(entities) == 2

        # Filter by source
        entities = in_memory_storage.get_entities(limit=10, source="test")
        assert len(entities) == 3

    def test_count_entities(self, in_memory_storage, sample_entities):
        """Test count_entities."""
        # Add entities
        for entity in sample_entities:
            in_memory_storage._session.add(entity)
        in_memory_storage._session.commit()

        total = in_memory_storage.count_entities()
        assert total == 3

        # Count with filter
        count = in_memory_storage.count_entities(entity_type="character")
        assert count == 2

    def test_find_relationships_with_filters(self, in_memory_storage, sample_entities, sample_relationships):
        """Test find_relationships with filters."""
        # Add entities and relationships
        for entity in sample_entities:
            in_memory_storage._session.add(entity)
        for rel in sample_relationships:
            in_memory_storage._session.add(rel)
        in_memory_storage._session.commit()

        # Filter by subject
        rels = in_memory_storage.find_relationships(subject_id="test:entity:1", limit=10)
        assert len(rels) == 2

        # Filter by predicate
        rels = in_memory_storage.find_relationships(predicate="co_occurs_with", limit=10)
        assert len(rels) == 2

        # Filter by object
        rels = in_memory_storage.find_relationships(object_id="test:entity:3", limit=10)
        assert len(rels) == 2

    def test_count_relationships(self, in_memory_storage, sample_entities, sample_relationships):
        """Test count_relationships."""
        # Add entities and relationships
        for entity in sample_entities:
            in_memory_storage._session.add(entity)
        for rel in sample_relationships:
            in_memory_storage._session.add(rel)
        in_memory_storage._session.commit()

        total = in_memory_storage.count_relationships()
        assert total == 3

        # Count with filter
        count = in_memory_storage.count_relationships(subject_id="test:entity:1")
        assert count == 2

    def test_get_bundle_info(self, in_memory_storage):
        """Test get_bundle_info."""
        # Add bundle
        bundle = Bundle(
            bundle_id="test-bundle-123",
            domain="test",
            created_at=datetime.now(),
            bundle_version="v1",
        )
        in_memory_storage._session.add(bundle)
        in_memory_storage._session.commit()

        info = in_memory_storage.get_bundle_info()
        assert info is not None
        assert info.bundle_id == "test-bundle-123"
        assert info.domain == "test"

    def test_get_bundle_info_none(self, in_memory_storage):
        """Test get_bundle_info when no bundle exists."""
        info = in_memory_storage.get_bundle_info()
        assert info is None

    def test_is_bundle_loaded(self, in_memory_storage):
        """Test is_bundle_loaded."""
        # No bundle loaded
        assert not in_memory_storage.is_bundle_loaded("test-bundle-123")

        # Add bundle
        bundle = Bundle(
            bundle_id="test-bundle-123",
            domain="test",
            created_at=datetime.now(),
            bundle_version="v1",
        )
        in_memory_storage._session.add(bundle)
        in_memory_storage._session.commit()

        assert in_memory_storage.is_bundle_loaded("test-bundle-123")
        assert not in_memory_storage.is_bundle_loaded("other-bundle")


class TestPostgresStorage:
    """Direct tests for PostgresStorage using mocked database."""

    @pytest.fixture
    def postgres_storage(self):
        """Create PostgresStorage using in-memory SQLite (mocks PostgreSQL)."""
        from sqlmodel import delete

        # Use SQLite in-memory database to mock PostgreSQL
        # PostgresStorage uses SQLModel which is database-agnostic for basic operations
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        session = Session(engine)
        storage = PostgresStorage(session)
        yield storage
        # Clean up after test
        session.exec(delete(Relationship))
        session.exec(delete(Entity))
        session.exec(delete(Bundle))
        session.commit()
        session.close()
        engine.dispose()

    def test_postgres_storage_basic(self, postgres_storage, sample_entities):
        """Test basic PostgresStorage operations."""
        # Add entity
        postgres_storage._session.add(sample_entities[0])
        postgres_storage._session.commit()

        entity = postgres_storage.get_entity("test:entity:1")
        assert entity is not None
        assert entity.entity_id == "test:entity:1"

    def test_postgres_storage_filters(self, postgres_storage, sample_entities):
        """Test PostgresStorage with filters."""
        # Add entities
        for entity in sample_entities:
            postgres_storage._session.add(entity)
        postgres_storage._session.commit()

        entities = postgres_storage.get_entities(limit=10, entity_type="character")
        assert len(entities) == 2
