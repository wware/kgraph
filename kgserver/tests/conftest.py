"""
Pytest configuration and shared fixtures for GraphQL tests.
"""

# pylint: disable=protected-access
import pytest
from datetime import datetime
from storage.backends.sqlite import SQLiteStorage
from storage.models import Bundle, Entity, Relationship
from query.graphql_schema import Query
import strawberry


@pytest.fixture
def in_memory_storage():
    """Create an in-memory SQLite storage for testing."""
    storage = SQLiteStorage(":memory:")
    yield storage
    storage.close()


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            entity_id="test:entity:1",
            entity_type="character",
            name="Test Character 1",
            status="canonical",
            confidence=0.95,
            usage_count=10,
            source="test",
            synonyms=["TC1", "TestChar1"],
            properties={"test": "data"},
        ),
        Entity(
            entity_id="test:entity:2",
            entity_type="character",
            name="Test Character 2",
            status="canonical",
            confidence=0.90,
            usage_count=5,
            source="test",
            synonyms=["TC2"],
            properties={},
        ),
        Entity(
            entity_id="test:entity:3",
            entity_type="location",
            name="Test Location",
            status="canonical",
            confidence=1.0,
            usage_count=20,
            source="test",
            synonyms=[],
            properties={"type": "place"},
        ),
    ]


@pytest.fixture
def sample_relationships():
    """Create sample relationships for testing."""
    return [
        Relationship(
            subject_id="test:entity:1",
            predicate="co_occurs_with",
            object_id="test:entity:2",
            confidence=0.85,
            source_documents=["doc1", "doc2"],
            properties={"count": 5},
        ),
        Relationship(
            subject_id="test:entity:1",
            predicate="appears_in",
            object_id="test:entity:3",
            confidence=0.90,
            source_documents=["doc1"],
            properties={},
        ),
        Relationship(
            subject_id="test:entity:2",
            predicate="co_occurs_with",
            object_id="test:entity:3",
            confidence=0.75,
            source_documents=["doc2"],
            properties={},
        ),
    ]


@pytest.fixture
def populated_storage(in_memory_storage, sample_entities, sample_relationships):
    """Create storage with sample data."""
    # Add entities
    for entity in sample_entities:
        in_memory_storage._session.add(entity)

    # Add relationships
    for relationship in sample_relationships:
        in_memory_storage._session.add(relationship)

    in_memory_storage._session.commit()
    return in_memory_storage


@pytest.fixture
def graphql_context(populated_storage):
    """Create GraphQL context with populated storage."""
    return {"storage": populated_storage}


@pytest.fixture
def graphql_schema():
    """Create GraphQL schema for testing."""
    return strawberry.Schema(query=Query)


@pytest.fixture
def sample_bundle():
    """Create a sample bundle for testing."""
    from kgbundle import BundleManifestV1, BundleFile

    return BundleManifestV1(
        bundle_id="test-bundle-123",
        domain="test",
        created_at=datetime.now().isoformat(),
        bundle_version="v1",
        entities=BundleFile(path="entities.jsonl", format="jsonl"),
        relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
    )


@pytest.fixture
def storage_with_bundle(populated_storage, sample_bundle):
    """Create storage with bundle metadata."""
    bundle = Bundle(
        bundle_id=sample_bundle.bundle_id,
        domain=sample_bundle.domain,
        created_at=datetime.fromisoformat(sample_bundle.created_at),
        bundle_version=sample_bundle.bundle_version,
    )
    populated_storage._session.add(bundle)
    populated_storage._session.commit()
    return populated_storage
