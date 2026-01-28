"""Test fixtures and minimal test domain implementation.

This module provides:
- Minimal concrete implementations of abstract base classes (SimpleEntity,
  SimpleDocument, SimpleRelationship, SimpleDomainSchema) for use in unit tests
- Mock implementations of pipeline interfaces (document parsing, entity
  extraction/resolution, relationship extraction, embedding generation)
- Pytest fixtures that instantiate in-memory storage and mock components
- Helper factory functions for creating test entities and relationships

The test domain uses a simple convention where entities are denoted by
square brackets in document text (e.g., "[aspirin]" becomes an entity).
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Sequence

import pytest

from kgraph.canonical_id import CanonicalId
from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema, PredicateConstraint
from kgraph.entity import BaseEntity, EntityMention, EntityStatus, PromotionConfig
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.promotion import PromotionPolicy
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import EntityStorageInterface
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)

# --- Minimal Test Domain Implementations ---


class SimpleEntity(BaseEntity):
    """Minimal concrete entity implementation for unit tests.

    Provides a single entity type ("test_entity") with configurable type
    via the entity_type field. Inherits all standard entity fields from
    BaseEntity (entity_id, name, status, confidence, usage_count, etc.).
    """

    entity_type: str = "test_entity"

    def get_entity_type(self) -> str:
        """Return the entity's type identifier."""
        return self.entity_type


class SimpleRelationship(BaseRelationship):
    """Minimal concrete relationship implementation for unit tests.

    Uses the predicate field directly as the edge type. Supports any
    predicate string, allowing flexible relationship testing without
    schema constraints.
    """
    subject_entity_type: str = "test_entity"
    object_entity_type: str = "test_entity"

    def get_edge_type(self) -> str:
        """Return the relationship's edge type (same as predicate)."""
        return self.predicate


class SimpleDocument(BaseDocument):
    """Minimal concrete document implementation for unit tests.

    Represents documents as a single "body" section containing the full
    content. The document type defaults to "test_document" but can be
    customized for tests requiring multiple document types.
    """

    document_type: str = "test_document"

    def get_document_type(self) -> str:
        """Return the document's type identifier."""
        return self.document_type

    def get_sections(self) -> list[tuple[str, str]]:
        """Return document sections as (section_name, content) tuples.

        For SimpleDocument, returns a single "body" section with full content.
        """
        return [("body", self.content)]


class SimplePromotionPolicy(PromotionPolicy):
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        canonical_id_str = f"canonical-{type(entity)}-{id(entity)}"
        return CanonicalId(id=canonical_id_str, url=None, synonyms=())


class SimpleDomainSchema(DomainSchema):
    """Minimal domain schema defining types and validation for the test domain.

    Configures:
    - One entity type: "test_entity"
    - Two relationship types: "related_to", "causes"
    - One document type: "test_document"
    - Lenient promotion config: 2 usages, 0.7 confidence, no embedding required

    This schema is intentionally simple to allow straightforward testing
    of domain-agnostic graph operations without real-world complexity.
    """

    @property
    def name(self) -> str:
        """Return the domain name identifier."""
        return "test_domain"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        """Return mapping of entity type names to their classes."""
        return {"test_entity": SimpleEntity}

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        """Return mapping of relationship type names to their classes."""
        return {"related_to": SimpleRelationship, "causes": SimpleRelationship}

    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]:
        """Define predicate constraints for the test domain.

        For simplicity in testing, we'll allow all relationships to be valid
        between 'test_entity' types by default, but this can be overridden
        in specific tests if needed.
        """
        return {
            "related_to": PredicateConstraint(
                subject_types={"test_entity"}, object_types={"test_entity"}
            ),
            "causes": PredicateConstraint(
                subject_types={"test_entity"}, object_types={"test_entity"}
            ),
        }

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        """Return mapping of document type names to their classes."""
        return {"test_document": SimpleDocument}

    @property
    def promotion_config(self) -> PromotionConfig:
        """Return configuration for promoting provisional entities to canonical.

        Uses lenient thresholds suitable for testing: requires only 2 usages
        and 0.7 confidence, with no embedding requirement.
        """
        return PromotionConfig(
            min_usage_count=2,
            min_confidence=0.7,
            require_embedding=False,
        )

    def validate_entity(self, entity: BaseEntity) -> bool:
        """Check if the entity's type is registered in this schema."""
        return entity.get_entity_type() in self.entity_types

    def validate_relationship(
        self,
        relationship: BaseRelationship,
        entity_storage: EntityStorageInterface | None = None,
    ) -> bool:
        """Check if the relationship's predicate is registered in this schema.

        Also calls the superclass method to apply predicate constraints.
        """
        return super().validate_relationship(relationship, entity_storage=entity_storage)

    def get_promotion_policy(self, lookup=None) -> PromotionPolicy:
        return SimplePromotionPolicy(config=self.promotion_config)


class MockEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Mock embedding generator producing deterministic hash-based embeddings.

    Generates fixed-dimension vectors derived from the text's hash, ensuring
    identical text always produces identical embeddings. Useful for testing
    embedding-dependent logic (similarity search, merge detection) without
    requiring a real embedding model.

    Args:
        dim: Embedding vector dimension (default: 8).
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dim

    async def generate(self, text: str) -> tuple[float, ...]:
        """Generate a deterministic embedding from text using its hash."""
        h = hash(text)
        return tuple((h >> i) % 100 / 100.0 for i in range(self._dim))

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Generate embeddings for multiple texts sequentially."""
        return [await self.generate(t) for t in texts]


class MockDocumentParser(DocumentParserInterface):
    """Mock document parser that wraps raw bytes in a SimpleDocument.

    Decodes raw content as UTF-8 text and creates a SimpleDocument with
    a randomly generated document_id. Does not perform any actual parsing
    or structure extraction.
    """

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> BaseDocument:
        """Parse raw bytes into a SimpleDocument.

        Args:
            raw_content: Document bytes (decoded as UTF-8).
            content_type: MIME type of the content.
            source_uri: Optional URI identifying the document source.

        Returns:
            SimpleDocument instance with generated ID and current timestamp.
        """
        return SimpleDocument(
            document_id=str(uuid.uuid4()),
            content=raw_content.decode("utf-8"),
            content_type=content_type,
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
        )


class MockEntityExtractor(EntityExtractorInterface):
    """Mock entity extractor using bracket notation for entity detection.

    Extracts entities from document text by finding text enclosed in
    square brackets. For example, the text "Patient took [aspirin] for
    [headache]" yields two EntityMention objects for "aspirin" and
    "headache", each with 0.9 confidence and "test_entity" type.

    This simple convention allows tests to precisely control which
    entities are extracted without needing NLP or ML components.
    """

    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        """Extract entity mentions from bracketed text in the document.

        Args:
            document: Document to extract entities from.

        Returns:
            List of EntityMention objects for each bracketed term found.
        """
        import re

        mentions = []
        for match in re.finditer(r"\[([^\]]+)\]", document.content):
            mentions.append(
                EntityMention(
                    text=match.group(1),
                    entity_type="test_entity",
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.9,
                )
            )
        return mentions


class MockEntityResolver(EntityResolverInterface):
    """Mock entity resolver that links mentions to entities via name matching.

    Resolution strategy:
    1. Search existing storage for an entity with matching name and type
    2. If found, return existing entity with 0.95 confidence
    3. If not found, create a new provisional SimpleEntity with the
       mention's confidence score

    This simple name-based matching is sufficient for testing entity
    resolution and promotion logic without external knowledge bases.
    """

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        """Resolve an entity mention to an existing or new entity.

        Args:
            mention: The entity mention to resolve.
            existing_storage: Storage to search for existing entities.

        Returns:
            Tuple of (resolved entity, resolution confidence score).
        """
        existing = await existing_storage.find_by_name(mention.text, mention.entity_type, limit=1)
        if existing:
            return existing[0], 0.95

        entity = SimpleEntity(
            entity_id=str(uuid.uuid4()),
            status=EntityStatus.PROVISIONAL,
            name=mention.text,
            confidence=mention.confidence,
            usage_count=1,
            created_at=datetime.now(timezone.utc),
            source="mock_extractor",
        )
        return entity, mention.confidence

    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]:
        """Resolve multiple mentions sequentially."""
        return [await self.resolve(m, existing_storage) for m in mentions]


class MockRelationshipExtractor(RelationshipExtractorInterface):
    """Mock relationship extractor that chains adjacent entities.

    Creates "related_to" relationships between consecutively ordered
    entities. For a document with entities [A, B, C], produces edges
    A->B and B->C, each with 0.8 confidence.

    This simple linear chaining allows tests to verify relationship
    storage and traversal without complex extraction logic.
    """

    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        """Extract relationships between consecutive entities.

        Args:
            document: Source document for provenance tracking.
            entities: Ordered sequence of entities from the document.

        Returns:
            List of "related_to" relationships linking adjacent entities.
        """
        relationships: list[BaseRelationship] = []
        entity_list = list(entities)
        for i in range(len(entity_list) - 1):
            relationships.append(
                SimpleRelationship(
                    subject_id=entity_list[i].entity_id,
                    predicate="related_to",
                    object_id=entity_list[i + 1].entity_id,
                    confidence=0.8,
                    source_documents=(document.document_id,),
                    created_at=datetime.now(timezone.utc),
                )
            )
        return relationships


# --- Fixtures ---


@pytest.fixture
def test_domain() -> SimpleDomainSchema:
    """Provide a SimpleDomainSchema instance for domain-aware tests.

    The schema defines entity types, relationship types, and promotion
    configuration suitable for unit testing graph operations.
    """
    return SimpleDomainSchema()


@pytest.fixture
def entity_storage() -> InMemoryEntityStorage:
    """Provide a fresh in-memory entity storage instance.

    Each test receives an empty storage, ensuring test isolation.
    """
    return InMemoryEntityStorage()


@pytest.fixture
def relationship_storage() -> InMemoryRelationshipStorage:
    """Provide a fresh in-memory relationship storage instance.

    Each test receives an empty storage, ensuring test isolation.
    """
    return InMemoryRelationshipStorage()


@pytest.fixture
def document_storage() -> InMemoryDocumentStorage:
    """Provide a fresh in-memory document storage instance.

    Each test receives an empty storage, ensuring test isolation.
    """
    return InMemoryDocumentStorage()


@pytest.fixture
def embedding_generator() -> MockEmbeddingGenerator:
    """Provide a MockEmbeddingGenerator with default 8-dimensional vectors.

    Generates deterministic embeddings based on text hash for reproducible
    similarity comparisons in tests.
    """
    return MockEmbeddingGenerator()


@pytest.fixture
def document_parser() -> MockDocumentParser:
    """Provide a MockDocumentParser for converting raw bytes to SimpleDocument.

    Decodes content as UTF-8 without performing actual parsing or
    structure extraction.
    """
    return MockDocumentParser()


@pytest.fixture
def entity_extractor() -> MockEntityExtractor:
    """Provide a MockEntityExtractor using bracket notation.

    Extracts entities from text enclosed in square brackets (e.g.,
    "[aspirin]" becomes an entity mention).
    """
    return MockEntityExtractor()


@pytest.fixture
def entity_resolver() -> MockEntityResolver:
    """Provide a MockEntityResolver for name-based entity matching.

    Links mentions to existing entities by name or creates new
    provisional entities when no match is found.
    """
    return MockEntityResolver()


@pytest.fixture
def relationship_extractor() -> MockRelationshipExtractor:
    """Provide a MockRelationshipExtractor for linear entity chaining.

    Creates "related_to" relationships between consecutively ordered
    entities in a document.
    """
    return MockRelationshipExtractor()


def make_test_entity(
    name: str,
    status: EntityStatus = EntityStatus.PROVISIONAL,
    entity_id: str | None = None,
    usage_count: int = 0,
    confidence: float = 1.0,
    embedding: tuple[float, ...] | None = None,
    canonical_ids: dict[str, str] | None = None,
) -> SimpleEntity:
    """Factory function to create SimpleEntity instances with sensible defaults.

    Provides a concise way to create entities in tests without specifying
    all required fields. Generates a random UUID if entity_id is not provided.

    Args:
        name: Display name for the entity (required).
        status: Entity lifecycle status (default: PROVISIONAL).
        entity_id: Unique identifier (default: auto-generated UUID).
        usage_count: Number of document references (default: 0).
        confidence: Confidence score from extraction (default: 1.0).
        embedding: Optional semantic embedding vector.
        canonical_ids: Optional mapping of authority names to external IDs.

    Returns:
        Configured SimpleEntity instance with current timestamp.
    """
    return SimpleEntity(
        entity_id=entity_id or str(uuid.uuid4()),
        status=status,
        name=name,
        confidence=confidence,
        usage_count=usage_count,
        embedding=embedding,
        canonical_ids=canonical_ids or {},
        created_at=datetime.now(timezone.utc),
        source="test",
    )


def make_test_relationship(
    subject_id: str,
    object_id: str,
    predicate: str = "related_to",
    confidence: float = 1.0,
) -> SimpleRelationship:
    """Factory function to create SimpleRelationship instances with defaults.

    Provides a concise way to create relationships in tests. Both subject
    and object entity IDs are required; predicate and confidence have defaults.

    Args:
        subject_id: Entity ID of the relationship source.
        object_id: Entity ID of the relationship target.
        predicate: Relationship type name (default: "related_to").
        confidence: Confidence score for the relationship (default: 1.0).

    Returns:
        Configured SimpleRelationship instance with current timestamp.
    """
    return SimpleRelationship(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        created_at=datetime.now(timezone.utc),
    )
