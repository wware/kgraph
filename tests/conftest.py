"""Test fixtures and minimal test domain implementation."""

import uuid
from datetime import datetime, timezone
from typing import Sequence

import pytest

from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema
from kgraph.entity import BaseEntity, EntityMention, EntityStatus, PromotionConfig
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import EntityStorageInterface
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)


# --- Minimal Test Domain Implementations ---


class TestEntity(BaseEntity):
    """Simple entity for testing."""

    entity_type: str = "test_entity"

    def get_entity_type(self) -> str:
        return self.entity_type


class TestRelationship(BaseRelationship):
    """Simple relationship for testing."""

    def get_edge_type(self) -> str:
        return self.predicate


class TestDocument(BaseDocument):
    """Simple document for testing."""

    document_type: str = "test_document"

    def get_document_type(self) -> str:
        return self.document_type

    def get_sections(self) -> list[tuple[str, str]]:
        return [("body", self.content)]


class TestDomainSchema(DomainSchema):
    """Minimal domain schema for testing."""

    @property
    def name(self) -> str:
        return "test_domain"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {"test_entity": TestEntity}

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {"related_to": TestRelationship, "causes": TestRelationship}

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        return {"test_document": TestDocument}

    @property
    def promotion_config(self) -> PromotionConfig:
        return PromotionConfig(
            min_usage_count=2,
            min_confidence=0.7,
            require_embedding=False,
        )

    def validate_entity(self, entity: BaseEntity) -> bool:
        return entity.get_entity_type() in self.entity_types

    def validate_relationship(self, relationship: BaseRelationship) -> bool:
        return relationship.predicate in self.relationship_types


class MockEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Mock embedding generator that produces deterministic embeddings."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def generate(self, text: str) -> tuple[float, ...]:
        # Simple deterministic embedding based on text hash
        h = hash(text)
        return tuple((h >> i) % 100 / 100.0 for i in range(self._dim))

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [await self.generate(t) for t in texts]


class MockDocumentParser(DocumentParserInterface):
    """Mock parser that creates TestDocument from raw content."""

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> BaseDocument:
        return TestDocument(
            document_id=str(uuid.uuid4()),
            content=raw_content.decode("utf-8"),
            content_type=content_type,
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
        )


class MockEntityExtractor(EntityExtractorInterface):
    """Mock extractor that finds words in square brackets as entities."""

    async def extract(self, document: BaseDocument) -> list[EntityMention]:
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
    """Mock resolver that creates provisional entities."""

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        # Try to find existing entity by name
        existing = await existing_storage.find_by_name(mention.text, mention.entity_type, limit=1)
        if existing:
            return existing[0], 0.95

        # Create provisional entity
        entity = TestEntity(
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
        return [await self.resolve(m, existing_storage) for m in mentions]


class MockRelationshipExtractor(RelationshipExtractorInterface):
    """Mock extractor that creates relationships between adjacent entities."""

    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        relationships = []
        entity_list = list(entities)
        for i in range(len(entity_list) - 1):
            relationships.append(
                TestRelationship(
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
def test_domain() -> TestDomainSchema:
    """Provide a test domain schema."""
    return TestDomainSchema()


@pytest.fixture
def entity_storage() -> InMemoryEntityStorage:
    """Provide fresh in-memory entity storage."""
    return InMemoryEntityStorage()


@pytest.fixture
def relationship_storage() -> InMemoryRelationshipStorage:
    """Provide fresh in-memory relationship storage."""
    return InMemoryRelationshipStorage()


@pytest.fixture
def document_storage() -> InMemoryDocumentStorage:
    """Provide fresh in-memory document storage."""
    return InMemoryDocumentStorage()


@pytest.fixture
def embedding_generator() -> MockEmbeddingGenerator:
    """Provide mock embedding generator."""
    return MockEmbeddingGenerator()


@pytest.fixture
def document_parser() -> MockDocumentParser:
    """Provide mock document parser."""
    return MockDocumentParser()


@pytest.fixture
def entity_extractor() -> MockEntityExtractor:
    """Provide mock entity extractor."""
    return MockEntityExtractor()


@pytest.fixture
def entity_resolver() -> MockEntityResolver:
    """Provide mock entity resolver."""
    return MockEntityResolver()


@pytest.fixture
def relationship_extractor() -> MockRelationshipExtractor:
    """Provide mock relationship extractor."""
    return MockRelationshipExtractor()


def make_test_entity(
    name: str,
    status: EntityStatus = EntityStatus.PROVISIONAL,
    entity_id: str | None = None,
    usage_count: int = 0,
    confidence: float = 1.0,
    embedding: tuple[float, ...] | None = None,
    canonical_ids: dict[str, str] | None = None,
) -> TestEntity:
    """Helper to create test entities."""
    return TestEntity(
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
) -> TestRelationship:
    """Helper to create test relationships."""
    return TestRelationship(
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        confidence=confidence,
        created_at=datetime.now(timezone.utc),
    )
