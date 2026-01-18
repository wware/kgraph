"""Pipeline interface definitions for document processing and extraction."""

from abc import ABC, abstractmethod
from typing import Sequence

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity, EntityMention
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import EntityStorageInterface


class DocumentParserInterface(ABC):
    """Parse raw documents into structured form."""

    @abstractmethod
    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> BaseDocument:
        """Parse raw content into a structured document.

        Args:
            raw_content: Raw document bytes
            content_type: MIME type or format indicator
            source_uri: Optional source location

        Returns:
            Parsed document ready for extraction
        """


class EntityExtractorInterface(ABC):
    """Extract entity mentions from documents (Pass 1)."""

    @abstractmethod
    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        """Extract entity mentions from a document.

        Returns raw mentions that will later be resolved to
        canonical or provisional entities.
        """


class EntityResolverInterface(ABC):
    """Resolve entity mentions to canonical or provisional entities."""

    @abstractmethod
    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        """Resolve an entity mention to an entity.

        Attempts to match the mention to existing canonical entities.
        If no match found, creates a provisional entity.

        Args:
            mention: The extracted entity mention
            existing_storage: Storage to check for existing entities

        Returns:
            Tuple of (resolved entity, confidence score)
        """

    @abstractmethod
    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]:
        """Resolve multiple mentions efficiently.

        Default implementation calls resolve() for each mention.
        Override for batch optimization (e.g., batch embedding lookups).
        """


class RelationshipExtractorInterface(ABC):
    """Extract relationships from documents (Pass 2)."""

    @abstractmethod
    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        """Extract relationships between entities in a document.

        Args:
            document: The source document
            entities: Entities that were extracted from this document

        Returns:
            List of extracted relationships
        """
