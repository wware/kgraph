"""Storage interface definitions for the knowledge graph framework."""

from abc import ABC, abstractmethod
from typing import Sequence

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity
from kgraph.relationship import BaseRelationship


class EntityStorageInterface(ABC):
    """Abstract interface for entity storage operations."""

    @abstractmethod
    async def add(self, entity: BaseEntity) -> str:
        """Store an entity and return its ID.

        If an entity with the same ID already exists, implementations
        may either update it or raise an error depending on policy.
        """

    @abstractmethod
    async def get(self, entity_id: str) -> BaseEntity | None:
        """Retrieve an entity by ID, or None if not found."""

    @abstractmethod
    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]:
        """Retrieve multiple entities by ID.

        Returns a list in the same order as input IDs, with None for missing entities.
        """

    @abstractmethod
    async def find_by_embedding(
        self,
        embedding: Sequence[float],
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[BaseEntity, float]]:
        """Find entities similar to the given embedding.

        Returns (entity, similarity_score) tuples, sorted by descending similarity.
        Only returns entities with similarity >= threshold.
        """

    @abstractmethod
    async def find_by_name(
        self,
        name: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[BaseEntity]:
        """Find entities matching the given name or synonym.

        Optionally filter by entity type.
        """

    @abstractmethod
    async def find_provisional_for_promotion(
        self,
        min_usage: int,
        min_confidence: float,
    ) -> list[BaseEntity]:
        """Find provisional entities eligible for promotion.

        Returns entities with status=PROVISIONAL, usage_count >= min_usage,
        and confidence >= min_confidence.
        """

    @abstractmethod
    async def update(self, entity: BaseEntity) -> bool:
        """Update an existing entity.

        Returns True if the entity was found and updated, False otherwise.
        """

    @abstractmethod
    async def promote(
        self,
        entity_id: str,
        canonical_id: str,
    ) -> BaseEntity | None:
        """Promote a provisional entity to canonical status.

        Updates the entity's ID and status. Returns the updated entity,
        or None if the entity was not found.
        """

    @abstractmethod
    async def merge(
        self,
        source_ids: Sequence[str],
        target_id: str,
    ) -> bool:
        """Merge multiple entities into a target entity.

        Combines usage counts and synonyms. Source entities are removed.
        Returns True if merge succeeded.
        """

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity by ID. Returns True if found and deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored entities."""


class RelationshipStorageInterface(ABC):
    """Abstract interface for relationship storage operations."""

    @abstractmethod
    async def add(self, relationship: BaseRelationship) -> str:
        """Store a relationship and return an identifier.

        If a relationship with the same (subject, predicate, object) exists,
        implementations may merge or update it.
        """

    @abstractmethod
    async def get_by_subject(
        self,
        subject_id: str,
        predicate: str | None = None,
    ) -> list[BaseRelationship]:
        """Get all relationships where the given entity is the subject.

        Optionally filter by predicate type.
        """

    @abstractmethod
    async def get_by_object(
        self,
        object_id: str,
        predicate: str | None = None,
    ) -> list[BaseRelationship]:
        """Get all relationships where the given entity is the object.

        Optionally filter by predicate type.
        """

    @abstractmethod
    async def find_by_triple(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> BaseRelationship | None:
        """Find a specific relationship by its triple.

        Returns None if no such relationship exists.
        """

    @abstractmethod
    async def update_entity_references(
        self,
        old_entity_id: str,
        new_entity_id: str,
    ) -> int:
        """Update all relationships referencing old_entity_id to use new_entity_id.

        Used when merging or promoting entities.
        Returns the number of relationships updated.
        """

    @abstractmethod
    async def delete(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> bool:
        """Delete a specific relationship. Returns True if found and deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored relationships."""


class DocumentStorageInterface(ABC):
    """Abstract interface for document storage operations."""

    @abstractmethod
    async def add(self, document: BaseDocument) -> str:
        """Store a document and return its ID."""

    @abstractmethod
    async def get(self, document_id: str) -> BaseDocument | None:
        """Retrieve a document by ID, or None if not found."""

    @abstractmethod
    async def find_by_source(self, source_uri: str) -> BaseDocument | None:
        """Find a document by its source URI."""

    @abstractmethod
    async def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List document IDs with pagination."""

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID. Returns True if found and deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored documents."""
