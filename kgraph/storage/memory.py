"""In-memory storage implementations for testing and development.

This module provides dictionary-based implementations of the storage interfaces
that keep all data in memory. These implementations are suitable for:

- **Unit testing**: Fast, isolated tests without external dependencies
- **Development**: Quick iteration without database setup
- **Prototyping**: Experimenting with the framework before choosing a backend
- **Small datasets**: Demos and examples with limited data

**Not recommended for production** due to:
- No persistence (data is lost when the process exits)
- No concurrency control (not safe for multi-process access)
- Memory constraints (all data must fit in RAM)
- O(n) search operations (no indexing)

For production use, implement the storage interfaces with a proper database
backend (PostgreSQL with pgvector, Neo4j, etc.).
"""

import math
from typing import Sequence

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity, EntityStatus
from kgraph.relationship import BaseRelationship
from kgraph.storage.interfaces import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    producing a value between -1 (opposite) and 1 (identical direction).
    For normalized embeddings, this is equivalent to the dot product.

    Args:
        a: First embedding vector.
        b: Second embedding vector (must have same dimension as a).

    Returns:
        Cosine similarity score between -1 and 1. Returns 0.0 if vectors
        have different lengths, are empty, or have zero magnitude.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class InMemoryEntityStorage(EntityStorageInterface):
    """In-memory entity storage using a dictionary keyed by entity_id.

    Stores entities in a simple `dict[str, BaseEntity]` structure. All
    operations are O(1) for direct lookups and O(n) for searches.

    Thread safety: Not thread-safe. For concurrent access, use external
    synchronization or a production database backend.

    Example:
        ```python
        storage = InMemoryEntityStorage()
        await storage.add(my_entity)
        entity = await storage.get(my_entity.entity_id)
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty entity storage."""
        self._entities: dict[str, BaseEntity] = {}

    async def add(self, entity: BaseEntity) -> str:
        """Adds a new entity to the storage.

        If an entity with the same ID already exists, it will be overwritten.

        Note: This operation is not thread-safe.

        Args:
            entity: The `BaseEntity` object to add.

        Returns:
            The ID of the added entity.
        """
        self._entities[entity.entity_id] = entity
        return entity.entity_id

    async def get(self, entity_id: str) -> BaseEntity | None:
        """Retrieves an entity by its ID.

        Note: This operation is not thread-safe.

        Args:
            entity_id: The ID of the entity to retrieve.

        Returns:
            The `BaseEntity` object if found, otherwise `None`.
        """
        return self._entities.get(entity_id)

    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]:
        """Retrieves a batch of entities by their IDs.

        Note: This operation is not thread-safe.

        Args:
            entity_ids: A sequence of entity IDs to retrieve.

        Returns:
            A list of `BaseEntity` objects or `None` for each ID, in the
            same order as the input.
        """
        return [self._entities.get(eid) for eid in entity_ids]

    async def find_by_embedding(
        self,
        embedding: Sequence[float],
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[BaseEntity, float]]:
        """Finds entities with embeddings similar to a given vector.

        This performs a brute-force, O(n) search over all entities.

        Note: This operation is not thread-safe.

        Args:
            embedding: The embedding vector to compare against.
            threshold: The minimum cosine similarity to be considered a match.
            limit: The maximum number of similar entities to return.

        Returns:
            A list of tuples, each containing a matching `BaseEntity` and its
            similarity score, sorted by score in descending order.
        """
        results: list[tuple[BaseEntity, float]] = []
        for entity in self._entities.values():
            if entity.embedding is not None:
                similarity = _cosine_similarity(embedding, entity.embedding)
                if similarity >= threshold:
                    results.append((entity, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def find_by_name(
        self,
        name: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[BaseEntity]:
        """Finds entities by a case-insensitive name or synonym match.

        This performs an O(n) search over all entities.

        Note: This operation is not thread-safe.

        Args:
            name: The name to search for.
            entity_type: An optional entity type to narrow the search.
            limit: The maximum number of matching entities to return.

        Returns:
            A list of matching `BaseEntity` objects.
        """
        name_lower = name.lower()
        results: list[BaseEntity] = []
        for entity in self._entities.values():
            if entity_type is not None and entity.get_entity_type() != entity_type:
                continue
            if name_lower in entity.name.lower() or any(name_lower in syn.lower() for syn in entity.synonyms):
                results.append(entity)
                if len(results) >= limit:
                    break
        return results

    async def find_provisional_for_promotion(
        self,
        min_usage: int,
        min_confidence: float,
    ) -> list[BaseEntity]:
        """Finds provisional entities that meet promotion criteria.

        This performs an O(n) scan of all entities.

        Note: This operation is not thread-safe.

        Args:
            min_usage: The minimum `usage_count` required for promotion.
            min_confidence: The minimum `confidence` score required.

        Returns:
            A list of provisional `BaseEntity` objects that are eligible
            for promotion.
        """
        promotable_entities = [entity for entity in self._entities.values() if entity.status == EntityStatus.PROVISIONAL and entity.usage_count >= min_usage and entity.confidence >= min_confidence]
        return promotable_entities

    async def update(self, entity: BaseEntity) -> bool:
        """Updates an existing entity in the storage.

        If the entity ID does not exist, the operation fails.

        Note: This operation is not thread-safe.

        Args:
            entity: The entity object with updated data.

        Returns:
            `True` if the update was successful, `False` if the entity ID
            was not found.
        """
        if entity.entity_id not in self._entities:
            return False
        self._entities[entity.entity_id] = entity
        return True

    async def promote(
        self,
        entity_id: str,
        new_entity_id: str,
        canonical_ids: dict[str, str],
    ) -> BaseEntity | None:
        """Promotes a provisional entity to a canonical one.

        This involves changing the entity's ID, updating its status, and
        assigning canonical IDs. The old entity record is removed.

        Note: This operation is not thread-safe.

        Args:
            entity_id: The current ID of the provisional entity.
            new_entity_id: The new, canonical ID for the entity.
            canonical_ids: A dictionary of canonical IDs to assign.

        Returns:
            The updated, canonical `BaseEntity` object if the original entity
            was found, otherwise `None`.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return None

        # Create new entity with updated status, ID, and canonical IDs
        promoted = entity.model_copy(
            update={
                "entity_id": new_entity_id,
                "status": EntityStatus.CANONICAL,
                "canonical_ids": canonical_ids,
            }
        )

        # Remove old entry, add new one with the new ID
        del self._entities[entity_id]
        self._entities[new_entity_id] = promoted
        return promoted

    async def merge(
        self,
        source_ids: Sequence[str],
        target_id: str,
    ) -> bool:
        """Merges multiple source entities into a single target entity.

        This operation combines synonyms and usage counts from source entities
        into the target entity and then deletes the source entities.

        Note: This operation is not thread-safe.

        Args:
            source_ids: A sequence of entity IDs to merge and then delete.
            target_id: The ID of the entity that will absorb the sources.

        Returns:
            `True` if the merge was successful, `False` if the target or any
            source entity was not found.
        """
        target = self._entities.get(target_id)
        if target is None:
            return False

        # Collect data from sources
        sources = [self._entities.get(sid) for sid in source_ids]
        if any(s is None for s in sources):
            return False

        # Combine synonyms and usage counts
        all_synonyms = set(target.synonyms)
        total_usage = target.usage_count
        for source in sources:
            if source is not None:
                all_synonyms.add(source.name)
                all_synonyms.update(source.synonyms)
                total_usage += source.usage_count

        all_synonyms.discard(target.name)  # Don't include target name as synonym

        # Create merged entity
        merged = target.model_copy(
            update={
                "synonyms": tuple(sorted(all_synonyms)),
                "usage_count": total_usage,
            }
        )
        self._entities[target_id] = merged

        # Remove source entities
        for sid in source_ids:
            if sid in self._entities and sid != target_id:
                del self._entities[sid]

        return True

    async def delete(self, entity_id: str) -> bool:
        """Deletes an entity from storage by its ID.

        Note: This operation is not thread-safe.

        Args:
            entity_id: The ID of the entity to delete.

        Returns:
            `True` if the entity was found and deleted, `False` otherwise.
        """
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    async def count(self) -> int:
        """Returns the total number of entities in storage.

        Note: This operation is not thread-safe if other operations are
        modifying the storage concurrently.

        Returns:
            The total count of entities.
        """
        return len(self._entities)

    async def list_all(
        self,
        status: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[BaseEntity]:
        """Lists entities from storage, with optional filtering and pagination.

        This performs an O(n) scan if filtering by status.

        Note: This operation is not thread-safe.

        Args:
            status: An optional status (`"canonical"` or `"provisional"`) to
                    filter by.
            limit: The maximum number of entities to return.
            offset: The starting offset for pagination.

        Returns:
            A list of `BaseEntity` objects.
        """
        if status is None:
            entities = list(self._entities.values())
        else:
            entities = [e for e in self._entities.values() if e.status.value == status]
        return entities[offset : offset + limit]


class InMemoryRelationshipStorage(RelationshipStorageInterface):
    """In-memory relationship storage using triple keys.

    Stores relationships in a dictionary keyed by (subject_id, predicate, object_id)
    tuples. This ensures uniqueness of triples and provides O(1) lookup for
    specific relationships.

    Traversal queries (get_by_subject, get_by_object) are O(n) as they scan
    all relationships. For large graphs, use a database with proper indices.

    Thread safety: Not thread-safe. For concurrent access, use external
    synchronization or a production database backend.

    Example:
        ```python
        storage = InMemoryRelationshipStorage()
        await storage.add(my_relationship)
        outgoing = await storage.get_by_subject(entity_id)
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty relationship storage."""
        # Key is (subject_id, predicate, object_id) to ensure triple uniqueness
        self._relationships: dict[tuple[str, str, str], BaseRelationship] = {}

    def _make_key(self, rel: BaseRelationship) -> tuple[str, str, str]:
        """Create a dictionary key from a relationship's triple."""
        return (rel.subject_id, rel.predicate, rel.object_id)

    async def add(self, relationship: BaseRelationship) -> str:
        """Adds a new relationship to the storage.

        If a relationship with the same triple (subject, predicate, object)
        already exists, it will be overwritten.

        Note: This operation is not thread-safe.

        Args:
            relationship: The `BaseRelationship` object to add.

        Returns:
            A string representation of the relationship's key.
        """
        key = self._make_key(relationship)
        self._relationships[key] = relationship
        return f"{key[0]}:{key[1]}:{key[2]}"

    async def get_by_subject(
        self,
        subject_id: str,
        predicate: str | None = None,
    ) -> list[BaseRelationship]:
        """Retrieves all relationships originating from a given subject.

        This performs an O(n) scan of all relationships.

        Note: This operation is not thread-safe.

        Args:
            subject_id: The ID of the subject entity.
            predicate: An optional predicate to filter the relationships.

        Returns:
            A list of matching `BaseRelationship` objects.
        """
        results = []
        for rel in self._relationships.values():
            if rel.subject_id == subject_id:
                if predicate is None or rel.predicate == predicate:
                    results.append(rel)
        return results

    async def get_by_object(
        self,
        object_id: str,
        predicate: str | None = None,
    ) -> list[BaseRelationship]:
        """Retrieves all relationships pointing to a given object.

        This performs an O(n) scan of all relationships.

        Note: This operation is not thread-safe.

        Args:
            object_id: The ID of the object entity.
            predicate: An optional predicate to filter the relationships.

        Returns:
            A list of matching `BaseRelationship` objects.
        """
        results = []
        for rel in self._relationships.values():
            if rel.object_id == object_id:
                if predicate is None or rel.predicate == predicate:
                    results.append(rel)
        return results

    async def find_by_triple(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> BaseRelationship | None:
        """Finds a specific relationship by its full triple.

        This is an O(1) lookup.

        Note: This operation is not thread-safe.

        Args:
            subject_id: The ID of the subject entity.
            predicate: The predicate of the relationship.
            object_id: The ID of the object entity.

        Returns:
            The `BaseRelationship` object if found, otherwise `None`.
        """
        return self._relationships.get((subject_id, predicate, object_id))

    async def update_entity_references(
        self,
        old_entity_id: str,
        new_entity_id: str,
    ) -> int:
        """Updates all relationships that reference an old entity ID.

        This is used during entity promotion or merging to retarget
        relationships from an old ID to a new one.

        Note: This operation is not thread-safe.

        Args:
            old_entity_id: The entity ID to be replaced.
            new_entity_id: The new entity ID to use.

        Returns:
            The number of relationships that were updated.
        """
        updated_count = 0
        to_update: list[tuple[tuple[str, str, str], BaseRelationship]] = []

        for key, rel in self._relationships.items():
            new_subject = rel.subject_id
            new_object = rel.object_id
            needs_update = False

            if rel.subject_id == old_entity_id:
                new_subject = new_entity_id
                needs_update = True
            if rel.object_id == old_entity_id:
                new_object = new_entity_id
                needs_update = True

            if needs_update:
                updated_rel = rel.model_copy(
                    update={
                        "subject_id": new_subject,
                        "object_id": new_object,
                    }
                )
                to_update.append((key, updated_rel))
                updated_count += 1

        # Apply updates
        for old_key, new_rel in to_update:
            del self._relationships[old_key]
            new_key = self._make_key(new_rel)
            self._relationships[new_key] = new_rel

        return updated_count

    async def get_by_document(
        self,
        document_id: str,
    ) -> list[BaseRelationship]:
        """Retrieves all relationships sourced from a specific document.

        This performs an O(n) scan of all relationships.

        Note: This operation is not thread-safe.

        Args:
            document_id: The ID of the source document.

        Returns:
            A list of matching `BaseRelationship` objects.
        """
        return [rel for rel in self._relationships.values() if document_id in rel.source_documents]

    async def delete(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> bool:
        """Deletes a relationship from storage by its triple.

        Note: This operation is not thread-safe.

        Args:
            subject_id: The ID of the subject entity.
            predicate: The predicate of the relationship.
            object_id: The ID of the object entity.

        Returns:
            `True` if the relationship was found and deleted, `False` otherwise.
        """
        key = (subject_id, predicate, object_id)
        if key in self._relationships:
            del self._relationships[key]
            return True
        return False

    async def count(self) -> int:
        """Returns the total number of relationships in storage.

        Note: This operation is not thread-safe if other operations are
        modifying the storage concurrently.

        Returns:
            The total count of relationships.
        """
        return len(self._relationships)

    async def list_all(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[BaseRelationship]:
        """Lists all relationships from storage, with optional pagination.

        Note: This operation is not thread-safe.

        Args:
            limit: The maximum number of relationships to return.
            offset: The starting offset for pagination.

        Returns:
            A list of `BaseRelationship` objects.
        """
        relationships = list(self._relationships.values())
        return relationships[offset : offset + limit]


class InMemoryDocumentStorage(DocumentStorageInterface):
    """In-memory document storage using a dictionary keyed by document_id.

    Stores documents in a simple `dict[str, BaseDocument]` structure.
    Document lookups by ID are O(1); lookups by source URI are O(n).

    Thread safety: Not thread-safe. For concurrent access, use external
    synchronization or a production database backend.

    Example:
        ```python
        storage = InMemoryDocumentStorage()
        await storage.add(my_document)
        doc = await storage.get(my_document.document_id)
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty document storage."""
        self._documents: dict[str, BaseDocument] = {}

    async def add(self, document: BaseDocument) -> str:
        """Adds a new document to the storage.

        If a document with the same ID already exists, it will be
        overwritten.

        Note: This operation is not thread-safe.

        Args:
            document: The `BaseDocument` object to add.

        Returns:
            The ID of the added document.
        """
        self._documents[document.document_id] = document
        return document.document_id

    async def get(self, document_id: str) -> BaseDocument | None:
        """Retrieves a document by its ID.

        Note: This operation is not thread-safe.

        Args:
            document_id: The ID of the document to retrieve.

        Returns:
            The `BaseDocument` object if found, otherwise `None`.
        """
        return self._documents.get(document_id)

    async def find_by_source(self, source_uri: str) -> BaseDocument | None:
        """Finds a document by its source URI.

        This performs an O(n) scan of all documents.

        Note: This operation is not thread-safe.

        Args:
            source_uri: The source URI to search for.

        Returns:
            The `BaseDocument` object if found, otherwise `None`.
        """
        for doc in self._documents.values():
            if doc.source_uri == source_uri:
                return doc
        return None

    async def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """Lists document IDs from storage, with optional pagination.

        Note: This operation is not thread-safe.

        Args:
            limit: The maximum number of document IDs to return.
            offset: The starting offset for pagination.

        Returns:
            A list of document ID strings.
        """
        all_ids = list(self._documents.keys())
        return all_ids[offset : offset + limit]

    async def delete(self, document_id: str) -> bool:
        """Deletes a document from storage by its ID.

        Note: This operation is not thread-safe.

        Args:
            document_id: The ID of the document to delete.

        Returns:
            `True` if the document was found and deleted, `False` otherwise.
        """
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False

    async def count(self) -> int:
        """Returns the total number of documents in storage.

        Note: This operation is not thread-safe if other operations are
        modifying the storage concurrently.

        Returns:
            The total count of documents.
        """
        return len(self._documents)
