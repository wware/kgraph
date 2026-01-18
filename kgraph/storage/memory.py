"""In-memory storage implementations for testing and development."""

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
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class InMemoryEntityStorage(EntityStorageInterface):
    """In-memory entity storage for testing."""

    def __init__(self) -> None:
        self._entities: dict[str, BaseEntity] = {}

    async def add(self, entity: BaseEntity) -> str:
        self._entities[entity.entity_id] = entity
        return entity.entity_id

    async def get(self, entity_id: str) -> BaseEntity | None:
        return self._entities.get(entity_id)

    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]:
        return [self._entities.get(eid) for eid in entity_ids]

    async def find_by_embedding(
        self,
        embedding: Sequence[float],
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[BaseEntity, float]]:
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
        name_lower = name.lower()
        results: list[BaseEntity] = []
        for entity in self._entities.values():
            if entity_type is not None and entity.get_entity_type() != entity_type:
                continue
            if (
                name_lower in entity.name.lower()
                or any(name_lower in syn.lower() for syn in entity.synonyms)
            ):
                results.append(entity)
                if len(results) >= limit:
                    break
        return results

    async def find_provisional_for_promotion(
        self,
        min_usage: int,
        min_confidence: float,
    ) -> list[BaseEntity]:
        return [
            entity
            for entity in self._entities.values()
            if entity.status == EntityStatus.PROVISIONAL
            and entity.usage_count >= min_usage
            and entity.confidence >= min_confidence
        ]

    async def update(self, entity: BaseEntity) -> bool:
        if entity.entity_id not in self._entities:
            return False
        self._entities[entity.entity_id] = entity
        return True

    async def promote(
        self,
        entity_id: str,
        canonical_id: str,
    ) -> BaseEntity | None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return None
        # Create new entity with updated status and ID
        # We need to copy all fields - use model_copy with update
        promoted = entity.model_copy(
            update={
                "entity_id": canonical_id,
                "status": EntityStatus.CANONICAL,
            }
        )
        # Remove old entry, add new one
        del self._entities[entity_id]
        self._entities[canonical_id] = promoted
        return promoted

    async def merge(
        self,
        source_ids: Sequence[str],
        target_id: str,
    ) -> bool:
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
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    async def count(self) -> int:
        return len(self._entities)

    async def list_all(
        self,
        status: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[BaseEntity]:
        if status is None:
            entities = list(self._entities.values())
        else:
            entities = [
                e for e in self._entities.values()
                if e.status.value == status
            ]
        return entities[offset : offset + limit]


class InMemoryRelationshipStorage(RelationshipStorageInterface):
    """In-memory relationship storage for testing."""

    def __init__(self) -> None:
        # Key is (subject_id, predicate, object_id)
        self._relationships: dict[tuple[str, str, str], BaseRelationship] = {}

    def _make_key(self, rel: BaseRelationship) -> tuple[str, str, str]:
        return (rel.subject_id, rel.predicate, rel.object_id)

    async def add(self, relationship: BaseRelationship) -> str:
        key = self._make_key(relationship)
        self._relationships[key] = relationship
        return f"{key[0]}:{key[1]}:{key[2]}"

    async def get_by_subject(
        self,
        subject_id: str,
        predicate: str | None = None,
    ) -> list[BaseRelationship]:
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
        return self._relationships.get((subject_id, predicate, object_id))

    async def update_entity_references(
        self,
        old_entity_id: str,
        new_entity_id: str,
    ) -> int:
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
        return [
            rel for rel in self._relationships.values()
            if document_id in rel.source_documents
        ]

    async def delete(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> bool:
        key = (subject_id, predicate, object_id)
        if key in self._relationships:
            del self._relationships[key]
            return True
        return False

    async def count(self) -> int:
        return len(self._relationships)

    async def list_all(
        self,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[BaseRelationship]:
        relationships = list(self._relationships.values())
        return relationships[offset : offset + limit]


class InMemoryDocumentStorage(DocumentStorageInterface):
    """In-memory document storage for testing."""

    def __init__(self) -> None:
        self._documents: dict[str, BaseDocument] = {}

    async def add(self, document: BaseDocument) -> str:
        self._documents[document.document_id] = document
        return document.document_id

    async def get(self, document_id: str) -> BaseDocument | None:
        return self._documents.get(document_id)

    async def find_by_source(self, source_uri: str) -> BaseDocument | None:
        for doc in self._documents.values():
            if doc.source_uri == source_uri:
                return doc
        return None

    async def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        all_ids = list(self._documents.keys())
        return all_ids[offset : offset + limit]

    async def delete(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False

    async def count(self) -> int:
        return len(self._documents)
