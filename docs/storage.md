# Storage Backends

The framework defines storage interfaces for entities, relationships, and documents. Implement these to connect to your preferred database.

## Storage Interfaces

### EntityStorageInterface

```python
from kgraph.storage import EntityStorageInterface

class EntityStorageInterface(ABC):
    async def add(self, entity: BaseEntity) -> str: ...
    async def get(self, entity_id: str) -> BaseEntity | None: ...
    async def get_batch(self, entity_ids: Sequence[str]) -> list[BaseEntity | None]: ...
    async def find_by_embedding(
        self, embedding: Sequence[float], threshold: float, limit: int
    ) -> list[tuple[BaseEntity, float]]: ...
    async def find_by_name(
        self, name: str, entity_type: str | None, limit: int
    ) -> list[BaseEntity]: ...
    async def find_provisional_for_promotion(
        self, min_usage: int, min_confidence: float
    ) -> list[BaseEntity]: ...
    async def update(self, entity: BaseEntity) -> bool: ...
    async def promote(self, entity_id: str, canonical_id: str) -> BaseEntity | None: ...
    async def merge(self, source_ids: Sequence[str], target_id: str) -> bool: ...
    async def delete(self, entity_id: str) -> bool: ...
    async def count(self) -> int: ...
```

### RelationshipStorageInterface

```python
from kgraph.storage import RelationshipStorageInterface

class RelationshipStorageInterface(ABC):
    async def add(self, relationship: BaseRelationship) -> str: ...
    async def get_by_subject(
        self, subject_id: str, predicate: str | None
    ) -> list[BaseRelationship]: ...
    async def get_by_object(
        self, object_id: str, predicate: str | None
    ) -> list[BaseRelationship]: ...
    async def find_by_triple(
        self, subject_id: str, predicate: str, object_id: str
    ) -> BaseRelationship | None: ...
    async def update_entity_references(
        self, old_entity_id: str, new_entity_id: str
    ) -> int: ...
    async def delete(
        self, subject_id: str, predicate: str, object_id: str
    ) -> bool: ...
    async def count(self) -> int: ...
```

### DocumentStorageInterface

```python
from kgraph.storage import DocumentStorageInterface

class DocumentStorageInterface(ABC):
    async def add(self, document: BaseDocument) -> str: ...
    async def get(self, document_id: str) -> BaseDocument | None: ...
    async def find_by_source(self, source_uri: str) -> BaseDocument | None: ...
    async def list_ids(self, limit: int, offset: int) -> list[str]: ...
    async def delete(self, document_id: str) -> bool: ...
    async def count(self) -> int: ...
```

## In-Memory Implementation

The framework includes in-memory storage for testing and development:

```python
from kgraph.storage import (
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
    InMemoryDocumentStorage,
)

entity_storage = InMemoryEntityStorage()
relationship_storage = InMemoryRelationshipStorage()
document_storage = InMemoryDocumentStorage()
```

## Implementing a Database Backend

Example PostgreSQL implementation sketch:

```python
import asyncpg
from kgraph.storage import EntityStorageInterface
from kgraph import BaseEntity

class PostgresEntityStorage(EntityStorageInterface):
    def __init__(self, pool: asyncpg.Pool, domain_name: str):
        self._pool = pool
        self._domain = domain_name

    async def add(self, entity: BaseEntity) -> str:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO entities (id, domain, type, status, name, data)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET data = $6
                """,
                entity.entity_id,
                self._domain,
                entity.get_entity_type(),
                entity.status.value,
                entity.name,
                entity.model_dump_json(),
            )
        return entity.entity_id

    async def get(self, entity_id: str) -> BaseEntity | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM entities WHERE id = $1 AND domain = $2",
                entity_id,
                self._domain,
            )
        if row is None:
            return None
        # Deserialize based on entity type
        return self._deserialize(row["data"])

    async def find_by_embedding(
        self,
        embedding: Sequence[float],
        threshold: float = 0.8,
        limit: int = 10,
    ) -> list[tuple[BaseEntity, float]]:
        # Use pgvector for similarity search
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data, 1 - (embedding <=> $1::vector) as similarity
                FROM entities
                WHERE domain = $2
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                list(embedding),
                self._domain,
                threshold,
                limit,
            )
        return [(self._deserialize(r["data"]), r["similarity"]) for r in rows]

    # ... implement remaining methods
```

## Vector Search Considerations

For `find_by_embedding`, consider:

- **pgvector**: PostgreSQL extension for vector similarity
- **Pinecone/Weaviate/Qdrant**: Dedicated vector databases
- **FAISS**: In-process vector index for smaller datasets

The interface returns `(entity, similarity_score)` tuples sorted by descending similarity, allowing callers to filter or rank results.
