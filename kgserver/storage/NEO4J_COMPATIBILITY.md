# Neo4j Compatibility

> **Note**: This document describes the theoretical Neo4j compatibility of the storage interfaces. **Neo4j is not currently implemented or planned for this project.** The current implementations use PostgreSQL for production and SQLite for testing. This document is preserved for architectural reference only.

## 1. Introduction

The storage interfaces are designed to be **storage-agnostic**, enabling the knowledge graph server to be backed by various persistence technologies. While the current implementations target SQLite (for testing) and PostgreSQL (for production), the abstract interface design is suitable for graph database backends like **Neo4j**.

The interfaces define operations in terms of entities and relationships rather than tables and SQL, making them naturally compatible with graph database concepts.

## 2. Current Interface Design

The server uses a single abstract `StorageInterface` defined in `storage/interfaces.py`:

### StorageInterface
Combines all storage operations into a unified abstraction.

**Key methods:**
- `load_bundle(bundle_manifest, bundle_path)`: Load a data bundle into storage.
- `get_entity(entity_id)`: Retrieve an entity by ID.
- `get_entities(limit, offset)`: List entities with pagination.
- `find_relationships(subject_id, predicate, object_id, limit)`: Query relationships with optional filters.
- `get_relationship(subject_id, predicate, object_id)`: Get a specific relationship by triple.
- `get_relationships(limit, offset)`: List relationships with pagination.
- `close()`: Clean up connections.

## 3. Neo4j Compatibility

This generic interface maps naturally to Neo4j's graph model. The `find_relationships()` method, in particular, aligns well with graph traversal queries.

## 4. Current Implementation Note

The codebase currently provides two storage implementations:

### SQLiteStorage (`storage/backends/sqlite.py`)
- **Purpose**: Testing, development, small datasets
- **Features**: In-memory support, file-based storage
- **Tradeoffs**: Slower for complex relationship queries compared to graph databases

### PostgresStorage (`storage/backends/postgres.py`)
- **Purpose**: Production deployment
- **Features**: Robust relational model
- **Tradeoffs**: Relationship traversals require JOINs, harder to visualize graph structure

### Design Philosophy

The interface design is **intentionally abstract** to support backend swapping:

- The interfaces themselves make no assumptions about the backend technology
- A Neo4j backend could be used as an alternative implementation for the `StorageInterface`.

The clean abstraction means you can choose the right tool for each deployment scenario without changing the core server logic.

## 5. Conclusion

The storage interfaces are **compatible with Neo4j** and graph database implementations. The design is:

- **Graph-native**: Thinks in terms of entities and relationships rather than tables and rows
- **Traversal-friendly**: Methods like `find_relationships()` map directly to graph queries
- **Backend-agnostic**: No SQL-specific assumptions limit implementation choices
- **Clean abstractions**: Server logic works with any conforming implementation

Implementing a Neo4j backend would be straightforward and would unlock powerful graph capabilities such as native support for multi-hop traversals and pattern matching.