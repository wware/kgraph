# Knowledge Graph GraphQL API

This document describes the **GraphQL API** for querying the knowledge graph. The API is designed to be easy to use, hard to abuse, and domain-neutral - it doesn't force premature ontology decisions.

## API Design Principles

* **Read-only**: The API currently supports queries only (no mutations).
* **Explicit pagination**: All list queries require pagination parameters to prevent unbounded result sets.
* **Canonical fields**: Only standard server schema fields are first-class; domain-specific data stays in `properties: JSON`.
* **Narrow filtering**: Filtering supports exact matches and a few safe pattern-matching helpers.
* **No graph traversal**: The API does not currently support graph traversal queries.

---

## GraphQL Schema

### Scalars

* `JSON` (for `properties`)
* `DateTime` (for timestamps, returned as ISO 8601 strings)

### Types

**Entity**

* `entityId: ID!`
* `entityType: String!`
* `name: String`
* `status: String`
* `confidence: Float`
* `usageCount: Int`
* `source: String`
* `synonyms: [String!]!`
* `properties: JSON`

Note: Entities and relationships are treated as immutable and eternal - they have no `createdAt` timestamps. Ingestion happens in a separate process (`kgraph`), and the server treats all data as pre-existing.

**Relationship**

* `subjectId: ID!`
* `predicate: String!`
* `objectId: ID!`
* `confidence: Float`
* `sourceDocuments: [ID!]!`
* `properties: JSON`

Note: Relationships have internal database identifiers, but these are not exposed in the GraphQL schema. The relationship identity is the triple `(subjectId, predicate, objectId)`. See "Implementation details for API consumers" below for more information.

### Pagination types

The API uses simple pagination wrappers (not Relay-style connections) that provide all the information clients need for pagination.

**EntityPage**

* `items: [Entity!]!`
* `total: Int!`
* `limit: Int!`
* `offset: Int!`

**RelationshipPage**

* `items: [Relationship!]!`
* `total: Int!`
* `limit: Int!`
* `offset: Int!`

---

## Queries

### Fetch by ID

These queries retrieve a single entity or relationship by their identifiers:

* `entity(id: ID!): Entity` - Retrieve a single entity by its ID
* `relationship(subjectId: ID!, predicate: String!, objectId: ID!): Relationship` - Retrieve a single relationship by its triple (subject, predicate, object)

### List queries with pagination

These queries return paginated lists of entities or relationships:

* `entities(limit: Int = 100, offset: Int = 0, filter: EntityFilter): EntityPage!` - List entities with optional filtering
* `relationships(limit: Int = 100, offset: Int = 0, filter: RelationshipFilter): RelationshipPage!` - List relationships with optional filtering

Note: The `limit` parameter is capped at a maximum value (default 100, configurable via `GRAPHQL_MAX_LIMIT`). Requests exceeding the maximum are silently capped and logged. See "Implementation details for API consumers" below.

### Filters

Filters allow you to narrow down query results. The API supports a focused set of filter options:


**EntityFilter**

* `entityType: String` - Filter by exact entity type
* `name: String` - Filter by exact name match
* `nameContains: String` - Filter by name containing the specified string (case-insensitive)
* `source: String` - Filter by exact source value
* `status: String` - Filter by exact status value

**RelationshipFilter**

* `subjectId: ID` - Filter by subject entity ID
* `objectId: ID` - Filter by object entity ID
* `predicate: String` - Filter by exact predicate value

Note: The API does not currently support arbitrary boolean logic, regex patterns, or querying within the `properties` JSON field.

---

## Complete Schema Definition

The complete GraphQL schema in SDL (Schema Definition Language) format:

```graphql
scalar JSON
scalar DateTime

type Entity {
  entityId: ID!
  entityType: String!
  name: String
  status: String
  confidence: Float
  usageCount: Int
  source: String
  synonyms: [String!]!
  properties: JSON
}

type Relationship {
  subjectId: ID!
  predicate: String!
  objectId: ID!
  confidence: Float
  sourceDocuments: [ID!]!
  properties: JSON
}

input EntityFilter {
  entityType: String
  name: String
  nameContains: String
  source: String
  status: String
}

input RelationshipFilter {
  subjectId: ID
  objectId: ID
  predicate: String
}

type EntityPage {
  items: [Entity!]!
  total: Int!
  limit: Int!
  offset: Int!
}

type RelationshipPage {
  items: [Relationship!]!
  total: Int!
  limit: Int!
  offset: Int!
}

type Query {
  entity(id: ID!): Entity
  entities(limit: Int = 100, offset: Int = 0, filter: EntityFilter): EntityPage!

  relationships(limit: Int = 100, offset: Int = 0, filter: RelationshipFilter): RelationshipPage!
}
```

---

## API Characteristics

* **Pagination**: All list queries support `limit` and `offset` parameters, similar to REST API pagination.
* **Efficient counting**: The `total` field in paginated results is computed efficiently using database count operations.
* **Simple filtering**: Filter fields map directly to database queries for predictable performance.
* **Flexible properties**: The `properties` JSON field allows domain-specific data without requiring schema changes.

---

## Field Naming Convention

The API uses camelCase for GraphQL field names, which map to snake_case in the underlying database:

* GraphQL fields: `entityId`, `entityType`, `usageCount`, `sourceDocuments`
* Database fields: `entity_id`, `entity_type`, `usage_count`, `source_documents`

This convention is consistent throughout the API.

---

## Implementation details for API consumers

### Relationship `id` field

The `Relationship` type does not expose an `id` field in the GraphQL schema, even though relationships have internal identifiers in the database. The relationship identity is represented by the triple `(subjectId, predicate, objectId)`. If an `id` field is needed in the future, it can be added to the schema, but the current design uses the triple as the canonical identifier.

### Count queries and performance

Count queries (`total` in paginated results) are implemented in the storage layer using optimized database `COUNT(*)` operations. This ensures efficient counting even with large datasets and complex filters. The counts are computed server-side and returned as part of the pagination metadata.

### Bundle introspection query

The `bundle` query provides metadata about the loaded data bundle. It retrieves information from the bundle tracking system, including the bundle ID, domain, creation timestamp, and any associated metadata.

### Maximum limit enforcement

All paginated queries (`entities` and `relationships`) enforce a maximum limit to prevent resource exhaustion. The default maximum is 100 items per page, but this can be configured via the `GRAPHQL_MAX_LIMIT` environment variable. If a client requests a limit greater than the maximum, the server will silently cap it to the maximum value and log a warning message. This ensures that:

* Clients receive valid responses without errors
* Server logs capture when limits are exceeded (useful for monitoring in cloud deployments)
* The API remains user-friendly while protecting server resources

The actual limit applied is always returned in the `limit` field of the pagination response, so clients can detect when their requested limit was capped.

### Bundle query example

```graphql
query GetBundleInfo {
  bundle {
    bundleId
    domain
    createdAt
    metadata
  }
}
```

The `bundle` query returns `null` if no bundle has been loaded.

---

## Future: Graph Traversal Queries

### Why Not Now?

Graph traversal queries are not currently supported in the API. This is an intentional design decision to keep the initial API surface minimal and predictable. Graph traversal languages (like Cypher, Gremlin, or GraphQL extensions) introduce significant complexity:

* **Query complexity**: Traversal queries can be arbitrarily complex, making it difficult to predict performance and resource usage
* **Security concerns**: Complex queries can be used for denial-of-service attacks or to extract large amounts of data
* **Implementation complexity**: Supporting traversal requires sophisticated query planning, execution engines, and optimization
* **Domain-agnostic design**: The current API avoids making assumptions about graph structure that might not apply to all domains

### What Graph Traversal Would Enable

If graph traversal queries were added, they would enable powerful use cases:

* **Multi-hop queries**: Find entities connected through multiple relationship hops (e.g., "find all characters who co-occur with characters who appear in the same stories as Sherlock Holmes")
* **Path finding**: Discover paths between entities (e.g., "find the shortest path from entity A to entity B")
* **Pattern matching**: Query for subgraph patterns (e.g., "find all triangles of co-occurrence relationships")
* **Aggregations over paths**: Compute metrics along traversal paths (e.g., "find the most central entities by relationship count")
* **Recursive queries**: Traverse hierarchical or recursive relationship structures

These capabilities would make the API suitable for more sophisticated graph analytics and exploration workflows.

### Engineering Trade-offs to Consider

If you decide to implement graph traversal, consider these trade-offs:

**Performance and Scalability**
* Traversal queries can be expensive, especially with deep paths or large branching factors
* Need query timeout mechanisms and depth limits to prevent runaway queries
* Consider caching strategies for common traversal patterns
* May require specialized graph database backends (Neo4j, Amazon Neptune) for optimal performance

**Query Language Design**
* Choose between existing languages (Cypher, Gremlin) vs. custom GraphQL extensions
* Existing languages are powerful but add dependency on external query engines
* Custom GraphQL extensions integrate better but require more design and implementation work
* Consider whether to support both simple and advanced traversal syntax

**Security and Resource Management**
* Implement query complexity scoring to reject or limit expensive queries
* Add traversal depth limits (e.g., max 3-5 hops) to prevent exponential explosion
* Consider query result size limits even with pagination
* May need query cost estimation before execution

**Storage Backend Compatibility**
* Current SQL-based backends (PostgreSQL, SQLite) can support simple traversals via recursive CTEs
* Complex traversals may require graph database backends for acceptable performance
* Consider whether traversal should be optional (only if graph backend is available) or always supported
* Storage interface abstraction should hide traversal implementation details

**API Design**
* Should traversal be separate queries (e.g., `traverse(from: ID!, depth: Int!)`) or embedded in existing queries?
* How to paginate traversal results (by path? by entity? by depth level?)
* Should traversal support filtering at each hop?
* Consider backward compatibility - traversal shouldn't break existing simple queries

### Where to Implement

If you implement graph traversal, here's where the code should go:

**GraphQL Schema** (`query/graphql_schema.py`)
* Add traversal query methods to the `Query` class
* Define traversal-specific types (e.g., `Path`, `TraversalResult`, `TraversalStep`)
* Add traversal input types (e.g., `TraversalFilter`, `PathOptions`)
* Keep traversal queries separate from simple queries for clarity

**Storage Interface** (`storage/interfaces.py`)
* Add abstract traversal methods like `traverse_paths()` or `find_connected_entities()`
* Methods should accept traversal parameters (max depth, filters, direction)
* Return paginated results with path information

**Storage Backends** (`storage/backends/`)
* **PostgreSQL**: Implement using recursive CTEs (WITH RECURSIVE) for simple traversals
* **SQLite**: Similar recursive CTE approach, but with performance limitations
* **Future graph backends**: Use native graph query languages (Cypher for Neo4j, etc.)
* Consider adding a dedicated graph backend abstraction if multiple graph databases are supported

**Query Planning and Execution**
* Consider a separate module (e.g., `query/traversal.py`) for traversal query planning
* Parse traversal queries into execution plans
* Apply security limits (depth, result size) before execution
* Optimize traversal order and direction based on relationship cardinality

**Documentation and Examples**
* Update `query/graphql_examples.py` with traversal query examples
* Document traversal query syntax and limitations
* Provide performance guidance for different traversal patterns

The current architecture (separate storage interface, GraphQL schema layer, and backend implementations) provides a good foundation for adding traversal without disrupting existing functionality.
