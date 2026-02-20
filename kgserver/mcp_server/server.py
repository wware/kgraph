"""
MCP Server implementation using FastMCP.

Provides tools for querying the knowledge graph via the Model Context Protocol.
"""

from typing import Optional, Any
from contextlib import contextmanager, closing
from fastmcp import FastMCP
import strawberry

from query.storage_factory import get_engine, get_storage
from query.graphql_schema import Query

# Schema for executing GraphQL queries (same as HTTP /graphql)
_graphql_schema = strawberry.Schema(query=Query)

# Create the FastMCP server instance
mcp_server = FastMCP(
    name="knowledge-graph",
    instructions=("Query interface for knowledge graph with tools for finding entities, " "relationships, and bundle information. Supports multiple knowledge domains " "with canonical IDs."),
)


@contextmanager
def _get_storage():
    """Context manager for getting a storage instance with proper lifecycle management."""
    # Initialize engine if needed
    get_engine()
    # Get storage from generator - use it directly as a context manager
    # Since get_storage() is a generator, we can use it with contextlib.closing
    # or wrap it properly to avoid manual next() calls
    with closing(get_storage()) as storage_gen:
        storage = next(storage_gen, None)
        if storage is None:
            raise RuntimeError("Failed to get storage instance from generator")
        yield storage


@mcp_server.tool()
def get_entity(entity_id: str) -> dict | None:
    """
    Retrieve a specific entity by its ID.

    Returns the full entity data including all metadata, identifiers, and properties.

    Args:
        entity_id: The unique identifier of the entity

    Returns:
        Entity dictionary with fields: entityId, entityType, name, status,
        confidence, usageCount, source, canonicalUrl, synonyms, properties.
        Returns None if entity not found.
    """
    with _get_storage() as storage:
        entity = storage.get_entity(entity_id)
        if entity is None:
            return None
        return {
            "entityId": entity.entity_id,
            "entityType": entity.entity_type,
            "name": entity.name,
            "status": entity.status,
            "confidence": entity.confidence,
            "usageCount": entity.usage_count,
            "source": entity.source,
            "canonicalUrl": entity.canonical_url,
            "synonyms": entity.synonyms,
            "properties": entity.properties,
        }


@mcp_server.tool()
def list_entities(
    limit: int = 100,
    offset: int = 0,
    entity_type: Optional[str] = None,
    name: Optional[str] = None,
    name_contains: Optional[str] = None,
    source: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    """
    List entities with pagination and optional filtering.

    This tool provides flexible querying of entities in the knowledge graph.
    You can filter by type, name (exact or partial), source, and status.

    Args:
        limit: Maximum number of entities to return (default: 100, max: 100)
        offset: Number of entities to skip for pagination (default: 0)
        entity_type: Filter by entity type (e.g., "Disease", "Gene", "Drug")
        name: Exact name match filter
        name_contains: Partial name match filter (case-insensitive)
        source: Filter by source (e.g., "UMLS", "HGNC", "RxNorm")
        status: Filter by status (e.g., "canonical", "provisional")

    Returns:
        Dictionary with keys: items (list of entities), total (total count),
        limit, offset. Each entity has the same structure as get_entity.
    """
    with _get_storage() as storage:
        # Cap limit at 100
        limit = min(limit, 100)

        entities = storage.get_entities(
            limit=limit,
            offset=offset,
            entity_type=entity_type,
            name=name,
            name_contains=name_contains,
            source=source,
            status=status,
        )

        total = storage.count_entities(
            entity_type=entity_type,
            name=name,
            name_contains=name_contains,
            source=source,
            status=status,
        )

        return {
            "items": [
                {
                    "entityId": e.entity_id,
                    "entityType": e.entity_type,
                    "name": e.name,
                    "status": e.status,
                    "confidence": e.confidence,
                    "usageCount": e.usage_count,
                    "source": e.source,
                    "canonicalUrl": e.canonical_url,
                    "synonyms": e.synonyms,
                    "properties": e.properties,
                }
                for e in entities
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


@mcp_server.tool()
def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search for entities by name (convenience wrapper around list_entities).

    This performs a simple name-based search using the name_contains filter.
    For more advanced filtering, use list_entities directly.

    Args:
        query: Search query text (searches in entity names)
        entity_type: Optional entity type filter
        limit: Maximum number of results to return (default: 10, max: 100)

    Returns:
        List of matching entity dictionaries
    """
    with _get_storage() as storage:
        limit = min(limit, 100)

        entities = storage.get_entities(
            limit=limit,
            offset=0,
            entity_type=entity_type,
            name_contains=query,
        )

        return [
            {
                "entityId": e.entity_id,
                "entityType": e.entity_type,
                "name": e.name,
                "status": e.status,
                "confidence": e.confidence,
                "usageCount": e.usage_count,
                "source": e.source,
                "canonicalUrl": e.canonical_url,
                "synonyms": e.synonyms,
                "properties": e.properties,
            }
            for e in entities
        ]


@mcp_server.tool()
def get_relationship(
    subject_id: str,
    predicate: str,
    object_id: str,
) -> dict | None:
    """
    Retrieve a specific relationship by its triple (subject, predicate, object).

    Returns the full relationship data including confidence, source documents, and properties.

    Args:
        subject_id: The subject entity ID
        predicate: The relationship predicate/type
        object_id: The object entity ID

    Returns:
        Relationship dictionary with fields: subjectId, predicate, objectId,
        confidence, sourceDocuments, properties. Returns None if relationship not found.
    """
    with _get_storage() as storage:
        rel = storage.get_relationship(subject_id, predicate, object_id)
        if rel is None:
            return None
        return {
            "subjectId": rel.subject_id,
            "predicate": rel.predicate,
            "objectId": rel.object_id,
            "confidence": rel.confidence,
            "sourceDocuments": rel.source_documents,
            "properties": rel.properties,
        }


@mcp_server.tool()
def find_relationships(
    subject_id: Optional[str] = None,
    predicate: Optional[str] = None,
    object_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Find relationships with pagination and optional filtering.

    This tool provides flexible querying of relationships in the knowledge graph.
    You can filter by subject, predicate, object, or any combination.

    Args:
        limit: Maximum number of relationships to return (default: 100, max: 100)
        offset: Number of relationships to skip for pagination (default: 0)
        subject_id: Filter by subject entity ID
        predicate: Filter by relationship predicate/type
        object_id: Filter by object entity ID

    Returns:
        Dictionary with keys: items (list of relationships), total (total count),
        limit, offset. Each relationship has the same structure as get_relationship.
    """
    with _get_storage() as storage:
        limit = min(limit, 100)

        relationships = storage.find_relationships(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            limit=limit,
            offset=offset,
        )

        total = storage.count_relationships(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
        )

        return {
            "items": [
                {
                    "subjectId": r.subject_id,
                    "predicate": r.predicate,
                    "objectId": r.object_id,
                    "confidence": r.confidence,
                    "sourceDocuments": r.source_documents,
                    "properties": r.properties,
                }
                for r in relationships
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


@mcp_server.tool()
def get_bundle_info() -> dict | None:
    """
    Get bundle metadata for debugging and provenance.

    Returns information about the currently loaded knowledge graph bundle,
    including bundle ID, domain, creation timestamp, and metadata.

    Returns:
        Bundle dictionary with fields: bundleId, domain, createdAt, metadata.
        Returns None if no bundle is loaded.
    """
    with _get_storage() as storage:
        bundle = storage.get_bundle_info()
        if bundle is None:
            return None
        return {
            "bundleId": bundle.bundle_id,
            "domain": bundle.domain,
            "createdAt": bundle.created_at.isoformat() if bundle.created_at else None,
            "metadata": bundle.metadata,
        }


@mcp_server.tool()
def graphql_query(query: str, variables: Optional[dict[str, Any]] = None) -> dict:
    """
    Run an arbitrary GraphQL query against the knowledge graph.

    Uses the same schema as the HTTP /graphql endpoint. Use this for custom
    query shapes, multiple roots in one request, or when the discrete tools
    are not enough. The schema supports: entity(id), entities(limit, offset,
    filter), relationship(subjectId, predicate, objectId), relationships(limit,
    offset, filter), bundle.

    Args:
        query: GraphQL query string (e.g. "{ entity(id: \"MeSH:D001943\") { name } }").
        variables: Optional map of variable names to values for parameterized queries.

    Returns:
        Dictionary with "data" (result payload, or None if errors) and "errors"
        (list of error dicts, or None if successful). Same shape as standard
        GraphQL JSON responses.
    """
    with _get_storage() as storage:
        context = {"storage": storage}
        result = _graphql_schema.execute_sync(
            query,
            variable_values=variables or {},
            context_value=context,
        )
        return {
            "data": result.data,
            "errors": [{"message": e.message, "path": getattr(e, "path", None)} for e in (result.errors or [])] if result.errors else None,
        }
