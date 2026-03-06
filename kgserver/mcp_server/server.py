"""
MCP Server implementation using FastMCP.

Provides tools for querying the knowledge graph via the Model Context Protocol.
"""

import os
import zipfile
from collections import deque
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Optional, Any
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
def find_entities_within_hops(
    start_id: str,
    max_hops: int = 3,
    entity_type: Optional[str] = None,
) -> dict:
    """
    Find all entities within N hops of a starting entity using BFS traversal.

    Traverses the graph bidirectionally (follows both incoming and outgoing edges).
    Returns entities grouped by their hop distance from the start entity.
    Optionally filters results to a specific entity type.

    Args:
        start_id: The entity ID to start from (e.g. 'C0006142' for breast cancer)
        max_hops: Maximum number of hops to traverse (default 3, max 5)
        entity_type: Optional entity type filter (e.g. 'gene', 'drug', 'disease')

    Returns:
        Dictionary with start_id, max_hops, entity_type_filter, total_entities_found,
        and results_by_hop (hop distance -> list of entity dicts with entity_id,
        entity_type, name, status, hop_distance).
    """
    max_hops = min(max_hops, 5)  # Hard cap to prevent runaway traversal

    with _get_storage() as storage:
        # BFS
        visited: dict[str, int] = {start_id: 0}  # entity_id -> hop distance
        queue = deque([(start_id, 0)])

        while queue:
            current_id, distance = queue.popleft()
            if distance >= max_hops:
                continue

            # Outgoing edges
            outgoing = storage.find_relationships(subject_id=current_id, limit=200)
            for rel in outgoing:
                if rel.object_id not in visited:
                    visited[rel.object_id] = distance + 1
                    queue.append((rel.object_id, distance + 1))

            # Incoming edges
            incoming = storage.find_relationships(object_id=current_id, limit=200)
            for rel in incoming:
                if rel.subject_id not in visited:
                    visited[rel.subject_id] = distance + 1
                    queue.append((rel.subject_id, distance + 1))

        # Fetch entity details and group by distance
        results_by_hop: dict[int, list[dict]] = {}
        for entity_id, distance in visited.items():
            if entity_id == start_id:
                continue
            entity = storage.get_entity(entity_id=entity_id)
            if entity is None:
                continue
            if entity_type and entity.entity_type != entity_type:
                continue
            results_by_hop.setdefault(distance, []).append(
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "status": entity.status,
                    "hop_distance": distance,
                }
            )

        total = sum(len(entities) for entities in results_by_hop.values())
        return {
            "start_id": start_id,
            "max_hops": max_hops,
            "entity_type_filter": entity_type,
            "total_entities_found": total,
            "results_by_hop": {str(hop): entities for hop, entities in sorted(results_by_hop.items())},
        }


@mcp_server.tool()
async def ingest_paper(url: str) -> dict:
    """
    Ingest a medical paper from a URL into the knowledge graph.

    Kicks off a background job. Returns immediately with a job_id.
    Poll check_ingest_status(job_id) to track progress.
    Supports PMC full-text URLs and direct XML/JSON URLs.

    Returns:
        Dict with job_id, status, url, message.
    """
    from mcp_server.ingest_worker import enqueue_job

    with _get_storage() as storage:
        job = storage.create_ingest_job(url)
    await enqueue_job(job.id)
    return {
        "job_id": job.id,
        "status": "queued",
        "url": url,
        "message": "Job queued. Use check_ingest_status(job_id) to track progress.",
    }


@mcp_server.tool()
def check_ingest_status(job_id: str) -> dict:
    """
    Check the status of a paper ingestion job.

    Returns:
        Dict with job_id, status, url, paper_title, pmcid, entities_added,
        relationships_added, error, created_at, started_at, completed_at.
    Status values: queued | running | complete | failed | not_found
    """
    with _get_storage() as storage:
        job = storage.get_ingest_job(job_id)
    if job is None:
        return {
            "job_id": job_id,
            "status": "not_found",
            "url": "",
            "paper_title": None,
            "pmcid": None,
            "entities_added": 0,
            "relationships_added": 0,
            "error": "No such job",
            "created_at": None,
            "started_at": None,
            "completed_at": None,
        }
    return {
        "job_id": job.id,
        "status": job.status,
        "url": job.url,
        "paper_title": job.paper_title,
        "pmcid": job.pmcid,
        "entities_added": job.entities_added,
        "relationships_added": job.relationships_added,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def _get_bundle_path() -> Path:
    """Resolve BUNDLE_PATH; raise ValueError if unset or invalid."""
    path_str = os.getenv("BUNDLE_PATH")
    if not path_str:
        raise ValueError("BUNDLE_PATH is not set")
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"BUNDLE_PATH '{path_str}' does not exist")
    return path


def _read_from_bundle(relative_path: str) -> str:
    """Read file from bundle (directory or ZIP). Returns file contents as string."""
    bundle_path = _get_bundle_path()
    if bundle_path.suffix == ".zip":
        with zipfile.ZipFile(bundle_path, "r") as zf:
            with zf.open(relative_path) as f:
                return f.read().decode("utf-8", errors="replace")
    file_path = Path(bundle_path) / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"File not found in bundle: {relative_path}")
    return file_path.read_text(encoding="utf-8", errors="replace")


@mcp_server.tool()
def get_paper_source(paper_id: str, max_chars: Optional[int] = None) -> str:
    """
    Retrieve the raw JATS-XML source of a paper for mention-inspection diagnostics.

    Reads from bundle sources/ directory (or inside a ZIP bundle). Use with
    get_mentions to verify that extracted mentions match the source text.

    Args:
        paper_id: Paper identifier (e.g. PMC12345). Accepts with or without .xml suffix.
        max_chars: If set, truncate the returned string to this length. Note: blind
            truncation may cut mid-sentence for full JATS papers.
            # TODO: consider lxml/ElementTree snippet to extract just <abstract> and <body>
            instead of blind truncation.

    Returns:
        Raw XML string. Raises ValueError if BUNDLE_PATH unset or paper not found.
    """
    # Normalize: strip .xml if present, then always append
    base_id = paper_id.removesuffix(".xml") if paper_id.endswith(".xml") else paper_id
    filename = f"{base_id}.xml"
    relative_path = f"sources/{filename}"
    try:
        content = _read_from_bundle(relative_path)
    except FileNotFoundError:
        raise ValueError(f"Paper {paper_id} not found in sources/") from None
    if max_chars is not None:
        content = content[:max_chars]
    return content


@mcp_server.tool()
def get_mentions(paper_id: Optional[str] = None) -> list[dict]:
    """
    Retrieve entity mentions from the bundle for mention-inspection diagnostics.

    Reads mentions.jsonl (MentionRow schema). Filter by document_id when paper_id
    is provided. Use with get_paper_source to verify extracted mentions against
    the source text.

    Note: When filtering by paper_id, reads and parses the entire file then
    filters in memory. Fine for a diagnostic tool; not efficient for large corpora.

    Args:
        paper_id: If provided, filter to mentions where document_id matches.
            For medlit, values are like PMC12345. If None, return all mentions.

    Returns:
        List of mention dicts (entity_id, document_id, text_span, etc.).
        Empty list if mentions.jsonl missing or no matches.
    """
    try:
        content = _read_from_bundle("mentions.jsonl")
    except FileNotFoundError:
        return []
    from kgbundle import MentionRow

    # Normalize paper_id for filter (strip .xml if present)
    filter_doc_id = None
    if paper_id is not None:
        filter_doc_id = paper_id.removesuffix(".xml") if paper_id.endswith(".xml") else paper_id

    rows: list[dict] = []
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        try:
            row = MentionRow.model_validate_json(line)
            d = row.model_dump()
            if filter_doc_id is not None and d.get("document_id") != filter_doc_id:
                continue
            rows.append(d)
        except Exception:
            continue
    return rows


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
