"""
GraphQL schema for the Knowledge Graph API.

This schema uses proper Strawberry types for type safety and better GraphQL introspection.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional
import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

logger = logging.getLogger(__name__)

# Max limit for pagination (configurable via environment variable)
MAX_LIMIT = int(os.getenv("GRAPHQL_MAX_LIMIT", "100"))


@strawberry.type
class Entity:
    """Generic entity GraphQL type."""

    id: strawberry.ID = strawberry.field(name="id")
    entity_id: strawberry.ID = strawberry.field(name="entityId")
    entity_type: str = strawberry.field(name="entityType")
    name: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    usage_count: Optional[int] = strawberry.field(name="usageCount", default=None)
    source: Optional[str] = None
    canonical_url: Optional[str] = strawberry.field(name="canonicalUrl", default=None)
    synonyms: List[str] = strawberry.field(default_factory=list)
    properties: Optional[JSON] = None


@strawberry.type
class Relationship:
    """Generic relationship GraphQL type."""

    id: strawberry.ID = strawberry.field(name="id")
    subject_id: strawberry.ID = strawberry.field(name="subjectId")
    predicate: str
    object_id: strawberry.ID = strawberry.field(name="objectId")
    confidence: Optional[float] = None
    source_documents: List[str] = strawberry.field(name="sourceDocuments", default_factory=list)
    properties: Optional[JSON] = None


@strawberry.type
class EntityPage:
    """Paginated result for entities."""

    items: List[Entity]
    total: int
    limit: int
    offset: int


@strawberry.type
class RelationshipPage:
    """Paginated result for relationships."""

    items: List[Relationship]
    total: int
    limit: int
    offset: int


@strawberry.input
class EntityFilter:
    """Filter criteria for entity queries."""

    entity_type: Optional[str] = None
    name: Optional[str] = None  # exact match
    name_contains: Optional[str] = None  # ILIKE %...% (PostgreSQL) or LIKE %...% (SQLite)
    source: Optional[str] = None
    status: Optional[str] = None


@strawberry.input
class RelationshipFilter:
    """Filter criteria for relationship queries."""

    subject_id: Optional[strawberry.ID] = None
    object_id: Optional[strawberry.ID] = None
    predicate: Optional[str] = None


@strawberry.type
class BundleInfo:
    """Bundle metadata for debugging and provenance."""

    id: strawberry.ID = strawberry.field(name="id")
    bundle_id: str = strawberry.field(name="bundleId")
    domain: str
    created_at: Optional[datetime] = strawberry.field(name="createdAt", default=None)
    metadata: Optional[JSON] = None


@strawberry.type
class Query:
    @strawberry.field
    def entity(self, info: Info, id: str) -> Optional[Entity]:  # pylint: disable=redefined-builtin
        """Retrieve a single entity by its ID."""
        storage = info.context["storage"]
        entity_model = storage.get_entity(entity_id=id)
        if entity_model:
            return Entity(
                id=entity_model.entity_id,
                entity_id=entity_model.entity_id,
                entity_type=entity_model.entity_type,
                name=entity_model.name,
                status=entity_model.status,
                confidence=entity_model.confidence,
                usage_count=entity_model.usage_count,
                source=entity_model.source,
                canonical_url=entity_model.canonical_url,
                synonyms=entity_model.synonyms,
                properties=entity_model.properties,
            )
        return None

    @strawberry.field
    def entities(
        self,
        info: Info,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[EntityFilter] = None,  # pylint: disable=redefined-builtin
    ) -> EntityPage:
        """List entities with pagination and optional filtering."""
        # Enforce max limit
        if limit > MAX_LIMIT:
            logger.warning("Requested limit %d exceeds MAX_LIMIT %d, capping to %d", limit, MAX_LIMIT, MAX_LIMIT)
            limit = MAX_LIMIT

        storage = info.context["storage"]

        # Extract filter parameters
        filter_obj = filter  # Use local variable to avoid shadowing built-in
        entity_type = filter_obj.entity_type if filter_obj else None
        name = filter_obj.name if filter_obj else None
        name_contains = filter_obj.name_contains if filter_obj else None
        source = filter_obj.source if filter_obj else None
        status = filter_obj.status if filter_obj else None

        # Get total count
        total = storage.count_entities(
            entity_type=entity_type,
            name=name,
            name_contains=name_contains,
            source=source,
            status=status,
        )

        # Get paginated results
        entity_models = storage.get_entities(
            limit=limit,
            offset=offset,
            entity_type=entity_type,
            name=name,
            name_contains=name_contains,
            source=source,
            status=status,
        )

        items = [
            Entity(
                id=e.entity_id,
                entity_id=e.entity_id,
                entity_type=e.entity_type,
                name=e.name,
                status=e.status,
                confidence=e.confidence,
                usage_count=e.usage_count,
                source=e.source,
                canonical_url=e.canonical_url,
                synonyms=e.synonyms,
                properties=e.properties,
            )
            for e in entity_models
        ]

        return EntityPage(items=items, total=total, limit=limit, offset=offset)

    @strawberry.field
    def relationship(
        self,
        info: Info,
        subject_id: strawberry.ID,
        predicate: str,
        object_id: strawberry.ID,
    ) -> Optional[Relationship]:
        """Retrieve a single relationship by its triple."""
        storage = info.context["storage"]
        relationship_model = storage.get_relationship(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
        )
        if relationship_model:
            return Relationship(
                id=str(relationship_model.id),
                subject_id=relationship_model.subject_id,
                predicate=relationship_model.predicate,
                object_id=relationship_model.object_id,
                confidence=relationship_model.confidence,
                source_documents=relationship_model.source_documents,
                properties=relationship_model.properties,
            )
        return None

    @strawberry.field
    def relationships(
        self,
        info: Info,
        limit: int = 100,
        offset: int = 0,
        filter: Optional[RelationshipFilter] = None,  # pylint: disable=redefined-builtin
    ) -> RelationshipPage:
        """Find relationships with pagination and optional filtering."""
        # Enforce max limit
        if limit > MAX_LIMIT:
            logger.warning("Requested limit %d exceeds MAX_LIMIT %d, capping to %d", limit, MAX_LIMIT, MAX_LIMIT)
            limit = MAX_LIMIT

        storage = info.context["storage"]

        # Extract filter parameters
        filter_obj = filter  # Use local variable to avoid shadowing built-in
        subject_id = filter_obj.subject_id if filter_obj else None
        object_id = filter_obj.object_id if filter_obj else None
        predicate = filter_obj.predicate if filter_obj else None

        # Get total count
        total = storage.count_relationships(
            subject_id=subject_id,
            object_id=object_id,
            predicate=predicate,
        )

        # Get paginated results
        relationship_models = storage.find_relationships(
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            limit=limit,
            offset=offset,
        )

        items = [
            Relationship(
                id=str(r.id),
                subject_id=r.subject_id,
                predicate=r.predicate,
                object_id=r.object_id,
                confidence=r.confidence,
                source_documents=r.source_documents,
                properties=r.properties,
            )
            for r in relationship_models
        ]

        return RelationshipPage(items=items, total=total, limit=limit, offset=offset)

    @strawberry.field
    def bundle(self, info: Info) -> Optional[BundleInfo]:
        """Get bundle metadata for debugging and provenance."""
        storage = info.context["storage"]
        bundle_model = storage.get_bundle_info()

        if bundle_model:
            # Try to get metadata from bundle manifest if available
            # For now, return None for metadata as it's not stored in the Bundle model
            return BundleInfo(
                id=bundle_model.bundle_id,
                bundle_id=bundle_model.bundle_id,
                domain=bundle_model.domain,
                created_at=bundle_model.created_at,
                metadata=None,
            )
        return None
