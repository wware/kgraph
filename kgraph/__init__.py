"""
Knowledge Graph Framework - Domain-Agnostic Entity and Relationship Extraction.

A flexible framework for building knowledge graphs across multiple domains
(medical, legal, CS papers, etc.) with a two-pass ingestion process.
"""

# Re-export core types from kgschema
from kgschema.canonical_id import CanonicalId
from kgschema.document import BaseDocument
from kgschema.domain import DomainSchema
from kgschema.entity import BaseEntity, EntityMention, EntityStatus, PromotionConfig
from kgschema.promotion import PromotionPolicy
from kgschema.relationship import BaseRelationship

# Canonical ID utilities
from kgraph.canonical_id import (
    CanonicalIdCacheInterface,
    CanonicalIdLookupInterface,
    JsonFileCanonicalIdCache,
    check_entity_id_format,
    extract_canonical_id_from_entity,
)

# Ingestion framework
from kgraph.ingest import IngestionOrchestrator, IngestionResult

__all__ = [
    "BaseDocument",
    "BaseEntity",
    "BaseRelationship",
    "CanonicalId",
    "CanonicalIdCacheInterface",
    "CanonicalIdLookupInterface",
    "DomainSchema",
    "EntityMention",
    "EntityStatus",
    "IngestionOrchestrator",
    "IngestionResult",
    "JsonFileCanonicalIdCache",
    "PromotionConfig",
    "PromotionPolicy",
    "check_entity_id_format",
    "extract_canonical_id_from_entity",
]

__version__ = "0.1.0"
