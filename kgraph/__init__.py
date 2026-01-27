"""
Knowledge Graph Framework - Domain-Agnostic Entity and Relationship Extraction.

A flexible framework for building knowledge graphs across multiple domains
(medical, legal, CS papers, etc.) with a two-pass ingestion process.

This module uses lazy imports to avoid loading heavy dependencies (numpy, sklearn)
when only lightweight components are needed. For example:

    # This does NOT import numpy:
    from kgraph.query.bundle import BundleManifestV1, EntityRow, RelationshipRow

    # This DOES import numpy (when the symbol is accessed):
    from kgraph import IngestionOrchestrator
"""

from typing import TYPE_CHECKING

# Lightweight imports that don't pull in numpy/sklearn
from kgraph.canonical_id import (
    CanonicalId,
    CanonicalIdCacheInterface,
    CanonicalIdLookupInterface,
    JsonFileCanonicalIdCache,
    check_entity_id_format,
    extract_canonical_id_from_entity,
)
from kgraph.entity import (
    BaseEntity,
    EntityMention,
    EntityStatus,
    PromotionConfig,
)
from kgraph.relationship import BaseRelationship
from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema

# Type checking imports for IDE support (not executed at runtime)
if TYPE_CHECKING:
    from kgraph.ingest import IngestionOrchestrator, IngestionResult

__all__ = [
    "BaseEntity",
    "EntityMention",
    "EntityStatus",
    "PromotionConfig",
    "BaseRelationship",
    "BaseDocument",
    "CanonicalId",
    "CanonicalIdCacheInterface",
    "CanonicalIdLookupInterface",
    "check_entity_id_format",
    "extract_canonical_id_from_entity",
    "JsonFileCanonicalIdCache",
    "DomainSchema",
    "IngestionOrchestrator",
    "IngestionResult",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid loading numpy/sklearn on light imports."""
    if name in ("IngestionOrchestrator", "IngestionResult"):
        from kgraph.ingest import IngestionOrchestrator, IngestionResult
        return {"IngestionOrchestrator": IngestionOrchestrator, "IngestionResult": IngestionResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
