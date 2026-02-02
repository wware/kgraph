"""Canonical ID system for knowledge graph ingestion.

This package provides abstractions for working with canonical IDs (stable
identifiers from authoritative sources) throughout the ingestion pipeline.

Note: The CanonicalId model has been moved to kgschema.canonical_id but is
re-exported here for backwards compatibility.
"""

from kgschema.canonical_id import CanonicalId

from .helpers import check_entity_id_format, extract_canonical_id_from_entity
from .json_cache import JsonFileCanonicalIdCache
from .lookup import CanonicalIdLookupInterface
from .models import CanonicalIdCacheInterface

__all__ = [
    "CanonicalId",
    "CanonicalIdCacheInterface",
    "CanonicalIdLookupInterface",
    "JsonFileCanonicalIdCache",
    "check_entity_id_format",
    "extract_canonical_id_from_entity",
]
