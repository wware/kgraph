"""Canonical ID system for knowledge graph ingestion.

This package provides abstractions for working with canonical IDs (stable
identifiers from authoritative sources) throughout the ingestion pipeline.
"""

from .helpers import check_entity_id_format, extract_canonical_id_from_entity
from .json_cache import JsonFileCanonicalIdCache
from .lookup import CanonicalIdLookupInterface
from .models import CanonicalId, CanonicalIdCacheInterface

__all__ = [
    "CanonicalId",
    "CanonicalIdCacheInterface",
    "CanonicalIdLookupInterface",
    "JsonFileCanonicalIdCache",
    "check_entity_id_format",
    "extract_canonical_id_from_entity",
]
