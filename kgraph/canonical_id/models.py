"""Canonical ID cache interface for knowledge graph ingestion.

This module provides the CanonicalIdCacheInterface for caching canonical ID
lookups across different knowledge domains.
"""

from abc import ABC, abstractmethod
from typing import Optional

from kgschema.canonical_id import CanonicalId


class CanonicalIdCacheInterface(ABC):
    """Abstract interface for caching canonical ID lookups.

    This interface allows different storage backends (JSON file, database, etc.)
    to be used for caching canonical ID lookups. The cache stores mappings from
    (term, entity_type) pairs to CanonicalId objects.

    The cache supports:
    - Loading and saving with a tag/identifier (for multi-domain support)
    - Storing and fetching CanonicalId objects
    - Marking terms as "known bad" (failed lookups that shouldn't be retried)
    - Cache metrics (hits, misses, etc.)

    Implementations should be performant since the cache's purpose is to avoid
    expensive external API calls.
    """

    @abstractmethod
    def load(self, tag: str) -> None:
        """Load cache from storage.

        Args:
            tag: Identifier for the cache (e.g., "medlit", "sherlock", or a file path).
                 For file-based caches, this might be a file path.
                 For database caches, this might be a domain identifier.
        """
        pass

    @abstractmethod
    def save(self, tag: str) -> None:
        """Save cache to storage.

        Args:
            tag: Identifier for the cache (same as used in load()).
        """
        pass

    @abstractmethod
    def store(
        self,
        term: str,
        entity_type: str,
        canonical_id: CanonicalId,
    ) -> None:
        """Store a canonical ID in the cache.

        Args:
            term: The entity name/mention text (will be normalized internally)
            entity_type: Type of entity (e.g., "disease", "gene", "drug")
            canonical_id: The CanonicalId object to store
        """
        pass

    @abstractmethod
    def fetch(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        """Fetch a canonical ID from the cache.

        Args:
            term: The entity name/mention text (will be normalized internally)
            entity_type: Type of entity (e.g., "disease", "gene", "drug")

        Returns:
            CanonicalId if found in cache, None if not found or marked as "known bad"
        """
        pass

    @abstractmethod
    def mark_known_bad(self, term: str, entity_type: str) -> None:
        """Mark a term as "known bad" (failed lookup, don't retry).

        This allows the cache to remember that a lookup was attempted and failed,
        so we don't waste time retrying it.

        Args:
            term: The entity name/mention text (will be normalized internally)
            entity_type: Type of entity (e.g., "disease", "gene", "drug")
        """
        pass

    @abstractmethod
    def is_known_bad(self, term: str, entity_type: str) -> bool:
        """Check if a term is marked as "known bad".

        Args:
            term: The entity name/mention text (will be normalized internally)
            entity_type: Type of entity (e.g., "disease", "gene", "drug")

        Returns:
            True if the term is marked as "known bad", False otherwise
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, int]:
        """Get cache performance metrics.

        Returns:
            Dictionary with metrics such as:
            - "hits": Number of cache hits
            - "misses": Number of cache misses
            - "known_bad": Number of known bad entries
            - "total_entries": Total number of entries in cache
        """
        pass

    def _normalize_key(self, term: str, entity_type: str) -> str:
        """Normalize cache key for consistent lookups.

        Args:
            term: The entity name/mention text
            entity_type: Type of entity

        Returns:
            Normalized cache key string (e.g., "disease:breast cancer")
        """
        entity_type_normalized = entity_type.lower().strip()
        term_normalized = term.lower().strip()
        return f"{entity_type_normalized}:{term_normalized}"
