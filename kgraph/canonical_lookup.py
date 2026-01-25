"""Canonical ID lookup interface for promotion policies.

This module provides an abstract interface for looking up canonical IDs,
which promotion policies can use to assign canonical IDs to entities.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .canonical_id import CanonicalId


class CanonicalIdLookupInterface(ABC):
    """Abstract interface for looking up canonical IDs.

    This interface is used by promotion policies to look up canonical IDs
    for entities. It abstracts away the details of how lookups are performed
    (API calls, cache, etc.) so promotion policies can work with any lookup
    implementation.
    """

    @abstractmethod
    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        """Look up a canonical ID for a term.

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (e.g., "disease", "gene", "drug")

        Returns:
            CanonicalId if found, None otherwise
        """
        pass

    @abstractmethod
    def lookup_sync(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        """Synchronous version of lookup (for use in sync contexts).

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (e.g., "disease", "gene", "drug")

        Returns:
            CanonicalId if found, None otherwise
        """
        pass
