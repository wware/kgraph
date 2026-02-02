"""Helper functions for working with canonical IDs in promotion logic.

This module provides generic helper functions that can be used by promotion
policies to extract canonical IDs from entity data.
"""

from typing import Optional

from kgschema.canonical_id import CanonicalId
from kgschema.entity import BaseEntity


def extract_canonical_id_from_entity(
    entity: BaseEntity,
    priority_sources: Optional[list[str]] = None,
) -> Optional[CanonicalId]:
    """Extract canonical ID from entity's canonical_ids dict.

    Args:
        entity: The entity to extract canonical ID from
        priority_sources: Optional list of source keys to check in priority order.
                         If None, checks all sources in the dict.

    Returns:
        CanonicalId if found, None otherwise
    """
    if not entity.canonical_ids:
        return None

    if priority_sources:
        # Check sources in priority order
        for source in priority_sources:
            if source in entity.canonical_ids:
                canonical_id_str = entity.canonical_ids[source]
                return CanonicalId(id=canonical_id_str, url=None, synonyms=())
    else:
        # Return first available canonical ID
        for source, canonical_id_str in entity.canonical_ids.items():
            return CanonicalId(id=canonical_id_str, url=None, synonyms=())

    return None


def check_entity_id_format(entity: BaseEntity, format_patterns: dict[str, tuple[str, ...]]) -> Optional[CanonicalId]:
    """Check if entity_id matches any known canonical ID format.

    Args:
        entity: The entity to check
        format_patterns: Dict mapping entity types to tuples of format prefixes/patterns.
                        For example: {"gene": ("HGNC:",), "disease": ("C",)}

    Returns:
        CanonicalId if entity_id matches a format, None otherwise
    """
    entity_id = entity.entity_id
    entity_type = entity.get_entity_type()

    if entity_type not in format_patterns:
        return None

    patterns = format_patterns[entity_type]

    for pattern in patterns:
        if entity_id.startswith(pattern):
            return CanonicalId(id=entity_id, url=None, synonyms=())
        # Handle patterns like "C" (UMLS) where we check if it starts with C and is followed by digits
        if pattern == "C" and len(entity_id) > 1 and entity_id.startswith("C") and entity_id[1:].isdigit():
            return CanonicalId(id=entity_id, url=None, synonyms=())
        # Handle numeric patterns (HGNC, RxNorm)
        if pattern == "numeric" and entity_id.isdigit():
            # Format depends on entity type
            if entity_type == "gene":
                return CanonicalId(id=f"HGNC:{entity_id}", url=None, synonyms=())
            elif entity_type == "drug":
                return CanonicalId(id=f"RxNorm:{entity_id}", url=None, synonyms=())
        # Handle UniProt pattern (P/Q + 5-6 alphanumeric)
        if pattern == "uniprot" and (entity_id.startswith("P") or entity_id.startswith("Q")):
            if len(entity_id) >= 6 and entity_id[1:].isalnum():
                return CanonicalId(id=f"UniProt:{entity_id}", url=None, synonyms=())

    return None
