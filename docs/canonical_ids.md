# Canonical IDs

Canonical IDs are stable identifiers from authoritative sources (UMLS, MeSH, HGNC, RxNorm, UniProt, DBPedia, etc.) that uniquely identify entities across different knowledge bases. The `kgraph` framework provides abstractions for working with canonical IDs throughout the ingestion pipeline.

## Overview

The canonical ID system consists of:

1. **`CanonicalId`** - A Pydantic model representing a canonical identifier with ID, URL, and synonyms
2. **`CanonicalIdCacheInterface`** - Abstract interface for caching canonical ID lookups
3. **`CanonicalIdLookupInterface`** - Abstract interface for looking up canonical IDs
4. **Helper functions** - Utilities for extracting canonical IDs from entities

These abstractions enable:
- **Consistent handling** of canonical IDs across different domains
- **Flexible caching** strategies (JSON file, database, etc.)
- **Reusable promotion logic** that works with any lookup implementation
- **URL generation** for canonical entities

## CanonicalId Model

The `CanonicalId` model represents a canonical identifier with its metadata:

```python
from kgraph import CanonicalId

# Create a canonical ID with all fields
cid = CanonicalId(
    id="MeSH:D000570",
    url="https://meshb.nlm.nih.gov/record/ui?ui=D000570",
    synonyms=("breast cancer", "breast neoplasms")
)

# Minimal canonical ID (just the ID string)
cid = CanonicalId(id="UMLS:C12345")

# CanonicalId is immutable (frozen)
assert str(cid) == "UMLS:C12345"
```

### Fields

- **`id`** (str): The canonical ID string (e.g., "UMLS:C12345", "MeSH:D000570", "HGNC:1100")
- **`url`** (Optional[str]): URL to the authoritative source page for this ID
- **`synonyms`** (tuple[str, ...]): Alternative names/terms that map to this canonical ID

## Canonical ID Cache

The `CanonicalIdCacheInterface` provides an abstract interface for caching canonical ID lookups, allowing different storage backends to be used.

### JsonFileCanonicalIdCache

A JSON file-based implementation suitable for development and small-to-medium datasets:

```python
from kgraph import JsonFileCanonicalIdCache, CanonicalId
from pathlib import Path

# Create cache
cache = JsonFileCanonicalIdCache(cache_file=Path("canonical_id_cache.json"))

# Load existing cache
cache.load("canonical_id_cache.json")

# Store a canonical ID
cid = CanonicalId(id="MeSH:D000570", url="https://example.com")
cache.store("breast cancer", "disease", cid)

# Fetch from cache
fetched = cache.fetch("breast cancer", "disease")
assert fetched.id == "MeSH:D000570"

# Mark as known bad (failed lookup)
cache.mark_known_bad("nonexistent term", "disease")
assert cache.is_known_bad("nonexistent term", "disease") is True

# Get cache metrics
metrics = cache.get_metrics()
print(f"Hits: {metrics['hits']}, Misses: {metrics['misses']}")

# Save cache
cache.save("canonical_id_cache.json")
```

### Cache Features

- **Automatic key normalization**: Case-insensitive, whitespace-stripped lookups
- **Known bad tracking**: Remembers failed lookups to avoid retrying
- **Backward compatibility**: Automatically migrates old cache format (dict[str, str])
- **Metrics**: Tracks hits, misses, and cache size

## Canonical ID Lookup Interface

The `CanonicalIdLookupInterface` provides an abstract interface for looking up canonical IDs, which promotion policies use:

```python
from kgraph import CanonicalIdLookupInterface

class MyLookupService(CanonicalIdLookupInterface):
    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        # Your lookup logic here
        canonical_id_str = await self._query_authority(term, entity_type)
        if canonical_id_str:
            url = self._build_url(canonical_id_str)
            return CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
        return None

    def lookup_sync(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        # Synchronous version for use in sync contexts
        # ...
```

## Helper Functions

The `canonical_helpers` module provides reusable functions for extracting canonical IDs from entities:

### extract_canonical_id_from_entity

Extracts a canonical ID from an entity's `canonical_ids` dictionary:

```python
from kgraph import extract_canonical_id_from_entity

entity = MyEntity(
    name="Breast Cancer",
    canonical_ids={
        "umls": "C0006142",
        "mesh": "MeSH:D000570",
        "dbpedia": "DBPedia:Breast_cancer"
    }
)

# Extract with priority order
cid = extract_canonical_id_from_entity(
    entity,
    priority_sources=["umls", "mesh", "dbpedia"]
)
assert cid.id == "C0006142"  # Returns UMLS (first in priority)

# Extract without priority (returns first available)
cid = extract_canonical_id_from_entity(entity)
assert cid.id in ("C0006142", "MeSH:D000570", "DBPedia:Breast_cancer")
```

### check_entity_id_format

Checks if an entity's `entity_id` matches a known canonical ID format:

```python
from kgraph import check_entity_id_format

entity = MyEntity(entity_id="HGNC:1100", entity_type="gene")

# Check if entity_id matches HGNC format
format_patterns = {"gene": ("HGNC:",)}
cid = check_entity_id_format(entity, format_patterns)
assert cid.id == "HGNC:1100"

# Supports multiple patterns
format_patterns = {
    "gene": ("HGNC:", "numeric"),  # "HGNC:1100" or just "1100"
    "disease": ("C",),  # UMLS: "C0006142"
    "protein": ("UniProt:", "uniprot"),  # "UniProt:P38398" or "P38398"
}
```

## Using in Promotion Policies

Promotion policies use these abstractions to assign canonical IDs:

```python
from kgraph import (
    PromotionPolicy,
    CanonicalId,
    extract_canonical_id_from_entity,
    check_entity_id_format,
)
from kgraph.canonical_id import CanonicalIdLookupInterface

class MyPromotionPolicy(PromotionPolicy):
    def __init__(self, config, lookup: CanonicalIdLookupInterface):
        super().__init__(config)
        self.lookup = lookup

    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        # Strategy 1: Check entity.canonical_ids
        cid = extract_canonical_id_from_entity(entity, priority_sources=["umls", "hgnc"])
        if cid:
            return cid

        # Strategy 2: Check entity_id format
        format_patterns = {"gene": ("HGNC:",), "disease": ("C",)}
        cid = check_entity_id_format(entity, format_patterns)
        if cid:
            return cid

        # Strategy 3: External lookup
        return await self.lookup.lookup(entity.name, entity.get_entity_type())
```

## Implementation Example: Medical Literature

The medical literature example (`examples/medlit`) demonstrates a complete implementation:

```python
from kgraph.canonical_id import JsonFileCanonicalIdCache
from kgraph.canonical_id import CanonicalIdLookupInterface

class CanonicalIdLookup(CanonicalIdLookupInterface):
    def __init__(self, cache_file: Path):
        self._cache = JsonFileCanonicalIdCache(cache_file=cache_file)
        self._cache.load(str(cache_file))

    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        # Check cache first
        cached = self._cache.fetch(term, entity_type)
        if cached:
            return cached

        # Query authority APIs (UMLS, HGNC, RxNorm, UniProt, DBPedia)
        canonical_id_str = await self._query_authority(term, entity_type)
        if canonical_id_str:
            url = build_canonical_url(canonical_id_str, entity_type)
            cid = CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
            self._cache.store(term, entity_type, cid)
            return cid
        else:
            self._cache.mark_known_bad(term, entity_type)
            return None
```

## Best Practices

1. **Use the abstractions**: Implement `CanonicalIdLookupInterface` rather than creating domain-specific lookup classes
2. **Cache aggressively**: Canonical ID lookups are expensive (API calls), so cache everything
3. **Handle known bads**: Mark failed lookups as "known bad" to avoid retrying
4. **Use helper functions**: Leverage `extract_canonical_id_from_entity` and `check_entity_id_format` in promotion policies
5. **Include URLs**: When possible, include URLs in `CanonicalId` objects for better user experience
6. **Normalize keys**: The cache interface normalizes keys automatically, but be consistent in your code

## Migration from Old Format

The `JsonFileCanonicalIdCache` automatically migrates old cache files (format: `dict[str, str]`) to the new format (format: `dict[str, dict]` with `CanonicalId` data). Migration happens automatically on load.
