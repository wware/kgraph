# Canonical ID Lookup for Medical Entities

This document describes how the medlit pipeline acquires canonical IDs from authoritative medical ontology sources for entities like diseases, genes, drugs, and proteins.

## Overview

Medical knowledge graphs require standardized identifiers to link entities across different papers and databases. The medlit pipeline uses a two-pronged approach:

1. **During Extraction**: The LLM can call a lookup tool to find canonical IDs while extracting entities from text
2. **During Promotion**: Provisional entities without canonical IDs can be enriched via API lookup before promotion

### Two-pass vs legacy

The **two-pass pipeline** (Pass 1 → Pass 2) is the canonical flow. It assigns authoritative IDs from the bundle and optionally via `CanonicalIdLookup` in Pass 2; it does not use usage/confidence promotion. The legacy promotion path is no longer available via a supported CLI.

### Pass 2 (two-pass pipeline)

Pass 2 can optionally use the same `CanonicalIdLookup` and cache file. Use `--canonical-id-cache path/to/cache.json` to enable authority lookups for entities that do not already have an authoritative ID from the bundle or synonym cache. Pass 2 output uses **entity_id** as the stable merge key (always present) and **canonical_id** only when the entity has an authoritative ontology ID; otherwise **canonical_id** is null. For runs without network or API access, use `--no-canonical-id-lookup` to disable lookups even when a cache path is set.

## Supported Ontologies

| Entity Type | Ontology | ID Format | Example |
|-------------|----------|-----------|---------|
| Disease | UMLS or MeSH* | C + digits or `MeSH:D` + digits | `C0006142` or `MeSH:D001943` |
| Gene | HGNC (HUGO Gene Nomenclature Committee) | `HGNC:` + number | `HGNC:1100` (BRCA1) |
| Drug | RxNorm | `RxNorm:` + number | `RxNorm:161` (aspirin) |
| Protein | UniProt | P/Q + 5 alphanumeric | `P38398` (BRCA1 protein) |

*MeSH (Medical Subject Headings) is used as a fallback for diseases when `UMLS_API_KEY` is not set. MeSH is freely accessible without authentication.

## Architecture

### CanonicalIdLookup Class

The `CanonicalIdLookup` class in `pipeline/authority_lookup.py` handles all external API calls:

```python
from examples.medlit.pipeline import CanonicalIdLookup

async with CanonicalIdLookup(cache_file=Path("cache.json")) as lookup:
    # Async lookup
    canonical_id = await lookup.lookup_canonical_id("BRCA1", "gene")
    # Returns: "HGNC:1100"
```

### API Endpoints Used

- **UMLS**: `https://uts-ws.nlm.nih.gov/rest/search/current` (requires API key)
- **MeSH**: `https://id.nlm.nih.gov/mesh/lookup/descriptor` (no auth, fallback for diseases)
- **HGNC**: `https://rest.genenames.org/search/symbol/{term}`
- **RxNorm**: `https://rxnav.nlm.nih.gov/REST/rxcui.json?name={term}`
- **UniProt**: `https://rest.uniprot.org/uniprotkb/search?query=protein_name:{term}`

### Persistent Caching

Lookup results are cached to a JSON file to avoid redundant API calls:

- Cache persists across program runs
- Only **successful** lookups are saved to disk
- Failed lookups (NULLs) are cached in memory only for the current run
- This allows new lookup improvements to benefit all terms on subsequent runs
- Cache is automatically saved when the lookup service is closed

### LLM Tool Calling

When using Ollama for entity extraction, the LLM can call `lookup_canonical_id` as a tool:

```python
# The LLM receives this tool definition and can call it during extraction
def lookup_canonical_id(term: str, entity_type: str) -> str | None:
    """Look up canonical ID for a medical entity.
    
    Args:
        term: The entity name (e.g., "BRCA1", "breast cancer")
        entity_type: Type of entity - one of: disease, gene, drug, protein
    
    Returns:
        Canonical ID if found, or null if not found
    """
```

This allows entities to be enriched with canonical IDs immediately during extraction, rather than waiting for the promotion phase.

## Configuration

### Environment Variables

- `UMLS_API_KEY`: Required for UMLS lookups (get from [UTS](https://uts.nlm.nih.gov/uts/signup-login))

### CLI Options

```bash
# Pass 1 and Pass 2 with canonical ID cache (Pass 2 only uses the cache)
uv run python -m examples.medlit.scripts.pass1_extract --input-dir ./papers --output-dir pass1_bundles --llm-backend anthropic
uv run python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles --output-dir ./merged --synonym-cache ./merged/synonym_cache.json --canonical-id-cache canonical_id_cache.json
```

## Lookup Strategy

The `MedLitPromotionPolicy.assign_canonical_id()` method uses a multi-strategy approach:

1. **Check existing canonical_ids**: If the entity already has a canonical ID from a known source (umls, hgnc, rxnorm, uniprot), use it

2. **Check entity_id format**: If the entity_id matches a canonical format (e.g., starts with "C" followed by digits for UMLS), use it directly

3. **External API lookup**: Query the appropriate ontology API based on entity type

4. **Return None**: If no canonical ID can be found, the entity remains provisional

## Example Flow

```
Document: "BRCA1 mutations are associated with breast cancer..."

1. LLM extracts entity: {name: "BRCA1", type: "gene"}
2. LLM calls tool: lookup_canonical_id("BRCA1", "gene")
3. Tool queries HGNC API → returns "HGNC:1100"
4. Entity created with canonical_id_hint in metadata
5. Resolver sees hint, creates canonical entity directly

Alternative (without tool calling):
1. LLM extracts entity: {name: "BRCA1", type: "gene"}
2. Entity created as provisional (no canonical ID)
3. Promotion phase: policy calls lookup.lookup_canonical_id()
4. HGNC API returns "HGNC:1100"
5. Entity promoted to canonical with assigned ID
```

## Error Handling

- API failures are logged but don't halt processing
- Network timeouts default to 10 seconds per request
- Failed lookups are cached in memory only (not persisted) to avoid repeated API calls within a run
- The sync wrapper (`lookup_canonical_id_sync`) has a 15-second timeout for tool calls

## Troubleshooting

### Entities not getting canonical IDs when they should

If common terms like "breast cancer" or "BRCA1" are not getting canonical IDs, check:

1. **Network connectivity** - Can you reach the ontology APIs?
2. **Entity type mismatch** - Is the LLM extracting with the wrong type? (e.g., "Mitochondria" as "protein")
3. **Old cache file** - If you have a cache file from before recent improvements, delete it:

```bash
rm canonical_id_cache.json
```

Note: Since failed lookups are no longer persisted to disk, stale cache entries are much less of an issue than before.

### Verifying lookups are working

You can test individual lookups manually:

```bash
# Test MeSH (diseases)
curl "https://id.nlm.nih.gov/mesh/lookup/descriptor?label=breast+neoplasms&match=contains&limit=1"

# Test HGNC (genes)
curl "https://rest.genenames.org/fetch/symbol/BRCA1" -H "Accept: application/json"

# Test RxNorm (drugs)
curl "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin"
```

## Files

- `pipeline/authority_lookup.py`: CanonicalIdLookup class with API integrations
- `pipeline/llm_client.py`: LLM client with tool calling support
- `pipeline/mentions.py`: Entity extractor with optional tool calling
- `promotion.py`: Promotion policy with external lookup integration
- `scripts/pass1_extract.py`, `pass2_dedup.py`, `pass3_build_bundle.py`: canonical three-pass CLI (see INGESTION.md and run-ingest.sh)
