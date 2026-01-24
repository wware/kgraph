# Canonical ID Lookup for Medical Entities

This document describes how the medlit pipeline acquires canonical IDs from authoritative medical ontology sources for entities like diseases, genes, drugs, and proteins.

## Overview

Medical knowledge graphs require standardized identifiers to link entities across different papers and databases. The medlit pipeline uses a two-pronged approach:

1. **During Extraction**: The LLM can call a lookup tool to find canonical IDs while extracting entities from text
2. **During Promotion**: Provisional entities without canonical IDs can be enriched via API lookup before promotion

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
- Both positive results (canonical IDs) and negative results (not found) are cached
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
python -m examples.medlit.scripts.ingest \
    --input-dir ./papers \
    --output-dir ./bundle \
    --use-ollama \
    --cache-file canonical_id_cache.json
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
- Failed lookups cache `None` (as `"NULL"` in JSON) to avoid repeated failed requests
- The sync wrapper (`lookup_canonical_id_sync`) has a 15-second timeout for tool calls

## Troubleshooting

### Entities not getting canonical IDs when they should

If common terms like "breast cancer" or "BRCA1" are not getting canonical IDs (showing "No canonical_id_hint provided"), the most likely cause is **stale cache entries**.

The cache stores both successful lookups and failures (`"NULL"`). If the lookup logic has been updated (e.g., MeSH fallback added), old failure entries may be incorrect.

**Solution**: Delete the cache file and re-run:

```bash
rm canonical_id_cache.json
# Or whatever path you specified with --cache-file
```

The cache will be rebuilt with fresh lookups.

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
- `scripts/ingest.py`: CLI wiring and lifecycle management
