# Determinism and Reproducibility

This document addresses concerns about non-deterministic behavior in the knowledge graph ingestion pipeline, particularly regarding LLM usage and provenance tracking.

## Current State

### LLM Temperature Settings

- **Current temperature**: `0.1` (default in `OllamaLLMClient`)
- **Impact**: Even at low temperature, LLMs can exhibit non-deterministic behavior due to:
  - Floating-point precision differences
  - Model implementation details
  - Hardware differences (CPU vs GPU)
  - Model version changes

### Non-Deterministic Sources

1. **LLM Generation** (entity extraction, relationship extraction)
   - Temperature = 0.1 (not fully deterministic)
   - No seed parameter currently set
   - Model version not tracked in provenance

2. **Network Operations**
   - DBPedia SPARQL queries (transient failures/timeouts)
   - External API calls (MeSH, UMLS, HGNC, etc.)
   - Cache state differences between runs

3. **Async Execution Order**
   - Parallel entity/relationship extraction
   - Race conditions in concurrent lookups

4. **Cache State**
   - Previous failed lookups marked as "known bad"
   - Cache file state affects subsequent runs

## Recommendations for Full Determinism

### 1. Set Temperature to 0.0

```python
# In OllamaLLMClient
temperature: float = 0.0  # Fully deterministic (if supported by model)
```

**Note**: Some models may not support temperature=0.0. Verify with your model.

### 2. Add Seed Parameter

If Ollama supports it, add a seed parameter for reproducibility:

```python
options: dict[str, Any] = {
    "temperature": 0.0,
    "seed": 42,  # Fixed seed for reproducibility
}
```

### 3. Track LLM Parameters in Provenance

Add model parameters to provenance metadata:

```python
class ModelInfo(BaseModel):
    name: str
    provider: str
    temperature: float  # Track actual temperature used
    seed: Optional[int] = None  # Track seed if used
    version: Optional[str] = None  # Model version
```

### 4. Document Cache State

- Include cache file hash in provenance
- Or: Use deterministic cache keys that don't depend on execution order

### 5. Make Network Operations Deterministic

- Retry with exponential backoff (deterministic retry count)
- Timeout values should be consistent
- Don't mark transient failures as "known bad" (already fixed)

## Current Fixes Applied

1. ✅ Removed "mark_known_bad" for DBPedia extraction failures
2. ✅ Changed exception logging to WARNING level (visible without DEBUG)
3. ⚠️ Temperature still at 0.1 (should be 0.0 for full determinism)
4. ⚠️ No seed parameter set
5. ⚠️ LLM parameters not tracked in provenance

## Testing Reproducibility

To verify reproducibility:

1. Run ingestion twice with identical inputs
2. Compare entity IDs and canonical IDs
3. Compare relationship structures
4. Check for any differences in promoted entities

If differences are found, investigate:
- LLM temperature/seed settings
- Cache state differences
- Network timing issues
- Async execution order

## Future Work

- [ ] Set temperature to 0.0 (or minimum supported)
- [ ] Add seed parameter support
- [ ] Track LLM parameters in provenance
- [ ] Add reproducibility tests
- [ ] Document model version in provenance
- [ ] Consider deterministic async execution order
