# Implementation Plan: Semantic Evidence Validation for Relationship Extraction

**Created:** 2026-02-17  
**Status:** Approved; updated per Claude review (caching, integration detail, CLI, trace fields)  
**Goal:** Replace or augment strict string-based evidence validation with cosine-similarity (embedding-based) checks so that relationships are not rejected when the evidence uses abbreviations, related terms, or partial matches (e.g. "MBC" vs "Male breast cancer", "breast cancer" vs "Male breast cancer (MBC)").

---

## 1. Problem statement

During relationship extraction, the pipeline rejects many otherwise-valid relationships because **evidence validation** requires the subject and object entity names (or synonyms) to appear **verbatim** in the evidence text. Examples of rejections that we want to accept:

| Evidence snippet | Entity name | Current result | Desired result |
|------------------|-------------|----------------|----------------|
| "...increased prevalence in **MBC**, such as BRCA2..." | Male breast cancer | Rejected (evidence_missing_object) | Accept (MBC ≈ Male breast cancer) |
| "...metastatic **pleural effusions** is **breast cancer**." | Male breast cancer (MBC), pleural effusion | Rejected (evidence_missing_subject) | Accept (breast cancer ≈ Male breast cancer; effusions ≈ effusion) |
| "...cytological **pleural metastasis** from ductal breast carcinoma..." | pleural effusion | Rejected (evidence_missing_object) | Accept or tune (pleural metastasis related to pleural effusion) |

**Current behavior:** All 23 relationships in a sample run were dropped due to `evidence_missing_subject` or `evidence_missing_object`; the pipeline produced 0 relationships for that paper.

**Target behavior:** Accept relationships when the evidence **semantically** supports the subject and object (e.g. cosine similarity between evidence embedding and entity embedding above a configurable threshold), while still rejecting clearly irrelevant evidence.

---

## 2. Current implementation (reference)

- **Evidence check:** `examples/medlit/pipeline/relationships.py`
  - `_evidence_contains_both_entities(evidence, subject_name, object_name, subject_entity, object_entity)` (sync).
  - Uses normalized substring matching: entity name and `entity.synonyms` must appear literally in normalized evidence text.
  - Returns `(ok: bool, drop_reason: str | None, detail: dict)` with `detail["subject_in_evidence"]`, `detail["object_in_evidence"]`.
- **Call site:** `MedLitRelationshipExtractor._process_llm_item()` (sync) calls the above and drops the relationship if `evidence_ok` is False.
- **Upstream:** `_extract_with_llm()` (async) builds an entity index, calls the LLM, then iterates over items and calls `_process_llm_item()` for each (sync). So evidence validation today is synchronous and has no access to an embedding generator.

**Existing assets we can reuse:**

- **Cosine similarity:** `kgraph/storage/memory.py` defines `_cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float` (pure Python). Alternatively `kgraph/ingest.py` uses `sklearn.metrics.pairwise.cosine_similarity` for entity-merge matrices.
- **Entity embeddings:** By relationship-extraction time, entities come from storage and typically already have `.embedding` set (from entity resolution). So we can compare evidence to subject/object **without** generating new embeddings for entity names in most cases.
- **Embedding generator:** The ingest pipeline already has an `embedding_generator` (e.g. `OllamaMedLitEmbeddingGenerator`); it is not currently passed into `MedLitRelationshipExtractor`.

---

## 3. Proposed design

### 3.1 Options considered

- **A. String-only relaxation:** Add fuzzy/alias/abbreviation rules (e.g. strip parentheticals, word-overlap, known abbreviations). No embeddings. Simpler but brittle and hard to maintain.
- **B. Semantic-only:** Replace string check with an embedding-based check. Clean but adds an embedding API call per candidate relationship (evidence text).
- **C. Hybrid (recommended):** Keep string check first; if it fails, run a semantic check using embeddings. Fast when strings match; semantic fallback when they don’t.

**Recommendation:** Implement **C (hybrid)** with a configuration option to use **semantic-only** or **string-only** (for environments without an embedding generator or for debugging).

### 3.2 Components

1. **Semantic evidence validator**
   - Inputs: evidence string, subject entity, object entity, optional embedding generator, threshold.
   - If entities have `.embedding`, use them; otherwise (or if generator provided) generate embedding for subject/object name.
   - Generate one embedding for the evidence text.
   - Compute `cosine_similarity(evidence_embedding, subject_embedding)` and same for object.
   - Return valid if both similarities ≥ threshold; plus detail dict (e.g. `subject_similarity`, `object_similarity`, `subject_in_evidence`, `object_in_evidence`) for tracing.

2. **Integration point**
   - Evidence validation runs inside the **async** path (`_extract_with_llm`). So we can add an **async** semantic check and await it when an embedding generator is configured.
   - Flow per item: resolve subject/object entities (unchanged) → run **string** check → if string check fails and embedding_generator is set, run **semantic** check → accept if either string or semantic passes (for hybrid).

3. **Configuration**
   - `MedLitRelationshipExtractor`: add optional `embedding_generator` and optional `evidence_similarity_threshold: float = 0.5`.
   - Ingest script: pass the same `embedding_generator` used for entity resolution into `MedLitRelationshipExtractor` when building the orchestrator.

4. **Trace / observability**
   - When semantic check runs, include in the decision trace: `subject_similarity`, `object_similarity`, and whether the pass was due to `string_match` or `semantic_match`. This keeps existing trace format extensible and debuggable.

### 3.3 Threshold guidance

- **0.3:** Very permissive (risk of false positives).
- **0.5:** Moderate; good default (evidence should be clearly related to the entity).
- **0.7:** Strict (evidence must strongly reference the entity).
- **0.9:** Very strict (near-exact semantic match).

Plan: default **0.5**, make it configurable so it can be tuned per run or environment.

### 3.4 Edge cases and caching

- **Missing entity embedding:** If `subject_entity.embedding` or `object_entity.embedding` is None, call `embedding_generator.generate(entity.name)` and cache by entity name for the duration of the document (see below).
- **Empty evidence:** Keep current behavior: reject with `evidence_empty`; do not call the embedding API.
- **Embedding generator None:** If not provided, use only string-based validation (current behavior).
- **Evidence embedding cache (mandatory):** Multiple relationships often cite the same evidence quote. Cache evidence embeddings per document to avoid redundant API calls. Example: 10 relationships, 5 same evidence → 1 API call + 4 cache hits instead of 5 calls.
- **Entity name embedding cache (recommended):** When an entity has no `.embedding`, we generate from `entity.name`; cache by name for the document so the same entity name is not embedded more than once.

**Cache lifecycle:** Both caches are per-document (or per `_extract_with_llm` call). Clear or create fresh dict at the start of each document’s relationship extraction. Pass the evidence cache into `_evidence_contains_both_entities_semantic` so the helper can read/write it.

**Explicit cache attributes on MedLitRelationshipExtractor:**

```python
def __init__(self, ..., embedding_generator=None, evidence_similarity_threshold=0.5):
    # ... existing init ...
    self._embedding_generator = embedding_generator
    self._evidence_similarity_threshold = evidence_similarity_threshold
    # Per-document caches; reset at start of each _extract_with_llm call
    self._evidence_embedding_cache: dict[str, tuple[float, ...]] = {}
    self._entity_name_embedding_cache: dict[str, tuple[float, ...]] = {}
```

---

## 4. File-by-file changes

### 4.1 `examples/medlit/pipeline/relationships.py`

- **Imports:** Add `from kgraph.storage.memory import _cosine_similarity` (or a local cosine helper if we prefer not to depend on storage). If we need async embedding calls, we already have async context in `_extract_with_llm`.
- **New async helper:** `async def _evidence_contains_both_entities_semantic(evidence, subject_entity, object_entity, embedding_generator, threshold, evidence_cache, entity_name_cache) -> tuple[bool, str | None, dict]`. Returns same shape as `_evidence_contains_both_entities`. Uses `evidence_cache` (read/write keyed by evidence string) so each evidence text is embedded at most once per document. Uses `entity_name_cache` for entities missing `.embedding`. Detail must include `subject_in_evidence`, `object_in_evidence`, `subject_similarity`, `object_similarity`; caller adds `method` and `threshold` when semantic is used.
- **MedLitRelationshipExtractor:**
  - Add `__init__` parameters: `embedding_generator` (optional), `evidence_similarity_threshold: float = 0.5`.
  - Store as instance attributes.
- **Evidence validation flow (in async path) — concrete integration:**

  At the start of `_extract_with_llm`, reset caches: `self._evidence_embedding_cache = {}`, `self._entity_name_embedding_cache = {}`.

  For each item from the LLM response, the flow is:

  ```python
  # 1. Resolve entities (unchanged)
  subject_entity = _pick_unique(entity_index.get(_normalize_entity_name(subject_name), []))
  object_entity = _pick_unique(entity_index.get(_normalize_entity_name(object_name), []))
  if not subject_entity or not object_entity:
      # drop with subject_unmatched / object_unmatched
      ...

  # 2. String evidence check (sync, fast)
  evidence_ok, reason, detail = _evidence_contains_both_entities(
      evidence, subject_name, object_name, subject_entity, object_entity
  )

  # 3. If string failed but we have semantic capability, try semantic
  if not evidence_ok and self._embedding_generator is not None:
      evidence_ok, reason, detail = await _evidence_contains_both_entities_semantic(
          evidence,
          subject_entity,
          object_entity,
          self._embedding_generator,
          self._evidence_similarity_threshold,
          self._evidence_embedding_cache,
          self._entity_name_embedding_cache,
      )
      if evidence_ok:
          detail["method"] = "semantic_match"
          detail["threshold"] = self._evidence_similarity_threshold

  if not evidence_ok:
      # record rejection in trace, return None, decision
      ...

  # 4. Continue with predicate semantics, swap, type constraints (unchanged)
  ```

  So the semantic check is an **async override** when string check fails; when it passes, we continue with the rest of `_process_llm_item` (predicate semantics, swap, type constraints, etc.). If we keep `_process_llm_item` sync, the above block lives in `_extract_with_llm`, which does the entity resolution and evidence check (including async semantic fallback), then calls into the rest of the processing (or we split so the sync part is a helper called with pre-validated evidence).

- **Trace:** When semantic check is used, set in `decision["evidence_check"]`: `subject_similarity`, `object_similarity`, `method` (e.g. `"semantic_match"` or `"string_match"` when string passed), and `threshold` so traces are self-documenting.

### 4.2 `examples/medlit/scripts/ingest.py`

- In `build_orchestrator`, where `MedLitRelationshipExtractor(...)` is constructed, add:
  - `embedding_generator=embedding_generator` (the same one used for the resolver and orchestrator).
  - `evidence_similarity_threshold` from args (see CLI below).
- **CLI (Priority 2):** Add arguments so users can control validation and threshold without code changes:
  - `--evidence-validation-mode [string|semantic|hybrid]` — default: `hybrid`. If `string`, pass `embedding_generator=None` to the relationship extractor. If `semantic`, skip string check and use only semantic (requires embedding generator). If `hybrid`, current design (string first, semantic fallback).
  - `--evidence-similarity-threshold FLOAT` — default: `0.5`. Passed into `MedLitRelationshipExtractor(evidence_similarity_threshold=args.evidence_similarity_threshold)`.

### 4.3 `kgraph/storage/memory.py`

- No change required; we only **use** `_cosine_similarity`. If we prefer not to import from storage in the medlit pipeline, we can copy a small `_cosine_similarity` helper into `relationships.py` or a shared util.

### 4.4 Tests

- **Unit tests for semantic evidence helper:** In `tests/` or `examples/medlit/tests/`, add tests for `_evidence_contains_both_entities_semantic` using a **mock** embedding generator that returns fixed vectors (e.g. identical vector for “MBC” and “Male breast cancer” so similarity is 1.0; orthogonal vectors for unrelated terms so similarity is 0). Assert return shape and `ok`/detail given threshold.
- **Existing evidence tests:** In `tests/test_relationship_swap.py` there are tests for `_evidence_contains_both_entities` (string). Keep them; they validate string behavior. New semantic tests should not replace them.
- **Integration (optional):** A single integration test that runs relationship extraction with a mock embedding generator and checks that a relationship with abbreviation in evidence is accepted when semantic is enabled could be added later.

### 4.5 Documentation

- In `examples/medlit/README.md` or `CANONICAL_IDS.md` (or a new “Relationship extraction” section), document:
  - That evidence validation can use semantic similarity when an embedding generator is provided.
  - The default threshold (0.5) and how to change it if we expose it (e.g. env or CLI).
  - That entity embeddings are reused when present, and only the evidence string is embedded per candidate.

---

## 5. API summary (for implementer)

- **New async function (relationships.py):**
  - `async def _evidence_contains_both_entities_semantic(evidence: str, subject_entity: BaseEntity, object_entity: BaseEntity, embedding_generator, threshold: float, evidence_cache: dict, entity_name_cache: dict) -> tuple[bool, str | None, dict[str, Any]]`
  - **Evidence embedding:** If `evidence not in evidence_cache`, `evidence_cache[evidence] = await embedding_generator.generate(evidence)`; then `evidence_emb = evidence_cache[evidence]`.
  - **Subject/object embeddings:** Use `subject_entity.embedding` if present; else if `subject_entity.name not in entity_name_cache`, `entity_name_cache[subject_entity.name] = await embedding_generator.generate(subject_entity.name)`; then use cached embedding. Same for object.
  - `subject_sim = _cosine_similarity(evidence_emb, subject_emb)`, same for object.
  - `ok = subject_sim >= threshold and object_sim >= threshold`.
  - `detail = {"subject_in_evidence": subject_sim >= threshold, "object_in_evidence": object_sim >= threshold, "subject_similarity": subject_sim, "object_similarity": object_sim}`. Caller adds `method` and `threshold` when semantic is used.
  - Return `(ok, None, detail)` if ok else `(False, "evidence_missing_subject" | "evidence_missing_object" | "evidence_missing_subject_and_object", detail)`.

- **MedLitRelationshipExtractor:**
  - `__init__(self, llm_client=..., domain=..., trace_dir=..., embedding_generator=..., evidence_similarity_threshold=0.5)`.

- **Call flow in _extract_with_llm (high level):**
  - At start: reset `self._evidence_embedding_cache = {}`, `self._entity_name_embedding_cache = {}`.
  - For each item: get subject/object entities; if either missing, drop (unchanged).
  - Run `evidence_ok, reason, detail = _evidence_contains_both_entities(...)` (string).
  - If not evidence_ok and self._embedding_generator is not None: `evidence_ok, reason, detail = await _evidence_contains_both_entities_semantic(..., self._evidence_embedding_cache, self._entity_name_embedding_cache)`; if ok, set `detail["method"] = "semantic_match"`, `detail["threshold"] = self._evidence_similarity_threshold`.
  - If still not evidence_ok, drop and record reason/detail (unchanged).
  - Else continue with predicate semantics, swap, type constraints, etc.

---

## 6. Rollback and configuration

- If `embedding_generator` is None, behavior is unchanged (string-only).
- **CLI flags (see 4.2):** `--evidence-validation-mode` (string | semantic | hybrid) and `--evidence-similarity-threshold` give full control. For example, `--evidence-validation-mode string` passes `embedding_generator=None` to the relationship extractor even when the rest of the pipeline uses embeddings.
- Threshold is configurable via `--evidence-similarity-threshold` (default 0.5).

---

## 7. Success criteria

- With semantic evidence validation enabled (default when embedding_generator is provided) and threshold 0.5:
  - Sample paper that previously yielded **0** relationships yields on the order of **8–15** relationships when the evidence uses abbreviations or related terms (e.g. MBC, breast cancer, pleural effusions).
- Trace files still contain `evidence_check` with at least `subject_in_evidence`, `object_in_evidence`; when semantic is used, they also contain `subject_similarity` and `object_similarity`.
- Existing tests (string-based evidence) still pass.
- New unit tests for the semantic helper (with mock embeddings) pass.

---

## 8. Out of scope for this plan

- Changing how entity names or synonyms are stored (e.g. adding alias lists) for string matching; that can be a follow-up.
- Batching evidence embeddings across items (reduces API calls but adds complexity; can be a later optimization).
- Changing predicate semantics or type-constraint validation; this plan is only about evidence validation.

---

*End of implementation plan. Ready for Claude (or human) review.*
