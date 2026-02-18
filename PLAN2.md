# Implementation Plan: Embedding Cache Fix and Ollama Performance (TODO2)

This plan addresses **TODO2.md**: embedding caching that was not working (causing expensive repeated API calls) and performance improvements for the Ollama client (thread pool, batch embeddings).

**How to use this plan:** Work through Phase 1 and Phase 3 first (can be done in parallel). Phase 1 fixes the cache so lookups hit; Phase 3 verifies the ingest script and adds one comment. Phase 2 (Ollama batch + optional executor) is optional. Phase 4 tests go in `tests/test_caching.py`. **You do not need to open any other files** — all required names, line numbers, and code snippets are in the "Code reference" and "Exact edits" sections below. **Success:** After Phase 1 + 3, a second ingest run over the same corpus shows cache hits; no duplicate API calls for the same normalized text.

---

## What was missing? (Why “caching was never used”)

The caching **was** wired in: the ingest script creates a `CachedEmbeddingGenerator` when `embeddings_cache_file` is set (always in main, since `output_dir` defaults to `medlit_bundle`), passes that single instance to the orchestrator, resolver, relationship extractor, and authority lookup, and calls `cache.load()` before extraction and `save_cache()` in a `finally` block. So the intended path was “use cache.”

The most plausible reasons it still looked like “cache never used” (lots of API calls despite “Embeddings cache loaded (110 entries)”):

1. **Keys not normalized on load**  
   `FileBasedEmbeddingsCache.load()` does **not** normalize keys when building the in-memory cache; it uses keys exactly as they appear in the JSON file. `get()` and `put()` **do** normalize (via `_normalize_key`). So:
   - If the cache file was ever written with non-normalized keys (e.g. by an older version, or by a different tool), then in memory we have e.g. `"Aspirin"` but lookups use `"aspirin"` → **every lookup misses**.
   - Even if our current `put()` always writes normalized keys, any existing file with mixed-case or different whitespace would still cause misses until we **normalize keys when loading** so that in-memory keys match the convention used in get/put.

2. **Normalization too weak**  
   Current normalization is `text.lower().strip()`. If the same logical text is sometimes stored with different whitespace or formatting (e.g. `"Ki67"` vs `"ki67"`, or `"entity  (type)"` vs `"entity (type)"`), we can get different keys and unnecessary misses. Stronger normalization (e.g. collapse internal whitespace) would make the cache more effective.

3. **No visibility**  
   There was no logging of cache hits vs misses, so it was hard to see whether the cache was being hit or whether every request was a miss. Adding hit/miss (and optionally key) logging would have shown that “cache is used but all misses.”

So the **missing piece** was almost certainly **key consistency between load and get/put**: either normalize keys when loading (so any existing file works), or ensure the same normalization is applied everywhere and that we don’t rely on the file having been written with that convention. Optionally, stronger normalization and hit/miss logging would have made the behavior obvious.

---

## Summary of Goals

| Priority | Goal |
|----------|------|
| **Required** | Fix embedding cache so repeated requests for the same (or equivalently normalized) text hit the cache and do not call the API. |
| **Required** | Harden cache: normalize keys on load, optional stronger normalization, asyncio lock, optional hit/miss logging. |
| **Required** | Verify ingest script: single cached generator, cache load before use, save on exit, correct path. |
| **Optional** | Ollama client: batch embedding (one request for many texts), configurable thread pool / executor. |

---

## Cache at entity level (canonical/resolved), not just mention level

**Idea:** Once an entity is resolved, reuse its embedding for all future uses of that entity instead of re-embedding by mention text or entity name each time.

**Are we already doing that?** Partially.

- **Relationship evidence check** (`relationships.py`): We **already** prefer `entity.embedding`. We only call the generator when `subject_entity.embedding` or `object_entity.embedding` is None, and we have a per-document `entity_name_cache` keyed by `entity.name`. So when the entity has an embedding (e.g. set at creation in the same run), we don’t call the API.
- **Ingest (new entities):** We embed once per new entity (`entity.name`) and set `entity.embedding`; we never re-embed when we only update usage_count for an existing entity.
- **Resolution:** We embed **every** `mention.text` to run similarity search. We don’t have the entity yet, so we can’t “reuse entity embedding” there. We could, however, do a **normalized name lookup first** and only embed when there’s no exact/normalized name match, which would cut resolution-time embeddings for the common case where the mention text matches an existing entity’s name.

**Would entity-level caching help?** Yes, in two ways:

1. **Persistent cache keyed by `entity_id`**  
   When we generate an embedding for a **known entity** (e.g. after resolution when we set `entity.embedding`, or when we need an embedding for an entity in the relationship phase), we could also store it in the file cache under `entity_id` (e.g. a dedicated key space or a second cache). Then:
   - In a **later run** (or after reload), when we need an embedding for that entity (e.g. relationship check and the entity was loaded without an embedding from the bundle), we’d look up by `entity_id` and reuse, instead of re-embedding `entity.name`.  
   - Today the cache is **text-only**; if the bundle doesn’t persist embeddings, every run re-embeds entity names. Keying by `entity_id` would make “one embedding per entity per cache” explicit and would help as soon as we have a stable entity_id (canonical or provisional).

2. **Resolution: name-first, embed only when needed**  
   Before calling `embedding_generator.generate(mention.text)` for similarity search, try to resolve by **normalized name** (e.g. lookup existing entities by normalized `mention.text`). If we find a unique match, return it without embedding. Only embed when we need to disambiguate or when there’s no name match. That reduces resolution-time API calls for exact/near-exact name matches.

**Effectiveness:** High for (1) when entities are reused across many relationship checks or across runs and embeddings aren’t persisted in the bundle. Medium for (2) when a large share of mentions match an existing entity by name. Both are compatible with the current text-level cache (we can add entity_id as an additional key space or a separate cache layer).

**Recommendation:** Add to PLAN2 as an optional Phase 2b or 3: (a) extend the embedding cache (or add a small layer) to support `get_by_entity_id` / `put_by_entity_id` and have call sites that have an entity (e.g. relationship evidence check) prefer `entity.embedding` then entity_id cache then generate and cache by entity_id and by name; (b) in the resolver, try normalized name lookup before embedding and only embed when necessary.

**Addressing “we need embedding in phase 2 before promotion”:** Your concern is valid if we ever **removed** embedding from resolution or **only** keyed cache by entity_id and left phase 2 with no way to get embeddings. In the intended design we do neither:

- **Resolution (pass 1):** We keep embedding-based matching. The “name-first” idea is a **fast path only**: try normalized name lookup first; **if no match**, then embed `mention.text` and run `find_by_embedding` as today. So we still need and use the embedding API in pass 1 for any mention that doesn’t exactly match an existing entity name (e.g. “BRCA-1” → “BRCA1”, “p53” → “TP53”). Embedding is not removed from resolution.
- **Phase 2 (relationship extraction):** We already need embeddings for the evidence check. We use `entity.embedding` when set; when it’s `None` (e.g. entity was matched by name and never embedded, or loaded from a bundle without embeddings), we generate on demand and can cache by `entity.name` and/or `entity_id`. So we still “benefit” from embedding in phase 2 — we still run the evidence check; we just generate the embedding there when missing. That can mean more API calls in phase 2 for name-matched entities that don’t have `entity.embedding` set, but we don’t lose the semantic check. An entity_id cache would then ensure we only generate once per entity per run.
- **Before promotion:** Provisionals have stable `entity_id`s for the duration of the run. So caching by `entity_id` (and/or by `entity.name`) in phase 2 still helps: first time we need an embedding for that entity we generate and cache; subsequent uses in the same run hit the cache. We do not key only by “canonical” identity; we key by the current `entity_id` (canonical or provisional), so we benefit before promotion too.

So: we keep embedding in resolution for semantic matching; we keep embedding in phase 2 for evidence check (on demand when `entity.embedding` is None); and entity-level (entity_id) cache is an **additional** layer so we don’t re-embed the same entity across documents or within phase 2. The text-level cache remains primary for resolution (same mention text → same embedding); entity_id cache is an extra win when we have a resolved entity and want to reuse its embedding later.

**Entity-level cache (optional / future phase):** Not in scope for Phase 1–4 below. When implementing: (a) extend cache or add a layer with `get_by_entity_id(entity_id)` / `put_by_entity_id(entity_id, embedding)`; (b) in relationship evidence check (and any caller that has an entity), lookup order: `entity.embedding` → entity_id cache → generate and store by entity_id and by name; (c) in resolver, before embedding: try normalized name lookup; only call `generate(mention.text)` when there is no exact/normalized name match.

---

## Phase 1: Diagnose and fix embedding cache

**Minimum required to fix the cache:** Tasks 1.1 and 1.3. Task 1.4 is no code change. Tasks 1.2 and 1.5 are optional (stronger normalization and debug logging).

### 1.1 Root causes (from TODO2 and code review)

Possible reasons the cache appeared not to work:

1. **Keys not normalized on load**  
   `FileBasedEmbeddingsCache.load()` fills `_cache` with keys exactly as stored in the JSON file. If the file was ever written with non-normalized keys (e.g. "Aspirin") or by code that didn’t normalize, then a lookup with normalized key "aspirin" would miss. **Fix:** When loading, normalize every key before inserting into `_cache` so in-memory cache always uses the same key convention as `get()`/`put()`.

2. **Key normalization too weak**  
   Current `_normalize_key` is `text.lower().strip()`. Variations like "entity (type)" vs "entity  (type)" (double space) or different Unicode could still differ. **Fix:** Optionally collapse internal whitespace to a single space (e.g. `" ".join(text.lower().strip().split())`) so cache keys are stable across minor formatting differences. Document the chosen normalization and keep it consistent in get/put/load/save.

3. **Concurrent access**  
   `FileBasedEmbeddingsCache` and `InMemoryEmbeddingsCache` do not use a lock. Under async concurrency, overlapping `get`/`put`/`save`/`load` can corrupt `_cache` or `_dirty`. **Fix:** Add an `asyncio.Lock` and hold it only around the **critical section** in each method (dict access and counter updates), **not** around the entire method body. See "Lock scope (deadlock avoidance)" below — wrapping whole methods would deadlock because `get_batch` calls `get`, `put_batch` calls `put` (InMemory), and `put`/`put_batch` call `save` (FileBased).

4. **Cache not actually used for some call paths**  
   If any code path creates its own embedding generator (e.g. a new `OllamaMedLitEmbeddingGenerator`) or gets a non-cached reference, that path would bypass the cache. **Verification:** Confirm that the only embedding generator used during ingestion is the one returned by `build_orchestrator` (the cached one when `embeddings_cache_file` is set) and that it is passed into the orchestrator, resolver, relationship extractor, and authority lookup. No ad-hoc instantiation of the base generator in the pipeline.

### 1.2 Implementation tasks

| # | Task | File(s) | Details |
|---|------|---------|--------|
| 1.1 | Normalize keys on load | `kgraph/pipeline/caching.py` | In `FileBasedEmbeddingsCache.load()`, replace the line that builds `_cache` from `data.items()` with the exact replacement in the Code reference below. When `self.config.normalize_keys` is True, use `self._normalize_key(k)` for each key so in-memory keys match `get()`/`put()`. If two file keys normalize to the same key, last wins. |
| 1.2 | Optional stronger key normalization | `kgraph/pipeline/caching.py` | Add optional “collapse whitespace” step in `_normalize_key` (e.g. config flag `normalize_collapse_whitespace: bool = True` and `" ".join(text.lower().strip().split())`). Ensure all get/put/load/save use the same logic. |
| 1.3 | Add asyncio lock to caches | `kgraph/pipeline/caching.py` | In both cache classes: in `__init__`, set `self._lock = asyncio.Lock()`. Hold the lock only around the **critical section** in each method (see "Lock scope (deadlock avoidance)" below). Do **not** wrap the entire method body, or re-entrant calls will deadlock. |
| 1.4 | Save uses normalized keys | `kgraph/pipeline/caching.py` | No code change if 1.1 is done: `save()` writes `self._cache`; after 1.1, keys are normalized. |
| 1.5 | Optional cache debug logging | `kgraph/pipeline/caching.py` | In `get()`, add optional DEBUG log: "CACHE HIT" or "CACHE MISS" with key prefix (e.g. first 40 chars). Gate on config or env (e.g. `KG_EMBED_CACHE_DEBUG=1`). |

#### Lock scope (deadlock avoidance)

The cache has an internal call graph: **get_batch** calls **get** (both caches); **put_batch** calls **put** (InMemory); **put** and **put_batch** call **save** (FileBased). If the lock is held for the **entire** body of each method, a non-reentrant lock will deadlock (e.g. get_batch acquires lock, then get() tries to acquire it again).

**Correct approach:** Hold the lock only around the minimal critical section in each method.

- **get(key):** Compute `key` (normalize if config) **outside** the lock. Acquire lock → if key in _cache: increment _hits, move_to_end(key), read value into a local; else: increment _misses, value = None → release. Return value after release.
- **get_batch:** Either (a) implement the loop inline: acquire lock once, for each text do lookup + move_to_end + append to results, update counters, release; or (b) leave get_batch as a loop over `await self.get(text)` and ensure get() only locks around its critical section (option (b) is minimal change).
- **put:** Compute `key` outside the lock. Acquire lock → update dict, _dirty, evict if needed, update _updates_since_save → release. Then **outside** the lock, if auto_save_interval reached, call `await self.save()`.
- **put_batch:** Critical section under lock (the whole for-loop that updates _cache, _dirty, evictions, _updates_since_save). Call `await self.save()` only **after** releasing the lock when interval reached.
- **save:** Acquire lock → read _dirty (early return if False), copy `_cache` into a local dict, set _dirty = False → release. Then do all file I/O (temp file, rename) **outside** the lock.
- **load:** Do file read and `data = json.load(f)` **first** (no lock). Then acquire lock → assign `_cache` from loaded data (use normalized keys per task 1.1), set _dirty = False → release.
- **get_stats:** Sync method; cannot use `async with self._lock`. Leave as-is (reads may be a best-effort snapshot). No lock required for correctness of get/put/save/load.
- **clear:** Async; holds lock for entire body (no internal call to get/put/save/load). Acquire lock → clear _cache, reset all counters (and _dirty, _updates_since_save for FileBased) → release.

**Example (get) — pattern to follow:**

```python
async def get(self, text: str) -> Optional[tuple[float, ...]]:
    key = self._normalize_key(text) if self.config.normalize_keys else text
    async with self._lock:
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            result = self._cache[key]  # tuple, immutable; safe to return after release
        else:
            self._misses += 1
            result = None
    return result
```

So: lock protects only in-memory state (`_cache`, `_dirty`, counters). I/O and any call to another cache method happen outside the lock.

### 1.3 Verification

- **Unit tests**  
  - **Normalization on load:** Create a `FileBasedEmbeddingsCache` with a JSON file that has mixed-case keys (e.g. `{"Aspirin": [0.1, 0.2, ...]}`). Call `await cache.load()`. Then `await cache.get("aspirin")` and `await cache.get("Aspirin")` must both return the same embedding. With `CachedEmbeddingGenerator` and a mock base generator, base `generate()` must be called only once for that text.  
  - **Concurrency:** Spawn multiple async tasks that call `get`/`put`/`get_batch`/`put_batch` on the same cache; call `save()` and `load()` in between; assert no exceptions and that all expected keys are present after load.  
- **Integration**  
  - Run ingest on 2–3 papers with `embeddings_cache_file` set (e.g. `--output-dir medlit_bundle`). First run: note "Embeddings cache saved (N entries)". Second run (same corpus): note "Embeddings cache loaded (N entries)" and, if hit/miss stats are implemented, confirm hits > 0 and cache size stable or slightly larger.

### 1.4 Deliverables

- [ ] Keys normalized on load in `FileBasedEmbeddingsCache`.
- [ ] Optional stronger normalization (collapse whitespace) and config.
- [ ] `asyncio.Lock` in both cache implementations; held only around critical sections (not whole methods) to avoid deadlock.
- [ ] Optional cache hit/miss debug logging.
- [ ] Tests for load normalization and concurrent access.
- [ ] (Optional) Remove or reduce temporary DEBUG prints in `OllamaMedLitEmbeddingGenerator` when touching that file in Phase 2.

---

## Phase 2: Ollama client – thread pool and batch embeddings

### 2.1 Current state

- **LLM client** (`examples/medlit/pipeline/llm_client.py`): Already uses `asyncio.to_thread(_generate)` for sync Ollama calls, so the event loop is not blocked. No dedicated thread pool.
- **Embedding generator** (`examples/medlit/pipeline/embeddings.py`): `OllamaMedLitEmbeddingGenerator.generate_batch()` is implemented as a loop over `generate(t)` — one HTTP request per text. Ollama’s `/api/embeddings` (or equivalent) may support batch input; if so, a single batch request would reduce latency and load.

### 2.2 Thread pool (optional)

- **Goal:** Allow limiting concurrent Ollama calls (e.g. max N in flight) to avoid overloading the server or to prioritize other work.
- **Approach:** Add an optional `ThreadPoolExecutor` (or use a bounded semaphore) to the Ollama LLM client. When provided, use `loop.run_in_executor(self._executor, _generate)` instead of `asyncio.to_thread(_generate)`. Default can remain `asyncio.to_thread` so behavior is unchanged unless the user configures an executor.
- **Scope:** If we only care about embeddings (not chat), the thread pool can be added to the embedding generator instead of (or in addition to) the LLM client. Plan below assumes we can add it to the embedding generator for embedding-specific concurrency control.

### 2.3 Batch embeddings (Ollama)

- **Goal:** Reduce the number of HTTP requests when embedding many texts (e.g. evidence strings, entity names in a window).
- **Ollama API:** Check Ollama’s embedding API (e.g. `POST /api/embed` or `/api/embeddings`; see [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md)). If it accepts multiple inputs (e.g. `input` or `prompts` as a list) and returns a list of vectors in order, implement a single batch request in `OllamaMedLitEmbeddingGenerator.generate_batch()`.
- **Implementation:**  
  - In `examples/medlit/pipeline/embeddings.py`, add a method (e.g. `_generate_batch_ollama(texts: list[str])`) that posts one request with all texts and parses the response into a list of tuples.  
  - If the API returns embeddings in the same order as the input, use that order; otherwise match by index or as documented.  
  - `generate_batch()`: call the new batch method when the backend supports it; otherwise fall back to the current loop over `generate(t)`.  
- **CachedEmbeddingGenerator:** Already implements batch: get_batch for cache, then only for misses call `base_generator.generate_batch(miss_texts)`. So once the base generator has a real batch implementation, the cached path will benefit (fewer network round-trips for misses).

### 2.4 Implementation tasks

| # | Task | File(s) | Details |
|---|------|---------|--------|
| 2.1 | Verify Ollama embedding batch API | (research) | Confirm Ollama’s embedding endpoint and request/response format for multiple prompts. Document in code or PLAN2. |
| 2.2 | Implement batch embedding in Ollama generator | `examples/medlit/pipeline/embeddings.py` | Only if batch API exists: one-request batch (e.g. prompts list). Set generate_batch() to use it; handle empty and single-item; preserve order. If no batch API, skip. |
| 2.3 | Optional: executor for embedding generator | `examples/medlit/pipeline/embeddings.py` | Add optional `executor: ThreadPoolExecutor | None` to the constructor. When executor is set and a call hits the network, run the blocking HTTP call via `loop.run_in_executor(self._executor, sync_wrapper)` (obtain `loop` with `asyncio.get_running_loop()`); otherwise keep current async httpx path. |
| 2.4 | Optional: executor for LLM client | `examples/medlit/pipeline/llm_client.py` | If we want a dedicated pool for chat: add optional executor to `OllamaLLMClient` and use it in `generate` / `_call_llm_for_json` instead of `asyncio.to_thread`. Default remains current behavior. |

### 2.5 Deliverables

- [ ] Ollama embedding batch API documented and, if supported, implemented in `OllamaMedLitEmbeddingGenerator.generate_batch()`.
- [ ] (Optional) Configurable executor for embedding generator and/or LLM client for bounded concurrency.
- [ ] Tests: batch embedding returns correct number and order of vectors; cache + batch path (batch of mixed hits/misses) produces correct results.

---

## Phase 3: Ingest script and cache lifecycle

### 3.1 Ensure cache is loaded and saved

- **Load:** In `examples/medlit/scripts/ingest.py`, locate the call to `cached_embedding_generator.cache.load()` (or equivalent). Verify it runs in `main()` **before** the first use of embeddings (e.g. before the loop that calls `orchestrator.ingest_document` or `extract_entities_from_document`). If load is conditional (e.g. only when cache file path is set), confirm that path is always set when running full ingest (e.g. `output_dir / "embeddings_cache.json"`).
- **Save:** Locate the `finally` block (or equivalent) that calls `save_cache()`. Verify it runs on normal exit and on exception so new embeddings are not lost. Confirm `save_cache()` calls `cache.save()` and that `FileBasedEmbeddingsCache.save()` writes to `config.cache_file`; confirm the ingest script passes the same path used for load (e.g. `embeddings_cache_file = output_dir / "embeddings_cache.json"`).

### 3.2 Single embedding generator instance

- In `examples/medlit/scripts/ingest.py`, trace the single `embedding_generator` (or `cached_embedding_generator`) returned by `build_orchestrator`. Verify that **the same** instance is passed to: (1) `IngestionOrchestrator(..., embedding_generator=...)`, (2) `MedLitRelationshipExtractor(..., embedding_generator=...)` (or equivalent; may be None when validation is "string"), (3) `_initialize_lookup(..., embedding_generator=...)` if used for canonical ID reranking, (4) the resolver (via orchestrator or pipeline setup). Grep for `EmbeddingGenerator`, `OllamaMedLitEmbeddingGenerator`, or `embedding_generator` in the medlit pipeline and ingest script to ensure no second instance is constructed for embeddings.
- **Exact edit:** In `examples/medlit/scripts/ingest.py`, in `build_orchestrator`, find the line `embedding_generator = cached_embedding_generator` (grep for it if needed). Insert a new line immediately **after** that line with exactly this comment: `# Single cached embedding generator is shared so all embedding calls hit the same cache.`

### 3.3 Log cache stats

- After load: already prints "Embeddings cache loaded (N entries)".
- After save: already prints "Embeddings cache saved (N entries)".
- Optionally print hit/miss counts at the end (e.g. "Cache stats: hits=X, misses=Y") so the user can verify that the cache is being hit on repeated text.

### 3.4 Deliverables

- [ ] Verified: `cache.load()` runs before first embedding use in ingest; `save_cache()` in `finally`; path for load and save is the same (e.g. `output_dir / "embeddings_cache.json"`).
- [ ] Verified: one embedding generator instance is passed to orchestrator, relationship extractor, lookup, and resolver; no other embedding generator is constructed in the pipeline. Comment added in ingest script.
- [ ] Optional: at end of run, log cache stats (e.g. from `get_cache_stats()`: hits, misses, size) so user can confirm cache is being hit on second run.

---

## Phase 4: Testing and validation

### 4.1 Unit tests — all in tests/test_caching.py

- **Normalization on load:** In the test, create a temporary JSON file (e.g. `tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)` then write `{"Aspirin": [0.1, 0.2], "  ibuprofen  ": [0.3, 0.4]}`, close, pass path to `EmbeddingCacheConfig(cache_file=Path(path), normalize_keys=True)` and `FileBasedEmbeddingsCache(config=config)`. Call `await cache.load()`. Assert `await cache.get("aspirin")` and `await cache.get("Aspirin")` return the same tuple; assert `await cache.get("ibuprofen")` returns the ibuprofen vector. Assert `len(cache._cache)` is 2 (Aspirin/aspirin collapse to one key; ibuprofen one key). Clean up temp file in test teardown.
- **Concurrency:** Create one cache instance. Spawn e.g. 10 async tasks that each call `get`/`put` on a mix of keys (use asyncio.gather). Then call `await cache.save()` and `await cache.load()` (for file-based). Assert no exceptions; assert all keys that were put are still present in `cache._cache`.
- **CachedEmbeddingGenerator:** Use a mock base generator that records each `generate(text)` and `generate_batch(texts)` in a list. (1) Call `await cached_gen.generate("x")` twice; assert the mock was called once with "x". (2) Call `await cached_gen.generate_batch(["a", "b", "a"])`; assert the mock’s `generate_batch` was called once with a list of two items (e.g. ["a", "b"] — deduplicated or in order of misses); assert the returned list has 3 elements in the same order as input.

### 4.2 Integration / manual check

- Run: ingest 2–3 papers with `--output-dir medlit_bundle` (or equivalent so `embeddings_cache.json` is used). First run: note final “Embeddings cache saved (N entries)”. Second run on same corpus: note “Embeddings cache loaded (N entries)” and, if cache stats are logged at end, hits > 0 and size ≈ same or slightly larger. If debug logging is enabled (`KG_EMBED_CACHE_DEBUG=1`), confirm “CACHE HIT” lines on second run.

### 4.3 Deliverables

- [ ] New or extended tests for cache normalization on load and concurrency.
- [ ] Test for CachedEmbeddingGenerator (single and batch) with mock base generator.
- [ ] Brief note in docs or README on how to enable and verify embedding cache (e.g. `--output-dir` with embeddings_cache.json, second run should show cache hits).
- [ ] Run before marking PLAN2 complete: `uv run pytest tests/test_caching.py -v`

---

## Dependency order

1. **Phase 1** (cache fixes) — do first; unblocks correct cache behavior. Implements normalize-on-load, lock, optional stronger normalization, optional debug logging.
2. **Phase 3** (ingest script verification) — can run in parallel with Phase 1. Verify load/save path and single generator; add comment and optional stats.
3. **Phase 2** (Ollama batch + optional executor) — optional; do after Phase 1 and 3. Improves throughput once cache is correct.
4. **Phase 4** (tests and validation) — add tests as you implement Phase 1 and 2; run `uv run pytest tests/test_caching.py -v` before considering PLAN2 complete.
5. **Entity-level cache** (optional / future) — deferred; implement after Phase 1–4 if desired (see “Entity-level cache (optional / future phase)” above).

---

## File checklist (summary)

| Area | File(s) |
|------|---------|
| Cache fixes | `kgraph/pipeline/caching.py` (normalize on load, lock, optional stronger normalization, optional debug) |
| Ollama embeddings | `examples/medlit/pipeline/embeddings.py` (batch API, optional executor; remove DEBUG prints) |
| Ollama LLM | `examples/medlit/pipeline/llm_client.py` (optional executor if desired) |
| Ingest | `examples/medlit/scripts/ingest.py` (verify cache path and single generator; optional stats log) |
| Tests | `tests/test_caching.py` (and any new tests for load normalization and concurrency) |

---

## Code reference (exact locations and signatures)

Use this section so you don’t have to re-open files to find names and structure. Line numbers in this doc may have shifted; search for the quoted code or method names when in doubt.

**Step-by-step (no other files needed):** (1) In `kgraph/pipeline/caching.py`: add `import asyncio`; in `FileBasedEmbeddingsCache.load()` replace the line that builds `_cache` with the exact snippet in "Exact replacement" below; in both cache classes' `__init__` add `self._lock = asyncio.Lock()`; in both classes add `async with self._lock:` only around the critical section in each of `get`, `put`, `get_batch`, `put_batch`, `save`, `load`, and `clear` (see "Lock scope (deadlock avoidance)" — do not wrap full method bodies or you risk deadlock). Omit lock from sync method `get_stats`; it remains best-effort. (2) Optional: add `normalize_collapse_whitespace` to `EmbeddingCacheConfig` and override `_normalize_key` in both caches (snippet below). Optional: in `get()` log CACHE HIT/MISS at DEBUG when `os.environ.get("KG_EMBED_CACHE_DEBUG")` is set. (3) In `examples/medlit/scripts/ingest.py`: add a new line after `embedding_generator = cached_embedding_generator` with comment `# Single cached embedding generator is shared so all embedding calls hit the same cache.` (4) Add tests in `tests/test_caching.py` per Phase 4.1; run `uv run pytest tests/test_caching.py -v`.

### kgraph/pipeline/caching.py

**EmbeddingCacheConfig** (lines 45–60): `model_config = {"frozen": True}`; fields: `max_cache_size`, `cache_file`, `auto_save_interval`, `normalize_keys: bool = True`. Add optional `normalize_collapse_whitespace: bool = True` here for task 1.2.

**_normalize_key** is defined **once** on `EmbeddingsCacheInterface` (lines 150–159):

```python
def _normalize_key(self, text: str) -> str:
    """Normalize cache key for consistent lookups."""
    return text.lower().strip()
```

Neither `InMemoryEmbeddingsCache` nor `FileBasedEmbeddingsCache` overrides it; both use it in `get`/`put`/`put_batch` as:

- `key = self._normalize_key(text) if self.config.normalize_keys else text`

For task 1.2 (collapse whitespace): (1) Add to `EmbeddingCacheConfig`: `normalize_collapse_whitespace: bool = Field(False, description="Collapse internal whitespace to single space")`. (2) Override `_normalize_key` in **both** `InMemoryEmbeddingsCache` and `FileBasedEmbeddingsCache` with this body (exact):

```python
def _normalize_key(self, text: str) -> str:
    base = text.lower().strip()
    if getattr(self.config, "normalize_collapse_whitespace", False):
        return " ".join(base.split())
    return base
```

The interface has no `config`, so the override must be in the two concrete classes only.

**FileBasedEmbeddingsCache.load()** — replace the line that builds `_cache` from the loaded JSON (search for `_cache = OrderedDict` in `load()` if line numbers have shifted). Current line:

```python
self._cache = OrderedDict((key, tuple(value)) for key, value in data.items())
```

**Exact replacement** (use this so load uses the same keys as get/put; last key wins on duplicates):

```python
self._cache = OrderedDict(
    (self._normalize_key(k) if self.config.normalize_keys else k, tuple(v))
    for k, v in data.items()
)
```

**Where to add the lock:** Add `import asyncio` at the top of the file (no asyncio import currently). In `InMemoryEmbeddingsCache.__init__` (line 184) and `FileBasedEmbeddingsCache.__init__` (line 328), add `self._lock = asyncio.Lock()`. In each method, hold the lock **only around the critical section** (dict/counter access), not around the whole method — see "Lock scope (deadlock avoidance)" above. In particular: do not hold the lock across `await self.get()`, `await self.put()`, or `await self.save()`; do file I/O and any call to another cache method after releasing the lock.

**Stats:** Both caches have `get_stats(self) -> dict[str, int]` returning `hits`, `misses`, `size`, `evictions`. `CachedEmbeddingGenerator.get_cache_stats()` (line 625) returns `self.cache.get_stats()`.

**Task 1.5 (optional debug logging) — exact behavior:** At top of file add `import os` if not present. In both cache classes’ `get()` method: after computing `key = self._normalize_key(text) if self.config.normalize_keys else text`, add a block that runs only when `os.environ.get("KG_EMBED_CACHE_DEBUG")` is truthy: if `key in self._cache` log at DEBUG level `"CACHE HIT %s" % (key[:40],)`, else log `"CACHE MISS %s" % (key[:40],)`. Use `logging.getLogger(__name__).debug(...)` (add `import logging` at top if not present).

### examples/medlit/scripts/ingest.py

**Cache path:** `embeddings_cache_file = output_dir / "embeddings_cache.json"` (line 1191). Passed into `build_orchestrator(..., embeddings_cache_file=embeddings_cache_file)` (1193–1200).

**Cache load:** In `main()`, after `_initialize_pipeline(args)` (1230). Lines 1232–1236:

```python
if cached_embedding_generator is not None:
    await cached_embedding_generator.cache.load()
    if not quiet:
        stats = cached_embedding_generator.get_cache_stats()
        print(f"  Embeddings cache loaded ({stats.get('size', 0)} entries)", ...)
```

So load runs **before** any document ingestion (entity extraction happens later in `main()`). Good.

**Cache save:** In the `finally` block of `main()` (lines 1361–1366):

```python
finally:
    if not llm_timeout_abort and cached_embedding_generator is not None:
        await cached_embedding_generator.save_cache()
        if not quiet:
            stats = cached_embedding_generator.get_cache_stats()
            print(f"  Embeddings cache saved ({stats.get('size', 0)} entries)", ...)
```

**Single embedding generator:** `build_orchestrator` (line 176) returns `(orchestrator, lookup, cached_embedding_generator)`. When `embeddings_cache_file` is set it builds `FileBasedEmbeddingsCache` + `CachedEmbeddingGenerator` and sets `embedding_generator = cached_embedding_generator` (234); that same `embedding_generator` is passed to: `IngestionOrchestrator(..., embedding_generator=embedding_generator)` (291), relationship extractor via `rel_embedding_generator = ... embedding_generator` (265, 269), resolver (291), and `_initialize_lookup(..., embedding_generator=embedding_generator)` (773). So one instance is shared. Add the “single cached generator” comment near the place `embedding_generator` is first assigned (e.g. around 224–236) or near `build_orchestrator`’s docstring.

### examples/medlit/pipeline/embeddings.py

**OllamaMedLitEmbeddingGenerator:** Single-text request uses `POST {ollama_host}/api/embeddings` with body `{"model": self.model, "prompt": text}` (lines 64–69). Response: `data["embedding"]` or `data["embeddings"][0]`. For Phase 2 batch, check Ollama docs for a batch endpoint or body (e.g. `prompt` as list); if none, keep current `generate_batch` as loop over `generate(t)`.

**generate_batch** (lines 96–106): Currently `return [await self.generate(t) for t in texts]` — one request per text. Replace with one batch request when the API supports it.

**DEBUG prints to remove or gate (Phase 2):** Lines 61–62, 64–65, 71, 73–74, 80–81 (e.g. `print("DEBUG: ...")`, `print("EMBED STATUS:", ...)`). Remove or guard with a debug flag so production logs stay clean.

---

## References

- **TODO2.md** — user report of cache not working; cost on A100; desire for thread pool and batch embeddings.
- **kgraph/pipeline/caching.py** — `EmbeddingsCacheInterface`, `FileBasedEmbeddingsCache.load()` (line 499), `InMemoryEmbeddingsCache`, `CachedEmbeddingGenerator`, `_normalize_key`, `get_stats()` / `get_cache_stats()`.
- **kgraph/pipeline/embedding.py** — `EmbeddingGeneratorInterface`, `generate`, `generate_batch`.
- **examples/medlit/pipeline/embeddings.py** — `OllamaMedLitEmbeddingGenerator`, `generate_batch()` (line 96), single-request URL and body above.
- **examples/medlit/scripts/ingest.py** — `build_orchestrator` (176), cache load (1232–1236), save in `finally` (1361–1366), `embeddings_cache_file` (1191).
