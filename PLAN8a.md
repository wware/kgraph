# Plan 8a: Authoritative Canonical IDs in Pass 2 (PLAN8a)

Execute steps in order from the **repository root**. Reference: **examples/medlit/CANONICAL_IDS.md**.

**Goal:** In Pass 2, reserve "canonical_id" for IDs from authoritative ontologies (MeSH/UMLS, HGNC, RxNorm, UniProt). Use a stable merge key (entity_id) for every entity; when we have an authoritative ID use it as the merge key and set canonical_id; when we do not, use a synthetic slug only as entity_id and set canonical_id to null. Integrate `CanonicalIdLookup` into Pass 2 so new entities are resolved via API when possible.

**Terminology (fixed for this plan):**
- **entity_id** (or **id**): Stable merge key used for dedup and for relationship subject/object. Always present. Either an authoritative ID string or a synthetic slug (e.g. `canon-<uuid>`).
- **canonical_id**: Optional. Set only when the merge key is an authoritative ontology ID; null when the entity is identified only by a synthetic slug. Never output a synthetic slug as canonical_id.

---

## Phase 1 (documentation — small and safe)

Do this first; no code removal. It clarifies that the new two-pass pipeline does not use the old promotion machinery and labels the legacy path.

**1.1 Document that the new pipeline does not use promotion**

- In **examples/medlit/INGESTION.md** (or the main "Two-pass ingestion" section): Add one or two sentences stating that the two-pass flow (Pass 1 → Pass 2) does **not** use promotion (no usage/confidence thresholds, no `PromotionPolicy`). Canonical vs provisional is reflected only by whether an entity has an authoritative `canonical_id` in the Pass 2 output (present) or `canonical_id` null (provisional in that sense).
- Optionally in **examples/medlit/CANONICAL_IDS.md**: In the "Overview" or a new "Two-pass vs legacy" subsection, note that Pass 2 assigns authoritative IDs via bundle + optional lookup and does not run the legacy promotion step.

**1.2 Label the old ingest as legacy**

- In **run-ingest.sh**: Add a short comment at the top (e.g. `# Legacy pipeline: uses examples.medlit.scripts.ingest (entity extraction → promotion → relationship extraction). For the two-pass pipeline use pass1_extract then pass2_dedup.`) so it is clear this script is the old path.
- Optionally in **examples/medlit/scripts/ingest.py**: In the module docstring, add one line such as "Legacy pipeline: uses PromotionPolicy and run_promotion between entity and relationship extraction. The two-pass pipeline (pass1_extract, pass2_dedup) does not use promotion."

**1.3 No removal**

- Do not remove or change any promotion code in kgschema, kgraph, or examples. Phase 2 (see top-level **SIMPLIFY_PROMOTION.md**) may be considered later for removing unused promotion machinery.

---

## Step 0. Pre-flight

From repo root:

```bash
./lint.sh 2>&1 | tail -25
uv run pytest examples/medlit/tests/ tests/test_medlit_domain.py -v --tb=short 2>&1 | tail -60
```

Note any failures. After each step below, re-run the same and fix regressions.

---

## Step 1. Helper: extract authoritative ID from bundle entity

**File:** `examples/medlit/pipeline/dedup.py`

Add a function that, given an `ExtractedEntityRow` (or equivalent dict), returns the best authoritative ID if present, else `None`:

- **Signature:** `def _authoritative_id_from_entity(e: ExtractedEntityRow) -> Optional[str]:`
- **Logic:** Check in order: `e.canonical_id`, `e.umls_id`, `e.hgnc_id`, `e.rxnorm_id`, `e.uniprot_id`. Return the first non-empty string that *looks* authoritative (see Step 2). If none, return `None`.
- **Import:** Use `ExtractedEntityRow` from `examples.medlit.bundle_models` (already have `PerPaperBundle`; add `ExtractedEntityRow` to imports if needed).

Do not call any external API in this step; only read fields from the entity row.

---

## Step 2. Helper: is this string an authoritative ID?

**File:** `examples/medlit/pipeline/dedup.py`

Add a function used to avoid treating synthetic slugs as authoritative:

- **Signature:** `def _is_authoritative_id(s: str) -> bool:`
- **Logic:** Return `False` if `s` is empty or starts with `"canon-"`. Return `True` for known prefixes/patterns: MeSH (e.g. `D` + digits, or `MeSH:D` + digits), UMLS (`C` + digits), HGNC (`HGNC:` or numeric), RxNorm (`RxNorm:` or numeric), UniProt (`P`/`Q` + alphanumeric), DBPedia (`DBPedia:`). You can mirror the patterns used in `examples/medlit/pipeline/canonical_urls.py` (e.g. `build_canonical_url`) or keep a small list of prefixes: `("MeSH:", "C", "HGNC:", "RxNorm:", "UniProt:", "P", "Q", "DBPedia:")` and accept numeric-only for HGNC/RxNorm. Ensure `canon-` is never considered authoritative.

Use this in Step 1 when choosing which field from the entity row to return (only return a value that passes `_is_authoritative_id`).

---

## Step 3. Entity class to lookup type mapping

**File:** `examples/medlit/pipeline/dedup.py`

Add a mapping from bundle `entity_class` (e.g. `"Disease"`, `"Gene"`) to the string expected by `CanonicalIdLookup.lookup_canonical_id_sync(term, entity_type)`:

- **Signature:** `def _entity_class_to_lookup_type(entity_class: str) -> Optional[str]:`
- **Supported:** `Disease` → `"disease"`, `Gene` → `"gene"`, `Drug` → `"drug"`, `Protein` → `"protein"`. Optionally: `Biomarker` → `"disease"` (or `None` to skip lookup). Any other class → `None` (skip lookup).

Use lowercase; the authority lookup already normalizes with `.lower()`.

---

## Step 4. Pass 2: optional CanonicalIdLookup and new ID assignment logic

**File:** `examples/medlit/pipeline/dedup.py`

**4.1** Add parameters to `run_pass2`:

- `canonical_id_cache_path: Optional[Path] = None` — path to the JSON cache file for `CanonicalIdLookup`. If `None`, do not create a lookup; when assigning a new merge key for an entity that has no authoritative ID from the bundle and no entry in the synonym cache, use a synthetic slug only.
- No change to `synonym_cache_path` or other existing parameters.

**4.2** At the start of `run_pass2`, optionally create a `CanonicalIdLookup` instance:

- If `canonical_id_cache_path` is not `None`, instantiate `CanonicalIdLookup(cache_file=canonical_id_cache_path)`. Use the sync API only (`lookup_canonical_id_sync`). Do not use async; Pass 2 is currently synchronous. Ensure the lookup is closed when done (e.g. use a context manager if the class supports it, or call a close/cleanup method at the end of `run_pass2` if documented). If the class does not support context manager, call lookup synchronously and do not hold the instance beyond the function (check `authority_lookup.py` for lifecycle).

**4.3** Change `get_or_assign_canonical` so it can accept an optional `ExtractedEntityRow` (or optional precomputed authoritative_id) for the current entity:

- **Option A (recommended):** Change signature to `get_or_assign_canonical(paper_id, local_id, name, entity_class, entity_row=None)`. When `entity_row` is provided, first call `_authoritative_id_from_entity(entity_row)`. If that returns a string, use it as the merge key (and do not call lookup or slug). When `entity_row` is None or authoritative_id is None, keep current behavior: lookup synonym cache; if miss, then if lookup is available call `lookup.lookup_canonical_id_sync(name, _entity_class_to_lookup_type(entity_class))`; if that returns a string use it; else use `_canonical_id_slug()`.
- **Option B:** Before calling `get_or_assign_canonical` for an entity, compute `auth_id = _authoritative_id_from_entity(e)` and pass it in; then inside `get_or_assign_canonical` accept an optional `authoritative_id: Optional[str] = None` and use it first when present.

Ensure that whenever we use a synthetic slug, we never write it to an output field named `canonical_id` (see Step 5).

**4.4** Update all call sites of `get_or_assign_canonical` to pass the current entity row (or precomputed authoritative_id) when available. In the two loops that iterate over `bundle.entities`, you have access to `e` (ExtractedEntityRow); pass `e` (or `auth_id`) into `get_or_assign_canonical`.

**4.5** Internal state: keep using a single map from (name_lower, entity_class) and (paper_id, local_id) to the **merge key** (string). The merge key may be authoritative or synthetic. No change to synonym cache keying; cache still stores the merge key (which might now be MeSH:, HGNC:, etc., or canon-xxx).

---

## Step 5. Output schema: entity_id and canonical_id

**File:** `examples/medlit/pipeline/dedup.py`

In the section that builds `canonical_entities` (the dict that becomes `entities.json`):

- For each entity row written to the output, include:
  - **entity_id** (required): The merge key (authoritative ID or synthetic slug). Use this for relationships and for any internal reference.
  - **canonical_id** (optional): Set to the merge key **only if** `_is_authoritative_id(merge_key)` is true; otherwise set to `null` (or omit the key). So synthetic slugs never appear as `canonical_id`.
- Keep **class**, **name**, **synonyms**, **source**, **source_papers** as they are. Rename or add so that the JSON has both `entity_id` and `canonical_id` as above.
- When writing relationships to `relationships.json`, use `entity_id` (the merge key) for `subject` and `object` fields.

**Backward compatibility:** If downstream code or docs expect a single "canonical_id" field that is always present, you can keep outputting `entity_id` as the main identifier and `canonical_id` only when authoritative; document that `canonical_id` may be null. No need to remove `entity_id` from the schema.

---

## Step 6. SAME_AS merge: prefer authoritative ID as winner

**File:** `examples/medlit/pipeline/dedup.py`

When resolving SAME_AS (two entities merged into one), if one side has an authoritative merge key and the other has a synthetic one, choose the authoritative key as the winner (so the merged entity has a non-null canonical_id in output). Current code assigns one side’s ID to both; add a step: when `sub_id != obj_id`, if one of them is authoritative and the other is not, set the winner to the authoritative one (rewrite `local_to_canonical` and `name_type_to_canonical` so the authoritative ID is the one that remains). If both are authoritative or both synthetic, keep current behavior (e.g. keep sub_id as winner).

---

## Step 7. Pass 2 CLI: canonical-id-cache and no-lookup flag

**File:** `examples/medlit/scripts/pass2_dedup.py`

- Add `--canonical-id-cache` (type=Path, default=None): path to the canonical ID lookup cache file (e.g. `canonical_id_cache.json`). If provided, Pass 2 will use `CanonicalIdLookup` to resolve entities that do not already have an authoritative ID from the bundle or synonym cache.
- Add `--no-canonical-id-lookup` (flag, default=False): if set, do not perform authority lookups even if `--canonical-id-cache` is set (useful for quick runs or when network/API is unavailable). When `--no-canonical-id-lookup` is true, ignore `--canonical-id-cache` for lookup purposes (still allow specifying cache for future use if desired; or simply disable lookup when flag is set).
- Pass `canonical_id_cache_path` into `run_pass2` (add the parameter to `run_pass2` as in Step 4 and wire it from the parser). When `--no-canonical-id-lookup` is True, pass `None` for the cache path so no lookup is created.

---

## Step 8. CanonicalIdLookup lifecycle in Pass 2

**File:** `examples/medlit/pipeline/dedup.py`

`CanonicalIdLookup` has an async `close()` and no sync context manager. When using only `lookup_canonical_id_sync`, the async HTTP client is not used; the sync path uses a temporary `httpx.Client` per call. At the end of `run_pass2`, if a lookup instance was created, call `lookup._save_cache(force=True)` so the persistent cache file is updated (e.g. in a `try`/`finally` block). Do not call `close()` from sync code. Add a one-line comment in code: "Save lookup cache so results persist across runs."

---

## Step 9. Tests

**File:** `examples/medlit/tests/test_two_pass_ingestion.py` (or a new `test_pass2_canonical_ids.py`)

- Add a test that runs Pass 2 with a small fixture bundle where one entity has `umls_id` or `canonical_id` set to an authoritative value (e.g. `C0006142`). Assert that the merged entities JSON contains that entity with `entity_id` equal to the authoritative ID and `canonical_id` equal to the same (non-null).
- Add a test that runs Pass 2 with a bundle where an entity has no authoritative ID and no synonym cache entry, and either (a) no lookup is used, or (b) a mock lookup is used. Assert that the entity appears with a synthetic `entity_id` (e.g. starting with `canon-`) and `canonical_id` is null.
- If you added `_is_authoritative_id`, add a unit test for it: assert true for `C0006142`, `HGNC:1100`, `MeSH:D001943`, false for `canon-abc123`, empty string.

Update existing tests that assert on `canonical_id` so they expect either the new output shape (entity_id + optional canonical_id) or the old shape, depending on what you keep; avoid regressions.

---

## Step 10. Docs

**File:** `examples/medlit/CANONICAL_IDS.md`

- Add a short section "Pass 2 (two-pass pipeline)" that states: Pass 2 can optionally use the same `CanonicalIdLookup` and cache file. Use `--canonical-id-cache path/to/cache.json` to enable. Pass 2 output uses `entity_id` as the stable merge key and `canonical_id` only when the entity has an authoritative ontology ID; otherwise `canonical_id` is null. Point to `--no-canonical-id-lookup` for runs without network/API.

**File:** `examples/medlit/INGESTION.md` (or README)

- Mention that Pass 2 supports `--canonical-id-cache` for authoritative canonical IDs and that output distinguishes `entity_id` (always present) from `canonical_id` (present only when from an ontology).

---

## Step 11. Verification

From repo root:

```bash
# Unit/ingestion tests
uv run pytest examples/medlit/tests/ -v --tb=short

# Optional: run Pass 1 + Pass 2 with a single paper and --canonical-id-cache
# (requires network and optionally UMLS_API_KEY for diseases)
uv run python -m examples.medlit.scripts.pass1_extract --input-dir examples/medlit/pmc_xmls --output-dir /tmp/p8a_pass1 --llm-backend ollama --papers "PMC12756687.xml" --limit 1
uv run python -m examples.medlit.scripts.pass2_dedup --bundle-dir /tmp/p8a_pass1 --output-dir /tmp/p8a_pass2 --canonical-id-cache /tmp/p8a_cache.json
# Inspect /tmp/p8a_pass2/entities.json: entity_id and canonical_id should follow the rules above.
```

Run lint and fix any issues:

```bash
./lint.sh 2>&1 | tail -30
```

---

## Summary checklist

- [ ] Phase 1: Docs (INGESTION.md, CANONICAL_IDS.md, run-ingest.sh, ingest.py docstring) — new pipeline doesn’t use promotion; old ingest labeled legacy
- [ ] Step 0: Pre-flight
- [ ] Step 1: `_authoritative_id_from_entity(e)` in dedup.py
- [ ] Step 2: `_is_authoritative_id(s)` in dedup.py
- [ ] Step 3: `_entity_class_to_lookup_type(entity_class)` in dedup.py
- [ ] Step 4: `run_pass2(..., canonical_id_cache_path=...)`, optional CanonicalIdLookup, new ID assignment (bundle → cache → lookup → slug), pass entity row into get_or_assign_canonical
- [ ] Step 5: Output entities have entity_id (merge key) and canonical_id (only when authoritative)
- [ ] Step 6: SAME_AS merge prefers authoritative ID as winner
- [ ] Step 7: pass2_dedup CLI: --canonical-id-cache, --no-canonical-id-lookup
- [ ] Step 8: CanonicalIdLookup lifecycle (save cache, close) in run_pass2
- [ ] Step 9: Tests for authoritative vs synthetic IDs and for _is_authoritative_id
- [ ] Step 10: CANONICAL_IDS.md and INGESTION.md/README updated
- [ ] Step 11: Verification and lint
