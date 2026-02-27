# PLAN11: Pass 1a vocabulary seeding + UMLS type validation

**Goal:** Reduce cross-paper entity duplication and LLM type misclassification by (1) running a fast lightweight extraction across all papers before full Pass 1b extraction, seeding a shared vocabulary that Pass 1b uses to normalize entity names and types; and (2) adding a UMLS semantic type cross-check that catches misclassifications like "cortisol typed as gene" programmatically.

**Background:** PLAN11 is independent of any other plan. (A former IMPLEMENTATION_PLAN.md for semantic evidence validation targeted the obsolete MedLitRelationshipExtractor; it was removed. The kernel idea—optional evidence validation as post-processing—is noted in INGESTION.md § "Future work: evidence validation".)

---

## Problem statement

Current Pass 1 extracts each paper in isolation. The LLM has no knowledge of entities seen in other papers, so:

- The same drug (pasireotide, osilodrostat, desmopressin) gets a fresh entity record per paper, with no synonym linkage across papers.
- Entity types are inconsistent across papers (biologicalprocess vs biological_process vs biological process) because there is no shared type vocabulary enforced at extraction time.
- Type misclassifications (cortisol as gene, ACTH as gene) are not caught until a human audits the graph.

Pass 2 dedup is supposed to merge these, but it can only merge what the synonym cache already knows. If two papers both introduce "pasireotide" as a new entity with no prior synonym cache entry, Pass 2 sees two new entities with no overlap signal and creates two records.

---

## Design

### Pass 1a: fast vocabulary extraction

A new script `examples/medlit/scripts/pass1a_vocab.py` that:

1. Iterates over all input XML files (same `--input-dir` as Pass 1b).
2. For each paper, sends a **cheap structured prompt** to the LLM asking only for: entity names, entity types (from a fixed enum), and any abbreviations or synonyms mentioned in the paper. No relationships, no evidence, no section structure.
3. Collects all results into a shared **vocabulary file** (`pass1_vocab/vocab.json`).
4. Runs UMLS type validation on any entity with a canonical_id (see §UMLS below) and flags or corrects misclassified types.
5. Writes the final vocabulary as a pre-seeded synonym cache (`pass1_vocab/seeded_synonym_cache.json`) in the same format Pass 2 expects for `--synonym-cache`.

Pass 1a is embarrassingly parallel — each paper's prompt is independent, no shared state needed during extraction. Output is merged after all papers complete.

**LLM prompt design for Pass 1a:**

Simpler and cheaper than Pass 1b. Something like:

```
Extract all named biomedical entities from this paper.
For each entity return:
  - name: canonical form (not an abbreviation)
  - type: one of [disease, gene, drug, protein, mutation, symptom, biomarker,
                  pathway, procedure, biologicalprocess, anatomicalstructure,
                  clinicaltrial, institution, author, studydesign,
                  statisticalmethod, adverseevent, hypothesis]
  - abbreviations: list of abbreviations or alternate names used in this paper
  - umls_id: UMLS CUI if you are confident, else null

Return JSON array only.
```

The fixed type enum is the key constraint. Pass 1b currently gets this wrong because the enum is not enforced in its prompt. Enforcing it in Pass 1a and then passing the vocabulary into Pass 1b gives Pass 1b a concrete list to match against rather than free-form string generation.

**Vocabulary file format (`vocab.json`):**

```json
[
  {
    "name": "pasireotide",
    "type": "drug",
    "abbreviations": ["SOM230"],
    "umls_id": "C2975503",
    "source_papers": ["PMC11548364", "PMC11560769", "PMC11779774"],
    "umls_type_validated": true,
    "umls_type_conflict": null
  },
  ...
]
```

**Seeded synonym cache format:** Same structure as Pass 2's synonym cache (see `examples/medlit/pipeline/synonym_cache.py`): a JSON object keyed by normalized (lowercased) entity name; each value is a list of entries with `entity_a`, `entity_b`, `resolution`, `confidence`, etc. To seed: for each vocab entry with `name`, `type`, and (when present) `umls_id` or other canonical id, add one cache entry so that `lookup_entity(cache, name, type)` returns that canonical_id. Concretely: for each vocab entity, set `cache[normalize(name)] = [{"entity_a": {"name": name, "class": type, "canonical_id": umls_id or null}, "entity_b": <same>, "resolution": "merged", "confidence": 1.0, "asserted_by": "pass1a", "source_papers": source_papers}]`. Pass 2's `--synonym-cache` argument points to this file; Pass 2 loads from it and writes back to the same path (so the seed is updated with Pass 2 merges).

---

### Pass 1b: full extraction with vocabulary context

Pass 1b is the existing `pass1_extract.py` with two additions:

1. **Vocabulary context in prompt:** If `--vocab-file` is provided and the file exists, load the vocabulary (list of `{name, type, abbreviations, umls_id, ...}`). Build the system prompt by appending a section: *"The following entities have already been identified across the corpus. Use these exact names and types where applicable rather than creating new variants."* followed by a compact list (e.g. one line per entity: `name (type)` or a small JSON slice). If the vocab is large (>500 entries), include only a subset (e.g. first 300 by name) or all if small; no need for MeSH/keyword filtering in the initial implementation. The prompt is built in `pass1_extract.py` (where `_default_system_prompt()` lives); add a helper e.g. `_build_system_prompt_with_vocab(base_prompt: str, vocab_entries: list[dict] | None) -> str` and use it when `vocab_file` is set.

2. **Type normalization post-processing:** After the LLM returns entities, normalize each entity's `class` (type) against the fixed enum before writing the Pass 1b bundle. The bundle and dedup expect **exact** entity_class strings: `lookup_entity` and `_entity_class_to_lookup_type` in `dedup.py` compare `entity_a.get("class") == entity_class`, and dedup uses `(name_lower, entity_class)` as a key. The codebase uses **PascalCase** (each word capitalized). See `_entity_class_to_lookup_type` in `examples/medlit/pipeline/dedup.py` for the five types that get authority lookup: `"Disease"`, `"Gene"`, `"Drug"`, `"Protein"`, `"Biomarker"`. Other types in the enum must use the same convention: e.g. `"BiologicalProcess"` (not `"Biologicalprocess"`), `"AnatomicalStructure"`, `"AdverseEvent"`. Define a single canonical enum (lowercase) and a **fixed mapping to bundle class** as follows (use this exact mapping in the normalizer):

   | Normalized (lowercase) | Bundle `class` (output) |
   |------------------------|-------------------------|
   | disease | Disease |
   | gene | Gene |
   | drug | Drug |
   | protein | Protein |
   | biomarker | Biomarker |
   | symptom | Symptom |
   | procedure | Procedure |
   | mutation | Mutation |
   | pathway | Pathway |
   | biologicalprocess | BiologicalProcess |
   | anatomicalstructure | AnatomicalStructure |
   | clinicaltrial | ClinicalTrial |
   | institution | Institution |
   | author | Author |
   | studydesign | StudyDesign |
   | statisticalmethod | StatisticalMethod |
   | adverseevent | AdverseEvent |
   | hypothesis | Hypothesis |

   Normalize input: strip spaces, lowercase, collapse spaces/underscores (e.g. "biological process" → "biologicalprocess", "biological_process" → "biologicalprocess"); then map via the table above. Unknown types: map to `"Other"` so the pipeline does not see free-form strings.

No other changes to Pass 1b logic. Relationships, evidence, section structure all unchanged.

---

### UMLS type validation

After Pass 1a collects entities with `umls_id`, cross-check the assigned `type` against the UMLS semantic type for that CUI.

**Implementation:**

The existing `examples/medlit/pipeline/authority_lookup.py` handles UMLS lookups. Add a **sync** function (so Pass 1a can call it without async):

```python
def validate_umls_type(umls_id: str, assigned_type: str) -> tuple[bool, str | None]:
    """Return (ok, correct_type_if_known).

    Looks up the UMLS semantic type group for umls_id (e.g. via UMLS REST API
    concept endpoint: https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}
    or /rest/content/current/CUI/{cui}/atoms and semantic types). Returns
    (False, expected_type) when assigned_type is incompatible with the UMLS
    semantic type; (True, None) when compatible; (False, None) when ambiguous.
    """
```

Use the UMLS API key from the same source as `CanonicalIdLookup` (env `UMLS_API_KEY`). If the API is unavailable or the CUI has no semantic types, return `(True, None)` (no correction). For unit tests, use a small hardcoded dict mapping CUI → semantic type group so tests do not require a live API.

**Caching:** The sync API call can be slow (one network round-trip per entity with a CUI). Pass 1a may process many papers and entities; without caching, 50 entities with umls_ids would mean 50 serial sync calls. The implementation **must** cache results by CUI for the duration of the Pass 1a run (e.g. a module-level or function-scope dict: `_umls_type_cache: dict[str, tuple[bool, str | None]]` keyed by `(umls_id, assigned_type)` or just `umls_id` if the return value is independent of assigned_type). Optionally persist the cache to disk (e.g. alongside `vocab.json`) for cross-run reuse; at minimum, use an in-memory cache so each CUI is looked up at most once per run.

The mapping needed is coarse — UMLS semantic type groups map cleanly onto the entity type enum:

| UMLS semantic type group | Compatible entity types |
|--------------------------|------------------------|
| Pharmacologic Substance, Clinical Drug | drug |
| Disease or Syndrome, Finding | disease, symptom, adverseevent |
| Gene or Genome | gene |
| Amino Acid, Peptide, Protein | protein, biomarker |
| Biologic Function | biologicalprocess |
| Body Part, Organ | anatomicalstructure |
| Diagnostic Procedure, Therapeutic Procedure | procedure |
| Laboratory Procedure | procedure, biomarker |
| Quantitative Concept | biomarker |

This mapping can be a small hardcoded dict — it doesn't need to be exhaustive for the common misclassification cases.

**Behavior on conflict:**

- Log a warning with entity name, umls_id, assigned type, and expected type.
- Correct the type automatically if the UMLS mapping is unambiguous (e.g. C0020268 cortisol → semantic type "Pharmacologic Substance" → type should be `drug` or `biomarker`, not `gene`).
- If ambiguous (entity could be gene or biomarker depending on context), flag in vocab output as `umls_type_conflict: "gene or biomarker"` and leave the LLM's assignment in place for human review.

**Where it runs:** Inside `pass1a_vocab.py`, after LLM extraction, before writing `vocab.json`. Not in Pass 1b (Pass 1b inherits corrected types from the vocab).

---

## File changes

| File | Change |
|------|--------|
| `examples/medlit/scripts/pass1a_vocab.py` | **New script.** CLI: `--input-dir`, `--output-dir` (writes `vocab.json` + `seeded_synonym_cache.json`), `--llm-backend`, `--papers` (optional filter). If `output_dir/vocab.json` already exists, load it, run extraction for the current batch of papers, **merge** new entities into the vocab (same name+type → add to `source_papers`), then write. Expose a function `run_pass1a(input_dir, output_dir, llm_backend, papers=None)` so the ingest worker can call it. |
| `examples/medlit/scripts/pass1_extract.py` | Add `--vocab-file` (optional `Path`). If provided and file exists: load vocab (list of dicts), call `_build_system_prompt_with_vocab(_default_system_prompt(), vocab_entries)` to build the system prompt, and pass it to the LLM. After parsing LLM response, run type normalization on each entity's `class` (see § Pass 1b) before building `ExtractedEntityRow` and writing the bundle. Extend `run_pass1(..., vocab_file: Optional[Path] = None)` to accept and use vocab_file. |
| `examples/medlit/pipeline/authority_lookup.py` | Add `validate_umls_type(umls_id: str, assigned_type: str) -> tuple[bool, str | None]` (sync) and a module-level mapping from UMLS semantic type group → allowed entity types. Use UMLS REST API when API key is set; otherwise return `(True, None)`. **Cache results by CUI** for the duration of the run (in-memory dict at minimum) so each CUI is looked up at most once; optional disk cache for cross-run reuse. |
| `run-ingest.sh` | Add Pass 1a step before Pass 1b. After Pass 1a, run Pass 1b with `--vocab-file pass1_vocab/vocab.json`. Pass `--synonym-cache pass1_vocab/seeded_synonym_cache.json` to Pass 2. |
| `kgserver/mcp_server/ingest_worker.py` | In `_ensure_workspace_dirs`, add and create `vocab_dir = root / "pass1_vocab"`, and return a 4-tuple `(bundles_dir, merged_dir, output_dir, vocab_dir)`. In `_run_ingest_job_impl`, unpack the four values. Before `run_pass1`, call `run_pass1a(input_dir, vocab_dir, llm_backend, papers=None)` so the single paper's vocab is merged into the persistent vocab; then call `run_pass1(..., vocab_file=vocab_dir / "vocab.json")`. In `_run_pass2_pass3_load`, accept `workspace_root` and set `synonym_cache_path = workspace_root / "pass1_vocab" / "seeded_synonym_cache.json"` if that file exists, else `merged_dir / "synonym_cache.json"`, and pass it to `run_pass2(..., synonym_cache_path=...)`. |
| `examples/medlit/INGESTION.md` | Document the four-pass flow (1a → 1b → 2 → 3), the vocabulary file format (`vocab.json`), and that `--vocab-file` and `--synonym-cache` point to `pass1_vocab/` when using Pass 1a. |

---

## Implementation order (recommended)

1. **authority_lookup.py:** Add UMLS semantic type mapping dict and `validate_umls_type(umls_id, assigned_type)`. Add unit tests (mock mapping, no live API).
2. **pass1a_vocab.py:** New script and `run_pass1a(...)`: iterate papers, cheap LLM prompt, merge into vocab.json, run UMLS type validation on entities with umls_id, write seeded_synonym_cache.json. Merge-with-existing when output_dir already has vocab.json.
3. **pass1_extract.py:** Add `_build_system_prompt_with_vocab`, type normalizer (raw type string → bundle `class`), `--vocab-file` and `run_pass1(..., vocab_file=...)`. Unit test prompt and normalizer.
4. **run-ingest.sh:** Insert Pass 1a step; add `--vocab-file` to Pass 1b and `--synonym-cache` to Pass 2.
5. **ingest_worker.py:** Extend `_ensure_workspace_dirs` with vocab_dir; call run_pass1a then run_pass1 with vocab_file; use vocab_dir for synonym_cache_path in Pass 2 when file exists.
6. **INGESTION.md:** Doc updates as in the file table.

---

## Updated `run-ingest.sh` flow

```bash
# Pass 1a: fast vocabulary extraction across all papers
uv run python -m examples.medlit.scripts.pass1a_vocab \
  --input-dir examples/medlit/pmc_xmls \
  --output-dir pass1_vocab \
  --llm-backend anthropic \
  --papers $PAPER

# Pass 1b: full extraction with vocabulary context
uv run python -m examples.medlit.scripts.pass1_extract \
  --input-dir examples/medlit/pmc_xmls \
  --output-dir pass1_bundles \
  --llm-backend anthropic \
  --vocab-file pass1_vocab/vocab.json \
  --papers $PAPER

# Pass 2: dedup, seeded with pass1a synonym cache
uv run python -m examples.medlit.scripts.pass2_dedup \
  --bundle-dir pass1_bundles \
  --output-dir medlit_merged \
  --synonym-cache pass1_vocab/seeded_synonym_cache.json

# Pass 3: build bundle (unchanged)
uv run python -m examples.medlit.scripts.pass3_build_bundle \
  --merged-dir medlit_merged \
  --bundles-dir pass1_bundles \
  --output-dir medlit_bundle
```

**Incremental behavior:** Pass 1a is idempotent per paper (same as Pass 1b). For incremental runs, re-run Pass 1a over all papers (including new ones) to rebuild the vocabulary, then Pass 1b (skips papers that already have a bundle). Alternatively, Pass 1a can be made incremental later (e.g. skip papers whose IDs are already in `vocab.json` source_papers).

---

## Persistent workspace additions (PLAN10 integration)

In `kgserver/mcp_server/ingest_worker.py`, extend `_ensure_workspace_dirs` to create and return the vocab dir:

```python
vocab_dir = root / "pass1_vocab"
vocab_dir.mkdir(parents=True, exist_ok=True)
# Return (bundles_dir, merged_dir, output_dir, vocab_dir) — caller must unpack four values.
```

**Breaking change to existing API:** The return type changes from a 3-tuple to a 4-tuple. Any existing call site (e.g. `_run_ingest_job_impl`, which currently unpacks `bundles_dir, merged_dir, output_dir`) must be updated to unpack four values. Compatibility with the current codebase is the priority: update all call sites in this codebase when making the change.

Update the return type and all call sites (e.g. `_run_ingest_job_impl`) to unpack `vocab_dir`. Pass 1a runs before Pass 1b in `_run_ingest_job_impl`, outside the lock (same as Pass 1b). The merge step in `pass1a_vocab.py` (reading existing vocab.json, merging new entities, writing vocab.json and seeded_synonym_cache.json) should be atomic: write to a temp file in the same directory, then rename to the final path, so concurrent workers do not corrupt the file. Alternatively, run the Pass 1a merge under the same workspace lock if it is quick.

---

## Test strategy

1. **Unit test for `validate_umls_type`** (in `examples/medlit/tests/`): Mock or inject a small CUI → semantic type mapping (e.g. C0020268 → "Hormone" or "Amino Acid, Peptide, or Protein") so tests do not call the UMLS API. Assert cortisol (C0020268) with assigned type `gene` returns `(False, "drug")` or `(False, "biomarker")` per the mapping table; pasireotide (or a test CUI mapped to "Pharmacologic Substance") with type `drug` returns `(True, None)`. Test unknown CUI returns `(True, None)` (no correction).

2. **Unit test for vocab-in-prompt and type normalization:** In `pass1_extract.py`, the helper `_build_system_prompt_with_vocab(base, vocab_entries)` — test that when `vocab_entries` is a short list (e.g. 2 entities), the returned prompt contains the entity names and types. For type normalization: test a function that maps raw LLM type strings to bundle `class` (e.g. `"biological process"` → `"BiologicalProcess"`, `"gene"` → `"Gene"`). Prefer a dedicated normalizer function so it can be unit-tested without calling the LLM.

3. **Integration test** (optional, in `examples/medlit/tests/`): Run Pass 1a over two small fixture papers that both mention the same entity (e.g. pasireotide). Assert `vocab.json` contains one entry for that entity with both paper IDs in `source_papers`. Run Pass 1b with `--vocab-file pass1_vocab/vocab.json` on the same papers; assert the output bundles use the same entity name and type for that entity.

---

## Success criteria

- After Pass 1a + 1b + 2 + 3 over the Cushing's paper set: pasireotide appears as one entity, not three. Osilodrostat appears as one entity, not two.
- Cortisol is not typed as `gene` in any Pass 1b bundle.
- Entity type strings in all bundles match the fixed enum exactly (no spaces, no underscores where not expected).
- Pass 1a runtime: measure and document (e.g. "Pass 1a added X% to total pipeline runtime"). Target is roughly ≤20% increase since the prompt is smaller and has no relationship extraction; treat as an observable to tune, not a hard pass/fail gate.

---

## Out of scope

- Batching Pass 1a prompts across multiple papers in a single LLM call (future optimization).
- Cross-domain vocabulary sharing (e.g. sharing entity vocabulary between medlit and a legal domain).
- Replacing Pass 2 dedup with embedding-based entity resolution (a larger separate change).