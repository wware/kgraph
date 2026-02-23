# Plan: MedLit Ingestion Refactor (PLAN8)

Execute steps in order from the **repository root**. Reference: **examples/medlit/INGESTION_REFACTOR.md**.

**Scope:** Implement the two-pass ingestion process described in INGESTION_REFACTOR.md: (1) Pass 1 — LLM extraction producing immutable per-paper bundle JSON; (2) Pass 2 — deduplication and promotion with name/type index, SAME_AS resolution, canonical ID assignment, ontology lookup, and relationship ref updates. Add the SAME_AS predicate to the schema and ensure bundle format, provenance, and Evidence handling match the spec.

---

## Step 0. Pre-flight

From repo root:

```bash
./lint.sh 2>&1 | tail -20
uv run pytest examples/medlit/tests/ -v --tb=short 2>&1 | tail -50
```

Note any failures. After each step below, re-run the same and fix regressions.

---

## Step 1. Schema: predicate consistency (SUBTYPE_OF, SAME_AS, INDICATES)

**Goal:** Ensure SUBTYPE_OF is defined and used everywhere like the other predicates, and add SAME_AS (and INDICATES if missing) in one schema pass. Same pattern for each: add to PredicateType, add class, add to RELATIONSHIP_TYPE_MAP, add to domain. One commit, one pre-flight re-run.

**1. SUBTYPE_OF (verify or add everywhere)**

- **`examples/medlit_schema/base.py`** — `PredicateType` must include `SUBTYPE_OF = "subtype_of"`. Confirm it exists.
- **`examples/medlit_schema/relationship.py`** — `SubtypeOf(BaseMedicalRelationship)` must exist with `get_edge_type()` returning `"SUBTYPE_OF"`. Add to **`RELATIONSHIP_TYPE_MAP`** if missing: `"SUBTYPE_OF": SubtypeOf`.
- **`examples/medlit_schema/domain.py`** — In `relationship_types`, include `"SUBTYPE_OF": SubtypeOf`. In **`predicate_constraints`**, add `"SUBTYPE_OF": PredicateConstraint(subject_types={"disease"}, object_types={"disease"})` if missing.
- **`examples/medlit/vocab.py`** — `predicate_subtype_of = "subtype_of"` in `ALL_PREDICATES` and in `get_valid_predicates()` for the relevant type pairs (e.g. disease → disease).

**2. SAME_AS (add)**

- **`examples/medlit_schema/base.py`** — In `PredicateType`, add `SAME_AS = "same_as"` (e.g. after SUBTYPE_OF or in an "Identity" section).
- **`examples/medlit_schema/relationship.py`** — Add `SameAs` class (after `PartOf`, before Hypothesis section). Subclass `ResearchRelationship` (no evidence_ids required). Fields: `confidence`, `resolution` (Optional[Literal["merged", "distinct"]]), `note` (Optional[str]). `get_edge_type()` returns `"SAME_AS"`. Add to **`RELATIONSHIP_TYPE_MAP`:** `"SAME_AS": SameAs`.
- **`examples/medlit_schema/domain.py`** — Add `SameAs` to imports; add `"SAME_AS": SameAs` to `relationship_types`; add `"SAME_AS"` to `predicate_constraints` (e.g. subject_types and object_types that allow any entity pair, or document as "any → any" per spec).

**3. INDICATES (if missing)**

- If `RELATIONSHIP_TYPE_MAP` does not already include `"INDICATES"`: add `Indicates` class (subclass of `BaseMedicalRelationship`, `get_edge_type()` returns `"INDICATES"`), add to `RELATIONSHIP_TYPE_MAP`, and ensure `PredicateType.INDICATES` and domain/constraints exist.

**Verification:**

```bash
uv run python -c "
from examples.medlit_schema.base import PredicateType
from examples.medlit_schema.relationship import RELATIONSHIP_TYPE_MAP, SubtypeOf, SameAs, create_relationship
from examples.medlit_schema.domain import MedlitDomain
domain = MedlitDomain()
# SUBTYPE_OF
assert PredicateType.SUBTYPE_OF.value == 'subtype_of'
assert 'SUBTYPE_OF' in RELATIONSHIP_TYPE_MAP and RELATIONSHIP_TYPE_MAP['SUBTYPE_OF'] is SubtypeOf
assert 'SUBTYPE_OF' in domain.relationship_types and 'SUBTYPE_OF' in domain.predicate_constraints
r = create_relationship('SUBTYPE_OF', 'e01', 'e02', evidence_ids=['ev1'], source_papers=['PMC123'])
assert r.get_edge_type() == 'SUBTYPE_OF'
# SAME_AS
assert PredicateType.SAME_AS.value == 'same_as'
assert 'SAME_AS' in RELATIONSHIP_TYPE_MAP and RELATIONSHIP_TYPE_MAP['SAME_AS'] is SameAs
r2 = create_relationship('SAME_AS', 'e01', 'e02', confidence=0.5, note='test')
assert r2.get_edge_type() == 'SAME_AS'
print('OK')
"
```

Re-run Step 0 pre-flight.

---

## Step 2. Define per-paper bundle Pydantic model

**Goal:** Define a Pydantic model that matches the per-paper bundle JSON structure in INGESTION_REFACTOR.md so Pass 1 output and Pass 2 input are strictly typed.

**New file:** `examples/medlit/bundle_models.py`

**Content:**

- **PaperInfo** (or reuse from schema if exists): `doi`, `pmcid`, `title`, `authors`, `journal`, `year`, `study_type`, `eco_type` (optional).
- **ExtractionProvenance** (re-export or alias from `examples.medlit_schema.base`): ensure it includes `models`, `extraction_pipeline`, `prompt`, `execution`, `entity_resolution` (all optional where appropriate).
- **ExtractedEntityRow**: minimal entity record for bundle: `id`, entity type (see note below), `name`, optional `synonyms`, `symbol` (gene), `brand_names` (drug), etc., `source` (literal `"extracted"` for Pass 1), `canonical_id` (optional), `umls_id`/`hgnc_id`/`rxnorm_id`/etc. optional.
- **EvidenceEntityRow**: `id` (canonical format `{paper_id}:{section}:{paragraph_idx}:{method}`), entity type (see note), `entity_id`, `paper_id`, `text_span_id`, `text` (optional), `confidence`, `extraction_method`, `study_type`, `eco_type`, `source`.
- **RelationshipRow**: `subject`, `predicate`, `object`, `evidence_ids` (optional for SAME_AS), `source_papers`, `confidence`, `properties`, `section`, `asserted_by`, optional `resolution`, `note` for SAME_AS.
- **PerPaperBundle**: `paper`, `extraction_provenance`, `entities` (list of ExtractedEntityRow), `evidence_entities` (list of EvidenceEntityRow), `relationships` (list of RelationshipRow), `notes` (list of str).

**Note — `class` is a Python reserved word:** The spec uses a key `"class"` for entity type (e.g. `"Disease"`, `"Gene"`, `"Evidence"`). In Pydantic, do not name the attribute `class`. Use a field name such as `entity_class` or `kind`, and add `Field(alias="class")` plus `model_config = ConfigDict(populate_by_name=True)` so that (a) the model accepts the keyword in Python and (b) JSON serialization/deserialization uses the key `"class"` to match the spec. Apply this pattern in both ExtractedEntityRow and EvidenceEntityRow where entity type is stored.

Use `model_json_schema()` / `model_dump()` for serialization. Keep bundle JSON keys snake_case to match the spec.

**Verification:**

- Instantiate a minimal `PerPaperBundle` with one entity, one evidence, one relationship, and one note; call `model_dump()` and assert structure matches the spec example (paper, extraction_provenance, entities, evidence_entities, relationships, notes).
- Re-run Step 0 pre-flight.

---

## Step 3. Pass 1: LLM extraction → per-paper bundle JSON (single file per paper)

**Goal:** Refactor the ingestion flow so that after the LLM extraction step, the output is one JSON file per paper (e.g. `paper_{pmcid}.json` or `{pmcid}.bundle.json`) containing the per-paper bundle structure. The file is the single source of truth for that paper; it is **never** overwritten by Pass 2 (Pass 2 writes overlays or a separate promoted copy).

**Inputs:** Papers in any supported format (JATS-XML, HTML, LaTeX, PDF). Pass 1 requires an LLM; see **Pass 1: LLM backends and configuration** below for how to use Lambda Labs, Anthropic (Claude), OpenAI (ChatGPT), or similar.

**Process:**

1. For each paper: load content (JATS-XML, HTML, LaTeX, or PDF). For long papers, optionally split on structural markers (e.g. JATS `<sec>`, LaTeX `\section{}`, HTML `<h2>`).
2. Call the LLM with the system prompt and paper content (one call per paper or per section); request valid JSON only (no markdown fences).
3. Parse the response into a structure that maps to `PerPaperBundle` (entities with `source="extracted"`, evidence_entities with canonical IDs `{paper_id}:{section}:{paragraph_idx}:{method}`, relationships with `evidence_ids` for medical assertions, optional SAME_AS with `resolution=null`).
4. Build `extraction_provenance`: `models.llm` (name/version), `extraction_pipeline` (name, version, git_commit, git_commit_short, git_branch, git_dirty, repo_url), `prompt` (version, template, checksum), `execution` (timestamp ISO 8601 UTC, hostname, python_version, duration_seconds). Leave `entity_resolution` null in Pass 1. (Step 5 is an explicit audit to confirm provenance and Evidence are complete — implement what you can here; polish in Step 5.)
5. Write one file per paper to an output directory: e.g. `out/paper_PMC12756687.json`, using `PerPaperBundle.model_dump()` (or equivalent) so the JSON matches the spec.

**Files to create or modify:**

- **New or refactored script:** e.g. `examples/medlit/scripts/pass1_extract.py` (or refactor `scripts/ingest.py` to have a `--pass1-only` mode that writes per-paper bundles instead of running promotion/relationship stage).
- **System prompt:** Ensure the prompt includes entity type taxonomy, predicate vocabulary, direction conventions (e.g. TREATS = Drug→Disease), and per-entity/per-relationship JSON schema (from `PerPaperBundle` or medlit_schema Pydantic `model_json_schema()`). Include guidance for SAME_AS (when to emit, confidence bands) and for confidence scoring (definitive 0.9–1.0, hedged 0.4–0.6, etc.).
- **Evidence canonical ID:** When creating Evidence entities, use format `{paper_id}:{section}:{paragraph_idx}:{method}`; use `SectionType` values for section (abstract, introduction, methods, results, discussion, conclusion).

**Verification:**

- Run Pass 1 on one small paper (e.g. one of the PMC XMLs in `examples/medlit/pmc_xmls/`). Inspect the produced JSON: must have `paper`, `extraction_provenance`, `entities` (all `source="extracted"`), `evidence_entities`, `relationships`, `notes`. Evidence IDs in relationships must reference IDs in `evidence_entities`.
- **This is a manual verification step requiring a configured LLM backend** — there is no pytest command for it; Step 7 covers CI-safe tests (Pass 2 only, pre-baked fixtures).
- Re-run Step 0 pre-flight.

---

### Pass 1: LLM backends and configuration

Pass 1 needs an LLM to extract entities and relationships from papers. The plan must document how users run Pass 1 with different backends so they can choose the one they have (e.g. Anthropic API key, Lambda Labs instance, or OpenAI).

**Deliverable:** A user-facing section (in `examples/medlit/README.md` or `examples/medlit/INGESTION.md` or a dedicated `examples/medlit/LLM_SETUP.md`) that covers:

1. **Anthropic (Claude) — API key**
   - User sets `ANTHROPIC_API_KEY` (e.g. in `.env` or environment). Do not document committing the key; recommend `.env` and adding `.env` to `.gitignore` if not already.
   - How to run Pass 1 with Claude: e.g. `--llm-backend anthropic` or `LLM_BACKEND=anthropic`, and optionally `--anthropic-model MODEL` or `ANTHROPIC_MODEL=claude-sonnet-4-20250514` (or current default).
   - Point to Anthropic docs for API keys and rate limits if needed.

2. **Lambda Labs (or other self-hosted / OpenAI-compatible endpoint)**
   - User runs an LLM on a Lambda Labs instance (or similar) that exposes an OpenAI-compatible API. Document the environment variables or flags: e.g. `OPENAI_API_BASE_URL` (or `LAMBDA_BASE_URL`) and `OPENAI_API_KEY` (or `LAMBDA_API_KEY`) if required by the endpoint.
   - How to run Pass 1 with this backend: e.g. `--llm-backend openai` or `LLM_BACKEND=openai` and set base URL and key so the client points at the instance.
   - Note that this was the previous setup (“as before”) for users who already have a Lambda instance.

3. **OpenAI (ChatGPT)**
   - User sets `OPENAI_API_KEY`. How to run Pass 1 with OpenAI: e.g. `--llm-backend openai` and optionally `--openai-model gpt-4o` or equivalent.

4. **Single place that lists backends**
   - One table or list: Backend name | Env vars / flags | When to use (e.g. “Anthropic API key on hand”, “Lambda instance already running”, “OpenAI only”).

**Implementation note:** Pass 1 script or config should accept a backend selector (env or CLI) and instantiate the appropriate client (Anthropic SDK, OpenAI SDK, or custom client for Lambda). Keep API keys out of code and in environment or a secrets mechanism.

---

## Step 4. Pass 2: Deduplication, promotion, and synonym cache

**Goal:** Implement the deduplication and promotion pass that reads all per-paper bundle JSONs and produces a merged graph with canonical IDs. Include the **synonym cache** as part of this step — it is how Pass 2 persists across runs and stays idempotent; implement load/save/lookup in Pass 2 from the start rather than adding it later.

**Input:** Directory of per-paper bundle JSON files (output of Pass 1).

**Process (implement in order):**

1. **Load synonym cache (if present).** On startup, load the synonym cache file (e.g. `synonym_cache.json`). Use it when building the name/type index and assigning canonical IDs so existing (name, type) → canonical_id and known SAME_AS ambiguities are reused.
2. **Build name/type index:** Scan all bundles; collect `(name.lower(), class)` for every entity. Entities with the same (name.lower(), class) across papers receive the same canonical ID. Use cache in `lookup_entity(name, type)` where a resolved link exists. Assign a canonical_id (UUID or human-readable slug) for each unique (name, class) and record mapping from bundle-local id → canonical_id.
3. **Auto-resolve high-confidence SAME_AS:** For each relationship with `predicate == "SAME_AS"` and `confidence >= 0.85`: merge the two entities (union synonyms/brand_names), assign a single canonical_id to both, set `resolution="merged"`, `asserted_by="automated"`. Update the name/type index so both names point to the same canonical_id.
4. **Assign canonical IDs:** Every entity that does not yet have a canonical_id gets one; write back into the in-memory representation (or overlay) for each bundle.
5. **Ontology lookup (stub for first pass):** Implement the *scaffolding* for ontology lookup but do **not** call external APIs (UMLS, HGNC, RxNorm, etc.) in this plan. The Pydantic validator requires non-extracted entities to have ontology IDs; so the stub must leave every entity as `source="extracted"` and not change any ontology id fields. That satisfies the validator and leaves a clear place to plug in real lookups later.
6. **Update relationship refs:** Replace bundle-local subject/object IDs with canonical_ids. Remove auto-merged SAME_AS edges from the merged graph (or mark as resolved); keep low/medium confidence SAME_AS with both sides as canonical IDs for review.
7. **Accumulate triples:** For the same (subject_canonical, predicate, object_canonical) across papers: union `source_papers`, union `evidence_ids`, retain max `confidence`.
8. **Save synonym cache.** When writing Pass 2 output (or at end of Pass 2), write all SAME_AS relationships (including resolved ones) to the synonym cache file. Format: index by normalized name (e.g. lowercase); each entry list of `{entity_a, entity_b, confidence, asserted_by, resolution, source_papers}`. This makes Pass 2 idempotent across runs.

**Output:** Either (a) overlay files that add canonical_id and ontology fields per entity (keeping original bundle files read-only), or (b) a single merged graph (e.g. entities.json + relationship list + per-doc edges) plus optional overlay JSON per paper. Original Pass 1 bundle files must remain unchanged on disk.

**Files to create or modify:**

- **New script or module:** e.g. `examples/medlit/scripts/pass2_dedup.py` or `examples/medlit/pipeline/dedup.py`. Read all `PerPaperBundle` from a directory; implement the steps above; write overlay or merged output.
- **Synonym cache module:** e.g. `examples/medlit/pipeline/synonym_cache.py` with `load_synonym_cache(path)`, `save_synonym_cache(path, data)`, `lookup_entity(name, type)` returning optional canonical_id or ambiguity info. Used by Pass 2 at start (load) and end (save).
- **Dedup policy (from spec):** Same (name.lower(), class) across papers → merge to one canonical_id. Same name, different class → keep separate; optionally emit SAME_AS (confidence 0.3) if not present. SAME_AS ≥ 0.85 → auto-merge. SAME_AS < 0.85 → keep link for review. Same (subject, predicate, object) from two papers → accumulate source_papers and evidence_ids.

**Verification:**

- Run Pass 1 on two papers that share at least one entity (same name/class). Run Pass 2 on the two bundle files. Confirm shared entity has one canonical_id; relationships reference canonical_ids; source_papers/evidence_ids accumulated for repeated triples. Confirm synonym cache is created and, on a second run with an additional bundle reusing the same entity name, canonical_id is reused from cache.
- Re-run Step 0 pre-flight.

---

## Step 5. Audit: Extraction provenance and Evidence in Pass 1

**Goal:** Explicit audit/polish pass. Step 3 implements provenance and Evidence; this step verifies that Pass 1 output is complete and correct. If something was missed in Step 3, fix it here — the implementer did not fail Step 3; this step exists to catch gaps.

**Checklist:**

- **extraction_provenance:** Pass 1 output must include `extraction_provenance.models.llm` (name/version), `extraction_provenance.extraction_pipeline` (name, version, git_commit, git_commit_short, git_branch, git_dirty, repo_url), `extraction_provenance.prompt` (version, template, checksum), `extraction_provenance.execution` (timestamp ISO 8601 UTC, hostname, python_version, duration_seconds). Leave `entity_resolution` null in Pass 1 (Pass 2 can fill it).
- **Evidence:** Each Evidence entity in `evidence_entities` must have `id` = `{paper_id}:{section}:{paragraph_idx}:{method}`. Use `SectionType` for section. Every medical relationship (except SAME_AS) must have `evidence_ids` referencing these Evidence IDs.

**Verification:**

- Run Pass 1 on one small paper; inspect the produced JSON. Confirm all provenance fields and Evidence ID format above. Fix any missing fields or incorrect IDs.
- Re-run Step 0 pre-flight.

---

## Step 6. Immutable extraction record and docs

**Goal:** Document and enforce that Pass 1 bundle JSON is never modified in place; Pass 2 writes to overlay or separate directory.

**Implementation:**

- In Pass 2, never open Pass 1 bundle files in write mode. Write overlay files (e.g. `paper_PMC12345.overlay.json` with only canonical_id and ontology fields) or write promoted copies to a different directory (e.g. `promoted/`) so the original `out/paper_PMC12345.json` is untouched.
- **Docs:** In `examples/medlit/README.md` or a new `examples/medlit/INGESTION.md`, describe: Pass 1 output = per-paper bundle JSON (immutable); Pass 2 input = directory of those JSONs; Pass 2 output = overlay or merged graph; synonym cache. Add one-line usage for Pass 1 and Pass 2 scripts. **Document Pass 1 LLM backends** per the "Pass 1: LLM backends and configuration" section: Anthropic (API key), Lambda Labs (OpenAI-compatible endpoint), OpenAI (ChatGPT), with env vars and backend selector.
- **Out-of-scope note:** The spec mentions a review GUI for unreviewed SAME_AS links (`predicate="SAME_AS"`, `resolution=null`). That GUI is **not** in scope for this plan; do not promise it in the docs. Document that low/medium-confidence SAME_AS links are preserved for future review tooling.

**Verification:**

- After running Pass 2, diff or checksum one Pass 1 bundle file before and after; it must be unchanged. README or INGESTION.md must describe the two passes, immutability, and LLM backends; and must note that the review GUI is not yet implemented.
- Re-run Step 0 pre-flight.

---

## Step 7. Integration test (Pass 2 only; pre-baked fixtures)

**Goal:** Add a CI-safe integration test that exercises Pass 2 only, using pre-baked fixture bundle JSONs. No live LLM calls — fast, deterministic, no cost. Pass 1 is tested separately (e.g. manual run, or a unit test with `--dry-run` / mocked LLM).

**Fixture strategy:**

- **Pre-bake two small per-paper bundle JSONs** and commit them under e.g. `examples/medlit/tests/fixtures/bundles/`. One can be the PMC12756687 extraction from INGESTION_REFACTOR.md (paper, entities, evidence_entities, relationships, notes). The second should share at least one entity (same name, same class) with the first so Pass 2 can merge and accumulate.
- The test **does not** call Pass 1 or any LLM API. It only runs Pass 2 (or the dedup module) on the fixture directory.

**New file:** e.g. `examples/medlit/tests/test_two_pass_ingestion.py`

**Content:**

- Load the two fixture bundles from disk.
- Run Pass 2 (dedup + synonym cache) on the fixture directory.
- Assert: entities with same (name, class) get the same canonical_id; a relationship appearing in both papers has union source_papers and union evidence_ids; SAME_AS with confidence >= 0.85 is merged; synonym cache is written (and optionally that a second run reuses canonical_id from cache).
- Pass 1 is tested separately: either run it manually with a real LLM, or add a separate test that uses `--dry-run` / a mock LLM to produce a minimal bundle and then runs Pass 2 on it.

**Verification:**

```bash
uv run pytest examples/medlit/tests/test_two_pass_ingestion.py -v --tb=short
```

Re-run full Step 0 pre-flight.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Schema: SUBTYPE_OF consistency + SAME_AS + INDICATES (if missing); PredicateType, class, RELATIONSHIP_TYPE_MAP, domain, vocab; single verification script |
| 2 | Add `examples/medlit/bundle_models.py`: PerPaperBundle, ExtractedEntityRow, EvidenceEntityRow, RelationshipRow, provenance types |
| 3 | Pass 1: script to write one JSON per paper; build extraction_provenance (full audit in Step 5); document LLM backends |
| 4 | Pass 2: dedup (name/type index, SAME_AS resolution, canonical IDs, ontology stub, relationship refs, accumulation) **and** synonym cache (load at start, save at end, lookup_entity); ontology stub leaves all source="extracted" |
| 5 | Audit: verify extraction_provenance and Evidence format in Pass 1 output; fix any gaps |
| 6 | Immutable bundles + docs (two passes, LLM backends, review GUI out of scope) |
| 7 | Integration test: pre-baked fixture bundles only; test Pass 2; no live LLM; Pass 1 tested separately |

---

## Reference: INGESTION_REFACTOR.md locations

- **Schema:** `examples/medlit_schema/` (entity.py, relationship.py, base.py)
- **Pass 1 output format:** INGESTION_REFACTOR.md "Output: Per-paper bundle JSON" (paper, extraction_provenance, entities, evidence_entities, relationships, notes)
- **Pass 2 steps:** INGESTION_REFACTOR.md "Pass 2: Deduplication and Promotion" (Steps 1–6)
- **SAME_AS:** INGESTION_REFACTOR.md "The SAME_AS Predicate" (schema addition, when to emit, resolution lifecycle)
- **Synonym cache:** INGESTION_REFACTOR.md "Synonym and Identity Cache"
- **Provenance:** INGESTION_REFACTOR.md "Extraction Provenance"

---

## Implementation summary (executed)

**Step 1: Schema (SUBTYPE_OF, SAME_AS, INDICATES)**

- **examples/medlit_schema/base.py:** Added `SAME_AS = "same_as"` to `PredicateType`.
- **examples/medlit_schema/relationship.py:** Added `SameAs(ResearchRelationship)` and `Indicates(BaseMedicalRelationship)`; added `"SUBTYPE_OF"`, `"SAME_AS"`, `"INDICATES"` to `RELATIONSHIP_TYPE_MAP`.
- **examples/medlit_schema/domain.py:** Imported `SameAs`, `Indicates`; added to `relationship_types` and `predicate_constraints` (including `SUBTYPE_OF`).

**Step 2: Per-paper bundle models**

- **examples/medlit/bundle_models.py:** Added `PaperInfo`, `ExtractedEntityRow`, `EvidenceEntityRow`, `RelationshipRow`, `PerPaperBundle` with `entity_class`/`object_id` and `Field(alias="class")`/`Field(alias="object")` for JSON. `to_bundle_dict()` and `from_bundle_dict()` for serialization.

**Step 3: Pass 1 script and LLM backends**

- **examples/medlit/pipeline/pass1_llm.py:** `Pass1LLMInterface`, `AnthropicPass1LLM`, `OpenAIPass1LLM`, `OllamaPass1LLM`, and `get_pass1_llm(backend)`.
- **examples/medlit/scripts/pass1_extract.py:** CLI `--input-dir`, `--output-dir`, `--llm-backend anthropic|openai|ollama`, `--limit`; loads papers via `JournalArticleParser`, calls LLM, builds `ExtractionProvenance` (git, execution, prompt), writes one `paper_{id}.json` per paper.
- **examples/medlit/LLM_SETUP.md:** Describes Anthropic, OpenAI, Lambda Labs, and Ollama setup and env vars.

**Step 4: Pass 2 dedup and synonym cache**

- **examples/medlit/pipeline/synonym_cache.py:** `load_synonym_cache`, `save_synonym_cache`, `lookup_entity`, `add_same_as_to_cache`.
- **examples/medlit/pipeline/dedup.py:** `run_pass2(bundle_dir, output_dir, synonym_cache_path)`: load bundles and cache, name/type index (using cache), high-confidence SAME_AS merge, canonical ID assignment, ontology stub (no-op), relationship refs, triple accumulation, save cache; writes `entities.json`, `relationships.json`, and synonym cache; does not modify input bundles.
- **examples/medlit/scripts/pass2_dedup.py:** CLI for Pass 2.

**Step 5 & 6: Audit and docs**

- Pass 1 already fills provenance and Evidence-style IDs; no code change.
- **examples/medlit/INGESTION.md:** Two-pass flow, immutability, one-line usage, review GUI out of scope.
- **examples/medlit/README.md:** Short “Two-pass ingestion” section with links to INGESTION.md and LLM_SETUP.md.

**Step 7: Integration test and fixtures**

- **examples/medlit/tests/fixtures/bundles/paper_PMC12756687.json** and **paper_PMC99999999.json:** Minimal per-paper bundles (second shares “male breast cancer” Disease for merging).
- **examples/medlit/tests/test_two_pass_ingestion.py:** Tests Pass 2 merge by (name, class), synonym cache write, no modification of inputs, relationship aggregation, and fixture load.

**Other**

- **tests/test_medlit_domain.py:** `expected_relationships` updated to include `INDICATES` and `SAME_AS`.
- **examples/medlit/pipeline/dedup.py:** Uses `rel.object_id` (not `rel.object`) for `RelationshipRow`.

All 82 medlit tests and medlit domain tests pass. Pass 1 verification is manual (requires a configured LLM).
