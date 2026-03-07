# Ingestion Pipeline Redesign — Implementation Plan

## Goal

Replace the current medlit ingestion pipeline with the architecture described in `ingestion_redesign.md`. Key outcomes:

- **Single source of truth** for entity types and predicates in YAML config files
- **Collaborative iteration** — domain experts edit config files, not Python
- **Richer trust/provenance** — linguistic trust enum, study design signal, citation linkage
- **Better deduplication** — authority lookup before fuzzy synonym merging
- **Schema binding** — Jinja2 templates in `kgraph/`, config in `examples/medlit/config/`

**Context:** Postgres DBs are blown away every run; no migration of existing data is required.

**Reference:** Full design in `ingestion_redesign.md`. This plan is the executable implementation order.

---

## Current Pipeline (Baseline)

| Step | Script | Output |
|------|--------|--------|
| Pass 1a | `pass1a_vocab` | `pass1_vocab/` (vocab.json, synonym cache) |
| Pass 1b | `pass1_extract` | `pass1_bundles/` (paper_*.json) |
| Pass 2 | `pass2_dedup` | `medlit_merged/` |
| Pass 3 | `pass3_build_bundle` | `medlit_bundle/` |

Schema is scattered across: `pass1_extract.py` (NORMALIZED_TO_BUNDLE_CLASS, prompt string), `pass1a_vocab.py` (duplicate), `dedup.py` (_entity_class_to_lookup_type), `examples/medlit_schema/` (relationship.py, domain.py, base.py).

---

## Implementation Phases

### Phase A: Schema Consolidation (Low Risk)

**Objective:** Create YAML config files as the single source of truth. Wire existing pipeline to load from them. No behavior change yet; just consolidation.

#### Task A1: Create `examples/medlit/config/` and entity_types.yaml

**Create:** `examples/medlit/config/entity_types.yaml`

**Content:** Extract from `pass1_extract.NORMALIZED_TO_BUNDLE_CLASS` and `pass1_extract._default_system_prompt()`. Structure:

```yaml
# Entity type taxonomy. Key: normalized (lowercase, no spaces). Value: bundle class (PascalCase).
# Descriptions are for schema binding in prompts.
types:
  disease:
    bundle_class: Disease
    description: "Diseases, conditions, syndromes. Use the most specific term."
  gene:
    bundle_class: Gene
    description: "Genes by symbol or name. Prefer HGNC symbol when known."
  drug:
    bundle_class: Drug
    description: "Drugs, compounds, therapeutic agents."
  # ... (all 20 from NORMALIZED_TO_BUNDLE_CLASS)
```

Include all types from `pass1_extract.py` lines 152–173. Add `description` from the prompt's entity type section (lines 216–221).

**Create:** `examples/medlit/config/README.md` — one paragraph: "Schema config for medlit. Domain experts edit these files. Pipeline loads at runtime."

**Acceptance:** File exists, parses as valid YAML, contains all current entity types.

---

#### Task A2: Create predicates.yaml

**Create:** `examples/medlit/config/predicates.yaml`

**Content:** Extract from `pass1_extract` prompt ("TREATS, INCREASES_RISK, INDICATES, ASSOCIATED_WITH, SAME_AS, SUBTYPE_OF, etc.") and `examples/medlit_schema/relationship.py` / `domain.py`. Structure:

```yaml
# Predicate definitions. Key: predicate name (UPPER_SNAKE). Used in extraction prompt and dedup.
predicates:
  TREATS:
    description: "Drug or procedure used therapeutically to address a disease."
    subject_types: [Drug, Procedure]
    object_types: [Disease]
  INCREASES_RISK:
    description: "Gene, mutation, or factor that raises risk of a disease."
    subject_types: [Gene, Mutation]
    object_types: [Disease]
  INDICATES:
    description: "Biomarker or test that indicates presence or status of a condition."
    # ...
  ASSOCIATED_WITH:
    description: "General association; use when more specific predicate does not apply."
    subject_types: [Any]
    object_types: [Any]
  SAME_AS:
    description: "Entity A is the same as entity B (for coreference/merge)."
    subject_types: [Any]
    object_types: [Any]
  SUBTYPE_OF:
    description: "Entity A is a subtype of entity B (e.g. Cushing's disease subtype_of Cushing's syndrome)."
    subject_types: [Disease]
    object_types: [Disease]
```

Include all predicates currently in the prompt and medlit_schema. Add any others that appear in pass1_bundles (CAUSES, INHIBITS, REGULATES, etc.) with placeholder descriptions.

**Acceptance:** File exists, parses as valid YAML. Dedup's `SAME_AS` and all prompt predicates are present.

---

#### Task A3: Create domain_instructions.md

**Create:** `examples/medlit/config/domain_instructions.md`

**Content:** Extract the domain-specific guidance from `pass1_extract._default_system_prompt()`:

- Entity type classification rules (Protein vs Enzyme vs Hormone, Biomarker exclusions, counterexamples)
- "Use 'class' for entity type"
- "Predicates: TREATS, INCREASES_RISK, ..." (can reference predicates.yaml)
- Evidence id format

Structure with clear headers: `## Entity type classification`, `## Predicates`, `## Evidence format`.

**Acceptance:** File exists. Content matches current prompt's domain guidance. Readable by non-engineer.

---

#### Task A4: Add config loader module

**Create:** `examples/medlit/pipeline/config_loader.py`

**Functions:**
- `load_entity_types(config_dir: Path) -> dict` — load entity_types.yaml, return dict suitable for template injection
- `load_predicates(config_dir: Path) -> dict` — load predicates.yaml
- `load_domain_instructions(config_dir: Path) -> str` — load domain_instructions.md as string
- `get_schema_version(config_dir: Path) -> str` — SHA256 of sorted concatenation of the three config files' contents (entity_types.yaml, predicates.yaml, domain_instructions.md), returned as 8-char hex prefix. Deterministic: same config → same version. Short enough for filenames.

**Acceptance:** Module loads config from `examples/medlit/config/`. Tests in `examples/medlit/tests/test_config_loader.py` verify load and schema_version changes when config changes.

---

#### Task A5: Refactor pass1_extract to use config

**Modify:** `examples/medlit/scripts/pass1_extract.py`

- Add `--config-dir` argument (default: `examples/medlit/config/`)
- Replace `NORMALIZED_TO_BUNDLE_CLASS` with `load_entity_types(config_dir)`
- Replace `normalize_entity_type` to use loaded config (or keep logic, derive from config)
- Build `_default_system_prompt()` by loading `domain_instructions.md` and `predicates.yaml`, formatting into the prompt string (no Jinja2 yet — just string format)
- Include schema_version in provenance (PromptInfo or ExtractionProvenance)

**Acceptance:** `uv run python -m examples.medlit.scripts.pass1_extract --input-dir ... --output-dir ... --llm-backend anthropic --limit 1` produces identical output to before (or equivalent). Existing tests pass.

---

#### Task A6: Refactor pass1a_vocab to use config

**Modify:** `examples/medlit/scripts/pass1a_vocab.py`

- Add `--config-dir` argument
- Remove `PASS1A_TYPE_ENUM` and `NORMALIZED_TO_BUNDLE_CLASS` — load from `entity_types.yaml` instead
- Build `PASS1A_SYSTEM_PROMPT` from config (entity type list)

**Acceptance:** Pass 1a runs successfully. Vocab output is equivalent.

---

#### Task A7: Refactor dedup to use config for entity_class mapping

**Modify:** `examples/medlit/pipeline/dedup.py`

- Add optional `config_dir` parameter to `run_pass2` (or load from fixed path)
- Replace `_entity_class_to_lookup_type` hardcoded dict with lookup derived from `entity_types.yaml` (bundle_class → lookup type: Disease→disease, Gene→gene, etc.)
- Keep `SAME_AS` string — it comes from predicates.yaml; add a `load_predicates()` call to get the list, or a constant `SAME_AS_PREDICATE = "SAME_AS"` that is documented as coming from config

**Acceptance:** Pass 2 runs successfully. Dedup behavior unchanged. No more "Must match _entity_class_to_lookup_type" comment — it's derived from config.

---

### Phase B: Jinja2 Templates (Low Risk)

**Objective:** Move prompt structure into Jinja2 templates in `kgraph/`. Config remains in `examples/medlit/config/`. Pipeline renders templates with config.

#### Task B1: Create kgraph/templates/ directory

**Create:** `kgraph/templates/` (in kgraph package, not examples)

**Create:** `kgraph/templates/entity_relationship_extraction.j2`

**Content:** Extract the structure from `pass1_extract._default_system_prompt()`. Use injection points:
- `{{ entity_types }}` — pre-formatted string (bulleted list) from entity_types.yaml
- `{{ predicates }}` — pre-formatted string (bulleted list) from predicates.yaml
- `{{ domain_instructions }}` — raw content of domain_instructions.md
- `{{ vocab_section }}` — optional, for Pass 1b vocab context (same as current)

**Pre-formatting:** `render.py` formats entity_types and predicates into strings before injection. The template receives only `{{ variable }}` placeholders — no loops or conditionals. This keeps the template legible and the legibility guarantee intact.

**Acceptance:** Template exists. Manual render with sample config produces readable prompt.

---

#### Task B2: Add template renderer to kgraph

**Create:** `kgraph/templates/__init__.py` and `kgraph/templates/render.py`

**Function:** `render_extraction_prompt(config_dir: Path, vocab_entries: list | None = None) -> str`

- Load entity_types, predicates, domain_instructions from config_dir
- **Pre-format** entity_types and predicates into bullet-list strings (do not pass raw dicts/list to template)
- Render `entity_relationship_extraction.j2` with `{{ entity_types }}`, `{{ predicates }}`, `{{ domain_instructions }}` as already-rendered text
- Append vocab_section if provided
- Return full prompt string

**Acceptance:** `render_extraction_prompt(Path("examples/medlit/config"))` returns a prompt equivalent to current `_default_system_prompt()`.

---

#### Task B3: Wire pass1_extract to use Jinja2 template

**Modify:** `examples/medlit/scripts/pass1_extract.py`

- Remove `_default_system_prompt()` body
- Call `render_extraction_prompt(config_dir, vocab_entries)` from kgraph.templates
- Pass `--config-dir` to specify medlit config
- Ensure schema_version (hash of config) is in provenance

**Acceptance:** Pass 1b produces equivalent output. Prompt checksum in bundle reflects config content.

---

### Phase C: Dedup Ordering Change (Medium Risk)

**Objective:** Hit authority lookup (CanonicalIdLookupInterface) before synonym cache in the dedup flow.

#### Task C1: Reorder get_or_assign_canonical in dedup.py

**Modify:** `examples/medlit/pipeline/dedup.py` — function `get_or_assign_canonical`

**Current order:** (1) index, (2) entity_row authoritative ID, (3) synonym cache, (4) authority lookup
**New order:** (1) index, (2) entity_row authoritative ID, (3) **authority lookup**, (4) synonym cache

Move the `if lookup is not None:` block to execute before `lookup_entity(cache, ...)`.

**Acceptance:** Pass 2 with `--canonical-id-cache` produces merged output. Run A/B: same corpus, old vs new order, compare entity counts and merge quality. Document any behavioral differences.

---

### Phase D: Trust Signals and Schema Extensions (Medium Risk)

**Objective:** Add linguistic_trust enum (asserted/suggested/speculative). Extend relationship schema. Keep backward compatibility where possible.

#### Task D1: Add linguistic_trust to extraction prompt and output

**Modify:** `kgraph/templates/entity_relationship_extraction.j2` and `domain_instructions.md`

- Add instruction: "For each relationship, classify linguistic trust: asserted (direct), suggested (soft), speculative (hedged)."
- Add to expected JSON: `linguistic_trust: "asserted" | "suggested" | "speculative"`

**Modify:** `examples/medlit/bundle_models.py` — `RelationshipRow`

- Add `linguistic_trust: Optional[Literal["asserted", "suggested", "speculative"]] = None`
- If absent in legacy bundles, default to `"asserted"` or derive from confidence band for backward compat

**Modify:** Pass 3 bundle builder — ensure linguistic_trust flows to output

**Acceptance:** New extractions include linguistic_trust. Old bundles without it still load (default or omit). Tests updated.

---

#### Task D2: Add study design trust signal (per-paper)

*(D2 and D3 both touch bundle_models.py; do them in one pass.)*

**Create:** `StudyDesignMetadata` model (in `examples/medlit/bundle_models.py` or kgbundle)

**Fields:** `study_type: str | None`, `sample_size: int | None`, `multicenter: bool`, `held_out_validation: bool`

**Extraction:** In Pass 1b (`pass1_extract`), as a **second LLM call per paper** after the main entity/relationship extraction. Same script, same document load — no standalone script. Simple structured prompt against the Methods section (and abstract if needed). Returns these four fields. Doesn't have to be fancy — even a basic structured prompt that returns the fields gets the data.

**Storage:** Attach to the paper-level node in the bundle. Not on each edge. Pass 3 output includes `study_design` (or equivalent) on the paper metadata.

**Rationale:** This is the other half of the two-dimensional trust model (design doc Phase 4b). Without it you have linguistic trust per relationship but no paper-level study quality signal. Both dimensions are needed.

**Acceptance:** Study design metadata extracted per paper, stored on paper node. Pass 3 bundle includes it. Defaults (None, False) when extraction fails or Methods is missing. Spot-check 5 papers to verify study design output is reasonable.

---

#### Task D3: Add provenance list structure to relationship schema

**Modify:** Bundle models and Pass 3 output

- Relationship record: support `provenance: list[{section, sentence, citation_markers}]` in addition to or replacing single evidence_ids
- For now: accumulate evidence per (sub, pred, obj) as list; don't flatten
- Document as "provenance list" for Phase 4c cross-paper support later

**Acceptance:** Schema supports multiple provenance records per relationship. Pass 3 emits it when available.

---

### Phase E: Chunking and Per-Chunk Extraction (Higher Risk)

**Objective:** Replace whole-paper extraction with per-chunk extraction. Overlapping chunks.

**Sequencing note:** Phase D adds trust signals while still using whole-paper prompts. When Phase E lands, extraction splits:
- **Per-chunk:** Entity extraction, relationship extraction, `linguistic_trust` (per relationship, in the relationship pass)
- **Per-paper:** Study design metadata (Task D2) — extracted once per document from Methods/abstract

The implementer of E must preserve this split. Do not move study design extraction to per-chunk; it stays per-paper. Linguistic trust is per-relationship and thus naturally per-chunk when relationships are extracted per-chunk.

#### Task E1: Add chunking module

**Create:** `kgraph/pipeline/chunker.py` or extend `examples/medlit/pipeline/pmc_chunker.py`

- Input: document text, section boundaries
- Output: list of chunks with overlap (e.g. 512 tokens, 64 overlap)
- Each chunk: `{chunk_id, text, section, start_offset, end_offset}`
- Ensure no sentence orphaned at boundary (overlap or sentence-aware splitting)

**Acceptance:** Chunker produces overlapping chunks. Unit tests with sample document.

---

#### Task E2: Per-chunk entity extraction

**Modify:** Pass 1b flow (or new script)

- Parse document → chunks
- For each chunk: call LLM with entity extraction template, entity_types, domain_instructions
- Collect entities per chunk; merge by (name, type) across chunks within document
- Output: per-document bundle with entities from all chunks (deduplicated within doc)

**Acceptance:** Per-chunk extraction produces entity list equivalent or better than whole-paper. Compare on 3–5 papers.

---

#### Task E3: Per-chunk relationship extraction

- After entity resolution within document, for each chunk: call LLM with relationship template, entity_list (resolved), predicates, domain_instructions
- Collect relationships; merge by (sub, pred, obj) across chunks
- Accumulate provenance (chunk_id, section, sentence) per occurrence

**Acceptance:** Per-chunk relationship extraction produces relationship list. Redundant extractions merged with provenance accumulated.

---

### Phase F: Phase 0 (Abstract Pre-Pass) and Citation Integration (Deferred)

**Objective:** Abstract pre-pass for claim hierarchy. Citation marker capture and resolution.

**Tasks:** Define in a follow-on plan. Phase 0 and Phase 7 require:
- New template `abstract_prepass.j2`
- JATS reference list parsing
- Citation marker parsing in prompts
- Claim hierarchy data model

**For PLAN2:** Document as "Phase F — Future work". Not in initial execution scope.

---

## Implementation Order

Execute in this order. Each phase completes before the next begins.

1. **Phase A** (Tasks A1–A7) — Schema consolidation. Establishes single source of truth.
2. **Phase B** (Tasks B1–B3) — Jinja2 templates. Enables config-driven prompts.
3. **Phase C** (Task C1) — Dedup ordering. One code change, validate with A/B.
4. **Phase D** (Tasks D1–D3) — Trust signals (linguistic_trust, study design), provenance list. Schema extensions. Do D2 and D3 in one pass through bundle_models.py.
5. **Phase E** (Tasks E1–E3) — Chunking, per-chunk extraction. Higher risk; validate quality. Preserve per-paper study design extraction; linguistic trust moves with per-chunk relationship extraction.
6. **Phase F** — Deferred.

---

## Verification Commands

After each phase:

```bash
# Full pipeline
./run-ingest.sh

# Or stepwise
uv run python -m examples.medlit.scripts.pass1a_vocab --input-dir examples/medlit/pmc_xmls --output-dir pass1_vocab --llm-backend anthropic --papers PMC12771675.xml
uv run python -m examples.medlit.scripts.pass1_extract --input-dir examples/medlit/pmc_xmls --output-dir pass1_bundles --llm-backend anthropic --vocab-file pass1_vocab/vocab.json --papers PMC12771675.xml
uv run python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles --output-dir medlit_merged --synonym-cache pass1_vocab/seeded_synonym_cache.json
uv run python -m examples.medlit.scripts.pass3_build_bundle --merged-dir medlit_merged --bundles-dir pass1_bundles --output-dir medlit_bundle
```

```bash
# Tests
uv run pytest examples/medlit/tests/ tests/test_medlit_domain.py tests/test_medlit_entities.py tests/test_medlit_relationships.py -v
```

---

## Risk Summary

| Phase | Risk | Mitigation |
|-------|------|------------|
| A | Config load bugs | Unit tests for config_loader. Compare output before/after. |
| B | Template render differs from hardcoded | Checksum comparison. Side-by-side prompt diff. |
| C | Dedup behavior change | A/B run on same corpus. Compare entity/relationship counts. |
| D | Schema breakage | Optional fields, defaults. Tests for backward compat. |
| D (study design) | New LLM call per paper could fail silently or produce garbage | Defaults on failure (None, False). Spot-check 5 papers' study design output. |
| E | Quality regression | Benchmark on 5–10 papers. Compare entity/relationship recall. |

---

## Files Created/Modified Summary

**New files:**
- `examples/medlit/config/entity_types.yaml`
- `examples/medlit/config/predicates.yaml`
- `examples/medlit/config/domain_instructions.md`
- `examples/medlit/config/README.md`
- `examples/medlit/pipeline/config_loader.py`
- `kgraph/templates/__init__.py`
- `kgraph/templates/entity_relationship_extraction.j2`
- `kgraph/templates/render.py`
- `examples/medlit/tests/test_config_loader.py`
- `kgraph/pipeline/chunker.py` (Phase E)

**Modified files:**
- `examples/medlit/scripts/pass1_extract.py`
- `examples/medlit/scripts/pass1a_vocab.py`
- `examples/medlit/pipeline/dedup.py`
- `examples/medlit/bundle_models.py` (RelationshipRow, StudyDesignMetadata, provenance)
- `examples/medlit/pipeline/bundle_builder.py` (if needed for provenance, study_design)
