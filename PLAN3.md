# Plan 3: Medlit Entity Quality Fixes

Plan for addressing type misclassification, provenance gaps, missing section metadata, and synonym/entity overlaps in the medlit pipeline. **Executable without supervision** — follow phases in order.

---

## Summary of Issues

| # | Issue | Root Cause | Fix Location |
|---|-------|------------|--------------|
| 1 | Type misclassification (catecholamines, ACTH, aldosterone synthase as protein) | Pass 1 prompt lacks priority rules and type definitions; `protein` used as catch-all | `pass1_extract.py` |
| 2 | usage_count vs total_mentions divergence | usage_count = unique papers; total_mentions = evidence count; gap is expected | Document only |
| 3 | first_seen_section null for all entities | Section never parsed from evidence_id; hardcoded to None | `bundle_builder.py` |
| 4 | Provenance: PMC_PLACEHOLDER, PMC_UNKNOWN, PMC_extracted | PMC_PLACEHOLDER not in denylist; UNKNOWN/extracted already filtered | `bundle_builder.py` |
| 5a | Spelling overlaps (hyperglycaemia/hyperglycemia) | No British/American spelling normalization | `dedup.py` |
| 5b | Abbreviation overlaps (STAT3↔full name, GPX4↔glutathione peroxidase 4) | Abbreviation expansion; spelling map does not apply | Punted — see section 5b |
| 6 | gut microbiota as biomarker | Edge case; optionally add prompt guidance | `pass1_extract.py` (optional) |

**Priority:** Provenance (4) and first_seen_section (3) are highest operational value. Type misclassification (1) improves downstream queries. Spelling normalization (5a) reduces duplicates. Abbreviation expansion (5b) is deferred.

---

## 1. Type Misclassification — Pass 1 Prompt

**Goal:** Classify entities at the most specific functional role. Hormones → hormone, enzymes → enzyme, not protein. Add explicit priority rules and type definitions.

**Files:**
- `examples/medlit/scripts/pass1_extract.py`
- `examples/medlit/scripts/pass1a_vocab.py` (if adding new types to vocab)

**Steps:**

1. **Add `Hormone` and `Enzyme` to the type taxonomy** in `pass1_extract.py`:
   - Extend `NORMALIZED_TO_BUNDLE_CLASS` with `"hormone": "Hormone"` and `"enzyme": "Enzyme"`.
   - Ensure `normalize_entity_type()` maps these correctly.

2. **Update `_default_system_prompt()`** with:
   - **Priority rule:** "Classify at the most specific functional role. If an entity is both a hormone and a protein, classify it as Hormone. Enzymes should be typed Enzyme, not Protein."
   - **Type definitions with boundaries:**
     - "Protein: structural or signaling proteins that are NOT better classified as Enzyme, Hormone, Receptor, or Antibody."
     - "Hormone: peptide or steroid hormones (e.g. ACTH, cortisol, catecholamines)."
     - "Enzyme: proteins with catalytic function (e.g. aldosterone synthase, kinases)."
   - **Counterexamples:** "ACTH and cortisol are hormones; aldosterone synthase is an enzyme."

3. **Update `pass1a_vocab.py`** if it has a type mapping: add `hormone` and `enzyme` to the vocab type map.

4. **Optional — gut microbiota:** Add to prompt: "Biomarker: measurable clinical indicators. Do not use for microbial communities (e.g. gut microbiota) — use BiologicalProcess or AnatomicalStructure instead if applicable." Both `BiologicalProcess` and `AnatomicalStructure` exist in `NORMALIZED_TO_BUNDLE_CLASS` (pass1_extract.py).

5. **Tests:** Add a test that prompts with a sample containing ACTH, cortisol, and aldosterone synthase; assert extracted types are Hormone, Hormone, Enzyme (or equivalent). Alternatively, add a unit test for `normalize_entity_type("hormone")` → `"Hormone"`.

---

## 2. usage_count vs total_mentions — Document Semantics

**Goal:** Confirm and document that usage_count = unique papers; total_mentions = evidence count. No code change.

**Files:**
- `examples/medlit/pipeline/bundle_builder.py` (docstring)
- `kgbundle` or schema docs if applicable

**Steps:**

1. **Add docstring to `_entity_usage_from_bundles`** explaining:
   - `usage_count` = number of unique papers in `supporting_documents` (each paper counted once per entity).
   - `total_mentions` = sum of evidence_ids across all relationships where the entity appears (subject or object).
   - Large gaps (e.g. usage_count=7, total_mentions=65) are expected when an entity appears many times within a few papers.

2. **Optional:** Add a short comment in `_merged_entity_to_entity_row` where usage counts are applied, referencing the docstring.

---

## 3. first_seen_section — Parse from evidence_id

**Goal:** Populate `first_seen_section` by parsing the section from evidence_id. Format: `{paper_id}:{section}:{paragraph_idx}:{method}`.

**Files:**
- `examples/medlit/pipeline/bundle_builder.py`

**Pre-implementation verification:** Confirm `evidence_ids` is a flat list of strings. In `bundle_builder.py`, `rel.evidence_ids` comes from `RelationshipRow.evidence_ids: list[str]` (see `bundle_models.py`). Fixtures use `["PMC12756687:abstract:0:llm"]`. If the structure ever changes (e.g. tuples or objects), adapt the parse logic accordingly.

**Steps:**

1. **Add helper** `_section_from_evidence_id(evidence_id: str) -> str | None`:
   - Split by `:`. If `len(parts) >= 2`, return `parts[1]`. Otherwise return `None`.

2. **In `_entity_usage_from_bundles`**, when setting `first_seen_document` and `first_seen_section`:
   - Change `for _ in evidence_ids` to `for evidence_id in evidence_ids`.
   - When `rec["first_seen_document"] is None` and paper_id is valid, set:
     - `rec["first_seen_document"] = paper_id`
     - `rec["first_seen_section"] = _section_from_evidence_id(evidence_id)` (instead of `None`).

3. **Fallback:** If `_section_from_evidence_id` returns `None`, keep `first_seen_section` as `None` (e.g. malformed IDs).

4. **Tests:** Add test that entity with evidence_id `PMC123:abstract:0:llm` gets `first_seen_section="abstract"`.

---

## 4. Provenance — Add PMC_PLACEHOLDER to Denylist

**Goal:** Exclude `PMC_PLACEHOLDER` from supporting_documents and first_seen_document, same as PMC_UNKNOWN and PMC_extracted.

**Files:**
- `examples/medlit/pipeline/bundle_builder.py`

**Steps:**

1. **Update `PROVENANCE_DENYLIST`**:
   ```python
   PROVENANCE_DENYLIST = frozenset({"PMC_UNKNOWN", "PMC_extracted", "PMC_PLACEHOLDER"})
   ```

2. **Ensure existing logic applies:** `_entity_usage_from_bundles` already skips denylisted paper_ids for `supporting_documents` and `first_seen_document`. No further changes needed.

3. **Tests:** Add or extend test: entity with only `PMC_PLACEHOLDER` in relationships gets `usage_count=0`, `first_seen_document=None`.

---

## 5a. Spelling Normalization (British/American)

**Goal:** Merge British/American spelling variants (e.g. hyperglycaemia/hyperglycemia) via a spelling map in dedup. Does **not** address abbreviation expansion (STAT3↔full name, GPX4↔glutathione peroxidase 4).

**Files:**
- `examples/medlit/pipeline/dedup.py`

**Steps:**

1. **Add `_normalize_for_dedup(name: str) -> str`** in `dedup.py`:
   - Lowercase, strip.
   - Apply British→American spelling map for common biomedical terms, e.g.:
     - `hyperglycaemia` → `hyperglycemia`
     - `haemoglobin` → `hemoglobin`
     - `tumour` → `tumor`
     - `oesophagus` → `esophagus`
     - `leukaemia` → `leukemia`
   - Return normalized form. Keep mapping small and extensible (e.g. `SPELLING_NORMALIZATIONS: dict[str, str]`).

2. **Use in index and lookup:** When building `name_type_to_canonical` and when looking up:
   - For each entity, add keys for both `(name.lower(), entity_class)` and `(_normalize_for_dedup(name), entity_class)`.
   - When looking up, check `(name.lower(), entity_class)` first, then `(_normalize_for_dedup(name), entity_class)`.
   - **Scope:** Only use normalized form as additional lookup key; do not replace original. Both name and normalized form should point to same canonical.

3. **Synonym indexing:** When adding synonyms to the index, also add `_normalize_for_dedup(syn)` for each synonym, so "hyperglycaemia" matches entity with synonym "hyperglycemia".

4. **Tests:** Two entities with same class, names "hyperglycaemia" and "hyperglycemia", merge to one canonical.

---

## 5b. Abbreviation Expansion (STAT3/GPX4) — Punted

**Goal:** STAT3 ↔ "signal transducer and activator of transcription 3" and GPX4 ↔ "glutathione peroxidase 4" are abbreviation-expansion pairs, not spelling variants. The spelling map in 5a does not catch them.

**Options:**
- **Synonym-driven merge:** PLAN2 synonym indexing already merges when A has B as synonym and B exists as standalone (or vice versa). If the LLM emits both symbol and full name as synonyms on one entity, the other paper's entity will merge when it matches. If neither paper emits the synonym, no merge.
- **Embedding-based dedup:** PLAN2 embedding pass catches provisional pairs with high similarity. May merge STAT3/full name if both are provisional.
- **Abbreviation dictionary or LLM lookup:** Would require a biomedical abbreviation dictionary or a separate LLM call to expand abbreviations. Higher effort.

**Decision:** Explicitly punt. No separate task in this plan. If abbreviation pairs persist after PLAN2 + 5a, consider a future enhancement (e.g. abbreviation dictionary lookup in dedup).

---

## 6. Optional: Type Validation Pass

**Goal:** Lightweight post-extraction check for type correctness. Defer unless prompt changes (section 1) prove insufficient.

**Files:**
- New: `examples/medlit/pipeline/type_validation.py` (optional)
- `examples/medlit/scripts/pass1_extract.py` (optional integration)

**Steps (deferred):**

1. Add optional `validate_entity_types(bundle, llm_client)` that takes `(entity_name, entity_type, context_snippet)` and returns `TypeValidationResult(ok: bool, suggested_type: str | None, reason: str)`.
2. Integrate after extraction; apply corrections when confidence is high.
3. Document as future enhancement in PLAN3 if not implemented.

---

## Execution Order

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| 1 | Type taxonomy + prompt (1), Document usage_count (2) | None |
| 2 | first_seen_section (3), Provenance denylist (4) | None |
| 3 | Spelling normalization (5a) | None |

Phases 1–3 can be done in parallel. Phase 1 affects Pass 1 output; phases 2–3 affect Pass 3 and Pass 2 respectively.

---

## File Touch Summary

| File | Changes |
|------|---------|
| `examples/medlit/scripts/pass1_extract.py` | Add Hormone, Enzyme to NORMALIZED_TO_BUNDLE_CLASS; expand _default_system_prompt with priority rules, type definitions, counterexamples |
| `examples/medlit/scripts/pass1a_vocab.py` | Add hormone, enzyme to type map if present |
| `examples/medlit/pipeline/bundle_builder.py` | _section_from_evidence_id; populate first_seen_section; add PMC_PLACEHOLDER to PROVENANCE_DENYLIST; docstring for usage_count/total_mentions |
| `examples/medlit/pipeline/dedup.py` | _normalize_for_dedup with spelling map; use in index and lookup (5a only; 5b punted) |
| `examples/medlit/tests/test_bundle_builder.py` or `test_pass3*.py` | first_seen_section, provenance denylist |
| `examples/medlit/tests/test_dedup.py` | Spelling normalization merge |

---

## Verification

After each change:

1. Run `uv run pytest examples/medlit/tests/ -v`
2. Run `lint.sh`
3. Run full pipeline on a small corpus (5–10 papers):
   - Inspect `entities.json` — `first_seen_section` populated where evidence has section; no PMC_PLACEHOLDER in supporting_documents
   - For type changes: re-run Pass 1 on sample papers; verify ACTH, cortisol, aldosterone synthase types
4. For spelling: run Pass 2 on bundles containing hyperglycaemia/hyperglycemia; verify single merged entity
