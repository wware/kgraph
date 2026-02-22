# UNIMPLEMENTED — Plan: Mitigate relationship extraction performance issues (PLAN6)

Execute steps in order from the **repository root**. All edits are in `examples/medlit/pipeline/relationships.py` unless stated otherwise. Reference: **A.md** (Gemini observations).

**Scope:** 6.1 Skip semantic when string says entity missing; 6.2 Shorten prompt (remove signature table); 6.3 Predicate hierarchy post-filter (Option B). 6.4 Batch semantic checks is specified so it can be implemented later without supervision.

---

## Step 0. Pre-flight

From repo root:

```bash
(./lint.sh && uv run pytest examples/medlit/ -v -k "relationship or medlit" --tb=short) 2>&1 | tail -30
```

Note any failures. After each step below, re-run the same and fix regressions.

---

## Step 1. 6.1 — Skip semantic when string check returns "entity missing"

**Goal:** When the string evidence check fails with `evidence_missing_subject` or `evidence_missing_object`, do not call the semantic (embedding) path; return immediately and save embedding calls.

**File:** `examples/medlit/pipeline/relationships.py`

**Location:** In `_process_llm_item`, immediately after the block that sets `evidence_detail["method"] = "string_match"` and before the block that calls `_evidence_contains_both_entities_semantic` (current lines 723–738).

**Current code (lines 723–738):**

```python
        evidence_ok, evidence_drop_reason, evidence_detail = _evidence_contains_both_entities(evidence, subject_name, object_name, subject_entity, object_entity)
        if evidence_ok:
            evidence_detail["method"] = "string_match"
        if not evidence_ok and self._embedding_generator is not None:
            evidence_ok, evidence_drop_reason, evidence_detail = await _evidence_contains_both_entities_semantic(
                ...
            )
```

**Change:** Insert a new block between the string check and the semantic call. After setting `evidence_detail["method"] = "string_match"` when `evidence_ok`, add:

- If string check failed **and** `evidence_drop_reason` is `"evidence_missing_subject"` or `"evidence_missing_object"`: set `decision["evidence_check"] = evidence_detail`, `decision["drop_reason"] = evidence_drop_reason`, and `return None, decision`. Do **not** call `_evidence_contains_both_entities_semantic` for these cases.

**Exact edit:**

Find:

```python
        evidence_ok, evidence_drop_reason, evidence_detail = _evidence_contains_both_entities(evidence, subject_name, object_name, subject_entity, object_entity)
        if evidence_ok:
            evidence_detail["method"] = "string_match"
        if not evidence_ok and self._embedding_generator is not None:
```

Replace with:

```python
        evidence_ok, evidence_drop_reason, evidence_detail = _evidence_contains_both_entities(evidence, subject_name, object_name, subject_entity, object_entity)
        if evidence_ok:
            evidence_detail["method"] = "string_match"
        if not evidence_ok and evidence_drop_reason in ("evidence_missing_subject", "evidence_missing_object"):
            decision["evidence_check"] = evidence_detail
            decision["drop_reason"] = evidence_drop_reason
            return None, decision
        if not evidence_ok and self._embedding_generator is not None:
```

**Verification:** Run tests. In a trace where string check returns `evidence_missing_object`, the decision must have `evidence_check` with `object_in_evidence: false` and **no** `method: "semantic_match"` (and no embedding calls for that item).

---

## Step 2. 6.2 — Shorten prompt: remove signature table and type-matching line

**Goal:** Remove the PREDICATE SIGNATURES block and the “You MUST choose a predicate whose subject/object types match the signature table” line from the LLM prompt. Keep allowed predicates and guidelines. Add a short note that types are validated server-side. Python type enforcement (predicate_constraints) is unchanged.

**File:** `examples/medlit/pipeline/relationships.py`

**Location:** Method `_build_llm_prompt` (starts around line 453).

### 2a. Remove signature table construction and use

**Delete** the block that builds `sig_lines` (lines 469–479) and **remove** from the f-string the PREDICATE SIGNATURES section and the guideline line that references the signature table.

**Find (in _build_llm_prompt):**

```python
        # 2) Build a compact "signature table" from predicate_constraints
        # Example: treats: drug -> disease
        sig_lines: list[str] = []
        for key in predicate_keys:
            lc = key.lower()
            c = self._domain.predicate_constraints.get(key)
            if c:
                subj = ", ".join(sorted(c.subject_types))
                obj = ", ".join(sorted(c.object_types))
                sig_lines.append(f"- {lc}: ({subj}) -> ({obj})")
            else:
                sig_lines.append(f"- {lc}: (no type constraints defined)")

        # Optional: a tiny bit of extra guidance
```

Replace with:

```python
        # Optional: a tiny bit of extra guidance
```

(Remove only the signature-building block; keep the comment "Optional: a tiny bit of extra guidance" and the rest of the method.)

### 2b. Update the prompt body (f-string)

**Find in the same method’s return f"""...""" block:**

- The line: `- You MUST choose a predicate whose subject/object types match the signature table. If none apply, output no relationship.`
- The line: `- Output ONLY valid JSON: a JSON array of relationship objects.`
- The block:
  ```
  PREDICATE SIGNATURES (subject_type -> object_type):
  {chr(10).join(sig_lines)}

  GUIDELINES:
  ```

**Replace** that guideline line with one line: `- We validate subject/object types server-side; use entity names from the list.`

**Remove** the entire `PREDICATE SIGNATURES (subject_type -> object_type):` and `{chr(10).join(sig_lines)}` from the f-string. Keep `GUIDELINES:` and the rest of the prompt.

**Exact old block (in the f-string):**

```
Your job:
- Find relationships between the listed entities that are explicitly supported by the text.
- You MUST choose a predicate whose subject/object types match the signature table. If none apply, output no relationship.
- Output ONLY valid JSON: a JSON array of relationship objects.
- Use ONLY the allowed predicates listed below (lowercase strings).

ALLOWED PREDICATES:
{", ".join(allowed_predicates)}

PREDICATE SIGNATURES (subject_type -> object_type):
{chr(10).join(sig_lines)}

GUIDELINES:
```

**New block:**

```
Your job:
- Find relationships between the listed entities that are explicitly supported by the text.
- We validate subject/object types server-side; use entity names from the list.
- Output ONLY valid JSON: a JSON array of relationship objects.
- Use ONLY the allowed predicates listed below (lowercase strings).

ALLOWED PREDICATES:
{", ".join(allowed_predicates)}

GUIDELINES:
```

### 2c. Update examples section wording

**Find:**

```python
        examples_section = """
EXAMPLES (follow subject/object order and predicate from the signature table):
```

**Replace with:**

```python
        examples_section = """
EXAMPLES (follow subject/object order and use allowed predicates):
```

**Verification:** Run tests. Generate a trace (e.g. one paper with `--use-ollama --limit 1`); the trace’s `prompt` must not contain "PREDICATE SIGNATURES" or "signature table". When the LLM returns invalid types, decisions must still show `drop_reason: "type_constraint_mismatch"`.

---

## Step 3. 6.3 — Predicate hierarchy post-filter (Option B)

**Goal:** After collecting all accepted relationships for the window, for each (subject_id, object_id) that has more than one predicate, keep only the relationship with the highest predicate specificity (e.g. prefer `indicates` over `associated_with`). Use a single, explicit hierarchy map; predicates not in the map have specificity 0.

**File:** `examples/medlit/pipeline/relationships.py`

### 3a. Add constant and helper

**Location:** Near the top of the file, after `DEFAULT_TRACE_DIR = ...` (around line 26) and before the first function. Add:

```python
# Predicate specificity for deduplication: when the same (subject_id, object_id) has multiple
# predicates, we keep the one with highest specificity (e.g. indicates > associated_with).
# Predicates not listed have specificity 0.
PREDICATE_SPECIFICITY: dict[str, int] = {
    "indicates": 2,
    "associated_with": 1,
}
```

Then add a helper function (e.g. after `_build_entity_index` or before `_evidence_contains_both_entities`):

```python
def _deduplicate_relationships_by_predicate_specificity(
    relationships: list[BaseRelationship],
) -> list[BaseRelationship]:
    """For each (subject_id, object_id), keep only the relationship with highest predicate specificity."""
    if not relationships:
        return relationships
    key_to_best: dict[tuple[str, str], BaseRelationship] = {}
    for rel in relationships:
        key = (rel.subject_id, rel.object_id)
        pred = getattr(rel, "predicate", None) or getattr(rel, "get_edge_type", lambda: "")()
        if isinstance(pred, str):
            pred = pred.lower()
        else:
            pred = str(pred).lower()
        specificity = PREDICATE_SPECIFICITY.get(pred, 0)
        existing = key_to_best.get(key)
        if existing is None:
            key_to_best[key] = rel
            continue
        existing_pred = getattr(existing, "predicate", None) or getattr(existing, "get_edge_type", lambda: "")()
        if isinstance(existing_pred, str):
            existing_pred = existing_pred.lower()
        else:
            existing_pred = str(existing_pred).lower()
        existing_spec = PREDICATE_SPECIFICITY.get(existing_pred, 0)
        if specificity > existing_spec:
            key_to_best[key] = rel
    return list(key_to_best.values())
```

**Note:** `MedicalClaimRelationship` and the code use a `predicate` attribute (see line 813). So for `rel` we should use `getattr(rel, "predicate", None)` and if that’s None fall back to `get_edge_type()` if it exists. Checking the class: it’s `MedicalClaimRelationship` with `predicate`. So simplify the helper to use only `getattr(rel, "predicate", "")` and normalize to lowercase. If the relationship type doesn’t have `predicate`, the code that builds `rel` in this file sets `predicate=predicate`. So we can use:

```python
def _deduplicate_relationships_by_predicate_specificity(
    relationships: list[BaseRelationship],
) -> list[BaseRelationship]:
    """For each (subject_id, object_id), keep only the relationship with highest predicate specificity."""
    if not relationships:
        return relationships
    key_to_best: dict[tuple[str, str], BaseRelationship] = {}
    for rel in relationships:
        key = (rel.subject_id, rel.object_id)
        pred = (getattr(rel, "predicate", "") or "").strip().lower()
        specificity = PREDICATE_SPECIFICITY.get(pred, 0)
        existing = key_to_best.get(key)
        if existing is None:
            key_to_best[key] = rel
            continue
        existing_pred = (getattr(existing, "predicate", "") or "").strip().lower()
        existing_spec = PREDICATE_SPECIFICITY.get(existing_pred, 0)
        if specificity > existing_spec:
            key_to_best[key] = rel
    return list(key_to_best.values())
```

### 3b. Call the helper in _extract_with_llm

**Location:** In `_extract_with_llm`, after the loop that appends to `relationships` (after `if relationship: relationships.append(relationship)`) and **before** building `trace["final_relationships"]`.

**Find:**

```python
                    if relationship:
                        relationships.append(relationship)

            # Record final relationships in trace
            trace["final_relationships"] = [
```

**Replace with:**

```python
                    if relationship:
                        relationships.append(relationship)

            relationships = _deduplicate_relationships_by_predicate_specificity(relationships)

            # Record final relationships in trace
            trace["final_relationships"] = [
```

**Verification:** Run tests. For a run where the LLM returns both `associated_with` and `indicates` for the same (subject_id, object_id), `trace["final_relationships"]` must contain only one relationship for that pair (the one with predicate `indicates`).

---

## Step 4. 6.4 — Batch semantic evidence checks (specification for later)

**Goal:** Replace per-item semantic evidence validation with a two-pass flow: (1) collect items that need semantic check; (2) batch-embed all unique evidence and entity names; (3) run similarity for each item from precomputed embeddings. No per-item `generate()` calls for semantic checks.

**File:** `examples/medlit/pipeline/relationships.py`

**Order of implementation:**

1. **Refactor so semantic check can use precomputed embeddings.**  
   - Add a function (e.g. `_evidence_contains_both_entities_semantic_from_embeddings(evidence_emb, subject_emb, object_emb, threshold) -> tuple[bool, str | None, dict]`) that takes three embedding tuples and returns the same shape as `_evidence_contains_both_entities_semantic` (ok, drop_reason, detail with subject_in_evidence, object_in_evidence, subject_similarity, object_similarity).  
   - Optionally have `_evidence_contains_both_entities_semantic` call the embedding generator, then call this helper with the resulting embeddings, so existing call sites keep working until you switch to batch.

2. **In _extract_with_llm, two-pass flow.**  
   - **Pass 1:** Iterate over parsed items. For each item: resolve subject/object entities; run string evidence check only. If string passes, treat as accepted for evidence and continue to type/semantic-predicate/constraint checks (either inline or record for pass 2). If string fails and `evidence_drop_reason` is `evidence_missing_subject` or `evidence_missing_object`, record decision and skip (no semantic). If string fails and embedding_generator is set and reason is something else (e.g. `evidence_empty`, `evidence_missing_subject_and_object`), add (item, subject_entity, object_entity, evidence, decision_slot) to a list `needs_semantic`.  
   - **Batch embed:** Build list of unique strings: all evidence texts and all (subject_entity.name, object_entity.name) for `needs_semantic`. Call `embedding_generator.generate_batch(list_of_texts)`. Map each text to its embedding (same order as list_of_texts).  
   - **Pass 2 (or continue in same loop for semantic items):** For each entry in `needs_semantic`, get evidence_emb, subject_emb, object_emb from the map; call `_evidence_contains_both_entities_semantic_from_embeddings`; fill decision and either append to relationships or leave as rejected. Append all decisions to `trace["decisions"]` in the same order as items so trace structure is unchanged.

3. **Marker-disease special case:** Apply the same “context_inferred” logic (evidence names marker, disease in window, disease context words) before or after the batch semantic pass so that decisions remain consistent.

4. **Trace:** Each decision must still include `evidence_check` with `method`, and when semantic is used, `subject_similarity`, `object_similarity`, `threshold`. No change to trace schema.

**Verification:** Same acceptance/drop outcomes as before; fewer total `generate` or `generate_batch` calls per window (observe via logging or cache stats if available).

---

## Step 5. Verification (after 6.1–6.3)

From repo root:

```bash
uv run pytest examples/medlit/ -v -k "relationship or medlit" --tb=short 2>&1 | tail -40
```

Optional end-to-end (requires Ollama or mock):

```bash
uv run python -m examples.medlit.scripts.ingest --input-dir examples/medlit/pmc_xmls --limit 1 --use-ollama --stop-after relationships 2>&1 | tail -20
```

Inspect a trace file under `/tmp/kgraph-relationship-traces/` (or path from script): confirm 6.1 (no semantic when entity missing), 6.2 (no signature block in prompt), 6.3 (at most one predicate per (subject_id, object_id) in final_relationships).

---

## Summary

| Step | Item | File | Action |
|------|------|------|--------|
| 1 | 6.1 Skip semantic when entity missing | relationships.py | After string check, if drop_reason in (evidence_missing_subject, evidence_missing_object), set decision and return without calling semantic. |
| 2 | 6.2 Shorten prompt | relationships.py | Remove sig_lines build and PREDICATE SIGNATURES from prompt; replace type-matching line with server-side validation note; update examples line. |
| 3 | 6.3 Predicate hierarchy (Option B) | relationships.py | Add PREDICATE_SPECIFICITY and _deduplicate_relationships_by_predicate_specificity; call dedupe after building relationships in _extract_with_llm. |
| 4 | 6.4 Batch semantic | relationships.py | (Spec only) Two-pass + batch embed + semantic-from-embeddings helper; implement when ready. |
