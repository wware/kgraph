# What are Next Steps for this thing?

Following a bunch of work, I have this little review from ChatGPT.

## Small suggestions / potential follow-ups ⚠️ (not blockers)

1.  **`ValidationIssue.value` is typed as `str | None`**. [GitHub](https://github.com/wware/kgraph/commit/7c3d5778045027f79cda39110bd09819dc05253e)  
    That’s fine for display, but you may eventually want either:
    
    -   `value: Any | None` (so callers can inspect programmatically), *or*
        
    -   keep `value` as `str` but add `raw_value_type: str | None` (or similar) for better debugging.  
        Right now you’re committing to “value is pre-stringified,” which is okay, just a design choice to be aware of.
        
2.  Consider whether you want the same structured approach for **`validate_relationship`** eventually. The docs snippet you updated only mentions `validate_entity`’s return change; `validate_relationship` appears to still be a bool in the docs list. [GitHub+1](https://github.com/wware/kgraph/commit/7c3d5778045027f79cda39110bd09819dc05253e)  
    Totally fine to do entity first, relationships later—just flagging symmetry.
    
3.  The `field` attribute is a single string. [GitHub](https://github.com/wware/kgraph/commit/7c3d5778045027f79cda39110bd09819dc05253e)  
    That works, but if you later want nested/structured locations (like Pydantic’s `loc` tuples), you might want `field_path: tuple[str, ...]` (or `list[str]`) instead. You can defer this until you feel the pain.

At the present time I will not take action on these.

## Current Status (2026-02-04)

### ✅ Completed: Stage Models and --stop-after Support

Commit `2056bda` implemented:
- Pydantic models for each pipeline stage (`examples/medlit/stage_models.py`)
- `--stop-after {entities|promotion|relationships}` flag in ingest.py
- JSON output to stdout for debugging intermediate states
- Progress tracking with configurable reporting interval

**Usage:** (Legacy ingest.py removed; three-pass pipeline has no stop-after. For extraction state use Pass 1 only; promotion is not in the three-pass flow.)
```bash
# Pass 1 only: entity and relationship extraction per paper
uv run python -m examples.medlit.scripts.pass1_extract --input-dir papers/ --output-dir pass1_bundles --llm-backend ollama
# Pass 2/3: see run-ingest.sh and INGESTION.md
```

---

## Next: Relationship Extraction Tracing

**Goal:** Answer these questions for any paper without guessing:
- Did the LLM emit relationships?
- Were they parsed correctly?
- Which ones were dropped and why?
- Were any swapped?

### Action Plan

#### Task 1: Add `generate_json_with_raw()` to LLM Client

**File:** `examples/medlit/pipeline/llm_client.py`

**Changes:**
1. Add method to `LLMClientInterface` ABC:
   ```python
   async def generate_json_with_raw(
       self, prompt: str, temperature: float = 0.1
   ) -> tuple[dict[str, Any] | list[Any], str]:
       """Returns (parsed_json, raw_text). Default returns placeholder raw."""
       data = await self.generate_json(prompt, temperature)
       return data, "<raw unavailable>"
   ```

2. Override in `OllamaLLMClient`:
   - Refactor JSON parsing into `_parse_json_from_text(raw: str) -> Any`
   - Implement `generate_json_with_raw()` to return both parsed and raw

**Definition of Done:**
- [ ] `generate_json_with_raw()` exists and returns `(parsed, raw_text)`
- [ ] Existing `generate_json()` still works unchanged
- [ ] Unit test verifies raw text is captured

---

#### Task 2: Add Per-Paper Relationship Trace Files

**File:** `examples/medlit/pipeline/relationships.py`

**Changes to `_extract_with_llm()`:**

1. Create trace dict at start of method:
   ```python
   trace = {
       "document_id": document.document_id,
       "llm_model": getattr(self._llm, "model", None),
       "created_at": datetime.now(timezone.utc).isoformat(),
       "prompt": None,
       "raw_llm_output": None,
       "parsed_json": None,
       "decisions": [],
       "final_relationships": [],
   }
   ```

2. Call `generate_json_with_raw()` and capture both:
   ```python
   response, raw = await self._llm.generate_json_with_raw(prompt)
   trace["prompt"] = prompt
   trace["raw_llm_output"] = raw
   trace["parsed_json"] = response
   ```

3. For each candidate relationship, record a decision:
   ```python
   decision = {
       "item": item,  # Original LLM output
       "matched_subject": subject_entity is not None,
       "matched_object": object_entity is not None,
       "semantic_ok": True/False,
       "swap_applied": True/False,
       "accepted": True/False,
       "drop_reason": "subject_unmatched" | "semantic_mismatch" | None,
   }
   trace["decisions"].append(decision)
   ```

4. Write trace file at end:
   ```python
   trace_dir = Path("/tmp/kgraph-relationship-traces")
   trace_dir.mkdir(parents=True, exist_ok=True)
   trace_path = trace_dir / f"{document.document_id}.relationships.trace.json"
   trace_path.write_text(json.dumps(trace, indent=2))
   ```

**Definition of Done:**
- [ ] Running ingest creates trace files in `/tmp/kgraph-relationship-traces/`
- [ ] Each trace file contains: prompt, raw_llm_output, parsed_json, decisions, final_relationships
- [ ] Drop reasons are recorded for every rejected relationship

---

#### Task 3: Run Single-Paper Test and Analyze

**After Tasks 1-2 are complete:**

```bash
# Process one paper (ingest.py removed; use pass1_extract)
uv run python -m examples.medlit.scripts.pass1_extract \
  --input-dir examples/medlit/pmc_xmls/ \
  --output-dir pass1_bundles \
  --llm-backend ollama \
  --papers "PMC10759991.xml"

# Examine traces (if relationship tracing is re-enabled in pass1 pipeline)
cat /tmp/kgraph-relationship-traces/*.relationships.trace.json | jq '.decisions[] | select(.accepted == false)'
```

**Definition of Done:**
- [ ] Can identify dominant failure mode (entity matching? semantic filter? type constraints?)
- [ ] Document findings in this file

---

### Deferred (Do After Trace Analysis)

These are good ideas but should wait until we understand the actual failure modes:

- **Replay mode** (`--replay trace.json`) - iterate without LLM calls
- **Structured `ValidationResult`** with repair suggestions
- **Predicate signature table** generated from domain constraints in prompts
- **Entity name normalization** (fuzzy matching for "HER2 protein" vs "HER2")
- **Better JSON bracket parsing** (handle brackets inside quoted strings)





# THIS IS FROM A COUPLE DAYS AGO

> **Note:** This document was created before the major refactoring that split the codebase into kgschema/, kgbundle/, kgraph/, and kgserver/ packages. The general intent and approach remain valid, but file paths and module references have been updated to match the current architecture.

## Quick Reference: Current Module Locations

### kgraph Framework
- **Domain schemas (ABCs):** `kgschema/domain.py` (DomainSchema ABC)
- **Relationship models (ABCs):** `kgschema/relationship.py` (BaseRelationship)
- **Entity models (ABCs):** `kgschema/entity.py` (BaseEntity, EntityMention, etc.)
- **Ingestion orchestrator:** `kgraph/ingest.py` (IngestionOrchestrator)
- **Bundle export:** `kgraph/export.py`
- **Bundle models:** `kgbundle/models.py` (EntityRow, RelationshipRow, BundleManifestV1)
- **Storage backends:** `kgserver/storage/backends/postgres.py`, `kgserver/storage/backends/sqlite.py`
- **MCP server:** `kgserver/mcp_server/`

### examples/medlit (Current Implementation)
- **Domain implementation:** `examples/medlit/domain.py` (MedLitDomainSchema)
- **Entity models:** `examples/medlit/entities.py` (basic Disease, Gene, Drug, etc.)
- **Relationship models:** `examples/medlit/relationships.py` (MedicalClaimRelationship)
- **Relationship extraction:** `examples/medlit/pipeline/relationships.py` (MedLitRelationshipExtractor)
- **LLM client:** `examples/medlit/pipeline/llm_client.py` (LLMClientInterface, OllamaLLMClient)

### med-lit-schema (Advanced Medical Schema)
- **Location:** `/home/wware/kgraph/med-lit-schema/`
- **Entity models:** `med-lit-schema/entity.py` (Disease, Gene, Drug, Protein, Mutation, Symptom, Biomarker, etc.)
- **Relationship models:** `med-lit-schema/relationship.py` (Treats, Causes, Inhibits, etc. - 40+ types)
- **Base models:** `med-lit-schema/base.py` (PredicateType enum, EvidenceItem, Measurement)
- **Entity collection:** `med-lit-schema/entity.py` (InMemoryEntityCollection)
- **Storage:** `med-lit-schema/storage/backends/` (SQLite, PostgreSQL with domain/persistence separation)
- **Ingestion:** `med-lit-schema/ingest/` (PMC parsing, NER, claims, evidence pipelines)
- **Query server:** `med-lit-schema/query/server.py` (FastAPI with REST, GraphQL, MCP)

### Other
- **Trace output location:** `/tmp/kgraph-relationship-traces/` (if implemented)

## Implementation Status (Updated 2026-02-04)

**✅ Implemented:**
- LLM client abstraction (`examples/medlit/pipeline/llm_client.py`)
- Relationship extractor with domain validation (`examples/medlit/pipeline/relationships.py`)
- Domain schema with predicate constraints (`kgschema/domain.py`, `examples/medlit/domain.py`)
- Subject/object swapping logic based on type constraints
- **Stage models for pipeline debugging** (`examples/medlit/stage_models.py`)
- **`--stop-after` flag** to halt pipeline at any stage and dump JSON

**❌ Not Yet Implemented:**
- `generate_json_with_raw()` method in LLM client (see Task 1 above)
- Per-paper relationship trace JSON files (see Task 2 above)
- Replay mode for debugging without calling the LLM
- Structured `ValidationResult` with repair suggestions
- Predicate signature table exposed in prompts

## Near term

### Migrate examples/medlit to use med-lit-schema

> **See MEDLIT_SCHEMA_SPEC.md** for the complete work order, including:
> - Context and rationale for this migration
> - Detailed migration tasks and implementation steps
> - Benefits, risks, and timeline estimates
> - Phase-by-phase implementation plan

---

### Query the Knowledge Graph via Claude Code (MCP)

The best UI for querying the knowledge graph is conversational interaction through Claude Code using the MCP (Model Context Protocol) server. This setup is already configured and working!

**Prerequisites:**
1. PostgreSQL database running with the knowledge graph data
2. Knowledge graph bundle loaded into the database

**Setup Steps:**

1. **Start the database** (if not already running):
   ```bash
   cd /home/wware/kgraph
   docker compose up -d postgres
   ```

2. **Configure Claude Code MCP** by adding to `~/.claude/mcp.json`:
   ```json
   {
       "mcpServers": {
           "knowledge-graph": {
               "command": "/home/wware/.local/bin/uv",
               "args": [
                   "--directory", "/home/wware/kgraph/kgserver",
                   "run",
                   "python", "-m", "kgserver.mcp_server"
               ],
               "env": {
                   "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/kgserver"
               }
           }
       }
   }
   ```

   > **Architecture Update:** The MCP server is now in `kgserver/mcp_server/` and uses the separated kgschema/ and kgbundle/ packages.

3. **Restart Claude Code** to load the MCP server configuration

**Available MCP Tools:**
- `mcp__knowledge-graph__list_entities` - List entities with filtering
- `mcp__knowledge-graph__search_entities` - Search entities by name
- `mcp__knowledge-graph__get_entity` - Get entity by ID
- `mcp__knowledge-graph__find_relationships` - Find relationships
- `mcp__knowledge-graph__get_relationship` - Get specific relationship
- `mcp__knowledge-graph__get_bundle_info` - Get bundle metadata

**Example Queries:**
- "Please list the first ten entities"
- "Please list the first ten relationships"
- "Show me relationships as English sentences with common names"
- "Find all genes related to breast cancer"
- "What drugs target BRCA2?"

The MCP server runs as a subprocess and communicates via stdio, providing Claude Code with direct access to query the knowledge graph conversationally.

### Debug Relationship Ingestion Pipeline

Entity extraction is working well, but relationship extraction needs improvement. The next step is to ingest a single paper with detailed debug logging to understand the relationship extraction pipeline.

> **Architecture Update:** The relationship extraction is now in `examples/medlit/pipeline/relationships.py` using the `MedLitRelationshipExtractor` class, which implements the `RelationshipExtractorInterface` from `kgraph/pipeline/interfaces.py`. The LLM client abstraction is in `examples/medlit/pipeline/llm_client.py`.

**Goal:**
- Run ingestion on one paper with verbose debug output
- Examine how the LLM extracts relationships
- Identify issues with relationship quality, accuracy, or coverage
- Improve prompts, validation, or post-processing

**Tasks:**
1. Add comprehensive debug logging to relationship extraction pipeline
   - Log LLM prompts sent to the model
   - Log raw LLM responses before parsing
   - Log parsed relationships before validation
   - Log validation failures and reasons
   - Log final relationships written to output

   **Current State:** The relationship extractor is at `examples/medlit/pipeline/relationships.py:25-50+`. The LLM client interface is at `examples/medlit/pipeline/llm_client.py`.

2. For single-paper analysis use the three-pass pipeline with one paper:
   ```bash
   uv run python -m examples.medlit.scripts.pass1_extract --input-dir examples/medlit/pmc_xmls --output-dir pass1_bundles --llm-backend ollama --papers "PMC12748354.xml"
   ```
   (Legacy `examples/medlit/scripts/ingest.py` was removed in PLAN10.)

3. Analyze the debug output to identify:
   - Are relationships being extracted?
   - Are they semantically correct?
   - Are canonical IDs being mapped properly?
   - Are relationships being filtered/rejected incorrectly?
   - Is the LLM prompt giving good guidance?
   - **Pay special attention to:**
     - Entity type checking on predicates (e.g., "treats" should have drug→disease)
     - Automatic subject/object swapping when types are reversed
     - Are swaps happening correctly?
     - Are valid relationships being rejected due to type mismatches?
     - Are invalid relationships being accepted due to missing type constraints?

4. Iterate on improvements:
   - Refine LLM prompts for better relationship extraction
   - Improve relationship type vocabulary/schema
   - Add better validation rules
   - Enhance canonical ID resolution for relationship entities
   - Review and refine predicate type constraints in domain schema
   - Fix or improve automatic subject/object swapping logic

---

# Thoughts and comments from ChatGPT


Nice — this is a very “do the simplest thing that proves it works” roadmap. If you want the relationship pipeline to stop being a black box, the trick is to make *every stage* observable and replayable.

## 1) Make relationship extraction replayable (before you make it “better”)

### Add a “relationship trace” artifact per paper

Write a single JSON file alongside the output that captures the full lineage:

* paper id, model id/version, timestamp
* extracted entity list (the exact list passed into relation extraction)
* prompt (rendered) + inputs used to render it
* raw model response (verbatim)
* parsed relations (as structured objects)
* canonicalization results (per subject/object)
* validation results (pass/fail + reasons + suggested fixes)
* post-processing actions taken (swap, drop, normalize predicate, etc.)

This turns debugging into: “open one file, see exactly what happened.”

**Shape suggestion (one file per paper):**

```json
{
  "paper_id": "PMC12748354",
  "model": "gpt-4.1-mini",
  "stages": [
    {"name": "prompt", "data": {...}},
    {"name": "raw_response", "data": {...}},
    {"name": "parsed", "data": {...}},
    {"name": "canonicalized", "data": {...}},
    {"name": "validated", "data": {...}},
    {"name": "final", "data": {...}}
  ]
}
```

### Keep the prompt stable and version it

Put the relationship prompt in a dedicated module/file and give it a `PROMPT_VERSION = "rel-extract-v3"` constant that gets logged. You’ll thank yourself later when comparing runs.

---

## 2) Add “structured parse” as a hard requirement

If you’re not already doing this: require the model to output **strict JSON** that matches a schema. Do not accept “almost JSON”. If parsing fails, log and bail (for debug runs). Relationship extraction is *much* easier to reason about when parsing is deterministic.

Minimal relation object:

```json
{
  "predicate": "treats",
  "subject": {"name": "olaparib", "type": "drug"},
  "object": {"name": "breast cancer", "type": "disease"},
  "evidence": {"quote": "...", "offsets": [123, 210]}
}
```

Even if you later canonicalize to IDs, keep the original `name/type` because it’s what the model “thought.”

---

## 3) Make type constraints explicit and machine-checkable

You called out the key issue: predicate type checking (drug→disease, gene→disease, etc.). Two practical moves:

### A. Put predicate signatures in one table

Something like:

| predicate       | allowed (subject_type, object_type) | swap_allowed |
| --------------- | ----------------------------------- | ------------ |
| treats          | (drug, disease)                     | true         |
| associated_with | (gene, disease), (variant, disease) | true         |
| inhibits        | (drug, protein), (drug, gene)       | true         |
| expresses       | (gene, tissue)                      | false        |

Then validation is mechanical.

### B. Validate with **reasons** and **suggested repair**

When a relation fails, emit something like:

* `TYPE_MISMATCH: treats expects (drug->disease), got (disease->drug)`
* suggested repair: `SWAP_SUBJECT_OBJECT`

or

* `UNKNOWN_PREDICATE: "improves_outcomes"` → suggest mapping to canonical predicate `improves` or `associated_with`, or drop.

That “suggested repair” becomes your post-processor’s to-do list.

---

## 4) Implement swapping as a first-class transformation (and log it)

Swapping should not be an “if mismatch then swap” one-liner hidden in code. Treat it as a transformation step with:

* condition: predicate allows swap AND swapped types match a signature
* action: swap + annotate:

  * `transform: "swap"`
  * `reason: "treats expected drug->disease"`

Also: if both directions match (possible with symmetric predicates like `associated_with`) then swapping shouldn’t trigger; mark predicate as symmetric.

---

## 5) Your `ingest_single_debug` script should support “stop points”

For rapid iteration, make the script able to run *just one stage*:

* `--stage rel_prompt` (render prompt + dump)
* `--stage rel_call` (call LLM + dump raw)
* `--stage rel_parse`
* `--stage rel_validate`
* `--stage rel_write`

Plus a `--replay trace.json` mode that skips the model call and re-runs parse/validate/postproc against recorded raw output. That’s how you iterate quickly without paying latency/tokens every time.

---

## 6) Add 5 tiny tests that will pay off immediately

1. **Type validation happy path**
   `treats(drug, disease)` passes.

2. **Type mismatch swap**
   `treats(disease, drug)` becomes swapped and passes, with transform logged.

3. **Type mismatch no swap allowed**
   `expresses(tissue, gene)` fails with reason, no swap.

4. **Unknown predicate**
   Fails with reason, optionally maps if you allow a synonym map.

5. **Canonicalization failure**
   Subject resolves, object doesn’t → relation either dropped or kept with `unresolved_entity`, but you decide and test it.

These tests force you to define the policy decisions you’ll otherwise keep re-litigating while debugging.

---

## 7) Practical prompt tweak that often helps relationship quality

In the relation extraction prompt, require:

* **one evidence quote per relation** (short)
* **entity mentions must appear in the quote**
* **no relations without explicit textual support**

This single constraint dramatically reduces “LLM guessed it” relations and makes debugging easier because every relation comes with a grounded snippet.

---

## A tight “do next” checklist

1. Implement per-paper relationship trace JSON (prompt/raw/parsed/validated/final).
2. Make relationship extraction output strict JSON only.
3. Centralize predicate signatures + swap rules.
4. Implement swap as an explicit transformation step with logging.
5. Add `--replay` + `--stage` options to `ingest_single_debug`.
6. Add the 5 tests above.

If you paste (or point me at) the modules that currently do: (a) relation prompt creation, (b) parsing, (c) validation, I can sketch the exact seams to hook the trace logging and replay in, without turning the codebase inside out.

---

# ChatGPT continues...

Great — that’s a clean map of the seams. Here’s how I’d wire in **(1) trace logging + replay**, **(2) structured JSON hardening**, and **(3) swap/type validation visibility**, with minimal churn and clear ownership per module.

## A. One "RelationshipTrace" object, created in one place

### Where it should live

Create a tiny module in kgraph (so both examples + kgserver can reuse it):

* `kgraph/pipeline/trace.py`

> **Architecture Note:** This fits with the current structure where `kgraph/pipeline/` contains pipeline component interfaces (`kgraph/pipeline/interfaces.py`). The trace module would be cross-cutting infrastructure used by domain-specific extractors in `examples/*/pipeline/`.

It defines:

* a `RelationshipTrace` Pydantic model (or dataclass) with `stages: list[...]`
* helper methods: `add_stage(name, data)`, `write(path)`, `read(path)`

**Why here:** `examples/medlit/...` is domain-specific; `kgserver/...` is storage-specific. This is cross-cutting.

### What it captures (minimum viable)

Stages:

1. `inputs` — paper_id, model name, entities provided, doc chunking info
2. `prompt` — rendered prompt text + prompt version
3. `raw_response` — exact string returned by LLM
4. `parsed` — list of parsed relationship dicts (still names/types)
5. `canonicalized` — mapping results for each entity mention
6. `validated` — per-rel verdict + reasons + suggested repairs
7. `transformed` — swaps/normalizations applied
8. `final` — what gets handed to `IngestionOrchestrator`/storage

Write it to something like:

* `output/debug/traces/{paper_id}.relationships.trace.json`

---

## B. Relationship prompt + model call: instrument in one method

### Primary seam: `MedLitRelationshipExtractor._extract_with_llm`

> **Architecture Update:** This method is now in `examples/medlit/pipeline/relationships.py`. The class implements `RelationshipExtractorInterface` from `kgraph/pipeline/interfaces.py`.

This is the right place to:

* attach `trace.add_stage("prompt", ...)`
* attach `trace.add_stage("raw_response", ...)`
* enforce strict JSON output (fail fast in debug mode)
* attach `trace.add_stage("parsed", ...)`

#### Suggested minimal edits

Add optional parameters:

* `trace: RelationshipTrace | None = None`
* `debug: bool = False`
* `replay_raw_path: Path | None = None` (or `replay_trace_path`)

Pseudo-flow:

1. render prompt → trace stage
2. if replay: load raw response from file/trace
3. else call LLM → trace stage
4. parse strict JSON → trace stage
5. return parsed objects + trace

**Important:** don’t bury printing in log spam; always put artifacts in the trace file. Logging can then just say “wrote trace: …”.

---

## C. Parsing: split “parse” from “convert to BaseRelationship”

Right now `_extract_relationships_with_llm` parses JSON into `MedicalClaimRelationship` objects. Keep that, but introduce an intermediate “raw” model:

* `MedLitExtractedRelationship` (Pydantic)

  * subject: {name, type}
  * object: {name, type}
  * predicate
  * evidence quote
  * optional confidence/section

Then convert:

`MedLitExtractedRelationship` → `MedicalClaimRelationship`/`BaseRelationship`

**Why:** validation + swapping need to see the original types/names independent of canonical IDs and builder decisions.

So in that method:

* parse into `MedLitExtractedRelationship[]`
* log those into trace stage `parsed`
* conversion happens later in orchestration (or right after parse, but still preserve the raw list)

---

## D. Validation: make it return *structured reasons*, not just pass/fail

### Primary seam: `DomainSchema.validate_relationship`

> **Architecture Update:** The `DomainSchema` abstract base class is in `kgschema/domain.py`, and the medical literature implementation is in `examples/medlit/domain.py` as `MedLitDomainSchema`. The `PredicateConstraint` model already exists in `kgschema/domain.py:59-72`.

Right now it likely returns bool / raises / returns maybe list of errors. Make it (or add a sibling method) that returns a structured result:

* `ValidationResult`

  * `ok: bool`
  * `errors: list[ValidationErrorDetail]`
  * `suggested_repairs: list[RepairAction]` (e.g., SWAP, NORMALIZE_PREDICATE, DROP)

**Do this without breaking callers:**

* keep `validate_relationship(rel) -> bool` for compatibility
* add `validate_relationship_detailed(rel) -> ValidationResult`
* have `validate_relationship` call the detailed method and return `.ok`

Then, in `IngestionOrchestrator.extract_relationships_from_document`:

> **Architecture Update:** The `IngestionOrchestrator` is now in `kgraph/ingest.py`. However, note that the current architecture has the relationship extractor itself calling the domain validation, not the orchestrator. See `examples/medlit/pipeline/relationships.py` for where validation and swapping currently happen.

* call detailed validation
* `trace.add_stage("validated", ...)`
* if suggested repair includes `SWAP` and swap produces a valid signature, apply it and record `transformed`

This is where your "type checking on predicates" and "automatic swapping" becomes transparent.

---

## E. Swapping: treat it as a transformation step with audit

### Where it belongs

Either:

* in `DomainSchema` (domain owns type rules), OR
* in `RelationshipBuilder` (builder owns shape), BUT…

I’d put the *decision* in `DomainSchema` (because that’s where predicate signatures live) and the *application* in the orchestrator/builder.

Concretely:

* `DomainSchema.suggest_repairs(rel)` → returns `[RepairAction.SWAP]` if:

  * predicate has swap_allowed
  * swapped types match an allowed signature
* orchestrator applies swap and logs to trace:

  * before/after + reason

That avoids hidden behavior inside builder constructors.

---

## F. Storage: don't lose debugging context when loading JSONL

> **Architecture Update:** The storage backends are now in `kgserver/storage/backends/`. The bundle loading is in `kgserver/query/bundle_loader.py`. The `RelationshipRow` model is in the lightweight `kgbundle/models.py` package.

Your `kgserver/storage/backends/*._load_relationships` does Pydantic validation (`RelationshipRow.model_validate`).

Two small upgrades:

1. If `RelationshipRow` allows an optional `provenance` or `debug` field, include:

   * `source_paper_id`
   * `extraction_trace_path` (or trace id)
   * `evidence_quote`

2. On `ValidationError`, include *which file + line number + relationship id* in the exception message or logs.

This makes “it got rejected on load” debuggable without reproducing the run.

---

## G. Replay mode: cheapest win for iteration speed

Add to your script:

```bash
# Replay/debug: legacy ingest_single_debug was part of removed ingest. For single-paper debug use pass1_extract; replay/validate would need to be reimplemented in the pass1 pipeline if desired.
uv run python -m examples.medlit.scripts.pass1_extract --input-dir examples/medlit/pmc_xmls --output-dir pass1_bundles --llm-backend ollama --papers "PMC12748354.xml"
```

Implementation approach (if adding replay to pass1):

* A replay helper would load the trace
* re-runs parse/validate/transform steps on recorded `raw_response` or recorded `parsed`
* never hits the model

This lets you iterate on:

* parsing rules
* predicate signature table
* swap logic
* validation policies

…without burning model calls.

---

## H. Exact "touch points" in your listed modules

> **Architecture Update:** All module paths have been verified and updated to match the current codebase structure.

### `examples/medlit/pipeline/relationships.py` (✓ Exists)

* Add `trace` + `replay` capability to `_extract_with_llm` method
* Parse into `MedLitExtractedRelationship[]` first (intermediate model before `MedicalClaimRelationship`)
* Record `prompt`, `raw_response`, `parsed`
* Current file: `examples/medlit/pipeline/relationships.py:25+`

### `kgraph/ingest.py` (✓ Exists - `IngestionOrchestrator`)

* Owns: `validated` + `transformed` + `final` stages
* Calls `domain.validate_relationship_detailed`
* Applies repairs (swap, normalize predicate)
* Records everything
* **Note:** Currently, validation happens in the extractor, not the orchestrator. May need to refactor.

### `kgschema/domain.py` + `examples/medlit/domain.py` (✓ Exist)

* Add `validate_relationship_detailed` method (currently only has async `validate_relationship`)
* Centralize predicate signatures (already has `PredicateConstraint` model at `kgschema/domain.py:59-72`)
* Make errors machine-readable (codes + context)
* Implementation: `examples/medlit/domain.py` as `MedLitDomainSchema`

### `kgraph/builders.py` (✓ Exists)

* If it currently validates, either:
  * leave it, but make it accept pre-validated relations
  * or have it optionally accept `validation_result` to avoid double work
* Ensure builder doesn't silently swap; swapping should already be done

### `kgserver/storage/backends/*.py` (✓ Exist)

* Files: `kgserver/storage/backends/postgres.py`, `kgserver/storage/backends/sqlite.py`
* Improve error context on JSONL load
* Optionally preserve provenance fields for later inspection
* Bundle loading happens in `kgserver/query/bundle_loader.py`

---

## What I’d do first (2–3 hour slice)

1. Implement `RelationshipTrace` + write/read helpers.
2. Instrument `_extract_relationships_with_llm` to emit:

   * prompt + raw response + parsed list
3. Add `validate_relationship_detailed` with:

   * error codes for predicate unknown + type mismatch
   * suggested repair = SWAP when applicable
4. In `IngestionOrchestrator`, log validation results + apply swap + log transform.
5. Make `ingest_single_debug` write the trace JSON and support `--replay`.

That’s enough to answer, with certainty, these questions you listed:

* “Are relationships being extracted?”
* “Are they being rejected due to type mismatches?”
* “Are swaps happening, and are they correct?”

---

If you paste the current `DomainSchema.validate_relationship` and the part of `_extract_relationships_with_llm` that builds/parses the prompt/JSON (just those functions), I can give you concrete patch-style edits (signatures + exact stage payloads) that match your existing types.

---

# Summary of Architecture Changes

> **Added during post-refactoring update:**

The original discussion with ChatGPT assumed a single monolithic codebase. The refactoring introduced clear separation:

1. **kgschema/** - Pure data models (BaseEntity, BaseRelationship, DomainSchema ABCs, PredicateConstraint)
2. **kgbundle/** - Lightweight exchange format (BundleManifestV1, EntityRow, RelationshipRow)
3. **kgraph/** - Framework code (IngestionOrchestrator, promotion, export, builders)
4. **kgserver/** - Query server (FastAPI, MCP, GraphQL, storage backends)
5. **examples/medlit/** - Medical literature domain implementation
6. **examples/sherlock/** - Literary domain implementation

The key insight is that:
- **Domain-specific** code lives in `examples/*/`
- **Cross-cutting infrastructure** lives in `kgraph/`
- **Pure schemas** live in `kgschema/`
- **Query/serving** lives in `kgserver/`

This separation is preserved throughout the recommendations below, but the specific file paths have been updated accordingly.

---

# okay

This is super actionable — you already have *two* different “swap/type mismatch” mechanisms, and right now they fight each other:

* `DomainSchema.validate_relationship()` detects "looks swapped" but **returns False** (and only logs).
* `MedLitRelationshipExtractor._extract_with_llm()` does a pre-swap check (`_should_swap_subject_object`) and **actually swaps**.

> **Architecture Update:** These references are now to `kgschema/domain.py` (base) and `examples/medlit/domain.py` (implementation) for validation, and `examples/medlit/pipeline/relationships.py` for the extractor.

So depending on where validation happens and what entity types are available at that moment, you can end up either:

* swapping “early” and passing, or
* *not* swapping and then getting rejected later with a warning that it *could* have been swapped.

Below is the cleanest refactor that makes this deterministic and traceable with minimal edits.

---

## 1) Fix the policy bug in `DomainSchema.validate_relationship`: “unknown types ⇒ True”

This line is going to hide a ton of real failures:

```py
if subject_type is None or object_type is None:
    ... 
    return True  # Allow for now if types can't be determined
```

For relationship extraction debugging, you want the opposite: **unknown types should be invalid (or “unknown”)**, not silently accepted.

### Minimal change

Add a `strict: bool = False` parameter (default False so you don’t break existing behavior), and in debug runs call it with `strict=True`.

```py
async def validate_relationship(..., strict: bool = False) -> bool:
    ...
    if subject_type is None or object_type is None:
        logger.warning(...)
        return not strict
```

That one change will immediately surface whether “types can’t be determined” is the reason relationships are slipping through.

---

## 2) Add a detailed validator (without breaking existing callers)

Keep `validate_relationship()` but add:

* `validate_relationship_detailed()` returning a structured result with:

  * `ok`
  * `errors` (codes + context)
  * `suggested_repairs` (SWAP, etc.)

### Why this matters *right now*

Your base validator already computes the swapped validity:

```py
swapped_subject_valid = object_type in constraints.subject_types
swapped_object_valid = subject_type in constraints.object_types
```

…but it throws away the computation and just logs + `return False`.

Instead, return “invalid, but swappable.”

**Sketch (small, contained):**

```py
class RelationshipValidationIssue(BaseModel, frozen=True):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)

class RelationshipRepair(BaseModel, frozen=True):
    action: str  # "swap_subject_object"
    reason: str

class RelationshipValidationResult(BaseModel, frozen=True):
    ok: bool
    issues: tuple[RelationshipValidationIssue, ...] = ()
    repairs: tuple[RelationshipRepair, ...] = ()
    inferred_types: tuple[str, str] | None = None
```

Then:

* if swapped would work, return `ok=False` with `repairs=("swap_subject_object", ...)`.

Your orchestrator (or extractor) can apply repairs and revalidate.

---

## 3) Decide: who owns swapping?

Right now swapping exists in **two** layers:

* extractor swaps (using entities it has in hand)
* domain validator *suggests* swap (but can’t apply)

You want **one owner**. The cleanest with your current architecture:

### Let swapping happen in the **relationship extractor**, and make the domain validator **never suggest swap** unless you explicitly ask.

Because in `_extract_with_llm` you *do* have `BaseEntity` objects (types are known). In `DomainSchema.validate_relationship`, you might only have IDs and need storage lookups.

So I’d do:

* In `MedLitRelationshipExtractor._extract_with_llm`: keep swap, but record it (see trace section).
* In `DomainSchema.validate_relationship`: change swapped-case log message to **DEBUG** and return False as it does now, *or* return a detailed “swappable” result if you adopt the detailed validator.

But don’t keep two independent swap policies.

If you prefer the opposite (domain owns swap policy), then remove `_should_swap_subject_object` from extractor entirely, and have the orchestrator apply repairs after validation. That’s also fine — just pick one.

Given your current code, **extractor-owned swap is the least invasive**.

---

## 4) Make the LLM extraction prompt align with the actual constraint engine

In the prompt you hardcode rules like:

* `increases_risk: gene/ethnicity → disease`

But the *actual* truth lives in `MedLitDomainSchema.predicate_constraints`. If these diverge, you’ll get garbage and/or false rejects.

### Minimal improvement

Generate the “PREDICATE TYPE RULES” section from `self._domain.predicate_constraints` at runtime:

* For each predicate:

  * subject types list
  * object types list
  * (optionally) an example

That guarantees the model sees the same constraints the validator enforces.

---

## 5) The biggest immediate bug in `_extract_with_llm`: entity matching by name only

This line is a quiet relationship-killer:

```py
entity_by_name = {e.name.lower(): e for e in entities}
...
subject_entity = entity_by_name.get(subject_name.lower())
```

If the LLM returns `HER2 protein` but your entity is `HER2`, you miss it and drop the relationship.

### Minimal fix (no embeddings, no fancy)

Build a small alias map:

* exact name
* name stripped of punctuation
* name with common suffixes removed (`protein`, `gene`, `mutation`, etc.)
* maybe `upper()` forms for genes

Example approach:

```py
def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\b(protein|gene|mutation|variant|pathway)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

Then map `norm(e.name)` to entity.

This alone will increase extracted relationships a lot and reduce “none found” cases.

---

## 6) Add trace logging exactly where it matters in your current code

You can do this without introducing new framework pieces yet.

### In `_extract_with_llm`, capture:

* `entity_list` (or at least entity ids/names/types)
* `text_sample` length + origin (abstract vs content)
* `prompt`
* `response` from `generate_json` **and** raw text (if your client can provide it)
* parsed items list
* per-item resolution: subject/object string → matched entity_id or “unmatched”
* per-item validation decisions:

  * semantic check pass/fail
  * swap applied yes/no and why

Even just dumping a JSON file with this data for one paper will tell you immediately whether you’re losing relationships due to:

* entity matching,
* predicate naming,
* semantics filter,
* swapping,
* validator reject.

---

## 7) One concrete change you should make right now in `DomainSchema.validate_relationship`

This behavior is unintentionally forgiving:

> “Could not determine types … Skipping type validation … return True”

That makes the type constraint system *look like it’s working* while it’s often not being applied.

For relationship debugging, you want a hard fail or at least an explicit “UNKNOWN_TYPES” failure record.

So: add the `strict` flag as above, and for your debug ingestion path set it `True`.

---

## 8) Quick “next patch” checklist (in your code as shown)

1. **DomainSchema**

   * add `strict` to `validate_relationship`
   * (optional) add `validate_relationship_detailed` returning errors/repairs

2. **MedLitRelationshipExtractor**

   * generate predicate type rules from domain constraints
   * improve entity matching with normalization/aliases
   * record swap actions and resolution failures (trace file)

3. **ingest_single_debug script**

   * enable strict validation
   * write the per-paper trace JSON

---

If you paste your `LLMClientInterface.generate_json()` implementation (and whether it can provide the raw model text), I can show the exact instrumentation lines to capture raw output vs parsed JSON — that’s usually the single most useful debug artifact.

# got it

Perfect — you *can* capture “raw output” today with almost no disruption, because `generate_json()` already has `response_text` before parsing. Right now that raw text is thrown away, which makes relationship debugging much harder than it needs to be.

Below are the **highest-leverage changes** I’d make, in order, and why.

---

## 1) Add an optional “return raw text too” API (smallest change, biggest payoff)

### Problem

`generate_json()` returns parsed JSON only. `_extract_with_llm` can’t log the exact LLM response that produced a parse/validation failure.

### Minimal fix

Add a new method (don’t break callers):

```py

class LLMClientInterface(ABC):
    ...
    async def generate_json_with_raw(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any] | list[Any], str]:
        data = await self.generate_json(prompt, temperature)
        return data, "<raw unavailable>"
```

Then override it in `OllamaLLMClient`:

```py
async def generate_json_with_raw(self, prompt: str, temperature: float = 0.1):
    response_text = await ...  # same as generate_json
    raw = response_text.strip()
    parsed = self._parse_json_from_text(raw)  # extract into helper
    return parsed, raw
```

Now `MedLitRelationshipExtractor._extract_with_llm()` can do:

```py
response, raw = await self._llm.generate_json_with_raw(prompt)
trace["raw_response"] = raw
trace["parsed"] = response
```

That alone will let you answer: “Did the model comply? Did it include extra prose? Did it emit malformed JSON? Did it output entity names that don’t match?”

---

## 2) Split JSON parsing into a helper and make it safer

### Your current parser has a subtle correctness bug

This function doesn’t respect JSON strings:

```py
def find_matching_bracket(text, start, open_char, close_char):
    count = 1
    ...
    if text[i] == open_char: count += 1
    elif text[i] == close_char: count -= 1
```

If the LLM includes evidence like:
`"evidence": "… [see Table 1] …"`
your bracket counter can get confused and cut early/late.

### Minimal improvement (still simple)

Add “inside string” tracking and escape handling. You don’t need a full JSON parser, just a bracket matcher that ignores bracket chars inside strings.

Pseudo:

* track `in_string: bool`
* track `escape: bool`
* only count brackets when `not in_string`

This *dramatically* reduces “No valid JSON found” false negatives.

---

## 3) Stop lying in the system prompt (it says entity extraction expert)

Right now your JSON method always uses:

```py
"You are a biomedical entity extraction expert..."
```

But you’re calling it for relationship extraction too. That prompt mismatch can absolutely hurt output quality and adherence.

### Best minimal change

Let the caller pass a `system_prompt` override:

```py
async def generate_json(..., system_prompt: str | None = None)
```

Default stays what you have, but in relationships code you pass:

* `"You extract biomedical relationships. Return ONLY valid JSON as requested."`

This will reduce “entities-only” bias and improve predicate consistency.

---

## 4) Make `generate_json()` optionally strict: no “best effort” extraction by bracket hunting

Bracket hunting is nice in production, but for debug runs you want to see failures clearly.

Add `strict: bool = False`:

* `strict=True` tries `json.loads(raw)` first (after stripping ``` fences)
* if that fails, raise with the raw text (or store it in trace)

In debug runs, this prevents “we parsed *some* JSON fragment that isn’t actually the intended output.”

---

## 5) Instrument relationship extraction with `raw_response` + “resolution diagnostics”

Given your `_extract_with_llm` pipeline, the most common relationship drop reasons will be:

1. subject/object name doesn’t map to an extracted entity (`entity_by_name` miss)
2. predicate not in domain vocabulary (or typos)
3. semantic mismatch filter rejects it
4. type constraints swapped / invalid
5. JSON parse partial / wrong shape

So for each candidate relationship dict, write a “decision record”:

```json
{
  "item": {... original dict ...},
  "resolved": {"subject": "hit|miss", "object": "hit|miss"},
  "semantic_ok": true,
  "swap_applied": false,
  "final_action": "accepted|dropped",
  "drop_reason": "object_unmatched|semantic_mismatch|..."
}
```

This is the single easiest way to learn where you’re bleeding relationships.

---

## 6) One more thing: name normalization is *more important* than better prompts

Right now you do exact lookup on `e.name.lower()`. That guarantees you’ll miss lots of legit relations.

Once you have the `raw_response`, you’ll see this immediately: model outputs “HER2 protein” vs entity “HER2”.

So after you add `raw_response`, do the cheap normalization/alias map next. It will probably multiply relationship yield.

---

# Concrete edits I’d apply immediately (summary)

### In `LLMClientInterface`

* add `generate_json_with_raw()`
* optionally add `system_prompt` parameter in `generate_json()`

### In `OllamaLLMClient`

* refactor JSON parsing into `_parse_json_from_text(raw: str) -> Any`
* implement `generate_json_with_raw()` returning `(parsed, raw)`
* improve bracket matching to ignore quoted strings

### In `MedLitRelationshipExtractor._extract_with_llm`

* call `generate_json_with_raw`
* dump: prompt, raw, parsed
* record per-item drop reasons and swap actions into a trace JSON file

---

If you want, paste just the `LLMClientInterface` file path + the `MedLitRelationshipExtractor._extract_with_llm` call site you want to modify (where it calls `generate_json(prompt)`), and I’ll write the exact patch blocks (diff-style) for the `generate_json_with_raw` addition + a safe bracket matcher.

---

# And on and on

> I see a pattern here, where I give you one or two files and you generate a half dozen or a dozen to-do items and then say, show me one or two other files. If we keep doing this I will have hundreds of to-do items and no idea which of them I should really do. How can we break this cycle before that happens?

That’s a *really* good meta-question, and you’re absolutely right about the pattern. What’s happening isn’t that the project needs hundreds of things — it’s that I’m operating in **“architectural review mode”**, which naturally explodes into possibilities.

Let’s break the cycle deliberately.

---

## The real problem (diagnosis)

You’re not missing ideas.
You’re missing **a stopping rule**.

Right now:

* You show a file
* I enumerate *everything that could be improved*
* Nothing tells us **which improvements are necessary to answer your current question**
* So the list grows unbounded

This is exactly how senior engineers burn weeks without shipping.

---

## Reframe the goal (this is the key move)

You are **not** trying to:

> “Make relationship ingestion good”

You are trying to:

> **Answer one concrete question with evidence**

> **“Why are relationship extractions not behaving as expected, and where are they being lost or mangled?”**

Everything else is noise until that question is answered.

So we need to agree on:

1. **One question**
2. **One experiment**
3. **One artifact**
4. **One decision point**

Anything that doesn’t directly support *that* gets postponed.

---

## A hard constraint we should adopt

Let’s adopt this rule explicitly:

> **No new to-do items unless they are required to complete the current experiment.**

If I violate that, you should stop me.

---

## The single experiment we should run

### Experiment goal

> Run ingestion on **one paper**, and produce a **single JSON file** that explains — line by line — what happened to every candidate relationship.

Not “fix it”.
Not “improve prompts”.
Just **observe**.

### Success criterion

You can open one file and answer:

* Did the LLM emit relationships?
* Were they parsed?
* Were they dropped?
* Why?
* Were any swapped?
* Were any invalid per domain rules?

If we can’t answer those questions, *nothing else matters*.

---

## The single artifact we should produce

**One file**, e.g.:

```
PMC12748354.relationship_trace.json
```

Containing (minimum):

```json
{
  "paper_id": "...",
  "prompt": "...",
  "raw_llm_output": "...",
  "parsed_items": [...],
  "decisions": [
    {
      "item": {...},
      "matched_subject": true,
      "matched_object": false,
      "semantic_ok": true,
      "swap_applied": false,
      "accepted": false,
      "reason": "object_name_not_resolved"
    }
  ],
  "final_relationships": [...]
}
```

That’s it.

No schemas, no replay modes, no refactors yet.

---

## The **only** to-do items that matter right now

Here is the *entire* to-do list — capped at **3 items**:

### ✅ TODO 1 (mandatory)

**Capture raw LLM output**

Modify `OllamaLLMClient.generate_json()` (or add `generate_json_with_raw`) so `_extract_with_llm` can access:

* the raw text returned by Ollama
* the parsed JSON (if any)

> Without this, you are blind.

---

### ✅ TODO 2 (mandatory)

**Log per-relationship decision outcomes**

In `_extract_with_llm`, for each candidate relationship:

* subject matched? (yes/no)
* object matched? (yes/no)
* semantic check passed?
* swap applied?
* accepted or dropped?
* drop reason

Append those to a list and write them to disk.

> This replaces speculation with facts.

---

### ✅ TODO 3 (mandatory)

**Run ingestion on exactly one paper**

* No batch mode
* No retries
* No fixes yet

Just generate the trace file.

---

## Explicitly postponed (write this down)

We are **not** doing these yet:

* prompt refinement
* predicate vocabulary redesign
* domain validation refactors
* replay tooling
* better JSON parsing
* alias resolution improvements
* swapping policy redesign

Not because they’re bad ideas — but because **we don’t yet know which one is actually needed**.

---

## How I’ll change how I respond (commitment)

From this point on, I will:

* Ask **“Does this help complete the current experiment?”**
* If not, I will explicitly say **“postponed”**
* I will not ask for more files unless:

  * a TODO above cannot be completed without them

If I suggest more than **one new task**, you should stop me.

---

## What I need from you (one choice)

Pick **one** of these and say it explicitly:

1. **“Let’s do the single-paper trace experiment.”**
   → I will give you *only* the exact code changes needed for TODO 1 and TODO 2, nothing else.

2. **“I want to redefine the goal first.”**
   → We’ll write a 2–3 sentence “definition of done” and freeze it.

Either way, we break the cycle *now*.

You’re right to call this out — this is exactly the moment where good projects either converge or sprawl.

---

# Let's do option 1

Cool. Here are **only** the exact code changes for:

* **TODO 1:** capture raw LLM output
* **TODO 2:** write a per-item decision trace JSON file from `_extract_with_llm`

No extra refactors, no new architecture.

---

# Patch 1 — add `generate_json_with_raw()` to the LLM client

Apply this to your `llm_client.py` (whatever its actual path is in your tree — it’s the file you pasted).

### What this does

* Adds a new optional API: `generate_json_with_raw(prompt) -> (parsed, raw_text)`
* Implements it in `OllamaLLMClient` by reusing your current logic
* Leaves existing callers of `generate_json()` unchanged

````diff
diff --git a/examples/medlit/pipeline/llm_client.py b/examples/medlit/pipeline/llm_client.py
index 1111111..2222222 100644
--- a/examples/medlit/pipeline/llm_client.py
+++ b/examples/medlit/pipeline/llm_client.py
@@ -1,10 +1,12 @@
 """LLM client abstraction for entity and relationship extraction.
 
 Provides a unified interface for Ollama LLM integration with tool calling support.
 """
 
 import asyncio
 from abc import ABC, abstractmethod
 from typing import Any, Callable, Optional
 import json
+from typing import Tuple
 
 try:
     import ollama
 
@@ -36,6 +38,21 @@ class LLMClientInterface(ABC):
         Raises:
             ValueError: If response is not valid JSON.
         """
 
+    async def generate_json_with_raw(
+        self,
+        prompt: str,
+        temperature: float = 0.1,
+    ) -> Tuple[dict[str, Any] | list[Any], str]:
+        """Generate structured JSON response AND return the raw model text.
+
+        Default implementation calls generate_json() and returns a placeholder raw text.
+        Subclasses should override if they can provide the raw response.
+        """
+        data = await self.generate_json(prompt, temperature)
+        return data, "<raw unavailable>"
+
     async def generate_json_with_tools(
         self,
         prompt: str,
@@ -94,6 +111,60 @@ class OllamaLLMClient(LLMClientInterface):
         self.timeout = timeout
         self._client = ollama.Client(host=host, timeout=timeout)
 
+    def _parse_json_from_text(self, response_text: str) -> dict[str, Any] | list[Any]:
+        """Extract and parse a JSON array/object from response text using existing heuristics."""
+        response_text = response_text.strip()
+
+        # Remove markdown code blocks if present
+        if response_text.startswith("```"):
+            lines = response_text.split("\n")
+            response_text = "\n".join(lines[1:-1]).strip()
+
+        def find_matching_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
+            count = 1
+            i = start + 1
+            while i < len(text) and count > 0:
+                if text[i] == open_char:
+                    count += 1
+                elif text[i] == close_char:
+                    count -= 1
+                i += 1
+            return i if count == 0 else -1
+
+        # Try to find JSON array first
+        json_start = response_text.find("[")
+        if json_start != -1:
+            json_end = find_matching_bracket(response_text, json_start, "[", "]")
+            if json_end != -1:
+                json_text = response_text[json_start:json_end]
+                try:
+                    return json.loads(json_text)
+                except json.JSONDecodeError:
+                    pass
+
+        # Try to find JSON object
+        json_start = response_text.find("{")
+        if json_start != -1:
+            json_end = find_matching_bracket(response_text, json_start, "{", "}")
+            if json_end != -1:
+                json_text = response_text[json_start:json_end]
+                try:
+                    return json.loads(json_text)
+                except json.JSONDecodeError:
+                    pass
+
+        raise ValueError(f"No valid JSON found in response: {response_text[:200]}")
+
     async def generate(
         self,
         prompt: str,
@@ -130,6 +201,44 @@ class OllamaLLMClient(LLMClientInterface):
     async def generate_json(
         self,
         prompt: str,
         temperature: float = 0.1,
     ) -> dict[str, Any] | list[Any]:
         """Generate structured JSON response from a prompt."""
 
         def _generate():
             response = self._client.chat(
                 model=self.model,
                 messages=[
                     {
                         "role": "system",
                         "content": "You are a biomedical entity extraction expert. Extract entities and return ONLY valid JSON in the exact format requested. Never return the original text, only the extracted entities.",
                     },
                     {"role": "user", "content": prompt},
                 ],
                 options={"temperature": temperature},
             )
             return response["message"]["content"]
 
         try:
             # Add timeout wrapper to prevent indefinite hangs
             response_text = await asyncio.wait_for(asyncio.to_thread(_generate), timeout=self.timeout)
         except asyncio.TimeoutError:
             raise TimeoutError(f"Ollama JSON generation timed out after {self.timeout}s")
-        response_text = response_text.strip()
-
-        # Remove markdown code blocks if present
-        if response_text.startswith("```"):
-            lines = response_text.split("\n")
-            response_text = "\n".join(lines[1:-1])
-
-        # Find and extract first complete JSON structure
-        def find_matching_bracket(text: str, start: int, open_char: str, close_char: str) -> int:
-            """Find the matching closing bracket for an opening bracket."""
-            count = 1
-            i = start + 1
-            while i < len(text) and count > 0:
-                if text[i] == open_char:
-                    count += 1
-                elif text[i] == close_char:
-                    count -= 1
-                i += 1
-            return i if count == 0 else -1
-
-        # Try to find JSON array first
-        json_start = response_text.find("[")
-        if json_start != -1:
-            json_end = find_matching_bracket(response_text, json_start, "[", "]")
-            if json_end != -1:
-                json_text = response_text[json_start:json_end]
-                try:
-                    return json.loads(json_text)
-                except json.JSONDecodeError:
-                    pass
-
-        # Try to find JSON object
-        json_start = response_text.find("{")
-        if json_start != -1:
-            json_end = find_matching_bracket(response_text, json_start, "{", "}")
-            if json_end != -1:
-                json_text = response_text[json_start:json_end]
-                try:
-                    return json.loads(json_text)
-                except json.JSONDecodeError:
-                    pass
-
-        raise ValueError(f"No valid JSON found in response: {response_text[:200]}")
+        return self._parse_json_from_text(response_text)
+
+    async def generate_json_with_raw(
+        self,
+        prompt: str,
+        temperature: float = 0.1,
+    ) -> tuple[dict[str, Any] | list[Any], str]:
+        """Same as generate_json(), but also returns raw model output text."""
+
+        def _generate():
+            response = self._client.chat(
+                model=self.model,
+                messages=[
+                    {
+                        "role": "system",
+                        "content": "You are a biomedical entity extraction expert. Extract entities and return ONLY valid JSON in the exact format requested. Never return the original text, only the extracted entities.",
+                    },
+                    {"role": "user", "content": prompt},
+                ],
+                options={"temperature": temperature},
+            )
+            return response["message"]["content"]
+
+        try:
+            response_text = await asyncio.wait_for(asyncio.to_thread(_generate), timeout=self.timeout)
+        except asyncio.TimeoutError:
+            raise TimeoutError(f"Ollama JSON generation timed out after {self.timeout}s")
+
+        raw = response_text.strip()
+        parsed = self._parse_json_from_text(raw)
+        return parsed, raw
````

---

# Patch 2 — write a single-paper relationship trace from `_extract_with_llm`

Apply this to `examples/medlit/pipeline/relationships.py` (the file you pasted).

### What this does

* Calls `generate_json_with_raw()` (falls back safely if not available)
* Writes one JSON trace file per paper to:

`/tmp/kgraph-relationship-traces/<document_id>.relationships.trace.json`

* Captures:

  * prompt
  * raw LLM output
  * parsed JSON (as returned by client)
  * per-item decisions (match/mismatch/semantic/swap/drop reason)
  * final accepted relationships (IDs/predicate/confidence)

```diff
diff --git a/examples/medlit/pipeline/relationships.py b/examples/medlit/pipeline/relationships.py
index 3333333..4444444 100644
--- a/examples/medlit/pipeline/relationships.py
+++ b/examples/medlit/pipeline/relationships.py
@@ -8,8 +8,11 @@
 from datetime import datetime, timezone
 from typing import TYPE_CHECKING, Any, Optional, Sequence
 
+import json
+from pathlib import Path
+
 from kgraph.document import BaseDocument
 from kgraph.domain import Evidence, Provenance
 from kgraph.entity import BaseEntity
 from kgraph.pipeline.interfaces import RelationshipExtractorInterface
 from kgraph.relationship import BaseRelationship
@@ -271,6 +274,45 @@ class MedLitRelationshipExtractor(RelationshipExtractorInterface):
     async def _extract_with_llm(
         self,
         document: BaseDocument,
         entities: Sequence[BaseEntity],
     ) -> list[BaseRelationship]:
         """Extract relationships using LLM."""
         if not self._llm:
             return []
+
+        # ---- trace setup (single file per paper) ----
+        trace_dir = Path("/tmp/kgraph-relationship-traces")
+        trace_dir.mkdir(parents=True, exist_ok=True)
+        trace_path = trace_dir / f"{document.document_id}.relationships.trace.json"
+        trace: dict[str, Any] = {
+            "document_id": document.document_id,
+            "source_uri": getattr(document, "source_uri", None),
+            "llm_model": getattr(self._llm, "model", None),
+            "created_at": datetime.now(timezone.utc).isoformat(),
+            "prompt": None,
+            "raw_llm_output": None,
+            "parsed_json": None,
+            "decisions": [],
+            "final_relationships": [],
+        }
 
         # Build entity context for LLM
         entity_by_name: dict[str, BaseEntity] = {e.name.lower(): e for e in entities}
         entity_list = "\n".join(f"- {e.name} ({e.get_entity_type()}): {e.entity_id}" for e in entities[:50])  # Limit to avoid huge prompts
 
@@ -288,6 +330,8 @@ class MedLitRelationshipExtractor(RelationshipExtractorInterface):
 
         prompt = f"""Extract medical relationships from the text below.
@@ -322,6 +366,8 @@ class MedLitRelationshipExtractor(RelationshipExtractorInterface):
 
         try:
-            response = await self._llm.generate_json(prompt)
+            trace["prompt"] = prompt
+
+            # Prefer raw+parsed if available
+            if hasattr(self._llm, "generate_json_with_raw"):
+                response, raw = await self._llm.generate_json_with_raw(prompt)
+                trace["raw_llm_output"] = raw
+            else:
+                response = await self._llm.generate_json(prompt)
+                trace["raw_llm_output"] = "<raw unavailable>"
+
+            trace["parsed_json"] = response
             relationships: list[BaseRelationship] = []
 
             if isinstance(response, list):
                 for item in response:
                     if isinstance(item, dict):
                         subject_name = item.get("subject", "").strip()
                         predicate = item.get("predicate", "").lower()
                         object_name = item.get("object", "").strip()
                         confidence = float(item.get("confidence", 0.5))
                         evidence = item.get("evidence", "")
+
+                        decision: dict[str, Any] = {
+                            "item": item,
+                            "subject_name": subject_name,
+                            "object_name": object_name,
+                            "predicate": predicate,
+                            "confidence": confidence,
+                            "matched_subject": False,
+                            "matched_object": False,
+                            "semantic_ok": None,
+                            "swap_applied": False,
+                            "accepted": False,
+                            "drop_reason": None,
+                            "resolved": {
+                                "subject_id": None,
+                                "subject_type": None,
+                                "object_id": None,
+                                "object_type": None,
+                            },
+                        }
 
                         # Find entities by name
                         subject_entity = entity_by_name.get(subject_name.lower())
                         object_entity = entity_by_name.get(object_name.lower())
+
+                        decision["matched_subject"] = subject_entity is not None
+                        decision["matched_object"] = object_entity is not None
 
                         if subject_entity and object_entity:
                             # Validate predicate semantics match evidence
-                            if not self._validate_predicate_semantics(predicate, evidence):
+                            semantic_ok = self._validate_predicate_semantics(predicate, evidence)
+                            decision["semantic_ok"] = semantic_ok
+                            if not semantic_ok:
                                 print(f"  Warning: Semantic mismatch - predicate '{predicate}' " f"does not match evidence: {evidence[:100]}...")
+                                decision["drop_reason"] = "semantic_mismatch"
+                                trace["decisions"].append(decision)
                                 continue  # Skip this relationship
 
                             # Check if we need to swap subject and object based on type constraints
                             if self._should_swap_subject_object(predicate, subject_entity, object_entity):
                                 print(
                                     f"  Swapping subject/object for predicate '{predicate}': "
                                     f"({subject_entity.name} [{subject_entity.get_entity_type()}] ↔ "
                                     f"{object_entity.name} [{object_entity.get_entity_type()}])"
                                 )
                                 # Swap the entities
                                 subject_entity, object_entity = object_entity, subject_entity
+                                decision["swap_applied"] = True
 
                             # Create structured provenance for LLM extraction
@@ -374,6 +467,8 @@ class MedLitRelationshipExtractor(RelationshipExtractorInterface):
                             rel = MedicalClaimRelationship(
                                 subject_id=subject_entity.entity_id,
                                 predicate=predicate,
                                 object_id=object_entity.entity_id,
                                 confidence=confidence,
                                 source_documents=(document.document_id,),
                                 evidence=evidence_obj,
                                 created_at=datetime.now(timezone.utc),
                                 last_updated=None,
                                 metadata={"extraction_method": "llm"},
                             )
                             relationships.append(rel)
+
+                            decision["accepted"] = True
+                            decision["resolved"] = {
+                                "subject_id": subject_entity.entity_id,
+                                "subject_type": subject_entity.get_entity_type(),
+                                "object_id": object_entity.entity_id,
+                                "object_type": object_entity.get_entity_type(),
+                            }
+                            trace["decisions"].append(decision)
+                        else:
+                            # Drop because at least one endpoint wasn't resolved
+                            if not decision["matched_subject"] and not decision["matched_object"]:
+                                decision["drop_reason"] = "subject_and_object_unmatched"
+                            elif not decision["matched_subject"]:
+                                decision["drop_reason"] = "subject_unmatched"
+                            else:
+                                decision["drop_reason"] = "object_unmatched"
+                            trace["decisions"].append(decision)
 
+            # Record final relationships in trace
+            trace["final_relationships"] = [
+                {
+                    "subject_id": r.subject_id,
+                    "predicate": r.predicate,
+                    "object_id": r.object_id,
+                    "confidence": getattr(r, "confidence", None),
+                }
+                for r in relationships
+            ]
+
+            # Write trace once per run
+            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
+            print(f"  Wrote relationship trace: {trace_path}")
+
             return relationships
 
         except Exception as e:
             print(f"Warning: LLM relationship extraction failed: {e}")
+            # Best-effort trace write if we have anything
+            try:
+                trace["error"] = repr(e)
+                trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
+                print(f"  Wrote relationship trace (with error): {trace_path}")
+            except Exception:
+                pass
             return []
```

---

# Run it (your existing script)

Now when you run your single-paper ingestion, you should see:

* “Wrote relationship trace: …”

And you’ll have a file like:

`/tmp/kgraph-relationship-traces/PMC12748354.relationships.trace.json`

Open that file and you’ll immediately see whether you’re dropping relationships due to:

* unmatched entity names
* semantic filter
* swap behavior
* or the model not actually returning what you think it returned

That’s the entire experiment.

If you paste the first trace you get (or just the `decisions` section + the `raw_llm_output` header), I’ll help interpret it — **without** adding new to-dos until we’ve identified the dominant failure mode.


---
## Historical Note: ChatGPT Patch Proposals (2026-02-04)

> **⚠️ CORRECTION:** The section below ("Results of running single-paper test") was
> **aspirational documentation** that described patches as if they had been applied.
> Verification on 2026-02-04 confirmed that the actual codebase does NOT have:
> - `generate_json_with_raw()` in `llm_client.py`
> - Trace file writing in `relationships.py`
>
> See **"Action Plan"** at the top of this file for the actual implementation tasks.

~~### ✅ Patch 1: Added generate_json_with_raw() to LLM Client~~ **NOT IMPLEMENTED**

~~### ✅ Patch 2: Added Relationship Trace Logging~~ **NOT IMPLEMENTED**

~~### ✅ Test Run Results~~ **NOT VERIFIED**

