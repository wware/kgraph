# What are Next Steps for this thing?

## Near term

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
                   "python", "-m", "mcp_server.stdio_server"
               ],
               "env": {
                   "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/kgserver"
               }
           }
       }
   }
   ```

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

2. Create a debug ingestion script for single-paper analysis:
   ```bash
   python -m examples.medlit.scripts.ingest_single_debug --paper PMC12748354
   ```

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

## A. One “RelationshipTrace” object, created in one place

### Where it should live

Create a tiny module in kgraph (so both examples + kgserver can reuse it):

* `kgraph/pipeline/trace.py`

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

### Primary seam: `MedLitRelationshipExtractor._extract_relationships_with_llm`

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

* call detailed validation
* `trace.add_stage("validated", ...)`
* if suggested repair includes `SWAP` and swap produces a valid signature, apply it and record `transformed`

This is where your “type checking on predicates” and “automatic swapping” becomes transparent.

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

## F. Storage: don’t lose debugging context when loading JSONL

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
python -m examples.medlit.scripts.ingest_single_debug \
  --paper PMC12748354 \
  --replay output/debug/traces/PMC12748354.relationships.trace.json \
  --stage validate
```

Implementation approach:

* `ingest_single_debug` loads the trace
* re-runs parse/validate/transform steps on recorded `raw_response` or recorded `parsed`
* never hits the model

This lets you iterate on:

* parsing rules
* predicate signature table
* swap logic
* validation policies

…without burning model calls.

---

## H. Exact “touch points” in your listed modules

### `examples/medlit/pipeline/relationships.py`

* Add `trace` + `replay` capability to `_extract_relationships_with_llm`
* Parse into `MedLitExtractedRelationship[]` first
* Record `prompt`, `raw_response`, `parsed`

### `kgraph/ingest.py` (`IngestionOrchestrator`)

* Owns: `validated` + `transformed` + `final` stages
* Calls `domain.validate_relationship_detailed`
* Applies repairs (swap, normalize predicate)
* Records everything

### `kgraph/domain.py` (+ domain extensions)

* Add `validate_relationship_detailed`
* Centralize predicate signatures and swap policy
* Make errors machine-readable (codes + context)

### `kgraph/builders.py`

* If it currently validates, either:

  * leave it, but make it accept pre-validated relations
  * or have it optionally accept `validation_result` to avoid double work
* Ensure builder doesn’t silently swap; swapping should already be done

### `kgserver/storage/backends/*.py`

* Improve error context on JSONL load
* Optionally preserve provenance fields for later inspection

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

# okay

This is super actionable — you already have *two* different “swap/type mismatch” mechanisms, and right now they fight each other:

* `DomainSchema.validate_relationship()` detects “looks swapped” but **returns False** (and only logs).
* `MedLitRelationshipExtractor._extract_with_llm()` does a pre-swap check (`_should_swap_subject_object`) and **actually swaps**.

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

