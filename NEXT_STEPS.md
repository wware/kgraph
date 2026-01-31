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

