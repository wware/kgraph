# Mention Inspection Diagnostic — Implementation Plan

## Goal

Add a lightweight diagnostic capability to debug cases where an unrelated paper appears in query results. Supports inspecting raw mentions and their source XML via MCP commands. Root cause may be: bad mention (Pass 1 hallucination), false-positive entity merge (Pass 2), or bundling artifact (Pass 3).

---

## Pipeline Context

The medlit pipeline is orchestrated by `run-ingest.sh`:

| Step | Script | Output |
|------|--------|--------|
| Pass 1a | `pass1a_vocab` | `pass1_vocab/` (vocab.json, synonym cache) |
| Pass 1b | `pass1_extract` | `pass1_bundles/` (paper_*.json per paper) |
| Pass 2 | `pass2_dedup` | `medlit_merged/` (entities.json, relationships.json, id_map.json) |
| Pass 3 | `pass3_build_bundle` | `medlit_bundle/` (mentions.jsonl, entities.jsonl, relationships.jsonl, evidence.jsonl, manifest.json) |

**Note:** `mentions.jsonl` is produced in Pass 3, not Pass 1. The bundle is loaded by kgserver via `BUNDLE_PATH`. MCP tools are in `kgserver/mcp_server/server.py`.

---

## Docker / Bundle Storage

**Architecture:** API and MCP are **separate containers**. Both need access to the bundle for `get_paper_source` and `get_mentions` (MCP reads `sources/` and `mentions.jsonl` directly).

**Current state:**
- **API** has `BUNDLE_PATH: /bundle` and `volumes: - ./medlit_bundle:/bundle:ro`
- **MCP** previously had neither: it used the baked-in `COPY medlit_bundle /bundle` from the image, which could be stale or absent

**Fix:** Add `BUNDLE_PATH` and the same volume to `mcpserver` in `docker-compose.yml` so both containers share the host bundle:

```yaml
mcpserver:
  environment:
    BUNDLE_PATH: /bundle
  volumes:
    - ./medlit_bundle:/bundle:ro
```

**ZIP bundles:** When `BUNDLE_PATH` points to a `.zip` file (e.g. `mount: ./bundle.zip:/bundle/bundle.zip`), the API extracts to `tempfile.TemporaryDirectory()` (ephemeral) for loading into Postgres. The MCP tools must read from the ZIP directly. Use `Path(bundle_path).suffix == ".zip"` for detection — not `zipfile.is_zipfile()`, which reads the file and raises if the path is a directory (the common case).

---

## Task 1: Add `sources/` Directory to medlit_bundle

**Location:** `medlit_bundle/sources/`

**Content:** One JATS-XML file per ingested paper, e.g. `PMC12345.xml`, `PMC67890.xml`.

**Population:** During ingestion, copy XML from `examples/medlit/pmc_xmls/` into `medlit_bundle/sources/`. The list of papers comes from the `PAPER` variable in `run-ingest.sh` (or equivalent when run via ingest_worker).

**Implementation choice:** Pass 3 is the cleaner home (it already knows which papers it processed). Add optional `--pmc-xmls-dir` to `pass3_build_bundle.py`; if provided, copy XML for papers in the bundles into `output_dir/sources/`. If threading `--pmc-xmls-dir` through ingest_worker and other callers is awkward, fall back to a shell step in `run-ingest.sh` after Pass 3: create `medlit_bundle/sources/` and copy each `$PMC.xml` from `pmc_xmls/` for every PMC in `$PAPER`. Settle this early during implementation.
- Store raw XML (no stripping for now); document as future cleanup if stripping is added.

---

## Task 2: MCP Tool `get_paper_source`

**File:** `kgserver/mcp_server/server.py`

**Signature:**
```python
@mcp_server.tool()
def get_paper_source(paper_id: str, max_chars: int | None = None) -> str:
```

**Behavior:**
- Resolve bundle path from `os.getenv("BUNDLE_PATH")`. If unset, return error.
- Support both directory and ZIP: use `Path(bundle_path).suffix == ".zip"` for detection; if ZIP, `zipfile.ZipFile(path).open(f"sources/{paper_id}.xml")`; otherwise read from `{bundle_path}/sources/{paper_id}.xml`.
- Normalize `paper_id`: strip any `.xml` suffix, then always append `.xml`, so both `"PMC12345"` and `"PMC12345.xml"` resolve to `sources/PMC12345.xml`.
- If `max_chars` is set, truncate the string to that length before returning.
- Add `# TODO` near `max_chars` implementation: consider lxml/ElementTree snippet to extract just `<abstract>` and `<body>` instead of blind truncation.
- If file not found, raise or return a clear error message: `Paper {paper_id} not found in sources/`.

---

## Task 3: MCP Tool `get_mentions`

**File:** `kgserver/mcp_server/server.py`

**Signature:**
```python
@mcp_server.tool()
def get_mentions(paper_id: str | None = None) -> list[dict]:
```

**Behavior:**
- Resolve bundle path from `os.getenv("BUNDLE_PATH")`.
- Support both directory and ZIP: use `Path(bundle_path).suffix == ".zip"` for detection; if ZIP, read via `zipfile.ZipFile(path).open("mentions.jsonl")`; otherwise from `{bundle_path}/mentions.jsonl`.
- Parse each line as JSON (MentionRow schema).
- **Performance note:** When filtering by `paper_id`, the implementation reads and parses the entire file then filters in memory. Fine for a diagnostic tool; add a code comment so nobody assumes it's efficient for large corpora.
- Schema field for document: `document_id` (not `paper_id`). For medlit, values are like `PMC12345`.
- If `paper_id` is provided, filter to rows where `document_id == paper_id` (or equivalent match).
- If `paper_id` is `None`, return all mentions.
- Return a list of dicts (e.g. `model_dump()` or `dict(row)`), not raw JSONL strings.
- If `mentions.jsonl` is missing, return empty list or clear error.

---

## Task 4: LLM Verification Workflow (Documentation Only)

Document the verification pattern for use in chat or scripts:

1. Call `get_paper_source("PMC12345", max_chars=8000)`.
2. Call `get_mentions("PMC12345")`.
3. Prompt the LLM: *"Here is the source text of paper PMC12345. Here is the list of entity mentions extracted from it. For each mention, flag it as OK or BOGUS. Flag BOGUS if: (a) the entity string does not appear in the source text, (b) the entity type appears wrong given context, or (c) the span looks garbled or truncated. Return a JSON list with fields: [identifier], verdict, reason."*

**Identifier field:** MentionRow has no `mention_id`; it has `entity_id`, `document_id`, `text_span`, `start_offset`, etc. When writing the doc, verify the schema and use a stable identifier for the prompt — e.g. a composite like `entity_id:document_id:start_offset` or a synthetic index added when serializing. Avoid `mention_id` if it doesn't exist in the schema.

**Note:** `max_chars=8000` will cut off mid-sentence for most full JATS papers. Callers may want to extract just `<abstract>` and `<body>` lead sections rather than relying on blind truncation — a refinement, not a blocker.

No code changes required for this step; it is a usage pattern.

---

## Task 5: Keep Merge-Bug Hypothesis in Mind

If the bogus paper link appeared *after* Pass 2 deduplication, the bug may be a false-positive entity merge. The mention inspection workflow rules out (or confirms) the extraction layer. If mentions look clean, the next step is to inspect `medlit_merged/entities.json` and `id_map.json` for over-eager merges.

**Merge-bug signature in `id_map.json`:** Multiple source IDs mapping to the same canonical ID indicates a merge. If promoting this to `DIAGNOSTICS.md`, add that concrete lookup pattern.

A follow-on `get_entity_provenance(entity_id)` tool is out of scope for this pass.

---

## Implementation Order

1. **Docker-compose fix** — Add `BUNDLE_PATH` and `volumes: - ./medlit_bundle:/bundle:ro` to `mcpserver` (prerequisite for Tasks 2 and 3 to be testable).
2. **Task 1** — Add `sources/` population (in `pass3_build_bundle` or `run-ingest.sh`).
3. **Task 2** — Implement `get_paper_source` MCP tool.
4. **Task 3** — Implement `get_mentions` MCP tool.
5. **Task 4** — Add a short doc section (e.g. in `examples/medlit/INGESTION.md` or a new `examples/medlit/DIAGNOSTICS.md`) describing the LLM verification workflow.
