# Mention Inspection Diagnostics

When an unrelated paper appears in query results, use these MCP tools to debug. Root cause may be: bad mention (Pass 1 hallucination), false-positive entity merge (Pass 2), or bundling artifact (Pass 3).

## MCP Tools

- **`get_paper_source(paper_id, max_chars=None)`** — Raw JATS-XML of a paper from `bundle/sources/`.
- **`get_mentions(paper_id=None)`** — Entity mentions from `mentions.jsonl`, optionally filtered by `document_id`.

Both require `BUNDLE_PATH` and read from the bundle directory or ZIP. The bundle must include `sources/` (populated by Pass 3 with `--pmc-xmls-dir`).

## LLM Verification Workflow

1. Call `get_paper_source("PMC12345", max_chars=8000)`.
2. Call `get_mentions("PMC12345")`.
3. Prompt the LLM:

   *"Here is the source text of paper PMC12345. Here is the list of entity mentions extracted from it. For each mention, flag it as OK or BOGUS. Flag BOGUS if: (a) the entity string does not appear in the source text, (b) the entity type appears wrong given context, or (c) the span looks garbled or truncated. Return a JSON list with fields: identifier, verdict, reason."*

**Identifier:** MentionRow has no `mention_id`. Use a stable composite such as `entity_id:document_id:start_offset`, or add a synthetic index when serializing (e.g. `mention_0`, `mention_1`) so the LLM can reference each item.

**Truncation:** `max_chars=8000` will cut off mid-sentence for most full JATS papers. For better coverage, extract just `<abstract>` and `<body>` lead sections (e.g. via lxml/ElementTree) instead of blind truncation.

## If Mentions Look Clean: Merge-Bug Hypothesis

If the bogus paper link appeared *after* Pass 2 deduplication, the bug may be a false-positive entity merge. Inspect `medlit_merged/entities.json` and `medlit_merged/id_map.json`.

**Merge-bug signature in `id_map.json`:** Multiple source IDs mapping to the same canonical ID indicates a merge. Look for entries where several `paper_id` keys resolve to one canonical entity.
