# Ingestion Pipeline — Remaining Work

Items left from PLAN2.md after the initial implementation. See PLAN2.md and ingestion_redesign.md for full details.

---

## Phase E: Chunking and per-chunk extraction (higher risk)

### E1: Add chunking module

- Create `kgraph/pipeline/chunker.py` or extend `examples/medlit/pipeline/pmc_chunker.py`
- Input: document text, section boundaries
- Output: list of chunks with overlap (e.g. 512 tokens, 64 overlap)
- Each chunk: `{chunk_id, text, section, start_offset, end_offset}`

### E2: Per-chunk entity extraction

- Parse document → chunks
- For each chunk: call LLM with entity extraction template
- Merge entities by (name, type) across chunks within document
- Output: per-document bundle with deduplicated entities

### E3: Per-chunk relationship extraction

- After entity resolution within document, for each chunk: call LLM with relationship template
- Merge relationships by (sub, pred, obj) across chunks
- Accumulate provenance (chunk_id, section, sentence) per occurrence

**Sequencing:** Study design (D2) stays per-paper; do not move it to per-chunk when E lands.

---

## Phase F: Abstract pre-pass and citation integration (deferred)

- Phase 0 abstract pre-pass for claim hierarchy
- JATS reference list parsing
- Citation marker parsing in prompts
- Claim hierarchy data model

**Status:** Documented as future work. Not in initial execution scope.

---

## Near-term / Demo Readiness

### Robustness

- **fetch_vocab JSON retry**: LLM occasionally returns malformed JSON (unbalanced braces). Mitigations to consider (in rough order of effort):
  - Use a JSON repair library (e.g. `json-repair`) as a fallback before giving up on a response
  - Add explicit closing reminder to prompts ("ensure all arrays and objects are properly closed")
  - Use response prefilling (`[` or `{`) to anchor the model and reduce preamble drift
  - Flatten deeply nested output schemas — nesting gives the model more opportunities to mis-close near the token limit
  - Split large extraction calls into two smaller calls to reduce per-response size
- **Entity name uniqueness constraint**: `Entity` table has `UniqueConstraint("name", "entity_type")` which can silently drop entities during bundle load if two distinct entities share the same name and type (e.g. two authors with the same name). Audit or relax this constraint.
- **Deterministic truncation in full-graph mode**: `extract_full_graph` has no ordering on `get_entities` — truncated views are non-deterministic. Sort by `usage_count DESC` so the most-used entities are always included.

### Graph Viz UX

- **Missing color legend entries for `author` and `institution`**: New entity types from ORCID/ROR ingestion. Verify `domain_spec.py` has entries with distinct colors; otherwise they render as grey.
- **No persistent loading indicator**: The "Load Graph" button gives no feedback while D3 renders a large graph, making it look frozen.

### ORCID / Author Identity

- **ORCID lookup by researcher name**: Authors without ORCID tags in the XML get provisional IDs. Consider querying the ORCID public API (`https://pub.orcid.org/v3.0/search?q=family-name:X+given-names:Y`) during authority lookup to assign canonical ORCID IDs to untagged authors. Needs deduplication care (name collisions).
- **ORCID coverage is XML-tag-dependent**: Only authors with `<contrib-id contrib-id-type="orcid">` tags get canonical IDs. Set demo expectations accordingly.
- **Bare ROR URL as entity ID**: Verify the pipeline never emits a bare `https://ror.org/...` as a primary entity ID (instead of the `ROR:xxxxxxxx` prefix form). If it does, check that `formatEntityIdLink` in `graph-viz.js` handles it gracefully rather than rendering an unlinked raw URL.

### Operational

- **Document BUNDLE_FORCE_RELOAD workflow**: `BUNDLE_FORCE_RELOAD` is not set in `docker-compose.yml`. If the bundle ID is unchanged across re-ingestions, a container restart won't reload data. Add a note to the ops docs.
- **patch_paper_titles.py**: Audit whether this script is still needed or has been superseded by the pipeline.
- **Full redeploy on the droplet** (move this to a proper ops doc when one exists):
  ```
  docker compose --profile api down -v && docker image prune -a && git pull && docker compose --profile api build && docker compose --profile api up -d && docker compose --profile api logs -f
  ```
