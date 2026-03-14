# PLAN7: Unified BFS Subgraph with JSON API and MCP Tool

**Status:** Not started.

**Goal:** Unify the BFS subgraph implementation with orthogonal topology and presentation filters, add an LLM-friendly JSON POST endpoint, and expose a `bfs_subgraph` MCP tool. Support both "pruned" (topology-filtered) and "stubbed" (presentation-filtered) behavior in a single code path.

---

## Terminology

| Term | Meaning |
|------|---------|
| **Topology filter** | Applied during BFS. Edges with confidence below `min_confidence` are skipped; their neighbors are not traversed. Produces a smaller subgraph ("pruned"). Predicate filtering is exclusively a presentation concern via `edge_filter`. |
| **Presentation filter** | Applied at serialization. Nodes/edges not matching (node_filter.entity_types, edge_filter.predicates) receive minimal stub serialization; matching items receive full metadata. Topology is unchanged ("stubbed"). |
| **Full node** | All available entity metadata (id, entity_type, name, status, confidence, usage_count, source, canonical_url, synonyms, properties). |
| **Stub node** | Only `{id, entity_type}`. |
| **Full edge** | All relationship metadata including provenance (subject, predicate, object, confidence, source_documents, properties). |
| **Stub edge** | Only `{subject, predicate, object}`. |

---

## Design: Orthogonal Filters

One implementation supports four combinations:

| Topology filter | Presentation filter | Result |
|-----------------|---------------------|--------|
| None | None | Full BFS, full serialization (current graph_api) |
| Set | None | Pruned BFS, full serialization (current subgraph_api) |
| None | Set | Full BFS, stubbed serialization (bfsql.md) |
| Set | Set | Pruned BFS, stubbed serialization |

---

## JSON Query Format (Input)

```json
{
  "seeds": ["<entity_id>", ...],
  "max_hops": <int>,
  "max_nodes": <int>,
  "topology_filter": {
    "min_confidence": <float>
  },
  "node_filter": {
    "entity_types": ["<type>", ...]
  },
  "edge_filter": {
    "predicates": ["<predicate>", ...]
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `seeds` | Yes | Array of canonical entity IDs. BFS starts from all; result is union of neighborhoods. |
| `max_hops` | Yes | Max graph distance from any seed. 1–3 typical; cap at 5. |
| `max_nodes` | No | Max nodes to include (default 500, max 2000). When hit, `truncated` is true. |
| `topology_filter` | No | Prunes during BFS. `min_confidence`: skip edges below this. Omit = no pruning. (Not exposed to LLM; see note below.) |
| `node_filter` | No | Presentation. Nodes with entity_type in `entity_types` get full serialization; others get stub. Omit = full for all nodes. |
| `edge_filter` | No | Presentation. Edges with predicate in `predicates` get full serialization; others get stub. Omit = full for all edges. |

---

## JSON Response Format (Output)

```json
{
  "seeds": ["<entity_id>", ...],
  "max_hops": 2,
  "node_count": <int>,
  "edge_count": <int>,
  "truncated": <bool>,
  "nodes": [
    { "id": "<entity_id>", "entity_type": "<type>", ... }
  ],
  "edges": [
    { "subject": "<id>", "predicate": "<pred>", "object": "<id>", ... }
  ]
}
```

- **Full node:** `id`, `entity_type`, `name`, `status`, `confidence`, `usage_count`, `source`, `canonical_url`, `synonyms`, `properties`.
- **Stub node:** `id`, `entity_type` only.
- **Full edge:** `subject`, `predicate`, `object`, `confidence`, `source_documents`, `properties` (including evidence/provenance keys if present).
- **Stub edge:** `subject`, `predicate`, `object` only.
- `truncated`: true when `max_nodes` limit was hit.

---

## LLM Prompt Documentation

Include the following in system prompts or tool descriptions for `bfs_subgraph` / `bfs_query`:

### Using `bfs_subgraph`

To explore the knowledge graph, call `bfs_subgraph` with a JSON body. The query performs a breadth-first search from one or more seed nodes and returns the resulting subgraph.

Nodes and edges are either **full** (all metadata and provenance) or **stub** (identity only), depending on `node_filter` and `edge_filter`. These filters affect only what data is returned, not which nodes and edges are included.

**If you do not yet have a canonical entity ID, call `search_entities` first.**

#### Query structure

```json
{
  "seeds": ["<id>", ...],
  "max_hops": <int>,
  "node_filter": {
    "entity_types": ["Publication", "Disease", "Drug", ...]
  },
  "edge_filter": {
    "predicates": ["AUTHORED", "TREATS", "INHIBITS", ...]
  }
}
```

- Stub nodes: `{id, entity_type}` only.
- Stub edges: `{subject, predicate, object}` only.
- Omitting a filter returns full data for all nodes or edges.

#### Example 1: Author's publications with provenance

```json
{
  "seeds": ["PERSON:12345"],
  "max_hops": 1,
  "node_filter": { "entity_types": ["Publication"] },
  "edge_filter": { "predicates": ["AUTHORED"] }
}
```

Result: Full data on Publication nodes, full provenance on AUTHORED edges. Other nodes/edges at hop 1 appear as stubs.

#### Example 2: Disease neighborhood, focus on drugs

```json
{
  "seeds": ["C0085084"],
  "max_hops": 2,
  "node_filter": { "entity_types": ["Drug"] }
}
```

Result: Full data on Drug nodes. Other types and all edges appear as stubs.

#### Example 3: Shared connections between two entities

```json
{
  "seeds": ["PERSON:12345", "PERSON:67890"],
  "max_hops": 2,
  "node_filter": { "entity_types": ["Publication", "Disease"] },
  "edge_filter": { "predicates": ["AUTHORED", "DISCUSSES"] }
}
```

Result: Union of both neighborhoods. Full data on Publications and Diseases; full provenance on AUTHORED and DISCUSSES. Everything else stubbed.

#### Depth guidance

- **1 hop:** Direct relationships (author's publications, drug indications).
- **2 hops:** Indirect connections (diseases via genes a drug targets, co-authors).
- **3+ hops:** Can be large; use targeted filters.

---

### Note: topology_filter intentionally omitted from LLM prompt

The `topology_filter` (min_confidence-based pruning) exists in the implementation and Pydantic model for programmatic callers who want confidence-based pruning. It is **intentionally omitted** from the docstring, MCP tool description, and all prompt documentation above. The LLM sees a clean two-filter model: `node_filter` and `edge_filter`, both presentation-only. **Do not add topology_filter to the LLM-facing docs** — a future reader might assume it was forgotten; it was a deliberate simplification.

---

## Implementation Plan

### Phase 1: Core BFS with Orthogonal Filters

#### Step 1.1: Extend `_extract_subgraph_multi_seed` in graph_traversal.py

**File:** `kgserver/query/graph_traversal.py`

**1.1.1** Add optional `node_filter` and `edge_filter` parameters. Keep existing `min_confidence` (topology filter). Do not add predicate-based topology filtering; predicates are presentation-only via `edge_filter`.

**1.1.2** Add a new function `extract_subgraph_bfs` that:
- Accepts: `storage`, `seed_ids`, `hops`, `max_nodes`, `min_confidence` (topology), `node_filter` (optional dict with `entity_types`), `edge_filter` (optional dict with `predicates`).
- Returns: `(nodes: list[dict], edges: list[dict], truncated: bool)` where each node/edge is already serialized as full or stub based on filters.
- Reuses the BFS loop from `_extract_subgraph_multi_seed`; topology filter applies only min_confidence (skip edges below threshold).
- After BFS, for each entity: if `node_filter` is None or `entity.entity_type` in `node_filter.get("entity_types", [])`, serialize full; else stub.
- For each relationship: if `edge_filter` is None or `rel.predicate` in `edge_filter.get("predicates", [])`, serialize full; else stub.
- Use existing nomenclature from the codebase (e.g. `id` or `entity_id` for nodes per GraphNode/storage; `subject`/`object` or `subject_id`/`object_id` for edges per Relationship).

**1.1.3** Add Pydantic models for the JSON request/response if not already present. Define `BfsSubgraphRequest` and `BfsSubgraphResponse` in a suitable module (e.g. `graph_traversal.py` or a new `bfs_models.py`).

**Verification:** Unit test that with no filters, output matches current full serialization; with node_filter, non-matching nodes are stubs.

---

#### Step 1.2: Serialization helpers

**File:** `kgserver/query/graph_traversal.py` (or `kgserver/query/bfs_serialize.py`)

**1.2.1** Add `_entity_to_full_node(entity) -> dict` returning full node dict (id, entity_type, name, status, confidence, usage_count, source, canonical_url, synonyms, properties).

**1.2.2** Add `_entity_to_stub_node(entity) -> dict` returning minimal node dict (field names per existing codebase convention).

**1.2.3** Add `_relationship_to_full_edge(rel) -> dict` returning full edge with subject, predicate, object, confidence, source_documents, properties. Sanitize source_documents (exclude placeholders like `paper_id`, `PMC_PLACEHOLDER`).

**1.2.4** Add `_relationship_to_stub_edge(rel) -> dict` returning minimal edge dict (field names per existing codebase convention).

---

### Phase 2: REST API

#### Step 2.1: POST endpoint with JSON body

**File:** `kgserver/query/routers/subgraph_api.py`

**2.1.1** Add Pydantic model `BfsSubgraphRequest`:
```python
class BfsSubgraphRequest(BaseModel):
    seeds: list[str]
    max_hops: int = Field(ge=1, le=5)
    max_nodes: int = Field(default=500, ge=1, le=2000)
    topology_filter: Optional[dict] = None  # min_confidence only; not in LLM prompt
    node_filter: Optional[dict] = None       # entity_types
    edge_filter: Optional[dict] = None       # predicates
```

**2.1.2** Add `BfsSubgraphResponse` model matching the JSON response format (seeds, max_hops, node_count, edge_count, truncated, nodes, edges).

**2.1.3** Add POST route `@router.post("", response_model=BfsSubgraphResponse)` that:
- Accepts `BfsSubgraphRequest` as JSON body.
- Validates all seeds exist. If any seed ID is not in the graph, return 400 with a clear error identifying the unknown ID(s). Do not proceed with a partial result.
- Calls `extract_subgraph_bfs` with topology_filter (min_confidence only) and presentation filters (node_filter, edge_filter).
- Returns `BfsSubgraphResponse`.

**2.1.4** Keep existing GET endpoint unchanged for backward compatibility. POST is the only way to submit a JSON body.

**Verification:** POST with valid JSON body returns BfsSubgraphResponse; GET with query params unchanged.

---

### Phase 3: MCP Tool `bfs_subgraph`

#### Step 3.1: Add tool to MCP server

**File:** `kgserver/mcp_server/server.py`

**3.1.1** Add `@mcp_server.tool()` function `bfs_subgraph` with signature:
```python
def bfs_subgraph(
    seeds: list[str],
    max_hops: int = 2,
    topology_filter: Optional[dict] = None,  # accepted but not in docstring
    node_filter: Optional[dict] = None,
    edge_filter: Optional[dict] = None,
) -> dict:
```

**3.1.2** Implementation:
- Use `_get_storage()` context manager.
- Validate all seeds exist; if any missing, raise/return error with unknown ID(s). Do not proceed with partial result.
- Import and call `extract_subgraph_bfs` from `query.graph_traversal` (or wherever it lives).
- Convert topology_filter: `min_confidence` only (no predicates).
- Return the dict representation of `BfsSubgraphResponse` (nodes, edges, seeds, max_hops, node_count, edge_count, truncated).

**3.1.3** Docstring: Include the LLM prompt documentation from "Using `bfs_subgraph`" above (abbreviated). Mention that `search_entities` should be called first if entity ID is unknown. **Do not mention topology_filter** — it is intentionally omitted from LLM-facing documentation.

---

#### Step 3.2: Register MCP tool descriptor

**Location:** MCP tool descriptors are typically auto-generated from the FastMCP server. Ensure the tool is exported. If there is a manual descriptor file (e.g. in `mcps/user-knowledge-graph/tools/`), add `bfs_subgraph.json` with:
- `name`: "bfs_subgraph"
- `description`: Summary + pointer to full docs
- `arguments`: seeds (required), max_hops, node_filter, edge_filter (topology_filter accepted by implementation but not documented for LLM)

**Verification:** MCP server lists `bfs_subgraph`; calling it returns the expected JSON structure.

---

### Phase 4: Tests

#### Step 4.1: Unit tests for BFS serialization

**File:** `kgserver/tests/test_bfs_subgraph.py` (new)

**4.1.1** Test `extract_subgraph_bfs` with no filters: all nodes and edges full.
**4.1.2** Test with `node_filter.entity_types`: matching nodes full, others stub.
**4.1.3** Test with `edge_filter.predicates`: matching edges full, others stub.
**4.1.4** Test with `topology_filter.min_confidence`: edges below threshold skipped; subgraph is smaller.
**4.1.5** Test multi-seed: union of neighborhoods.
**4.1.6** Test truncation when max_nodes hit.

---

#### Step 4.2: API tests

**File:** `kgserver/tests/test_subgraph_api.py` or extend existing

**4.2.1** POST with valid JSON body returns 200 and BfsSubgraphResponse shape.
**4.2.2** POST with unknown seed ID returns 400 with error identifying the unknown ID(s).
**4.2.3** GET without JSON (query params) still returns legacy format (entities, relationships, query).

---

#### Step 4.3: MCP tool test

**File:** `kgserver/tests/test_bfs_subgraph_mcp.py` (new) or extend `test_find_entities_within_hops.py`

**4.3.1** Call `bfs_subgraph` with mock storage; verify response structure (nodes, edges, seeds, max_hops, node_count, edge_count, truncated).
**4.3.2** Verify stub vs full based on node_filter and edge_filter.

---

## File Summary

| File | Changes |
|------|---------|
| `kgserver/query/graph_traversal.py` | Add `extract_subgraph_bfs`, serialization helpers, BfsSubgraphRequest/Response models |
| `kgserver/query/subgraph.py` | Optional: add wrapper for `extract_subgraph_bfs` if needed by REST |
| `kgserver/query/routers/subgraph_api.py` | Add POST endpoint, request/response models |
| `kgserver/mcp_server/server.py` | Add `bfs_subgraph` tool |
| `kgserver/tests/test_bfs_subgraph.py` | New: unit tests for BFS + serialization |
| `kgserver/tests/test_subgraph_api.py` | Extend: POST tests, seed validation 400 |
| `kgserver/tests/test_bfs_subgraph_mcp.py` | New: MCP tool tests |
| `mcps/user-knowledge-graph/tools/bfs_subgraph.json` | New: MCP tool descriptor (if manually maintained) |

---

## Constants and Limits

- `MAX_HOPS = 5` (existing)
- `MAX_NODES_LIMIT = 2000` (existing)
- `DEFAULT_MAX_NODES = 500` (existing)
- `max_hops` in request: clamp to 1–5
- `seeds`: require at least one; if any seed ID does not exist in the graph, return 400 with clear error identifying the unknown ID(s). Do not proceed with a partial result.

---

## Reference: bfsql.md

The design in this plan is derived from `bfsql.md`. That file remains the canonical design doc for the BFS query language. This plan adds topology filters (pruning) and unifies with the existing implementation.
