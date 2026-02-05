# Force-Directed Graph Visualization

Interactive graph visualization for kgserver using D3.js force simulation.

## Overview

This feature provides an interactive graph visualization accessible via a REST endpoint and static HTML page. The design separates data retrieval (API) from rendering (client-side JS) for flexibility and extensibility.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Browser                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  /graph-viz (Static HTML + D3.js)                        │  │
│  │  - Force simulation rendering                            │  │
│  │  - Click handlers for node/edge details                  │  │
│  │  - Controls: center entity, hop depth, layout options    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GET /api/v1/graph/subgraph                              │  │
│  │  ?center_id=...&hops=2&include_all=false                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      kgserver (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  query/routers/graph_api.py                              │  │
│  │  - SubgraphRequest/Response models                       │  │
│  │  - BFS traversal logic                                   │  │
│  │  - D3-compatible JSON output                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  query/graph_traversal.py                                │  │
│  │  - extract_subgraph(center_id, hops, storage)            │  │
│  │  - extract_full_graph(storage, max_nodes)                │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Models

### GraphNode

D3-compatible node representation:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | entity_id |
| `label` | string | name or entity_id fallback |
| `entity_type` | string | Type for styling |
| `properties` | dict | Full entity data for detail panel |

### GraphEdge

D3-compatible edge representation:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | subject_id |
| `target` | string | object_id |
| `label` | string | predicate (human-readable) |
| `predicate` | string | predicate (raw) |
| `properties` | dict | Full relationship data for detail panel |

### SubgraphResponse

| Field | Type | Description |
|-------|------|-------------|
| `nodes` | list[GraphNode] | Nodes in subgraph |
| `edges` | list[GraphEdge] | Edges in subgraph |
| `center_id` | string | null | Focal entity (if subgraph mode) |
| `hops` | int | Depth traversed |
| `truncated` | bool | True if max_nodes limit was hit |
| `total_entities` | int | Total in full graph |
| `total_relationships` | int | Total relationships |

## API Endpoints

### GET /api/v1/graph/subgraph

Retrieve a subgraph for visualization.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `center_id` | string | null | Entity to center on |
| `hops` | int | 2 | BFS depth (1-5) |
| `max_nodes` | int | 500 | Safety limit |
| `include_all` | bool | false | If true, return entire graph |

**Behavior:**
- If `include_all=true`: returns entire graph (up to max_nodes)
- Otherwise: BFS from center_id for 'hops' levels

### GET /api/v1/graph/node/{entity_id}

Get full details for a single node (for click-to-expand).

### GET /api/v1/graph/edge

Get full details for a single edge.

**Parameters:** `subject_id`, `predicate`, `object_id`

## Graph Traversal Algorithm

BFS from center entity:

```
visited = {center_id}
frontier = {center_id}
for hop in range(hops):
    next_frontier = set()
    for entity_id in frontier:
        # Get relationships where entity is subject OR object
        rels = storage.find_relationships(subject_id=entity_id) 
             + storage.find_relationships(object_id=entity_id)
        for rel in rels:
            neighbor = rel.object_id if rel.subject_id == entity_id else rel.subject_id
            if neighbor not in visited:
                next_frontier.add(neighbor)
                visited.add(neighbor)
    frontier = next_frontier
    if len(visited) >= max_nodes:
        break  # truncate
```

## Visualization Features

### D3 Force Configuration

```javascript
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(edges).id(d => d.id).distance(150))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width/2, height/2))
    .force("collision", d3.forceCollide().radius(30));
```

### Interactive Features

- **Nodes**: Circles colored by `entity_type`, labeled with name
- **Edges**: Lines with arrowheads showing direction, labeled with predicate
- **Controls**:
  - Text input for `center_id` (with autocomplete)
  - Slider for `hops` (1-5, default 2)
  - Checkbox for "Show entire graph"
  - Reload button
- **Click handlers**:
  - Click node → side panel shows entity details
  - Click edge → side panel shows relationship details
  - Double-click node → re-center graph on that node
- **Navigation**: Zoom and pan support via d3-zoom

## Safety Limits

| Parameter | Default | Max | Purpose |
|-----------|---------|-----|---------|
| `hops` | 2 | 5 | Prevent exponential blowup |
| `max_nodes` | 500 | 2000 | Browser performance |
| `max_edges` | 5000 | 10000 | Rendering limits |

## File Structure

```
kgserver/query/
├── routers/
│   ├── rest_api.py          # existing
│   ├── graphiql_custom.py   # existing
│   └── graph_api.py         # graph subgraph endpoint
├── models/
│   └── graph.py             # GraphNode, GraphEdge, SubgraphResponse
├── graph_traversal.py       # BFS logic
├── static/
│   ├── graph-viz.html       # main visualization page
│   ├── graph-viz.js         # D3 rendering logic
│   └── graph-viz.css        # styling
└── server.py                # mount new router + static
```

## Future Enhancements

- **Evidence-weighted forces**: Adjust link distance/strength based on relationship confidence or evidence count
- **Filtering**: Add `entity_types` and `predicates` params to filter visible nodes/edges
- **Export**: Button to export current view as SVG or PNG
- **Cluster detection**: Highlight connected components or communities
- **Time-based filtering**: If relationships have timestamps, add temporal filtering
- **Search integration**: Autocomplete for `center_id` using entity search
