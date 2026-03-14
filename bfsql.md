# KGServer BFS Query Language

## Overview

The BFS query ([breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search)) provides a single, general-purpose mechanism for retrieving subgraphs from the
knowledge graph. It is designed to be constructed easily by an LLM while minimizing unnecessary
context window consumption in the response.

The design separates two orthogonal concerns:

- **Topology**: BFS from one or more seed nodes up to a specified hop depth defines which nodes
  and edges are included in the subgraph. Filtering has no effect on this.
- **Presentation**: Node and edge filters determine which items in that subgraph receive full
  metadata vs. a minimal stub. This is purely a serialization decision applied after the
  subgraph is computed.

Stub items carry only enough information to understand the graph's topology without consuming
context on irrelevant details. The LLM sees the full shape of the neighborhood while receiving
rich data only where it matters.

---

## Query Format

```json
{
  "seeds": ["<entity_id>", ...],
  "max_hops": <int>,
  "node_filter": {
    "entity_types": ["<type>", ...]
  },
  "edge_filter": {
    "predicates": ["<predicate>", ...]
  }
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `seeds` | Yes | Array of one or more canonical entity IDs to use as BFS starting points. All seeds are expanded simultaneously; the result is the union of their neighborhoods. |
| `max_hops` | Yes | Maximum graph distance from any seed node. Values of 1–3 are typical; larger values may return very large subgraphs. |
| `node_filter` | No | Controls which nodes receive full metadata. Nodes not matching the filter appear as stubs. Omit to receive full data on all nodes. |
| `node_filter.entity_types` | No | List of entity type names. A node matches if its type is in this list. |
| `edge_filter` | No | Controls which edges receive full metadata including provenance. Edges not matching appear as stubs. Omit to receive full data on all edges. |
| `edge_filter.predicates` | No | List of predicate names. An edge matches if its predicate is in this list. |

### Notes

- If `node_filter` is omitted entirely, all nodes in the subgraph receive full data.
- If `edge_filter` is omitted entirely, all edges receive full data including provenance.
- Omitting both filters is appropriate for small subgraphs or debugging, but will produce
  large responses on dense neighborhoods.
- Multiple seeds are useful when you want to explore the shared neighborhood of several
  entities simultaneously — for example, finding publications co-authored by two researchers,
  or finding diseases connected to a set of genes.
- If you do not yet have a canonical entity ID, call `search_entities()` first to resolve a
  name to an ID.

---

## Response Format

```json
{
  "seeds": ["<entity_id>", ...],
  "max_hops": 2,
  "node_count": <int>,
  "edge_count": <int>,
  "nodes": [
    {
      "id": "<entity_id>",
      "entity_type": "<type>",
      "<additional fields>": "..."
    }
  ],
  "edges": [
    {
      "subject": "<entity_id>",
      "predicate": "<predicate>",
      "object": "<entity_id>",
      "<additional fields>": "..."
    }
  ]
}
```

### Full Node

A node that matches the `node_filter` (or when no filter is specified) includes all available
metadata:

```json
{
  "id": "PUB:PMC2386281",
  "entity_type": "Publication",
  "title": "The Diagnosis of Cushing's Syndrome",
  "canonical_id": "PMID:18493314",
  "year": 2008,
  "journal": "Reviews in Endocrine and Metabolic Disorders",
  "authors": ["Stewart PM"],
  "abstract_snippet": "..."
}
```

### Stub Node

A node that does not match the `node_filter` appears as a stub with only identity information:

```json
{
  "id": "PERSON:67890",
  "entity_type": "Person"
}
```

### Full Edge

An edge that matches the `edge_filter` (or when no filter is specified) includes all provenance
and metadata:

```json
{
  "subject": "PERSON:12345",
  "predicate": "AUTHORED",
  "object": "PUB:PMC2386281",
  "confidence": 0.97,
  "provenance": [
    {
      "source_doc": "PMC2386281",
      "section": "metadata",
      "method": "structured_extraction",
      "evidence_type": "primary_authorship"
    }
  ]
}
```

### Stub Edge

An edge that does not match the `edge_filter` appears with topology only:

```json
{
  "subject": "PERSON:12345",
  "predicate": "COLLEAGUE_OF",
  "object": "PERSON:67890"
}
```

---

## LLM Prompt

The following section can be included in a system prompt or tool description to instruct an
LLM how to construct BFS queries.

---

### Using `bfs_query()`

To explore the knowledge graph, call `bfs_query()` with a JSON body. The query performs a
breadth-first search from one or more seed nodes and returns the resulting subgraph.

Nodes and edges in the subgraph are either **full** (all metadata and provenance included) or
**stub** (identity only), depending on the filters you specify. Filtering affects only what
data is returned, not which nodes and edges are included in the subgraph.

**If you do not yet have a canonical entity ID, call `search_entities()` first.**

#### Query structure

```json
{
  "seeds": ["<id>", ...],       // one or more starting entity IDs
  "max_hops": <int>,            // graph distance from seeds (1-3 recommended)
  "node_filter": {              // optional: which nodes get full data
    "entity_types": [...]       // e.g. ["Publication", "Disease", "Drug"]
  },
  "edge_filter": {              // optional: which edges get full data + provenance
    "predicates": [...]         // e.g. ["AUTHORED", "TREATS", "INHIBITS"]
  }
}
```

Stub nodes contain only `{id, entity_type}`.
Stub edges contain only `{subject, predicate, object}`.
Omitting a filter entirely returns full data for all nodes or edges respectively.

---

#### Example 1: Find an author's publications with provenance

You know Dr. Stewart's entity ID and want to retrieve her publications along with the evidence
that links her to each one. You don't need full metadata on other entities in the neighborhood.

```json
{
  "seeds": ["PERSON:12345"],
  "max_hops": 1,
  "node_filter": {
    "entity_types": ["Publication"]
  },
  "edge_filter": {
    "predicates": ["AUTHORED"]
  }
}
```

Result: Full data on Publication nodes, full provenance on AUTHORED edges. Any other nodes
or edges at hop 1 (e.g. institutional affiliations) appear as stubs.

---

#### Example 2: Explore a disease neighborhood, focusing on drugs

You want to understand what drugs are connected to Cushing's syndrome within two hops, without
being overwhelmed by the full metadata of every gene, symptom, and pathway in the neighborhood.

```json
{
  "seeds": ["UMLS:C0085084"],
  "max_hops": 2,
  "node_filter": {
    "entity_types": ["Drug"]
  }
}
```

Result: Full data on Drug nodes. All other node types (Disease, Gene, Symptom, etc.) appear
as stubs. All edges appear as stubs since no `edge_filter` targets specific predicates — you
can see the topology of the neighborhood without consuming context on provenance you haven't
asked for.

---

#### Example 3: Find shared connections between two entities

You want to explore what two researchers have in common — shared publications, shared diseases
they've written about, or shared collaborators — within two hops of either of them.

```json
{
  "seeds": ["PERSON:12345", "PERSON:67890"],
  "max_hops": 2,
  "node_filter": {
    "entity_types": ["Publication", "Disease"]
  },
  "edge_filter": {
    "predicates": ["AUTHORED", "DISCUSSES"]
  }
}
```

Result: BFS expands from both researchers simultaneously. The returned subgraph is the union
of both neighborhoods. Full data on Publications and Diseases they connect to; full provenance
on AUTHORED and DISCUSSES edges. Everything else is stubbed. Nodes reachable from both seeds
appear once, making shared connections directly visible.

---

## Design Considerations

### Topology and presentation are orthogonal

The subgraph returned by a BFS query is determined entirely by `seeds` and `max_hops`. Filters
have no effect on which nodes or edges are included — they only control how much data each
item carries in the response. This means the LLM always sees an accurate picture of the graph's
topology, regardless of what it filtered for. A stub node is not a missing node; it is a node
whose full metadata was not requested.

### Why stubs rather than omission

Omitting non-matching nodes entirely would produce a misleading picture of the graph. If
Dr. Jones appears as a stub rather than disappearing, the LLM knows that Stewart has a
connection to another person in the graph, and can issue a follow-up query for Jones if needed.
Omission would make the graph appear sparser than it is, causing the LLM to miss connections
it doesn't know to ask about.

### Edge provenance is expensive context

A single edge with strong multi-source support may carry a long provenance list — source
documents, confidence scores, evidence types, extraction methods. Returning full provenance on
every edge in a two-hop neighborhood would dominate the context window on most queries. The
`edge_filter` gives the LLM precise control over where that cost is paid.

### Multiple seeds enable relational queries

Accepting an array of seeds rather than a single seed allows the LLM to express relational
questions — "what do these entities have in common?" — without requiring a specialized query
type. The union semantics are simple to implement and simple to reason about.

### Filters are independently optional

`node_filter` and `edge_filter` compose independently. You can request full node data with
stub edges (useful when you want entity details but not provenance), full edge provenance with
stub nodes (useful when you want to audit support for a relationship without loading entity
metadata), both (for focused high-detail queries), or neither (for small graphs or debugging).
No special cases are required.

### BFS depth guidance

Depth 1 is appropriate for direct relationships: an author's publications, a drug's known
indications, a gene's associated diseases. Depth 2 surfaces indirect connections: diseases
associated with genes that a drug targets, co-authors of papers a researcher has written.
Depth 3 and beyond can return very large subgraphs on well-connected nodes and should be
used with targeted filters. When in doubt, start at depth 1 and increase.
