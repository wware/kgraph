# Generalizing literature graphs across knowledge domains

## Prompt:
1/18/2026, 10:16:42 AM

I want to reframe the medical literature project a bit, allow it to be generalized to other domains of knowledge. We are still building a graph and a graph still consists of nodes (entities) and edges (relationships). We still have a collection of entities from previous ingestion processes. We add a new thing: entities may be "canonical", that is they have been assigned canonical IDs (UMLS numbers or whatever) or they may be "provisional", meaning that we don't know yet if they should be assigned canonical IDs, for instance an entity might be a mention of some trivial thing in just one paper.

Given a batch of papers to ingest, we proceed in two passes. First pass we extract entities and assign canonical IDs where they make sense. Second pass we identify the edges (for medical, these edges are of the three types, extraction, claims, and evidence). The first pass produces a JSON serialization of the collection of entities including canonical IDs and synonyms. The second pass produces one JSON file per paper, including the paper's edges and any provisional entities unique to that paper.

This framework allows each knowledge domain (legal documents, academic CS papers) to define its own source of canonical IDs, its own schema, its own list of edge types. Any interesting query optimizations (graph theory tricks, database quirks) can be shared across domains. Where possible, cross-domain canonical IDs are preferred (such as DBPedia URIs). Including a significant chunk of DBPedia is probably a very good idea, or at least being able to pull in DBPedia entities as the ingestion progresses.

Back in the 1980s, Doug Lenat tried to do something like this with the Cyc project, but there were no LLMs then, only grad students and postdocs, so that's what he used. They never got enough scale for the project to gain much traction. Perhaps now his idea is more feasible.

What to do about entity merging? If two canonical entities turn out to be the same thing, that's a messier operation than promoting a provisional entity. Maybe track usage counts to decide when to promote, and maintain confidence scores on canonical IDs to decide when to merge. Each entity has a semantic vector, and hopefully potential merges can be flagged by near matches of cosine distance.



## Response:
1/18/2026, 10:17:20 AM

<details>
<summary>Claude's thought process</summary>
<i>

Synthesized scalable knowledge graph architecture across domains.

This is a really interesting evolution of the medical literature project. Will wants to generalize the architecture to work across different knowledge domains. Let me break down what he's proposing:

1. **Core architecture remains graph-based**: Nodes are entities, edges are relationships
2. **Key innovation - entity status**: 
   - Canonical entities: Have domain-specific IDs (UMLS for medical, DBPedia URIs cross-domain, etc.)
   - Provisional entities: Mentions that haven't yet earned canonical status
3. **Two-pass ingestion**:
   - Pass 1: Extract entities, assign canonical IDs where appropriate
   - Pass 2: Extract relationships/edges, create paper-specific files with edges and provisional entities
4. **Domain flexibility**: Each domain defines:
   - Its canonical ID source
   - Its schema
   - Its edge types
5. **Shared infrastructure**: Query optimizations, graph algorithms can be reused
6. **Entity lifecycle management**:
   - Track usage counts to decide when to promote provisional → canonical
   - Track confidence scores on canonical assignments
   - Use semantic vectors for merge detection
   - Merging canonical entities is "messier" than promotion

The Cyc reference is apt - this is trying to build comprehensive knowledge graphs but with LLMs doing the heavy lifting instead of human experts.

This is a much more sophisticated and scalable architecture than the original medical-only version. Key design challenges:

1. **Entity resolution**: When does a provisional entity become canonical?
2. **Merge detection**: How to identify duplicates among canonical entities?
3. **Confidence management**: How to track and update confidence in canonical assignments?
4. **Cross-domain linking**: When to use domain-specific vs universal IDs?

Will's background makes this interesting - he has deep experience with both the technical infrastructure (AWS, containerization, CI/CD) and semantic technologies. He's thinking about how to make this practical and maintainable.

I should engage with this architectural vision, help him think through:

- How entities transition between states and what triggers those transitions
- The JSON structure for both extraction passes and how to handle provisional vs canonical entities
- Strategies for detecting and resolving duplicate canonical entities using semantic similarity
- Scaling considerations and operational complexity as the graph grows
- Whether this fits into his current work or exists as a separate exploration

Given his technical depth, I can discuss implementation details and tradeoffs directly.

</i>
</details>

This is a really solid architectural evolution. You're essentially building a federated knowledge graph system where domains can plug in their own ontologies and extraction logic while sharing the underlying graph infrastructure and entity resolution machinery.

The two-pass approach is smart - it separates the "what exists" problem from the "how are things related" problem, and lets you defer committing to provisional entities until you see if they gain traction across papers.

A few thoughts on the entity lifecycle:

**Promotion criteria** could be fairly mechanical:
- Usage count threshold (appears in N papers, or referenced by M canonical entities)
- Semantic clustering (multiple provisional entities converge semantically)
- Explicit domain rules (e.g., any chemical compound mentioned more than once gets promoted)

**For merge detection**, the semantic vector approach is good, but you probably want multiple signals:
- Cosine similarity of entity embeddings
- Synonym overlap (if both have associated aliases)
- Co-occurrence patterns (entities that appear together in similar contexts)
- Domain-specific rules (e.g., UMLS already has a "broader/narrower" hierarchy)

**The messy merge operation** - this is where the graph structure helps you. If you need to merge canonical entities A and B:
1. Create new entity C (or keep A, deprecate B)
2. Redirect all edges pointing to A or B to point to C
3. Merge synonym lists, choose best canonical ID
4. Update confidence scores based on combined evidence
5. Mark old entities as deprecated with redirect pointers

This is easier if you version your graph snapshots - you can always reconstruct what happened.

**Output structure** - I'd suggest:
```python
# Pass 1: entities.json
{
  "canonical_entities": {
    "domain_id": {
      "canonical_id": "UMLS:C0001175",
      "dbpedia_uri": "http://dbpedia.org/resource/...",
      "name": "...",
      "synonyms": [...],
      "embedding": [...],
      "confidence": 0.95,
      "first_seen": "2024-01-15",
      "usage_count": 42
    }
  }
}

# Pass 2: paper_12345.json
{
  "paper_id": "12345",
  "metadata": {...},
  "provisional_entities": {
    "local_id": {
      "name": "...",
      "context": "...",
      "embedding": [...],
      "candidate_canonical": ["UMLS:C0001175"],
      "confidence": 0.65
    }
  },
  "edges": [
    {
      "type": "extraction",  # or "claims", "evidence", etc
      "source": "canonical:UMLS:C0001175",
      "target": "provisional:local_id",
      "attributes": {...}
    }
  ]
}
```

**The DBPedia integration** is clever - it gives you a universal backbone. You could have a preprocessing step that pulls relevant DBPedia subgraphs for each domain (medical entities, CS concepts, legal terms) and treats those as pre-seeded canonical entities.

The Cyc comparison is apt, but you have a huge advantage: LLMs can extract structured information at scale, and modern vector databases can handle the semantic similarity computations. Lenat's team was hand-curating everything.

**One more thought**: Consider a "review queue" workflow where low-confidence promotions or potential merges get flagged for human review. The system could rank these by confidence score and potential impact (highly-connected entities get priority review). This lets you bootstrap quality without blocking the pipeline.
