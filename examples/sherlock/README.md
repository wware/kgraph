# Sherlock Holmes Knowledge Graph Example

This directory contains a **complete, working reference example** demonstrating
how to build a domain-specific knowledge graph using **kgraph**, based on the
Sherlock Holmes canon.

It ingests *The Adventures of Sherlock Holmes* from Project Gutenberg and
constructs a queryable knowledge graph of:

* **Characters** (e.g. Sherlock Holmes, Irene Adler)
* **Locations** (e.g. 221B Baker Street, Scotland Yard)
* **Stories**
* **Relationships** (e.g. appears_in, co_occurs_with)

The example demonstrates **entity enrichment** with DBPedia, linking
fictional characters and locations to their DBPedia URIs when available.

The example is intentionally **readable, explicit, and idiomatic**, prioritizing clarity over cleverness.

---

## Why this example exists

This project is meant to answer a concrete question:

> *“What does a correct, end-to-end kgraph domain look like?”*

Specifically, it demonstrates:

* How to define a **DomainSchema**
* How to implement each **pipeline interface**
* How to enable **external entity enrichment** with DBPedia
* How canonical and provisional entities are created and promoted
* How documents, entities, and relationships flow through ingestion
* How to structure a non-trivial domain in a maintainable way

If you are planning to build your own kgraph-based domain, this example is
intended to be **copied, modified, and adapted**.

---

## High-level architecture

At a high level, the pipeline looks like this:

```
Gutenberg text
     ↓
DocumentParser
     ↓
EntityExtractor (mentions only)
     ↓
EntityResolver (canonical vs provisional)
     ↓
EntityEnricher (DBPedia URIs)
     ↓
RelationshipExtractor
     ↓
(optional) Embeddings
     ↓
Storage + Promotion
```

Key architectural principles illustrated here:

* **Parsers parse; extractors extract; resolvers resolve**
* **Relationship extractors never create entities**
* **Domains own entity and relationship vocabularies**
* **Pipeline components are swappable and testable**

---

## Directory structure

```
examples/sherlock/
├── README.md                 # This file
├── domain.py                 # DomainSchema + entity/relationship/document classes
├── data.py                   # Curated characters, locations, and stories
├── pipeline/
│   ├── parser.py             # SherlockDocumentParser
│   ├── mentions.py           # SherlockEntityExtractor (mentions only)
│   ├── resolve.py            # SherlockEntityResolver
│   ├── relationships.py      # SherlockRelationshipExtractor
│   └── embeddings.py         # SimpleEmbeddingGenerator (stub)
├── sources/
│   └── gutenberg.py          # Download + split utilities
├── scripts/
│   ├── ingest.py             # CLI-style ingestion entrypoint
│   └── query.py              # Demo queries
└── output/                   # Generated artifacts (gitignored)
```

---

## Quick start

From the **repository root**:

```bash
# Run the ingestion pipeline
uv run python -m examples.sherlock.scripts.ingest

# Run example queries
uv run python -m examples.sherlock.scripts.query
```

Expected behavior:

* Downloads and splits 12 Sherlock Holmes stories
* Ingests each story through the full kgraph pipeline
* Creates canonical entities for known characters and locations
* Enriches entities with DBPedia URIs (e.g., Sherlock Holmes → http://dbpedia.org/resource/Sherlock_Holmes)
* Generates relationships based on co-occurrence
* Prints a summary and example queries

---

## What gets built

After ingestion, the graph typically contains:

* **Stories**: 12 canonical story entities
* **Characters**: canonical characters from the curated list, enriched with DBPedia URIs
* **Locations**: canonical locations from the curated list, enriched with DBPedia URIs
* **Relationships**:

  * `appears_in` (character → story)
  * `co_occurs_with` (character ↔ character)

Provisional entities are also created internally when needed and may be
promoted depending on domain rules.

### DBPedia Enrichment

The example enables DBPedia enrichment for characters and locations. Well-known entities
get linked to their DBPedia resources:

* **Sherlock Holmes** → `http://dbpedia.org/resource/Sherlock_Holmes`
* **Dr. Watson** → `http://dbpedia.org/resource/Dr._Watson`
* **London** → `http://dbpedia.org/resource/London`
* **Baker Street** → `http://dbpedia.org/resource/Baker_Street`

Enrichment is optional and configurable. See `scripts/ingest.py` for the configuration
and [docs/enrichment.md](../../docs/enrichment.md) for details.

---

## Canonical vs provisional entities

This example explicitly demonstrates the **entity lifecycle**:

* **Canonical entities**

  * Come from curated domain knowledge (`data.py`)
  * Have stable, meaningful IDs (e.g. `holmes:char:SherlockHolmes`)
* **Provisional entities**

  * Created from raw mentions
  * Scoped to a document
  * Eligible for promotion based on usage and confidence

Promotion rules are defined in `SherlockDomainSchema.promotion_config`.

---

## Extending this example

This project is designed to be adapted.

### Add new characters or locations

Edit `data.py`:

```python
KNOWN_CHARACTERS["holmes:char:NewCharacter"] = {
    "name": "New Character",
    "aliases": ["Alias One", "Alias Two"],
    "role": "client",
}
```

### Add new relationship types

1. Define a new `BaseRelationship` subclass in `domain.py`
2. Register it in `SherlockDomainSchema.relationship_types`
3. Emit it from the relationship extractor

### Replace components

All pipeline components are swappable. For example:

* Replace the embedding generator with a real model
* Replace in-memory storage with a persistent backend
* Replace the co-occurrence heuristic with something smarter
* Disable enrichment by removing the `entity_enrichers` parameter
* Add additional enrichers (e.g., Wikidata, custom authority sources)

See `scripts/ingest.py` for the canonical wiring.

---

## `build_orchestrator()`

The ingestion script uses a helper function:

```python
build_orchestrator()
```

This function exists to:

* Provide a **single authoritative pipeline configuration**
* Demonstrate **entity enrichment** setup
* Reduce cognitive load for extenders
* Serve as executable documentation
* Make experimentation easy (swap one component, keep the rest)

You are not required to use it, but it is recommended for clarity and reuse.

---

## Design philosophy

This example intentionally favors:

* Explicitness over magic
* Correctness over shortcuts
* Separation of responsibilities
* Debuggability and testability

If something feels “a bit verbose,” that is by design — this is a teaching
artifact as much as a demo.

---

## Known limitations (intentional)

* Co-occurrence is based on simple paragraph heuristics
* Embeddings are stubbed and not semantically meaningful
* No coreference resolution or NLP libraries are used

These choices keep the example **understandable and dependency-light**.

---

## Related documentation

* `kgraph/docs/domains.md`
* `kgraph/docs/pipeline.md`
* `kgraph/docs/enrichment.md` ← **Entity enrichment guide**
* `kgraph/domain.py`
* `kgraph/pipeline/interfaces.py`

---

## Final note

If you are building your own kgraph domain, the recommended workflow is:

1. Copy this directory
2. Rename it
3. Replace the domain vocabulary
4. Iterate one pipeline component at a time

This Sherlock example exists so you don’t have to invent everything from scratch.
