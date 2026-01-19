# Sherlock Holmes Example (kgraph)

This example demonstrates how to build a small, domain-specific knowledge graph using **kgraph**.
It downloads public-domain Sherlock Holmes stories from Project Gutenberg, extracts entities
(characters, locations, stories), and produces relationships (appears_in, co_occurs_with).

## Quick start

```bash
python -m examples.sherlock.ingest
python -m examples.sherlock.query
```
