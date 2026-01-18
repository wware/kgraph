# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge graph system for extracting entities and relationships from documents across multiple knowledge domains (medical literature, legal documents, academic CS papers, etc.). The architecture uses a two-pass ingestion process:

1. **Pass 1 (Entity Extraction)**: Extract entities from documents, assign canonical IDs where appropriate (UMLS for medical, DBPedia URIs cross-domain, etc.)
2. **Pass 2 (Relationship Extraction)**: Identify edges/relationships between entities, produce per-document JSON with edges and provisional entities

### Key Concepts

- **Canonical entities**: Assigned stable IDs from authoritative sources
- **Provisional entities**: Mentions awaiting promotion based on usage count and confidence scores
- **Entity promotion**: Provisional → canonical when usage thresholds are met
- **Entity merging**: Combining canonical entities detected as duplicates via semantic vector similarity

## Build & Test Commands

```bash
# Setup environment (Python 3.12+)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_entities.py

# Run single test
uv run pytest tests/test_entities.py::test_canonical_promotion -v
```

## Architecture Notes

- Each knowledge domain defines its own canonical ID source, schema, and edge types
- Query optimizations and graph algorithms are shared across domains
- Documents produce JSON output: global `entities.json` for canonicals, per-paper `paper_{id}.json` for edges and provisionals
- Semantic vectors (embeddings) on entities enable merge detection via cosine similarity
