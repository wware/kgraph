# kgraph

A domain-agnostic framework for building knowledge graphs from documents. Supports entity extraction, relationship mapping, and a two-pass ingestion pipeline that works across any knowledge domain (medical, legal, academic, etc.).

## Features

- **Domain-agnostic**: Define your own entity types, relationships, and validation rules
- **Two-pass ingestion**: Extract entities first, then relationships between them
- **Entity lifecycle**: Provisional entities promoted to canonical based on usage/confidence
- **Canonical ID system**: Abstractions for working with authoritative identifiers (UMLS, MeSH, HGNC, etc.)
- **Embedding support**: Semantic similarity for entity matching and duplicate detection
- **Async-first**: All storage and pipeline interfaces use async/await
- **Immutable models**: Thread-safe Pydantic models with frozen=True

## Installation

```bash
# Clone the repository
git clone https://github.com/wware/kgraph.git
cd kgraph

# Set up environment (requires Python 3.12+)
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Define your domain

```python
from kgraph import BaseEntity, BaseRelationship, DomainSchema, EntityStatus

class PersonEntity(BaseEntity):
    def get_entity_type(self) -> str:
        return "person"

    def get_canonical_id_source(self) -> str | None:
        return "my_authority" if self.status == EntityStatus.CANONICAL else None

class KnowsRelationship(BaseRelationship):
    def get_edge_type(self) -> str:
        return "knows"

class MyDomain(DomainSchema):
    @property
    def name(self) -> str:
        return "my_domain"

    @property
    def entity_types(self) -> dict:
        return {"person": PersonEntity}

    @property
    def relationship_types(self) -> dict:
        return {"knows": KnowsRelationship}

    # ... implement remaining abstract methods
```

### 2. Set up storage and pipeline

```python
from kgraph.storage import InMemoryEntityStorage, InMemoryRelationshipStorage, InMemoryDocumentStorage
from kgraph import IngestionOrchestrator

orchestrator = IngestionOrchestrator(
    domain=MyDomain(),
    parser=my_parser,
    entity_extractor=my_extractor,
    entity_resolver=my_resolver,
    relationship_extractor=my_rel_extractor,
    embedding_generator=my_embedder,
    entity_storage=InMemoryEntityStorage(),
    relationship_storage=InMemoryRelationshipStorage(),
    document_storage=InMemoryDocumentStorage(),
)
```

### 3. Ingest documents

```python
result = await orchestrator.ingest_document(
    raw_content=b"Document text mentioning Alice and Bob...",
    content_type="text/plain",
)

print(f"Extracted {result.entities_extracted} entities")
print(f"Created {result.relationships_extracted} relationships")
```

## Architecture

```
Document → Parser → Pass 1 (Entity Extraction) → Pass 2 (Relationship Extraction) → Storage
                           ↓                              ↓
                    EntityMention[]                BaseRelationship[]
                           ↓
                    Entity Resolution
                           ↓
                    BaseEntity[] (canonical or provisional)
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Implementing a Domain](docs/domains.md)
- [Storage Backends](docs/storage.md)
- [Pipeline Components](docs/pipeline.md)
- [API Reference](docs/api.md)

## Testing

```bash
uv run pytest
```

## Project Structure

```
kgraph/
├── entity.py              # BaseEntity, EntityStatus, EntityMention, PromotionConfig
├── relationship.py        # BaseRelationship
├── document.py            # BaseDocument
├── domain.py              # DomainSchema ABC
├── ingest.py              # IngestionOrchestrator
├── promotion.py           # PromotionPolicy ABC
├── canonical_id.py         # CanonicalId model
├── canonical_cache.py     # CanonicalIdCacheInterface ABC
├── canonical_cache_json.py  # JsonFileCanonicalIdCache implementation
├── canonical_lookup.py     # CanonicalIdLookupInterface ABC
├── canonical_helpers.py    # Helper functions for promotion policies
├── storage/
│   ├── interfaces.py      # Storage ABCs
│   └── memory.py          # In-memory implementation
└── pipeline/
    ├── interfaces.py       # Parser, Extractor, Resolver ABCs
    └── embedding.py       # EmbeddingGeneratorInterface
```

## License

MIT
