# Knowledge Graph Framework

A domain-agnostic framework for building knowledge graphs from documents. Supports entity extraction, relationship mapping, and a two-pass ingestion pipeline that works across any knowledge domain (medical, legal, academic, etc.).

## Quick Start

```bash
# Install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest
```

## Core Concepts

### Entities

Entities are the nodes in your knowledge graph. They can be:

- **Canonical**: Assigned a stable ID from an authoritative source (e.g., UMLS for medical terms, DBPedia for general concepts)
- **Provisional**: Awaiting promotion based on usage count and confidence scores

```python
from kgraph import BaseEntity, EntityStatus

class DrugEntity(BaseEntity):
    def get_entity_type(self) -> str:
        return "drug"

    def get_canonical_id_source(self) -> str | None:
        return "RxNorm" if self.status == EntityStatus.CANONICAL else None
```

### Relationships

Relationships are edges connecting entities. Each domain defines valid predicates.

```python
from kgraph import BaseRelationship

class TreatsRelationship(BaseRelationship):
    def get_edge_type(self) -> str:
        return "treats"
```

### Domain Schema

Define your domain by implementing `DomainSchema`:

```python
from kgraph import DomainSchema, PromotionConfig

class MedicalDomain(DomainSchema):
    @property
    def name(self) -> str:
        return "medical"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {"drug": DrugEntity, "disease": DiseaseEntity}

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {"treats": TreatsRelationship, "causes": CausesRelationship}

    # ... implement remaining abstract methods
```

## Documentation

- [Architecture Overview](architecture.md)
- [Implementing a Domain](domains.md)
- [Storage Backends](storage.md)
- [Pipeline Components](pipeline.md)
- [API Reference](api.md)
