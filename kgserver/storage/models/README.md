# Storage Models (SQLModel Schemas)

This directory contains SQLModel persistence schemas that define the database structure for storing knowledge graphs.

## Overview

These are **Persistence Models** - flattened database representations optimized for storage and querying.

## Available Models

### Entity (`entity.py`)

Single table with primary key `entity_id`.

**Key Fields:**
```python
entity_id: str              # Primary key
entity_type: str            # Type of the entity
name: Optional[str]         # Entity name
synonyms: List[str]         # List of synonyms
properties: dict[str, Any]  # Flexible JSONB properties
```

**Indexes:**
- Primary key on `entity_id`
- Index on `entity_type` for filtering
- Index on `name` for lookups

**Example:**
```python
from storage.models.entity import Entity

entity = Entity(
    entity_id="E001",
    entity_type="person",
    name="Alice",
    synonyms=["Alicia"],
    properties={"age": 30, "city": "New York"}
)
```

### Relationship (`relationship.py`)

Table for storing relationships between entities.

**Key Fields:**
```python
id: UUID                    # Primary key
subject_id: str             # ID of the subject entity
predicate: str              # Type of the relationship (e.g., "knows", "has_job")
object_id: str              # ID of the object entity
confidence: Optional[float] # Confidence score (0.0 - 1.0)
source_documents: List[str] # List of documents supporting this relationship
properties: dict[str, Any]  # Relationship-specific attributes
```

**Indexes:**
- Primary key on `id`
- Composite unique constraint on `(subject_id, predicate, object_id)`
- Index on `subject_id`
- Index on `object_id`
- Index on `predicate`

**Example:**
```python
from storage.models.relationship import Relationship

relationship = Relationship(
    subject_id="E001",  # Alice
    predicate="knows",
    object_id="E002",   # Bob
    confidence=0.9,
    source_documents=["doc1.txt"],
    properties={"since": "2020-01-01"}
)
```

## Database Tables

When using PostgreSQL or SQLite, tables are created with these names:

- `entity`
- `relationship`

## Performance Optimization

### Indexes

All models include appropriate indexes:

**Entity:**
- Primary key: `entity_id`
- Index: `entity_type` for filtering
- Index: `name` for lookups

**Relationship:**
- Primary key: `id`
- Composite unique constraint: `(subject_id, predicate, object_id)`
- Index: `subject_id`
- Index: `object_id`
- Index: `predicate`

### JSON vs Separate Columns

Use JSON (`properties`) for:
- Flexible/optional attributes
- Rarely queried fields

Use separate columns for:
- Frequently queried fields
- Fields requiring indexes
- Fields with strict types

## Testing

Models are tested in `tests/storage/models/`:

```bash
# Test entity model
pytest tests/storage/models/test_entity.py -v

# Test all models
pytest tests/storage/models/ -v
```

## Further Reading

- [Storage Layer Overview](../README.md)
- [Storage Backends](../backends/README.md)