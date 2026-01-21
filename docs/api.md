# API Reference

## Core Classes

### kgraph.entity

#### EntityStatus

```python
class EntityStatus(str, Enum):
    CANONICAL = "canonical"    # Stable ID from authoritative source
    PROVISIONAL = "provisional" # Awaiting promotion
```

#### BaseEntity

Abstract base for all domain entities.

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | `str` | Domain-specific canonical ID or provisional UUID |
| `status` | `EntityStatus` | Canonical or provisional |
| `name` | `str` | Primary name/label |
| `synonyms` | `tuple[str, ...]` | Alternative names |
| `embedding` | `tuple[float, ...] \| None` | Semantic vector |
| `dbpedia_uri` | `str \| None` | Cross-domain linking |
| `confidence` | `float` | Confidence in canonical assignment (0.0-1.0) |
| `usage_count` | `int` | Number of references |
| `created_at` | `datetime` | Creation timestamp |
| `source` | `str` | Origin indicator |
| `metadata` | `dict` | Domain-specific data |

Abstract methods:
- `get_entity_type() -> str`: Domain-specific type identifier
- `get_canonical_id_source() -> str | None`: Authoritative ID source

#### EntityMention

Raw entity extraction from a document.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Exact text span |
| `entity_type` | `str` | Domain-specific type |
| `start_offset` | `int` | Start position in source |
| `end_offset` | `int` | End position in source |
| `confidence` | `float` | Extraction confidence |
| `context` | `str \| None` | Surrounding text |
| `metadata` | `dict` | Extraction metadata |

#### PromotionConfig

Controls provisional-to-canonical promotion thresholds.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_usage_count` | `int` | 3 | Minimum appearances |
| `min_confidence` | `float` | 0.8 | Minimum confidence |
| `require_embedding` | `bool` | True | Embedding required |

### kgraph.relationship

#### BaseRelationship

Abstract base for domain relationships.

| Field | Type | Description |
|-------|------|-------------|
| `subject_id` | `str` | Source entity ID |
| `predicate` | `str` | Relationship type |
| `object_id` | `str` | Target entity ID |
| `confidence` | `float` | Confidence score |
| `source_documents` | `tuple[str, ...]` | Supporting document IDs |
| `created_at` | `datetime` | Creation timestamp |
| `last_updated` | `datetime \| None` | Last update timestamp |
| `metadata` | `dict` | Domain-specific data |

Abstract methods:
- `get_edge_type() -> str`: Edge type category

### kgraph.document

#### BaseDocument

Abstract base for processed documents.

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | `str` | Unique identifier |
| `title` | `str \| None` | Document title |
| `content` | `str` | Full text content |
| `content_type` | `str` | MIME type |
| `source_uri` | `str \| None` | Original location |
| `created_at` | `datetime` | Ingestion timestamp |
| `metadata` | `dict` | Domain-specific data |

Abstract methods:
- `get_document_type() -> str`: Domain-specific document type
- `get_sections() -> list[tuple[str, str]]`: Section name/content pairs

### kgraph.domain

#### DomainSchema

Abstract schema definition for a knowledge domain.

Abstract properties:
- `name -> str`: Domain identifier
- `entity_types -> dict[str, type[BaseEntity]]`: Entity type registry
- `relationship_types -> dict[str, type[BaseRelationship]]`: Relationship type registry
- `document_types -> dict[str, type[BaseDocument]]`: Document type registry
- `canonical_id_sources -> dict[str, str]`: Entity type to ID authority mapping

Optional override:
- `promotion_config -> PromotionConfig`: Domain-specific promotion thresholds

Abstract methods:
- `validate_entity(entity: BaseEntity) -> bool`
- `validate_relationship(rel: BaseRelationship) -> bool`

Optional override:
- `get_valid_predicates(subject_type: str, object_type: str) -> list[str]`

### kgraph.ingest

#### IngestionOrchestrator

Orchestrates two-pass document ingestion.

Constructor parameters:
- `domain: DomainSchema`
- `parser: DocumentParserInterface`
- `entity_extractor: EntityExtractorInterface`
- `entity_resolver: EntityResolverInterface`
- `relationship_extractor: RelationshipExtractorInterface`
- `embedding_generator: EmbeddingGeneratorInterface`
- `entity_storage: EntityStorageInterface`
- `relationship_storage: RelationshipStorageInterface`
- `document_storage: DocumentStorageInterface`

Methods:

```python
async def ingest_document(
    self,
    raw_content: bytes,
    content_type: str,
    source_uri: str | None = None,
) -> DocumentResult
```

```python
async def ingest_batch(
    self,
    documents: Sequence[tuple[bytes, str, str | None]],
) -> IngestionResult
```

```python
async def run_promotion(self) -> list[BaseEntity]
```

```python
async def merge_entities(
    self,
    source_ids: Sequence[str],
    target_id: str,
) -> bool
```

#### DocumentResult

Result of processing a single document.

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | `str` | Processed document ID |
| `entities_extracted` | `int` | Total mentions found |
| `entities_new` | `int` | New entities created |
| `entities_existing` | `int` | Existing entities matched |
| `relationships_extracted` | `int` | Relationships created |
| `errors` | `tuple[str, ...]` | Error messages |

#### IngestionResult

Result of batch ingestion.

| Field | Type | Description |
|-------|------|-------------|
| `documents_processed` | `int` | Total documents |
| `documents_failed` | `int` | Documents with errors |
| `total_entities_extracted` | `int` | Total mentions |
| `total_relationships_extracted` | `int` | Total relationships |
| `document_results` | `tuple[DocumentResult, ...]` | Per-document results |
| `errors` | `tuple[str, ...]` | Global errors |

## Storage Interfaces

See [Storage Backends](storage.md) for full interface documentation.

- `EntityStorageInterface`
- `RelationshipStorageInterface`
- `DocumentStorageInterface`

In-memory implementations:
- `InMemoryEntityStorage`
- `InMemoryRelationshipStorage`
- `InMemoryDocumentStorage`

## Pipeline Interfaces

See [Pipeline Components](pipeline.md) for implementation guidance.

- `DocumentParserInterface`
- `EntityExtractorInterface`
- `EntityResolverInterface`
- `RelationshipExtractorInterface`
- `EmbeddingGeneratorInterface`

## Bundle Export

### kgraph.export

#### write_bundle

Export a knowledge graph to a bundle format that can be loaded by the KG server.

```python
async def write_bundle(
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
    bundle_path: Path,
    domain: str,
    label: Optional[str] = None,
    docs: Optional[Path] = None,
    description: Optional[str] = None,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `entity_storage` | `EntityStorageInterface` | Storage interface for entities |
| `relationship_storage` | `RelationshipStorageInterface` | Storage interface for relationships |
| `bundle_path` | `Path` | Directory path for the bundle output |
| `domain` | `str` | Knowledge domain identifier (e.g., "sherlock", "medical") |
| `label` | `str \| None` | Optional human-readable bundle label |
| `docs` | `Path \| None` | Optional path to directory containing documentation assets (markdown files, images, etc.) |
| `description` | `str \| None` | Optional description for bundle metadata |

**Bundle Output:**

The function creates a bundle directory containing:

- `manifest.json` - Bundle metadata and file references
- `entities.jsonl` - Entity data in JSONL format
- `relationships.jsonl` - Relationship data in JSONL format
- `documents.jsonl` - (optional) List of documentation assets with paths and content types
- `docs/` - (optional) Directory containing documentation files (markdown, images, etc.)

**Documentation Assets:**

If the `docs` parameter is provided, the function will:

1. Recursively copy all files from the source directory to `bundle_path/docs/`
2. Create `documents.jsonl` listing each asset with its path (relative to bundle root) and MIME type
3. Add a `documents` field to the manifest referencing `documents.jsonl`

Example:

```python
from kgraph.export import write_bundle
from pathlib import Path

# Export bundle with documentation
await write_bundle(
    entity_storage=my_entity_storage,
    relationship_storage=my_relationship_storage,
    bundle_path=Path("./my_bundle"),
    domain="medical",
    label="medical-literature-2024",
    docs=Path("./my_docs"),  # Contains markdown files, images, etc.
    description="Medical literature knowledge graph for 2024"
)
```
