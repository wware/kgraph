# Medical Literature Domain Extension

This package provides a kgraph domain extension for extracting knowledge from biomedical journal articles. It rewrites the med-lit-schema project as a kgraph domain package, following the same pattern as the Sherlock example.

## Architecture

### Key Design Decisions

1. **Papers are NOT documents.jsonl**: Source papers are `JournalArticle(BaseDocument)` instances used for extraction, not documentation assets. The `documents.jsonl` in bundles is for human-readable documentation only.

2. **Canonical IDs**: Entities use authoritative identifiers (UMLS, HGNC, RxNorm, UniProt) directly in `entity_id`, with additional mappings in `canonical_ids`.

3. **Pattern A Relationships**: All medical predicates (TREATS, CAUSES, INCREASES_RISK, etc.) use a single `MedicalClaimRelationship` class. The `predicate` field distinguishes the relationship type.

4. **Rich Metadata**: Paper metadata (study type, sample size, MeSH terms) and extraction provenance are preserved in `BaseDocument.metadata` and `BaseRelationship.metadata`.

## Domain Components

### Documents

- **JournalArticle**: Maps from med-lit-schema's `Paper` to kgraph's `BaseDocument`
  - `paper_id` → `document_id` (prefer `doi:`, else `pmid:`, else `paper_id`)
  - `PaperMetadata` → `metadata` dict
  - `extraction_provenance` → `metadata["extraction"]`

### Entities

- **DiseaseEntity**: Uses UMLS IDs (e.g., `C0006142`)
- **GeneEntity**: Uses HGNC IDs (e.g., `HGNC:1100`)
- **DrugEntity**: Uses RxNorm IDs (e.g., `RxNorm:1187832`)
- **ProteinEntity**: Uses UniProt IDs (e.g., `P38398`)
- **SymptomEntity**, **ProcedureEntity**, **BiomarkerEntity**, **PathwayEntity**

### Relationships

- **MedicalClaimRelationship**: Single class for all medical predicates
  - Supports TREATS, CAUSES, INCREASES_RISK, ASSOCIATED_WITH, INTERACTS_WITH, etc.
  - Evidence and provenance stored in `metadata`
  - Multi-paper aggregation via `source_documents` tuple

### Domain Schema

- **MedLitDomainSchema**: Defines entity types, relationship types, document types
  - Promotion config: min_usage_count=2, min_confidence=0.75
  - Predicate validation via `get_valid_predicates()` (e.g., Drug→Disease supports TREATS)

## Pipeline Components

### Parser

- **JournalArticleParser**: Converts raw input (JSON, PMC XML) to `JournalArticle`
  - Currently supports JSON format (from med-lit-schema's Paper)
  - PMC XML parsing TODO (can reuse med-lit-schema's pmc_parser.py logic)

### TODO: Remaining Pipeline Components

- **EntityExtractor**: Extract entity mentions from journal articles
- **EntityResolver**: Resolve mentions to canonical entities (UMLS, HGNC, etc.)
- **RelationshipExtractor**: Extract relationships with evidence and provenance
- **EmbeddingGenerator**: Generate biomedical embeddings for entities

## Usage Example

```python
from kgraph.ingest import IngestionOrchestrator
from examples.medlit.domain import MedLitDomainSchema
from examples.medlit.pipeline.parser import JournalArticleParser

# Create domain schema
domain = MedLitDomainSchema()

# Create parser
parser = JournalArticleParser()

# Create orchestrator (with other pipeline components)
orchestrator = IngestionOrchestrator(
    domain=domain,
    parser=parser,
    # ... other components
)

# Ingest a paper
with open("paper.json", "rb") as f:
    result = await orchestrator.ingest_document(
        raw_content=f.read(),
        content_type="application/json",
        source_uri="https://example.com/paper.json"
    )
```

## Mapping from med-lit-schema

### Paper → JournalArticle

| med-lit-schema | kgraph |
|----------------|--------|
| `Paper.paper_id` | `JournalArticle.document_id` (prefer `doi:`, else `pmid:`) |
| `Paper.title` | `JournalArticle.title` |
| `Paper.abstract` | `JournalArticle.abstract` |
| `Paper.abstract + full_text` | `JournalArticle.content` |
| `Paper.authors` | `JournalArticle.authors` |
| `PaperMetadata` | `JournalArticle.metadata` |
| `ExtractionProvenance` | `JournalArticle.metadata["extraction"]` |

### Entity Mapping

| med-lit-schema | kgraph |
|----------------|--------|
| `Disease.entity_id` (UMLS) | `DiseaseEntity.entity_id` |
| `Disease.umls_id` | `DiseaseEntity.canonical_ids["umls"]` |
| `Disease.mesh_id` | `DiseaseEntity.canonical_ids["mesh"]` |
| `Gene.entity_id` (HGNC) | `GeneEntity.entity_id` |
| `Gene.hgnc_id` | `GeneEntity.canonical_ids["hgnc"]` |
| `Drug.entity_id` (RxNorm) | `DrugEntity.entity_id` |
| `Drug.rxnorm_id` | `DrugEntity.canonical_ids["rxnorm"]` |

### Relationship Mapping

| med-lit-schema | kgraph |
|----------------|--------|
| `AssertedRelationship.subject_id` | `MedicalClaimRelationship.subject_id` |
| `AssertedRelationship.predicate` | `MedicalClaimRelationship.predicate` |
| `AssertedRelationship.object_id` | `MedicalClaimRelationship.object_id` |
| `AssertedRelationship.confidence` | `MedicalClaimRelationship.confidence` |
| `AssertedRelationship.evidence` | `MedicalClaimRelationship.metadata["evidence"]` |
| `AssertedRelationship.section` | `MedicalClaimRelationship.metadata["section"]` |
| Paper ID | `MedicalClaimRelationship.source_documents` (tuple) |

## Usage

### Basic Ingestion

Process Paper JSON files and generate a bundle:

```bash
cd /path/to/kgraph
uv run python -m examples.medlit.scripts.ingest \
    --input-dir /path/to/json_papers \
    --output-dir medlit_bundle \
    --limit 10  # Optional: limit number of papers for testing
```

### Processing All Papers

```bash
uv run python -m examples.medlit.scripts.ingest \
    --input-dir /home/wware/med-lit-schema/output/json_papers \
    --output-dir medlit_bundle
```

This will:
1. Process all Paper JSON files in the input directory
2. Extract entities and relationships (if present in the Paper JSON)
3. Generate a kgraph bundle with:
   - `manifest.json` - Bundle metadata
   - `entities.jsonl` - All extracted entities
   - `relationships.jsonl` - All extracted relationships
   - `documents.jsonl` - Documentation assets (if provided)
   - `docs/` - Documentation directory

## Next Steps

1. **Entity/Relationship Extraction**: The current pipeline works with pre-extracted entities/relationships from Paper JSON. To extract from raw text:
   - Integrate NER models (BioBERT, scispaCy) into `MedLitEntityExtractor`
   - Integrate relationship extraction (LLM, pattern matching) into `MedLitRelationshipExtractor`

2. **PMC XML parser**: Port med-lit-schema's PMC XML parsing logic to `JournalArticleParser`

3. **Tests**: Add unit tests for domain components

4. **Integration**: Test end-to-end ingestion → export → kgserver loading

5. **Enhancements**:
   - Better entity resolution (embedding similarity, external authority lookup)
   - Biomedical embedding models (BioBERT, etc.)
   - Relationship extraction from text (currently uses pre-extracted relationships)

## References

- [kgraph architecture docs](../../docs/architecture.md)
- [Domain extension guide](../../docs/domains.md)
- [Sherlock example](../sherlock/) - Reference implementation
- [med-lit-schema](../../../med-lit-schema/) - Original project
