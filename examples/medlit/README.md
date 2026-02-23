# Medical Literature Domain Extension

This package provides a kgraph domain extension for extracting knowledge from biomedical journal articles. It rewrites the med-lit-schema project as a kgraph domain package, following the same pattern as the Sherlock example.

## Architecture

### Key Design Decisions

1. **Papers are NOT doc_assets.jsonl**: Source papers are `JournalArticle(BaseDocument)` instances used for extraction, not documentation assets. The `doc_assets.jsonl` in bundles is for human-readable documentation only.

2. **Canonical IDs**: Entities use authoritative identifiers (UMLS, HGNC, RxNorm, UniProt) directly in `entity_id`, with additional mappings in `canonical_ids`.

3. **Pattern A Relationships**: All medical predicates (treats, causes, increases_risk, etc.) use a single `MedicalClaimRelationship` class. The `predicate` field distinguishes the relationship type.

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
- **LocationEntity**: Geographic locations for epidemiological analysis (uses provisional IDs)
- **EthnicityEntity**: Ethnic/population groups for health disparities research (uses provisional IDs)

### Relationships

- **MedicalClaimRelationship**: Single class for all medical predicates
  - Supports treats, causes, increases_risk, associated_with, interacts_with, etc.
  - Evidence and provenance stored in `metadata`
  - Multi-paper aggregation via `source_documents` tuple

### Domain Schema

- **MedLitDomainSchema**: Defines entity types, relationship types, document types
  - Promotion config: min_usage_count=2, min_confidence=0.75
  - Predicate validation via `get_valid_predicates()` (e.g., Drug→Disease supports treats)

## Pipeline Components

### Parser

- **JournalArticleParser**: Converts raw input (JSON, PMC XML) to `JournalArticle`
  - Currently supports JSON format (from med-lit-schema's Paper)
  - PMC XML parsing TODO (can reuse med-lit-schema's pmc_parser.py logic)

### Entity Extractors

- **MedLitNEREntityExtractor**: Local NER model (e.g. BC5CDR) for fast entity extraction; optional `.[ner]` dependency.
- **LLM-based extractor**: Ollama-based entity extraction with type normalization.

### TODO: Remaining Pipeline Components

- **EntityResolver**: Resolve mentions to canonical entities (UMLS, HGNC, etc.)
- **RelationshipExtractor**: Extract relationships with evidence and provenance
- **EmbeddingGenerator**: Generate biomedical embeddings for entities

## Two-pass ingestion (per-paper bundles)

Pass 1 writes one JSON bundle per paper (immutable); Pass 2 deduplicates and merges without modifying those files. See **INGESTION.md** for the two-pass flow and **LLM_SETUP.md** for configuring the LLM (Anthropic, OpenAI, Ollama) for Pass 1.

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

Process Paper JSON/XML files and generate a bundle. You can use either **NER** (local, fast) or **LLM** (Ollama) for entity extraction; relationship extraction still uses the LLM when `--use-ollama` is set.

**With NER entity extraction (recommended for speed):**

```bash
pip install -e ".[ner]"   # one-time: install transformers/torch for NER
cd /path/to/kgraph
uv run python -m examples.medlit.scripts.ingest \
    --input-dir /path/to/papers \
    --output-dir medlit_bundle \
    --entity-extractor ner \
    --use-ollama \
    --limit 10  # Optional: limit number of papers for testing
```

**With LLM entity extraction (default):**

```bash
uv run python -m examples.medlit.scripts.ingest \
    --input-dir /path/to/papers \
    --output-dir medlit_bundle \
    --entity-extractor llm \
    --use-ollama \
    --limit 10
```

### Processing with Parallel Extraction

Use multiple workers for faster processing:

```bash
uv run python -m examples.medlit.scripts.ingest \
    --input-dir /path/to/papers \
    --output-dir medlit_bundle \
    --entity-extractor ner \
    --ner-model tner/roberta-base-bc5cdr \
    --use-ollama \
    --workers 3 \
    --progress-interval 15
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir` | (required) | Directory containing Paper JSON/XML files |
| `--output-dir` | `medlit_bundle` | Output directory for the bundle |
| `--entity-extractor` | `llm` | Entity extraction: `llm` (Ollama) or `ner` (local NER model) |
| `--ner-model` | `tner/roberta-base-bc5cdr` | HuggingFace model for NER (BC5CDR chemical/disease) |
| `--use-ollama` | `false` | Use Ollama for relationship extraction (and entity extraction if `--entity-extractor llm`) |
| `--ollama-model` | `llama3.1:8b` | Ollama model name |
| `--ollama-host` | `localhost:11434` | Ollama server URL |
| `--ollama-timeout` | `300` | Timeout in seconds for Ollama requests |
| `--workers` | `1` | Number of parallel workers for extraction |
| `--progress-interval` | `30` | Progress report interval in seconds |
| `--limit` | (none) | Limit number of papers to process |
| `--cache-file` | (auto) | Path to canonical ID lookup cache file |

This will:
1. Process all Paper JSON files in the input directory
2. Extract entities and relationships (if present in the Paper JSON)
3. Generate a kgraph bundle with:
   - `manifest.json` - Bundle metadata
   - `entities.jsonl` - All extracted entities
   - `relationships.jsonl` - All extracted relationships
   - `doc_assets.jsonl` - Documentation assets (if provided)
   - `docs/` - Documentation directory

## Resource Requirements

### Current Implementation (Ollama LLM)

The current implementation uses **Ollama** for LLM-based extraction:

- **Entity Extraction**: LLM-based NER with type normalization and validation
- **Relationship Extraction**: LLM-based with semantic validation (prevents contradictory predicates)
- **Embeddings**: Ollama embeddings for entity similarity and merge detection
- **Canonical ID Lookup**: External API calls to UMLS, UniProt, HGNC, DBPedia (with caching)

Requirements:
- **Ollama** running locally (default: `http://localhost:11434`)
- Recommended model: `llama3.1:8b` or larger
- Network access for canonical ID lookups (cached to reduce API calls)

### Extraction Features

1. **Entity Type Normalization**:
   - Handles LLM mistakes (`"test"` → `"procedure"`, `"marker"` → `"biomarker"`)
   - Handles pipe-separated types (`"drug|protein"` → `"drug"`)
   - Validates against domain schema

2. **Relationship Validation**:
   - Semantic validation prevents contradictory predicates
   - Evidence-based filtering (e.g., rejects "treats" with only negative evidence)

3. **Parallel Processing**:
   - `--workers N` for concurrent extraction
   - Progress reporting with rate estimates

4. **Bundle Version Tracking**:
   - Git commit hash included in bundle manifest for reproducibility

## Completed Features

- ✅ **NER-based Entity Extraction**: Optional local NER model (BC5CDR) for fast entity extraction; install with `.[ner]`
- ✅ **LLM-based Entity Extraction**: Extracts entities from raw text using Ollama
- ✅ **LLM-based Relationship Extraction**: Extracts relationships with semantic validation
- ✅ **Entity Type Normalization**: Handles LLM mistakes and validates against schema
- ✅ **Parallel Processing**: `--workers` flag for concurrent extraction
- ✅ **Progress Reporting**: Real-time progress with rate estimates
- ✅ **Canonical ID Lookup**: UMLS, UniProt, HGNC, DBPedia with caching
- ✅ **Location/Ethnicity Entities**: Epidemiological entity types
- ✅ **Bundle Version Tracking**: Git hash in manifest for reproducibility
- ✅ **Unit Tests**: Tests for normalization, progress tracking, git hash

## Next Steps

1. **PMC XML parser**: Improve PMC XML parsing for complex article structures

2. **Integration**: Test end-to-end ingestion → export → kgserver loading

3. **Enhancements**:
   - Biomedical embedding models (BioBERT, PubMedBERT)
   - Better entity resolution (embedding similarity thresholds)
   - Canonical IDs for location/ethnicity (GeoNames, ISO codes)

## References

- [kgraph architecture docs](../../docs/architecture.md)
- [Domain extension guide](../../docs/domains.md)
- [Sherlock example](../sherlock/) - Reference implementation
- [med-lit-schema](../../../med-lit-schema/) - Original project
