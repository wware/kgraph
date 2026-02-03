# Work Order: MedLit as an Extension of kgschema/

## Executive Summary

**Goal:** Create `examples/medlit_schema/` as a definitions-only package demonstrating domain-specific extension of the kgraph framework for medical literature knowledge graphs.

**Key Design Decisions:**
1. ✅ **Evidence as first-class entity** (not relationship metadata) for database indexing, canonical IDs, and multi-hop traceability
2. ✅ **Canonical ID format for Evidence:** `{paper_id}:{section}:{paragraph}:{method}` enables immediate promotion
3. ✅ **Rich bibliographic model** from `med-lit-schema/`: Paper with authors, journal, study metadata, MeSH terms, extraction provenance
4. ✅ **Mandatory evidence for medical relationships:** Pydantic validation prevents evidence-free assertions
5. ✅ **Ontology integration:** UMLS, HGNC, RxNorm, UniProt for entities; ECO, OBI, STATO for evidence classification

**Traceability Requirement:** Users must be able to trace from any claim back to source papers with minimal hops and maximum clarity.

**Navigation path:** `Relationship → Evidence → TextSpan → Paper` (all first-class entities, all indexed)

## Architecture Parallel

This plan creates `examples/medlit_schema/` as a **definitions-only** package that mirrors `kgschema/` structure, but deepened for medical literature. The parallel is:

```
kgschema/                    → examples/medlit_schema/     (definitions only)
kgraph/                      → examples/medlit/            (implementations)
```

**Key principle:** `medlit_schema/` inherits the sophistication of `med-lit-schema/` (rich bibliographic model, provenance tracking, ontology integration) while maintaining architectural isolation (no functional code).

**Migration strategy:** Extract schema definitions from both `examples/medlit/` and `med-lit-schema/` into `examples/medlit_schema/`, leaving implementation code in `examples/medlit/`.

---

## Context: Why This Migration?

### Migrating from current examples/medlit implementation

The `med-lit-schema` directory contains a mature, production-quality medical literature schema that's significantly more advanced than the current `examples/medlit` implementation. This migration would bring substantial improvements in schema quality, entity resolution, and provenance tracking.

**Current State:**
- `examples/medlit/` has basic entity/relationship models built on kgschema ABCs
- `med-lit-schema/` (in `/home/wware/kgraph/med-lit-schema`) has comprehensive medical domain models with:
  - Rich entity types (Disease, Gene, Drug, Protein, Mutation, Symptom, Biomarker, Pathway, Procedure, etc.)
  - Comprehensive relationship types with ~40+ predicates
  - Full provenance tracking with EvidenceItem and study type weighting
  - Entity resolution with canonical IDs (UMLS, HGNC, RxNorm, UniProt)
  - InMemoryEntityCollection for entity registry
  - Hypothesis tracking with ontology references (IAO, OBI, STATO, ECO, SEPIO)
  - Storage backends (SQLite, PostgreSQL) with domain/persistence model separation
  - Ingestion pipelines for PMC papers

**Architecture Decision: Keep med-lit-schema as External Dependency**

It makes sense to keep `med-lit-schema` as a separate package that `examples/medlit` imports from, rather than merging it:

**Benefits:**
1. **Separation of Concerns**: med-lit-schema is domain-specific, kgraph is framework-level
2. **Reusability**: Other projects can use med-lit-schema independently
3. **Versioning**: Can version med-lit-schema separately from kgraph framework
4. **Testing**: Can test medical schema independently of framework
5. **Maintenance**: Clear ownership boundaries (medical domain vs. graph framework)

**Migration Tasks:**

1. **Update pyproject.toml dependencies**
   ```toml
   [project]
   dependencies = [
       "med-lit-schema @ file:///home/wware/kgraph/med-lit-schema",
       # or when published:
       # "med-lit-schema>=0.1.0",
   ]
   ```

2. **Update examples/medlit/entities.py**
   - Import from med-lit-schema instead of defining locally:
     ```python
     from med_lit_schema.entity import (
         Disease, Gene, Drug, Protein, Mutation,
         Symptom, Biomarker, Pathway, Procedure,
         InMemoryEntityCollection,
     )
     ```
   - Remove local entity class definitions
   - Keep any kgraph-specific adapters if needed

3. **Update examples/medlit/relationships.py**
   - Import from med-lit-schema:
     ```python
     from med_lit_schema.relationship import (
         Treats, Causes, IncreasesRisk, DecreasesRisk,
         Inhibits, Activates, AssociatedWith, etc.
     )
     from med_lit_schema.base import PredicateType
     ```
   - Remove local relationship class definitions

4. **Update examples/medlit/domain.py (MedLitDomainSchema)**
   - Adapt to reference med-lit-schema entity/relationship types
   - Update predicate_constraints to use med-lit-schema's PredicateType enum
   - Leverage med-lit-schema's evidence weighting system

5. **Update examples/medlit/pipeline/relationships.py**
   - Use med-lit-schema's EvidenceItem for rich provenance
   - Adopt study_type weighting from med-lit-schema
   - Update to use comprehensive PredicateType vocabulary

6. **Update examples/medlit/pipeline/mentions.py and resolve.py**
   - Integrate with InMemoryEntityCollection for entity resolution
   - Use canonical ID systems (UMLS, HGNC, RxNorm, UniProt)

7. **Test Integration**
   - Verify all tests pass with med-lit-schema imports
   - Add tests for canonical entity resolution
   - Validate evidence provenance tracking

8. **Update Documentation**
   - Update examples/medlit/README.md to reference med-lit-schema
   - Document the dependency relationship
   - Add examples showing how to use med-lit-schema features

**Benefits of Migration:**

- ✅ **Richer Schema**: 40+ relationship types vs current basic set
- ✅ **Better Provenance**: EvidenceItem with study types, confidence weighting
- ✅ **Entity Resolution**: Canonical IDs from medical ontologies (UMLS, HGNC, etc.)
- ✅ **Production-Ready**: Already has storage backends, ingestion pipelines, API server
- ✅ **Scientific Rigor**: Hypothesis tracking, ontology references (IAO, OBI, STATO)
- ✅ **Domain/Persistence Separation**: Clean mapper pattern between domain and storage

**Risks & Mitigation:**

- **Risk**: Breaking changes to existing examples/medlit code
  - **Mitigation**: Create adapters/facades if needed for backward compatibility
- **Risk**: Dependency on external package
  - **Mitigation**: Pin version, or vendor the code if needed
- **Risk**: Med-lit-schema may have different design assumptions than kgraph
  - **Mitigation**: Use adapters to bridge differences; contribute improvements back to med-lit-schema

**Timeline Estimate:** 1-2 days for basic migration, plus 2-3 days for full integration testing and documentation.

---

## 1) Lock the extension boundary

**Do:** Decide and document: *"kgschema/ stays domain-agnostic; medlit_schema/ adds domain types + rules only."*

**Files:** `examples/medlit_schema/README.md` — add a short "Extension boundary" section.

**Done when:** There's a 10–15 line statement explaining:
- What never goes into `kgschema/` (no med-specific fields, no ingestion logic)
- How `medlit_schema/` mirrors `kgschema/` structure
- The parallel relationship: `kgschema→kgraph` mirrors `medlit_schema→examples/medlit`

---

## 2) Create medlit_schema package skeleton mirroring kgschema structure

**Do:** Create `examples/medlit_schema/` with internal structure matching `kgschema/`:

**Files:**

```
examples/medlit_schema/__init__.py
examples/medlit_schema/domain.py           # MedlitDomain(DomainSchema)
examples/medlit_schema/entity.py           # Disease, Gene, Drug, Paper, Author, etc.
examples/medlit_schema/relationship.py     # Treats, Causes, AuthoredBy, etc.
examples/medlit_schema/document.py         # PaperDocument
examples/medlit_schema/storage.py          # Storage interfaces if needed
examples/medlit_schema/base.py             # EvidenceItem, PaperMetadata, etc.
```

**Inheritance from med-lit-schema:**
- Rich bibliographic model: Paper with authors, journal, publication_date, MeSH terms
- Provenance tracking: ExtractionProvenance, ModelInfo, PromptInfo
- Ontology integration: IAO, OBI, STATO, ECO, SEPIO references
- Scientific entities: Hypothesis, StudyDesign, StatisticalMethod

**Done when:** You can `from examples.medlit_schema.domain import MedlitDomain` without importing anything from `kgraph/` (definitions only).

---

## 3) Implement `MedlitDomain(DomainSchema)` with initial entity types

**Do:** Implement a domain class extending `DomainSchema` ABC from `kgschema/`.

**Files:** `examples/medlit_schema/domain.py`

**Initial entity types (phase-1, from med-lit-schema):**

| Category | Types |
|---|---|
| **Biomedical Core** | `Disease`, `Gene`, `Drug`, `Protein`, `Mutation`, `Symptom`, `Biomarker`, `Pathway` |
| **Paper/Bibliographic** | `Paper`, `Author`, `ClinicalTrial` |
| **Provenance** | `TextSpan` (for anchoring evidence to paper sections) |
| **Scientific Method** | `Hypothesis`, `StudyDesign`, `StatisticalMethod`, `EvidenceLine` (optional, advanced) |

**Initial relationship types:**
- Medical: `TREATS`, `CAUSES`, `PREVENTS`, `INCREASES_RISK`, `SIDE_EFFECT`
- Biological: `ENCODES`, `BINDS_TO`, `INHIBITS`, `UPREGULATES`, `DOWNREGULATES`
- Bibliographic: `AUTHORED_BY`, `CITES`, `STUDIED_IN`
- Scientific: `PREDICTS`, `TESTED_BY`, `SUPPORTS`, `REFUTES`

**Done when:** `MedlitDomain` registers these types and can be instantiated without errors.

---

## 4) Implement rich Paper model with bibliographic metadata

**Do:** Create comprehensive Paper entity with fields from `med-lit-schema/entity.py`:

**Paper fields:**
- **Core identification**: `paper_id` (PMC ID), `pmid`, `doi`
- **Content**: `title`, `abstract`
- **Bibliographic**: `authors: List[str]`, `publication_date`, `journal`
- **Study metadata**: Nested `PaperMetadata` with:
  - Study characteristics: `study_type`, `sample_size`, `study_population`, `primary_outcome`, `clinical_phase`
  - Indexing: `mesh_terms: List[str]`
- **Provenance**: `extraction_provenance: Optional[ExtractionProvenance]`

**TextSpan for evidence anchoring:**
- `paper_id`: Link to paper
- `section_type`: Enum ("abstract", "introduction", "methods", "results", "discussion", "conclusion")
- `paragraph_idx`: Integer index
- `sentence_idx`: Optional sentence index
- `text_span`: Optional actual text snippet
- `start_offset`, `end_offset`: Optional character offsets

**Files:**

```
examples/medlit_schema/entity.py           # Paper, Author, PaperMetadata
examples/medlit_schema/base.py             # TextSpan, ExtractionProvenance, ModelInfo
```

**Done when:** You can represent *"this claim from Paper PMC123, section 'results', paragraph 5"* and link Evidence to specific paper locations.

---

## 5) Implement Evidence as first-class entity with traceability

**CRITICAL REQUIREMENT:** Evidence must enable traceability - users must track back to source papers with minimal hops and maximum clarity.

**Design Decision (FINALIZED):** Evidence as **first-class entity** (not relationship metadata).

**Rationale:**
1. ✅ **Database indexing** - Efficient queries like "all evidence from Section 2" or "all MRI-based evidence"
2. ✅ **Canonical IDs** - Can be promoted immediately: `PMCID12345:Sec2-Para3-MRI-results`
3. ✅ **Graph navigation** - Traceability via edges: `Claim → SUPPORTED_BY → Evidence → ANCHORED_IN → TextSpan → PART_OF → Paper`
4. ✅ **Reusable** - Same evidence can support multiple claims without duplication
5. ✅ **Complex queries** - GraphQL/Cypher can filter by evidence properties efficiently

**Evidence entity fields (based on med-lit-schema EvidenceItem):**
- `entity_id`: Canonical ID format `{paper_id}:{section}:{paragraph}:{method}`
- `paper_id`: PMC ID of source paper
- `text_span_id`: Link to TextSpan entity (for exact location)
- `confidence`: Float 0.0-1.0
- `extraction_method`: Enum ("scispacy_ner", "llm", "table_parser", "pattern_match", "manual")
- `study_type`: Enum ("observational", "rct", "meta_analysis", "case_report", "review")
- `sample_size`: Optional int
- **Ontology references** (from med-lit-schema):
  - `eco_type`: ECO evidence type ID (e.g., "ECO:0007673" for RCT)
  - `obi_study_design`: OBI study design ID (e.g., "OBI:0000008" for RCT)
  - `stato_methods`: List of STATO statistical method IDs

**Promotion rule:** Evidence entities with valid canonical IDs are promoted immediately (usage count = 1), as they represent publicly shareable claims about paper content.

**Files:** `examples/medlit_schema/entity.py` (add Evidence entity)

**Done when:**
- Evidence entity can be instantiated with canonical ID
- Evidence links to both Paper and TextSpan entities
- Multi-step query possible: "Find all RCT evidence supporting drug-disease claims"

---

## 6) Implement medical relationships with mandatory evidence references

**Do:** Create relationship classes that require Evidence entity references.

**Pattern (from med-lit-schema):** All medical assertion relationships must include evidence.

**Example relationship classes:**

```python
class Treats(BaseRelationship):
    """Drug treats disease relationship."""
    subject_id: str  # Drug entity_id
    predicate: Literal["TREATS"]
    object_id: str   # Disease entity_id
    evidence_ids: List[str]  # REQUIRED: List of Evidence entity_ids

    # Optional quantitative data
    response_rate: Optional[float]
    measurements: List[Measurement] = []

    @field_validator('evidence_ids')
    def evidence_required(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Medical relationships must include evidence")
        return v
```

**Key relationship types to implement:**
- `Treats`: Drug → Disease (with response_rate, measurements)
- `Causes`: Gene/Protein → Disease (with risk_factor data)
- `Inhibits`: Drug → Protein (with binding affinity)
- `AuthoredBy`: Paper → Author (no evidence required - bibliographic fact)

**Files:** `examples/medlit_schema/relationship.py`

**Done when:**
- Medical relationships require non-empty `evidence_ids`
- Bibliographic relationships (AuthoredBy, Cites) work without evidence
- Validation prevents evidence-free medical claims

---

## 7) Implement biomedical entities with canonical ID system

**Do:** Create entity classes extending `BaseEntity` from `kgschema/`, adding medical-specific fields from `med-lit-schema/entity.py`.

**Base pattern (BaseMedicalEntity):**
```python
class BaseMedicalEntity(BaseEntity):
    """Base for all medical entities."""
    entity_id: str           # Canonical ID (UMLS, HGNC, RxNorm, etc.)
    entity_type: EntityType  # Enum
    name: str
    synonyms: List[str] = []
    abbreviations: List[str] = []
    embedding: Optional[List[float]] = None  # For semantic search
    source: Literal["umls", "mesh", "rxnorm", "hgnc", "uniprot", "extracted"]
```

**Key entity types:**

1. **Disease**: `umls_id`, `mesh_id`, `icd10_codes`, `category`
2. **Gene**: `symbol`, `hgnc_id`, `chromosome`, `entrez_id`
3. **Drug**: `rxnorm_id`, `brand_names`, `drug_class`, `mechanism`
4. **Protein**: `uniprot_id`, `gene_id`, `function`, `pathways`

**Entity lifecycle:**
- **Provisional**: `source="extracted"`, no ontology ID yet
- **Canonical**: `source="umls|hgnc|rxnorm"`, has ontology ID

**Files:** `examples/medlit_schema/entity.py`

**Done when:**
- Can represent both provisional and canonical entities
- Each entity type has appropriate ontology ID fields
- Embeddings supported for semantic similarity

---

## 8) Add domain-specific validation rules

**Do:** Implement validation that enforces medlit schema guarantees.

**Critical invariants:**
1. **Medical relationships require evidence**: All assertion relationships (Treats, Causes, etc.) must have non-empty `evidence_ids`
2. **Canonical entities have ontology IDs**: If `source != "extracted"`, then appropriate ontology ID field must be set (umls_id, hgnc_id, etc.)
3. **Evidence traceability**: Every Evidence entity must reference valid `paper_id` and `text_span_id`
4. **Evidence promotion**: Evidence entities with proper canonical ID format are immediately promotable

**Implementation approaches:**
- Option 1: Pydantic validators on each class (recommended, type-safe)
- Option 2: Domain-level validation in `MedlitDomain.validate()` method
- Option 3: Hybrid (class validators + domain-level checks)

**Files:**
- `examples/medlit_schema/entity.py` (add validators to entity classes)
- `examples/medlit_schema/relationship.py` (add validators to relationship classes)
- Optional: `examples/medlit_schema/validation.py` (domain-wide validation utilities)

**Done when:**
- Pydantic validation prevents invalid medical data
- Unit tests verify each invariant

---

## 9) Create unit tests for schema validation

**Do:** Write tests that verify Pydantic models and domain invariants without touching ingestion logic.

**Test categories:**

1. **Domain registration** (`tests/test_medlit_domain.py`):
   - MedlitDomain instantiates correctly
   - All entity types registered
   - All relationship types registered

2. **Entity validation** (`tests/test_medlit_entities.py`):
   - Disease with UMLS ID validates
   - Gene with HGNC ID validates
   - Provisional entities (no ontology ID) validate
   - Canonical entities without ontology ID fail

3. **Relationship validation** (`tests/test_medlit_relationships.py`):
   - Treats relationship with evidence validates
   - Treats relationship without evidence fails
   - Bibliographic relationships work without evidence

4. **Evidence traceability** (`tests/test_evidence_traceability.py`):
   - Evidence entity with paper_id and text_span_id validates
   - Evidence canonical ID format correct
   - Evidence without paper_id fails
   - Can navigate: Relationship → Evidence → TextSpan → Paper

5. **Paper model** (`tests/test_paper_model.py`):
   - Paper with full metadata validates
   - PaperMetadata with study_type validates
   - ExtractionProvenance serializes correctly

**Files:** `tests/test_medlit_*.py`

**Done when:**
- `uv run pytest tests/test_medlit_*.py` passes
- Error messages clearly explain validation failures
- ~15-20 tests total covering all invariants

---

## 10) Create golden example demonstrating two-pass pipeline

**Do:** Build a minimal but complete example showing the full pipeline with medical literature.

**Example scenario:** "Olaparib treats BRCA-mutated breast cancer" from a mini abstract.

**Files:**

```
examples/medlit_golden/
├── input/
│   └── PMC999_abstract.txt          # Minimal abstract text
├── expected/
│   ├── pass1_entities.jsonl         # Canonical entities: Drug, Disease, Gene, Paper
│   ├── pass1_evidence.jsonl         # Evidence entities with canonical IDs
│   └── pass2_relationships.jsonl    # Treats relationship with evidence_ids
├── README.md                         # How to run and verify
└── verify.sh                         # Diff script
```

**Example content (input):**

```
Title: Efficacy of Olaparib in BRCA-Mutated Breast Cancer
Abstract: Olaparib, a PARP inhibitor, showed significant efficacy in
BRCA1/2-mutated metastatic breast cancer patients (N=302, RCT).
Response rate: 59.9% (95% CI: 52-66%, p<0.001).
```

**Expected outputs:**

Pass 1 entities:
- Paper (PMC999)
- Drug (Olaparib, RxNorm:1187832)
- Disease (Breast Cancer, UMLS:C0006142)
- Gene (BRCA1, HGNC:1100)
- Evidence (PMC999:abstract:para1:rct)
- TextSpan (PMC999:abstract:para1)

Pass 2 relationships:
- Treats (Olaparib → Breast Cancer, evidence_ids=[PMC999:abstract:para1:rct])
- IncreasesRisk (BRCA1 → Breast Cancer)

**Done when:**
- Pipeline can process input and produce expected outputs
- Outputs match golden files (or diff is explained)
- Demonstrates Evidence traceability: Treats → Evidence → TextSpan → Paper

---

## What to not do yet

Prevents scope creep and TODO explosion.

**Schema scope:**
- ✅ DO: Core biomedical entities (Disease, Gene, Drug, Protein) from med-lit-schema
- ✅ DO: Paper with rich metadata (authors, journal, study_type, sample_size)
- ✅ DO: Evidence as first-class entity with traceability
- ❌ DON'T: Advanced scientific entities (Hypothesis, StudyDesign) until core is stable
- ❌ DON'T: Figure/Table entities until core document-text-evidence flow works
- ❌ DON'T: Citation network (Cites, References) until basic relationships work

**Implementation scope:**
- ✅ DO: Schema definitions only in `examples/medlit_schema/`
- ✅ DO: Pydantic validation and unit tests
- ❌ DON'T: Ingestion pipeline code (stays in `examples/medlit/`)
- ❌ DON'T: Storage implementations (will use kgraph framework)
- ❌ DON'T: Query APIs (will come later via kgserver)

**Migration strategy:**
- ✅ DO: Port schema commitments (entity types, relationship types, validation rules)
- ✅ DO: Preserve evidence-first design from med-lit-schema
- ❌ DON'T: Port implementation code (parsers, extractors, resolvers)
- ❌ DON'T: Port storage backends (will use kgraph/kgserver patterns)

---

## Where to start

Items **2–7** are the critical path to a working medlit_schema that proves the concept:

**Phase 1 (Definitions):**
1. Package skeleton (Item 2)
2. Domain registration (Item 3)
3. Paper + TextSpan models (Item 4)
4. Evidence entity (Item 5)

**Phase 2 (Core medical KG):**
5. Medical entities (Item 7)
6. Relationships with evidence (Item 6)
7. Validation rules (Item 8)

**Phase 3 (Verification):**
8. Unit tests (Item 9)
9. Golden example (Item 10)

**Flow:** `Paper` → `TextSpan` → `Evidence` → `Relationship` → `Validation`

This establishes the foundation: evidence-backed medical assertions with full traceability to source papers.
