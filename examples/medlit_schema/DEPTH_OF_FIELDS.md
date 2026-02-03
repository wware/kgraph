# DEPTH_OF_FIELDS: Enrich examples/medlit_schema to Production-Ready Depth

## Executive Summary

**Goal**: Transform `examples/medlit_schema/` from a minimal teaching example (~400 LOC) into a production-ready, reusable schema package (~2,500+ LOC) with the full richness of `med-lit-schema` while maintaining the clean separation of definitions vs. implementation.

**Scope**: Schema definitions only (no functional infrastructure code like pipelines, storage backends, or servers).

**Architectural Principle**:
```
kgschema/          → examples/medlit_schema/    (definitions only)
kgraph/            → examples/medlit/           (implementations)
```

**Current State**:
- `entity.py`: 171 lines → **Target: ~1,100 lines**
- `relationship.py`: 153 lines → **Target: ~600 lines**
- `base.py`: 76 lines → **Target: ~300 lines**
- `domain.py`: 141 lines → **Target: ~170 lines**
- **Total**: ~540 lines → **Target: ~2,170 lines** of pure schema definitions

**Timeline**: ~30 hours (4 working days) of focused development

---

## Schema Version & Compatibility

Add to `examples/medlit_schema/__init__.py`:

```python
"""
Medical Literature Domain Schema for kgraph.

This package provides production-ready schema definitions for medical
literature knowledge graphs, with full provenance tracking, ontology
integration, and evidence-based relationships.

Schema version: 1.0.0
Compatible with: kgschema >=0.2.0
Ontologies: UMLS, HGNC, RxNorm, UniProt, ECO, OBI, STATO, SEPIO
"""

__version__ = "1.0.0"
__schema_version__ = "1.0.0"
```

---

## Phase 1: Enhance Base Models & Types (base.py)

**Current**: 76 lines with minimal enums and basic models
**Target**: ~300 lines with comprehensive type system
**Estimated Time**: 2-3 hours

### Task 1.1: Expand PredicateType Enum

**Source**: `med-lit-schema/base.py` lines 73-110

**Add missing predicates to create complete vocabulary**:

```python
class PredicateType(str, Enum):
    # Research/Bibliographic (existing)
    AUTHORED_BY = "authored_by"
    CITES = "cites"
    STUDIED_IN = "studied_in"

    # Medical - Causal (existing + new)
    CAUSES = "causes"
    PREVENTS = "prevents"  # NEW
    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"  # NEW

    # Medical - Therapeutic (existing + new)
    TREATS = "treats"
    MANAGES = "manages"  # NEW
    CONTRAINDICATED_FOR = "contraindicated_for"  # NEW
    SIDE_EFFECT = "side_effect"

    # Biological - Molecular (existing + new)
    BINDS_TO = "binds_to"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"  # NEW
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"
    ENCODES = "encodes"
    METABOLIZES = "metabolizes"  # NEW
    PARTICIPATES_IN = "participates_in"  # NEW

    # Diagnostic (new)
    DIAGNOSES = "diagnoses"  # NEW
    DIAGNOSED_BY = "diagnosed_by"  # NEW
    INDICATES = "indicates"  # NEW

    # Relationships (new)
    PRECEDES = "precedes"  # NEW
    CO_OCCURS_WITH = "co_occurs_with"  # NEW
    ASSOCIATED_WITH = "associated_with"  # NEW
    INTERACTS_WITH = "interacts_with"  # NEW

    # Location (new)
    LOCATED_IN = "located_in"  # NEW
    AFFECTS = "affects"  # NEW

    # Scientific Method (existing + new)
    PREDICTS = "predicts"
    REFUTES = "refutes"
    TESTED_BY = "tested_by"
    SUPPORTS = "supports"

    # Research Metadata (new)
    CITED_BY = "cited_by"  # NEW
    CONTRADICTS = "contradicts"  # NEW
    PART_OF = "part_of"  # NEW
    GENERATES = "generates"  # NEW
```

**Result**: Complete predicate vocabulary (~40 predicates)

### Task 1.2: Add EntityType Enum

**Source**: `med-lit-schema/base.py` lines 214-246

```python
class EntityType(str, Enum):
    """
    All possible entity types in the knowledge graph.

    This enum provides type safety for entity categorization and enables
    validation of entity-relationship compatibility.
    """

    # Core medical entities
    DISEASE = "disease"
    SYMPTOM = "symptom"
    DRUG = "drug"
    GENE = "gene"
    MUTATION = "mutation"
    PROTEIN = "protein"
    PATHWAY = "pathway"
    ANATOMICAL_STRUCTURE = "anatomical_structure"
    PROCEDURE = "procedure"
    TEST = "test"
    BIOMARKER = "biomarker"

    # Research metadata
    PAPER = "paper"
    AUTHOR = "author"
    INSTITUTION = "institution"
    CLINICAL_TRIAL = "clinical_trial"

    # Scientific method entities (ontology-based)
    HYPOTHESIS = "hypothesis"        # IAO:0000018
    STUDY_DESIGN = "study_design"    # OBI study designs
    STATISTICAL_METHOD = "statistical_method"  # STATO methods
    EVIDENCE_LINE = "evidence_line"  # SEPIO evidence structures
```

### Task 1.3: Add Supporting Models

**Source**: `med-lit-schema/base.py` lines 112-288

**Add these models with full documentation**:

1. **ClaimPredicate** - structured claim representation
   ```python
   class ClaimPredicate(BaseModel):
       """
       Describes the nature of a claim made in a paper.

       Examples:
           - "Olaparib significantly improved progression-free survival" (TREATS)
           - "BRCA1 mutations increase breast cancer risk by 5-fold" (INCREASES_RISK)
           - "Warfarin and aspirin interact synergistically" (INTERACTS_WITH)

       Attributes:
           predicate_type: The type of relationship asserted in the claim
           description: A natural language description of the predicate as it appears in the text
       """
       predicate_type: PredicateType
       description: str
   ```

2. **Provenance** - source tracking
   ```python
   class Provenance(BaseModel):
       """
       Information about the origin of a piece of data.

       Attributes:
           source_type: The type of source (e.g., 'paper', 'database', 'model_extraction')
           source_id: An identifier for the source (e.g., DOI, database record ID)
           source_version: The version of the source, if applicable
           notes: Additional notes about the provenance
       """
       source_type: str
       source_id: str
       source_version: Optional[str] = None
       notes: Optional[str] = None
   ```

3. **EvidenceType** - ontology-linked evidence classification
   ```python
   class EvidenceType(BaseModel):
       """
       The type of evidence supporting a relationship, linked to evidence ontologies.

       Examples:
           - RCT: ontology_id="ECO:0007673", ontology_label="randomized controlled trial evidence"
           - Observational: ontology_id="ECO:0000203", ontology_label="observational study evidence"
           - Case report: ontology_id="ECO:0006016", ontology_label="case study evidence"

       Attributes:
           ontology_id: Identifier from an evidence ontology (ECO, SEPIO)
           ontology_label: Human-readable label for the ontology term
           description: A fuller description of the evidence type
       """
       ontology_id: str
       ontology_label: str
       description: Optional[str] = None
   ```

4. **EntityReference** - lightweight entity pointers
   ```python
   class EntityReference(BaseModel):
       """
       Reference to an entity in the knowledge graph.

       Lightweight pointer to a canonical entity (Disease, Drug, Gene, etc.)
       with the name as it appeared in this specific paper.

       Attributes:
           id: Canonical entity ID
           name: Entity name as mentioned in paper
           type: Entity type (drug, disease, gene, protein, etc.)
       """
       id: str = Field(..., description="Canonical entity ID")
       name: str = Field(..., description="Entity name as mentioned in paper")
       type: EntityType = Field(..., description="Entity type")
   ```

5. **Polarity** enum - SUPPORTS/REFUTES/NEUTRAL
   ```python
   class Polarity(str, Enum):
       """Polarity of evidence relative to a claim."""
       SUPPORTS = "supports"
       REFUTES = "refutes"
       NEUTRAL = "neutral"
   ```

6. **Edge type hierarchy** - structured edge models
   ```python
   PaperId = uuid.UUID
   EdgeId = uuid.UUID

   class Edge(BaseModel):
       """Base edge in the knowledge graph."""
       id: EdgeId
       subject: EntityReference
       object: EntityReference
       provenance: Provenance

   class ExtractionEdge(Edge):
       """Edge from automated extraction."""
       extractor: ModelInfo
       confidence: float

   class ClaimEdge(Edge):
       """Edge representing a claim from a paper."""
       predicate: ClaimPredicate
       asserted_by: PaperId
       polarity: Polarity

   class EvidenceEdge(Edge):
       """Edge representing evidence for a claim."""
       evidence_type: EvidenceType
       strength: float
   ```

**Estimated**: +220 lines

---

## Phase 2: Enrich Entity Definitions (entity.py)

**Current**: 171 lines with basic entities
**Target**: ~1,100 lines with rich domain models
**Estimated Time**: 6-8 hours

### Task 2.1: Add Missing Entity Classes

**Add from `med-lit-schema`**:

1. **Procedure** (with medical procedure fields)
   ```python
   class Procedure(BaseMedicalEntity):
       """
       Represents medical tests, diagnostics, treatments.

       Attributes:
           type: Procedure category (diagnostic, therapeutic, preventive)
           invasiveness: Invasiveness level (non-invasive, minimally invasive, invasive)
       """
       type: Optional[str] = None
       invasiveness: Optional[str] = None

       def get_entity_type(self) -> str:
           return "procedure"
   ```

2. **Institution** (for author affiliations)
   ```python
   class Institution(BaseEntity):
       """
       Represents research institutions and affiliations.

       Attributes:
           name: Institution name
           country: Country location
           department: Department or division
       """
       name: str
       country: Optional[str] = None
       department: Optional[str] = None

       def get_entity_type(self) -> str:
           return "institution"
   ```

**Estimated**: +40 lines

### Task 2.2: Enrich Existing Medical Entities with Documentation

**For each entity (Disease, Gene, Drug, Protein, Mutation, Symptom, Biomarker, Pathway)**:

Add **comprehensive docstrings** following this pattern from `med-lit-schema/entity.py` lines 90-115:

```python
class Disease(BaseMedicalEntity):
    """
    Represents medical conditions, disorders, and syndromes.

    Uses UMLS as the primary identifier system with additional mappings to
    MeSH and ICD-10 for interoperability with clinical systems.

    Attributes:
        umls_id: UMLS Concept ID (e.g., "C0006142" for Breast Cancer)
        mesh_id: Medical Subject Heading ID for literature indexing
        icd10_codes: List of ICD-10 diagnostic codes
        category: Disease classification (genetic, infectious, autoimmune, etc.)

    Example:
        >>> breast_cancer = Disease(
        ...     entity_id="C0006142",
        ...     name="Breast Cancer",
        ...     synonyms=["Breast Carcinoma", "Mammary Cancer"],
        ...     umls_id="C0006142",
        ...     mesh_id="D001943",
        ...     icd10_codes=["C50.9"],
        ...     category="genetic"
        ... )
    """
    umls_id: Optional[str] = None
    mesh_id: Optional[str] = None
    icd10_codes: List[str] = []
    category: Optional[str] = None

    def get_entity_type(self) -> str:
        return "disease"
```

**Apply this pattern to all medical entities**:
- Disease (UMLS, MeSH, ICD-10)
- Gene (HGNC, Entrez, chromosomal location)
- Drug (RxNorm, brand names, mechanism)
- Protein (UniProt, function, pathways)
- Mutation (variant notation, consequence)
- Symptom (severity scales)
- Biomarker (LOINC codes, measurement types)
- Pathway (KEGG, Reactome IDs)

**Estimated**: +300 lines of documentation across 8 entity types

### Task 2.3: Enhance Paper & Bibliographic Entities

**Expand Paper class with comprehensive docstring**:

**Source**: `med-lit-schema/entity.py` lines 429-615

```python
class Paper(BaseEntity):
    """
    A research paper with extracted entities, relationships, and full provenance.

    This is the COMPLETE representation of a paper in the knowledge graph, combining:

    1. Bibliographic metadata (authors, journal, identifiers)
    2. Text content (title, abstract)
    3. Study metadata (study type, sample size, etc.)
    4. Extraction provenance (how extraction was performed)

    Design philosophy:

    - Top-level fields are FREQUENTLY QUERIED (paper_id, title, authors, publication_date)
    - Nested objects group related data (paper_metadata for study info, extraction_provenance for pipeline info)

    Why certain fields are top-level:

    - `paper_id`: Primary key, referenced everywhere
    - `title`, `abstract`: Core content, always displayed
    - `authors`: Essential for citations, frequently filtered
    - `publication_date`: Frequently used for filtering by recency
    - `journal`: Frequently used for quality filtering

    Why other fields are nested:

    - `paper_metadata`: Study details, accessed together for evidence assessment
    - `extraction_provenance`: Technical details, only for debugging/reproducibility

    Attributes:
        paper_id: Unique identifier - PMC ID preferred, but can be DOI or PMID
        pmid: PubMed ID - different from PMC ID
        doi: Digital Object Identifier
        title: Full paper title
        abstract: Complete abstract text
        authors: List of author names in citation order
        publication_date: Publication date in ISO format (YYYY-MM-DD)
        journal: Journal name
        paper_metadata: Extended metadata including study type, sample size, MeSH terms
        extraction_provenance: Complete provenance of how extraction was performed

    Example:
        >>> paper = Paper(
        ...     paper_id="PMC8437152",
        ...     pmid="34567890",
        ...     doi="10.1234/nejm.2023.001",
        ...     title="Efficacy of Olaparib in BRCA-Mutated Breast Cancer",
        ...     abstract="Background: PARP inhibitors have shown promise...",
        ...     authors=["Smith J", "Johnson A", "Williams K"],
        ...     publication_date=datetime(2023, 6, 15),
        ...     journal="New England Journal of Medicine",
        ...     paper_metadata=PaperMetadata(
        ...         study_type="rct",
        ...         sample_size=302,
        ...         mesh_terms=["Breast Neoplasms", "PARP Inhibitors"]
        ...     )
        ... )
    """
    paper_id: str  # PMC ID
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = []
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    paper_metadata: PaperMetadata = PaperMetadata()
    extraction_provenance: Optional[ExtractionProvenance] = None

    def get_entity_type(self) -> str:
        return "paper"
```

**Enhance PaperMetadata with query-focused documentation**:

**Source**: `med-lit-schema/entity.py` lines 391-422

```python
class PaperMetadata(BaseModel):
    """
    Extended metadata about the research paper.

    Combines study characteristics (for evidence quality assessment) with
    bibliographic information (for citations and filtering).

    This is MORE than just storage - these fields enable critical queries:
    - "Show me only RCT evidence for this drug-disease relationship"
    - "What's the sample size distribution for studies on this topic?"
    - "Find papers from high-impact journals on this mutation"

    Attributes:
        study_type: Type of study (observational, rct, meta_analysis, case_report, review)
        sample_size: Study sample size - larger = more reliable
        study_population: Description of study population
        primary_outcome: Primary outcome measured
        clinical_phase: Clinical trial phase if applicable
        mesh_terms: Medical Subject Headings - NLM's controlled vocabulary for indexing

    Example:
        >>> metadata = PaperMetadata(
        ...     study_type="rct",
        ...     sample_size=302,
        ...     study_population="Women with BRCA1/2-mutated metastatic breast cancer",
        ...     primary_outcome="Progression-free survival",
        ...     clinical_phase="III",
        ...     mesh_terms=["Breast Neoplasms", "BRCA1 Protein", "PARP Inhibitors"]
        ... )
    """
    study_type: Optional[str] = None
    sample_size: Optional[int] = None
    study_population: Optional[str] = None
    primary_outcome: Optional[str] = None
    clinical_phase: Optional[str] = None
    mesh_terms: List[str] = []
```

**Enhance Author class**:

**Source**: `med-lit-schema/entity.py` lines 683-702

```python
class Author(BaseEntity):
    """
    Represents a researcher or author of scientific publications.

    Attributes:
        orcid: ORCID identifier (unique researcher ID)
        name: Full name of the researcher
        affiliations: List of institutional affiliations
        h_index: Citation metric indicating research impact

    Example:
        >>> author = Author(
        ...     entity_id="0000-0001-2345-6789",
        ...     orcid="0000-0001-2345-6789",
        ...     name="Jane Smith",
        ...     affiliations=["Harvard Medical School", "Massachusetts General Hospital"],
        ...     h_index=45
        ... )
    """
    orcid: Optional[str] = None
    name: Optional[str] = None
    affiliations: List[str] = []
    h_index: Optional[int] = None

    def get_entity_type(self) -> str:
        return "author"
```

**Enhance ClinicalTrial class**:

**Source**: `med-lit-schema/entity.py` lines 712-742

```python
class ClinicalTrial(BaseEntity):
    """
    Represents a clinical trial registered on ClinicalTrials.gov.

    Attributes:
        nct_id: ClinicalTrials.gov identifier (e.g., "NCT01234567")
        title: Official trial title
        phase: Trial phase (I, II, III, IV)
        status: Current status (recruiting, completed, terminated, etc.)
        intervention: Description of treatment being tested

    Example:
        >>> trial = ClinicalTrial(
        ...     entity_id="NCT01234567",
        ...     nct_id="NCT01234567",
        ...     title="Study of Drug X in Patients with Disease Y",
        ...     phase="III",
        ...     status="completed",
        ...     intervention="Drug X 100mg daily"
        ... )
    """
    nct_id: Optional[str] = None
    title: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    intervention: Optional[str] = None

    def get_entity_type(self) -> str:
        return "clinical_trial"
```

**Estimated**: +200 lines

### Task 2.4: Add Scientific Method Entities (Full Implementations)

**Add complete classes with all fields and ontology integration**:

**Source**: `med-lit-schema/entity.py` lines 749-902

1. **Hypothesis** - hypothesis tracking with ontology references
   ```python
   class Hypothesis(BaseEntity):
       """
       Represents a scientific hypothesis tracked across the literature.

       Uses IAO (Information Artifact Ontology) for standardized representation
       of hypotheses as information content entities. Enables tracking of
       hypothesis evolution: from proposal through testing to acceptance/refutation.

       Attributes:
           iao_id: IAO identifier (typically IAO:0000018 for hypothesis)
           sepio_id: SEPIO identifier for assertions (SEPIO:0000001)
           proposed_by: Paper ID where hypothesis was first proposed
           proposed_date: Date when hypothesis was first proposed
           status: Current status (proposed, supported, controversial, refuted)
           description: Natural language description of the hypothesis
           predicts: List of entity IDs that this hypothesis predicts outcomes for

       Example:
           >>> hypothesis = Hypothesis(
           ...     entity_id="HYPOTHESIS:amyloid_cascade_alzheimers",
           ...     name="Amyloid Cascade Hypothesis",
           ...     iao_id="IAO:0000018",
           ...     sepio_id="SEPIO:0000001",
           ...     proposed_by="PMC123456",
           ...     proposed_date="1992",
           ...     status="controversial",
           ...     description="Beta-amyloid accumulation drives Alzheimer's disease pathology",
           ...     predicts=["C0002395"]  # Alzheimer's disease
           ... )
       """
       iao_id: Optional[str] = None
       sepio_id: Optional[str] = None
       proposed_by: Optional[str] = None
       proposed_date: Optional[str] = None
       status: Optional[str] = None
       description: Optional[str] = None
       predicts: List[str] = []

       def get_entity_type(self) -> str:
           return "hypothesis"
   ```

2. **StudyDesign** - study design with OBI/STATO ontologies
   ```python
   class StudyDesign(BaseEntity):
       """
       Represents a study design or experimental protocol.

       Uses OBI (Ontology for Biomedical Investigations) to standardize
       study design classifications. Enables filtering by evidence quality
       based on study design.

       Attributes:
           obi_id: OBI identifier for study design type
           stato_id: STATO identifier for study design (if applicable)
           design_type: Human-readable design type
           description: Description of the study design
           evidence_level: Quality level (1-5, where 1 is highest quality)

       Example:
           >>> rct = StudyDesign(
           ...     entity_id="OBI:0000008",
           ...     name="Randomized Controlled Trial",
           ...     obi_id="OBI:0000008",
           ...     stato_id="STATO:0000402",
           ...     design_type="interventional",
           ...     evidence_level=1
           ... )
       """
       obi_id: Optional[str] = None
       stato_id: Optional[str] = None
       design_type: Optional[str] = None
       description: Optional[str] = None
       evidence_level: Optional[int] = None

       def get_entity_type(self) -> str:
           return "study_design"
   ```

3. **StatisticalMethod** - statistical methods with STATO ontology
   ```python
   class StatisticalMethod(BaseEntity):
       """
       Represents a statistical method or test used in analysis.

       Uses STATO (Statistics Ontology) to standardize statistical method
       classifications. Enables tracking of analytical approaches across studies.

       Attributes:
           stato_id: STATO identifier for the statistical method
           method_type: Category of method (hypothesis_test, regression, etc.)
           description: Description of the method
           assumptions: Key assumptions of the method

       Example:
           >>> ttest = StatisticalMethod(
           ...     entity_id="STATO:0000288",
           ...     name="Student's t-test",
           ...     stato_id="STATO:0000288",
           ...     method_type="hypothesis_test",
           ...     description="Parametric test comparing means of two groups"
           ... )
       """
       stato_id: Optional[str] = None
       method_type: Optional[str] = None
       description: Optional[str] = None
       assumptions: List[str] = []

       def get_entity_type(self) -> str:
           return "statistical_method"
   ```

4. **EvidenceLine** - structured evidence with SEPIO framework
   ```python
   class EvidenceLine(BaseEntity):
       """
       Represents a line of evidence using SEPIO framework.

       Uses SEPIO (Scientific Evidence and Provenance Information Ontology)
       to represent structured evidence chains. Links evidence items to
       assertions they support or refute.

       Attributes:
           sepio_type: SEPIO evidence line type ID
           eco_type: ECO evidence type ID
           assertion_id: ID of the assertion this evidence supports
           supports: List of hypothesis IDs this evidence supports
           refutes: List of hypothesis IDs this evidence refutes
           evidence_items: List of paper IDs providing evidence
           strength: Evidence strength classification
           provenance: Provenance information

       Example:
           >>> evidence = EvidenceLine(
           ...     entity_id="EVIDENCE_LINE:olaparib_brca_001",
           ...     name="Clinical evidence for Olaparib in BRCA-mutated breast cancer",
           ...     sepio_type="SEPIO:0000084",
           ...     eco_type="ECO:0007673",
           ...     assertion_id="ASSERTION:olaparib_brca",
           ...     supports=["HYPOTHESIS:parp_inhibitor_synthetic_lethality"],
           ...     evidence_items=["PMC999888", "PMC888777"],
           ...     strength="strong"
           ... )
       """
       sepio_type: Optional[str] = None
       eco_type: Optional[str] = None
       assertion_id: Optional[str] = None
       supports: List[str] = []
       refutes: List[str] = []
       evidence_items: List[str] = []
       strength: Optional[str] = None
       provenance: Optional[str] = None

       def get_entity_type(self) -> str:
           return "evidence_line"
   ```

**Estimated**: +180 lines

### Task 2.5: Enhance Evidence Entity with Canonical ID Schema

**Enhance existing Evidence class with comprehensive documentation**:

```python
class Evidence(BaseEntity):
    """
    Evidence for a relationship, treated as a first-class entity.

    Evidence entities have immediate canonical ID promotion using format:
    {paper_id}:{section}:{paragraph}:{method}

    Example canonical ID: "PMC8437152:results:5:llm"

    This format enables:
    - Immediate promotion (no provisional state needed)
    - Efficient lookups by paper/section
    - Deduplication across extraction runs
    - Database indexing for queries like "all evidence from Section 2"

    Attributes:
        entity_id: Canonical ID in format {paper_id}:{section}:{paragraph}:{method}
        paper_id: PMC ID of source paper
        text_span_id: Reference to TextSpan entity (for exact location)
        confidence: Confidence score 0.0-1.0
        extraction_method: Method used (scispacy_ner, llm, table_parser, pattern_match, manual)
        study_type: Type of study (observational, rct, meta_analysis, case_report, review)
        sample_size: Number of subjects in the study
        eco_type: ECO evidence type ID (e.g., "ECO:0007673" for RCT)
        obi_study_design: OBI study design ID (e.g., "OBI:0000008" for RCT)
        stato_methods: List of STATO statistical method IDs used

    Schema Rules:
    - entity_id MUST follow canonical ID format
    - paper_id and text_span_id MUST be non-empty
    - Evidence entities are immediately promotable (no usage threshold)

    Example:
        >>> evidence = Evidence(
        ...     entity_id="PMC999888:results:3:llm",
        ...     paper_id="PMC999888",
        ...     text_span_id="PMC999888:results:3",
        ...     confidence=0.92,
        ...     extraction_method=ExtractionMethod.LLM,
        ...     study_type=StudyType.RCT,
        ...     sample_size=302,
        ...     eco_type="ECO:0007673",
        ...     obi_study_design="OBI:0000008",
        ...     stato_methods=["STATO:0000288"]
        ... )
    """
    paper_id: str
    text_span_id: str
    confidence: float
    extraction_method: ExtractionMethod
    study_type: StudyType
    sample_size: Optional[int] = None
    eco_type: Optional[str] = None
    obi_study_design: Optional[str] = None
    stato_methods: List[str] = []

    @field_validator("paper_id", "text_span_id")
    def ids_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("paper_id and text_span_id must not be empty")
        return v

    def get_entity_type(self) -> str:
        return "evidence"
```

**Add Measurement class** (enhance existing):

**Source**: `med-lit-schema/entity.py` lines 975-1028

```python
class Measurement(BaseModel):
    """
    Quantitative measurements associated with relationships.

    Stores numerical data with appropriate metadata for statistical
    analysis and evidence quality assessment.

    Attributes:
        value: The numerical value
        unit: Unit of measurement (if applicable)
        value_type: Type of measurement (effect_size, p_value, etc.)
        p_value: Statistical significance
        confidence_interval: 95% confidence interval
        study_population: Description of study population
        measurement_context: Additional context about the measurement

    Example:
        >>> measurement = Measurement(
        ...     value=0.59,
        ...     value_type="response_rate",
        ...     p_value=0.001,
        ...     confidence_interval=(0.52, 0.66),
        ...     study_population="BRCA-mutated breast cancer patients"
        ... )
    """
    value: float
    unit: Optional[str] = None
    value_type: Literal[
        "effect_size",
        "odds_ratio",
        "hazard_ratio",
        "p_value",
        "ci",
        "correlation",
        "response_rate",
        "risk_ratio",
        "penetrance",
        "sensitivity",
        "specificity",
    ]

    # Statistical context
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None

    # Study context
    study_population: Optional[str] = None
    measurement_context: Optional[str] = None
```

**Estimated**: +120 lines

### Task 2.6: Add Provenance Metadata Classes

**Add from `med-lit-schema/entity.py` lines 283-362**:

These support full reproducibility tracking for research:

1. **ExtractionPipelineInfo** - git commit, version tracking
   ```python
   class ExtractionPipelineInfo(BaseModel):
       """
       Information about the extraction pipeline version.

       Tracks the exact code version that performed entity/relationship extraction.
       Essential for reproducibility and debugging extraction quality issues.

       Attributes:
           name: Pipeline name (e.g., 'ollama_langchain_ingest')
           version: Semantic version of the pipeline
           git_commit: Full git commit hash
           git_commit_short: Short git commit hash (7 chars)
           git_branch: Git branch name
           git_dirty: Whether working directory had uncommitted changes
           repo_url: Repository URL
       """
       name: str
       version: str
       git_commit: str
       git_commit_short: str
       git_branch: str
       git_dirty: bool
       repo_url: str
   ```

2. **PromptInfo** - prompt version, checksum
   ```python
   class PromptInfo(BaseModel):
       """
       Information about the prompt used.

       Tracks prompt evolution. Critical for understanding extraction behavior changes.

       Attributes:
           version: Prompt version identifier
           template: Prompt template name
           checksum: SHA256 of actual prompt text for exact reproduction
       """
       version: str
       template: str
       checksum: Optional[str] = None
   ```

3. **ExecutionInfo** - timestamp, hostname, duration
   ```python
   class ExecutionInfo(BaseModel):
       """
       Information about when and where extraction was performed.

       Useful for debugging issues related to specific machines or time periods.

       Attributes:
           timestamp: ISO 8601 UTC timestamp
           hostname: Hostname of machine that ran extraction
           python_version: Python version
           duration_seconds: Extraction duration in seconds
       """
       timestamp: str
       hostname: str
       python_version: str
       duration_seconds: Optional[float] = None
   ```

4. **EntityResolutionInfo** - canonical matching stats
   ```python
   class EntityResolutionInfo(BaseModel):
       """
       Information about entity resolution process.

       Tracks how entities were matched to canonical IDs. Helps identify when
       entity deduplication is working poorly.

       Attributes:
           canonical_entities_matched: Number of entities matched to existing canonical IDs
           new_entities_created: Number of new canonical entities created
           similarity_threshold: Similarity threshold used for matching
           embedding_model: Embedding model used for similarity
       """
       canonical_entities_matched: int
       new_entities_created: int
       similarity_threshold: float
       embedding_model: str
   ```

5. **ExtractionProvenance** - complete audit trail (enhance existing)
   ```python
   class ExtractionProvenance(BaseModel):
       """
       Complete provenance metadata for an extraction.

       This is the complete audit trail of how extraction was performed.
       Enables:
       - Reproducing exact extraction with same code/models/prompts
       - Comparing outputs from different pipeline versions
       - Debugging quality issues
       - Tracking pipeline evolution over time
       - Meeting reproducibility requirements for research

       Example queries enabled by provenance:
       - "Find all papers extracted with prompt v1 so I can re-extract with v2"
       - "Which papers were extracted with uncommitted code changes?"
       - "Compare entity extraction quality between llama3.1:70b and claude-4"

       Attributes:
           extraction_pipeline: Pipeline version info
           models: Models used, keyed by role (e.g., 'llm', 'embeddings')
           prompt: Prompt version info
           execution: Execution environment info
           entity_resolution: Entity resolution details if applicable
       """
       extraction_pipeline: ExtractionPipelineInfo
       models: dict[str, ModelInfo]
       prompt: PromptInfo
       execution: ExecutionInfo
       entity_resolution: Optional[EntityResolutionInfo] = None
   ```

**Estimated**: +100 lines

**Note**: Skip `ExtractedEntity`, `EntityMention`, `ProcessedPaper` - these are pipeline artifacts that belong in `examples/medlit/`, not schema definitions.

---

## Phase 3: Enrich Relationship Definitions (relationship.py)

**Current**: 153 lines with basic relationships
**Target**: ~600 lines with rich provenance
**Estimated Time**: 4-5 hours

### Task 3.1: Add BaseMedicalRelationship

**Source**: `med-lit-schema/relationship.py` lines 35-102

**Add comprehensive base class for medical relationships**:

```python
class BaseMedicalRelationship(BaseRelationship):
    """
    Base class for all medical relationships with comprehensive provenance tracking.

    All medical relationships inherit from this class and include evidence-based
    provenance fields to support confidence scoring, contradiction detection,
    and temporal tracking of medical knowledge.

    Combines lightweight tracking (just paper IDs) with optional rich provenance
    (detailed Evidence objects) and quantitative measurements.

    Schema Rules:
    - Medical assertion relationships MUST have non-empty evidence_ids
    - Bibliographic relationships (AuthoredBy, Cites) do NOT require evidence

    Attributes:
        subject_id: Entity ID of the subject (source node)
        predicate: Relationship type
        object_id: Entity ID of the object (target node)
        evidence_ids: REQUIRED list of Evidence entity IDs (for medical assertions)
        confidence: Confidence score (0.0-1.0) based on evidence strength
        source_papers: List of PMC IDs supporting this relationship (lightweight)
        evidence_count: Number of papers providing supporting evidence
        contradicted_by: List of PMC IDs with contradicting findings
        first_reported: Date when this relationship was first observed
        last_updated: Date of most recent supporting evidence
        evidence: List of detailed EvidenceItem objects (optional, for rich provenance)
        measurements: List of quantitative measurements (optional)
        properties: Flexible dict for relationship-specific properties

    Example (lightweight):
        >>> relationship = Treats(
        ...     subject_id="RxNorm:1187832",
        ...     predicate="TREATS",
        ...     object_id="C0006142",
        ...     evidence_ids=["PMC123:results:5:llm", "PMC456:abstract:2:llm"],
        ...     source_papers=["PMC123", "PMC456"],
        ...     confidence=0.85,
        ...     evidence_count=2,
        ...     response_rate=0.59
        ... )

    Example (rich provenance):
        >>> relationship = Treats(
        ...     subject_id="RxNorm:1187832",
        ...     predicate="TREATS",
        ...     object_id="C0006142",
        ...     evidence_ids=["PMC123:results:5:rct"],
        ...     confidence=0.85,
        ...     evidence=[EvidenceItem(paper_id="PMC123", study_type="rct", sample_size=302)],
        ...     measurements=[Measurement(value=0.59, value_type="response_rate")],
        ...     response_rate=0.59
        ... )
    """
    # Required: References to Evidence entities
    evidence_ids: list[str] = Field(default_factory=list)

    # Core provenance (always present)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    # Lightweight tracking
    source_papers: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    contradicted_by: list[str] = Field(default_factory=list)
    first_reported: Optional[str] = None
    last_updated: Optional[str] = None

    # Rich provenance (optional)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    # Measurements (optional)
    measurements: list[Measurement] = Field(default_factory=list)

    # Relationship-specific properties (flexible)
    properties: dict = Field(default_factory=dict)

    @field_validator('evidence_ids')
    def evidence_required_for_medical_assertions(cls, v):
        """
        Medical assertion relationships must include evidence.

        This validator is overridden in non-medical relationship classes
        (like ResearchRelationship) that don't require evidence.
        """
        if not v or len(v) == 0:
            raise ValueError("Medical relationships must include evidence")
        return v
```

**Estimated**: +80 lines

### Task 3.2: Enrich All Medical Relationships

**For each medical relationship, add domain-specific fields**:

1. **Treats** (Drug → Disease)

   **Source**: `med-lit-schema/relationship.py` lines 135-167

   ```python
   class Treats(BaseMedicalRelationship):
       """
       Represents a therapeutic relationship between a drug and a disease.

       Direction: Drug → Disease

       Attributes:
           efficacy: Effectiveness measure or description
           response_rate: Percentage of patients responding (0.0-1.0)
           line_of_therapy: Treatment sequence (first-line, second-line, etc.)
           indication: Specific approved use or condition

       Example:
           >>> treats = Treats(
           ...     subject_id="RxNorm:1187832",  # Olaparib
           ...     object_id="C0006142",  # Breast Cancer
           ...     evidence_ids=["PMC999:results:5:rct", "PMC888:abstract:2:rct"],
           ...     efficacy="significant improvement in PFS",
           ...     response_rate=0.59,
           ...     line_of_therapy="second-line",
           ...     indication="BRCA-mutated breast cancer",
           ...     source_papers=["PMC999", "PMC888"],
           ...     confidence=0.85
           ... )
       """
       efficacy: Optional[str] = None
       response_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
       line_of_therapy: Optional[Literal["first-line", "second-line", "third-line", "maintenance", "salvage"]] = None
       indication: Optional[str] = None

       def get_edge_type(self) -> str:
           return "TREATS"
   ```

2. **Causes** (Gene/Mutation → Disease)

   **Source**: `med-lit-schema/relationship.py` lines 104-133

   ```python
   class Causes(BaseMedicalRelationship):
       """
       Represents a causal relationship between a disease and a symptom.

       Direction: Disease → Symptom (or Gene/Mutation → Disease)

       Attributes:
           frequency: How often the symptom occurs (always, often, sometimes, rarely)
           onset: When the symptom typically appears (early, late)
           severity: Typical severity of the symptom (mild, moderate, severe)

       Example:
           >>> causes = Causes(
           ...     subject_id="C0006142",  # Breast Cancer
           ...     object_id="C0030193",  # Pain
           ...     evidence_ids=["PMC123:results:3:llm"],
           ...     frequency="often",
           ...     onset="late",
           ...     severity="moderate",
           ...     source_papers=["PMC123"],
           ...     confidence=0.75
           ... )
       """
       frequency: Optional[Literal["always", "often", "sometimes", "rarely"]] = None
       onset: Optional[Literal["early", "late"]] = None
       severity: Optional[Literal["mild", "moderate", "severe"]] = None

       def get_edge_type(self) -> str:
           return "CAUSES"
   ```

3. **IncreasesRisk** (Gene/Mutation → Disease)

   **Source**: `med-lit-schema/relationship.py` lines 169-201

   ```python
   class IncreasesRisk(BaseMedicalRelationship):
       """
       Represents genetic risk factors for diseases.

       Direction: Gene/Mutation → Disease

       Attributes:
           risk_ratio: Numeric risk increase (e.g., 2.5 means 2.5x higher risk)
           penetrance: Percentage who develop condition (0.0-1.0)
           age_of_onset: Typical age when disease manifests
           population: Studied population or ethnic group

       Example:
           >>> risk = IncreasesRisk(
           ...     subject_id="HGNC:1100",  # BRCA1
           ...     object_id="C0006142",  # Breast Cancer
           ...     evidence_ids=["PMC123:results:7:llm", "PMC456:discussion:2:llm"],
           ...     risk_ratio=5.0,
           ...     penetrance=0.72,
           ...     age_of_onset="40-50 years",
           ...     population="Ashkenazi Jewish",
           ...     source_papers=["PMC123", "PMC456"],
           ...     confidence=0.92
           ... )
       """
       risk_ratio: Optional[float] = Field(None, gt=0.0)
       penetrance: Optional[float] = Field(None, ge=0.0, le=1.0)
       age_of_onset: Optional[str] = None
       population: Optional[str] = None

       def get_edge_type(self) -> str:
           return "INCREASES_RISK"
   ```

4. **SideEffect** (Drug → Symptom)

   **Source**: `med-lit-schema/relationship.py` lines 337-366

   ```python
   class SideEffect(BaseMedicalRelationship):
       """
       Represents adverse effects of medications.

       Direction: Drug → Symptom

       Attributes:
           frequency: How often it occurs (common, uncommon, rare)
           severity: Severity level (mild, moderate, severe)
           reversible: Whether the side effect resolves after stopping the drug

       Example:
           >>> side_effect = SideEffect(
           ...     subject_id="RxNorm:1187832",  # Olaparib
           ...     object_id="C0027497",  # Nausea
           ...     evidence_ids=["PMC999:results:8:llm"],
           ...     frequency="common",
           ...     severity="mild",
           ...     reversible=True,
           ...     source_papers=["PMC999"],
           ...     confidence=0.75
           ... )
       """
       frequency: Optional[Literal["common", "uncommon", "rare"]] = None
       severity: Optional[Literal["mild", "moderate", "severe"]] = None
       reversible: bool = True

       def get_edge_type(self) -> str:
           return "SIDE_EFFECT"
   ```

**Update Prevents relationship similarly**:
```python
class Prevents(BaseMedicalRelationship):
    """Drug prevents disease relationship."""

    efficacy: Optional[str] = None
    risk_reduction: Optional[float] = Field(None, ge=0.0, le=1.0)

    def get_edge_type(self) -> str:
        return "PREVENTS"
```

**Estimated**: +150 lines

### Task 3.3: Add Missing Medical Relationships

**Add from `med-lit-schema`**:

1. **AssociatedWith** (General associations)

   **Source**: `med-lit-schema/relationship.py` lines 203-239

   ```python
   class AssociatedWith(BaseMedicalRelationship):
       """
       Represents a general association between entities.

       This is used for relationships where causality is not established but
       statistical association exists.

       Valid directions:
           - Disease → Disease (comorbidities)
           - Gene → Disease
           - Biomarker → Disease

       Attributes:
           association_type: Nature of association (positive, negative, neutral)
           strength: Association strength (strong, moderate, weak)
           statistical_significance: p-value from statistical tests

       Example:
           >>> assoc = AssociatedWith(
           ...     subject_id="C0011849",  # Diabetes
           ...     object_id="C0020538",  # Hypertension
           ...     evidence_ids=["PMC111:results:4:llm"],
           ...     association_type="positive",
           ...     strength="strong",
           ...     statistical_significance=0.001,
           ...     source_papers=["PMC111"],
           ...     confidence=0.80
           ... )
       """
       association_type: Optional[Literal["positive", "negative", "neutral"]] = None
       strength: Optional[Literal["strong", "moderate", "weak"]] = None
       statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0)

       def get_edge_type(self) -> str:
           return "ASSOCIATED_WITH"
   ```

2. **InteractsWith** (Drug-drug interactions)

   **Source**: `med-lit-schema/relationship.py` lines 241-274

   ```python
   class InteractsWith(BaseMedicalRelationship):
       """
       Represents drug-drug interactions.

       Direction: Drug ↔ Drug (bidirectional)

       Attributes:
           interaction_type: Nature of interaction (synergistic, antagonistic, additive)
           severity: Clinical severity (major, moderate, minor)
           mechanism: Pharmacological mechanism of interaction
           clinical_significance: Description of clinical implications

       Example:
           >>> interaction = InteractsWith(
           ...     subject_id="RxNorm:123",  # Warfarin
           ...     object_id="RxNorm:456",  # Aspirin
           ...     evidence_ids=["PMC789:discussion:3:llm"],
           ...     interaction_type="synergistic",
           ...     severity="major",
           ...     mechanism="Additive anticoagulant effect",
           ...     clinical_significance="Increased bleeding risk",
           ...     source_papers=["PMC789"],
           ...     confidence=0.90
           ... )
       """
       interaction_type: Optional[Literal["synergistic", "antagonistic", "additive"]] = None
       severity: Optional[Literal["major", "moderate", "minor"]] = None
       mechanism: Optional[str] = None
       clinical_significance: Optional[str] = None

       def get_edge_type(self) -> str:
           return "INTERACTS_WITH"
   ```

3. **ContraindicatedFor** (Drug → Disease/Condition)

   **Source**: `med-lit-schema/relationship.py` lines 297-304

   ```python
   class ContraindicatedFor(BaseMedicalRelationship):
       """
       Drug -[CONTRAINDICATED_FOR]-> Disease/Condition

       Attributes:
           severity: Contraindication severity (absolute, relative)
           reason: Why contraindicated
       """
       severity: Optional[Literal["absolute", "relative"]] = None
       reason: Optional[str] = None

       def get_edge_type(self) -> str:
           return "CONTRAINDICATED_FOR"
   ```

4. **DiagnosedBy** (Disease → Procedure/Biomarker)

   **Source**: `med-lit-schema/relationship.py` lines 306-335

   ```python
   class DiagnosedBy(BaseMedicalRelationship):
       """
       Represents diagnostic tests or biomarkers used to diagnose a disease.

       Direction: Disease → Procedure/Biomarker

       Attributes:
           sensitivity: True positive rate (0.0-1.0)
           specificity: True negative rate (0.0-1.0)
           standard_of_care: Whether this is standard clinical practice

       Example:
           >>> diagnosis = DiagnosedBy(
           ...     subject_id="C0006142",  # Breast Cancer
           ...     object_id="LOINC:123",  # Mammography
           ...     evidence_ids=["PMC555:methods:2:llm"],
           ...     sensitivity=0.87,
           ...     specificity=0.91,
           ...     standard_of_care=True,
           ...     source_papers=["PMC555"],
           ...     confidence=0.88
           ... )
       """
       sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
       specificity: Optional[float] = Field(None, ge=0.0, le=1.0)
       standard_of_care: bool = False

       def get_edge_type(self) -> str:
           return "DIAGNOSED_BY"
   ```

5. **ParticipatesIn** (Gene/Protein → Pathway)

   **Source**: `med-lit-schema/relationship.py` lines 286-294

   ```python
   class ParticipatesIn(BaseMedicalRelationship):
       """
       Gene/Protein -[PARTICIPATES_IN]-> Pathway

       Attributes:
           role: Function in pathway
           regulatory_effect: Type of regulation (activates, inhibits, modulates)
       """
       role: Optional[str] = None
       regulatory_effect: Optional[Literal["activates", "inhibits", "modulates"]] = None

       def get_edge_type(self) -> str:
           return "PARTICIPATES_IN"
   ```

**Estimated**: +120 lines

### Task 3.4: Add Research Metadata Relationships

**Source**: `med-lit-schema/relationship.py` lines 373-460

**Add ResearchRelationship base class**:

```python
class ResearchRelationship(BaseRelationship):
    """
    Base class for research metadata relationships.

    These relationships connect papers, authors, and clinical trials.
    Unlike medical relationships, they don't require provenance tracking
    since they represent bibliographic metadata rather than medical claims.

    Attributes:
        subject_id: ID of the subject entity
        predicate: Relationship type
        object_id: ID of the object entity
        properties: Flexible dict for relationship-specific properties
    """
    properties: dict = Field(default_factory=dict)

    # Override parent validator - research relationships don't need evidence
    @field_validator('evidence_ids', mode='before')
    def evidence_not_required(cls, v):
        return v or []
```

**Add research relationship types**:

1. **Cites** (Paper → Paper)
   ```python
   class Cites(ResearchRelationship):
       """
       Represents a citation from one paper to another.

       Direction: Paper → Paper (citing → cited)

       Attributes:
           context: Section where citation appears
           sentiment: How the citation is used (supports, contradicts, mentions)
       """
       context: Optional[Literal["introduction", "methods", "results", "discussion"]] = None
       sentiment: Optional[Literal["supports", "contradicts", "mentions"]] = None

       def get_edge_type(self) -> str:
           return "CITES"
   ```

2. **StudiedIn** (Entity → Paper)
   ```python
   class StudiedIn(ResearchRelationship):
       """
       Links medical entities to papers that study them.

       Direction: Any medical entity → Paper

       Attributes:
           role: Importance in the paper (primary_focus, secondary_finding, mentioned)
           section: Where discussed (results, methods, discussion, introduction)
       """
       role: Optional[Literal["primary_focus", "secondary_finding", "mentioned"]] = None
       section: Optional[Literal["results", "methods", "discussion", "introduction"]] = None

       def get_edge_type(self) -> str:
           return "STUDIED_IN"
   ```

3. **AuthoredBy** (Paper → Author)
   ```python
   class AuthoredBy(ResearchRelationship):
       """
       Paper -[AUTHORED_BY]-> Author

       Attributes:
           position: Author position (first, last, corresponding, middle)
       """
       position: Optional[Literal["first", "last", "corresponding", "middle"]] = None

       def get_edge_type(self) -> str:
           return "AUTHORED_BY"
   ```

4. **PartOf** (Paper → ClinicalTrial)
   ```python
   class PartOf(ResearchRelationship):
       """
       Paper -[PART_OF]-> ClinicalTrial

       Attributes:
           publication_type: Type of publication (protocol, results, analysis)
       """
       publication_type: Optional[Literal["protocol", "results", "analysis"]] = None

       def get_edge_type(self) -> str:
           return "PART_OF"
   ```

**Estimated**: +70 lines

### Task 3.5: Add Hypothesis Relationships

**Source**: `med-lit-schema/relationship.py` lines 467-584

1. **Predicts** (Hypothesis → Entity)
   ```python
   class Predicts(BaseMedicalRelationship):
       """
       Represents a hypothesis predicting an observable outcome.

       Direction: Hypothesis → Entity (Disease, Outcome, etc.)

       Attributes:
           prediction_type: Nature of prediction (positive, negative, conditional)
           conditions: Conditions under which prediction holds
           testable: Whether the prediction is empirically testable
       """
       prediction_type: Optional[Literal["positive", "negative", "conditional"]] = None
       conditions: Optional[str] = None
       testable: bool = True

       def get_edge_type(self) -> str:
           return "PREDICTS"
   ```

2. **Refutes** (Evidence/Paper → Hypothesis)
   ```python
   class Refutes(BaseMedicalRelationship):
       """
       Represents evidence that refutes a hypothesis.

       Direction: Evidence/Paper → Hypothesis

       Attributes:
           refutation_strength: Strength of refutation (strong, moderate, weak)
           alternative_explanation: Alternative explanation for observations
           limitations: Limitations of the refuting evidence
       """
       refutation_strength: Optional[Literal["strong", "moderate", "weak"]] = None
       alternative_explanation: Optional[str] = None
       limitations: Optional[str] = None

       def get_edge_type(self) -> str:
           return "REFUTES"
   ```

3. **TestedBy** (Hypothesis → Paper/ClinicalTrial)
   ```python
   class TestedBy(BaseMedicalRelationship):
       """
       Represents a hypothesis being tested by a study or clinical trial.

       Direction: Hypothesis → Paper/ClinicalTrial

       Attributes:
           test_outcome: Result of the test (supported, refuted, inconclusive)
           methodology: Study methodology used
           study_design_id: OBI study design ID
       """
       test_outcome: Optional[Literal["supported", "refuted", "inconclusive"]] = None
       methodology: Optional[str] = None
       study_design_id: Optional[str] = None

       def get_edge_type(self) -> str:
           return "TESTED_BY"
   ```

4. **Generates** (Study → Evidence)
   ```python
   class Generates(BaseMedicalRelationship):
       """
       Represents a study generating evidence for analysis.

       Direction: ClinicalTrial/Paper → Evidence

       Attributes:
           evidence_type: Type of evidence generated (experimental, observational, etc.)
           eco_type: ECO evidence type ID
           quality_score: Quality assessment score
       """
       evidence_type: Optional[str] = None
       eco_type: Optional[str] = None
       quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

       def get_edge_type(self) -> str:
           return "GENERATES"
   ```

**Estimated**: +100 lines

### Task 3.6: Add Relationship Factory Function

**Source**: `med-lit-schema/relationship.py` lines 591-640

```python
def create_relationship(
    predicate: PredicateType,
    subject_id: str,
    object_id: str,
    **kwargs
) -> Union[BaseMedicalRelationship, ResearchRelationship]:
    """
    Factory function to create the appropriate relationship type.

    This allows you to use either the generic interface or the strongly-typed one.

    Args:
        predicate: The type of relationship
        subject_id: ID of the subject entity
        object_id: ID of the object entity
        **kwargs: Additional fields specific to the relationship type

    Returns:
        Appropriately typed relationship instance

    Example:
        >>> rel = create_relationship(
        ...     PredicateType.TREATS,
        ...     subject_id="RxNorm:1187832",
        ...     object_id="C0006142",
        ...     evidence_ids=["PMC999:results:5:rct"],
        ...     response_rate=0.59,
        ...     source_papers=["PMC999"]
        ... )
        >>> isinstance(rel, Treats)
        True
    """
    relationship_classes = {
        PredicateType.TREATS: Treats,
        PredicateType.CAUSES: Causes,
        PredicateType.PREVENTS: Prevents,
        PredicateType.INCREASES_RISK: IncreasesRisk,
        PredicateType.SIDE_EFFECT: SideEffect,
        PredicateType.ENCODES: Encodes,
        PredicateType.BINDS_TO: BindsTo,
        PredicateType.INHIBITS: Inhibits,
        PredicateType.UPREGULATES: Upregulates,
        PredicateType.DOWNREGULATES: Downregulates,
        PredicateType.ASSOCIATED_WITH: AssociatedWith,
        PredicateType.INTERACTS_WITH: InteractsWith,
        PredicateType.CONTRAINDICATED_FOR: ContraindicatedFor,
        PredicateType.DIAGNOSED_BY: DiagnosedBy,
        PredicateType.PARTICIPATES_IN: ParticipatesIn,
        PredicateType.CITES: Cites,
        PredicateType.STUDIED_IN: StudiedIn,
        PredicateType.AUTHORED_BY: AuthoredBy,
        PredicateType.PART_OF: PartOf,
        PredicateType.PREDICTS: Predicts,
        PredicateType.REFUTES: Refutes,
        PredicateType.TESTED_BY: TestedBy,
        PredicateType.GENERATES: Generates,
    }

    cls = relationship_classes.get(predicate, BaseMedicalRelationship)
    return cls(subject_id=subject_id, object_id=object_id, **kwargs)
```

**Estimated**: +50 lines

---

## Phase 4: Enhance Domain Registration (domain.py)

**Current**: 141 lines with basic domain
**Target**: ~170 lines with complete type registry
**Estimated Time**: 1 hour

### Task 4.1: Update Entity Type Registry

**Update** `entity_types` property to include all new entities:

```python
@property
def entity_types(self) -> dict[str, type[BaseEntity]]:
    return {
        # Core biomedical
        "disease": Disease,
        "gene": Gene,
        "drug": Drug,
        "protein": Protein,
        "mutation": Mutation,
        "symptom": Symptom,
        "biomarker": Biomarker,
        "pathway": Pathway,
        "procedure": Procedure,  # NEW

        # Paper/bibliographic
        "paper": Paper,
        "author": Author,
        "clinical_trial": ClinicalTrial,
        "institution": Institution,  # NEW

        # Scientific method
        "hypothesis": Hypothesis,
        "study_design": StudyDesign,
        "statistical_method": StatisticalMethod,
        "evidence_line": EvidenceLine,

        # Provenance
        "evidence": Evidence,
    }
```

**Estimated**: +5 lines

### Task 4.2: Update Relationship Type Registry

**Update** `relationship_types` property with all ~25 relationship classes:

```python
@property
def relationship_types(self) -> dict[str, type[BaseRelationship]]:
    return {
        # Medical assertions
        "TREATS": Treats,
        "CAUSES": Causes,
        "PREVENTS": Prevents,
        "INCREASES_RISK": IncreasesRisk,
        "SIDE_EFFECT": SideEffect,
        "ASSOCIATED_WITH": AssociatedWith,  # NEW
        "INTERACTS_WITH": InteractsWith,  # NEW
        "CONTRAINDICATED_FOR": ContraindicatedFor,  # NEW
        "DIAGNOSED_BY": DiagnosedBy,  # NEW

        # Biological
        "ENCODES": Encodes,
        "BINDS_TO": BindsTo,
        "INHIBITS": Inhibits,
        "UPREGULATES": Upregulates,
        "DOWNREGULATES": Downregulates,
        "PARTICIPATES_IN": ParticipatesIn,  # NEW

        # Bibliographic
        "AUTHORED_BY": AuthoredBy,
        "CITES": Cites,
        "STUDIED_IN": StudiedIn,
        "PART_OF": PartOf,  # NEW

        # Scientific method
        "PREDICTS": Predicts,
        "TESTED_BY": TestedBy,
        "SUPPORTS": Supports,
        "REFUTES": Refutes,
        "GENERATES": Generates,  # NEW
    }
```

**Estimated**: +10 lines

### Task 4.3: Expand Predicate Constraints

**Expand** the `predicate_constraints` dictionary to cover all predicates:

```python
@property
def predicate_constraints(self) -> dict[str, PredicateConstraint]:
    return {
        # Medical assertions
        "TREATS": PredicateConstraint(subject_types={"drug"}, object_types={"disease"}),
        "CAUSES": PredicateConstraint(subject_types={"gene", "mutation", "disease"}, object_types={"disease", "symptom"}),
        "PREVENTS": PredicateConstraint(subject_types={"drug"}, object_types={"disease"}),
        "INCREASES_RISK": PredicateConstraint(subject_types={"gene", "mutation"}, object_types={"disease"}),
        "SIDE_EFFECT": PredicateConstraint(subject_types={"drug"}, object_types={"symptom", "disease"}),
        "ASSOCIATED_WITH": PredicateConstraint(
            subject_types={"disease", "gene", "biomarker"},
            object_types={"disease", "gene", "biomarker"}
        ),
        "INTERACTS_WITH": PredicateConstraint(subject_types={"drug"}, object_types={"drug"}),
        "CONTRAINDICATED_FOR": PredicateConstraint(subject_types={"drug"}, object_types={"disease"}),
        "DIAGNOSED_BY": PredicateConstraint(subject_types={"disease"}, object_types={"procedure", "biomarker"}),

        # Biological
        "ENCODES": PredicateConstraint(subject_types={"gene"}, object_types={"protein"}),
        "BINDS_TO": PredicateConstraint(subject_types={"drug", "protein"}, object_types={"protein"}),
        "INHIBITS": PredicateConstraint(subject_types={"drug", "protein"}, object_types={"protein", "pathway"}),
        "UPREGULATES": PredicateConstraint(subject_types={"drug", "gene"}, object_types={"gene", "pathway"}),
        "DOWNREGULATES": PredicateConstraint(subject_types={"drug", "gene"}, object_types={"gene", "pathway"}),
        "PARTICIPATES_IN": PredicateConstraint(subject_types={"gene", "protein"}, object_types={"pathway"}),

        # Bibliographic
        "AUTHORED_BY": PredicateConstraint(subject_types={"paper"}, object_types={"author"}),
        "CITES": PredicateConstraint(subject_types={"paper"}, object_types={"paper"}),
        "STUDIED_IN": PredicateConstraint(
            subject_types={"disease", "drug", "gene", "protein"},
            object_types={"clinical_trial", "paper"}
        ),
        "PART_OF": PredicateConstraint(subject_types={"paper"}, object_types={"clinical_trial"}),

        # Scientific method
        "PREDICTS": PredicateConstraint(subject_types={"hypothesis"}, object_types={"disease", "outcome"}),
        "TESTED_BY": PredicateConstraint(subject_types={"hypothesis"}, object_types={"study_design", "paper"}),
        "SUPPORTS": PredicateConstraint(subject_types={"evidence"}, object_types={"hypothesis"}),
        "REFUTES": PredicateConstraint(subject_types={"evidence", "paper"}, object_types={"hypothesis"}),
        "GENERATES": PredicateConstraint(subject_types={"paper", "clinical_trial"}, object_types={"evidence"}),
    }
```

**Estimated**: +30 lines

### Task 4.4: Enhance Validation Methods

**Add comprehensive docstrings** to validation methods:

```python
def validate_entity(self, entity: BaseEntity) -> bool:
    """
    Validate medical entity against domain invariants.

    Schema Rules (enforced by Pydantic validators on entity classes):

    1. Canonical entities (source != "extracted") MUST have ontology ID
       - Disease: umls_id required
       - Gene: hgnc_id required
       - Drug: rxnorm_id required
       - Protein: uniprot_id required

    2. Evidence entities MUST have valid canonical ID format
       - Format: {paper_id}:{section}:{paragraph}:{method}
       - Example: "PMC8437152:results:5:llm"

    3. Paper entities MUST have at least one of: doi, pmid, paper_id

    Implementation note: This method is a passthrough since validation
    is enforced by Pydantic field validators on entity classes.
    See entity.py for @field_validator implementations.

    Args:
        entity: Entity to validate

    Returns:
        True if valid (validation errors raise exceptions from Pydantic)
    """
    return True

async def validate_relationship(self, relationship: BaseRelationship, entity_storage=None) -> bool:
    """
    Validate medical relationship against domain rules.

    Schema Rules (enforced by Pydantic validators on relationship classes):

    1. Medical assertion relationships MUST have non-empty evidence_ids
       - Applies to: Treats, Causes, IncreasesRisk, SideEffect, etc.
       - Bibliographic relationships (Cites, AuthoredBy) are exempt

    2. Subject and object entity types MUST match predicate constraints
       - See predicate_constraints property for type compatibility matrix
       - Example: TREATS requires subject=drug, object=disease

    3. Evidence IDs MUST reference valid Evidence entities
       - If entity_storage provided, can verify Evidence entities exist

    Implementation note: Type checking is enforced by Pydantic.
    Evidence existence checking requires entity_storage (implementation detail).

    Args:
        relationship: Relationship to validate
        entity_storage: Optional storage for evidence lookup

    Returns:
        True if valid (validation errors raise exceptions from Pydantic)
    """
    return True
```

**Estimated**: +20 lines

---

## Phase 5: Documentation & Examples

**Estimated Time**: 5-6 hours

### Task 5.1: Create Comprehensive README.md

**Add sections**:

1. **Schema Richness Overview** - production-ready features
2. **Ontology Integration** - all supported ontologies
3. **Provenance Tracking** - reproducibility features
4. **Evidence Model** - first-class entity design
5. **Usage Examples** - realistic medical scenarios
6. **Predicate Compatibility Matrix** - relationship constraints
7. **Comparison with med-lit-schema** - relationship and migration path

**Estimated**: +200 lines

### Task 5.2: Create ONTOLOGY_GUIDE.md

**New file documenting**:
- All ontology systems (UMLS, HGNC, RxNorm, UniProt, ECO, OBI, STATO, SEPIO)
- How to look up canonical IDs
- Validation rules for ontology IDs
- Examples of multi-ontology entity resolution
- Links to ontology browsers and documentation

**Estimated**: +200 lines

### Task 5.3: Create Comprehensive Code Examples

**Create** `examples/` directory with realistic examples:

1. **complete_paper_example.py** - Full paper with all metadata
2. **hypothesis_tracking_example.py** - Scientific method entities
3. **entity_resolution_example.py** - Canonical ID lookup patterns
4. **evidence_traceability_example.py** - Multi-hop evidence queries

**Estimated**: +330 lines total

---

## Phase 6: Testing Strategy

**Estimated Time**: 6-7 hours

### Task 6.1: Update Existing Tests

**Files to update** with new fields and validation:
- `tests/test_medlit_domain.py` - add tests for new entity types (+50 lines)
- `tests/test_medlit_entities.py` - add tests for new entity fields (+100 lines)
- `tests/test_medlit_relationships.py` - add tests for new relationship fields (+50 lines)

**Estimated**: +200 lines

### Task 6.2: Add New Test Files

**Create**:

1. **tests/test_scientific_method_entities.py** - Hypothesis, StudyDesign, etc. (+100 lines)
2. **tests/test_provenance_tracking.py** - ExtractionProvenance, ModelInfo, etc. (+80 lines)
3. **tests/test_evidence_canonical_id.py** - Evidence canonical ID format validation (+60 lines)
4. **tests/test_relationship_factory.py** - create_relationship function (+40 lines)
5. **tests/test_ontology_validation.py** - ontology ID validation rules (+120 lines)
6. **tests/test_measurement.py** - Measurement class with all value types (+50 lines)

**Estimated**: +450 lines

---

## Implementation Timeline

| Phase | Tasks | Estimated LOC | Estimated Time |
|-------|-------|--------------|----------------|
| 1. Base Models | 1.1-1.3 | +220 | 2-3 hours |
| 2. Entities | 2.1-2.6 | +940 | 6-8 hours |
| 3. Relationships | 3.1-3.6 | +570 | 4-5 hours |
| 4. Domain | 4.1-4.4 | +65 | 1 hour |
| 5. Documentation | 5.1-5.3 | +730 | 5-6 hours |
| 6. Testing | 6.1-6.2 | +650 | 6-7 hours |
| **Total** | | **~3,175 LOC** | **~30 hours** |

---

## Success Criteria

### ✅ Schema Completeness
- All 17+ entity types with full fields and comprehensive docstrings
- All 25+ relationship types with domain-specific fields
- Complete ontology integration (UMLS, HGNC, RxNorm, UniProt, ECO, OBI, STATO, SEPIO)
- Evidence canonical ID format documented as schema rule

### ✅ Provenance Richness
- BaseMedicalRelationship with dual-mode provenance (lightweight + rich)
- Evidence entity with text-level traceability
- Complete ExtractionProvenance with reproducibility metadata
- Measurement class with statistical context

### ✅ Documentation Quality
- Every entity/relationship has comprehensive docstring with:
  - Purpose and usage description
  - Ontology system explanation
  - All attributes documented
  - Realistic medical examples
- Ontology guide with lookup instructions
- Predicate compatibility matrix
- Migration guide from minimal schema

### ✅ Validation
- All tests pass (existing + new ~650 lines)
- No breaking changes to existing code
- Pydantic validators enforce all schema rules
- Clean separation: definitions only, no implementation code

### ✅ Architectural Integrity
- No functional code (storage, pipelines, utilities) in medlit_schema/
- Clear documentation of what belongs in medlit_schema/ vs medlit/
- Schema version metadata
- Ready for standalone publication

---

## Risk Mitigation

### Risk 1: Divergence from med-lit-schema
- **Mitigation**: Document intentional differences in README.md
- **Mitigation**: Maintain compatibility notes for migration

### Risk 2: Breaking changes to examples/medlit_golden
- **Mitigation**: Update golden example incrementally
- **Mitigation**: Provide migration script if needed

### Risk 3: Over-complication of "examples" package
- **Mitigation**: Keep implementation in examples/medlit/, only definitions in examples/medlit_schema/
- **Mitigation**: Regular review: "Does this belong here or in medlit/?"

### Risk 4: Maintenance burden of duplicated definitions
- **Mitigation**: Consider eventual consolidation strategy
- **Mitigation**: Med-lit-schema could depend on this schema in the future

---

## Future Considerations

**After completion, consider**:

1. **Publishing** `examples/medlit_schema` as standalone `kgraph-medlit-schema` package on PyPI
2. **Deprecating** `med-lit-schema` in favor of unified schema
3. **Adding** validation rules as executable code (currently docstrings only)
4. **Creating** schema versioning and migration utilities
5. **Building** interactive schema explorer/documentation site
6. **Extending** to other medical literature sources (ClinicalTrials.gov, FDA labels)

---

## Implementation Order

**Recommended sequence**:

### Week 1: Core Schema (Phases 1-2)
- Day 1: Phase 1 (Base Models) - foundation types
- Day 2-3: Phase 2 (Entities) - medical entities with docs

### Week 2: Relationships & Domain (Phases 3-4)
- Day 1: Phase 3.1-3.3 (Core medical relationships)
- Day 2: Phase 3.4-3.6 (Research + hypothesis relationships)
- Day 3: Phase 4 (Domain registration)

### Week 3: Documentation & Testing (Phases 5-6)
- Day 1-2: Phase 5 (Documentation + examples)
- Day 3-4: Phase 6 (Comprehensive testing)

### Week 4: Polish & Review
- Integration testing
- Documentation review
- Migration guide
- Release preparation

---

## Validation Checkpoints

**After each phase**, verify:

1. ✅ All imports work without errors
2. ✅ Pydantic models validate correctly
3. ✅ Examples run successfully
4. ✅ Tests pass
5. ✅ No functional code added (definitions only)
6. ✅ Documentation is clear and accurate

**Before final release**, verify:

1. ✅ All success criteria met
2. ✅ No breaking changes to existing code
3. ✅ README.md is comprehensive
4. ✅ ONTOLOGY_GUIDE.md is complete
5. ✅ Examples are realistic and helpful
6. ✅ Test coverage is adequate (~650 lines)
7. ✅ Schema version metadata is set
