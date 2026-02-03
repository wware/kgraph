# Notes on JATS-XML Parsing for Medlit Schema Ingestion

## Goal
To develop a robust parser that can transform JATS-XML scientific articles into structured data conforming to the `medlit_schema` in the `kgraph` framework. This involves extracting:
- `Paper` entities (bibliographic metadata)
- `TextSpan` entities (granular text locations)
- `BaseMedicalEntity` types (e.g., Disease, Gene, Drug, Protein, Mutation, Symptom, Biomarker, Pathway, Procedure, Author, ClinicalTrial, Institution, Hypothesis, StudyDesign, StatisticalMethod)
- `Evidence` entities (canonical records of observations)
- Relationships between entities, critically linked to `Evidence` entities.

## Key Challenges
1.  **Heterogeneous XML Structure**: JATS-XML can vary significantly between publishers and journals, requiring flexible parsing strategies.
2.  **Text Extraction Accuracy**: Accurately extracting text content, section titles, and character offsets is crucial for `TextSpan` creation.
3.  **Entity Identification & Normalization**: Identifying mentions of biomedical entities in text and linking them to canonical IDs (UMLS, HGNC, etc.) is complex.
4.  **Evidence Grounding**: Correctly associating asserted facts and relationships with their precise `TextSpan` and `Paper` of origin.
5.  **Relationship Inference**: Moving beyond simple co-occurrence to infer meaningful, typed relationships.

## Proposed Steps for Parser Development

### 1. XML Parsing & Document Structure
-   **Tooling**: Utilize `lxml` for efficient and robust XML parsing, providing XPath capabilities for navigating the JATS tree. Alternatively, `BeautifulSoup` can be used for simpler XML structures.
-   **Initial Document Object**: Create an internal representation of the JATS-XML document that facilitates navigation and extraction.

### 2. `Paper` Entity Extraction
-   **Metadata Mapping**: Map JATS elements such as `<front-matter>`, `<article-meta>`, `<title-group>`, `<contrib-group>`, `<pub-date>`, `<abstract>`, `<article-id>` (PMID, DOI, PMCID) to the fields of the `Paper` entity.
-   **Author Handling**: Extract author names and affiliations, potentially creating `Author` and `Institution` entities if detailed provenance is required.

### 3. `TextSpan` Entity Identification
-   **Section Iteration**: Traverse the JATS document (`<body>`, `<abstract>`, etc.) to identify logical sections.
-   **Offset Calculation**: For each section, maintain accurate character `start_offset` and `end_offset` for every text segment within it. This is critical for generating unique `TextSpan` `entity_id`s.
-   **`TextSpan` Creation**: Instantiate `TextSpan` entities, providing `paper_id`, `section` (e.g., "abstract", "introduction"), and precise character offsets. Optionally, store `text_content` for caching.

### 4. Entity Recognition & Canonicalization
-   **NLP Pipeline**: Integrate an NLP pipeline (e.g., spaCy with a biomedical model, or a custom NER model) to process the `text_content` from `TextSpan` entities.
-   **Candidate Generation**: Identify potential `BaseMedicalEntity` mentions within the text.
-   **Ontology Lookup**: For each mention, attempt to resolve it against biomedical ontologies (UMLS, HGNC, RxNorm, UniProt, MeSH) to obtain canonical IDs.
-   **Entity Object Creation**: Create instances of `Disease`, `Gene`, `Drug`, `Protein`, etc., populating their specific fields and ensuring `source` and `canonical_ids` are set appropriately. Handle provisional entities for those not found in ontologies, if applicable (though `Evidence` itself is always canonical).

### 5. `Evidence` Entity Creation
-   **Fact Extraction**: After entity recognition, analyze sentences or clauses within `TextSpan` entities to identify factual assertions. This might involve pattern matching, rule-based systems, or LLMs.
-   **`Evidence` Instantiation**: For each identified assertion, create an `Evidence` entity.
    -   Its `entity_id` should follow the format: `{paper_id}:{section}:{start_offset}-{end_offset}:{extraction_method}`.
    -   Link to the originating `paper_id` and `text_span_id`.
    -   Record `extraction_method` (e.g., "llm", "rule_based", "manual").
    -   Assign a `confidence` score.
    -   Populate `study_type`, `sample_size`, etc., from `PaperMetadata` or direct extraction.

### 6. Relationship Extraction & Grounding
-   **Relationship Inference**: Identify semantic relationships between `BaseMedicalEntity` objects found within or across `TextSpan` entities. This is the most complex step and could use:
    -   **Rule-based patterns**: For well-defined relationships.
    -   **Machine Learning models**: Trained for relation extraction.
    -   **LLM prompting**: For flexible and novel relationship discovery.
-   **`evidence_ids` Linkage**: Crucially, for every relationship created, ensure it includes one or more `evidence_ids` referencing the `Evidence` entities that support that assertion. This enforces the "relationship-only assertions" model.

### 7. Output Generation
-   **Serialization**: Convert all extracted `Paper`, `TextSpan`, `BaseMedicalEntity`, `Evidence`, and Relationship objects into JSONL format, suitable for ingestion into `kgraph`.
-   **`kgbundle` Integration**: Ensure the output matches the `kgbundle` specification for manifests, entities, and relationships.
