# MedLit Schema Ontology Integration Guide

This document outlines how the MedLit schema integrates with standard biomedical ontologies to ensure data quality, interoperability, and semantic richness.

## Core Principle: Canonicalization

A central goal of the schema is to move from provisional, text-extracted entities to canonical entities linked to established ontology identifiers. This process is managed through the `source` field in `BaseMedicalEntity` and entity-specific identifier fields.

-   **Provisional Entities**: When an entity is first extracted from text, it is considered "provisional." Its `source` is set to `"extracted"`, and it may not have a canonical ID.
-   **Canonical Entities**: Once the entity is resolved to a specific ontology concept, its `source` is updated (e.g., to `"umls"`, `"hgnc"`), and the corresponding ID field is populated.

The schema includes a Pydantic validator that enforces this: a canonical entity *must* have an associated ontology ID.

## Key Ontologies Used

The schema leverages several key ontologies, each tailored to a specific type of entity.

### For Core Biomedical Entities

| Entity    | Primary Ontology | ID Field(s)                             | Description                                                                                             |
| :-------- | :--------------- | :-------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| `Disease` | UMLS             | `umls_id`, `mesh_id`, `icd10_codes`       | UMLS provides a comprehensive vocabulary of biomedical concepts. MeSH and ICD-10 are used for mapping. |
| `Gene`      | HGNC             | `hgnc_id`, `entrez_id`                    | The HUGO Gene Nomenclature Committee provides standardized names for human genes.                       |
| `Drug`      | RxNorm           | `rxnorm_id`                             | RxNorm provides normalized names for clinical drugs.                                                    |
| `Protein`   | UniProt          | `uniprot_id`                            | UniProt is the central resource for protein sequence and function.                                    |
| `Biomarker` | LOINC            | `loinc_code`                            | LOINC provides universal codes for lab tests and clinical observations.                               |

### For Scientific Method and Evidence

The schema uses a set of ontologies to model the scientific process itself, enabling fine-grained tracking of evidence and hypotheses.

| Entity              | Ontology | ID Field(s)                               | Description                                                                                                   |
| :------------------ | :------- | :---------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| `Hypothesis`        | IAO      | `iao_id`, `sepio_id`                      | The Information Artifact Ontology (IAO) is used to model hypotheses as information content entities.        |
| `StudyDesign`       | OBI      | `obi_id`, `stato_id`                      | The Ontology for Biomedical Investigations (OBI) provides terms for describing study designs.                 |
| `StatisticalMethod` | STATO    | `stato_id`                                | The Statistics Ontology (STATO) provides a framework for describing statistical methods.                      |
| `Evidence`          | ECO      | `eco_type`, `obi_study_design`            | The Evidence & Conclusion Ontology (ECO) provides terms for different types of scientific evidence.           |
| `EvidenceLine`      | SEPIO    | `sepio_type`, `eco_type`                  | The Scientific Evidence and Provenance Information Ontology (SEPIO) is used to model structured evidence lines. |

## How Integration Works in Practice

1.  **Entity Definition**: Each entity class in `entity.py` that corresponds to an ontology concept includes a field for the canonical ID (e.g., `Disease` has `umls_id`).
2.  **Validation**: Pydantic validators on `BaseMedicalEntity` ensure that if an entity is marked as canonical (e.g., `source="umls"`), the corresponding ID field (`umls_id`) is not empty.
3.  **Ingestion Process (External to this Schema)**:
    -   An ingestion pipeline extracts a mention of an entity from text (e.g., "breast cancer").
    -   It creates a provisional `Disease` entity with `source="extracted"`.
    -   An entity linking step resolves this mention to a canonical concept (e.g., UMLS:C0006142).
    -   A new, canonical `Disease` entity is created with `source="umls"` and `umls_id="C0006142"`.
4.  **Promotion**: The `kgraph` promotion process uses these canonical IDs to merge duplicate entities and build a clean knowledge graph.

By embedding these ontology references directly into the schema, we ensure that the knowledge graph is not just a collection of strings, but a semantically grounded network of concepts.
