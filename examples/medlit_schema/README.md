# MedLit Schema

**Version**: 1.0.0

This directory contains the domain-specific schema for representing knowledge graphs of medical literature, serving as a `definitions-only` package. It extends the core `kgschema` with rich, domain-specific types for entities and relationships tailored to the biomedical field.

## Core Design Principles

1.  **Evidence as a First-Class Entity**: All medical claims (relationships) must be backed by evidence. The `Evidence` entity is a cornerstone of this schema, providing traceability from a claim back to its source in the literature.
2.  **Rich, Composable Models**: Entities and relationships are modeled as Pydantic classes, enabling validation, type safety, and easy composition.
3.  **Ontology Integration**: The schema is designed to integrate with major biomedical ontologies (UMLS, HGNC, RxNorm, UniProt, etc.) for entity normalization and canonicalization.
4.  **Traceability**: A primary goal is to ensure that any piece of information can be traced back to its source paper, section, and paragraph. The linkage `Relationship -> Evidence -> TextSpan -> Paper` is central to this.
5.  **Separation of Concerns**: This package contains only schema definitions (Pydantic models and domain registration). All implementation logic (ingestion, storage, querying) resides in other parts of the `kgraph` framework, such as `examples/medlit`.

## Schema Structure

The schema is organized into the following key files:

-   `entity.py`: Defines all entity types, from core biomedical concepts like `Disease`, `Gene`, and `Drug` to bibliographic entities like `Paper` and `Author`, and scientific concepts like `Hypothesis` and `Evidence`.
-   `relationship.py`: Defines the connections between entities. These range from medical relationships (`TREATS`, `CAUSES`) to biological (`ENCODES`) and bibliographic (`AUTHORED_BY`) ones. It also includes a factory function `create_relationship` for creating typed relationship instances.
-   `base.py`: Contains fundamental building blocks and enums like `PredicateType`, `EntityType`, and supporting models for provenance and evidence.
-   `domain.py`: The `MedlitDomain` class registers all the entity and relationship types and defines domain-specific validation rules and predicate constraints.
-   `document.py`: Defines document types, such as `PaperDocument`.

## Usage

This schema is not meant to be used directly for data storage or ingestion. Instead, it provides the definitions that the `kgraph` ingestion and storage systems use.

### Example: Creating a `Treats` Relationship

Here's how you might define a `Treats` relationship, complete with evidence:

```python
from examples.medlit_schema.relationship import create_relationship, Treats
from examples.medlit_schema.entity import Evidence

# First, define the evidence for the claim
evidence = Evidence(
    entity_id="PMC12345:results:5:llm",
    paper_id="PMC12345",
    text_span_id="PMC12345:results:5",
    confidence=0.95,
    extraction_method="llm",
    study_type="rct",
    sample_size=300,
)

# Create the relationship using the factory
treats_relationship = create_relationship(
    predicate="TREATS",
    subject_id="RxNorm:1187832",  # Olaparib
    object_id="UMLS:C0006142",   # Breast Cancer
    evidence_ids=[evidence.entity_id],
    response_rate=0.59,
    confidence=0.9,
)

assert isinstance(treats_relationship, Treats)
```

## Validation

The schema enforces a number of invariants through Pydantic validators:

-   **Medical relationships must have evidence**: Any attempt to create a medical relationship without `evidence_ids` will raise a `ValueError`.
-   **Canonical entities require ontology IDs**: Entities with a `source` other than `"extracted"` must have an appropriate ontology identifier (e.g., `umls_id`, `hgnc_id`).
-   **Evidence traceability**: `Evidence` entities must have a valid `paper_id` and `text_span_id`.

These validation rules are invoked automatically when creating instances of the schema's Pydantic models.