# MedLit Golden Example

This directory provides a "golden" example of a two-pass ingestion pipeline using the MedLit schema. It demonstrates how a small piece of text is processed to extract canonical entities, evidence, and relationships.

## Scenario

The input is a mini-abstract about the drug Olaparib and its use in treating BRCA-mutated breast cancer.

-   **Input**: `input/PMC999_abstract.txt`

## Expected Output

The ingestion process is expected to run in two passes:

1.  **Pass 1: Entity and Evidence Extraction**
    -   Extracts canonical entities for the drug, disease, and genes mentioned.
    -   Extracts a `Paper` entity representing the abstract itself.
    -   Extracts an `Evidence` entity that captures the context of the claim (e.g., that this is from an RCT with a specific sample size).
    -   **Output**: `expected/pass1_entities.jsonl` and `expected/pass1_evidence.jsonl`

2.  **Pass 2: Relationship Extraction**
    -   Using the entities and evidence from Pass 1, extracts relationships.
    -   Extracts a `TREATS` relationship between Olaparib and Breast Cancer.
    -   Extracts `INCREASES_RISK` relationships for the BRCA genes.
    -   Crucially, each relationship is linked to the `Evidence` entity from Pass 1.
    -   **Output**: `expected/pass2_relationships.jsonl`

## Verification

To verify that the pipeline is working correctly, you can run the `verify.sh` script. This script will (conceptually) run the ingestion pipeline on the input and diff the output against the expected files.

```bash
./verify.sh
```

This provides a concrete, testable example of the MedLit schema in action, demonstrating the core principles of evidence-based claims and traceability.