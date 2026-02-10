# Snapshot Semantics (v1 Draft)

## Purpose

This document defines the lifecycle boundaries for building a knowledge
graph snapshot. It clarifies when different identifiers become frozen
and what invariants a snapshot guarantees.

------------------------------------------------------------------------

## Identifier Categories

### 1. Intrinsic (Span/Addressing) Identifiers

These are frozen immediately during parsing and never change within a
snapshot:

-   `document_id`
-   Section / paragraph / sentence indices
-   Character offsets (`start_offset`, `end_offset`)
-   Mention identity = (`document_id`, offsets)

These represent source text location, not entity identity.

------------------------------------------------------------------------

### 2. Authority Canonical Identifiers

These are assigned during the **Promotion Phase**:

-   UMLS, MeSH, HGNC, RxNorm, DBPedia, etc.
-   Any internally minted stable canonical entity IDs

These are frozen **after the Promotion Phase** and do not change within
the snapshot.

------------------------------------------------------------------------

## Snapshot Build Lifecycle

A snapshot is produced by the following ordered phases:

1.  **Pass 1 -- Extraction & Resolution**
    -   Extract mentions with intrinsic identifiers.
    -   Resolve mentions to provisional or existing entities.
    -   Prepare candidates for authority canonicalization.
2.  **Promotion Phase -- Canonicalization Boundary**
    -   Assign authority canonical IDs.
    -   Optional: mint internal canonical entity IDs.
    -   After this phase, canonical IDs are locked for this snapshot.
3.  **Pass 2 -- Relationship Extraction**
    -   Extract relationships referencing stable entity IDs.
    -   No further promotion occurs.
4.  **Validation**
    -   Referential integrity checks.
    -   Schema validation.
5.  **Export**
    -   Bundle written to disk.
    -   Snapshot state is frozen.

------------------------------------------------------------------------

## Invariants

Within a single snapshot:

-   Intrinsic span identifiers never change.
-   Authority canonical IDs never change after promotion.
-   No entities are promoted after the Promotion Phase.
-   Relationship extraction operates only on stabilized entities.

------------------------------------------------------------------------

## Notes

-   A new snapshot may differ from a previous one if source text,
    authority data, or models change.
-   Canonical ID resolution is considered stable within a snapshot even
    if future snapshots refine or improve assignments.

------------------------------------------------------------------------

*End of Snapshot Semantics v1.*
