# MedLit Knowledge Graph — Ingestion Process

## Overview

Ingestion transforms JATS-XML papers into a structured knowledge graph. The process has
two distinct passes with very different computational profiles:

1. **LLM extraction pass** — slow, expensive, requires judgment. One API call per paper
   (or per section for long papers). Produces a per-paper JSON bundle where all entities
   have `source="extracted"` (provisional).
2. **Deduplication/promotion pass** — fast, cheap, mostly mechanical. Resolves entities
   to canonical ontology IDs, assigns canonical IDs, and merges cross-paper evidence.

These passes are deliberately separate. The LLM pass produces a durable per-paper record
of what was extracted. The dedup pass builds the merged graph without modifying the
extraction record.

---

## Schema Reference

All entity and relationship types are defined in `examples/medlit_schema/`. Key modules:

- `entity.py` — `Disease`, `Gene`, `Drug`, `Protein`, `Mutation`, `Symptom`,
  `Biomarker`, `Evidence`, and scientific method entities
- `relationship.py` — `Treats`, `Causes`, `IncreasesRisk`, `Indicates`, `AssociatedWith`,
  etc., plus `RELATIONSHIP_TYPE_MAP` factory
- `base.py` — `PredicateType`, `EntityType`, `ExtractionMethod`, `StudyType`,
  `SectionType`, `ExtractionProvenance`, `Measurement`

---

## Pass 1: LLM Extraction

### Input

- A paper in any readable format. Typical inputs include JATS-XML (PMC full-text),
  HTML (PMC web format), LaTeX source, or PDF. All of these can be passed directly
  to the Anthropic API without conversion — PDF as a base64 document input, text
  formats as raw content. A mechanical Markdown conversion step is unnecessary and
  risks introducing its own errors.
- Anthropic API key
- System prompt (preamble) containing: entity type taxonomy, predicate vocabulary,
  judgment guidance, and per-entity-type JSON Schema derived from the medlit Pydantic
  models via `Model.model_json_schema()`

### Process

1. For short papers (fits in one context window): pass the raw file content directly
   to the Anthropic API. PDF is passed as base64 with `media_type: "application/pdf"`.
   Text formats (JATS-XML, HTML, LaTeX) are passed as the `text` content of a user
   message.

   For long papers requiring section-by-section processing: split on structural
   markers before passing to the API. JATS-XML: split on `<sec>` tags. LaTeX: split
   on `\section{}`. HTML: split on `<h2>` or equivalent. PDF has no structural
   markers; very long PDFs may need a text extraction preprocessing step to identify
   section boundaries, or can be chunked by approximate token count as a fallback.

2. Call the Anthropic API with:
   - `system`: preamble (taxonomy + predicate vocabulary + schema + judgment rules +
     process instructions)
   - `messages`: raw paper content (or one section at a time for long papers)
   - `max_tokens`: 16384
   - Instruct the model to return only valid JSON, no preamble or markdown fences

3. Parse the response and write the per-paper bundle JSON.

### Output: Per-paper bundle JSON

The bundle is the single source of truth for that paper's contribution to the graph.
It is never overwritten by the dedup pass — canonical IDs are written as an overlay
or the bundle is treated as immutable after extraction.

All extracted entities have `source="extracted"` (provisional). The dedup/promotion
pass updates `source` to the appropriate ontology value (e.g. `"umls"`, `"hgnc"`)
and populates the corresponding ontology ID field.

```json
{
  "paper": {
    "doi": "10.1002/dc.70049",
    "pmcid": "PMC12756687",
    "title": "Pleural Metastasis From Male Breast Cancer: A Case Report",
    "authors": ["Patrizia Straccia", "Esther Diana Rossi"],
    "journal": "Diagnostic Cytopathology",
    "year": 2026,
    "study_type": "case_report",
    "eco_type": "ECO:0006016"
  },
  "extraction_provenance": {
    "models": {
      "llm": {"name": "claude-sonnet-4-20250514", "version": "20250514"}
    },
    "extraction_pipeline": {
      "name": "medlit_llm_ingest",
      "version": "0.1.0",
      "git_commit": "abc1234567890abcdef...",
      "git_commit_short": "abc1234",
      "git_branch": "main",
      "git_dirty": false,
      "repo_url": "https://github.com/org/medlit"
    },
    "prompt": {
      "version": "v1",
      "template": "medlit_extraction_v1",
      "checksum": "sha256:..."
    },
    "execution": {
      "timestamp": "2026-02-23T12:00:00Z",
      "hostname": "my-machine",
      "python_version": "3.12.0",
      "duration_seconds": 47.3
    },
    "entity_resolution": {
      "canonical_entities_matched": 3,
      "new_entities_created": 31,
      "similarity_threshold": 0.85,
      "embedding_model": "text-embedding-3-small"
    }
  },
  "entities": [
    {
      "id": "e01",
      "class": "Disease",
      "name": "male breast cancer",
      "synonyms": ["MBC", "male breast carcinoma"],
      "source": "extracted",
      "canonical_id": null,
      "umls_id": null
    },
    {
      "id": "g01",
      "class": "Gene",
      "name": "BRCA2",
      "symbol": "BRCA2",
      "source": "extracted",
      "canonical_id": null,
      "hgnc_id": null
    },
    {
      "id": "d01",
      "class": "Drug",
      "name": "eribulin",
      "brand_names": ["Halaven"],
      "source": "extracted",
      "canonical_id": null,
      "rxnorm_id": null
    }
  ],
  "evidence_entities": [
    {
      "id": "ev01",
      "class": "Evidence",
      "entity_id": "PMC12756687:abstract:0:llm",
      "paper_id": "PMC12756687",
      "text_span_id": "PMC12756687:abstract:0",
      "text": "positive for ER (70%)",
      "confidence": 0.95,
      "extraction_method": "llm",
      "study_type": "case_report",
      "eco_type": "ECO:0006016",
      "source": "extracted"
    }
  ],
  "relationships": [
    {
      "subject": "d01",
      "predicate": "TREATS",
      "object": "e02",
      "evidence_ids": ["PMC12756687:case_report:2:llm"],
      "source_papers": ["PMC12756687"],
      "confidence": 0.9,
      "properties": {"line_of_therapy": "adjuvant"},
      "section": "case_report",
      "asserted_by": "llm"
    },
    {
      "subject": "g01",
      "predicate": "INCREASES_RISK",
      "object": "e01",
      "evidence_ids": ["PMC12756687:introduction:1:llm"],
      "source_papers": ["PMC12756687"],
      "confidence": 0.55,
      "section": "introduction",
      "asserted_by": "llm",
      "note": "Hedged claim: 'BRCA2 may play pivotal roles'"
    },
    {
      "subject": "b01",
      "predicate": "INDICATES",
      "object": "e02",
      "evidence_ids": ["PMC12756687:abstract:0:llm"],
      "source_papers": ["PMC12756687"],
      "confidence": 0.95,
      "properties": {"result": "positive", "score": "70%"},
      "section": "abstract",
      "asserted_by": "llm"
    },
    {
      "subject": "b09",
      "predicate": "INDICATES",
      "object": "e02",
      "evidence_ids": ["PMC12756687:immunohistochemistry:3:llm"],
      "source_papers": ["PMC12756687"],
      "confidence": 0.95,
      "properties": {"result": "negative"},
      "section": "immunohistochemistry",
      "asserted_by": "llm"
    },
    {
      "subject": "g03",
      "predicate": "SAME_AS",
      "object": "b03",
      "confidence": 0.4,
      "asserted_by": "llm",
      "resolution": null,
      "note": "AR appears as both a risk gene (introduction) and an IHC biomarker (immunohistochemistry). May be same entity or legitimately distinct roles depending on schema."
    }
  ],
  "notes": [
    "AR gene (g03) and AR biomarker (b03) are the same gene product appearing in two contexts. Dedup pass should resolve via SAME_AS link.",
    "INCREASES_RISK relationships from Introduction use hedged language — confidence 0.4-0.6. IHC INDICATES relationships are definitive — confidence 0.9-0.95.",
    "eribulin (d01) has brand name Halaven — both appear in the paper. brand_names field populated; no SAME_AS needed."
  ]
}
```

---

## Entity Classes and Ontology Mapping

All extracted entities are created with `source="extracted"`. The promotion pass
resolves them to canonical ontology IDs. The Pydantic validator
`BaseMedicalEntity.canonical_entities_have_ontology_ids` enforces that any entity
with `source != "extracted"` must have a non-null ontology ID field.

| Class | Primary Ontology | ID Field | Examples from PMC12756687 |
|---|---|---|---|
| `Disease` | UMLS | `umls_id` | male breast cancer, ductal carcinoma, pleural effusion |
| `Gene` | HGNC | `hgnc_id` | BRCA1, BRCA2, PTEN, CHEK2, CYP17 |
| `Drug` | RxNorm | `rxnorm_id` | eribulin (Halaven), PARP inhibitors |
| `Protein` | UniProt | `uniprot_id` | AR protein (when acting as receptor) |
| `Biomarker` | LOINC | `loinc_code` | ER, PR, AR, Ki67, GATA-3, HER-2 |
| `Mutation` | HGVS/ClinVar | `variant_notation` | BRCA1 mutation, BRCA2 mutation |
| `Symptom` | UMLS | `umls_id` | pleural effusion (symptom context) |

---

## Relationship Predicates

Use the predicates defined in `PredicateType` and implemented in `RELATIONSHIP_TYPE_MAP`.
All medical assertion relationships (`BaseMedicalRelationship` subclasses) require
non-empty `evidence_ids` — enforced by Pydantic.

**Direction conventions matter.** `TREATS` is Drug→Disease, not Disease→Drug.
Include direction conventions with examples in the system prompt.

| Predicate | Class | Direction | Usage in PMC12756687 |
|---|---|---|---|
| `TREATS` | `Treats` | Drug → Disease | eribulin TREATS ductal carcinoma |
| `INCREASES_RISK` | `IncreasesRisk` | Gene/Disease → Disease | BRCA2 INCREASES_RISK male breast cancer |
| `INDICATES` | `Indicates` | Biomarker → Disease | ER INDICATES ductal carcinoma |
| `ASSOCIATED_WITH` | `AssociatedWith` | Any → Any | ER ASSOCIATED_WITH male breast cancer (expressed >90%) |
| `INHIBITS` | `Inhibits` | Drug → Gene/Protein | PARP inhibitor INHIBITS PARP enzyme |
| `DIAGNOSED_BY` | `DiagnosedBy` | Disease → Procedure | pleural metastasis DIAGNOSED_BY effusion cytology |
| `AFFECTS` | `Affects` | Disease → AnatomicalStructure | ductal carcinoma AFFECTS pleura |
| `SUBTYPE_OF` | `SubtypeOf` | Disease/Entity → Disease/Entity | ductal carcinoma SUBTYPE_OF breast cancer |
| `SAME_AS` | `SameAs` *(add — see below)* | Any → Any | AR gene SAME_AS AR biomarker |

### IHC positive vs. negative

There are no separate `ihc_positive_for` / `ihc_negative_for` predicates. Use
`INDICATES` for both, and record the result in `properties`:

```json
{"predicate": "INDICATES", "properties": {"result": "positive", "score": "70%"}}
{"predicate": "INDICATES", "properties": {"result": "negative"}}
```

### Confidence scoring

`confidence` is a float 0.0–1.0 (not a string). Guidance for the LLM:

| Language in paper | confidence |
|---|---|
| Definitive measurement (IHC score, dose, stated fact) | 0.9–1.0 |
| Clinical observation | 0.8–0.9 |
| Probable / likely | 0.6–0.8 |
| Hedged: "may", "might", "could" | 0.4–0.6 |
| Speculative / hypothetical | 0.2–0.4 |

---

## Evidence as a First-Class Entity

`Evidence` objects are created alongside relationships and referenced by `evidence_ids`.
The canonical ID format is: `{paper_id}:{section}:{paragraph_idx}:{method}`

Example: `"PMC12756687:abstract:0:llm"`

This enables:
- Immediate canonical ID with no provisional state needed for Evidence
- Efficient queries: "all evidence from Results section"
- Deduplication across extraction runs
- Full traceability to exact text location

The `evidence_entities` array in the bundle contains all Evidence objects for the paper.
Each relationship's `evidence_ids` references these by canonical ID.

---

## The `SAME_AS` Predicate

`SAME_AS` is a provisional identity link used to record the LLM's belief that two
bundle-local entities may refer to the same real-world concept, without collapsing them
prematurely.

### Schema addition required

`SAME_AS` is not yet in `PredicateType` or `RELATIONSHIP_TYPE_MAP`. Add it:

```python
# In base.py PredicateType enum:
SAME_AS = "same_as"

# In relationship.py:
class SameAs(ResearchRelationship):
    """
    Provisional identity link between two entities.

    Not a BaseMedicalRelationship — no evidence_ids required.
    Direction: conventionally lower bundle ID → higher bundle ID.

    Attributes:
        confidence: Strength of identity claim (0.0-1.0)
        resolution: Outcome after review ("merged", "distinct", null = unreviewed)
        note: Free text explaining the ambiguity
    """
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    resolution: Optional[Literal["merged", "distinct"]] = None
    note: Optional[str] = None

    def get_edge_type(self) -> str:
        return "SAME_AS"

# Add to RELATIONSHIP_TYPE_MAP:
"SAME_AS": SameAs,
```

### When the LLM should emit `SAME_AS`

- Same abbreviation used in two different type contexts within the paper
  (e.g. AR as a gene in Introduction, AR as a biomarker in IHC) → `confidence: 0.3–0.5`
- Different name forms clearly referring to the same drug — prefer `brand_names` field;
  `SAME_AS` only if genuinely ambiguous → `confidence: 0.8–0.9`
- Two entity names that appear to be synonyms across sections → `confidence: 0.5–0.7`

### Resolution lifecycle

| State | `resolution` | `asserted_by` | Action |
|---|---|---|---|
| Unreviewed | `null` | `"llm"` | Queued for review GUI |
| Auto-resolved high-confidence | `"merged"` | `"automated"` | Dedup collapses entities |
| Human confirmed same | `"merged"` | `"reviewed"` | Dedup collapses entities |
| Human confirmed distinct | `"distinct"` | `"reviewed"` | Link preserved, entities stay separate |

The review GUI filters on `predicate="SAME_AS"` and `resolution=null` to show
unreviewed links to the med student reviewer.

---

## Pass 2: Deduplication and Promotion

### Input

All per-paper bundle JSON files.

### Process

**Step 1 — Build name/type index**
Scan all bundles. Collect all `(name.lower(), class)` pairs. Exact matches across
papers get the same canonical ID.

**Step 2 — Auto-resolve high-confidence `SAME_AS`**
For `SAME_AS` links with `confidence >= 0.85`: merge entities, union
`synonyms`/`brand_names`, assign single canonical ID, set `resolution="merged"`,
`asserted_by="automated"`.

**Step 3 — Assign canonical IDs**
Every entity without a `canonical_id` gets one (UUID or human-readable slug).
Write `canonical_id` back into each bundle's entity records.

**Step 4 — Ontology lookup (optional, enriches graph)**
For entities where class and name map to a known ontology:
- `Disease` → UMLS → `umls_id`, `source="umls"`
- `Gene` → HGNC → `hgnc_id`, `source="hgnc"`
- `Drug` → RxNorm → `rxnorm_id`, `source="rxnorm"`
- `Biomarker` → LOINC → `loinc_code`, `source="loinc"`

Entities that cannot be resolved remain `source="extracted"` until manual review.

**Step 5 — Update relationship refs**
Replace bundle-local IDs (`e01`, `g01`) with canonical IDs. Auto-merged `SAME_AS`
links are removed (implicit in alias list). Low/medium-confidence `SAME_AS` links
keep canonical IDs on both sides and remain in the graph for the review GUI.

**Step 6 — Accumulate `source_papers` and `evidence_ids`**
For the same canonical (subject, predicate, object) triple across multiple papers:
union `source_papers`, union `evidence_ids`, retain highest `confidence`.

### Dedup policy

| Case | Action |
|---|---|
| Same name (case-folded), same class, across papers | Merge → single canonical ID |
| Same name, different class | Keep separate; emit `SAME_AS` (`confidence: 0.3`) if not already present |
| `SAME_AS` confidence ≥ 0.85 | Auto-merge → single canonical ID, union synonyms |
| `SAME_AS` confidence < 0.85 | Keep separate, preserve link, queue for human review |
| Same canonical (subject, predicate, object) from two papers | Accumulate `source_papers` + `evidence_ids` |

---

## Synonym and Identity Cache

The synonym cache is the set of all `SAME_AS` links across all bundles, indexed by
normalized entity name. It persists across ingestion runs.

```json
{
  "ar": [
    {
      "entity_a": {"name": "AR gene", "class": "Gene", "canonical_id": "canon-ar-gene"},
      "entity_b": {"name": "AR", "class": "Biomarker", "canonical_id": "canon-ar-biomarker"},
      "confidence": 0.4,
      "asserted_by": "llm",
      "resolution": null,
      "source_papers": ["10.1002/dc.70049"]
    }
  ]
}
```

During future ingestion runs, `lookup_entity(name, type)` checks this cache before
creating new entities, surfacing existing canonical IDs or flagging known ambiguities.

---

## Extraction Provenance

The `extraction_provenance` block records everything needed to reproduce the extraction
exactly. It maps to `ExtractionProvenance` in `base.py`.

| Field | Purpose |
|---|---|
| `models.llm.name` | Exact model string used (e.g. `claude-sonnet-4-20250514`) |
| `extraction_pipeline.git_commit` | Full commit hash — flags uncommitted changes |
| `prompt.checksum` | SHA256 of actual prompt text — detects prompt drift between runs |
| `execution.timestamp` | ISO 8601 UTC — enables temporal analysis of extraction quality |

Enables queries like:
- "Find all papers extracted with prompt v1 so I can re-extract with v2"
- "Compare entity extraction quality between model versions"
- "Which extractions used uncommitted code?"

The `entity_resolution` sub-object (`EntityResolutionInfo`) is populated by the dedup
pass, not the extraction pass, and records how many entities were matched to existing
canonical IDs vs. newly created. `similarity_threshold` and `embedding_model` are
relevant if embedding-based fuzzy matching is used (future work — exact-match only
for now, leave `entity_resolution: null` until then).

---

## Confidence and Trust

| `asserted_by` | Meaning |
|---|---|
| `"llm"` | LLM extraction pass — treat as hypothesis |
| `"automated"` | Resolved by ontology lookup or high-confidence `SAME_AS` — intermediate |
| `"reviewed"` | Confirmed by human curator or clinician — production-ready |

No downstream system should treat `asserted_by="llm"` relationships as ground truth
without acknowledging their provisional status.

`Evidence` entities carry an `eco_type` field for ECO ontology classification of
evidence quality. For a case report: `ECO:0006016`.

---

## Implementation Notes

**Predicate directions are enforced by the schema.** `TREATS` is Drug→Disease.
The system prompt must include direction conventions with examples for each predicate.

**`evidence_ids` required for medical assertions.** `BaseMedicalRelationship` enforces
this via Pydantic. The LLM must create `Evidence` objects in `evidence_entities` and
reference them by canonical ID in each relationship's `evidence_ids`. `SAME_AS` and
other `ResearchRelationship` subclasses are exempt from this requirement.

**Evidence canonical ID format.** `{paper_id}:{section}:{paragraph_idx}:{method}`.
Use `SectionType` values from `base.py` for the section segment: `abstract`,
`introduction`, `methods`, `results`, `discussion`, `conclusion`.

**`Measurement` for quantitative relationship data.** `BaseMedicalRelationship` has an
optional `measurements: list[Measurement]` field. Use it for IHC scores, risk ratios,
response rates, p-values, etc. rather than stuffing them into `properties`. Key fields:
`value`, `unit`, `value_type` (e.g. `"response_rate"`, `"risk_ratio"`, `"ihc_score"`),
`p_value`, `confidence_interval`, `study_population`.

**`Polarity`.** `base.py` defines a `Polarity` enum (SUPPORTS/REFUTES/NEUTRAL) used on
`ClaimEdge` in the base edge hierarchy. It is not currently on `BaseMedicalRelationship`
directly, but is worth tracking for hypothesis-testing relationships (`SUPPORTS`,
`REFUTES`, `TESTED_BY`) where the direction of evidence matters.

**Immutable extraction record.** The bundle JSON is never modified after extraction.
Canonical IDs and ontology lookups are written to a separate overlay or a promoted copy.
The original extraction record is preserved permanently for audit and re-extraction.

**`SAME_AS` in the `relationships` array.** It is distinguishable from clinical
relationships by predicate value and by the absence of `evidence_ids`. The review GUI
filters on `predicate="SAME_AS"` and `resolution=null`.

**Public URL (Anthropic API MCP connector only).** Anthropic's infrastructure calls back
to your KGServer from outside. The MCP endpoint must be publicly reachable — use ngrok
(`ngrok http 8001`) or an SSH reverse tunnel for local development. A self-managed MCP
client (Python MCP SDK) can connect to localhost directly and does not require a tunnel.
