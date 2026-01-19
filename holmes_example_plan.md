# Sherlock Holmes Knowledge Graph Example — Implementation Plan

## Overview

This example is a *reference implementation* showing how to build a domain-specific knowledge graph using **kgraph**. It ingests public-domain Sherlock Holmes stories (Project Gutenberg), extracts entities and relationships, and demonstrates basic querying.

### Purpose

Provide a complete, correct, and idiomatic example that extension authors can copy/adapt.

### Design goals

- Align closely with the `kgraph` ingestion pipeline and interfaces
- Keep pipeline responsibilities cleanly separated:
  - **Pass 1**: parse document → emit **mentions** → resolve into **entities**
  - **Pass 2**: derive **relationships** from document + resolved entities
- Demonstrate canonical vs provisional entities in a way users can understand
- Prefer readability and teachability over cleverness

---

## File Structure

```

examples/sherlock/
├── README.md
├── __init__.py
├── domain.py                 # DomainSchema + entity/relationship/document classes
├── data.py                   # curated data: characters, locations, stories (pure constants)
├── pipeline/
│   ├── __init__.py
│   ├── parser.py             # SherlockDocumentParser
│   ├── mentions.py           # SherlockEntityExtractor (mentions only)
│   ├── resolve.py            # SherlockEntityResolver
│   ├── relationships.py      # SherlockRelationshipExtractor
│   └── embeddings.py         # SimpleEmbeddingGenerator (or stub)
├── sources/
│   ├── __init__.py
│   └── gutenberg.py          # download + split utilities
├── scripts/
│   ├── __init__.py
│   ├── ingest.py             # CLI-style entrypoint; uses orchestrator
│   └── query.py              # demo queries
└── output/                   # gitignored

````

### Rationale for this layout

- `data.py` is intentionally “dumb”: no IO, no regex, no logic — just curated constants.
- `pipeline/` mirrors the interfaces and makes it easy to see what happens in each phase.
- `sources/` isolates external IO concerns (Gutenberg download + splitting).
- `scripts/` provides runnable entrypoints without turning the rest of the package into “script soup”.

---

## Component responsibilities

### `domain.py` (schema + types)

Defines the domain vocabulary and its concrete types:

- Entities:
  - `SherlockCharacter`
  - `SherlockLocation`
  - `SherlockStory`
- Relationships (initial minimal set):
  - `appears_in` (character → story)
  - `co_occurs_with` (character ↔ character)

**Note on scope:** you can keep `lives_at`, `ally_of`, `antagonist_of` in the *plan* as “future” predicates, but for a first implementation you’ll get a much cleaner working example by starting with only the predicates you can actually extract reliably from text (co-occurrence is the simplest).

Document type:
- `SherlockDocument` extends `BaseDocument` and carries `story_id`, `collection`, and any story metadata.

Domain schema:
- Registers entity types, relationship types, document types
- `promotion_config` demonstrates a realistic promotion threshold
- Validation ensures entity/relationship vocab correctness

---

### `data.py` (curated constants)

Contains:

- `KNOWN_CHARACTERS`: canonical IDs, display names, aliases, role
- `KNOWN_LOCATIONS`: canonical IDs, display names, aliases, location_type
- `ADVENTURES_STORIES`: canonical story IDs + title + year + collection + **Gutenberg split marker**

Important: `ADVENTURES_STORIES` should include *markers* that `sources/gutenberg.py` uses to split the Gutenberg master text into individual stories.

Canonical ID scheme:
- Characters: `holmes:char:<Name>`
- Locations: `holmes:loc:<Name>`
- Stories: `holmes:story:<Name>`

---

### `pipeline/parser.py` — `SherlockDocumentParser`

Input: raw bytes, content_type, source_uri
Output: `SherlockDocument`

Responsibilities:

- Validate content type is text/plain
- Decode with a robust strategy (`utf-8` with `errors="replace"` or `utf-8-sig`)
- Infer story title and look up metadata from `data.py`
- Populate:
  - `document_id`
  - `title`
  - `content`
  - `created_at`
  - `story_id` (canonical ID if known)
  - `collection`
  - `metadata` (publication year etc.)

---

### `pipeline/mentions.py` — `SherlockEntityExtractor` (mentions only)

Input: `BaseDocument`
Output: list of `EntityMention`

Responsibilities:

- Emit **mentions** only — no entity creation, no storage access.
- Emit a single **story mention** per document:
  - entity_type = `"story"`
  - canonical hint = document.story_id
- Emit character/location mentions via regex matching against alias lists.

Metadata guidance:
- Put a canonical ID hint in mention metadata when the extractor is confident:
  - `metadata={"canonical_id_hint": "..."}`
- Do *not* include document_id redundantly unless you need it (most pipelines already have doc context).

---

### `pipeline/resolve.py` — `SherlockEntityResolver`

Input: `EntityMention`, `EntityStorageInterface`
Output: `(BaseEntity, confidence)`

Responsibilities:

- If mention includes `canonical_id_hint`:
  - Return existing canonical entity if stored
  - Otherwise, create canonical entity from `data.py`
- If no canonical hint:
  - Create provisional entity (new `prov:` ID; stable ID is optional, UUID is fine)
  - Lower confidence appropriately (e.g., `mention.confidence * 0.5`)

Note: if the resolver creates canonical entities, it should create the *domain-specific subclass* (`SherlockCharacter`, etc.), not the bare `BaseEntity`. That makes the example more instructive.

---

### `pipeline/relationships.py` — `SherlockRelationshipExtractor` (relationships only)

Input: document + resolved entities
Output: list of `BaseRelationship`

Hard rule:
- **No entity creation occurs here.** If a story entity is required, it must already exist from mention extraction + resolution.

Initial relationship extraction:

1. `appears_in`: for each character seen in the document, create `character → story`
2. `co_occurs_with`: for character pairs co-mentioned within same paragraph
   - Use paragraph split heuristic: `document.content.split("\n\n")`
   - Count co-occurrence occurrences and map to a confidence curve:
     - e.g. `min(0.95, 0.60 + 0.10 * count)`

De-duplication:
- Normalize pairs so `(a, b)` and `(b, a)` collapse to a stable ordering.

Metadata:
- Store counts (`{"co_occurrence_count": n}`) to make the result explainable.

---

### `pipeline/embeddings.py` — `SimpleEmbeddingGenerator` (demo only)

Purpose:
- Demonstrate the embedding interface without requiring external dependencies.

Implementation:
- Deterministic hash-based vector is fine as a stub.
- Dimension should be small (e.g. 32) and documented as “not semantically meaningful”.

---

### `sources/gutenberg.py`

Responsibilities:

- Download the Gutenberg master text for *The Adventures of Sherlock Holmes*
- Cache it under `examples/sherlock/output/` or `examples/sherlock/.cache/` (preferred)
- Split into individual stories using markers from `data.py`
- Return `list[tuple[title, content]]`

Implementation details:

- Normalize to uppercase for marker scanning to avoid casing issues.
- Use the story metadata for titles and canonical IDs.

---

### `scripts/ingest.py`

This is the runnable entrypoint.

Responsibilities:

- Instantiate:
  - storages (in-memory storage is fine for a reference example)
  - domain (`SherlockDomainSchema`)
  - pipeline components (`parser`, `mentions`, `resolver`, `relationships`, `embeddings`)
  - `IngestionOrchestrator`
- Download stories and ingest each one
- Print per-document ingestion results
- Print final summary counts (docs/entities/relationships)
- Optionally export JSON to `output/` if supported by your orchestrator / storages

CLI expectations:
- Run as module:
  - `python -m examples.sherlock.scripts.ingest`

---

### `scripts/query.py`

This should demonstrate querying without requiring a graph database.

Two options (pick one, keep it simple):

**Option A (simplest):** re-ingest in memory, then run queries
Pros: no persistence complexity
Cons: slower, but fine for an example

**Option B:** load exported JSON from `output/`
Pros: “ingest once, query many times”
Cons: requires you to define a stable export format

Queries to include:

- Characters appearing in a given story (`appears_in` edges)
- Top co-occurrences for a character (`co_occurs_with`)
- Most mentioned characters (by `usage_count` if tracked)

Run as:
- `python -m examples.sherlock.scripts.query`

---

## Implementation sequence (practical)

1. `domain.py` — define vocabulary and ensure it imports cleanly
2. `data.py` — curated constants + story markers
3. `sources/gutenberg.py` — download + cache + split
4. `pipeline/parser.py` — parse into `SherlockDocument`
5. `pipeline/mentions.py` — emit mentions (including story mention)
6. `pipeline/resolve.py` — resolve mentions into canonical/provisional entities
7. `pipeline/relationships.py` — extract `appears_in` + `co_occurs_with`
8. `pipeline/embeddings.py` — stub embedding generator
9. `scripts/ingest.py` — runnable end-to-end ingestion
10. `scripts/query.py` — runnable query demo
11. `README.md` — document once the code is real

---

## Verification

From repo root:

```bash
uv pip install -e ".[dev]"

python -m examples.sherlock.scripts.ingest
python -m examples.sherlock.scripts.query
````

Expected (ballpark) results for Adventures:

* 12 documents (stories)
* Dozens of canonical characters (depending on how much you curate)
* A handful of canonical locations
* 12 story entities
* Hundreds of `appears_in` edges + many `co_occurs_with` edges

---

## “Later” ideas (deliberately out of scope for v1)

* `lives_at`, `ally_of`, `antagonist_of` extraction from linguistic patterns
* Better co-occurrence (sentence windowing, mention-based detection, coref)
* Persistence format + load path for `scripts/query.py`
* Adding more corpora (Memoirs, Return, etc.)

---

## References within kgraph

* `tests/conftest.py`: mock implementations and interface examples
* `kgraph/domain.py`: DomainSchema base
* `kgraph/entity.py`: BaseEntity, EntityMention, EntityStatus
* `kgraph/relationship.py`: BaseRelationship
* `kgraph/document.py`: BaseDocument
* `kgraph/pipeline/interfaces.py`: parser/extractor/resolver interfaces
* `kgraph/ingest.py`: orchestrator usage

```

---

If you want, I can also adjust this plan to match *your exact current* `kgraph` API surface (e.g., whether `IngestionOrchestrator` expects a context/builder now), but the structure above is already consistent with the reorganized example layout you just adopted.
::contentReference[oaicite:0]{index=0}
