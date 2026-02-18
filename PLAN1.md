# Implementation Plan: Provenance in the Bundle (TODO1)

This plan implements the design in **TODO1.md**: extend the V1 bundle contract to always include provenance (entity mentions and relationship evidence), persist it through export/load, and surface it in the graph visualization.

**How to use this plan:** Work through phases in order (1 → 2 → 3 → 4 → 5 → 6). Phase 1 is the bundle contract (kgbundle); Phase 2 collects provenance during ingestion (kgraph); Phase 3 writes it in export; Phase 4 loads it in KGServer; Phase 5 exposes it in the graph API and UI; Phase 6 is tests and backward compatibility. **All file paths, line references, and source types you need are in the "Code reference" section** so you can implement without opening other files for discovery. **Success:** A bundle exported after ingestion includes `mentions.jsonl` and `evidence.jsonl`; loading it in KGServer and querying the graph returns provenance on nodes/edges; the graph UI shows mentions and evidence in tooltips/panels.

---

## Summary of Goals

- **Bundle contract (V1, always-on):** Add provenance to the existing bundle format; no opt-in flag.
- **Entity provenance:** Each entity gets summary fields (first_seen_document, total_mentions, supporting_documents) and every mention is written as one line in `mentions.jsonl`.
- **Relationship evidence:** Each relationship gets summary fields (evidence_count, strongest_evidence_quote, evidence_confidence_avg) and every evidence span is one line in `evidence.jsonl`.
- **Graph visualization:** Entity tooltips/panel show mentions with document links; edge tooltips/panel show evidence quotes; optional “jump to document/section” links.

---

## Phase 1: Bundle contract (kgbundle)

**Owner:** kgbundle (shared contract).  
**Backward compatibility:** New fields are additive; existing consumers can ignore new files and new fields.

### 1.1 New models in `kgbundle/kgbundle/models.py`

Add Pydantic models:

```python
class MentionRow(BaseModel):
    """One entity mention occurrence (one line in mentions.jsonl)."""
    entity_id: str
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str
    context: Optional[str] = None
    confidence: float
    extraction_method: str  # e.g. "llm", "rule_based", "canonical_lookup"
    created_at: str  # ISO 8601

class EvidenceRow(BaseModel):
    """One evidence span supporting a relationship (one line in evidence.jsonl)."""
    relationship_key: str  # f"{subject_id}:{predicate}:{object_id}"
    document_id: str
    section: Optional[str] = None
    start_offset: int
    end_offset: int
    text_span: str
    confidence: float
    supports: bool = True  # True=supports, False=contradicts
```

### 1.2 Extend existing models in `kgbundle/kgbundle/models.py`

**Exact locations (current code):** `EntityRow` is lines 16–33; `RelationshipRow` is 36–50; `BundleManifestV1` is 75–93; `BundleFile` is 54–57.

- **EntityRow:** Add these optional fields (e.g. after `properties`, before the closing of the class):
  - `first_seen_document: Optional[str] = None`
  - `first_seen_section: Optional[str] = None`
  - `total_mentions: int = 0`
  - `supporting_documents: List[str] = Field(default_factory=list)`

- **RelationshipRow:** Add (e.g. after `properties`):
  - `evidence_count: int = 0`
  - `strongest_evidence_quote: Optional[str] = None`
  - `evidence_confidence_avg: Optional[float] = None`

- **BundleManifestV1:** Add after `doc_assets` (around line 88):
  - `mentions: Optional[BundleFile] = None`  # mentions.jsonl
  - `evidence: Optional[BundleFile] = None`  # evidence.jsonl

### 1.3 Bundle layout (after implementation)

```
bundle_dir/
├── manifest.json      # includes entities, relationships, mentions?, evidence?, doc_assets?
├── entities.jsonl    # EntityRow (with new provenance summary fields)
├── relationships.jsonl
├── mentions.jsonl    # MentionRow, one line per mention
├── evidence.jsonl    # EvidenceRow, one line per evidence span
└── doc_assets.jsonl  # unchanged
```

### 1.4 Deliverables

- [ ] `MentionRow` and `EvidenceRow` in `kgbundle/kgbundle/models.py` (full model code in 1.1 above).
- [ ] Export them from `kgbundle/kgbundle/__init__.py`: add `MentionRow` and `EvidenceRow` to the imports from `.models` and to `__all__`.
- [ ] `EntityRow` and `RelationshipRow` extended with provenance summary fields; `BundleManifestV1` extended with `mentions` and `evidence` (optional).
- [ ] Tests in kgbundle for new models and serialization.

---

## Phase 2: Collecting provenance during ingestion (kgraph)

Currently mentions exist only in the extraction loop and are not persisted. Relationship evidence exists on `BaseRelationship.evidence` in memory but is not written to storage. We need a way to record mentions and evidence so the exporter can write `mentions.jsonl` and `evidence.jsonl`.

### 2.1 Strategy: Provenance accumulator

Introduce a **provenance accumulator** that the orchestrator (and optionally extractors) feed during ingestion. At export time, the exporter reads from this accumulator to write mentions and evidence. This avoids changing the core entity/relationship storage schema and keeps kgraph storage backends unchanged for now.

- **Option A (recommended): In-memory accumulator passed to export**  
  - New type, e.g. `ProvenanceAccumulator` or `BundleProvenanceCollector`, with:
    - `add_mention(entity_id, document_id, section, start_offset, end_offset, text_span, context, confidence, extraction_method, created_at)`
    - `add_evidence(subject_id, predicate, object_id, document_id, section, start_offset, end_offset, text_span, confidence, supports=True)`
  - Orchestrator (or the code that resolves mentions and stores relationships) receives the accumulator and calls `add_mention` for each resolved mention and `add_evidence` when a relationship is stored (from `relationship.evidence`).
  - Export receives the same accumulator and writes `mentions.jsonl` and `evidence.jsonl` from it; also derives EntityRow provenance summary (first_seen_document, total_mentions, supporting_documents) and RelationshipRow evidence summary from accumulator data.

- **Option B: Persist in entity/relationship storage**  
  - Store mentions in entity storage (e.g. a `MentionStorageInterface` or entity.metadata["mentions"]) and evidence in relationship storage (e.g. relationship.properties["evidence_spans"] or a separate EvidenceStorageInterface). Export would read from storage.  
  - Requires new storage methods or schema (e.g. kgserver Entity/Relationship tables or in-memory stores extended). Heavier change.

**Recommendation:** Option A for a first iteration: minimal storage changes, clear path to write the new bundle files. We can later move to Option B if we need provenance to survive across runs or be queryable without export.

### 2.2 Where to call the accumulator (exact call sites)

- **Mentions:** In `kgraph/ingest.py`, in `extract_entities_from_document`: immediately after `await self.entity_storage.update(updated)` (around line 325) and after `await self.entity_storage.add(entity)` (around line 341), call `accumulator.add_mention(entity_id=entity.entity_id, document_id=document.document_id, section=document.metadata.get("section") if document.metadata else None, start_offset=mention.start_offset, end_offset=mention.end_offset, text_span=mention.text, context=mention.context, confidence=mention.confidence, extraction_method=document.metadata.get("extraction_method", "llm") if document.metadata else "llm", created_at=datetime.now(timezone.utc).isoformat())`. The orchestrator must receive an optional `provenance_accumulator` (e.g. in `__init__` or as an optional argument to this method); if `None`, skip the call. For streaming paths, use the chunk’s or document’s document_id and section the same way.
- **Evidence:** In `kgraph/ingest.py`, in `extract_relationships_from_document`: immediately after each `await self.relationship_storage.add(rel)` (around lines 444 and 452), if `rel.evidence` is not None, call the accumulator. For each evidence span: from `rel.evidence.primary` (a `Provenance`: document_id, section, start_offset, end_offset) and evidence text from `rel.evidence.notes.get("evidence_text", "")`; confidence from `rel.confidence`; `relationship_key = f"{rel.subject_id}:{rel.predicate}:{rel.object_id}"`. Call `accumulator.add_evidence(relationship_key=..., document_id=..., section=..., start_offset=..., end_offset=..., text_span=..., confidence=..., supports=True)`. Repeat for `rel.evidence.mentions` if you want each mention as a separate evidence row.

### 2.3 Entity provenance summary (for EntityRow)

Derive from accumulator (or from mention list per entity):

- `first_seen_document`: document_id of the earliest mention (by created_at or order of add_mention).
- `first_seen_section`: section of that first mention, if any.
- `total_mentions`: count of mention rows for this entity_id.
- `supporting_documents`: distinct list of document_ids for this entity_id.

These can be computed when writing the bundle (single pass over mentions).

### 2.4 Relationship evidence summary (for RelationshipRow)

Derive from accumulator (or from evidence rows per relationship):

- `evidence_count`: count of evidence rows for this (subject_id, predicate, object_id).
- `strongest_evidence_quote`: text_span of the evidence row with highest confidence (or first).
- `evidence_confidence_avg`: mean of evidence confidence for this relationship.

### 2.5 Deliverables

- [ ] Define `ProvenanceAccumulator` (or equivalent) interface and in-memory implementation in kgraph (e.g. `kgraph/provenance.py` or under `kgraph/export.py`).
- [ ] In the medlit (or generic) ingestion path: after resolving a mention and updating entity storage, call accumulator `add_mention` with data from document + EntityMention.
- [ ] When storing a relationship that has `evidence`: call accumulator `add_evidence` for primary provenance (and optionally mentions) using document_id, section, offsets, evidence text from notes, confidence, supports=True.
- [ ] Export (Phase 3) receives the accumulator and uses it to write mentions/evidence and to fill EntityRow/RelationshipRow provenance fields.

---

## Phase 3: Export (kgraph)

### 3.1 Export API

- **Exact location:** `kgraph/export.py`. `JsonlGraphBundleExporter.export_graph_bundle` starts at line 106; it currently takes `entity_storage`, `relationship_storage`, `bundle_path`, `domain`, `label`, `docs`, `description`. Add an optional parameter: `provenance_accumulator: Optional[Any] = None` (use the type you define in Phase 2). The helper `write_bundle` (line 226) must also accept and pass through `provenance_accumulator` to the exporter. If provided, write `mentions.jsonl` and `evidence.jsonl` and set the new EntityRow and RelationshipRow fields as below.

### 3.2 Writing entities.jsonl

- For each entity, build `EntityRow` as today, plus:
  - From accumulator: `first_seen_document`, `first_seen_section`, `total_mentions`, `supporting_documents`.
  - If no accumulator, leave these as default (None / 0 / []).

### 3.3 Writing relationships.jsonl

- For each relationship, build `RelationshipRow` as today, plus:
  - From accumulator: `evidence_count`, `strongest_evidence_quote`, `evidence_confidence_avg`.
  - If no accumulator, leave as default.

### 3.4 Writing mentions.jsonl

- If accumulator is present and has mention data: open `bundle_path/mentions.jsonl`, iterate accumulator’s mentions (or iterate by entity_id then by mention), write one `MentionRow.model_dump_json()` per line.

### 3.5 Writing evidence.jsonl

- If accumulator is present and has evidence data: open `bundle_path/evidence.jsonl`, iterate accumulator’s evidence, write one `EvidenceRow.model_dump_json()` per line. `relationship_key` = `f"{subject_id}:{predicate}:{object_id}"`.

### 3.6 Manifest

- If `mentions.jsonl` or `evidence.jsonl` was written, set `manifest.mentions` and/or `manifest.evidence` to `BundleFile(path="mentions.jsonl", format="jsonl")` and same for evidence.

### 3.7 Ingest script wiring (exact location)

- In `examples/medlit/scripts/ingest.py`: create the provenance accumulator in the same place you create the orchestrator (e.g. in `main()` or in the function that builds the pipeline, before the ingestion loop). Pass it into `IngestionOrchestrator` (orchestrator must accept an optional `provenance_accumulator` in its constructor or in the extract methods). When calling `write_bundle` (around line 1044), pass the same accumulator so the exporter can write `mentions.jsonl` and `evidence.jsonl` and fill EntityRow/RelationshipRow provenance fields.

### 3.8 Deliverables

- [ ] Export accepts optional provenance accumulator.
- [ ] Export writes `mentions.jsonl` and `evidence.jsonl` when accumulator is provided and non-empty.
- [ ] Export sets EntityRow and RelationshipRow provenance summary fields from accumulator.
- [ ] Export sets `manifest.mentions` and `manifest.evidence` when those files are written.
- [ ] Medlit ingest script creates and wires the accumulator through ingestion and export.

---

## Phase 4: KGServer load and storage

### 4.1 Backend storage for mentions and evidence

- **Option A:** New tables `Mention` and `Evidence` (or `relationship_evidence`) in SQLite/Postgres, with foreign keys or relationship_key to entity/relationship. Load `mentions.jsonl` and `evidence.jsonl` in `load_bundle` and insert into these tables.
- **Option B:** Store as JSON in existing tables (e.g. entity.properties["mentions"], relationship.properties["evidence_spans"]). Simpler but less good for querying by document or by entity.

**Recommendation:** Option A for queryability (e.g. “all mentions for this entity”, “all evidence for this relationship”, “all mentions in this document”).

### 4.2 Storage interface

- Extend `StorageInterface` (or add a separate interface) with methods such as:
  - `get_mentions_for_entity(entity_id) -> list[MentionRow]` (or domain model)
  - `get_evidence_for_relationship(subject_id, predicate, object_id) -> list[EvidenceRow]`
  - Optionally: `get_mentions_for_document(document_id)` for “jump to document” views.

- Implement in SQLite and Postgres backends: read from the new tables (or from JSON columns if Option B).

### 4.3 load_bundle (exact locations)

- **SQLite:** `kgserver/storage/backends/sqlite.py`. `load_bundle` is at line 41. After the existing entity and relationship load block (lines 53–71), add: if `bundle_manifest.mentions` is not None, open `f"{bundle_path}/{bundle_manifest.mentions.path}"` and for each line parse as JSON and insert into the Mention table (or append to an in-memory list if using Option B). Same for `bundle_manifest.evidence` and the Evidence table. Commit after.
- **Postgres:** `kgserver/storage/backends/postgres.py`. `load_bundle` is at line 24; same pattern: after loading entities and relationships (around 36–50), if `manifest.mentions` / `manifest.evidence` present, load the corresponding JSONL files into the new tables.
- Backward compatibility: if `manifest.mentions` or `manifest.evidence` is None, skip; old bundles continue to load without new tables.

### 4.4 Entity/Relationship models (kgserver)

- Entity and Relationship models can stay as they are for core fields. Provenance summary fields on EntityRow/RelationshipRow can be:
  - Stored in `entity.properties` / `relationship.properties` when loading from bundle (so graph API can return them), or
  - Recomputed from Mention/Evidence tables when serving.  
  Storing in properties keeps the main entity/relationship load path simple and ensures the graph API has first_seen_document, total_mentions, evidence_count, strongest_evidence_quote, etc., without extra queries.

### 4.5 Deliverables

- [ ] New tables (or equivalent) for mentions and evidence in SQLite and Postgres.
- [ ] `load_bundle` loads `mentions.jsonl` and `evidence.jsonl` when present in manifest.
- [ ] Storage interface (or concrete backends) expose `get_mentions_for_entity` and `get_evidence_for_relationship`.
- [ ] Entity/relationship payloads returned by the API include provenance summary (e.g. from properties or from new columns).

---

## Phase 5: Graph API and visualization

### 5.1 Graph API (REST / subgraph)

- **Exact locations:** Node/edge payloads are built in `kgserver/query/graph_traversal.py`: `_entity_to_node` (line 63) and `_relationship_to_edge` (line 82). The storage `Entity` and `Relationship` models are in `kgserver/storage/models/entity.py` and `relationship.py`; both have a `properties` dict. When loading the bundle (Phase 4), store provenance summary in `entity.properties` and `relationship.properties` (e.g. first_seen_document, total_mentions, evidence_count, strongest_evidence_quote) so the existing graph code can pass them through. In `_entity_to_node`, include `entity.properties.get("first_seen_document")`, `entity.properties.get("total_mentions")`, etc. in the returned `properties` dict (or add top-level fields). In `_relationship_to_edge`, include evidence summary from `rel.properties`.
- **Node payload:** Include first_seen_document, first_seen_section, total_mentions, supporting_documents; optionally first 5 mention snippets (document_id, text_span) for tooltips.
- **Edge payload:** Include evidence_count, strongest_evidence_quote, evidence_confidence_avg; optionally evidence snippets for the details panel.

### 5.2 Detail endpoints (optional)

- `GET /api/v1/graph/entity/{entity_id}/mentions` → full list of MentionRow (or equivalent) for that entity, so the UI can show “all mentions” with “jump to document” links.
- `GET /api/v1/graph/edge/evidence?subject_id=...&predicate=...&object_id=...` → full list of EvidenceRow for that relationship.

### 5.3 Graph visualization (D3/frontend)

- **Exact file:** `kgserver/query/static/graph-viz.js` (and optionally `kgserver/query/static/index.html` for panel structure). The graph API already returns node/edge data; extend the frontend to read provenance fields from the node/edge payload.
- **Entity tooltip:** Show first_seen_document, total_mentions, and one or two mention snippets (text_span + document_id). Link document_id to a URL if available (e.g. PMC).
- **Entity detail panel:** List mentions with document link, section, text_span, confidence, extraction_method. Use “jump to document” link (e.g. `PMC123#section-abstract`) if we have section/fragment IDs.
- **Edge tooltip:** Show strongest_evidence_quote and evidence_count.
- **Edge detail panel:** List evidence spans (document_id, section, text_span, confidence, supports). Document links as above.

### 5.4 Document fragment links

- If source documents are PMC or have a known URL scheme, format links as `base_url#section-{section}` or `base_url#offset-{start_offset}`. Implementation can be minimal (e.g. PMC URL + fragment) and extended later for other doc types.

### 5.5 Deliverables

- [ ] Subgraph/node/edge responses include provenance summary and optional mention/evidence snippets.
- [ ] Optional REST endpoints for full mention list per entity and full evidence list per relationship.
- [ ] graph-viz.js (or equivalent): entity tooltip and detail panel show mentions and document links; edge tooltip and detail panel show evidence quotes and links.
- [ ] (Optional) Deep links for document section/fragment where feasible.

---

## Phase 6: Testing and backward compatibility

### 6.1 Backward compatibility

- Bundles produced without an accumulator (or by old code) have no `mentions` or `evidence` in the manifest; entities and relationships have default provenance fields (None, 0, []). KGServer skips loading mention/evidence files when not in manifest; graph UI shows no mentions/evidence when fields are missing.
- New bundles always include provenance (per TODO1); no `--include-provenance` flag.

### 6.2 Tests

- **kgbundle:** In kgbundle test module: serialization/deserialization of MentionRow, EvidenceRow; EntityRow and RelationshipRow with new fields; manifest with mentions and evidence. Run: `uv run pytest kgbundle/ -v` (or the project’s test path for kgbundle).
- **kgraph:** Accumulator add_mention/add_evidence; export produces correct mentions.jsonl and evidence.jsonl and summary fields; ingest script runs without error with accumulator wired. Run: `uv run pytest tests/ -v -k "provenance or export or ingest"` (or equivalent).
- **kgserver:** load_bundle with manifest that has mentions/evidence; get_mentions_for_entity and get_evidence_for_relationship return expected rows; graph API returns provenance in node/edge payloads.
- **E2E:** Export a small bundle with provenance, load it in kgserver, fetch subgraph and entity/edge details, confirm mentions and evidence appear in the response and (if UI is tested) in the graph UI.

---

## Dependency order

1. **Phase 1** (kgbundle contract) — no dependencies.
2. **Phase 2** (provenance collection) — depends on Phase 1 (uses MentionRow/EvidenceRow or equivalent in accumulator).
3. **Phase 3** (export) — depends on Phase 1 and 2 (writes new files and row types using accumulator).
4. **Phase 4** (kgserver load) — depends on Phase 1 (reads new manifest fields and new JSONL files).
5. **Phase 5** (API and UI) — depends on Phase 4 (reads from storage and returns provenance to frontend).

---

## File checklist (summary)

| Area        | File(s) |
|------------|---------|
| Contract   | `kgbundle/kgbundle/models.py` |
| Accumulator| `kgraph/provenance.py` (or under export/ingest) |
| Ingestion  | `kgraph/ingest.py`, `examples/medlit/scripts/ingest.py`, medlit relationship extractor (evidence) |
| Export     | `kgraph/export.py` |
| KGServer   | `kgserver/storage/interfaces.py`, `kgserver/storage/backends/sqlite.py`, `kgserver/storage/backends/postgres.py`, `kgserver/storage/models/` (new Mention/Evidence if table-based), `kgserver/query/bundle_loader.py`, `kgserver/query/routers/graph_api.py` |
| Graph UI   | `kgserver/query/static/graph-viz.js`, `kgserver/query/static/index.html` (if detail panel structure changes) |

---

## Code reference (exact locations and source types)

Use this so you do not need to open other files for names and structure. Line numbers are approximate (codebase at plan authoring time).

### kgbundle/kgbundle/models.py

- **EntityRow** (lines 16–33): existing fields `entity_id`, `entity_type`, `name`, `status`, `confidence`, `usage_count`, `created_at`, `source`, `canonical_url`, `properties`. Add: `first_seen_document`, `first_seen_section`, `total_mentions`, `supporting_documents` (see Phase 1.2).
- **RelationshipRow** (36–50): existing fields `subject_id`, `object_id`, `predicate`, `confidence`, `source_documents`, `created_at`, `properties`. Add: `evidence_count`, `strongest_evidence_quote`, `evidence_confidence_avg`.
- **BundleManifestV1** (75–93): add `mentions: Optional[BundleFile] = None`, `evidence: Optional[BundleFile] = None` after `doc_assets`.
- **MentionRow / EvidenceRow:** define as in Phase 1.1 (full Pydantic models in same file). Use `from typing import Optional, List` (already present).

### kgschema (source types for accumulator)

- **EntityMention** (`kgschema/entity.py`, ~177–235): `text`, `entity_type`, `start_offset`, `end_offset`, `confidence`, `context`, `metadata`. Use these when calling `add_mention(text_span=mention.text, ...)`.
- **Provenance** (`kgschema/domain.py`, ~93–113): `document_id`, `source_uri`, `section`, `paragraph`, `start_offset`, `end_offset`. Use when converting `rel.evidence.primary` or `rel.evidence.mentions[i]` to `add_evidence(...)`.
- **Evidence** (`kgschema/domain.py`, ~116–121): `kind`, `source_documents`, `primary: Provenance | None`, `mentions: tuple[Provenance, ...]`, `notes: dict`. Evidence quote text: `rel.evidence.notes.get("evidence_text", "")`.
- **BaseRelationship** (`kgschema/relationship.py`, ~75–99): `subject_id`, `predicate`, `object_id`, `confidence`, `source_documents`, `evidence` (type Evidence | None), `created_at`, `metadata`.

### kgraph/ingest.py

- **extract_entities_from_document:** Loop over `mentions` (line ~309); resolve to `entity`; if existing, `entity_storage.update(updated)` at 325; else `entity_storage.add(entity)` at 341. `document` is in scope (`document.document_id`). Insert accumulator call right after each of 325 and 341 with `entity_id=entity.entity_id`, `document_id=document.document_id`, `section=document.metadata.get("section") if document.metadata else None`, `start_offset=mention.start_offset`, `end_offset=mention.end_offset`, `text_span=mention.text`, `context=mention.context`, `confidence=mention.confidence`, `extraction_method=...`, `created_at=datetime.now(timezone.utc).isoformat()`. Orchestrator must have an optional `provenance_accumulator` (e.g. constructor arg); if None, skip.
- **Streaming path (document_id and section):** When `use_streaming` is True (line 242), the code path still ends up with a single `document` in scope before the `for mention in mentions` loop: either `document` is set when parsing (line 257) or it is parsed after extraction (lines 275–286) and then `await self.document_storage.add(document)` runs. So by the time we reach the loop at 309, `document` is always set. Use `document_id=document.document_id` and `section=document.metadata.get("section") if document and getattr(document, "metadata", None) else None` for every `add_mention` call. If a future extractor attaches section per mention (e.g. `mention.metadata.get("section")`), prefer that when present.
- **extract_relationships_from_document:** After `relationship_storage.add(rel)` at 444 and 452, if `rel.evidence` is not None, call accumulator: from `rel.evidence.primary` (and optionally `rel.evidence.mentions`) get document_id, section, start_offset, end_offset; text_span from `rel.evidence.notes.get("evidence_text", "")`; relationship_key = `f"{rel.subject_id}:{rel.predicate}:{rel.object_id}"`; confidence from `rel.confidence`.

### kgraph/export.py

- **JsonlGraphBundleExporter.export_graph_bundle** (line 106): Add optional `provenance_accumulator` parameter. When building `EntityRow` (146–158), if accumulator is present, compute per-entity summary from accumulator mentions (first_seen_document, total_mentions, supporting_documents) and set on entity_row. When building `RelationshipRow` (165–173), if accumulator present, set evidence_count, strongest_evidence_quote, evidence_confidence_avg from accumulator. After the two `with open(...)` blocks for entities and relationships, if accumulator has data: write `bundle_path / "mentions.jsonl"` (one `MentionRow.model_dump_json()` per line) and `bundle_path / "evidence.jsonl"` (one `EvidenceRow.model_dump_json()` per line). When building `manifest` (201–211), set `mentions=BundleFile(path="mentions.jsonl", format="jsonl")` and `evidence=...` if those files were written.
- **write_bundle** (line 226): Add optional `provenance_accumulator` and pass it to `default_exporter.export_graph_bundle(...)`.

### examples/medlit/scripts/ingest.py

- **write_bundle** is called around line 1044 with `entity_storage`, `relationship_storage`, `bundle_path`, `domain`, `label`, `docs`, `description`. Create the accumulator earlier (e.g. in main, before the ingestion loop). Pass it into the orchestrator (orchestrator constructor or extract calls) and as a new argument to `write_bundle`.

### kgserver

- **load_bundle:** `kgserver/storage/backends/sqlite.py` line 41; `kgserver/storage/backends/postgres.py` line 24. After loading entities and relationships, if `bundle_manifest.mentions`: open `{bundle_path}/{bundle_manifest.mentions.path}`, parse each line as JSON, insert into Mention table (or store in a structure keyed by entity_id). Same for `bundle_manifest.evidence` and Evidence table. Storage interface: extend with `get_mentions_for_entity(entity_id)` and `get_evidence_for_relationship(subject_id, predicate, object_id)` (or by relationship_key).
- **Storing provenance summary in Entity/Relationship (no new columns):** The KGServer `Entity` and `Relationship` models (`kgserver/storage/models/entity.py`, `relationship.py`) have only a generic `properties: dict` for extra data; they do **not** have columns for `first_seen_document`, `evidence_count`, etc. When loading from bundle, each line of `entities.jsonl` is an EntityRow that will include the new optional fields. If you pass that dict to `Entity(**entity_data)`, SQLModel will reject unknown keys. So **before** calling `Entity(**entity_data)` in load_bundle: move provenance summary fields from `entity_data` into `entity_data["properties"]` and remove them from the top level. Concretely, after `entity_data = self._normalize_entity(entity_data)`, for each key in `("first_seen_document", "first_seen_section", "total_mentions", "supporting_documents")` that exists in `entity_data`, do `entity_data.setdefault("properties", {})[key] = entity_data.pop(key)` so that `Entity(**entity_data)` only receives the model’s defined fields. Same for relationship_data before `Relationship(**relationship_data)`: for each of `("evidence_count", "strongest_evidence_quote", "evidence_confidence_avg")` that exists in `relationship_data`, move it into `relationship_data["properties"]` and pop from top level. The existing `_normalize_entity` (lines 75–111) only flattens `metadata` and handles `canonical_url`; it does not move these provenance fields, so do this in the load_bundle loop (or extend _normalize_entity to do it).
- **Graph payloads:** `kgserver/query/graph_traversal.py`: `_entity_to_node` (63), `_relationship_to_edge` (82). Include provenance summary from `entity.properties` / `rel.properties` (populate these when loading bundle from EntityRow/RelationshipRow provenance fields). **Graph UI:** `kgserver/query/static/graph-viz.js` — extend tooltips and detail panels to show the new fields.
- **graph-viz.js tooltips and detail panels (exact pattern):** The UI uses `document.getElementById('tooltip')` and `document.getElementById('detail-panel')` with `panelTitle`, `panelContent`, `closePanel` (lines 25–28). **Node tooltip** (mouseenter on node, line 431): currently `showTooltip(event, `${d.label}\n(${d.entity_type})`)`. To add provenance in the tooltip, extend the string e.g. with `d.properties.total_mentions` and `d.properties.first_seen_document` when present. **Node detail panel** (click on node): `showNodeDetails(node)` (line 533) builds HTML from `const props = node.properties || {}` and a series of `createProperty('Label', props.key)`. It already shows Entity ID, Type, Name, Status, Confidence, Usage Count, Source (with `formatSourceLink`), Canonical URL, Synonyms. Add after the existing properties (e.g. after Usage Count or after Synonyms): `createProperty('First seen document', props.first_seen_document)`, `createProperty('Total mentions', props.total_mentions)`, and optionally `createProperty('Supporting documents', ...)` (format links with `formatSourceLink` if document IDs). **Edge detail panel:** `showEdgeDetails(edge)` (line 598) uses `const props = edge.properties || {}` and shows Predicate, Subject, Object, Confidence, Source Documents. Add: `createProperty('Evidence count', props.evidence_count)`, `createProperty('Strongest evidence', props.strongest_evidence_quote)`. Helper `createProperty(label, value)` (line 620) returns empty string if value is null/undefined/empty, otherwise a `<div class="property">` with label and value. So any new field in `node.properties` or `edge.properties` can be shown by adding one `html += createProperty('Label', props.field_name);` line.

### Step-by-step (no other files needed)

1. **Phase 1:** In `kgbundle/kgbundle/models.py` add `MentionRow` and `EvidenceRow` (full model code in Phase 1.1). Extend `EntityRow`, `RelationshipRow`, `BundleManifestV1` with the optional fields listed in Phase 1.2. Add tests for serialization.
2. **Phase 2:** Define `ProvenanceAccumulator` (e.g. in `kgraph/provenance.py`) with `add_mention(...)` and `add_evidence(...)` and in-memory lists. Add optional `provenance_accumulator` to `IngestionOrchestrator`; in `extract_entities_from_document` after 325 and 341 call `add_mention`; in `extract_relationships_from_document` after 444 and 452 call `add_evidence` when `rel.evidence` is set.
3. **Phase 3:** In `kgraph/export.py` add `provenance_accumulator` to `export_graph_bundle` and `write_bundle`; when present, compute EntityRow/RelationshipRow provenance fields, write mentions.jsonl and evidence.jsonl, set manifest.mentions and manifest.evidence. In medlit ingest script create accumulator and pass to orchestrator and write_bundle.
4. **Phase 4:** In kgserver storage backends add Mention/Evidence tables (or JSON in properties); extend load_bundle to load mentions and evidence when in manifest; add get_mentions_for_entity and get_evidence_for_relationship; when loading bundle, put provenance summary into entity.properties and relationship.properties.
5. **Phase 5:** In graph_traversal include provenance in node/edge properties; in graph_api add optional detail endpoints for mentions and evidence; in graph-viz.js show mentions and evidence in tooltips and panels.
6. **Phase 6:** Add tests and verify old bundles still load.

---

## References

- **TODO1.md** — design and bundle contract sketch.
- **kgschema/entity.py** — `EntityMention` (lines ~177–235: text, start_offset, end_offset, context, confidence).
- **kgschema/domain.py** — `Provenance` (~93–113), `Evidence` (~116–121: primary, mentions, notes).
- **kgschema/relationship.py** — `BaseRelationship.evidence` (line ~87).
- **kgbundle/kgbundle/models.py** — EntityRow (16–33), RelationshipRow (36–50), BundleManifestV1 (75–93).
- **kgraph/export.py** — `JsonlGraphBundleExporter.export_graph_bundle` (106), `write_bundle` (226).
- **kgraph/ingest.py** — entity add/update at 325 and 341; relationship add at 444 and 452.
- **kgserver/storage/backends/sqlite.py** — `load_bundle` (41); postgres.py `load_bundle` (24).
- **kgserver/query/graph_traversal.py** — `_entity_to_node` (63), `_relationship_to_edge` (82).
- **kgserver/query/static/graph-viz.js** — tooltips and detail panels.
