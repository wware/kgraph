# Plan 8b: Pass 3 — Bundle builder (medlit_merged → kgbundle)

Execute steps in order from the **repository root**. Reference: **kgbundle/kgbundle/models.py** (EntityRow, RelationshipRow, EvidenceRow, MentionRow, DocAssetRow, BundleManifestV1, BundleFile).

**Goal:** The two-pass pipeline produces `medlit_merged/` (entities.json + relationships.json + synonym_cache.json) but never the **kgbundle** format that kgserver loads. Pass 3 reads merged output plus raw Pass 1 bundles and writes a loadable bundle (entities.jsonl, relationships.jsonl, evidence.jsonl, mentions.jsonl, manifest.json, etc.). Pass 2 currently drops evidence_entities; Pass 3 must re-read Pass 1 bundles for evidence and mentions and for usage/total_mentions.

**Id map decision:** Pass 2 will write an **id_map** file so Pass 3 can resolve (paper_id, local_id) → merge_key without re-running merge logic. Schema: `{"<paper_id>": {"<local_id>": "<merge_key>", ...}, ...}` written as `merged_dir/id_map.json`.

---

## Step 0. Pre-flight

From repo root:

```bash
./lint.sh 2>&1 | tail -30
uv run pytest examples/medlit/tests/ tests/test_medlit_domain.py -v --tb=short 2>&1 | tail -50
```

Note any failures. After each step below, re-run and fix regressions.

---

## Step 1. Pass 2: write id_map.json

**File:** `examples/medlit/pipeline/dedup.py`

In `_run_pass2_impl`, after all entities have been assigned merge keys (after the "3) Assign merge keys" loop), write the **local_to_canonical** map to disk so Pass 3 can resolve (paper_id, local_id) → merge_key.

- **Path:** `output_dir / "id_map.json"`.
- **Format:** A single JSON object: keys = paper_id (string), values = object with keys = local entity id (e.g. `e01`, `d01`), values = merge_key (string). Example: `{"PMC12756687": {"e01": "canon-abc123", "g01": "HGNC:1100"}, "PMC99999999": {"e01": "canon-abc123"}}`.
- **When:** Write once after the merge-key assignment loops, before building canonical_entities (so the map is complete). Use the same `local_to_canonical: dict[tuple[str, str], str]` already in scope; convert to nested dict: `by_paper: dict[str, dict[str, str]] = {}`, then `for (paper_id, local_id), merge_key in local_to_canonical.items(): by_paper.setdefault(paper_id, {})[local_id] = merge_key`, then `json.dump(by_paper, f, indent=2)`.
- **CLI:** No new flags; id_map is always written when Pass 2 runs.

**Verification:** Run Pass 2 on fixture bundles; assert `merged_dir/id_map.json` exists and has the expected paper IDs and local IDs mapping to merge keys.

---

## Step 2. Pass 3 module: load merged output and id_map

**File:** `examples/medlit/pipeline/bundle_builder.py` (new)

Create a new module that implements the bundle build logic.

- **Function:** `def load_merged_output(merged_dir: Path) -> tuple[list[dict], list[dict], dict, dict]:`
  - Read `merged_dir / "entities.json"` → list of entity dicts (each has `entity_id`, `canonical_id`, `class`, `name`, `synonyms`, `source`, `source_papers`).
  - Read `merged_dir / "relationships.json"` → list of relationship dicts (each has `subject`, `predicate`, `object`, `evidence_ids`, `source_papers`, `confidence`).
  - Read `merged_dir / "id_map.json"` → dict mapping paper_id → { local_id → merge_key }.
  - Read `merged_dir / "synonym_cache.json"` → dict (for canonical_id_cache copy later).
  - Return `(entities, relationships, id_map, synonym_cache)`. If id_map.json or synonym_cache.json is missing, raise or return empty dict as appropriate (id_map is required).
- **Function:** `def load_pass1_bundles(bundles_dir: Path) -> list[tuple[str, PerPaperBundle]]:`
  - Glob `paper_*.json`, load each with `PerPaperBundle.from_bundle_dict(json.load(...))`, paper_id = bundle.paper.pmcid or path.stem.replace("paper_", ""). Return list of (paper_id, bundle).
- **Imports:** `from examples.medlit.bundle_models import PerPaperBundle`; `from kgbundle import EntityRow, RelationshipRow, EvidenceRow, MentionRow, DocAssetRow, BundleFile, BundleManifestV1`.

---

## Step 3. Pass 3: entity usage and supporting_documents from bundles

**File:** `examples/medlit/pipeline/bundle_builder.py`

Pass 2 does not output usage_count or total_mentions. Compute them from Pass 1 bundles by scanning relationships and evidence_ids.

- **Function:** `def _entity_usage_from_bundles(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]]) -> dict[str, dict]:`
  - Returns dict keyed by **merge_key** with keys: `usage_count` (int), `total_mentions` (int), `supporting_documents` (list[str] of paper_ids), `first_seen_document` (str or None), `first_seen_section` (str or None).
  - Algorithm: For each (paper_id, bundle), for each rel in bundle.relationships, resolve rel.subject and rel.object to merge_key via id_map[paper_id][local_id]. For each merge_key, increment a mention count and add paper_id to a set. For each evidence_id in rel.evidence_ids, the relationship’s subject/object are “mentioned” in that paper (count one mention per evidence per entity). So: for each rel, for each evidence_id, count one mention for subject and one for object (after resolving to merge keys). supporting_documents = distinct paper_ids where entity appears. first_seen = first paper_id by sort order (or earliest evidence if we have timestamps); first_seen_section from first evidence’s section if available.
  - Use id_map to resolve; if (paper_id, local_id) not in id_map, skip that ref (do not fail).

---

## Step 4. Pass 3: merged entity → EntityRow

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **Function:** `def _merged_entity_to_entity_row(ent: dict, usage: dict, created_at: str) -> EntityRow:`
  - `entity_id` ← ent["entity_id"].
  - `entity_type` ← ent["class"].lower() (EntityRow expects e.g. "disease", "gene").
  - `name` ← ent.get("name").
  - `status` ← "canonical" if ent.get("canonical_id") else "provisional".
  - `confidence` ← None or a default (e.g. 0.8 for canonical, 0.5 provisional).
  - `usage_count` ← usage.get("usage_count", 0).
  - `created_at` ← created_at (ISO 8601).
  - `source` ← "medlit:llm".
  - `canonical_url` ← if ent.get("canonical_id"), call `build_canonical_url(ent["canonical_id"], ent["class"].lower())` from `examples.medlit.pipeline.canonical_urls`, else None.
  - `properties` ← e.g. {"synonyms": ent.get("synonyms", [])} or similar; do not put list in a top-level EntityRow field that expects dict.
  - `first_seen_document`, `first_seen_section` ← from usage.
  - `total_mentions` ← usage.get("total_mentions", 0).
  - `supporting_documents` ← usage.get("supporting_documents", []).
  - Return `EntityRow(...)`.

---

## Step 5. Pass 3: relationship evidence stats from bundles

**File:** `examples/medlit/pipeline/bundle_builder.py`

Merged relationships have `evidence_ids` and `source_papers` but not evidence_count, strongest_evidence_quote, evidence_confidence_avg. Compute from bundles.

- **Function:** `def _relationship_evidence_stats(merged_rels: list[dict], bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]]) -> dict[tuple[str, str, str], dict]:`
  - Key = (subject_merge, predicate, object_merge). Value = dict with `evidence_count` (int), `strongest_evidence_quote` (str or None), `evidence_confidence_avg` (float or None).
  - For each merged rel, key = (rel["subject"], rel["predicate"], rel["object"]). To gather evidence: (1) Scan all bundles; for each bundle rel, resolve rel.subject and rel.object via id_map[paper_id][local_id]. If (sub_merge, rel.predicate, obj_merge) == key, collect rel.evidence_ids. (2) For each evidence_id, find the EvidenceEntityRow in the same bundle (ev.id == evidence_id). Build list of (ev.text, ev.confidence). Then evidence_count = len(list); strongest_evidence_quote = text from the pair with max confidence; evidence_confidence_avg = mean(confidence). If no evidence found, return 0, None, None for that key. Return dict mapping key → {evidence_count, strongest_evidence_quote, evidence_confidence_avg}.

---

## Step 6. Pass 3: merged relationship → RelationshipRow

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **Function:** `def _merged_rel_to_relationship_row(rel: dict, stats: dict, created_at: str) -> RelationshipRow:`
  - subject_id ← rel["subject"], object_id ← rel["object"], predicate ← rel["predicate"].
  - confidence ← rel.get("confidence").
  - source_documents ← rel.get("source_papers", []).
  - created_at ← created_at.
  - properties ← rel.get("properties", {}).
  - key = (rel["subject"], rel["predicate"], rel["object"]); evidence_count, strongest_evidence_quote, evidence_confidence_avg ← stats.get(key, {}).

---

## Step 7. Pass 3: evidence.jsonl from bundles

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **Function:** `def _build_evidence_rows(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]], merged_relationships: list[dict]) -> list[EvidenceRow]:`
  - For each (paper_id, bundle), for each rel in bundle.relationships, resolve rel.subject and rel.object to merge_key via id_map. relationship_key = f"{sub_merge}:{rel.predicate}:{obj_merge}".
  - For each evidence_id in rel.evidence_ids, find the EvidenceEntityRow in bundle.evidence_entities with ev.id == evidence_id. Build EvidenceRow(relationship_key=..., document_id=paper_id, section=ev entity’s section if any, start_offset=0, end_offset=len(ev.text) if ev.text else 0, text_span=ev.text or "", confidence=ev.confidence, supports=True). Append to list.
  - **Stub:** start_offset=0, end_offset=len(text_span) (EvidenceEntityRow has no character offsets).
  - Return list of EvidenceRow; caller will write one JSON line per row.

---

## Step 8. Pass 3: mentions.jsonl from bundles

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **Function:** `def _build_mention_rows(bundles: list[tuple[str, PerPaperBundle]], id_map: dict[str, dict[str, str]], created_at: str) -> list[MentionRow]:`
  - For each (paper_id, bundle), for each rel in bundle.relationships, resolve subject and object to merge_key. For each evidence_id in rel.evidence_ids, look up EvidenceEntityRow. Emit one MentionRow for the **subject** and one for the **object** (so each evidence span produces two mentions, one per entity). entity_id = merge_key; document_id = paper_id; section = from evidence entity if available; start_offset=0, end_offset=len(text_span); text_span = ev.text; confidence = ev.confidence; extraction_method = ev.extraction_method; created_at = created_at.
  - **Stub:** start_offset=0, end_offset=len(text_span). Add a one-line comment in code: "EvidenceEntityRow has no character offsets; stub 0 and len(text_span)."
  - Dedupe optional: if multiple evidence spans for same (entity_id, document_id), can keep one or all; keeping one per (entity_id, document_id, evidence_id) is fine.

---

## Step 9. Pass 3: doc_assets and manifest

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **doc_assets.jsonl:** DocAssetRow is for static docs (e.g. README). Emit a single row: path="docs/README.md", content_type="text/markdown". Do not put source papers here (papers are not doc assets per kgbundle).
- **manifest:** Build BundleManifestV1: bundle_version="v1", bundle_id=uuid.uuid4().hex, domain="medlit", label="medical-literature", created_at=now ISO 8601, entities=BundleFile(path="entities.jsonl", format="jsonl"), relationships=BundleFile(path="relationships.jsonl", format="jsonl"), doc_assets=BundleFile(path="doc_assets.jsonl", format="jsonl"), mentions=BundleFile(path="mentions.jsonl", format="jsonl"), evidence=BundleFile(path="evidence.jsonl", format="jsonl"), metadata={"entity_count": N, "relationship_count": M, "description": "Knowledge graph bundle from two-pass medlit pipeline"}).
- **canonical_id_cache.json:** Copy merged_dir/synonym_cache.json to output_dir/canonical_id_cache.json (same format, or document if server expects a different shape).
- **docs/README.md:** Create output_dir/docs, write a short README.md (e.g. "Medlit bundle built from Pass 1 + Pass 2 output.").

---

## Step 10. Pass 3: run_pass3 and write all files

**File:** `examples/medlit/pipeline/bundle_builder.py`

- **Function:** `def run_pass3(merged_dir: Path, bundles_dir: Path, output_dir: Path) -> dict[str, Any]:`
  - Load merged output and id_map (Step 2). If id_map.json missing, raise FileNotFoundError with a clear message.
  - Load Pass 1 bundles (Step 2).
  - created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z").
  - Compute entity usage from bundles (Step 3).
  - Compute relationship evidence stats (Step 5).
  - Build list of EntityRow (Step 4), RelationshipRow (Step 6), EvidenceRow (Step 7), MentionRow (Step 8).
  - output_dir.mkdir(parents=True, exist_ok=True). Create output_dir/docs.
  - Write entities.jsonl: one line per EntityRow (model_dump_json() + newline).
  - Write relationships.jsonl: one line per RelationshipRow.
  - Write evidence.jsonl: one line per EvidenceRow.
  - Write mentions.jsonl: one line per MentionRow.
  - Write doc_assets.jsonl: one line for docs/README.md.
  - Write manifest.json: BundleManifestV1.model_dump_json(indent=2).
  - Copy merged_dir/synonym_cache.json to output_dir/canonical_id_cache.json.
  - Write output_dir/docs/README.md with short static content.
  - Return summary dict (entity_count, relationship_count, evidence_count, mention_count, manifest path).

---

## Step 11. Pass 3 CLI script

**File:** `examples/medlit/scripts/pass3_build_bundle.py` (new)

- Shebang and docstring: "Pass 3: Build kgbundle from medlit_merged and pass1_bundles. Reads merged_dir (entities.json, relationships.json, id_map.json, synonym_cache.json) and bundles_dir (paper_*.json), writes output_dir in kgbundle format for kgserver."
- argparse: --merged-dir (required), --bundles-dir (required), --output-dir (default Path("medlit_bundle")).
- Add repo root to sys.path; load .env if python-dotenv available (same pattern as pass1_extract).
- Call run_pass3(merged_dir, bundles_dir, output_dir). Print summary to stderr; exit 0 on success.
- If id_map.json is missing, print clear error and exit 1.

---

## Step 12. Tests

**File:** `examples/medlit/tests/test_pass3_bundle_builder.py` (new)

- **Fixture:** Build a minimal merged dir: entities.json (one entity with entity_id and canonical_id), relationships.json (one row), id_map.json (one paper, two local ids → merge keys), synonym_cache.json (can be empty dict).
- **Fixture:** Build a minimal bundles_dir: one paper_*.json with one relationship and one evidence_entity (id matching the relationship’s evidence_ids).
- **Test:** run_pass3(merged_dir, bundles_dir, tmp_path). Assert output_dir/entities.jsonl exists, one line; output_dir/relationships.jsonl exists, one line; output_dir/evidence.jsonl exists; output_dir/mentions.jsonl exists; output_dir/manifest.json exists and validates as BundleManifestV1; manifest.entities.path == "entities.jsonl".
- **Test:** EntityRow in entities.jsonl has entity_id, status ("canonical" or "provisional"), usage_count/total_mentions populated from bundle scan.
- **Test:** EvidenceRow relationship_key uses merge keys (from id_map).
- **Test:** If id_map.json is missing in merged_dir, run_pass3 raises or script exits non-zero with clear message.

---

## Step 13. Docs

**File:** `examples/medlit/INGESTION.md`

- Add "Pass 3: Build kgbundle" section: input = merged_dir (output of Pass 2) and bundles_dir (output of Pass 1); output = kgbundle directory (entities.jsonl, relationships.jsonl, evidence.jsonl, mentions.jsonl, manifest.json, etc.) loadable by kgserver. Command: `python -m examples.medlit.scripts.pass3_build_bundle --merged-dir medlit_merged --bundles-dir pass1_bundles --output-dir medlit_bundle`. Pass 2 must have been run so that merged_dir contains id_map.json.
- Update one-line usage to show three passes: Pass 1 → Pass 2 → Pass 3 → medlit_bundle.

---

## Step 14. Verification

From repo root:

```bash
uv run pytest examples/medlit/tests/test_pass3_bundle_builder.py examples/medlit/tests/test_two_pass_ingestion.py -v --tb=short
# Full pipeline: Pass 1 (fixtures) → Pass 2 → Pass 3, then validate manifest and row counts
uv run pytest examples/medlit/tests/ -v --tb=short 2>&1 | tail -30
./lint.sh 2>&1 | tail -35
```

---

## Summary checklist

- [ ] Step 0: Pre-flight
- [ ] Step 1: Pass 2 writes id_map.json (merged_dir)
- [ ] Step 2: bundle_builder.py load_merged_output, load_pass1_bundles
- [ ] Step 3: _entity_usage_from_bundles
- [ ] Step 4: _merged_entity_to_entity_row (EntityRow)
- [ ] Step 5: _relationship_evidence_stats from bundles
- [ ] Step 6: _merged_rel_to_relationship_row (RelationshipRow)
- [ ] Step 7: _build_evidence_rows (EvidenceRow; stub offsets)
- [ ] Step 8: _build_mention_rows (MentionRow; stub offsets)
- [ ] Step 9: doc_assets (README), manifest, canonical_id_cache, docs/README.md
- [ ] Step 10: run_pass3 and write all files
- [ ] Step 11: pass3_build_bundle.py CLI
- [ ] Step 12: test_pass3_bundle_builder.py
- [ ] Step 13: INGESTION.md Pass 3 section
- [ ] Step 14: Verification and lint

---

## Non-obvious details (reference)

- **EntityRow.entity_type:** Use merged entity "class" lowercased (e.g. Disease → "disease").
- **MentionRow / EvidenceRow offsets:** EvidenceEntityRow has text_span_id and text but no character offsets; use start_offset=0, end_offset=len(text_span).
- **Evidence relationship_key:** Must use merge keys (subject_id:predicate:object_id); resolve via id_map when building from per-paper relationships.
- **doc_assets:** Only static docs (e.g. docs/README.md). Paper metadata goes in manifest.metadata (paper_count, paper_ids) if desired; not in DocAssetRow.

---

## Summary of what was implemented

- **Step 1 (id_map.json):** In `dedup.py`, after building `canonical_entities` and `triple_to_rel`, a `by_paper` dict is built from `local_to_canonical` and written to `output_dir / "id_map.json"` in the same block as entities.json and relationships.json (step 9). Test `test_pass2_writes_id_map` in `test_two_pass_ingestion.py` asserts the file exists and has the expected nested structure.

- **Pass 3 module:** New file `examples/medlit/pipeline/bundle_builder.py` implements `load_merged_output`, `load_pass1_bundles`, `_entity_usage_from_bundles`, `_merged_entity_to_entity_row`, `_relationship_evidence_stats`, `_merged_rel_to_relationship_row`, `_build_evidence_rows`, `_build_mention_rows`, and `run_pass3`. All kgbundle row types use the Pydantic models from `kgbundle`; `build_canonical_url` is imported from `examples.medlit.pipeline.canonical_urls`.

- **Entity usage:** `_entity_usage_from_bundles` scans each bundle’s relationships and evidence_ids, resolves subject/object to merge_key via id_map, and for each evidence_id counts one mention for subject and one for object. `usage_count` is set to `len(supporting_documents)`. `EvidenceEntityRow` has no section field, so `first_seen_section` is always None.

- **Evidence/mention offsets:** EvidenceRow and MentionRow use `start_offset=0`, `end_offset=len(text_span)`; in-code comments note that EvidenceEntityRow has no character offsets.

- **Evidence relationship_key:** Built as `f"{sub_merge}:{rel.predicate}:{obj_merge}"`; only bundle relationships whose (sub_merge, predicate, obj_merge) appears in the merged relationship list are emitted (so evidence rows align with merged relationships).

- **CLI:** `examples/medlit/scripts/pass3_build_bundle.py` takes `--merged-dir`, `--bundles-dir`, `--output-dir` (default `medlit_bundle`), checks for id_map.json before calling `run_pass3`, and exits 1 with a clear message if it is missing.

- **Tests:** `examples/medlit/tests/test_pass3_bundle_builder.py` adds fixtures for a minimal merged dir (two entities, one relationship, id_map, synonym_cache) and a minimal bundles dir (one paper with one relationship and one evidence_entity), plus tests for run_pass3 output files, EntityRow usage/status, EvidenceRow relationship_key using merge keys, and FileNotFoundError when id_map.json is missing.

- **Docs:** `examples/medlit/INGESTION.md` updated with a Pass 3 section and three-pass one-line usage (Pass 1 → Pass 2 → Pass 3 → medlit_bundle).

- **Verification:** Full medlit test suite and Pass 2 → Pass 3 pipeline (fixture bundles) pass; CLI correctly fails when given a directory without id_map.json and succeeds when given Pass 2 output plus bundles.
