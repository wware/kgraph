# MedLit ingestion (three passes)

Ingestion is split into three passes. **Pass 1** produces immutable per-paper bundle JSON files. **Pass 2** reads those bundles and writes a merged graph (entities, relationships, id_map, synonym cache) to a separate directory; it never modifies the Pass 1 files. **Pass 3** reads the merged output and Pass 1 bundles and writes a **kgbundle** directory (entities.jsonl, relationships.jsonl, evidence.jsonl, mentions.jsonl, manifest.json, etc.) loadable by kgserver.

The two-pass flow does **not** use promotion (no usage/confidence thresholds, no `PromotionPolicy`). Canonical vs provisional is reflected only by whether an entity has an authoritative `canonical_id` in the Pass 2 output (present) or `canonical_id` null (provisional in that sense).

## Pass 1: LLM extraction → per-paper bundle

- **Input:** Directory of paper files (JATS XML or JSON).
- **Output:** One JSON file per paper in an output directory (e.g. `paper_PMC12345.json`). Each file is a **per-paper bundle** (paper metadata, entities, evidence_entities, relationships, notes, extraction_provenance).
- **Immutable:** These files are the single source of truth for that paper’s extraction. Pass 2 does not overwrite them.

**Run:**

```bash
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic
```

Pass 1 requires an LLM. See **LLM_SETUP.md** for backends (Anthropic, OpenAI, Lambda Labs, Ollama) and environment variables.

## Pass 2: Deduplication and promotion

- **Input:** Directory of per-paper bundle JSON files (Pass 1 output).
- **Output:** A separate directory with `entities.json`, `relationships.json`, `id_map.json`, and `synonym_cache.json`. Original bundle files are **not** modified. Each entity has **entity_id** (stable merge key, always present) and **canonical_id** (set only when from an ontology; null otherwise). **id_map.json** maps (paper_id → local_id → merge_key) for Pass 3.
- **Synonym cache:** Pass 2 loads and saves a synonym cache so that (name, type) → merge key and SAME_AS links persist across runs (idempotent behavior).
- **Optional authority lookup:** Use `--canonical-id-cache path/to/cache.json` to resolve entities via MeSH/UMLS, HGNC, RxNorm, UniProt when possible. Use `--no-canonical-id-lookup` to disable lookups (e.g. offline).

**Run:**

```bash
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/
# With authority lookup (requires network for cache misses):
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/ --canonical-id-cache canonical_id_cache.json
```

## Pass 3: Build kgbundle

- **Input:** Pass 2 merged directory (`entities.json`, `relationships.json`, **id_map.json**, `synonym_cache.json`) and Pass 1 bundles directory (`paper_*.json`).
- **Output:** A kgbundle directory (e.g. `medlit_bundle/`) with `entities.jsonl`, `relationships.jsonl`, `evidence.jsonl`, `mentions.jsonl`, `manifest.json`, `doc_assets.jsonl`, `docs/README.md`, `canonical_id_cache.json`, in the format kgserver expects.
- **Requirement:** Pass 2 must have been run so that the merged directory contains **id_map.json**. If it is missing, Pass 3 exits with an error.

**Run:**

```bash
python -m examples.medlit.scripts.pass3_build_bundle --merged-dir pass2_merged/ --bundles-dir pass1_bundles/ --output-dir medlit_bundle
```

## One-line usage

```bash
# Pass 1 (requires LLM backend and API key or Ollama)
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic

# Pass 2 (no LLM; reads bundles, writes merged graph + id_map)
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/

# Pass 3 (build kgbundle for kgserver)
python -m examples.medlit.scripts.pass3_build_bundle --merged-dir pass2_merged/ --bundles-dir pass1_bundles/ --output-dir medlit_bundle
```

## Adding more papers to an existing bundle

You can add new papers **only if you still have the intermediate dirs** from the first run: `pass1_bundles/` (per-paper JSONs) and the Pass 2 merged dir (e.g. `medlit_merged/` or `pass2_merged/`). The final bundle dir (`medlit_bundle/`) alone is not enough—Pass 2 needs all per-paper bundles and the synonym cache to re-merge.

1. **Run Pass 1 for the new papers only**, writing into your **existing** Pass 1 dir. Pass 1 skips any paper that already has a `paper_<id>.json` file, so only new papers are extracted.

   ```bash
   # Same --output-dir as before; --papers lists only the new PMC IDs or filenames
   uv run python -m examples.medlit.scripts.pass1_extract \
     --input-dir examples/medlit/pmc_xmls \
     --output-dir pass1_bundles \
     --llm-backend anthropic \
     --papers "PMC12345678.xml,PMC87654321.xml"
   ```

2. **Re-run Pass 2** on the full `pass1_bundles/` dir, with the **same** merged output dir and synonym cache so the cache is reused and entity merge keys stay consistent.

   ```bash
   uv run python -m examples.medlit.scripts.pass2_dedup \
     --bundle-dir pass1_bundles \
     --output-dir medlit_merged \
     --synonym-cache medlit_merged/synonym_cache.json
   # Optional: add --canonical-id-cache medlit_bundle/canonical_id_cache.json if you use authority lookup
   ```

3. **Re-run Pass 3** to rebuild the final kgbundle (old + new papers).

   ```bash
   uv run python -m examples.medlit.scripts.pass3_build_bundle \
     --merged-dir medlit_merged \
     --bundles-dir pass1_bundles \
     --output-dir medlit_bundle
   ```

If you no longer have `pass1_bundles/` and the merged dir, you must re-run the full pipeline (Pass 1 on all papers, then Pass 2, then Pass 3) to get a bundle that includes the new papers.

## Review GUI (out of scope)

The spec (INGESTION_REFACTOR.md) describes a **review GUI** for unreviewed SAME_AS links (`predicate="SAME_AS"`, `resolution=null`). That GUI is **not** implemented in this plan. Low- and medium-confidence SAME_AS links are preserved in the graph and in the synonym cache for future review tooling.
