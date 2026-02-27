# MedLit ingestion (four passes)

Ingestion is split into four passes. **Pass 1a** runs a fast vocabulary extraction across all papers and writes `vocab.json` plus a **seeded synonym cache** (`seeded_synonym_cache.json`) so that Pass 1b and Pass 2 use consistent entity names and types. **Pass 1b** (Pass 1) produces immutable per-paper bundle JSON files, optionally using the vocabulary as context. **Pass 2** reads those bundles and writes a merged graph (entities, relationships, id_map, synonym cache); it loads the seeded cache when present so cross-paper entities merge correctly. **Pass 3** reads the merged output and Pass 1 bundles and writes a **kgbundle** directory loadable by kgserver.

The flow does **not** use promotion (no usage/confidence thresholds, no `PromotionPolicy`). Canonical vs provisional is reflected only by whether an entity has an authoritative `canonical_id` in the Pass 2 output (present) or `canonical_id` null (provisional in that sense).

## Pass 1a: Vocabulary extraction (optional but recommended)

- **Input:** Same directory of paper files as Pass 1b (JATS XML or JSON).
- **Output:** `pass1_vocab/vocab.json` (list of entities with name, type, abbreviations, umls_id, source_papers, umls_type_validated) and `pass1_vocab/seeded_synonym_cache.json` in the same format Pass 2 expects for `--synonym-cache`. If `vocab.json` already exists, new extractions are merged in (same name+type → add to source_papers).
- **Purpose:** Reduces cross-paper duplication and type inconsistency; UMLS type validation corrects misclassifications (e.g. cortisol as gene → drug/biomarker).

**Run:**

```bash
python -m examples.medlit.scripts.pass1a_vocab --input-dir pmc_xmls/ --output-dir pass1_vocab/ --llm-backend anthropic
```

## Pass 1b: LLM extraction → per-paper bundle

- **Input:** Directory of paper files (JATS XML or JSON). Optionally `--vocab-file pass1_vocab/vocab.json` so the extraction prompt includes the shared vocabulary and entity types are normalized.
- **Output:** One JSON file per paper in an output directory (e.g. `paper_PMC12345.json`). Each file is a **per-paper bundle** (paper metadata, entities, evidence_entities, relationships, notes, extraction_provenance).
- **Immutable:** These files are the single source of truth for that paper’s extraction. Pass 2 does not overwrite them.

**Run:**

```bash
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic
# With vocabulary context (after Pass 1a):
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic --vocab-file pass1_vocab/vocab.json
```

Pass 1 requires an LLM. See **LLM_SETUP.md** for backends (Anthropic, OpenAI, Lambda Labs, Ollama) and environment variables.

## Pass 2: Deduplication and promotion

- **Input:** Directory of per-paper bundle JSON files (Pass 1 output).
- **Output:** A separate directory with `entities.json`, `relationships.json`, `id_map.json`, and `synonym_cache.json`. Original bundle files are **not** modified. Each entity has **entity_id** (stable merge key, always present) and **canonical_id** (set only when from an ontology; null otherwise). **id_map.json** maps (paper_id → local_id → merge_key) for Pass 3.
- **Synonym cache:** Pass 2 loads and saves a synonym cache so that (name, type) → merge key and SAME_AS links persist across runs (idempotent behavior). When using Pass 1a, pass `--synonym-cache pass1_vocab/seeded_synonym_cache.json` so Pass 2 starts from the seeded vocabulary; it will write back to that path and update it with new merges.
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
# Pass 1a (optional; recommended for cross-paper consistency)
python -m examples.medlit.scripts.pass1a_vocab --input-dir pmc_xmls/ --output-dir pass1_vocab/ --llm-backend anthropic

# Pass 1b (requires LLM backend and API key or Ollama)
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic --vocab-file pass1_vocab/vocab.json

# Pass 2 (no LLM; reads bundles, writes merged graph + id_map; use seeded cache when Pass 1a was run)
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/ --synonym-cache pass1_vocab/seeded_synonym_cache.json

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

## Future work: evidence validation (optional)

The current pipeline does not validate that a relationship’s cited evidence text actually mentions (or is semantically similar to) the subject and object entities. An optional post-processing step could run after the Pass 1 LLM response: for each relationship, resolve its `evidence_ids` to evidence text, check that subject and object names (or synonyms) appear in that text—or that an embedding-based similarity is above a threshold—and drop or flag relationships that fail. That would reduce spurious relationships when the LLM cites evidence that doesn’t support the triple. Implementation would live in or alongside pass1_extract (or a small module called from it), not in the obsolete `MedLitRelationshipExtractor`.
