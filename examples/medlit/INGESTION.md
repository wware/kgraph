# MedLit two-pass ingestion

Ingestion is split into two passes. **Pass 1** produces immutable per-paper bundle JSON files. **Pass 2** reads those bundles and writes a merged graph (and synonym cache) to a separate directory; it never modifies the Pass 1 files.

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
- **Output:** A separate directory with `entities.json`, `relationships.json`, and `synonym_cache.json`. Original bundle files are **not** modified. Each entity has **entity_id** (stable merge key, always present) and **canonical_id** (set only when from an ontology; null otherwise).
- **Synonym cache:** Pass 2 loads and saves a synonym cache so that (name, type) → merge key and SAME_AS links persist across runs (idempotent behavior).
- **Optional authority lookup:** Use `--canonical-id-cache path/to/cache.json` to resolve entities via MeSH/UMLS, HGNC, RxNorm, UniProt when possible. Use `--no-canonical-id-lookup` to disable lookups (e.g. offline).

**Run:**

```bash
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/
# With authority lookup (requires network for cache misses):
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/ --canonical-id-cache canonical_id_cache.json
```

## One-line usage

```bash
# Pass 1 (requires LLM backend and API key or Ollama)
python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir pass1_bundles/ --llm-backend anthropic

# Pass 2 (no LLM; reads bundles, writes merged graph)
python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir pass2_merged/
```

## Review GUI (out of scope)

The spec (INGESTION_REFACTOR.md) describes a **review GUI** for unreviewed SAME_AS links (`predicate="SAME_AS"`, `resolution=null`). That GUI is **not** implemented in this plan. Low- and medium-confidence SAME_AS links are preserved in the graph and in the synonym cache for future review tooling.
