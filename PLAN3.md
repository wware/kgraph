# Implementation Plan: BioBERT / NER-Based Entity Extraction (PLAN3)

This plan replaces (or offers as an alternative to) **LLM-based entity extraction** in stage 2 with a **local, inference-only NER model** (e.g. BioBERT/PubMedBERT fine-tuned for biomedical NER). The goal is to make stage 2 "Extracting entities" **much faster** and more predictable while keeping the LLM for **relationship extraction** (stage 4), which benefits from generative reasoning.

**How to use this plan:** Work through phases in order. Phase 1 adds the NER extractor implementation and optional dependencies; Phase 2 wires it into the medlit ingest script and keeps the LLM path available; Phase 3 adds tests and documentation. **Success:** Running ingest with `--entity-extractor ner` (or equivalent) uses the NER model for entity extraction only; stage 2 completes in seconds per paper instead of minutes; relationship extraction still uses the LLM when `--use-ollama` is set.

---

## Why NER instead of LLM for entity extraction?

| Concern | LLM (current) | NER (BioBERT / token classification) |
|--------|----------------|--------------------------------------|
| **Speed** | Slow: full generation per chunk, round-trip to Ollama | Fast: single forward pass, no generation |
| **Bottleneck** | LLM calls dominate stage 2 | Embeddings and resolution dominate; NER is cheap |
| **Accuracy** | Depends on prompt; can output type-as-name | Trained on spans + labels; structured by design |
| **Where it fits** | Good for relationships (reasoning over triples) | Good for entity spans (recognize disease, gene, drug) |

The interface is unchanged: `EntityExtractorInterface.extract(document) -> list[EntityMention]`. The NER-based extractor returns the same `EntityMention` (text, entity_type, start_offset, end_offset, confidence, context, metadata) so the rest of the pipeline (resolver, promotion, relationship extraction) is unchanged.

---

## Summary of goals

| Priority | Goal |
|----------|------|
| **Required** | Implement an entity extractor that uses a local NER model (HuggingFace token classification) and returns `list[EntityMention]` with correct character offsets and medlit entity types. |
| **Required** | Map NER model labels (e.g. B-DISEASE, I-DISEASE, B-CHEMICAL, I-CHEMICAL) to medlit types (disease, drug, gene, etc.) and filter out unsupported types. |
| **Required** | Wire the NER extractor into the medlit ingest script so it can be selected via a flag (e.g. `--entity-extractor ner`) while keeping the existing LLM extractor as default or `--entity-extractor llm`. |
| **Required** | Keep dependencies optional: NER/transformers/torch only required when using the NER extractor; no change to kgraph core or kgschema. |
| **Optional** | Support multiple NER models (e.g. one for disease/chemical, one for gene) or a single multi-type model; start with one model that covers at least disease and chemical (drug). |

---

## Phase 1: NER-based entity extractor implementation

**Owner:** `examples/medlit` (optional dependency scope).  
**Location:** New module under `examples/medlit/pipeline/` (e.g. `ner_extractor.py` or `biobert_ner.py`).

### 1.1 Dependencies

- Add to **`examples/medlit`** (or a dedicated `examples/medlit/ner` extra) optional dependencies: `transformers`, `torch`, `tokenizers`. Do **not** add these to the root `kgraph` or `kgschema` so that users who only use LLM extraction do not need to install them.
- In the NER module, import `transformers` and use it only when the NER extractor is instantiated; catch `ImportError` and raise a clear error if the user selects NER without having installed the extra.

### 1.2 Model choice (actionable)

- Use a HuggingFace **token classification** (NER) model that outputs **BIO/BIOES** (or similar) labels for at least **disease** and **chemical** (to map to medlit `disease` and `drug`). Examples:
  - **`tner/roberta-large-bc5cdr`**: BC5CDR fine-tuned; labels typically include chemical and disease.
  - **`dslim/bert-base-NER`**: General NER; can be swapped later for a biomedical-specific model.
  - Prefer a model that has a **pipeline** or clear **label2id** so you can map model labels to medlit types without guessing.
- Document the chosen model and label mapping in the module docstring and in PLAN3. If the first model does not support genes, document that "gene" entities may be under-covered in NER-only mode and can be added in a follow-up (second model or hybrid).

### 1.3 Implement `EntityExtractorInterface`

- **Class name:** e.g. `MedLitNEREntityExtractor` or `BioBERTEntityExtractor`.
- **Constructor:** Accept at least:
  - `model_name_or_path: str` (HuggingFace model id or local path).
  - `domain: DomainSchema | None` (for medlit entity types and validation); optional default to medlit domain.
  - Optional `device: str | None` (e.g. `"cuda"`, `"cpu"`, `None` for auto).
  - Optional `max_length: int` for tokenizer (e.g. 512).
- **Load model once in `__init__`:** Use `transformers.AutoModelForTokenClassification.from_pretrained(...)` and `AutoTokenizer.from_pretrained(...)`. Store pipeline or model + tokenizer for reuse in `extract()`.
- **`async def extract(self, document: BaseDocument) -> list[EntityMention]`:**
  1. Get document text: prefer `document.content`; if the document has `abstract` and no content, use abstract (same as current LLM path).
  2. If text is empty or too short (e.g. &lt; 10 chars), return `[]`.
  3. Run NER: tokenize text, run model forward, decode predictions to spans (aggregate subword tokens into contiguous spans; map token positions back to **character offsets** using the tokenizer’s `offset_mapping`).
  4. For each span: (start_offset, end_offset, label, confidence). Use softmax on logits and take max prob as confidence; if the pipeline returns scores, use those.
  5. **Label mapping:** Map model labels (e.g. `B-DISEASE`, `I-DISEASE`, `B-CHEMICAL`, `I-CHEMICAL`, or raw tag names) to medlit `entity_type` strings: `disease`, `drug`, `gene`, `protein`, etc. Use a fixed dict (e.g. `LABEL_TO_MEDLIT_TYPE`) and normalize (strip B-/I-, lowercase). If a label has no mapping, skip that span or map to a default only if you document it.
  6. **Type filter:** If `domain` is provided, only keep mentions whose `entity_type` is in `domain.entity_types`. Otherwise keep all mapped types.
  7. **Reject type-as-name:** Reuse the same filter as in `mentions.py`: if the span text (after strip) is a known type label or equals the entity type, skip it (no `EntityMention`).
  8. Build `EntityMention(text=span_text, entity_type=..., start_offset=..., end_offset=..., confidence=..., context=optional_surrounding_text, metadata={"extraction_method": "ner", "model": model_name})`.
  9. Return the list of mentions.

### 1.4 Character offset handling

- Tokenizers often return subword pieces. Use the tokenizer’s **`return_offsets_mapping=True`** (or equivalent) to get (start, end) character offsets per token. Aggregate consecutive tokens that have the same predicted label (after stripping B-/I-) into one span; use the first token’s start and the last token’s end as `start_offset` and `end_offset`. Extract `text = document.content[start_offset:end_offset]` (or from the aggregated token slice) so that `EntityMention.text` is exactly the span from the document.
- Handle documents longer than `max_length`: split into overlapping or non-overlapping windows, run NER on each window, merge spans and **adjust offsets** so they are relative to the full document. Deduplicate overlapping spans (e.g. keep higher confidence or merge). This avoids silent truncation.

### 1.5 Sync vs async

- `EntityExtractorInterface.extract` is `async`. The NER forward pass is CPU/GPU bound. Implement `extract` as `async` and run the actual model inference in an executor (e.g. `asyncio.to_thread(run_ner, text)`) so the event loop is not blocked. Alternatively use a sync call inside the async method if the pipeline is fast enough; keep the method signature `async def extract(...)`.

---

## Phase 2: Wire NER extractor into ingest script

**Owner:** `examples/medlit/scripts/ingest.py` and config.

### 2.1 CLI / config

- Add a way to select the entity extractor. Options (choose one and document):
  - **CLI flag:** e.g. `--entity-extractor llm` (default) and `--entity-extractor ner`. When `ner`, do not require `--use-ollama` for entity extraction; still require it for relationship extraction if the user wants stage 4 to run.
  - **Config file:** e.g. in medlit config YAML, `entity_extractor: llm | ner`. Script reads it and instantiates the chosen extractor.
- If `--entity-extractor ner` (or config `ner`):
  - Do **not** raise "LLM extraction is required for entity extraction from XML". Instead, create `MedLitNEREntityExtractor(...)` (or the class name you chose) with a default or configurable model name.
  - Optionally allow `--ner-model` to override the default model id.
- If `--entity-extractor llm` or default:
  - Keep current behavior: require `use_ollama`, create `MedLitEntityExtractor(llm_client=..., domain=domain)`.

### 2.2 Single orchestrator path

- The orchestrator already accepts `entity_extractor: EntityExtractorInterface`. So:
  - When NER is selected: `entity_extractor = MedLitNEREntityExtractor(model_name_or_path=..., domain=domain)` (and optional device/max_length).
  - When LLM is selected: `entity_extractor = MedLitEntityExtractor(llm_client=llm_client, domain=domain)`.
- Pass the same `entity_extractor` into `IngestionOrchestrator` and into `BatchingEntityExtractor(base_extractor=entity_extractor, ...)`. No change to the orchestrator or batching logic; they already work with any `EntityExtractorInterface`.

### 2.3 Relationship extraction unchanged

- Relationship extraction (stage 4) continues to use `MedLitRelationshipExtractor` with `llm_client`. So when the user runs with `--entity-extractor ner --use-ollama`, stage 2 uses NER and stage 4 uses the LLM. When the user runs with `--entity-extractor ner` and **no** `--use-ollama`, stage 2 uses NER and stage 4 can be skipped or disabled (document that relationships require LLM).

### 2.4 Startup messages

- Print a clear message at startup: e.g. "Using NER-based entity extraction (model: ...)" or "Using LLM-based entity extraction..." so the user can confirm which path is active.

---

## Phase 3: Tests and documentation

### 3.1 Unit tests

- **File:** e.g. `examples/medlit/tests/test_ner_extractor.py`.
- **Tests:**
  1. **Label mapping:** Test that the extractor’s internal label-to-medlit mapping produces the expected `entity_type` for known model labels (mock or use a tiny model if needed).
  2. **Offset and text:** Feed a short document (e.g. "Patient with diabetes took aspirin."); assert that returned mentions have `start_offset`/`end_offset` consistent with `document.content[start:end] == mention.text`, and that `entity_type` is one of the domain types.
  3. **Type-as-name filter:** Ensure that if the model outputs a span with text "disease" or "gene", that mention is dropped (same behavior as in `mentions.py`).
  4. **Empty/short text:** Document with empty or very short content returns `[]`.
- **Optional:** Integration test that loads the real model and runs on one short paragraph; mark as slow or use a small model so CI stays fast.

### 3.2 Documentation

- **README or docstring:** In the NER module and in `examples/medlit/README.md`, document:
  - How to install the NER extra (e.g. `pip install -e ".[ner]"` or similar).
  - How to run ingest with NER: `--entity-extractor ner` (and optional `--ner-model`).
  - Which model is used by default and what entity types it supports (e.g. disease, drug); note if gene/protein require a different model or hybrid.
- **PLAN3.md:** Add a short "Implemented" or "Done" section at the top after implementation, with the chosen model name and any deviations from this plan.

---

## Code reference (minimal)

- **Interface:** `kgraph/pipeline/interfaces.py` — `EntityExtractorInterface`, method `async def extract(self, document: BaseDocument) -> list[EntityMention]`.
- **EntityMention:** `kgschema/entity.py` — `text`, `entity_type`, `start_offset`, `end_offset`, `confidence`, `context`, `metadata` (all required except context/metadata with defaults).
- **Medlit entity types:** `examples/medlit/domain.py` — `MedLitDomainSchema.entity_types`: disease, gene, drug, protein, symptom, procedure, biomarker, pathway, location, ethnicity.
- **Type-as-name filter:** `examples/medlit/pipeline/mentions.py` — `_is_type_masquerading_as_name(name, entity_type)` and `KNOWN_TYPE_LABELS`; reuse in NER extractor so NER output is filtered the same way.
- **Ingest script entity extractor creation:** `examples/medlit/scripts/ingest.py` — around lines 257–263: currently requires `llm_client` and builds `MedLitEntityExtractor`. Add a branch for NER when `--entity-extractor ner` (or config) is set.
- **BatchingEntityExtractor:** `kgraph/pipeline/streaming.py` — `BatchingEntityExtractor(base_extractor=...)`; works with any `EntityExtractorInterface`, including the new NER extractor.

---

## Out of scope (future)

- **Hybrid NER + LLM:** Run NER first, then optionally run LLM only on low-confidence or missing types; not required for PLAN3.
- **Multiple NER models:** One for disease/chemical, one for gene; can be added later by composing or chaining extractors.
- **Entity-level embedding cache for NER path:** Same as PLAN2; no change to embedding cache behavior in this plan.

---

## Success criteria

1. With `--entity-extractor ner` and the NER extra installed, stage 2 "Extracting entities" runs using the local NER model and completes in seconds per paper (order of magnitude faster than LLM).
2. Output entities have correct `entity_type` (disease, drug, etc.) and character-aligned `start_offset`/`end_offset` and `text`.
3. With `--entity-extractor llm --use-ollama`, behavior is unchanged from today (LLM-based entity extraction).
4. Relationship extraction (stage 4) still uses the LLM when `--use-ollama` is set, regardless of entity extractor choice.
5. Tests and README/documentation are updated so the plan is actionable and verifiable by you or another implementer without supervision.
