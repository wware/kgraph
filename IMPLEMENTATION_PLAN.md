# Implementation Plan: Wire Streaming Abstractions with PMC-Specific Chunker

This plan describes how to use the existing streaming abstractions (`DocumentChunkerInterface`, `StreamingEntityExtractorInterface`, `BatchingEntityExtractor`, `StreamingRelationshipExtractorInterface`, `WindowedRelationshipExtractor`) in the main ingestion path, and how to add a **PMC-specific chunker** that produces `DocumentChunk`s from raw XML bytes using `iter_pmc_windows` (memory-efficient, no full-document parse).

---

## 1. Goals

- **Single streaming path in the orchestrator**: Chunk → entity `extract_streaming` → (parse & store document) → relationship `extract_windowed`, instead of extractor-specific logic inside `extract()`.
- **Memory-efficient PMC**: Chunk from raw bytes via `iter_pmc_windows` so we never load the full XML tree for entity/relationship extraction.
- **Reuse**: Other domains can use `WindowedDocumentChunker(document.content)` and the same streaming extractors; only PMC needs a raw-byte chunker.
- **MedLit**: Use `BatchingEntityExtractor(MedLitEntityExtractor)` and `WindowedRelationshipExtractor(MedLitRelationshipExtractor)` with the PMC chunker; simplify `MedLitEntityExtractor` by removing its internal `iter_pmc_windows` path once wired.

---

## 2. Chunking from Raw Bytes

The current `DocumentChunkerInterface` is:

```python
async def chunk(self, document: BaseDocument) -> list[DocumentChunk]:
```

That requires a parsed document (and for PMC, parsing loads the full tree). To support memory-efficient PMC, we need a way to get chunks from **raw bytes + content type** without parsing the body.

### 2.1 Option A: Extend the existing interface (recommended)

Add an **optional** method to the same interface so one chunker can support both modes:

- **Existing**: `chunk(self, document: BaseDocument) -> list[DocumentChunk]` — used when we already have a parsed document (e.g. generic `WindowedDocumentChunker`).
- **New (optional)**: `chunk_from_raw(self, raw_content: bytes, content_type: str, document_id: str, source_uri: str | None = None) -> list[DocumentChunk]`.

If the orchestrator’s chunker implements `chunk_from_raw` and the request is for a supported content type (e.g. `application/xml`), the orchestrator uses `chunk_from_raw` and can skip a full parse until after entity extraction. Otherwise it parses first and uses `chunk(document)`.

- **Pros**: One interface; one chunker can support both document-based and raw-based chunking (e.g. PMC chunker).
- **Cons**: Optional method is a bit informal; could use a protocol or a separate “raw chunker” interface if you prefer stricter typing.

### 2.2 Option B: Separate interface

Define `RawDocumentChunkerInterface` with:

```python
async def chunk_from_raw(
    self,
    raw_content: bytes,
    content_type: str,
    document_id: str,
    source_uri: str | None = None,
) -> list[DocumentChunk]:
```

Orchestrator then accepts an optional `raw_chunker` (and optionally `raw_content_types: set[str]`). When `raw_chunker` is set and `content_type` is in the supported set, use `raw_chunker.chunk_from_raw(...)`; otherwise parse and use the regular chunker.

- **Pros**: Clear separation; no optional methods.
- **Cons**: Two chunker concepts; orchestrator has to know about both.

**Recommendation**: Start with **Option A** (optional `chunk_from_raw` on `DocumentChunkerInterface`) to keep one chunker per pipeline and simpler orchestrator logic. Can refactor to Option B later if needed.

---

## 3. PMC Chunker Implementation

- **Location**: `examples/medlit/pipeline/pmc_chunker.py` (or alongside `pmc_streaming.py`).
- **Class**: e.g. `PMCStreamingChunker(DocumentChunkerInterface)`.

**Behavior**:

1. **`chunk(self, document: BaseDocument)`**  
   Fallback for document-based use: delegate to a simple windowed chunker over `document.content` (or return a single chunk with full content). Used when the orchestrator already has a parsed document (e.g. non-PMC or when not using raw path).

2. **`chunk_from_raw(self, raw_content: bytes, content_type: str, document_id: str, source_uri: str | None = None)`**  
   - If `content_type` (normalized, e.g. strip `; charset=...`) is not `application/xml` / `text/xml`, return a single chunk with `raw_content.decode()` (or raise / return empty list if you want to force document-based path only for XML).
   - Otherwise call `iter_pmc_windows(raw_content, window_size=..., overlap=..., include_abstract_separately=True)` (reuse existing `pmc_streaming`).
   - For each `(window_index, text)`:
     - Build a `DocumentChunk`:
       - `content=text`
       - `chunk_index=window_index`
       - `document_id=document_id`
       - `start_offset`, `end_offset`: track cumulative character offset as you iterate so that chunk `i` has logical offsets into the “virtual” concatenated document (e.g. first window 0..len(text0), next window `step..step+len(text1)` with `step = window_size - overlap`). This allows downstream (e.g. `BatchingEntityExtractor`) to adjust mention offsets correctly.
       - `metadata`: e.g. `{"section": "abstract"}` for window 0 when `include_abstract_separately` is True, or `{"window_index": str(window_index)}`.
   - Return the list of `DocumentChunk`s in order.

**Config**: Accept `window_size`, `overlap`, `include_abstract_separately` (e.g. from `ChunkingConfig` or a small PMC-specific config) so they match current `iter_pmc_windows` defaults (4000, 800, True) or are overridable.

**Document ID**: When orchestrator calls `chunk_from_raw`, it must pass `document_id`. For PMC we can derive it from `source_uri` when not provided (e.g. `Path(source_uri).stem`) in the orchestrator before calling `chunk_from_raw`.

---

## 4. Orchestrator Changes

- **New optional fields** (all optional so existing callers remain valid):
  - `document_chunker: DocumentChunkerInterface | None = None`
  - `streaming_entity_extractor: StreamingEntityExtractorInterface | None = None`
  - `streaming_relationship_extractor: StreamingRelationshipExtractorInterface | None = None`

**Entity extraction flow** (when streaming is used):

1. **Chunks**  
   - If `document_chunker` is set and it has `chunk_from_raw` and `content_type` is supported (e.g. XML for PMC):
     - Resolve `document_id` (e.g. from `source_uri`: `Path(source_uri).stem` if `source_uri` else generate a temporary id).
     - `chunks = await document_chunker.chunk_from_raw(raw_content, content_type, document_id, source_uri)`.
   - Else:
     - Parse as today: `document = await self.parser.parse(...)`.
     - `chunks = await document_chunker.chunk(document)` (requires chunker).
   - If no chunker is set, keep **current behavior**: parse, store document, then `entity_extractor.extract(document, raw_content=..., content_type=...)` (no streaming).

2. **Streaming entity extraction**  
   If `streaming_entity_extractor` is set and we have `chunks`:
   - `mentions = []`
   - `async for batch in streaming_entity_extractor.extract_streaming(chunks): mentions.extend(batch)`
   - Then resolve, validate, store entities (same as now), and set `document_id` for entities from the chunker’s `document_id` (or from the parsed document when we parse later).

3. **Document for storage**  
   When we used raw chunking we don’t have a `BaseDocument` yet:
   - After entity extraction, parse once: `document = await self.parser.parse(raw_content, content_type, source_uri)`.
   - `await self.document_storage.add(document)`.
   - Use this `document.document_id` for entity `source` if we had used a temporary id earlier (or ensure chunker’s `document_id` matches parser’s so we don’t need to rewrite).

4. **Relationship extraction (streaming)**  
   If `streaming_relationship_extractor` is set and we have `chunks` and `document_entities`:
   - `async for rel_batch in streaming_relationship_extractor.extract_windowed(chunks, document_entities):` store and validate each batch.
   - Else: keep current single call `relationship_extractor.extract(document, document_entities)`.

**Summary of branching**:

- No chunker / no streaming extractors → current behavior (parse → extract → store; extract relationships in one go).
- Chunker + streaming entity extractor → get chunks (from raw or from document), run entity `extract_streaming`, then parse & store document, then run relationship extraction (streaming if `streaming_relationship_extractor` set, else single call).

Ensure `document_id` is consistent: e.g. when using PMC chunker, derive `document_id` from `source_uri` the same way the parser does (e.g. `Path(source_uri).stem`) so stored entities and document and relationship extraction all use the same id.

---

## 5. MedLit Pipeline Wiring

In `examples/medlit/scripts/ingest.py` (or wherever the orchestrator is built for MedLit):

1. **Chunker**  
   Instantiate `PMCStreamingChunker` (with desired window_size, overlap, etc.) and pass it as `document_chunker`.

2. **Entity extraction**  
   - Keep `MedLitEntityExtractor` for the actual per-chunk NER.
   - Wrap it: `streaming_entity_extractor = BatchingEntityExtractor(base_extractor=MedLitEntityExtractor(...), deduplicate=True)`.
   - Pass `streaming_entity_extractor` to the orchestrator.
   - Leave `entity_extractor` as the same `MedLitEntityExtractor` for backward compatibility if the orchestrator still needs a non-streaming path for non-XML or when chunker is not used; or set the orchestrator to use streaming whenever `streaming_entity_extractor` and `document_chunker` are present (see above).

3. **Relationship extraction**  
   - Wrap: `streaming_relationship_extractor = WindowedRelationshipExtractor(base_extractor=MedLitRelationshipExtractor(...))`.
   - Pass it to the orchestrator; orchestrator uses `extract_windowed(chunks, document_entities)` when this is set and chunks are available.

4. **Optional cleanup**  
   Once the orchestrator always uses the streaming path for MedLit XML:
   - Simplify `MedLitEntityExtractor.extract()`: remove the branch that uses `raw_content` and `iter_pmc_windows` (and the `raw_content`/`content_type` kwargs if nothing else needs them), so it only does per-chunk extraction from `document.content`. That keeps MedLit as a simple “chunk → NER” implementation used behind `BatchingEntityExtractor`.

---

## 6. DocumentChunk Offsets for PMC

`iter_pmc_windows` yields `(window_index, text)`. To fill `start_offset` and `end_offset` for each `DocumentChunk` (so that `BatchingEntityExtractor`’s offset adjustment is meaningful):

- Treat the concatenated section text as a single logical document and track a running offset.
- When yielding window 0: `start_offset=0`, `end_offset=len(text)`.
- For subsequent windows: use a step of `window_size - overlap` (same as in `iter_overlapping_windows`). So window 1: `start_offset = step`, `end_offset = step + len(text1)` (or track the actual cumulative length if section lengths differ).
- Alternatively, if you don’t need exact global offsets for PMC (e.g. only dedup by name/type), you can set `start_offset=0` and `end_offset=len(content)` for each chunk and document that PMC chunks are window-relative; then downstream can still merge by (name, type) and ignore offsets. Prefer proper cumulative offsets if you want consistent behavior with the rest of the pipeline.

---

## 7. Testing

- **PMC chunker**  
  - Unit tests: `chunk_from_raw` with a small PMC XML fixture; check number of chunks, `document_id`, `chunk_index` order, and that offsets are non-overlapping or overlapping as intended.
  - Optionally: test that `chunk(document)` fallback works when given a parsed `JournalArticle` with `content` set.

- **Orchestrator**  
  - Integration test: run entity extraction with a chunker + streaming entity extractor and (if possible) raw PMC bytes; then run relationship extraction with streaming relationship extractor; assert entities and relationships are stored and document is stored once.
  - Keep existing tests that don’t pass chunker/streaming extractors unchanged (current behavior).

- **MedLit**  
  - Run the existing MedLit ingest script on one or two PMC XMLs with the new wiring; confirm entity and relationship counts and that only one parse happens after entity extraction when using raw chunking.

---

## 8. Implementation Order

1. **PMC chunker**  
   - Add optional `chunk_from_raw` to `DocumentChunkerInterface` (and implement it as no-op or raise in `WindowedDocumentChunker`).  
   - Implement `PMCStreamingChunker` in `examples/medlit/pipeline/pmc_chunker.py` using `iter_pmc_windows`, with correct `DocumentChunk` construction and cumulative offsets.  
   - Unit tests for `PMCStreamingChunker.chunk_from_raw`.

2. **Orchestrator**  
   - Add optional `document_chunker`, `streaming_entity_extractor`, `streaming_relationship_extractor`.  
   - Implement branching: when chunker + streaming entity extractor are set, get chunks (from `chunk_from_raw` if supported and content type matches, else parse then `chunk(document)`), run `extract_streaming`, then parse & store document, then run relationship extraction (streaming or single call).  
   - Ensure `document_id` is consistent (e.g. from `source_uri` when using raw chunking).  
   - Integration test for the new path.

3. **MedLit wiring**  
   - In MedLit ingest script, create `PMCStreamingChunker`, `BatchingEntityExtractor(MedLitEntityExtractor(...))`, `WindowedRelationshipExtractor(MedLitRelationshipExtractor(...))`, and pass them into the orchestrator.  
   - Run manual test on a few PMC XMLs.

4. **Cleanup (optional)**  
   - Simplify `MedLitEntityExtractor`: remove internal `iter_pmc_windows` and `raw_content`/`content_type` streaming path once the orchestrator always uses the streaming path for MedLit XML.

---

## 9. Files to Touch (Summary)

| Area              | File(s) |
|-------------------|--------|
| Interface         | `kgraph/pipeline/streaming.py` — optional `chunk_from_raw` on `DocumentChunkerInterface`; `WindowedDocumentChunker` stub or no-op. |
| PMC chunker       | `examples/medlit/pipeline/pmc_chunker.py` — new `PMCStreamingChunker`. |
| Orchestrator      | `kgraph/ingest.py` — optional fields, branching, and streaming flow. |
| MedLit wiring     | `examples/medlit/scripts/ingest.py` — build and pass chunker + streaming extractors. |
| MedLit cleanup    | `examples/medlit/pipeline/mentions.py` — remove internal PMC streaming from `extract()` (optional, after wiring). |
| Tests             | New tests for PMC chunker; integration test for orchestrator streaming path. |

This plan gets the “nice” abstractions used end-to-end, keeps PMC memory-efficient via a PMC-specific chunker that implements the same interface and uses `iter_pmc_windows` under the hood, and keeps a clear path for other domains to use the same streaming pipeline with a document-based chunker.
