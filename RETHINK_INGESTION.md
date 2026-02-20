# Rethinking ingestion: Q&A over entity-scoped context

This document describes a proposed alternative to the current relationship-extraction stage. The goal is to get richer, more useful results by asking targeted questions about each entity with access to **all and only** the documents that mention that entity, rather than extracting fixed (subject, predicate, object) triples from isolated evidence windows.

**Status:** Design discussion. No implementation yet.

---

## Motivation

Asking natural questions about the graph (e.g. "What drugs treat breast cancer?", "What do we know about tamoxifen? Side effects?") yields more insightful, nuanced answers than querying the current relationship tables. The relationship-extraction pipeline forces the LLM to emit triples from small windows and a fixed predicate set; it often misses nuance (e.g. "no side effects extracted" vs. "papers don't report side effects") and underuses the full corpus. A per-entity Q&A pass, with the model allowed to consult every document that mentions that entity, aligns the pipeline with how we want to query the data.

---

## Dependency: we need mentions

To do this right, **we must have mention-level data before the Q&A stage.**

For **entity N**, we want the LLM to see **only the papers (documents) that mention entity N**. That implies:

1. **Mentions are produced and stored**  
   Each mention row links an `entity_id` to a `document_id` (and optionally section, offsets, text_span, context). The bundle must include a `mentions.jsonl` (or equivalent) that is loaded into storage so that we can answer: "Which documents mention this entity?"

2. **Scoping context by entity**  
   When processing entity N:
   - Call `get_mentions_for_entity(entity_id)` (or equivalent) to get all mention rows for N.
   - From those rows, obtain the distinct set of `document_id`s.
   - Retrieve only those documents (or the relevant sections/passages for those documents) as context for the LLM.

Without mentions, we would have to either (a) send all documents for every entity (infeasible at scale) or (b) guess which documents are relevant (unreliable). So **stages that produce and persist mentions are a hard dependency** for the Q&A approach. The current pipeline already has a notion of mentions (e.g. `mentions.jsonl` in the bundle, `get_mentions_for_entity` in storage); we need to ensure that by the time we run the Q&A stage, every entity we process has mention rows and we can resolve entity → list of documents.

---

## Proposed pipeline shape

- **Stages 1–3 unchanged (for this discussion):**
  - Stage 1: Parse, provenance (documents, sections, etc.).
  - Stage 2: NER (entity extraction). Entities and their **mentions** (entity_id, document_id, section, text_span, context, …) must be produced and written so they can be loaded into the bundle.
  - Stage 3: Promotion (provisional → canonical, merging, etc.). Still operate on entities; mentions stay attached.

- **Stage 4 (new): Per-entity Q&A**
  - Input: entities (after promotion) and **mentions** (entity_id → document_id and any passage/section info).
  - For each entity N:
    1. **Reclassification gate:** Ask the LLM: "This entity has been classified as [type X]. Given the following passages from papers that mention it, should it be classified as something different? If yes, what type?" Context = only the documents that mention N (e.g. passages or sections where N appears). If the answer is "yes, it should be type Y," we can update the entity type and proceed with the Y-type question set (or define a policy for when to run both).
    2. **Type-specific questions:** Ask a set of questions that depend on the (possibly corrected) entity type, **driven by the ontology** (entity types and their allowed predicates / relationship types). Examples for illustration: Drug → efficacy, safety, side effects, contraindications, interactions; Disease → treatments, risk factors, symptoms, diagnosis; Gene/protein → function, pathways, associations. The full set of types and the questions (or predicates) per type should come from the ontology, not a fixed hardcoded list.
    3. The LLM is given **only the documents (or passages) that mention entity N** as context, so it can synthesize across all relevant text.

- **Output of stage 4:** For each entity, we store the answers (and optionally any structured extractions derived from them). That becomes the queryable "what do we know about this entity?" layer.

---

## Design choices to settle

1. **Context budget**  
   "All documents mentioning entity N" can be large. We need a policy: e.g. cap total tokens per entity, or one representative passage per document, or retrieval over mention passages. This is the main scalability lever.

2. **What we store**  
   Options: (a) free-text answers only (e.g. in entity properties or an `entity_qa` table); (b) structured fields parsed from answers (e.g. `side_effects: [...]`); (c) both. (c) supports both narrative search and faceted/structured queries.

3. **Relationship graph**  
   The graph can become optional or derived: e.g. parse "Drug A: side effects include B, C" into `(A, side_effect, B)` and `(A, side_effect, C)` if we want graph queries. Or we keep a minimal graph from a separate lightweight pass and treat the Q&A layer as the primary rich representation.

4. **Question taxonomy**  
   The mapping from entity type to question templates (or predicates to elicit) should be derived from the ontology so the full set of types and relationship types is used. Some types may have "no further questions." Easy to iterate by updating the ontology rather than pipeline code.

5. **Reclassification policy**  
   If the model says "yes, reclassify as Y," do we (a) update type and run only the Y question set, (b) run both X and Y sets, or (c) something else? (a) is the simplest and avoids redundant questions.

---

## Ontology and predicates

Whatever we build should be driven by the **full domain ontology**, not a fixed list of predicates or relationship types. Entity types, relationship types (predicates), and any type-predicate constraints should come from the ontology so that:

- New entity types or predicates added to the ontology are automatically available to the pipeline (e.g. for question generation or for validating/emitting relationships).
- We avoid hardcoding a small set of predicates and instead use whatever the ontology defines.

**New predicates proposed by the LLM:** If during Q&A or extraction the LLM proposes a relationship type (predicate) that is **not** in the current ontology, we should **not** accept it automatically. There must be a **human-in-the-loop** step: the proposed predicate is queued for review, and a human approves or rejects adding it to the ontology (or to an allowed extension set). Only after approval can that predicate be used in stored relationships or in downstream question templates. This keeps the ontology authoritative and prevents predicate proliferation without oversight.

---

## Tool for the LLM to query paper contents

During the Q&A stage (and possibly at query time), the LLM needs a way to pull in relevant text from the corpus. We should provide something like a **tool** the LLM can use to query paper contents (e.g. "retrieve passages that discuss side effects of this drug" or "find sentences mentioning efficacy"). How that tool works affects both quality and cost.

**Windowing vs. retrieval**

- **Window-based (like today):** We could expose a tool that returns one or more **windows** from the current document set (e.g. the same overlapping-window scheme used in relationship extraction). The LLM calls the tool with document_id (or entity-scoped doc list) and optionally window index or a keyword. Pros: no extra precomputation; reuse existing chunker. Cons: windows are fixed in advance and may not align with "what is relevant to this question"; the LLM may need to request many windows to find the right content, and we still have to fit a subset into context.

- **Precomputed passage embeddings:** We could **precompute embedding vectors** for every paragraph (or every sentence) of every paper at ingestion time, store them in a vector store (e.g. pgvector, or a dedicated vector DB), and at Q&A time **embed the user question** (or the current Q&A prompt) and **retrieve the top-k most similar passages** for the documents that mention the entity. Pros: the tool returns the most semantically relevant chunks, so we use context budget on what matters. Cons: we must run an embedding model over the full corpus once (and again when new papers are added), store vectors, and run vector search at Q&A time; we need to choose granularity (paragraph vs. sentence—paragraph is usually a good tradeoff for readability and retrieval quality).

**Recommendation (for discussion):** A **hybrid** is plausible: (1) Use **mentions** to restrict to documents that mention the entity. (2) Within those documents, either (a) return fixed windows that overlap with mention locations (cheap, no embeddings), or (b) precompute **per-paragraph** (or per-section) embeddings, store them, and let the tool take a natural-language query and return the top-k passages by similarity. Option (b) scales better to "very large number of papers" because we never send full papers—we only send the k passages that best match the question. The tool API could look like: `query_paper_contents(entity_id, query_text, k=10)` returning up to k passages (with document_id and section/offset) so the LLM can cite them or synthesize. If we precompute embeddings, they are also useful for other retrieval scenarios (e.g. "find papers similar to this question" at query time).

**Precomputed embeddings: when and how**

- **When:** At ingestion time, after we have full text (and optionally after we have mentions, so we could restrict embedding to paragraphs that contain or neighbor mentions to save cost—though that complicates "query anything" later). Or in a separate batch job over the corpus.
- **Granularity:** Paragraph is a natural unit: long enough for coherence, short enough for many chunks per paper. Sentence-level gives finer retrieval but more chunks and more storage.
- **Storage:** Vectors need to be stored and indexed (pgvector, or a dedicated vector DB). Each row needs (document_id, passage_id or offset, text, embedding).
- **Use at Q&A time:** For entity N, we already restrict to docs mentioning N. The tool then: embed the current question (or a summary of the question set), retrieve top-k passages from those docs only, return their text (and provenance) to the LLM. That keeps the context window filled with the most relevant content instead of arbitrary windows.

---

## Scale and performance

We want to support a **very large number of papers** eventually. That forces choices about context budget, batching, and where GPUs (or other hardware) are used.

**Performance considerations**

1. **Context budget per entity:** For each entity we may have dozens or hundreds of mentioning documents. We cannot send all of them to the LLM in one call. So we **must** either (a) cap the number of documents or passages per entity, (b) use retrieval (e.g. embedding-based) to send only the top-k most relevant passages per question, or (c) do multi-turn Q&A (one question per call, each call with a subset of docs). Retrieval (or smart windowing) is the main lever to keep context size bounded while growing the corpus.

2. **Total LLM calls:** Current relationship extraction does roughly one LLM call per **window** (across all papers). The Q&A approach does one or more LLM calls per **entity** (reclassification + several type-specific questions). So total calls scale with number of entities, not number of windows. If we have 100k entities and 5 questions each, that is 500k+ calls unless we batch or combine questions. We may want to batch (e.g. one prompt with multiple questions per entity) or sample entities for an initial rollout.

3. **Embedding workload:** If we precompute paragraph (or sentence) embeddings for the full corpus, that is one embedding call per passage. For millions of paragraphs, that is a large batch job. It can be run offline, incrementally (new papers only), and with batching; it is embarrassingly parallel. Cost is dominated by embedding model inference (CPU or GPU).

4. **GPU resources**

   - **Embedding model:** Today the medlit pipeline uses Ollama for embeddings (e.g. nomic-embed-text), which can run on CPU or GPU. For very large corpora, running the embedding model on GPU speeds up precomputation; for Q&A-time retrieval we only embed the query (and maybe a few candidates), so GPU is less critical unless we do reranking at scale.
   - **LLM (Q&A):** The main cost is **LLM inference**—each Q&A call consumes context (retrieved passages + prompt) and generates an answer. If we use a local model (e.g. via Ollama), a single GPU can serve one model; throughput is limited by context length and batch size. If we use a hosted API, we pay per token and need to watch rate limits. For "very large number of papers" and millions of entities, we likely need either (a) significant GPU capacity for local inference, (b) a hosted LLM with high rate limits, or (c) aggressive sampling/caching (e.g. only run Q&A for high-value or high-usage entities, and fill in the rest lazily or from a smaller model).

5. **Comparison to current ingestion pipeline**

   - **Current:** Window-based; one relationship-extraction LLM call per window; embeddings used for evidence validation and entity resolution (on demand, with optional file cache). Throughput is limited by LLM calls per window and by embedding calls when semantic validation is on. Typically run on a single machine with Ollama (CPU or one GPU).
   - **Q&A approach:** One or more LLM calls per entity, each with a context that is either (a) a subset of windows from mentioning docs, or (b) top-k retrieved passages. So we trade "many small window calls" for "fewer but potentially larger context calls per entity." If we precompute embeddings, we add a one-time (or incremental) embedding job over the corpus; at Q&A time we add vector search and one embedding call per question (or per entity) to build context. Overall, the Q&A approach can be **more** LLM-expensive per entity than one window-sized call, because we send more context (retrieved passages) and may ask several questions. The benefit is better answers. To scale, we rely on retrieval to keep context size bounded, on batching/sampling, and on GPU or hosted LLM capacity for the LLM itself. Embedding precomputation is a one-time cost that pays off when we have many papers and many entities.

---

## Summary

- **Keep:** NER (stage 2) and promotion (stage 3) as they are.
- **Require:** Mentions are produced in (or by) stage 2 and loaded so that for any entity N we can get the set of documents that mention N.
- **Replace stage 4** with a per-entity Q&A pass: reclassification gate, then type-specific questions, with context scoped to **only the papers that mention that entity** (using mentions to drive that scope).
- **Result:** A "what do we know about entity N?" layer that aligns with how we want to ask questions, with the option to derive a relationship graph or keep a minimal one alongside it.
- **Ontology:** Use the full ontology (entity types, predicates, constraints); no fixed predicate list. If the LLM proposes a new predicate, a human must approve it before it is added to the ontology or used in the pipeline.
