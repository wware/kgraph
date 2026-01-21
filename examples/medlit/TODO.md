# Medical Literature Domain - Enhancement TODO

This document tracks enhancements to the med-lit domain extension for kgraph. The current implementation works with pre-extracted entities/relationships from Paper JSON format. These enhancements will enable extraction directly from raw text.

## 1. Integrate NER Models for Entity Extraction

**Status**: Not Started  
**Priority**: High  
**Component**: `MedLitEntityExtractor`

### Current State
- `MedLitEntityExtractor` only extracts from pre-extracted entities in Paper JSON
- Returns empty list if no pre-extracted entities found

### Implementation Options

#### Option A: BioBERT (HuggingFace)
**Model**: `d4data/biomedical-ner-all` or `dmis-lab/biobert-base-cased-v1.1`

**Pros**:
- Fast inference (can run on CPU)
- Good accuracy for biomedical entities
- Free (no API costs)
- Can batch process multiple documents

**Cons**:
- Requires model download (~400MB)
- May need GPU for large batches
- Less flexible than LLMs for edge cases

**Implementation**:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class BioBERTEntityExtractor(MedLitEntityExtractor):
    def __init__(self):
        self.ner_pipeline = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            tokenizer="d4data/biomedical-ner-all",
            aggregation_strategy="simple"
        )
    
    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        # Extract from document sections
        mentions = []
        for section_name, section_text in document.get_sections():
            results = self.ner_pipeline(section_text)
            for result in results:
                # Map NER labels to entity types
                entity_type = self._map_label_to_type(result["entity_group"])
                mentions.append(EntityMention(
                    text=result["word"],
                    entity_type=entity_type,
                    start_offset=result["start"],
                    end_offset=result["end"],
                    confidence=result["score"],
                    context=section_text[max(0, result["start"]-50):result["end"]+50],
                    metadata={"section": section_name}
                ))
        return mentions
```

**Entity Type Mapping**:
- `DISEASE` → `"disease"`
- `CHEMICAL` → `"drug"` (may need additional filtering)
- `GENE` → `"gene"` (if model supports it)
- May need additional models for proteins, pathways, etc.

#### Option B: scispaCy
**Model**: `en_ner_bc5cdr_md` (diseases and chemicals)

**Pros**:
- Very fast (optimized Cython)
- Lightweight
- Good for diseases and drugs

**Cons**:
- Limited entity types (only DISEASE and CHEMICAL)
- Dependency issues on Python 3.13+
- Less accurate than BioBERT for some cases

**Implementation**:
```python
import spacy

class SpacyEntityExtractor(MedLitEntityExtractor):
    def __init__(self):
        self.nlp = spacy.load("en_ner_bc5cdr_md")
    
    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        mentions = []
        for section_name, section_text in document.get_sections():
            doc = self.nlp(section_text)
            for ent in doc.ents:
                if ent.label_ == "DISEASE":
                    mentions.append(EntityMention(
                        text=ent.text,
                        entity_type="disease",
                        start_offset=ent.start_char,
                        end_offset=ent.end_char,
                        confidence=0.9,  # spaCy doesn't provide confidence
                        context=section_text[max(0, ent.start_char-50):ent.end_char+50],
                        metadata={"section": section_name}
                    ))
        return mentions
```

#### Option C: LLM-based Extraction (Ollama/OpenAI)
**Use for**: When you need high accuracy and flexible entity types

**Pros**:
- Most flexible (can extract any entity type)
- Can follow complex instructions
- Can extract entities with context and relationships
- **Meditron models** are specifically trained for medical tasks and excel at biomedical entity extraction

**Cons**:
- Slower than NER models
- More expensive (API costs or GPU costs)
- May be overkill for simple entity extraction

**Recommended Models for Medical Entity Extraction**:
- **Meditron-70B** (via Ollama) - Best medical accuracy, trained on PubMed/clinical data
- **Meditron-7B** (via Ollama) - Faster, still medical-specialized
- **GPT-4o** (via OpenAI) - Very good general performance, fast API

**See "LLM-based Relationship Extraction" below for implementation pattern**

### Recommendation
- **Start with BioBERT** (Option A) - good balance of speed, accuracy, and cost
- **Add scispaCy as fallback** if BioBERT unavailable
- **Use LLM extraction** only for papers where NER models fail or for specialized entity types

---

## 2. Integrate Relationship Extraction from Text

**Status**: Not Started  
**Priority**: High  
**Component**: `MedLitRelationshipExtractor`

### Current State
- `MedLitRelationshipExtractor` only extracts from pre-extracted relationships in Paper JSON
- Returns empty list if no pre-extracted relationships found

### Implementation Options

#### Option A: Pattern-Based Extraction
**Use for**: Simple, high-confidence relationships

**Pros**:
- Very fast
- No API/GPU costs
- Deterministic results

**Cons**:
- Limited to patterns you define
- Misses complex relationships
- Requires maintenance of pattern library

**Implementation** (from med-lit-schema's `claims_pipeline.py`):
```python
import re

PREDICATE_PATTERNS = [
    ("treats", [
        r"\b(treats?|treatment|therapeutic|efficacy)\b.*\b(?:of|for|in)\b",
        r"\b(effective|efficacious)\b.*\b(?:against|for|in)\b",
    ], "rct"),
    ("causes", [
        r"\b(causes?|leads? to|results? in|associated with)\b",
    ], "observational"),
    ("increases_risk", [
        r"\b(increases?|elevates?|raises?)\b.*\b(?:risk|probability|likelihood)\b",
        r"\b(risk factor|predisposes?)\b",
    ], "observational"),
]

class PatternRelationshipExtractor(MedLitRelationshipExtractor):
    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]:
        relationships = []
        entity_names = {e.name.lower(): e for e in entities}
        
        for section_name, section_text in document.get_sections():
            sentences = re.split(r'[.!?]+', section_text)
            for sentence in sentences:
                for predicate, patterns, evidence_type in PREDICATE_PATTERNS:
                    for pattern in patterns:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            # Try to find entities in sentence
                            found_entities = [e for name, e in entity_names.items() 
                                            if name in sentence.lower()]
                            if len(found_entities) >= 2:
                                # Create relationship (need to determine subject/object)
                                # This is simplified - real implementation needs more logic
                                relationships.append(...)
        return relationships
```

#### Option B: LLM-based Extraction (Recommended)
**Use for**: Complex relationships, high accuracy requirements

### LLM Provider Comparison

#### OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)

**Costs** (as of 2024):
- GPT-4: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- GPT-4o: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens (much cheaper!)
- GPT-3.5-turbo: ~$0.0015 per 1K input tokens, ~$0.002 per 1K output tokens

**Example costs for relationship extraction**:
- Paper abstract (~500 tokens) + prompt (~300 tokens) = 800 input tokens
- Response (~200 tokens) = 200 output tokens
- GPT-4o: (800 * $0.005/1000) + (200 * $0.015/1000) = $0.004 + $0.003 = **$0.007 per paper**
- 100 papers = **$0.70**
- 1000 papers = **$7.00**

**Pros**:
- No infrastructure to manage
- Reliable API
- Good performance (GPT-4o is fast)
- Easy to use

**Cons**:
- Ongoing API costs
- Rate limits (though generous)
- Data sent to external service (privacy consideration)

#### Ollama (Local LLM on LambdaLabs GPU)

**Costs**:
- LambdaLabs GPU instance (A100 40GB): ~$1.10/hour
- Can process ~10-20 papers per minute (depending on model size)
- 100 papers ≈ 5-10 minutes = **~$0.10**
- 1000 papers ≈ 50-100 minutes = **~$1.00-2.00**

**Pros**:
- No per-request costs (just GPU time)
- Data stays local (privacy)
- No rate limits
- Can use open-source models (Llama 3.1, Mistral, etc.)

**Cons**:
- Need to manage GPU instance
- Slower than OpenAI API (but still reasonable)
- Need to handle instance lifecycle (start/stop)
- May need to implement retry logic, batching

**Recommended Setup**:
- Use LambdaLabs spot instances for cost savings (~50% discount)
- Run batch jobs (process 100+ papers at once)
- Stop instance when done

**Model Recommendations**:
- **Meditron-70B** - **Best for medical tasks** - Specialized medical LLM trained on PubMed, clinical guidelines, and medical papers. Outperforms GPT-3.5 and Med-PaLM on medical reasoning. Slower (~2-3 papers/min). Based on Llama-2, trained on 48B tokens of medical data.
- **Meditron-7B** - Medical-specialized, faster version (~8-12 papers/min). Good balance of speed and medical accuracy.
- **Llama 3.1 70B** - Best general quality, slower (~2-3 papers/min)
- **Llama 3.1 8B** - Good quality, faster (~10-15 papers/min)
- **Mistral 7B** - Fast, decent quality (~15-20 papers/min)

**Note on Meditron**: Meditron is specifically designed for medical applications and trained on:
- ~46,000 clinical practice guidelines
- 16.1 million PubMed abstracts
- ~5 million full-text medical papers
- General domain data (RedPajama)

It achieves ~6% absolute gain over Llama-2 on medical tasks and approaches GPT-4 performance (within 5-10%) on medical benchmarks. Available on Ollama as `meditron-70b` and `meditron-7b`.

### LLM Prompt Design

**Entity Extraction Prompt** (if using LLM):
```
You are a biomedical entity extraction system. Extract all medical entities from the following text.

Text:
{text}

Extract entities of these types:
- disease: Medical conditions, disorders, syndromes
- drug: Medications, therapeutic substances
- gene: Genes (use HGNC format if known)
- protein: Proteins (use UniProt format if known)

For each entity, provide:
1. The exact text span
2. Entity type
3. Canonical ID if you know it (UMLS CUI, HGNC ID, RxNorm ID, UniProt ID)
4. Confidence (0.0-1.0)

Return JSON format:
{
  "entities": [
    {
      "text": "Type 2 Diabetes",
      "type": "disease",
      "canonical_id": "C0011860",
      "confidence": 0.95,
      "start_char": 123,
      "end_char": 140
    }
  ]
}
```

**Relationship Extraction Prompt** (Recommended):
```
You are a biomedical relationship extraction system. Extract relationships between entities from the following text.

Text:
{text}

Entities found in this text:
{entities_list}

Extract relationships with these predicates:
- treats: Drug treats Disease
- causes: Disease causes Symptom
- increases_risk: Gene/Mutation increases risk of Disease
- associated_with: General association between entities
- interacts_with: Drug interacts with Drug
- diagnosed_by: Disease diagnosed by Procedure/Biomarker

For each relationship, provide:
1. Subject entity ID
2. Predicate
3. Object entity ID
4. Evidence text (the sentence/paragraph supporting this)
5. Section where found (abstract, methods, results, discussion)
6. Confidence (0.0-1.0)

Return JSON format:
{
  "relationships": [
    {
      "subject_id": "RxNorm:1187832",
      "predicate": "treats",
      "object_id": "C0006142",
      "evidence": "Olaparib significantly improved progression-free survival in BRCA-mutated breast cancer patients.",
      "section": "abstract",
      "confidence": 0.95
    }
  ]
}
```

**Cost Optimization Strategies**:
1. **Batch processing**: Process multiple papers in one request (if LLM supports)
2. **Use cheaper models**: GPT-3.5-turbo for simple cases, GPT-4o only when needed
3. **Cache results**: Don't re-extract if paper already processed
4. **Two-stage approach**: Use pattern matching first, LLM only for complex cases
5. **LambdaLabs spot instances**: 50% cost savings for batch jobs

### Recommendation
- **Start with pattern-based extraction** (Option A) for common, high-confidence relationships
- **Add LLM extraction** (Option B) for complex cases and papers where patterns fail
- **Use Meditron-70B via Ollama on LambdaLabs** for batch processing - best medical accuracy, cost-effective for 100+ papers
- **Use Meditron-7B via Ollama** if you need faster processing with still-excellent medical accuracy
- **Use OpenAI API (GPT-4o)** for one-off extractions or when you need fastest results

---

## 3. PMC XML Parser

**Status**: Not Started  
**Priority**: Medium  
**Component**: `JournalArticleParser`

### Current State
- `JournalArticleParser` only handles JSON format
- PMC XML parsing raises `NotImplementedError`

### Implementation

**Reference**: med-lit-schema's `ingest/pmc_parser.py` has a complete implementation

**Key Components**:
1. Parse JATS XML format (standard for PMC)
2. Extract metadata (title, authors, journal, dates, identifiers)
3. Extract abstract and full text
4. Handle document structure (sections, paragraphs)

**Implementation**:
```python
import xml.etree.ElementTree as ET
from pathlib import Path

class PMCXMLParser(JournalArticleParser):
    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> JournalArticle:
        if content_type not in ("application/xml", "text/xml"):
            raise ValueError(f"Unsupported content_type: {content_type}")
        
        root = ET.fromstring(raw_content)
        
        # Extract identifiers
        pmc_id = self._extract_pmc_id(root)
        pmid = self._extract_pmid(root)
        doi = self._extract_doi(root)
        
        # Extract bibliographic metadata
        title = self._extract_title(root)
        authors = self._extract_authors(root)
        journal = self._extract_journal(root)
        publication_date = self._extract_publication_date(root)
        
        # Extract content
        abstract = self._extract_abstract(root)
        full_text = self._extract_full_text(root)
        content = f"{abstract}\n\n{full_text}" if full_text else abstract
        
        # Extract metadata
        mesh_terms = self._extract_mesh_terms(root)
        metadata = {
            "mesh_terms": mesh_terms,
            # ... other metadata
        }
        
        # Determine document_id (prefer DOI, else PMID, else PMC ID)
        document_id = f"doi:{doi}" if doi else (f"pmid:{pmid}" if pmid else pmc_id)
        
        return JournalArticle(
            document_id=document_id,
            title=title,
            content=content,
            content_type="text/plain",
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
            authors=tuple(authors),
            abstract=abstract,
            publication_date=publication_date,
            journal=journal,
            doi=doi,
            pmid=pmid,
            metadata=metadata,
        )
```

**Key XML Elements** (JATS format):
- `<article-meta>` - Metadata container
- `<article-id>` - Identifiers (PMC, PMID, DOI)
- `<title-group>` - Title
- `<contrib-group>` - Authors
- `<abstract>` - Abstract text
- `<body>` - Full text content
- `<sec>` - Sections within body

**Reference Implementation**: See `med-lit-schema/ingest/pmc_parser.py` for complete parsing logic

---

## 4. Better Entity Resolution with Embedding Similarity

**Status**: Not Started  
**Priority**: Medium  
**Component**: `MedLitEntityResolver`

### Current State
- `MedLitEntityResolver` only uses canonical_id_hint from pre-extracted entities
- Creates provisional entities if no canonical ID found
- No embedding-based similarity matching

### Implementation

**Use Cases**:
1. Match entity mentions to existing entities by semantic similarity
2. Merge duplicate entities (same entity, different names)
3. Resolve to canonical IDs using external authorities (UMLS, HGNC, etc.)

#### Option A: Embedding Similarity Matching

**Implementation**:
```python
class EmbeddingEntityResolver(MedLitEntityResolver):
    def __init__(self, domain: DomainSchema, embedding_generator: EmbeddingGeneratorInterface):
        super().__init__(domain=domain)
        self.embedding_generator = embedding_generator
        self.similarity_threshold = 0.85
    
    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        # Generate embedding for mention
        mention_embedding = await self.embedding_generator.generate(mention.text)
        
        # Search for similar entities
        similar_entities = await existing_storage.find_similar(
            embedding=mention_embedding,
            entity_type=mention.entity_type,
            threshold=self.similarity_threshold,
            top_k=5
        )
        
        if similar_entities:
            # Use most similar entity
            best_match, similarity = similar_entities[0]
            return best_match, similarity * mention.confidence
        
        # No match found - create provisional entity
        return await super().resolve(mention, existing_storage)
```

**Embedding Models for Biomedical Text**:
- **BioBERT**: `dmis-lab/biobert-base-cased-v1.1` - Best for biomedical entities
- **SciBERT**: `allenai/scibert_scivocab_uncased` - Good for scientific text
- **PubMedBERT**: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` - Trained on PubMed abstracts
- **OpenAI embeddings**: `text-embedding-3-small` or `text-embedding-3-large` - Good general purpose

**Cost Comparison**:
- **BioBERT/SciBERT**: Free (run locally), ~100ms per entity
- **OpenAI embeddings**: ~$0.00002 per entity (text-embedding-3-small), very fast

#### Option B: External Authority Lookup

**UMLS API** (for diseases):
- Requires UMLS license (free for research)
- Can resolve disease names to UMLS CUIs
- API rate limits apply

**HGNC API** (for genes):
- Free, no authentication needed
- Can resolve gene names to HGNC IDs
- REST API: `https://rest.genenames.org/fetch/symbol/{gene_name}`

**RxNorm API** (for drugs):
- Free, requires API key (free registration)
- Can resolve drug names to RxNorm IDs
- REST API: `https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}`

**Implementation**:
```python
import httpx

class AuthorityEntityResolver(MedLitEntityResolver):
    async def _lookup_umls(self, entity_name: str) -> str | None:
        # UMLS API lookup
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://uts-ws.nlm.nih.gov/rest/search/current",
                params={"string": entity_name, "apiKey": self.umls_api_key}
            )
            # Parse response and return CUI
            ...
    
    async def _lookup_hgnc(self, gene_name: str) -> str | None:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://rest.genenames.org/fetch/symbol/{gene_name}"
            )
            # Parse response and return HGNC ID
            ...
```

### Recommendation
- **Start with embedding similarity** (Option A) - works for all entity types
- **Add authority lookups** (Option B) for high-confidence canonical ID resolution
- **Use BioBERT embeddings** for best biomedical accuracy
- **Fall back to OpenAI embeddings** if local models unavailable

---

## Implementation Priority

1. **High Priority**: NER models (BioBERT) - Enables entity extraction from raw text
2. **High Priority**: LLM relationship extraction - Enables relationship extraction from raw text
3. **Medium Priority**: PMC XML parser - Enables direct processing of PMC files
4. **Medium Priority**: Embedding-based entity resolution - Improves entity matching quality

## Cost Summary

**For processing 100 papers**:
- **BioBERT NER**: Free (local), ~10 minutes
- **Pattern-based relationships**: Free, ~1 minute
- **LLM relationships (Ollama)**: ~$0.10-0.20 (LambdaLabs GPU, 10-20 min)
- **LLM relationships (OpenAI)**: ~$0.70 (GPT-4o)
- **Embeddings (BioBERT)**: Free (local), ~2 minutes
- **Embeddings (OpenAI)**: ~$0.02 (text-embedding-3-small)

**Total cost (Ollama approach)**: ~$0.10-0.20  
**Total cost (OpenAI approach)**: ~$0.72

**For processing 1000 papers**:
- **Ollama approach**: ~$1.00-2.00
- **OpenAI approach**: ~$7.00

**Recommendation**: Use Ollama on LambdaLabs for batch processing (better cost), OpenAI for one-off or when speed is critical.
