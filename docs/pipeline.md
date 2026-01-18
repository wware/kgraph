# Pipeline Components

The ingestion pipeline consists of pluggable components for parsing, extraction, resolution, and embedding generation.

## Component Interfaces

### DocumentParserInterface

Converts raw document bytes into structured `BaseDocument` instances:

```python
from kgraph.pipeline import DocumentParserInterface
from kgraph.document import BaseDocument

class DocumentParserInterface(ABC):
    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> BaseDocument: ...
```

Example implementation using an LLM for structure extraction:

```python
class LLMDocumentParser(DocumentParserInterface):
    def __init__(self, llm_client, document_class: type[BaseDocument]):
        self._llm = llm_client
        self._doc_class = document_class

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> BaseDocument:
        text = raw_content.decode("utf-8")

        # Use LLM to extract structure
        response = await self._llm.complete(
            f"Extract title and sections from this document:\n\n{text}"
        )

        return self._doc_class(
            document_id=str(uuid.uuid4()),
            title=response.title,
            content=text,
            content_type=content_type,
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
            metadata={"sections": response.sections},
        )
```

### EntityExtractorInterface

Extracts entity mentions from documents:

```python
from kgraph.pipeline import EntityExtractorInterface
from kgraph import EntityMention

class EntityExtractorInterface(ABC):
    async def extract(self, document: BaseDocument) -> list[EntityMention]: ...
```

`EntityMention` captures raw extractions before resolution:

```python
mention = EntityMention(
    text="aspirin",                    # Exact text span
    entity_type="drug",                # Domain-specific type
    start_offset=145,                  # Position in document
    end_offset=152,
    confidence=0.95,                   # Extraction confidence
    context="patients taking aspirin showed...",  # Surrounding text
    metadata={"extraction_model": "gpt-4"},
)
```

Example NER-based extractor:

```python
class SpacyEntityExtractor(EntityExtractorInterface):
    def __init__(self, nlp, type_mapping: dict[str, str]):
        self._nlp = nlp
        self._type_map = type_mapping  # spaCy label -> domain type

    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        doc = self._nlp(document.content)
        mentions = []
        for ent in doc.ents:
            if ent.label_ in self._type_map:
                mentions.append(EntityMention(
                    text=ent.text,
                    entity_type=self._type_map[ent.label_],
                    start_offset=ent.start_char,
                    end_offset=ent.end_char,
                    confidence=0.9,
                ))
        return mentions
```

### EntityResolverInterface

Maps mentions to canonical or provisional entities:

```python
from kgraph.pipeline import EntityResolverInterface
from kgraph.storage import EntityStorageInterface

class EntityResolverInterface(ABC):
    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]: ...

    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]: ...
```

Resolution strategy:

1. Search existing entities by name/synonym
2. Search by embedding similarity if available
3. Query external authority (UMLS, DBPedia, etc.)
4. Create provisional entity if no match found

```python
class HybridEntityResolver(EntityResolverInterface):
    def __init__(
        self,
        embedding_gen: EmbeddingGeneratorInterface,
        authority_client,  # External ID authority
        entity_factory,    # Creates domain-specific entities
    ):
        self._embedder = embedding_gen
        self._authority = authority_client
        self._factory = entity_factory

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        # Try name match first
        existing = await existing_storage.find_by_name(
            mention.text, mention.entity_type, limit=1
        )
        if existing:
            return existing[0], 0.95

        # Try embedding similarity
        embedding = await self._embedder.generate(mention.text)
        similar = await existing_storage.find_by_embedding(
            embedding, threshold=0.9, limit=1
        )
        if similar:
            return similar[0][0], similar[0][1]

        # Try external authority
        canonical = await self._authority.lookup(
            mention.text, mention.entity_type
        )
        if canonical:
            entity = self._factory.create_canonical(
                canonical_id=canonical.id,
                name=canonical.name,
                entity_type=mention.entity_type,
                embedding=embedding,
            )
            return entity, canonical.confidence

        # Create provisional
        entity = self._factory.create_provisional(
            name=mention.text,
            entity_type=mention.entity_type,
            embedding=embedding,
            confidence=mention.confidence,
        )
        return entity, mention.confidence
```

### RelationshipExtractorInterface

Extracts relationships between entities:

```python
from kgraph.pipeline import RelationshipExtractorInterface

class RelationshipExtractorInterface(ABC):
    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]: ...
```

Example LLM-based extractor:

```python
class LLMRelationshipExtractor(RelationshipExtractorInterface):
    def __init__(self, llm_client, relationship_factory, valid_predicates: list[str]):
        self._llm = llm_client
        self._factory = relationship_factory
        self._predicates = valid_predicates

    async def extract(
        self,
        document: BaseDocument,
        entities: Sequence[BaseEntity],
    ) -> list[BaseRelationship]:
        # Build entity context for LLM
        entity_list = "\n".join(
            f"- {e.name} ({e.get_entity_type()}): {e.entity_id}"
            for e in entities
        )

        prompt = f"""
        Given these entities:
        {entity_list}

        And this document:
        {document.content[:2000]}

        Extract relationships using these predicates: {self._predicates}
        Format: subject_id | predicate | object_id | confidence
        """

        response = await self._llm.complete(prompt)
        return self._parse_relationships(response, document.document_id)
```

### EmbeddingGeneratorInterface

Generates semantic vectors for entity similarity:

```python
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface

class EmbeddingGeneratorInterface(ABC):
    @property
    def dimension(self) -> int: ...

    async def generate(self, text: str) -> tuple[float, ...]: ...

    async def generate_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[float, ...]]: ...
```

Example OpenAI implementation:

```python
class OpenAIEmbedding(EmbeddingGeneratorInterface):
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self._client = client
        self._model = model
        self._dim = 1536  # Model-dependent

    @property
    def dimension(self) -> int:
        return self._dim

    async def generate(self, text: str) -> tuple[float, ...]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return tuple(response.data[0].embedding)

    async def generate_batch(
        self, texts: Sequence[str]
    ) -> list[tuple[float, ...]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [tuple(d.embedding) for d in response.data]
```

## Assembling the Pipeline

Connect components via `IngestionOrchestrator`:

```python
from kgraph import IngestionOrchestrator

orchestrator = IngestionOrchestrator(
    domain=my_domain,
    parser=my_parser,
    entity_extractor=my_entity_extractor,
    entity_resolver=my_resolver,
    relationship_extractor=my_rel_extractor,
    embedding_generator=my_embedder,
    entity_storage=entity_storage,
    relationship_storage=relationship_storage,
    document_storage=document_storage,
)

# Ingest a single document
result = await orchestrator.ingest_document(
    raw_content=document_bytes,
    content_type="text/plain",
    source_uri="https://example.com/doc.txt",
)

# Batch ingestion
results = await orchestrator.ingest_batch([
    (doc1_bytes, "text/plain", "source1"),
    (doc2_bytes, "application/pdf", "source2"),
])

# Run promotion after ingestion
promoted = await orchestrator.run_promotion()
```
