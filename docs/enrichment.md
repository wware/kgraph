# Entity Enrichment

Entity enrichment is the process of augmenting entities with external knowledge from authoritative sources like DBPedia, Wikidata, UMLS, and other linked open data (LOD) sources. This document explains how to use enrichment in kgraph and how to implement custom enrichers.

## Overview

The enrichment system in kgraph provides:

- **Pluggable architecture**: Add enrichment sources as needed
- **Optional feature**: Enrichment is opt-in via orchestrator configuration
- **Graceful degradation**: Failed lookups don't break the pipeline
- **Async-first**: Non-blocking I/O for external API calls
- **Caching support**: Reduce redundant API calls

### Pipeline Integration

Enrichment runs after entity resolution but before storage:

```
Document → Parser → EntityExtractor → EntityResolver → [Enricher(s)] → Storage
                                                             ↓
                                                      DBPedia/External APIs
```

This positioning allows enrichers to:
- Augment resolved entities with canonical IDs
- Add metadata from external sources
- Validate entity disambiguation
- Support entity merging via shared canonical URIs

## Using DBPedia Enrichment

The `DBPediaEnricher` is included with kgraph and provides out-of-the-box DBPedia integration.

### Basic Usage

```python
from kgraph import IngestionOrchestrator
from kgraph.pipeline.enrichment import DBPediaEnricher

# Create enricher with default settings
dbpedia_enricher = DBPediaEnricher()

# Configure orchestrator with enrichment
orchestrator = IngestionOrchestrator(
    domain=my_domain,
    parser=my_parser,
    entity_extractor=my_extractor,
    entity_resolver=my_resolver,
    relationship_extractor=my_rel_extractor,
    embedding_generator=my_embedder,
    entity_storage=entity_storage,
    relationship_storage=rel_storage,
    document_storage=doc_storage,
    entity_enrichers=[dbpedia_enricher],  # Add enricher(s) here
)

# Ingest documents - entities will be automatically enriched
result = await orchestrator.ingest_document(content, "text/plain")
```

### Configuration Options

```python
enricher = DBPediaEnricher(
    entity_types_to_enrich={'person', 'location', 'organization'},  # Whitelist types
    confidence_threshold=0.8,       # Only enrich high-confidence entities
    min_lookup_score=0.6,           # Minimum DBPedia match score
    cache_results=True,             # Enable caching (recommended)
    timeout=5.0,                    # HTTP timeout in seconds
)
```

**Configuration Parameters:**

- **`entity_types_to_enrich`**: Set of entity types to enrich (None = all types). Use this to focus enrichment on relevant entity types for your domain.

- **`confidence_threshold`**: Minimum confidence score (0.0-1.0) for an entity to be enriched. Lower-confidence entities are skipped to avoid polluting canonical IDs with uncertain matches.

- **`min_lookup_score`**: Minimum similarity score required to accept a DBPedia match. Higher values increase precision but may reduce recall.

- **`cache_results`**: Whether to cache lookup results in memory. Highly recommended to reduce API calls and improve performance.

- **`timeout`**: HTTP request timeout in seconds. Prevents slow API calls from blocking the pipeline.

### Accessing Enriched Data

Once enriched, entities have DBPedia URIs in their `canonical_ids` field:

```python
# Retrieve entity
entity = await entity_storage.get(entity_id)

# Check if DBPedia enrichment succeeded
if 'dbpedia' in entity.canonical_ids:
    dbpedia_uri = entity.canonical_ids['dbpedia']
    print(f"Entity '{entity.name}' linked to: {dbpedia_uri}")
```

## Domain-Specific Configuration

Different domains benefit from different enrichment strategies:

### Medical/Biomedical Domain

```python
enricher = DBPediaEnricher(
    entity_types_to_enrich={'drug', 'disease', 'protein'},
    confidence_threshold=0.85,  # High precision for medical data
    min_lookup_score=0.7,
)
```

Medical domains may also want to combine DBPedia with specialized enrichers like UMLS.

### Literary/Entertainment Domain

```python
enricher = DBPediaEnricher(
    entity_types_to_enrich={'character', 'location', 'work'},
    confidence_threshold=0.7,
    min_lookup_score=0.5,  # More lenient for fictional entities
)
```

### Academic/Research Domain

```python
enricher = DBPediaEnricher(
    entity_types_to_enrich={'person', 'institution', 'publication'},
    confidence_threshold=0.8,
    min_lookup_score=0.6,
)
```

## Implementing Custom Enrichers

You can implement custom enrichers for any external data source by extending `EntityEnricherInterface`.

### Basic Custom Enricher

```python
from kgraph.pipeline.enrichment import EntityEnricherInterface
from kgraph.entity import BaseEntity

class WikidataEnricher(EntityEnricherInterface):
    """Enriches entities with Wikidata QIDs."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    async def enrich_entity(self, entity: BaseEntity) -> BaseEntity:
        """Add Wikidata QID to entity.canonical_ids."""
        # Skip low-confidence entities
        if entity.confidence < self.confidence_threshold:
            return entity
        
        try:
            # Query Wikidata API
            qid = await self._query_wikidata(entity.name)
            
            if qid:
                # Add to canonical_ids
                ids = dict(entity.canonical_ids)
                ids['wikidata'] = qid
                return entity.model_copy(update={'canonical_ids': ids})
        
        except Exception as e:
            # Always handle errors gracefully
            logger.warning(f"Wikidata enrichment failed: {e}")
        
        return entity
    
    async def _query_wikidata(self, name: str) -> str | None:
        # Implement Wikidata API call
        pass
```

### Best Practices for Custom Enrichers

1. **Handle errors gracefully**: Return the original entity rather than raising exceptions
2. **Respect confidence thresholds**: Only enrich entities you trust
3. **Use caching**: Avoid redundant API calls
4. **Set timeouts**: Don't block the pipeline on slow APIs
5. **Log decisions**: Help debug enrichment behavior
6. **Test thoroughly**: Use mocked API responses in tests

### Domain-Specific Enricher Example

```python
class UMLSEnricher(EntityEnricherInterface):
    """Enriches medical entities with UMLS CUIs."""
    
    def __init__(self, api_key: str, entity_types: set[str] | None = None):
        self.api_key = api_key
        self.entity_types = entity_types or {'drug', 'disease', 'symptom'}
        self._cache = {}
    
    async def enrich_entity(self, entity: BaseEntity) -> BaseEntity:
        # Only enrich medical entity types
        if entity.get_entity_type() not in self.entity_types:
            return entity
        
        # Check cache
        if entity.name in self._cache:
            cui = self._cache[entity.name]
            if cui:
                ids = dict(entity.canonical_ids)
                ids['umls'] = cui
                return entity.model_copy(update={'canonical_ids': ids})
            return entity
        
        try:
            # Query UMLS API with authentication
            cui = await self._query_umls(entity.name)
            self._cache[entity.name] = cui
            
            if cui:
                ids = dict(entity.canonical_ids)
                ids['umls'] = cui
                return entity.model_copy(update={'canonical_ids': ids})
        
        except Exception as e:
            logger.warning(f"UMLS enrichment failed for '{entity.name}': {e}")
            self._cache[entity.name] = None
        
        return entity
    
    async def _query_umls(self, name: str) -> str | None:
        # Implement UMLS REST API call
        pass
```

## Chaining Multiple Enrichers

You can use multiple enrichers together to gather IDs from multiple sources:

```python
orchestrator = IngestionOrchestrator(
    # ... other params ...
    entity_enrichers=[
        DBPediaEnricher(entity_types_to_enrich={'person', 'location'}),
        WikidataEnricher(confidence_threshold=0.8),
        UMLSEnricher(api_key=umls_key, entity_types={'drug', 'disease'}),
    ],
)
```

Enrichers run sequentially in the order specified. Each enricher receives the entity enriched by previous enrichers, allowing them to build on each other's work.

## Performance Considerations

### Caching

Caching is essential for good performance:

- **In-memory caching**: Built into `DBPediaEnricher`, suitable for most use cases
- **Persistent caching**: For large-scale ingestion, consider Redis or file-based caches
- **Negative caching**: Cache failed lookups to avoid retrying known failures

### Rate Limiting

Be mindful of external API rate limits:

- **DBPedia Lookup**: No strict rate limits, but be respectful
- **Wikidata**: Rate limiting on the public API
- **UMLS**: Requires API key, has usage limits

Consider:
- Adding delays between requests
- Using batch APIs when available
- Running enrichment as a separate offline process for large datasets

### Batch Processing

For bulk enrichment, consider batching:

```python
class BatchDBPediaEnricher(DBPediaEnricher):
    async def enrich_batch(self, entities: list[BaseEntity]) -> list[BaseEntity]:
        """Optimize batch enrichment with parallel requests."""
        import asyncio
        
        tasks = [self.enrich_entity(entity) for entity in entities]
        return await asyncio.gather(*tasks)
```

## Troubleshooting

### No Enrichment Happening

1. Check that enrichers are configured: `entity_enrichers=[...]` in orchestrator
2. Verify entity confidence meets threshold
3. Check entity type is in whitelist (if configured)
4. Enable debug logging: `logging.getLogger('kgraph.pipeline.enrichment').setLevel(logging.DEBUG)`

### Low Match Rate

1. Lower `min_lookup_score` threshold
2. Add entity synonyms to improve matching
3. Check that entity names are normalized (capitalization, punctuation)
4. Consider domain-specific enrichment sources

### Performance Issues

1. Enable caching: `cache_results=True`
2. Increase timeout if API calls are slow
3. Consider batch processing for large document sets
4. Use async-compatible HTTP libraries (httpx, aiohttp)

### API Errors

1. Check network connectivity
2. Verify API endpoints are accessible
3. Review API rate limits
4. Check for API changes or downtime
5. Implement retry logic with exponential backoff

## Future Enhancements

The enrichment framework is designed for extensibility. Planned enhancements include:

- **Additional enrichers**: Wikidata, OpenAlex, domain-specific sources
- **Persistent caching**: Redis, file-based, database-backed
- **Batch APIs**: Parallel enrichment for better performance
- **Enrichment validation**: Quality checks and confidence scoring
- **Human-in-the-loop**: Interactive disambiguation for ambiguous entities
- **Local knowledge bases**: Support for offline DBPedia dumps and local enrichment

## Related Documentation

- [Pipeline Architecture](pipeline.md) - How enrichment fits into the ingestion pipeline
- [API Reference](api.md) - Detailed API documentation
- [Domain Implementation Guide](domains.md) - Creating domain-specific enrichers
