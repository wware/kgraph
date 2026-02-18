# Embedding cacheing is not working

## Prompt:
2/17/2026, 8:47:42 PM

We're going to need to do some performance work. I'm doing twenty papers on an expensive A100 instance and it's taken three hours. Can't have that.



## Response:
2/17/2026, 8:48:05 PM

````plaintext
Thought process: Identified multiple bottlenecks hindering pipeline performance and scalability.

The user is talking about performance issues - 20 papers taking 3 hours on an A100 instance, which is expensive. That's about 9 minutes per paper, which does seem quite slow for what should be a fairly efficient extraction pipeline.

Looking at the codebase, I can see several potential bottlenecks:

1. **LLM calls are likely the main bottleneck** - The pipeline uses Ollama with llama3.1:8b, and there are multiple LLM calls per paper:
   - Entity extraction (possibly windowed for large papers)
   - Relationship extraction 
   - Canonical ID lookups (UMLS, HGNC, RxNorm, etc.) - though these should be cached

2. **Streaming vs batch processing** - The code shows PMC streaming chunker that processes papers in windows, but each window might be making separate LLM calls

3. **I/O and caching issues** - Canonical ID lookups, embedding generation, etc.

4. **Inefficient parallelization** - Running papers sequentially rather than in parallel

Let me think about the specific performance optimizations:

1. **Batch LLM calls** - Instead of one call per window, batch multiple windows
2. **Parallel processing** - Process multiple papers simultaneously 
3. **Caching optimization** - Better embedding and canonical ID caching
4. **Model optimization** - Use smaller/faster models for some tasks, or optimize prompts
5. **Skip unnecessary work** - Don't re-extract if already processed

Looking at the code, I see there's already some caching infrastructure in place, but it might not be optimal.

The user is clearly focused on practical deployment concerns - 3 hours on an A100 for 20 papers means the current approach doesn't scale economically.
````

That's way too slow. 9 minutes per paper on an A100 won't scale. Let's look at the bottlenecks:

**Major suspects:**

1. **Sequential LLM calls per window** - If you're doing entity + relationship extraction on each 2k-character window separately, that's probably 10-20 LLM calls per paper. Each call has ~1-2 second overhead even on fast hardware.

2. **Canonical ID API calls** - UMLS, HGNC, RxNorm lookups are hitting external APIs with rate limits and latency. Even with caching, first-pass papers are slow.

3. **No batching/parallelization** - Processing papers one at a time instead of parallel.

**Quick wins:**

```python
# 1. Batch windows in single LLM call
prompt = f"""
Extract entities from these {len(windows)} text windows:

Window 1: {windows[0]}
Window 2: {windows[1]}
...

Return JSON array with window_index for each entity.
"""

# 2. Parallel paper processing
async def process_papers_parallel(papers, max_concurrent=4):
    semaphore = asyncio.Semaphore(max_concurrent) 
    tasks = [process_single_paper(paper, semaphore) for paper in papers]
    return await asyncio.gather(*tasks, return_exceptions=True)

# 3. Pre-populate canonical ID cache
# Run once: python -m examples.medlit.scripts.build_canonical_cache
```

**Medium-term optimizations:**

- **Smaller model for entity extraction** - llama3.1:8b → llama3.2:3b for first pass, full model only for relationships
- **Skip windows with no entities** - Run cheap NER first, only do full extraction on entity-rich windows  
- **Streaming relationship extraction** - Don't wait for all entities before starting relationships

What's your current `--ollama-timeout` setting? And are you seeing the bottleneck in entity extraction, relationship extraction, or canonical lookups?



## Prompt:
2/17/2026, 8:57:18 PM

Wait, I *am* caching embeddings! WTF. Apparently my cache isn't working! I wasted ten bucks finding that out, that instance costs over three bucks per hour. Damn.



## Response:
2/17/2026, 8:57:34 PM

````plaintext
Thought process: Diagnosed caching malfunction and formulated debugging strategy.

The user is frustrated because they spent $10 on an A100 instance (at $3+ per hour) only to discover their embedding cache isn't working properly. They're seeing tons of embedding API calls when they expected the cache to prevent most of them.

Looking at their logs, they're getting:
`​`​`
DEBUG: Inside async context manager
DEBUG: Set model to nomic-embed-text
DEBUG: URL is http://localhost:11434/api/embeddings
EMBED STATUS: 200
`​`​`

This pattern repeating constantly suggests the cache isn't working as expected. Let me think about why this might be happening:

1. **Cache key normalization issues** - Maybe the cache keys aren't matching due to whitespace, capitalization, or encoding differences
2. **Cache not loading properly** - The log shows "Embeddings cache loaded (110 entries)" but maybe it's not being used
3. **Cache scope issues** - Maybe the cache is being created fresh for each entity or each paper
4. **Text preprocessing differences** - The text being embedded might be different from what's in the cache due to normalization

The user is understandably frustrated about the cost. I should help them debug this quickly so they can avoid more waste.

Looking at the code, they have a CachedEmbeddingGenerator that wraps around a base generator with a FileBasedEmbeddingsCache. The cache should be preventing duplicate calls.

I should suggest some immediate debugging steps to figure out why the cache isn't working.
````

Ugh, that's infuriating! $10 down the drain because the cache isn't working. Let's debug this fast before you burn more money.

**Quick cache debug:**

1. **Check what's actually being embedded:**
```python
# Add this to your CachedEmbeddingGenerator.generate() method
def generate(self, text: str) -> tuple[float, ...]:
    print(f"EMBED REQUEST: '{text[:50]}...'")
    cached = await self.cache.get(text)
    if cached:
        print(f"CACHE HIT: {text[:20]}...")
        return cached
    else:
        print(f"CACHE MISS: {text[:20]}...")
        # ... rest of method
```

2. **Check cache key normalization:**
```bash
# Peek at your cache file
head -10 /path/to/your/embeddings_cache.json
```

**Likely culprits:**

- **Different text each time** - Entity names getting different context/formatting: `"aspirin"` vs `"Aspirin"` vs `"aspirin (drug)"`
- **Cache not being saved/loaded** - Cache loaded 110 entries but maybe not persisting new ones
- **Async cache corruption** - Multiple concurrent embedding requests corrupting the cache state

**Emergency brake:**
```python
# Add this to your embedding generator to force cache-only mode
def generate(self, text: str) -> tuple[float, ...] | None:
    cached = await self.cache.get(text)
    if cached:
        return cached
    else:
        print(f"SKIP EMBEDDING: {text[:30]}")
        return None  # Don't generate, just skip
```

What does your cache file look like? And are you seeing the same entity names being embedded multiple times?



## Prompt:
2/17/2026, 8:59:49 PM

Gotta fix that cache. Also, I want to use an asyncio thread pool in the Ollama client, just in case I can get better performance that way.



## Response:
2/17/2026, 9:00:03 PM

````plaintext
Thought process: Identified caching bottlenecks and asyncio optimization strategies.

The user wants to fix the caching issue that's causing their knowledge graph ingestion to be painfully slow (and expensive on an A100 instance). They're also interested in using asyncio thread pools in the Ollama client for better performance.

Let me focus on practical solutions to both issues:

1. Cache debugging and fixing
2. Asyncio thread pool implementation in Ollama client

This is a technical performance issue, so I should be direct and helpful with concrete solutions.
````

Right, let's fix both issues. The cache problem is costing you real money.

**Cache debugging strategy:**

```python
# Add this to your CachedEmbeddingGenerator to see what's happening
async def generate(self, text: str) -> tuple[float, ...]:
    cache_key = self.cache._normalize_key(text, "embedding")  
    print(f"EMBED: '{text}' -> key: '{cache_key}'")
    
    cached = await self.cache.get(text)
    if cached is not None:
        print(f"  ✓ CACHE HIT")
        self.cache._hits += 1
        return cached
    
    print(f"  ✗ CACHE MISS - generating...")
    embedding = await self.base_generator.generate(text)
    await self.cache.put(text, embedding)
    print(f"  ✓ CACHED for next time")
    return embedding
```

**Asyncio thread pool for Ollama:**

```python
# In your OllamaLLMClient class
class OllamaLLMClient(LLMClientInterface):
    def __init__(self, model: str = 'llama3.1:8b', host: str = 'http://localhost:11434', 
                 timeout: float = 300.0, max_workers: int = 4):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._sync_generate, 
            prompt
        )
    
    def _sync_generate(self, prompt: str) -> str:
        # Your existing synchronous Ollama call
        response = requests.post(f"{self.host}/api/generate", ...)
        return response.json()["response"]
```

**Batch embeddings for Ollama:**

```python
# For the embedding generator
async def generate_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        self.executor,
        self._sync_batch_embed,
        texts
    )

def _sync_batch_embed(self, texts: list[str]) -> list[tuple[float, ...]]:
    response = requests.post(f"{self.host}/api/embed", json={
        "model": "nomic-embed-text",
        "input": texts  # Ollama supports batch embedding
    })
    return [tuple(emb) for emb in response.json()["embeddings"]]
```

First priority: add that cache debugging to see why you're getting cache misses. Are you seeing duplicate entity names being embedded?
