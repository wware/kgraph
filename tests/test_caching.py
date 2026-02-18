"""Tests for embedding caching components.

Tests in-memory and file-based caching, as well as the cached embedding generator.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Sequence

import pytest

from kgraph.pipeline.caching import (
    CachedEmbeddingGenerator,
    EmbeddingCacheConfig,
    FileBasedEmbeddingsCache,
    InMemoryEmbeddingsCache,
)
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface

from tests.conftest import MockEmbeddingGenerator


class TestEmbeddingCacheConfig:
    """Test EmbeddingCacheConfig model."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = EmbeddingCacheConfig()

        assert config.max_cache_size == 10000
        assert config.cache_file is None
        assert config.auto_save_interval == 100
        assert config.normalize_keys is True

    def test_custom_config(self):
        """Test custom cache configuration."""
        config = EmbeddingCacheConfig(
            max_cache_size=5000,
            cache_file=Path("test.json"),
            auto_save_interval=50,
            normalize_keys=False,
        )

        assert config.max_cache_size == 5000
        assert config.cache_file == Path("test.json")
        assert config.auto_save_interval == 50
        assert config.normalize_keys is False

    def test_config_immutability(self):
        """Test that config is immutable (frozen=True)."""
        config = EmbeddingCacheConfig()

        with pytest.raises(Exception):  # Pydantic ValidationError
            config.max_cache_size = 20000


class TestInMemoryEmbeddingsCache:
    """Test InMemoryEmbeddingsCache implementation."""

    async def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = InMemoryEmbeddingsCache()
        embedding = (0.1, 0.2, 0.3, 0.4)

        await cache.put("test_text", embedding)
        result = await cache.get("test_text")

        assert result == embedding

    async def test_cache_miss(self):
        """Test get with cache miss."""
        cache = InMemoryEmbeddingsCache()

        result = await cache.get("nonexistent")

        assert result is None

    async def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = InMemoryEmbeddingsCache()
        embedding = (0.1, 0.2, 0.3)

        await cache.put("text1", embedding)
        await cache.get("text1")  # Hit
        await cache.get("text2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["evictions"] == 0

    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = EmbeddingCacheConfig(max_cache_size=3)
        cache = InMemoryEmbeddingsCache(config=config)

        # Add 4 items to a cache with max_size=3
        await cache.put("text1", (0.1,))
        await cache.put("text2", (0.2,))
        await cache.put("text3", (0.3,))
        await cache.put("text4", (0.4,))  # Should evict text1

        # text1 should be evicted (oldest)
        assert await cache.get("text1") is None
        assert await cache.get("text2") is not None
        assert await cache.get("text3") is not None
        assert await cache.get("text4") is not None

        stats = cache.get_stats()
        assert stats["evictions"] == 1

    async def test_lru_access_order(self):
        """Test that accessing items updates LRU order."""
        config = EmbeddingCacheConfig(max_cache_size=3)
        cache = InMemoryEmbeddingsCache(config=config)

        await cache.put("text1", (0.1,))
        await cache.put("text2", (0.2,))
        await cache.put("text3", (0.3,))

        # Access text1 to make it recently used
        await cache.get("text1")

        # Add text4, should evict text2 (now oldest)
        await cache.put("text4", (0.4,))

        assert await cache.get("text1") is not None  # Recently used, kept
        assert await cache.get("text2") is None  # Oldest, evicted
        assert await cache.get("text3") is not None
        assert await cache.get("text4") is not None

    async def test_batch_get(self):
        """Test batch get operation."""
        cache = InMemoryEmbeddingsCache()

        await cache.put("text1", (0.1,))
        await cache.put("text2", (0.2,))

        results = await cache.get_batch(["text1", "text2", "text3"])

        assert results[0] == (0.1,)
        assert results[1] == (0.2,)
        assert results[2] is None

    async def test_batch_put(self):
        """Test batch put operation."""
        cache = InMemoryEmbeddingsCache()

        texts = ["text1", "text2", "text3"]
        embeddings = [(0.1,), (0.2,), (0.3,)]

        await cache.put_batch(texts, embeddings)

        assert await cache.get("text1") == (0.1,)
        assert await cache.get("text2") == (0.2,)
        assert await cache.get("text3") == (0.3,)

    async def test_clear(self):
        """Test clearing the cache."""
        cache = InMemoryEmbeddingsCache()

        await cache.put("text1", (0.1,))
        await cache.put("text2", (0.2,))
        await cache.get("text1")  # Create some stats

        await cache.clear()

        assert await cache.get("text1") is None
        assert await cache.get("text2") is None

        stats = cache.get_stats()
        assert stats["size"] == 0
        # Note: hits and misses are reset by clear()
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # The two gets above after clear

    async def test_key_normalization(self):
        """Test that keys are normalized when enabled."""
        config = EmbeddingCacheConfig(normalize_keys=True)
        cache = InMemoryEmbeddingsCache(config=config)

        await cache.put("  TEST Text  ", (0.1,))

        # Should match with different casing/whitespace
        result = await cache.get("test text")
        assert result == (0.1,)

    async def test_no_key_normalization(self):
        """Test cache without key normalization."""
        config = EmbeddingCacheConfig(normalize_keys=False)
        cache = InMemoryEmbeddingsCache(config=config)

        await cache.put("TEST", (0.1,))

        # Should not match with different casing
        assert await cache.get("test") is None
        assert await cache.get("TEST") == (0.1,)


class TestFileBasedEmbeddingsCache:
    """Test FileBasedEmbeddingsCache implementation."""

    async def test_put_and_get(self):
        """Test basic put and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file)
            cache = FileBasedEmbeddingsCache(config=config)

            embedding = (0.1, 0.2, 0.3, 0.4)
            await cache.put("test_text", embedding)
            result = await cache.get("test_text")

            assert result == embedding

    async def test_persistence(self):
        """Test that cache persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, auto_save_interval=1)

            # Create cache and add data
            cache1 = FileBasedEmbeddingsCache(config=config)
            await cache1.put("text1", (0.1,))
            await cache1.save()

            # Create new cache instance and load
            cache2 = FileBasedEmbeddingsCache(config=config)
            await cache2.load()

            result = await cache2.get("text1")
            assert result == (0.1,)

    async def test_auto_save(self):
        """Test automatic saving at intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, auto_save_interval=2)
            cache = FileBasedEmbeddingsCache(config=config)

            # Add items - should auto-save after 2 updates
            await cache.put("text1", (0.1,))
            assert not cache_file.exists()  # Not saved yet

            await cache.put("text2", (0.2,))
            assert cache_file.exists()  # Auto-saved after 2 updates

    async def test_manual_save(self):
        """Test manual save operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, auto_save_interval=0)
            cache = FileBasedEmbeddingsCache(config=config)

            await cache.put("text1", (0.1,))
            assert not cache_file.exists()  # No auto-save

            await cache.save()
            assert cache_file.exists()  # Manually saved

    async def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "nonexistent.json"
            config = EmbeddingCacheConfig(cache_file=cache_file)
            cache = FileBasedEmbeddingsCache(config=config)

            # Should not raise error
            await cache.load()

            # Cache should be empty
            assert await cache.get("text1") is None

    async def test_normalize_keys_on_load(self):
        """Keys loaded from file are normalized so get() with different casing hits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            # Write file with mixed-case and extra whitespace keys
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "Aspirin": [0.1, 0.2],
                        "  ibuprofen  ": [0.3, 0.4],
                    },
                    f,
                )
            config = EmbeddingCacheConfig(cache_file=cache_file, normalize_keys=True)
            cache = FileBasedEmbeddingsCache(config=config)
            await cache.load()

            # Both lookups for aspirin (different casing) must return same embedding
            assert await cache.get("aspirin") == (0.1, 0.2)
            assert await cache.get("Aspirin") == (0.1, 0.2)
            assert await cache.get("ibuprofen") == (0.3, 0.4)
            # After load, in-memory keys are normalized so we have 2 entries
            assert len(cache._cache) == 2

    async def test_lru_eviction_with_persistence(self):
        """Test LRU eviction works with persistent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, max_cache_size=3, auto_save_interval=1)
            cache = FileBasedEmbeddingsCache(config=config)

            # Add 4 items to cache with max_size=3
            await cache.put("text1", (0.1,))
            await cache.put("text2", (0.2,))
            await cache.put("text3", (0.3,))
            await cache.put("text4", (0.4,))  # Should evict text1

            await cache.save()

            # Load in new instance
            cache2 = FileBasedEmbeddingsCache(config=config)
            await cache2.load()

            # Only most recent 3 should be in cache
            assert await cache2.get("text1") is None
            assert await cache2.get("text2") is not None
            assert await cache2.get("text3") is not None
            assert await cache2.get("text4") is not None

    async def test_batch_operations(self):
        """Test batch put and get with persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, auto_save_interval=3)
            cache = FileBasedEmbeddingsCache(config=config)

            texts = ["text1", "text2", "text3"]
            embeddings = [(0.1,), (0.2,), (0.3,)]

            await cache.put_batch(texts, embeddings)
            await cache.save()

            # Load in new instance
            cache2 = FileBasedEmbeddingsCache(config=config)
            await cache2.load()

            results = await cache2.get_batch(texts)
            assert results == list(embeddings)

    async def test_concurrent_access(self):
        """Concurrent get/put/save/load do not corrupt cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(
                cache_file=cache_file, auto_save_interval=50, normalize_keys=True
            )
            cache = FileBasedEmbeddingsCache(config=config)

            async def do_ops(i: int) -> None:
                key = f"key_{i % 20}"
                emb = (float(i), float(i + 1))
                await cache.put(key, emb)
                await cache.get(key)

            await asyncio.gather(*[do_ops(i) for i in range(50)])
            await cache.save()
            await cache.load()

            # All keys 0..19 should be present (normalized)
            for i in range(20):
                result = await cache.get(f"key_{i}")
                assert result is not None
            assert len(cache._cache) == 20


class TestCachedEmbeddingGenerator:
    """Test CachedEmbeddingGenerator wrapper."""

    async def test_cache_hit(self):
        """Test that cached values are returned without calling base generator."""
        base_gen = MockEmbeddingGenerator(dim=4)
        cache = InMemoryEmbeddingsCache()
        cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        # First call - should hit base generator
        emb1 = await cached_gen.generate("test")
        stats1 = cache.get_stats()

        # Second call - should hit cache
        emb2 = await cached_gen.generate("test")
        stats2 = cache.get_stats()

        assert emb1 == emb2
        assert stats2["hits"] == stats1["hits"] + 1
        assert stats2["misses"] == stats1["misses"]  # No additional misses

    async def test_cache_miss(self):
        """Test that cache misses call base generator."""
        base_gen = MockEmbeddingGenerator(dim=4)
        cache = InMemoryEmbeddingsCache()
        cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        # First call - cache miss
        emb = await cached_gen.generate("test")

        assert emb is not None
        assert len(emb) == 4

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["size"] == 1  # Should be cached now

    async def test_dimension_property(self):
        """Test that dimension is passed through from base generator."""
        base_gen = MockEmbeddingGenerator(dim=768)
        cache = InMemoryEmbeddingsCache()
        cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        assert cached_gen.dimension == 768

    async def test_batch_generation_with_cache(self):
        """Test batch generation with partial cache hits."""
        base_gen = MockEmbeddingGenerator(dim=4)
        cache = InMemoryEmbeddingsCache()

        # Pre-populate cache with one item
        await cache.put("text1", (0.1, 0.2, 0.3, 0.4))

        cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        # Request batch with one cached and one uncached
        results = await cached_gen.generate_batch(["text1", "text2"])

        assert len(results) == 2
        assert results[0] == (0.1, 0.2, 0.3, 0.4)  # From cache
        assert results[1] is not None  # Generated fresh

        # Both should now be in cache
        stats = cache.get_stats()
        assert stats["size"] == 2

    async def test_save_cache_convenience_method(self):
        """Test convenience method for saving cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            config = EmbeddingCacheConfig(cache_file=cache_file, auto_save_interval=0)
            cache = FileBasedEmbeddingsCache(config=config)

            base_gen = MockEmbeddingGenerator(dim=4)
            cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

            await cached_gen.generate("test")
            await cached_gen.save_cache()

            assert cache_file.exists()

    async def test_get_cache_stats(self):
        """Test getting cache statistics through wrapper."""
        base_gen = MockEmbeddingGenerator(dim=4)
        cache = InMemoryEmbeddingsCache()
        cached_gen = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        await cached_gen.generate("test1")
        await cached_gen.generate("test1")  # Hit
        await cached_gen.generate("test2")

        stats = cached_gen.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2

    async def test_cached_generator_calls_base_once_per_text(self):
        """Repeated generate() with same text calls base generator only once."""
        call_log: list[str] = []

        class RecordingMock(EmbeddingGeneratorInterface):
            def __init__(self) -> None:
                self._dim = 4

            @property
            def dimension(self) -> int:
                return self._dim

            async def generate(self, text: str) -> tuple[float, ...]:
                call_log.append(("generate", text))
                return (hash(text) % 1000 / 1000.0,) * self._dim

            async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
                call_log.append(("generate_batch", list(texts)))
                return [await self.generate(t) for t in texts]

        base = RecordingMock()
        cache = InMemoryEmbeddingsCache()
        cached = CachedEmbeddingGenerator(base_generator=base, cache=cache)

        await cached.generate("x")
        await cached.generate("x")
        assert len(call_log) == 1 and call_log[0] == ("generate", "x")

    async def test_cached_generator_batch_returns_correct_order(self):
        """generate_batch with mixed hits/misses returns list in input order."""
        base_gen = MockEmbeddingGenerator(dim=4)
        cache = InMemoryEmbeddingsCache()
        await cache.put("a", (0.1, 0.2, 0.3, 0.4))
        cached = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache)

        results = await cached.generate_batch(["a", "b", "a"])

        assert len(results) == 3
        assert results[0] == (0.1, 0.2, 0.3, 0.4)
        assert results[2] == (0.1, 0.2, 0.3, 0.4)
        assert results[1] is not None and len(results[1]) == 4
        # Only "b" was a miss; base generate_batch called once with ["b"]
        assert cache.get_stats()["misses"] == 1
        assert cache.get_stats()["size"] == 2


class TestCachingIntegration:
    """Integration tests for caching components."""

    async def test_end_to_end_caching_workflow(self):
        """Test complete caching workflow with persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "embeddings.json"
            config = EmbeddingCacheConfig(
                cache_file=cache_file,
                max_cache_size=100,
                auto_save_interval=5,
            )

            # Phase 1: Generate embeddings with caching
            base_gen = MockEmbeddingGenerator(dim=4)
            cache1 = FileBasedEmbeddingsCache(config=config)
            await cache1.load()

            cached_gen1 = CachedEmbeddingGenerator(base_generator=base_gen, cache=cache1)

            texts = [f"text{i}" for i in range(10)]
            embeddings1 = await cached_gen1.generate_batch(texts)
            await cached_gen1.save_cache()

            stats1 = cached_gen1.get_cache_stats()
            assert stats1["size"] == 10

            # Phase 2: Load cache in new instance
            cache2 = FileBasedEmbeddingsCache(config=config)
            await cache2.load()

            base_gen2 = MockEmbeddingGenerator(dim=4)
            cached_gen2 = CachedEmbeddingGenerator(base_generator=base_gen2, cache=cache2)

            # Should all be cache hits
            embeddings2 = await cached_gen2.generate_batch(texts)

            stats2 = cached_gen2.get_cache_stats()
            assert stats2["hits"] == 10
            assert stats2["misses"] == 0

            # Embeddings should match
            assert embeddings1 == embeddings2

    async def test_cache_with_many_items(self):
        """Test cache performance with many items."""
        cache = InMemoryEmbeddingsCache(config=EmbeddingCacheConfig(max_cache_size=1000))

        # Add 500 items
        for i in range(500):
            await cache.put(f"text{i}", tuple([float(i)] * 4))

        # Verify all are cached
        stats = cache.get_stats()
        assert stats["size"] == 500

        # Access all items (should all be hits)
        for i in range(500):
            result = await cache.get(f"text{i}")
            assert result is not None

        stats = cache.get_stats()
        assert stats["hits"] == 500
