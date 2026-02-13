"""Pipeline interfaces for document processing and extraction."""

from kgraph.pipeline.caching import (
    CachedEmbeddingGenerator,
    EmbeddingCacheConfig,
    EmbeddingsCacheInterface,
    FileBasedEmbeddingsCache,
    InMemoryEmbeddingsCache,
)
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.pipeline.streaming import (
    BatchingEntityExtractor,
    ChunkingConfig,
    DocumentChunk,
    DocumentChunkerInterface,
    StreamingEntityExtractorInterface,
    StreamingRelationshipExtractorInterface,
    WindowedDocumentChunker,
    WindowedRelationshipExtractor,
)

__all__ = [
    # Core interfaces
    "DocumentParserInterface",
    "EntityExtractorInterface",
    "EntityResolverInterface",
    "RelationshipExtractorInterface",
    "EmbeddingGeneratorInterface",
    # Streaming interfaces and implementations
    "DocumentChunkerInterface",
    "StreamingEntityExtractorInterface",
    "StreamingRelationshipExtractorInterface",
    "DocumentChunk",
    "ChunkingConfig",
    "WindowedDocumentChunker",
    "BatchingEntityExtractor",
    "WindowedRelationshipExtractor",
    # Caching interfaces and implementations
    "EmbeddingsCacheInterface",
    "EmbeddingCacheConfig",
    "InMemoryEmbeddingsCache",
    "FileBasedEmbeddingsCache",
    "CachedEmbeddingGenerator",
]
