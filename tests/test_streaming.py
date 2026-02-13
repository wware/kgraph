"""Tests for streaming pipeline components.

Tests document chunking, streaming entity extraction, and windowed relationship
extraction capabilities.
"""

import pytest

from kgraph.pipeline.streaming import (
    BatchingEntityExtractor,
    ChunkingConfig,
    DocumentChunk,
    StreamingEntityExtractorInterface,
    WindowedDocumentChunker,
    WindowedRelationshipExtractor,
)
from kgschema.document import BaseDocument
from kgschema.entity import EntityMention
from kgschema.relationship import BaseRelationship

from tests.conftest import (
    MockEntityExtractor,
    MockRelationshipExtractor,
    SimpleDocument,
    SimpleEntity,
    make_test_entity,
)


class TestDocumentChunk:
    """Test DocumentChunk model."""

    def test_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="This is a test chunk",
            start_offset=0,
            end_offset=20,
            chunk_index=0,
            document_id="doc1",
            metadata={"section": "introduction"},
        )

        assert chunk.content == "This is a test chunk"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 20
        assert chunk.chunk_index == 0
        assert chunk.document_id == "doc1"
        assert chunk.metadata["section"] == "introduction"

    def test_chunk_immutability(self):
        """Test that chunks are immutable (frozen=True)."""
        chunk = DocumentChunk(
            content="test",
            start_offset=0,
            end_offset=4,
            chunk_index=0,
            document_id="doc1",
        )

        with pytest.raises(Exception):  # Pydantic ValidationError
            chunk.content = "modified"


class TestChunkingConfig:
    """Test ChunkingConfig model."""

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()

        assert config.chunk_size == 2000
        assert config.overlap == 200
        assert config.respect_boundaries is True
        assert config.min_chunk_size == 500

    def test_custom_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=1000,
            overlap=100,
            respect_boundaries=False,
            min_chunk_size=200,
        )

        assert config.chunk_size == 1000
        assert config.overlap == 100
        assert config.respect_boundaries is False
        assert config.min_chunk_size == 200

    def test_config_immutability(self):
        """Test that config is immutable (frozen=True)."""
        config = ChunkingConfig()

        with pytest.raises(Exception):  # Pydantic ValidationError
            config.chunk_size = 3000


class TestWindowedDocumentChunker:
    """Test WindowedDocumentChunker implementation."""

    async def test_single_chunk_document(self):
        """Test chunking a document that fits in a single chunk."""
        chunker = WindowedDocumentChunker(config=ChunkingConfig(chunk_size=1000))
        doc = SimpleDocument(
            document_id="doc1",
            content="This is a short document.",
            metadata={},
        )

        chunks = await chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "This is a short document."
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 25
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc1"

    async def test_multiple_chunks_no_overlap(self):
        """Test chunking a document into multiple non-overlapping chunks."""
        chunker = WindowedDocumentChunker(
            config=ChunkingConfig(
                chunk_size=20,
                overlap=0,
                respect_boundaries=False,
            )
        )

        content = "a" * 50  # 50 characters
        doc = SimpleDocument(document_id="doc1", content=content, metadata={})

        chunks = await chunker.chunk(doc)

        # Should produce 3 chunks: 20, 20, 10
        assert len(chunks) == 3
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 20
        assert chunks[1].start_offset == 20
        assert chunks[1].end_offset == 40
        assert chunks[2].start_offset == 40
        assert chunks[2].end_offset == 50

    async def test_multiple_chunks_with_overlap(self):
        """Test chunking with overlap between chunks."""
        chunker = WindowedDocumentChunker(
            config=ChunkingConfig(
                chunk_size=30,
                overlap=10,
                respect_boundaries=False,
            )
        )

        content = "a" * 70
        doc = SimpleDocument(document_id="doc1", content=content, metadata={})

        chunks = await chunker.chunk(doc)

        # With chunk_size=30 and overlap=10, we move forward by 20 each time
        # Positions: 0-30, 20-50, 40-70
        assert len(chunks) == 3
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == 30
        assert chunks[1].start_offset == 20
        assert chunks[1].end_offset == 50
        assert chunks[2].start_offset == 40
        assert chunks[2].end_offset == 70

    async def test_respect_sentence_boundaries(self):
        """Test chunking that respects sentence boundaries."""
        chunker = WindowedDocumentChunker(
            config=ChunkingConfig(
                chunk_size=50,
                overlap=10,
                respect_boundaries=True,
                min_chunk_size=20,
            )
        )

        content = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        doc = SimpleDocument(document_id="doc1", content=content, metadata={})

        chunks = await chunker.chunk(doc)

        # Should break at sentence boundaries (after periods)
        assert len(chunks) > 1
        # First chunk should end at a sentence boundary
        assert chunks[0].content.endswith(".") or chunks[0].content.endswith("one.")

    async def test_chunk_metadata_preserved(self):
        """Test that document ID is preserved in chunks."""
        chunker = WindowedDocumentChunker(config=ChunkingConfig(chunk_size=20, overlap=5))
        doc = SimpleDocument(
            document_id="test_doc_123",
            content="a" * 50,
            metadata={},
        )

        chunks = await chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.document_id == "test_doc_123"

    async def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = WindowedDocumentChunker(config=ChunkingConfig(chunk_size=20, overlap=5))
        doc = SimpleDocument(document_id="doc1", content="a" * 100, metadata={})

        chunks = await chunker.chunk(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestBatchingEntityExtractor:
    """Test BatchingEntityExtractor implementation."""

    async def test_extract_from_single_chunk(self):
        """Test extracting entities from a single chunk."""
        base_extractor = MockEntityExtractor()
        batching_extractor = BatchingEntityExtractor(base_extractor=base_extractor)

        chunk = DocumentChunk(
            content="[aspirin] is a drug",
            start_offset=0,
            end_offset=19,
            chunk_index=0,
            document_id="doc1",
        )

        all_mentions = []
        async for mentions in batching_extractor.extract_streaming([chunk]):
            all_mentions.extend(mentions)

        # Should extract "aspirin"
        assert len(all_mentions) == 1
        assert all_mentions[0].text == "aspirin"
        assert all_mentions[0].start_pos == 1  # Adjusted from chunk position
        assert all_mentions[0].end_pos == 8

    async def test_extract_from_multiple_chunks(self):
        """Test extracting entities from multiple chunks."""
        base_extractor = MockEntityExtractor()
        batching_extractor = BatchingEntityExtractor(base_extractor=base_extractor)

        chunks = [
            DocumentChunk(
                content="[aspirin] is a drug",
                start_offset=0,
                end_offset=19,
                chunk_index=0,
                document_id="doc1",
            ),
            DocumentChunk(
                content="[ibuprofen] is also a drug",
                start_offset=20,
                end_offset=47,
                chunk_index=1,
                document_id="doc1",
            ),
        ]

        all_mentions = []
        async for mentions in batching_extractor.extract_streaming(chunks):
            all_mentions.extend(mentions)

        # Should extract both entities
        assert len(all_mentions) == 2
        assert all_mentions[0].text == "aspirin"
        assert all_mentions[1].text == "ibuprofen"

    async def test_offset_adjustment(self):
        """Test that entity offsets are adjusted for chunk position."""
        base_extractor = MockEntityExtractor()
        batching_extractor = BatchingEntityExtractor(base_extractor=base_extractor)

        # Entity at position 1-8 in chunk, which starts at offset 100
        chunk = DocumentChunk(
            content="[aspirin] is mentioned here",
            start_offset=100,
            end_offset=127,
            chunk_index=0,
            document_id="doc1",
        )

        all_mentions = []
        async for mentions in batching_extractor.extract_streaming([chunk]):
            all_mentions.extend(mentions)

        # Should have offset adjusted to original document position
        assert all_mentions[0].start_pos == 101  # 1 + 100
        assert all_mentions[0].end_pos == 108  # 8 + 100

    async def test_streaming_iteration(self):
        """Test that results are yielded incrementally."""
        base_extractor = MockEntityExtractor()
        batching_extractor = BatchingEntityExtractor(base_extractor=base_extractor)

        chunks = [
            DocumentChunk(content=f"[entity{i}]", start_offset=i * 20, end_offset=(i + 1) * 20, chunk_index=i, document_id="doc1")
            for i in range(5)
        ]

        iteration_count = 0
        async for mentions in batching_extractor.extract_streaming(chunks):
            iteration_count += 1
            assert len(mentions) >= 1  # At least one entity per chunk

        # Should yield results 5 times (once per chunk)
        assert iteration_count == 5


class TestWindowedRelationshipExtractor:
    """Test WindowedRelationshipExtractor implementation."""

    async def test_extract_from_single_window(self):
        """Test extracting relationships from a single window."""
        base_extractor = MockRelationshipExtractor()
        windowed_extractor = WindowedRelationshipExtractor(base_extractor=base_extractor)

        entity1 = make_test_entity(name="aspirin", entity_id="e1")
        entity2 = make_test_entity(name="pain", entity_id="e2")

        chunk = DocumentChunk(
            content="aspirin treats pain",
            start_offset=0,
            end_offset=19,
            chunk_index=0,
            document_id="doc1",
        )

        all_relationships = []
        async for rels in windowed_extractor.extract_windowed([chunk], [entity1, entity2]):
            all_relationships.extend(rels)

        # Should extract at least one relationship
        assert len(all_relationships) >= 1

    async def test_deduplication_across_windows(self):
        """Test that duplicate relationships are deduplicated across overlapping windows."""
        base_extractor = MockRelationshipExtractor()
        windowed_extractor = WindowedRelationshipExtractor(base_extractor=base_extractor)

        entity1 = make_test_entity(name="aspirin", entity_id="e1")
        entity2 = make_test_entity(name="pain", entity_id="e2")

        # Two overlapping chunks with same entities
        chunks = [
            DocumentChunk(content="aspirin treats pain", start_offset=0, end_offset=19, chunk_index=0, document_id="doc1"),
            DocumentChunk(content="aspirin treats pain again", start_offset=10, end_offset=35, chunk_index=1, document_id="doc1"),
        ]

        all_relationships = []
        async for rels in windowed_extractor.extract_windowed(chunks, [entity1, entity2]):
            all_relationships.extend(rels)

        # Should deduplicate based on (subject_id, predicate, object_id)
        # The exact count depends on mock behavior, but there shouldn't be exact duplicates
        unique_keys = {(r.subject_id, r.predicate, r.object_id) for r in all_relationships}
        assert len(all_relationships) == len(unique_keys)

    async def test_empty_window(self):
        """Test handling windows with no entities."""
        base_extractor = MockRelationshipExtractor()
        windowed_extractor = WindowedRelationshipExtractor(base_extractor=base_extractor)

        chunk = DocumentChunk(
            content="This text has no entities",
            start_offset=0,
            end_offset=25,
            chunk_index=0,
            document_id="doc1",
        )

        all_relationships = []
        async for rels in windowed_extractor.extract_windowed([chunk], []):
            all_relationships.extend(rels)

        # No entities = no relationships
        assert len(all_relationships) == 0

    async def test_single_entity_window(self):
        """Test handling windows with only one entity."""
        base_extractor = MockRelationshipExtractor()
        windowed_extractor = WindowedRelationshipExtractor(base_extractor=base_extractor)

        entity1 = make_test_entity(name="aspirin", entity_id="e1")

        chunk = DocumentChunk(
            content="aspirin is mentioned",
            start_offset=0,
            end_offset=20,
            chunk_index=0,
            document_id="doc1",
        )

        all_relationships = []
        async for rels in windowed_extractor.extract_windowed([chunk], [entity1]):
            all_relationships.extend(rels)

        # Single entity = no relationships (need at least 2)
        assert len(all_relationships) == 0


class TestIntegrationStreamingPipeline:
    """Integration tests for streaming pipeline."""

    async def test_full_streaming_pipeline(self):
        """Test complete streaming pipeline: chunk -> extract entities -> extract relationships."""
        # 1. Create a document and chunk it
        chunker = WindowedDocumentChunker(
            config=ChunkingConfig(chunk_size=50, overlap=10, respect_boundaries=True, min_chunk_size=20)
        )

        content = "[aspirin] treats [pain]. [ibuprofen] also treats [pain]. [aspirin] and [ibuprofen] are similar drugs."
        doc = SimpleDocument(document_id="doc1", content=content, metadata={})

        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 1

        # 2. Extract entities from chunks
        entity_extractor = MockEntityExtractor()
        streaming_extractor = BatchingEntityExtractor(base_extractor=entity_extractor)

        all_mentions = []
        async for mentions in streaming_extractor.extract_streaming(chunks):
            all_mentions.extend(mentions)

        # Should extract multiple entities
        assert len(all_mentions) >= 3  # aspirin, ibuprofen, pain

        # 3. Create entities from mentions (simplified)
        entities = [make_test_entity(name=m.text, entity_id=f"e{i}") for i, m in enumerate(all_mentions)]

        # 4. Extract relationships
        rel_extractor = MockRelationshipExtractor()
        windowed_rel_extractor = WindowedRelationshipExtractor(base_extractor=rel_extractor)

        all_relationships = []
        async for rels in windowed_rel_extractor.extract_windowed(chunks, entities):
            all_relationships.extend(rels)

        # Should extract some relationships
        assert len(all_relationships) >= 1

    async def test_large_document_chunking(self):
        """Test handling very large documents with many chunks."""
        chunker = WindowedDocumentChunker(
            config=ChunkingConfig(chunk_size=100, overlap=20, respect_boundaries=False, min_chunk_size=50)
        )

        # Create a large document
        content = " ".join([f"[entity{i}]" for i in range(100)])  # Many entities
        doc = SimpleDocument(document_id="large_doc", content=content, metadata={})

        chunks = await chunker.chunk(doc)

        # Should produce multiple chunks
        assert len(chunks) > 5

        # All chunks should be properly indexed
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == "large_doc"

        # Extract entities from all chunks
        entity_extractor = MockEntityExtractor()
        streaming_extractor = BatchingEntityExtractor(base_extractor=entity_extractor)

        all_mentions = []
        async for mentions in streaming_extractor.extract_streaming(chunks):
            all_mentions.extend(mentions)

        # Should extract many entities
        assert len(all_mentions) > 10
