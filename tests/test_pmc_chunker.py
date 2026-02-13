"""Tests for PMCStreamingChunker."""

import pytest

from examples.medlit.pipeline.pmc_chunker import (
    PMCStreamingChunker,
    document_id_from_source_uri,
)
MINIMAL_PMC_XML = b"""<?xml version="1.0"?>
<article xmlns="http://jats.nlm.nih.gov">
  <front><article-meta>
    <abstract><p>Abstract text here.</p></abstract>
  </article-meta></front>
  <body>
    <sec id="s1"><title>Intro</title><p>First section content.</p></sec>
    <sec id="s2"><p>Second section with more text.</p></sec>
  </body>
</article>
"""


@pytest.fixture
def chunker() -> PMCStreamingChunker:
    """PMC chunker with small window for tests."""
    return PMCStreamingChunker(window_size=100, overlap=20)


@pytest.mark.asyncio
async def test_chunk_from_raw_xml_returns_document_chunks(chunker: PMCStreamingChunker) -> None:
    """chunk_from_raw with XML returns list of DocumentChunk."""
    chunks = await chunker.chunk_from_raw(
        raw_content=MINIMAL_PMC_XML,
        content_type="application/xml",
        document_id="test_doc",
        source_uri="/path/to/PMC123.xml",
    )
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "test_doc"
        assert c.content
        assert c.chunk_index >= 0
        assert c.start_offset >= 0
        assert c.end_offset > c.start_offset


@pytest.mark.asyncio
async def test_chunk_from_raw_xml_content_type_with_charset(chunker: PMCStreamingChunker) -> None:
    """content_type with charset (e.g. application/xml; charset=utf-8) is treated as XML."""
    chunks = await chunker.chunk_from_raw(
        raw_content=MINIMAL_PMC_XML,
        content_type="application/xml; charset=utf-8",
        document_id="test_doc",
    )
    assert len(chunks) >= 1
    assert chunks[0].document_id == "test_doc"


@pytest.mark.asyncio
async def test_chunk_from_raw_non_xml_returns_single_chunk(chunker: PMCStreamingChunker) -> None:
    """Non-XML content_type returns a single chunk with decoded text."""
    raw = b"Plain text content here."
    chunks = await chunker.chunk_from_raw(
        raw_content=raw,
        content_type="text/plain",
        document_id="plain_doc",
    )
    assert len(chunks) == 1
    assert chunks[0].content == "Plain text content here."
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(chunks[0].content)


def test_document_id_from_source_uri() -> None:
    """document_id_from_source_uri returns stem of path."""
    assert document_id_from_source_uri("/path/to/PMC12757604.xml") == "PMC12757604"
    assert document_id_from_source_uri("PMC123.xml") == "PMC123"
    assert document_id_from_source_uri(None) == ""
