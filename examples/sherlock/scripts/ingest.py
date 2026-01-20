# examples/sherlock/ingest.py
from __future__ import annotations

import asyncio

from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)

# from .domain import SherlockDomainSchema
# from .download import download_adventures
# from .extractors import (
#     SherlockDocumentParser,
#     SherlockEntityExtractor,
#     SherlockEntityResolver,
#     SherlockRelationshipExtractor,
#     SimpleEmbeddingGenerator,
# )

from ..sources.gutenberg import download_adventures

# from ..data import KNOWN_CHARACTERS, KNOWN_LOCATIONS, ADVENTURES_STORIES
# from ..pipeline.parser import SherlockDocumentParser
# from ..pipeline.mentions import SherlockEntityExtractor
# from ..pipeline.resolve import SherlockEntityResolver
# from ..pipeline.relationships import SherlockRelationshipExtractor
# from ..pipeline.embeddings import SimpleEmbeddingGenerator
from ..pipeline._legacy_extractors import (
    SherlockDocumentParser,
    SherlockEntityExtractor,
    SherlockEntityResolver,
    SherlockRelationshipExtractor,
    SimpleEmbeddingGenerator,
)
from ..domain import SherlockDomainSchema


def build_orchestrator() -> IngestionOrchestrator:
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    return IngestionOrchestrator(
        domain=SherlockDomainSchema(),
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )


async def main() -> None:
    print("=" * 60)
    print("Sherlock Holmes Knowledge Graph - Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Downloading stories...")
    stories = download_adventures()
    print(f"      Loaded {len(stories)} stories")

    print("\n[2/4] Initializing pipeline...")
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    orchestrator = IngestionOrchestrator(
        domain=SherlockDomainSchema(),
        parser=SherlockDocumentParser(),
        entity_extractor=SherlockEntityExtractor(),
        entity_resolver=SherlockEntityResolver(),
        relationship_extractor=SherlockRelationshipExtractor(),
        embedding_generator=SimpleEmbeddingGenerator(),
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )

    print("\n[3/4] Ingesting stories...")
    for title, content in stories:
        res = await orchestrator.ingest_document(
            raw_content=content.encode("utf-8"),
            content_type="text/plain",
            source_uri=title,
        )
        print(f"      {title}: {res.entities_extracted} entities, {res.relationships_extracted} relationships")

    print("\n[4/4] Summary")
    doc_count = await document_storage.count()
    ent_count = await entity_storage.count()
    rel_count = await relationship_storage.count()

    print(
        f"""
    ╔══════════════════════════════════════╗
    ║       Knowledge Graph Summary        ║
    ╠══════════════════════════════════════╣
    ║  Documents:     {doc_count:>6}       ║
    ║  Entities:      {ent_count:>6}       ║
    ║  Relationships: {rel_count:>6}       ║
    ╚══════════════════════════════════════╝
    """
    )

    # If your orchestrator has export support, plug it in here.


if __name__ == "__main__":
    asyncio.run(main())
