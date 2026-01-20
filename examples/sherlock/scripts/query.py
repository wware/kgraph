# examples/sherlock/query.py
from __future__ import annotations

import asyncio

from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)

from ..sources.gutenberg import download_adventures
from ..pipeline import (
    SherlockDocumentParser,
    SherlockEntityExtractor,
    SherlockEntityResolver,
    SherlockRelationshipExtractor,
    SimpleEmbeddingGenerator,
)
from ..domain import SherlockDomainSchema


async def build_or_load() -> tuple[InMemoryEntityStorage, InMemoryRelationshipStorage]:
    # For demo purposes, just ingest again.
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    orch = IngestionOrchestrator(
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

    stories = download_adventures()
    for title, content in stories:
        await orch.ingest_document(
            raw_content=content.encode("utf-8"),
            content_type="text/plain",
            source_uri=title,
        )

    return entity_storage, relationship_storage


async def main() -> None:
    entity_storage, relationship_storage = await build_or_load()

    print("=" * 60)
    print("Sherlock Holmes Knowledge Graph - Query Examples")
    print("=" * 60)

    # Query 1: characters in a specific story
    story_id = "holmes:story:AScandalInBohemia"
    print(f"\n📖 Characters in {story_id}")
    rels = await relationship_storage.get_by_object(story_id, "appears_in")
    for r in rels:
        e = await entity_storage.get(r.subject_id)
        if e:
            print(f"  - {e.name}")

    # Query 2: show all canonical entities
    print("\n🤝 All canonical entities")
    for e in await entity_storage.list_all(status="canonical"):
        print(f"  - {e.name} -> {e.entity_id}")

    # Query 3: co-occurs with Lestrade? Irene Adler? Holmes?
    # who = "holmes:char:InspectorLestrade"
    # who = "holmes:char:IreneAdler"
    who = "holmes:char:SherlockHolmes"
    print(f"\n🤝 Co-occurs with {who}")
    incoming = await relationship_storage.get_by_object(who, "co_occurs_with")
    outgoing = await relationship_storage.get_by_subject(who, "co_occurs_with")
    rels = {(r.subject_id, r.object_id): r for r in (incoming + outgoing)}.values()
    for r in rels:
        other_id = r.subject_id if r.object_id == who else r.object_id
        other = await entity_storage.get(other_id)
        if other:
            print(f"  - {other.name} (confidence={r.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
