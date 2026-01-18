"""A simple example demonstrating the kgraph ingestion pipeline."""

import asyncio


async def main():
    """Run a simple ingestion pipeline."""
    # This example requires test dependencies to be installed
    # pip install "kgraph[dev]"

    try:
        from kgraph.ingest import IngestionOrchestrator
        from kgraph.storage.memory import (
            InMemoryDocumentStorage,
            InMemoryEntityStorage,
            InMemoryRelationshipStorage,
        )
        from tests.conftest import (
            MockDocumentParser,
            MockEmbeddingGenerator,
            MockEntityExtractor,
            MockEntityResolver,
            MockRelationshipExtractor,
            TestDomainSchema,
        )
    except ImportError:
        print("Please install dev dependencies: pip install -e .[dev]")
        return

    print("Initializing kgraph components for a 'test' domain...")

    # 1. Initialize components
    orchestrator = IngestionOrchestrator(
        domain=TestDomainSchema(),
        parser=MockDocumentParser(),
        entity_extractor=MockEntityExtractor(),
        entity_resolver=MockEntityResolver(),
        relationship_extractor=MockRelationshipExtractor(),
        embedding_generator=MockEmbeddingGenerator(),
        entity_storage=InMemoryEntityStorage(),
        relationship_storage=InMemoryRelationshipStorage(),
        document_storage=InMemoryDocumentStorage(),
    )

    # 2. Ingest documents
    print("\n--- Ingesting Documents ---")
    docs_to_ingest = [
        (b"Paper A cites [Paper B] and [Paper C].", "text/plain", "uri:A"),
        (
            b"Another paper about [Paper B] which is a cool paper.",
            "text/plain",
            "uri:B",
        ),
        (b"Paper D is cited by [Paper A].", "text/plain", "uri:D"),
    ]
    ingestion_result = await orchestrator.ingest_batch(docs_to_ingest)
    print(f"Processed {ingestion_result.documents_processed} documents.")
    print(f"Extracted {ingestion_result.total_entities_extracted} total entities.")
    print(f"Extracted {ingestion_result.total_relationships_extracted} total relationships.")

    # 3. List provisional entities
    print("\n--- Provisional Entities ---")
    provisional_entities = await orchestrator.entity_storage.list_all(status="provisional")
    for entity in provisional_entities:
        print(f"- {entity.name} (ID: {entity.entity_id}), Usage: {entity.usage_count}")

    # 4. Run promotion
    # The test domain promotes entities with usage_count >= 2
    print("\n--- Running Promotion ---")
    promoted = await orchestrator.run_promotion()
    print(f"Promoted {len(promoted)} entities.")
    for entity in promoted:
        print(f"- {entity.name} promoted to canonical ID: {entity.entity_id}")

    # 5. List canonical entities
    print("\n--- Canonical Entities ---")
    canonical_entities = await orchestrator.entity_storage.list_all(status="canonical")
    for entity in canonical_entities:
        print(f"- {entity.name} (ID: {entity.entity_id})")

    # 6. Find merge candidates
    # Let's add two similar entities to test this
    from kgraph.entity import EntityStatus
    from tests.conftest import make_test_entity

    e1 = make_test_entity(
        "USA",
        status=EntityStatus.CANONICAL,
        embedding=(0.1, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    e2 = make_test_entity(
        "United States",
        status=EntityStatus.CANONICAL,
        embedding=(0.11, 0.22, 0.91, 0.01, 0.0, 0.0, 0.0, 0.0),
    )
    await orchestrator.entity_storage.add(e1)
    await orchestrator.entity_storage.add(e2)

    print("\n--- Finding Merge Candidates (Similarity > 0.98) ---")
    candidates = await orchestrator.find_merge_candidates(similarity_threshold=0.98)
    print(f"Found {len(candidates)} merge candidates.")
    for e1, e2, sim in candidates:
        print(f"- Candidate pair: {e1.name} and {e2.name} (Similarity: {sim:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
