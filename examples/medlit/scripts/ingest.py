#!/usr/bin/env python3
"""Ingestion script for medical literature knowledge graph.

Processes Paper JSON files (from med-lit-schema) and generates a kgraph bundle.

Usage:
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle --limit 10
"""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from kgraph.ingest import IngestionOrchestrator
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from kgraph.export import write_bundle

from ..domain import MedLitDomainSchema
from ..pipeline.parser import JournalArticleParser
from ..pipeline.mentions import MedLitEntityExtractor
from ..pipeline.resolve import MedLitEntityResolver
from ..pipeline.relationships import MedLitRelationshipExtractor
from ..pipeline.embeddings import SimpleMedLitEmbeddingGenerator


def build_orchestrator() -> IngestionOrchestrator:
    """Build the ingestion orchestrator for medical literature domain."""
    domain = MedLitDomainSchema()

    return IngestionOrchestrator(
        domain=domain,
        parser=JournalArticleParser(),
        entity_extractor=MedLitEntityExtractor(),
        entity_resolver=MedLitEntityResolver(domain=domain),
        relationship_extractor=MedLitRelationshipExtractor(),
        embedding_generator=SimpleMedLitEmbeddingGenerator(),
        entity_storage=InMemoryEntityStorage(),
        relationship_storage=InMemoryRelationshipStorage(),
        document_storage=InMemoryDocumentStorage(),
    )


async def ingest_paper_json(
    orchestrator: IngestionOrchestrator,
    json_path: Path,
) -> tuple[str, int, int]:
    """Ingest a single Paper JSON file.

    Args:
        orchestrator: The ingestion orchestrator.
        json_path: Path to the Paper JSON file.

    Returns:
        Tuple of (paper_id, entities_extracted, relationships_extracted).
    """
    try:
        # Read JSON file
        with open(json_path, "rb") as f:
            raw_content = f.read()

        # Ingest the paper
        result = await orchestrator.ingest_document(
            raw_content=raw_content,
            content_type="application/json",
            source_uri=str(json_path),
        )

        return (result.document_id, result.entities_extracted, result.relationships_extracted)

    except Exception as e:
        print(f"  ERROR processing {json_path.name}: {e}")
        return (json_path.stem, 0, 0)


async def main() -> None:
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest medical literature papers and generate bundle")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing Paper JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="medlit_bundle",
        help="Output directory for the bundle (default: medlit_bundle)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process (for testing)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    # Find all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    if args.limit:
        json_files = json_files[: args.limit]

    if not json_files:
        print(f"ERROR: No JSON files found in {input_dir}")
        return

    print("=" * 60)
    print("Medical Literature Knowledge Graph - Ingestion Pipeline")
    print("=" * 60)
    print(f"\nInput directory: {input_dir}")
    print(f"Found {len(json_files)} paper(s) to process")
    if args.limit:
        print(f"(Limited to {args.limit} papers)")

    print("\n[1/3] Initializing pipeline...")
    orchestrator = build_orchestrator()
    entity_storage = orchestrator.entity_storage
    relationship_storage = orchestrator.relationship_storage
    document_storage = orchestrator.document_storage

    print("\n[2/3] Ingesting papers...")
    total_entities = 0
    total_relationships = 0
    processed = 0
    errors = 0

    for json_file in json_files:
        paper_id, entities, relationships = await ingest_paper_json(orchestrator, json_file)
        if entities > 0 or relationships > 0:
            print(f"  {json_file.name}: {entities} entities, {relationships} relationships")
            total_entities += entities
            total_relationships += relationships
            processed += 1
        else:
            errors += 1

    print(f"\n[3/3] Exporting bundle...")
    print(f"  Processed: {processed} papers")
    print(f"  Errors/skipped: {errors} papers")
    print(f"  Total entities: {total_entities}")
    print(f"  Total relationships: {total_relationships}")

    # Get final counts from storage
    doc_count = await document_storage.count()
    ent_count = await entity_storage.count()
    rel_count = await relationship_storage.count()

    print(f"\nFinal counts:")
    print(f"  Documents: {doc_count}")
    print(f"  Entities: {ent_count}")
    print(f"  Relationships: {rel_count}")

    # Create bundle
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple README for the bundle
    with TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        readme_file = temp_path / "README.md"
        readme_content = f"""# Medical Literature Knowledge Graph Bundle

This bundle contains extracted knowledge from biomedical journal articles.

## Statistics

- Papers processed: {processed}
- Total entities: {ent_count}
- Total relationships: {rel_count}

## Domain

- Domain: medlit
- Entity types: disease, gene, drug, protein, symptom, procedure, biomarker, pathway
- Relationship types: treats, causes, increases_risk, associated_with, interacts_with, etc.

## Source

Papers were processed from JSON format (med-lit-schema Paper format).
"""
        readme_file.write_text(readme_content)

        # Export the bundle
        await write_bundle(
            entity_storage=entity_storage,
            relationship_storage=relationship_storage,
            bundle_path=output_dir,
            domain="medlit",
            label="medical-literature",
            docs=temp_path,
            description="Knowledge graph bundle of biomedical journal articles",
        )

    print(f"\n✓ Bundle exported to: {output_dir}")
    print(f"  - manifest.json")
    print(f"  - entities.jsonl")
    print(f"  - relationships.jsonl")
    print(f"  - documents.jsonl (if docs provided)")
    print(f"  - docs/ (documentation)")


if __name__ == "__main__":
    asyncio.run(main())
