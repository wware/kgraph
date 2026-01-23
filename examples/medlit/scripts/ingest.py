#!/usr/bin/env python3
"""Ingestion script for medical literature knowledge graph.

Processes Paper JSON files (from med-lit-schema) and generates a kgraph bundle.

Usage:
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle --limit 10
"""

import argparse
import asyncio
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
from ..pipeline.embeddings import OllamaMedLitEmbeddingGenerator
from ..pipeline.llm_client import OllamaLLMClient


def build_orchestrator(
    use_ollama: bool = False,
    ollama_model: str = "meditron:70b",
    ollama_host: str = "http://localhost:11434",
) -> IngestionOrchestrator:
    """Build the ingestion orchestrator for medical literature domain.

    Args:
        use_ollama: If True, use Ollama LLM for entity and relationship extraction.
        ollama_model: Ollama model name (e.g., "meditron:70b").
        ollama_host: Ollama server URL.

    Returns:
        Configured IngestionOrchestrator instance.
    """
    domain = MedLitDomainSchema()

    # Create LLM client if requested
    llm_client = None
    if use_ollama:
        try:
            print(f"  Initializing Ollama LLM: {ollama_model} at {ollama_host}...")
            llm_client = OllamaLLMClient(model=ollama_model, host=ollama_host)
            print(f"  ✓ Ollama client created successfully")
        except Exception as e:
            print(f"  Warning: Failed to create Ollama client: {e}")
            print("  Continuing without LLM extraction...")
            import traceback
            traceback.print_exc()
            llm_client = None

    return IngestionOrchestrator(
        domain=domain,
        parser=JournalArticleParser(),
        entity_extractor=MedLitEntityExtractor(llm_client=llm_client),
        entity_resolver=MedLitEntityResolver(domain=domain),
        relationship_extractor=MedLitRelationshipExtractor(llm_client=llm_client),
        embedding_generator=OllamaMedLitEmbeddingGenerator(),
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
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use Ollama LLM for entity and relationship extraction",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="meditron:70b",
        help="Ollama model name (default: meditron:70b)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
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

    sep = "=" * 60
    lim = f"(Limited to {args.limit} papers)\n" if args.limit else ""
    print(f"""{sep}
Medical Literature Knowledge Graph - Ingestion Pipeline
{sep}

Input directory: {input_dir}
Found {len(json_files)} paper(s) to process
{lim}

[1/5] Initializing pipeline...""")
    orchestrator = build_orchestrator(
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )
    entity_storage = orchestrator.entity_storage
    relationship_storage = orchestrator.relationship_storage
    document_storage = orchestrator.document_storage

    print("\n[2/5] Ingesting papers...")
    total_entities = 0
    total_relationships = 0
    processed = 0
    errors = 0

    for json_file in json_files:
        _, entities, relationships = await ingest_paper_json(orchestrator, json_file)
        if entities > 0 or relationships > 0:
            print(f"  {json_file.name}: {entities} entities, {relationships} relationships")
            total_entities += entities
            total_relationships += relationships
            processed += 1
        else:
            errors += 1

    print(f"""
[3/5] Running entity promotion...""")

    # Debug: Check what provisional entities we have
    all_provisional = await entity_storage.list_all(status="provisional")
    print(f"      Found {len(all_provisional)} provisional entities")

    # Debug: Show their usage counts and confidence
    if all_provisional:
        print("      Sample entity stats:")
        for entity in all_provisional[:5]:  # Show first 5
            print(f"        - {entity.name}: usage={entity.usage_count}, confidence={entity.confidence:.2f}")

    # Debug: Check the thresholds
    config = orchestrator.domain.promotion_config
    print(f"      Promotion thresholds: usage>={config.min_usage_count}, confidence>={config.min_confidence}")

    promoted = await orchestrator.run_promotion()
    print(f"      Promoted {len(promoted)} entities to canonical status:")
    for entity in promoted:
        print(f"        - {entity.name} → {entity.entity_id}")

    print(f"""
[4/5] Summary""")
    doc_count = await document_storage.count()
    ent_count = await entity_storage.count()
    rel_count = await relationship_storage.count()

    # Show canonical vs provisional breakdown
    canonical_count = len(await entity_storage.list_all(status="canonical"))
    provisional_count = len(await entity_storage.list_all(status="provisional"))

    print(f"""
    ╔══════════════════════════════════════╗
    ║       Knowledge Graph Summary        ║
    ╠══════════════════════════════════════╣
    ║  Documents:               {doc_count:>10} ║
    ║  Entities (total):        {ent_count:>10} ║
    ║    - Canonical:           {canonical_count:>10} ║
    ║    - Provisional:         {provisional_count:>10} ║
    ║  Relationships:           {rel_count:>10} ║
    ╚══════════════════════════════════════╝
    """)

    print(f"""
[5/5] Exporting bundle...
      Processed: {processed} papers
      Errors/skipped: {errors} papers
""")

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

    print(f"""
✓ Bundle exported to: {output_dir}
     - manifest.json
     - entities.jsonl
     - relationships.jsonl
     - documents.jsonl (if docs provided)
     - docs/ (documentation)
""")


if __name__ == "__main__":
    asyncio.run(main())
