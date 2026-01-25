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
from kgraph.storage.interfaces import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from kgraph.export import write_bundle

from ..domain import MedLitDomainSchema
from ..pipeline.authority_lookup import CanonicalIdLookup
from ..pipeline.parser import JournalArticleParser
from ..pipeline.mentions import MedLitEntityExtractor
from ..pipeline.resolve import MedLitEntityResolver
from ..pipeline.relationships import MedLitRelationshipExtractor
from ..pipeline.embeddings import OllamaMedLitEmbeddingGenerator
from ..pipeline.llm_client import OllamaLLMClient


def build_orchestrator(
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    cache_file: Path | None = None,
) -> tuple[IngestionOrchestrator, CanonicalIdLookup | None]:
    """Build the ingestion orchestrator for medical literature domain.

    Args:
        use_ollama: If True, use Ollama LLM for entity and relationship extraction.
        ollama_model: Ollama model name (e.g., "llama3.1:8b").
        ollama_host: Ollama server URL.
        cache_file: Optional path for canonical ID lookup cache file.

    Returns:
        Tuple of (IngestionOrchestrator, CanonicalIdLookup or None).
        The lookup is returned so it can be properly closed after use.
    """
    domain = MedLitDomainSchema()

    # Create canonical ID lookup service for promotion phase
    # Note: We've learned not to use this during extraction (tool calling is too slow),
    # but it still contains useful information and is used during the promotion phase
    # to assign canonical IDs to provisional entities.
    lookup = None
    if use_ollama:
        try:
            cache_path = cache_file or Path("canonical_id_cache.json")
            print(f"  Initializing canonical ID lookup (cache: {cache_path})...")
            lookup = CanonicalIdLookup(cache_file=cache_path)
            print("  ✓ Canonical ID lookup created successfully")
        except Exception as e:
            print(f"  Warning: Failed to create canonical ID lookup: {e}")
            print("  Continuing without canonical ID lookup...")
            lookup = None

    # Create LLM client if requested
    llm_client = None
    if use_ollama:
        try:
            print(f"  Initializing Ollama LLM: {ollama_model} at {ollama_host}...")
            llm_client = OllamaLLMClient(model=ollama_model, host=ollama_host)
            print("  ✓ Ollama client created successfully")
        except Exception as e:
            print(f"  Warning: Failed to create Ollama client: {e}")
            print("  Continuing without LLM extraction...")
            import traceback

            traceback.print_exc()
            llm_client = None

    orchestrator = IngestionOrchestrator(
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

    return orchestrator, lookup


async def extract_entities_from_paper(
    orchestrator: IngestionOrchestrator,
    json_path: Path,
) -> tuple[str, int, int]:
    """Extract entities from a single Paper JSON file.

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

        # Extract entities from the paper
        result = await orchestrator.extract_entities_from_document(
            raw_content=raw_content,
            content_type="application/json",
            source_uri=str(json_path),
        )

        return (result.document_id, result.entities_extracted, result.relationships_extracted)

    except Exception as e:
        print(f"  ERROR processing {json_path.name}: {e}")
        return (json_path.stem, 0, 0)


async def extract_relationships_from_paper(
    orchestrator: IngestionOrchestrator,
    json_path: Path,
) -> tuple[str, int, int]:
    """Extract relationships from a single Paper JSON file.

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

        # Extract relationships from the paper
        result = await orchestrator.extract_relationships_from_document(
            raw_content=raw_content,
            content_type="application/json",
            source_uri=str(json_path),
        )

        return (result.document_id, result.entities_extracted, result.relationships_extracted)

    except Exception as e:
        print(f"  ERROR processing {json_path.name}: {e}")
        return (json_path.stem, 0, 0)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Path to canonical ID lookup cache file (default: canonical_id_cache.json)",
    )
    return parser.parse_args()


def find_json_files(input_dir: Path, limit: int | None) -> list[Path]:
    """Find and return JSON files to process."""
    json_files = sorted(input_dir.glob("*.json"))
    if limit:
        json_files = json_files[:limit]
    return json_files


async def extract_entities_phase(
    orchestrator: IngestionOrchestrator,
    json_files: list[Path],
) -> tuple[int, int]:
    """Extract entities from all papers. Returns (processed, errors)."""
    print("\n[2/5] Extracting entities from all papers...")
    processed = 0
    errors = 0

    for json_file in json_files:
        _, entities, _ = await extract_entities_from_paper(orchestrator, json_file)
        if entities > 0:
            print(f"  {json_file.name}: {entities} entities")
            processed += 1
        else:
            errors += 1

    return (processed, errors)


async def run_promotion_phase(
    orchestrator: IngestionOrchestrator,
    entity_storage: EntityStorageInterface,
    lookup: CanonicalIdLookup | None = None,
) -> None:
    """Run entity promotion and print results."""
    print("""
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

    promoted = await orchestrator.run_promotion(lookup=lookup)
    print(f"      Promoted {len(promoted)} entities to canonical status:")
    for entity in promoted:
        print(f"        - {entity.name} → {entity.entity_id}")


async def extract_relationships_phase(
    orchestrator: IngestionOrchestrator,
    json_files: list[Path],
) -> tuple[int, int]:
    """Extract relationships from all papers. Returns (processed, errors)."""
    print("""
[4/5] Extracting relationships from all papers...""")
    processed_rels = 0
    errors_rels = 0

    for json_file in json_files:
        _, _, relationships = await extract_relationships_from_paper(orchestrator, json_file)
        if relationships > 0:
            print(f"  {json_file.name}: {relationships} relationships")
            processed_rels += 1
        else:
            errors_rels += 1

    return (processed_rels, errors_rels)


async def print_summary(
    document_storage: DocumentStorageInterface,
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
) -> None:
    """Print the knowledge graph summary."""
    print("""
[5/5] Summary""")
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


async def export_bundle(
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
    output_dir: Path,
    processed: int,
    errors: int,
    cache_file: Path | None = None,
) -> None:
    """Export the knowledge graph bundle."""
    print(f"""
[6/6] Exporting bundle...
      Processed: {processed} papers
      Errors/skipped: {errors} papers
""")

    # Create bundle
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple README for the bundle
    with TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        readme_file = temp_path / "README.md"
        ent_count = await entity_storage.count()
        rel_count = await relationship_storage.count()
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

        # Copy cache file to bundle if it exists
        if cache_file and cache_file.exists():
            import shutil

            bundle_cache_path = output_dir / cache_file.name
            shutil.copy2(cache_file, bundle_cache_path)
            print(f"  ✓ Copied canonical ID cache to bundle: {cache_file.name}")

    cache_note = f"\n     - {cache_file.name}" if cache_file and cache_file.exists() else ""
    print(f"""
✓ Bundle exported to: {output_dir}
     - manifest.json
     - entities.jsonl
     - relationships.jsonl
     - documents.jsonl (if docs provided)
     - docs/ (documentation){cache_note}
""")


def _handle_keyboard_interrupt(lookup: CanonicalIdLookup | None) -> None:
    """Handle KeyboardInterrupt by saving cache before exiting."""
    print("\n  Interrupted by user (Ctrl+C)")
    if not lookup:
        return

    try:
        # Debug: show cache state
        total_entries = len(lookup._cache)  # pylint: disable=protected-access
        successful_entries = len([v for v in lookup._cache.values() if v != "NULL"])  # pylint: disable=protected-access
        print(f"  Cache state: {total_entries} total entries, {successful_entries} successful lookups")

        lookup._save_cache(force=True)  # pylint: disable=protected-access
        cache_path = lookup.cache_file.absolute()

        # Verify file exists
        if cache_path.exists():
            file_size = cache_path.stat().st_size
            print(f"  ✓ Canonical ID cache saved to: {cache_path} ({file_size} bytes)")
        else:
            print(f"  ✗ Cache file not found at: {cache_path}")
            print("     This should not happen - save may have failed silently")
    except Exception as e:
        print(f"  ✗ Warning: Failed to save cache: {e}")
        import traceback

        traceback.print_exc()
        print(f"     Cache file location: {lookup.cache_file.absolute()}")


async def _cleanup_lookup_service(lookup: CanonicalIdLookup | None) -> None:
    """Clean up lookup service and save cache."""
    if not lookup:
        return

    try:
        await lookup.close()
        cache_path = lookup.cache_file.absolute()
        print(f"  ✓ Canonical ID cache saved to: {cache_path}")
    except Exception:
        # If close() fails, at least try to save cache (force save as safety)
        try:
            lookup._save_cache(force=True)  # pylint: disable=protected-access
            cache_path = lookup.cache_file.absolute()
            print(f"  ✓ Canonical ID cache saved to: {cache_path} (emergency save)")
        except Exception as e:
            print(f"  ✗ Failed to save cache: {e}")
            print(f"     Cache file location: {lookup.cache_file.absolute()}")


async def main() -> None:
    """Main ingestion function."""
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    json_files = find_json_files(input_dir, args.limit)
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

    cache_file = Path(args.cache_file) if args.cache_file else None
    orchestrator, lookup = build_orchestrator(
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        cache_file=cache_file,
    )
    entity_storage = orchestrator.entity_storage
    relationship_storage = orchestrator.relationship_storage
    document_storage = orchestrator.document_storage

    try:
        processed, errors = await extract_entities_phase(orchestrator, json_files)
        await run_promotion_phase(orchestrator, entity_storage, lookup=lookup)
        await extract_relationships_phase(orchestrator, json_files)
        await print_summary(document_storage, entity_storage, relationship_storage)
        cache_file_path = lookup.cache_file if lookup else cache_file
        await export_bundle(entity_storage, relationship_storage, output_dir, processed, errors, cache_file=cache_file_path)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(lookup)
        raise
    finally:
        await _cleanup_lookup_service(lookup)


if __name__ == "__main__":
    asyncio.run(main())
