#!/usr/bin/env python3
"""Ingestion script for medical literature knowledge graph.

Processes Paper JSON files (from med-lit-schema) and generates a kgraph bundle.

Usage:
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle
    python -m examples.medlit.scripts.ingest --input-dir /path/to/json_papers --output-dir medlit_bundle --limit 10
"""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from kgraph.ingest import IngestionOrchestrator
from kgraph.logging import setup_logging
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

logger = setup_logging()


@dataclass
class ProgressTracker:
    """Track and report progress during long-running operations."""

    total: int
    completed: int = 0
    start_time: float = field(default_factory=time.time)
    last_report_time: float = field(default_factory=time.time)
    report_interval: float = 30.0  # Report every 30 seconds

    def increment(self) -> None:
        """Increment completed count and report if interval elapsed."""
        self.completed += 1
        now = time.time()
        if now - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = now

    def report(self) -> None:
        """Print progress report."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.completed / elapsed
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            pct = (self.completed / self.total * 100) if self.total > 0 else 0

            print(f"\n  Progress: {self.completed}/{self.total} ({pct:.1f}%)")
            print(f"    Elapsed: {elapsed/60:.1f} min")
            if remaining > 0:
                print(f"    Estimated remaining: {remaining/60:.1f} min")
            print(f"    Rate: {rate:.2f} papers/sec")


def build_orchestrator(
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    ollama_timeout: float = 300.0,
    cache_file: Path | None = None,
) -> tuple[IngestionOrchestrator, CanonicalIdLookup | None]:
    """Build the ingestion orchestrator for medical literature domain.

    Args:
        use_ollama: If True, use Ollama LLM for entity and relationship extraction.
                    Required for extracting entities from XML (embedding extraction
                    requires pre-populated entity storage).
        ollama_model: Ollama model name (e.g., "llama3.1:8b").
        ollama_host: Ollama server URL.
        ollama_timeout: Timeout in seconds for Ollama requests (default: 300).
        cache_file: Optional path for canonical ID lookup cache file.

    Returns:
        Tuple of (IngestionOrchestrator, None).
        The lookup service is created during the promotion phase, not during initialization.
    """
    domain = MedLitDomainSchema()

    # Create storage instances first (needed for embedding extractor)
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    # Create embedding generator (used by both extractors)
    embedding_generator = OllamaMedLitEmbeddingGenerator(ollama_host=ollama_host)

    # Canonical ID lookup service will be created at the start of promotion phase
    # (stage 3) to load cache at that time, not during initialization

    # Create LLM client if requested (for relationship extraction)
    llm_client = None
    if use_ollama:
        try:
            print(f"  Initializing Ollama LLM: {ollama_model} at {ollama_host} (timeout: {ollama_timeout}s)...")
            llm_client = OllamaLLMClient(model=ollama_model, host=ollama_host, timeout=ollama_timeout)
            print("  ✓ Ollama client created successfully")
        except Exception as e:
            print(f"  Warning: Failed to create Ollama client: {e}")
            print("  Continuing without LLM extraction...")
            import traceback

            traceback.print_exc()
            llm_client = None

    # Entity extraction always requires LLM (embedding extraction needs pre-populated storage)
    if not llm_client:
        raise ValueError("LLM extraction is required for entity extraction from XML. Use --use-ollama to enable LLM extraction, or provide documents with pre-extracted entities.")

    print("  Using LLM-based entity extraction...")
    entity_extractor = MedLitEntityExtractor(llm_client=llm_client, domain=domain)
    print("  ✓ LLM-based extractor created")

    orchestrator = IngestionOrchestrator(
        domain=domain,
        parser=JournalArticleParser(),
        entity_extractor=entity_extractor,
        entity_resolver=MedLitEntityResolver(
            domain=domain,
            embedding_generator=embedding_generator,  # Enable embedding-based resolution
            similarity_threshold=0.85,  # Require 85% similarity for matches
        ),
        relationship_extractor=MedLitRelationshipExtractor(llm_client=llm_client),
        embedding_generator=embedding_generator,
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
    )

    return orchestrator, None


async def extract_entities_from_paper(
    orchestrator: IngestionOrchestrator,
    file_path: Path,
    content_type: str,
) -> tuple[str, int, int]:
    """Extract entities from a single paper file (JSON or XML).

    Args:
        orchestrator: The ingestion orchestrator.
        file_path: Path to the paper file (JSON or XML).
        content_type: MIME type ("application/json" or "application/xml").

    Returns:
        Tuple of (paper_id, entities_extracted, relationships_extracted).
    """
    try:
        # Read file
        with open(file_path, "rb") as f:
            raw_content = f.read()

        # Extract entities from the paper
        result = await orchestrator.extract_entities_from_document(
            raw_content=raw_content,
            content_type=content_type,
            source_uri=str(file_path),
        )
        if result.errors:
            logger.error(result.errors)
            return (file_path.stem, 0, 0)

        return (result.document_id, result.entities_extracted, result.relationships_extracted)

    except Exception:
        logger.exception(file_path.name)
        return (file_path.stem, 0, 0)


async def extract_relationships_from_paper(
    orchestrator: IngestionOrchestrator,
    file_path: Path,
    content_type: str,
) -> tuple[str, int, int]:
    """Extract relationships from a single paper file (JSON or XML).

    Args:
        orchestrator: The ingestion orchestrator.
        file_path: Path to the paper file (JSON or XML).
        content_type: MIME type ("application/json" or "application/xml").

    Returns:
        Tuple of (paper_id, entities_extracted, relationships_extracted).
    """
    try:
        # Read file
        with open(file_path, "rb") as f:
            raw_content = f.read()

        # Extract relationships from the paper
        result = await orchestrator.extract_relationships_from_document(
            raw_content=raw_content,
            content_type=content_type,
            source_uri=str(file_path),
        )

        return (result.document_id, result.entities_extracted, result.relationships_extracted)

    except Exception as e:
        print(f"  ERROR processing {file_path.name}: {e}")
        return (file_path.stem, 0, 0)


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
        help="Use Ollama LLM for relationship extraction (entity extraction uses embeddings by default)",
    )
    parser.add_argument(
        "--use-llm-extraction",
        action="store_true",
        default=True,
        help="Use LLM for entity extraction (required for XML input; embedding extraction needs pre-populated storage)",
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
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: from OLLAMA_HOST env var or http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds for Ollama requests (default: 300)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Path to canonical ID lookup cache file (default: canonical_id_cache.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for extraction (default: 1)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Progress report interval in seconds (default: 30)",
    )
    return parser.parse_args()


def find_input_files(input_dir: Path, limit: int | None) -> list[tuple[Path, str]]:
    """Find and return input files to process (JSON or XML).

    Returns:
        List of (file_path, content_type) tuples.
    """
    files: list[tuple[Path, str]] = []
    # Find JSON files
    json_files = sorted(input_dir.glob("*.json"))
    files.extend((f, "application/json") for f in json_files)
    # Find XML files
    xml_files = sorted(input_dir.glob("*.xml"))
    files.extend((f, "application/xml") for f in xml_files)
    # Sort by filename
    files.sort(key=lambda x: x[0].name)
    if limit:
        files = files[:limit]
    return files


async def extract_entities_phase(
    orchestrator: IngestionOrchestrator,
    input_files: list[tuple[Path, str]],
    max_workers: int = 1,
    progress_interval: float = 30.0,
) -> tuple[int, int]:
    """Extract entities from all papers. Returns (processed, errors)."""
    workers_msg = f" (workers: {max_workers})" if max_workers > 1 else ""
    print(f"\n[2/5] Extracting entities from all papers{workers_msg}...")

    tracker = ProgressTracker(total=len(input_files), report_interval=progress_interval)
    results: list[tuple[str, int, int]] = []

    async def extract_with_progress(file_path: Path, content_type: str) -> tuple[str, int, int]:
        result = await extract_entities_from_paper(orchestrator, file_path, content_type)
        tracker.increment()
        return result

    if max_workers > 1:
        # Parallel extraction with semaphore
        semaphore = asyncio.Semaphore(max_workers)

        async def extract_with_limit(file_path: Path, content_type: str) -> tuple[str, int, int]:
            async with semaphore:
                return await extract_with_progress(file_path, content_type)

        results = await asyncio.gather(*[extract_with_limit(file_path, content_type) for file_path, content_type in input_files])
    else:
        # Sequential extraction
        for file_path, content_type in input_files:
            results.append(await extract_with_progress(file_path, content_type))

    # Final report
    tracker.report()

    processed = 0
    errors = 0
    for paper_id, entities, _ in results:
        if entities > 0:
            print(f"  {paper_id}: {entities} entities")
            processed += 1
        else:
            errors += 1

    return (processed, errors)


async def run_promotion_phase(
    orchestrator: IngestionOrchestrator,
    entity_storage: EntityStorageInterface,
    cache_file: Path | None = None,
    use_ollama: bool = False,
) -> CanonicalIdLookup | None:
    """Run entity promotion and print results.

    Creates the canonical ID lookup service at the start (loading cache),
    runs promotion, saves cache immediately after completion, and returns
    the lookup service for cleanup.

    Returns:
        CanonicalIdLookup instance if created, None otherwise.
    """
    print("""
[3/5] Running entity promotion...""")

    # Create lookup service at START of promotion phase (loads cache now)
    lookup: CanonicalIdLookup | None = None
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

    # Save cache IMMEDIATELY after promotion phase completes
    if lookup:
        try:
            lookup._save_cache(force=True)  # pylint: disable=protected-access
            cache_path = lookup.cache_file.absolute()
            print(f"  ✓ Canonical ID cache saved to: {cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")

    return lookup


async def extract_relationships_phase(
    orchestrator: IngestionOrchestrator,
    input_files: list[tuple[Path, str]],
    max_workers: int = 1,
    progress_interval: float = 30.0,
) -> tuple[int, int]:
    """Extract relationships from all papers. Returns (processed, errors)."""
    workers_msg = f" (workers: {max_workers})" if max_workers > 1 else ""
    print(f"""
[4/5] Extracting relationships from all papers{workers_msg}...""")

    tracker = ProgressTracker(total=len(input_files), report_interval=progress_interval)
    results: list[tuple[str, int, int]] = []

    async def extract_with_progress(file_path: Path, content_type: str) -> tuple[str, int, int]:
        result = await extract_relationships_from_paper(orchestrator, file_path, content_type)
        tracker.increment()
        return result

    if max_workers > 1:
        # Parallel extraction with semaphore
        semaphore = asyncio.Semaphore(max_workers)

        async def extract_with_limit(file_path: Path, content_type: str) -> tuple[str, int, int]:
            async with semaphore:
                return await extract_with_progress(file_path, content_type)

        results = await asyncio.gather(*[extract_with_limit(file_path, content_type) for file_path, content_type in input_files])
    else:
        # Sequential extraction
        for file_path, content_type in input_files:
            results.append(await extract_with_progress(file_path, content_type))

    # Final report
    tracker.report()

    processed_rels = 0
    errors_rels = 0
    for paper_id, _, relationships in results:
        if relationships > 0:
            print(f"  {paper_id}: {relationships} relationships")
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
        metrics = lookup._cache.get_metrics()  # pylint: disable=protected-access
        print(f"  Cache state: {metrics['total_entries']} total entries, {metrics['hits']} hits, {metrics['misses']} misses")

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
    """Clean up lookup service.

    Note: Cache is already saved immediately after promotion phase,
    so this just closes the HTTP client.
    """
    if not lookup:
        return

    try:
        await lookup.close()
    except Exception as e:
        print(f"  ✗ Warning: Failed to close lookup service: {e}")


async def main() -> None:
    """Main ingestion function."""
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return

    input_files = find_input_files(input_dir, args.limit)
    if not input_files:
        print(f"ERROR: No JSON or XML files found in {input_dir}")
        return

    sep = "=" * 60
    lim = f"(Limited to {args.limit} papers)\n" if args.limit else ""
    json_count = sum(1 for _, ct in input_files if ct == "application/json")
    xml_count = sum(1 for _, ct in input_files if ct == "application/xml")
    print(f"""{sep}
Medical Literature Knowledge Graph - Ingestion Pipeline
{sep}

Input directory: {input_dir}
Found {len(input_files)} paper(s) to process ({json_count} JSON, {xml_count} XML)
{lim}

[1/5] Initializing pipeline...""")

    cache_file = Path(args.cache_file) if args.cache_file else None
    orchestrator, lookup = build_orchestrator(
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
        cache_file=cache_file,
    )
    entity_storage = orchestrator.entity_storage
    relationship_storage = orchestrator.relationship_storage
    document_storage = orchestrator.document_storage

    try:
        processed, errors = await extract_entities_phase(
            orchestrator,
            input_files,
            max_workers=args.workers,
            progress_interval=args.progress_interval,
        )
        lookup = await run_promotion_phase(orchestrator, entity_storage, cache_file=cache_file, use_ollama=args.use_ollama)
        await extract_relationships_phase(
            orchestrator,
            input_files,
            max_workers=args.workers,
            progress_interval=args.progress_interval,
        )
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
