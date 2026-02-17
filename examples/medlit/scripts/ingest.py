#!/usr/bin/env python3
"""Ingestion script for medical literature knowledge graph.

Processes Paper JSON files (from med-lit-schema) and generates a kgraph bundle.

The pipeline has three stages:
    1. Entity Extraction (per-paper): Extract entities, most provisional initially
    2. Promotion (batch): De-duplicate and promote provisionals to canonical
    3. Relationship Extraction (per-paper): Extract relationships using canonical entities

Use --stop-after to halt at any stage and dump JSON to stdout for debugging/testing.

Usage:
    # Full pipeline
    python -m examples.medlit.scripts.ingest --input-dir /path/to/papers --output-dir medlit_bundle --use-ollama

    # Stop after entity extraction and dump JSON
    python -m examples.medlit.scripts.ingest --input-dir /path/to/papers --use-ollama --stop-after entities

    # Stop after promotion and dump JSON
    python -m examples.medlit.scripts.ingest --input-dir /path/to/papers --use-ollama --stop-after promotion
"""

import argparse
import asyncio
import fnmatch
import os
import sys
import time
import uuid
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from logging import DEBUG
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from kgraph.export import write_bundle
from kgraph.ingest import IngestionOrchestrator
from kgraph.logging import setup_logging
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.caching import (
    CachedEmbeddingGenerator,
    EmbeddingCacheConfig,
    FileBasedEmbeddingsCache,
)
from kgraph.pipeline.streaming import (
    BatchingEntityExtractor,
    WindowedRelationshipExtractor,
)
from kgraph.storage.memory import (
    InMemoryDocumentStorage,
    InMemoryEntityStorage,
    InMemoryRelationshipStorage,
)
from kgschema.storage import (
    DocumentStorageInterface,
    EntityStorageInterface,
    RelationshipStorageInterface,
)
from ..domain import MedLitDomainSchema
from ..pipeline.authority_lookup import CanonicalIdLookup
from ..pipeline.embeddings import OllamaMedLitEmbeddingGenerator
from ..pipeline.llm_client import LLMTimeoutError, OllamaLLMClient
from ..pipeline.mentions import MedLitEntityExtractor
from ..pipeline.config import load_medlit_config
from ..pipeline.pmc_chunker import PMCStreamingChunker
from ..pipeline.parser import JournalArticleParser
from ..pipeline.relationships import MedLitRelationshipExtractor
from ..pipeline.resolve import MedLitEntityResolver
from ..stage_models import (
    EntityExtractionStageResult,
    IngestionPipelineResult,
    PaperEntityExtractionResult,
    PaperRelationshipExtractionResult,
    PromotedEntityRecord,
    PromotionStageResult,
    RelationshipExtractionStageResult,
)

# Trace output base directory
TRACE_BASE_DIR = Path("/tmp/kgraph-traces")

logger = setup_logging()


@dataclass
class TraceCollector:
    """Collects paths to trace files written during ingestion.

    Each ingestion run gets a unique UUID, and trace files are organized as:
    /tmp/kgraph-traces/{run_id}/entities/{doc_id}.entities.trace.json
    /tmp/kgraph-traces/{run_id}/promotion/promotions.trace.json
    /tmp/kgraph-traces/{run_id}/relationships/{doc_id}.relationships.trace.json
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_files: list[Path] = field(default_factory=list)

    @property
    def trace_dir(self) -> Path:
        """Get the trace directory for this run."""
        return TRACE_BASE_DIR / self.run_id

    @property
    def entity_trace_dir(self) -> Path:
        """Get the entity trace directory for this run."""
        return self.trace_dir / "entities"

    @property
    def promotion_trace_dir(self) -> Path:
        """Get the promotion trace directory for this run."""
        return self.trace_dir / "promotion"

    @property
    def relationship_trace_dir(self) -> Path:
        """Get the relationship trace directory for this run."""
        return self.trace_dir / "relationships"

    def add(self, path: Path) -> None:
        """Add a trace file path."""
        self.trace_files.append(path)

    def collect_from_directory(self, directory: Path, pattern: str = "*.trace.json") -> None:
        """Collect all trace files matching pattern from a directory."""
        if directory.exists():
            for trace_file in sorted(directory.glob(pattern)):
                if trace_file not in self.trace_files:
                    self.trace_files.append(trace_file)

    def print_summary(self) -> None:
        """Print summary of all trace files written."""
        if not self.trace_files:
            return
        print(f"\nTrace files written (run_id: {self.run_id}):", file=sys.stderr)
        for trace_file in sorted(self.trace_files):
            print(f"  Wrote {trace_file}", file=sys.stderr)


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
        """Print progress report to stderr."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            rate = self.completed / elapsed
            remaining = (self.total - self.completed) / rate if rate > 0 else 0
            pct = (self.completed / self.total * 100) if self.total > 0 else 0

            print(f"\n  Progress: {self.completed}/{self.total} ({pct:.1f}%)", file=sys.stderr)
            print(f"    Elapsed: {elapsed/60:.1f} min", file=sys.stderr)
            if remaining > 0:
                print(f"    Estimated remaining: {remaining/60:.1f} min", file=sys.stderr)
            print(f"    Rate: {rate:.2f} papers/sec", file=sys.stderr)


def build_orchestrator(
    use_ollama: bool = False,
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    ollama_timeout: float = 300.0,
    cache_file: Path | None = None,
    relationship_trace_dir: Path | None = None,
    embeddings_cache_file: Path | None = None,
    evidence_validation_mode: str = "hybrid",
    evidence_similarity_threshold: float = 0.5,
) -> tuple[IngestionOrchestrator, CanonicalIdLookup | None, CachedEmbeddingGenerator | None]:
    """Builds and configures the ingestion orchestrator and its components.

    This function sets up the entire pipeline, including storage,
    extractors, resolvers, and the main orchestrator instance.

    Args:
        use_ollama: If True, initializes the Ollama LLM client for extraction tasks.
                    This is mandatory for the current entity and relationship
                    extraction strategies.
        ollama_model: The name of the Ollama model to use (e.g., "llama3.1:8b").
        ollama_host: The URL of the Ollama server.
        ollama_timeout: The timeout in seconds for requests to the Ollama server.
        cache_file: An optional path to a file for caching canonical ID lookups.
                    This is not used during initialization but passed for later use.
        relationship_trace_dir: Optional directory for writing relationship trace files.
                                If None, uses the default location.
        embeddings_cache_file: Optional path for a persistent embeddings cache (JSON).
                               If set, wraps the embedding generator with
                               CachedEmbeddingGenerator + FileBasedEmbeddingsCache.

    Returns:
        A tuple containing:
        - An instance of `IngestionOrchestrator` configured for the pipeline.
        - `None`, as the `CanonicalIdLookup` service is initialized later,
          just before the promotion phase.
        - The CachedEmbeddingGenerator if embeddings_cache_file was set, else None
          (caller should await cache.load() before use and save_cache() when done).
    """
    domain = MedLitDomainSchema()

    # Create storage instances first (needed for embedding extractor)
    entity_storage = InMemoryEntityStorage()
    relationship_storage = InMemoryRelationshipStorage()
    document_storage = InMemoryDocumentStorage()

    # Create embedding generator (used by resolver and for new-entity embeddings)
    base_embedding_generator = OllamaMedLitEmbeddingGenerator(ollama_host=ollama_host)
    cached_embedding_generator: CachedEmbeddingGenerator | None = None
    embedding_generator: EmbeddingGeneratorInterface
    if embeddings_cache_file is not None:
        embeddings_cache = FileBasedEmbeddingsCache(
            config=EmbeddingCacheConfig(cache_file=embeddings_cache_file),
        )
        cached_embedding_generator = CachedEmbeddingGenerator(
            base_generator=base_embedding_generator,
            cache=embeddings_cache,
        )
        embedding_generator = cached_embedding_generator
    else:
        embedding_generator = base_embedding_generator

    # Canonical ID lookup service will be created at the start of promotion phase
    # (stage 3) to load cache at that time, not during initialization

    # Create LLM client if requested (for relationship extraction)
    llm_client = None
    if use_ollama:
        try:
            print(f"  Initializing Ollama LLM: {ollama_model} at {ollama_host} (timeout: {ollama_timeout}s)...", file=sys.stderr)
            llm_client = OllamaLLMClient(model=ollama_model, host=ollama_host, timeout=ollama_timeout)
            print("  ✓ Ollama client created successfully", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to create Ollama client: {e}", file=sys.stderr)
            print("  Continuing without LLM extraction...", file=sys.stderr)
            import traceback

            traceback.print_exc()
            llm_client = None

    # Entity extraction always requires LLM (embedding extraction needs pre-populated storage)
    if not llm_client:
        raise ValueError("LLM extraction is required for entity extraction from XML. Use --use-ollama to enable LLM extraction, or provide documents with pre-extracted entities.")

    print("  Using LLM-based entity extraction...", file=sys.stderr)
    entity_extractor = MedLitEntityExtractor(llm_client=llm_client, domain=domain)
    print("  ✓ LLM-based extractor created", file=sys.stderr)

    # When mode is "string", disable semantic evidence validation (no embedding generator)
    rel_embedding_generator = (
        None if evidence_validation_mode == "string" else embedding_generator
    )
    relationship_extractor = MedLitRelationshipExtractor(
        llm_client=llm_client,
        trace_dir=relationship_trace_dir,
        embedding_generator=rel_embedding_generator,
        evidence_similarity_threshold=evidence_similarity_threshold,
    )
    chunker_cfg = load_medlit_config()["chunker"]
    document_chunker = PMCStreamingChunker(
        window_size=chunker_cfg["window_size"],
        overlap=chunker_cfg["overlap"],
    )
    streaming_entity_extractor = BatchingEntityExtractor(
        base_extractor=entity_extractor,
        deduplicate=True,
    )
    streaming_relationship_extractor = WindowedRelationshipExtractor(
        base_extractor=relationship_extractor,
    )

    orchestrator = IngestionOrchestrator(
        domain=domain,
        parser=JournalArticleParser(),
        entity_extractor=entity_extractor,
        entity_resolver=MedLitEntityResolver(
            domain=domain,
            embedding_generator=embedding_generator,  # Enable embedding-based resolution
            similarity_threshold=0.85,  # Require 85% similarity for matches
        ),
        relationship_extractor=relationship_extractor,
        embedding_generator=embedding_generator,
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        document_storage=document_storage,
        document_chunker=document_chunker,
        streaming_entity_extractor=streaming_entity_extractor,
        streaming_relationship_extractor=streaming_relationship_extractor,
    )

    return orchestrator, None, cached_embedding_generator


async def extract_entities_from_paper(
    orchestrator: IngestionOrchestrator,
    file_path: Path,
    content_type: str,
) -> tuple[str, int, int]:
    """Extracts entities from a single document file.

    This function reads a file, passes its content to the ingestion
    orchestrator's entity extraction pipeline, and handles any exceptions
    that occur during the process. It is designed to be called concurrently.

    Args:
        orchestrator: The configured `IngestionOrchestrator` instance.
        file_path: The `Path` to the input document (e.g., a JSON or XML file).
        content_type: The MIME type of the file, such as "application/json".

    Returns:
        A tuple containing:
        - The document ID (typically the file stem).
        - The number of entities successfully extracted.
        - The number of relationships extracted (will be 0 in this phase).
        Returns (file_stem, 0, 0) on failure.
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
    """Extracts relationships from a single document file.

    This function reads a file and passes its content to the ingestion
    orchestrator's relationship extraction pipeline. It is designed to be
    called concurrently after the entity promotion phase is complete.

    Args:
        orchestrator: The configured `IngestionOrchestrator` instance.
        file_path: The `Path` to the input document (e.g., a JSON or XML file).
        content_type: The MIME type of the file, such as "application/json".

    Returns:
        A tuple containing:
        - The document ID (typically the file stem).
        - The number of entities extracted (0 in this phase).
        - The number of relationships successfully extracted.
        Returns (file_stem, 0, 0) on failure.
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
    """Parses and validates command-line arguments for the ingestion script.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ingest medical literature papers and generate bundle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  entities      - Extract entities from papers (most provisional initially)
  promotion     - De-duplicate and promote provisionals to canonical
  relationships - Extract relationships using canonical entities

Examples:
  # Full pipeline with bundle output
  python -m examples.medlit.scripts.ingest --input-dir papers/ --use-ollama

  # Stop after entities and dump JSON to stdout
  python -m examples.medlit.scripts.ingest --input-dir papers/ --use-ollama --stop-after entities

  # Stop after promotion, save JSON to file
  python -m examples.medlit.scripts.ingest --input-dir papers/ --use-ollama --stop-after promotion > promotion_state.json

  # Process specific papers using glob patterns
  python -m examples.medlit.scripts.ingest --input-dir papers/ --use-ollama --input-papers 'PMC1234*.xml,PMC56*.xml'

  # Run full pipeline with trace files for debugging
  python -m examples.medlit.scripts.ingest --input-dir papers/ --use-ollama --trace-all
""",
    )
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
        "--stop-after",
        type=str,
        choices=["entities", "promotion", "relationships"],
        default=None,
        help="Stop after specified stage and dump JSON to stdout (for debugging/testing)",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (useful when piping JSON to file)",
    )
    parser.add_argument(
        "--input-papers",
        type=str,
        default=None,
        help="Comma-separated list of glob patterns to filter papers, e.g. 'PMC1234*.xml,PMC56*.xml'",
    )
    parser.add_argument(
        "--trace-all",
        action="store_true",
        help="Write trace JSON files for all stages (currently only relationship traces are implemented)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Logging is done at DEBUG level",
    )
    parser.add_argument(
        "--evidence-validation-mode",
        type=str,
        choices=("string", "semantic", "hybrid"),
        default="hybrid",
        help="Evidence validation: string (exact match only), semantic (embedding similarity), hybrid (string then semantic fallback). Default: hybrid",
    )
    parser.add_argument(
        "--evidence-similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity for semantic evidence validation (default: 0.5)",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.debug:
        logger.setLevel(DEBUG)
        stream_log = logging.getLogger("kgraph.pipeline.streaming")
        stream_log.setLevel(DEBUG)
        if not stream_log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s"))
            stream_log.addHandler(handler)
    return args


def find_input_files(
    input_dir: Path,
    limit: int | None,
    input_papers: str | None = None,
) -> list[tuple[Path, str]]:
    """Finds all processable JSON and XML files in the input directory.

    Args:
        input_dir: The directory to search for input files.
        limit: An optional integer to limit the number of files returned.
        input_papers: Optional comma-separated glob patterns to filter files,
                      e.g. 'PMC1234*.xml,PMC56*.xml'

    Returns:
        A sorted list of tuples, where each tuple contains:
        - A `Path` object for a found file.
        - A string with the file's MIME content type.
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

    # Filter by glob patterns if specified
    if input_papers:
        patterns = [p.strip() for p in input_papers.split(",")]
        filtered: list[tuple[Path, str]] = []
        for file_path, content_type in files:
            for pattern in patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    filtered.append((file_path, content_type))
                    break
        files = filtered

    if limit:
        files = files[:limit]
    return files


async def extract_entities_phase(
    orchestrator: IngestionOrchestrator,
    input_files: list[tuple[Path, str]],
    max_workers: int = 1,
    progress_interval: float = 30.0,
    quiet: bool = False,
    trace_all: bool = False,  # noqa: ARG001 - reserved for future use
) -> tuple[int, int, EntityExtractionStageResult]:
    """Coordinates the entity extraction phase for all input files.

    This function manages the concurrent execution of the entity extraction
    process across multiple files, using a semaphore to limit parallelism.
    It also tracks and reports progress.

    Args:
        orchestrator: The configured `IngestionOrchestrator` instance.
        input_files: A list of file paths and their content types to process.
        max_workers: The maximum number of concurrent extraction tasks.
        progress_interval: The interval in seconds for reporting progress.
        quiet: If True, suppress progress output.
        trace_all: If True, write per-paper entity trace files.
                   TODO: Entity tracing not yet implemented. Would write to
                   /tmp/kgraph-entity-traces/{doc_id}.entities.trace.json

    Returns:
        A tuple containing:
        - The number of files processed successfully.
        - The number of files that resulted in errors.
        - EntityExtractionStageResult with detailed results.
    """
    workers_msg = f" (workers: {max_workers})" if max_workers > 1 else ""
    if not quiet:
        print(f"\n[2/5] Extracting entities from all papers{workers_msg}...", file=sys.stderr)

    tracker = ProgressTracker(total=len(input_files), report_interval=progress_interval)
    results: list[tuple[str, int, int, str | None, tuple[str, ...]]] = []

    async def extract_with_progress(file_path: Path, content_type: str) -> tuple[str, int, int, str | None, tuple[str, ...]]:
        result = await extract_entities_from_paper(orchestrator, file_path, content_type)
        tracker.increment()
        # Return (doc_id, entities, relationships, source_uri, errors)
        return (*result, str(file_path), ())

    if max_workers > 1:
        # Parallel extraction with semaphore
        semaphore = asyncio.Semaphore(max_workers)

        async def extract_with_limit(file_path: Path, content_type: str) -> tuple[str, int, int, str | None, tuple[str, ...]]:
            async with semaphore:
                return await extract_with_progress(file_path, content_type)

        results = await asyncio.gather(*[extract_with_limit(file_path, content_type) for file_path, content_type in input_files])
    else:
        # Sequential extraction
        for file_path, content_type in input_files:
            results.append(await extract_with_progress(file_path, content_type))

    # Final report
    if not quiet:
        tracker.report()

    processed = 0
    errors_count = 0
    paper_results: list[PaperEntityExtractionResult] = []
    total_new = 0
    total_existing = 0

    for paper_id, entities, _, source_uri, errs in results:
        if entities > 0:
            if not quiet:
                print(f"  {paper_id}: {entities} entities", file=sys.stderr)
            processed += 1
            # For now, we don't have detailed new/existing breakdown per paper
            # This would require modifying extract_entities_from_paper
            total_new += entities  # Approximation
        else:
            errors_count += 1

        paper_results.append(
            PaperEntityExtractionResult(
                document_id=paper_id,
                source_uri=source_uri,
                extracted_at=datetime.now(timezone.utc),
                entities_extracted=entities,
                entities_new=entities,  # Approximation
                entities_existing=0,
                entities=(),  # Would need to collect from orchestrator
                errors=errs,
            )
        )

    # Build entity type counts from storage
    entity_type_counts: dict[str, int] = {}
    all_entities = await orchestrator.entity_storage.list_all()
    for entity in all_entities:
        etype = entity.get_entity_type()
        entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

    provisional_count = len(await orchestrator.entity_storage.list_all(status="provisional"))
    canonical_count = len(await orchestrator.entity_storage.list_all(status="canonical"))

    stage_result = EntityExtractionStageResult(
        stage="entities",
        completed_at=datetime.now(timezone.utc),
        papers_processed=processed,
        papers_failed=errors_count,
        total_entities_extracted=sum(r.entities_extracted for r in paper_results),
        total_entities_new=total_new,
        total_entities_existing=total_existing,
        paper_results=tuple(paper_results),
        entity_type_counts=entity_type_counts,
        provisional_count=provisional_count,
        canonical_count=canonical_count,
    )

    return (processed, errors_count, stage_result)


def _initialize_lookup(
    use_ollama: bool,
    cache_file: Path | None,
    quiet: bool,
    embedding_generator: Any = None,
) -> CanonicalIdLookup | None:
    """Initializes the canonical ID lookup service."""
    if not use_ollama:
        return None

    try:
        cache_path = cache_file or Path("canonical_id_cache.json")
        if not quiet:
            print(f"  Initializing canonical ID lookup (cache: {cache_path})...", file=sys.stderr)
        lookup = CanonicalIdLookup(
            cache_file=cache_path,
            embedding_generator=embedding_generator,
            similarity_threshold=0.5,
        )
        if not quiet:
            print("  ✓ Canonical ID lookup created successfully", file=sys.stderr)
        return lookup
    except Exception as e:
        if not quiet:
            print(f"  Warning: Failed to create canonical ID lookup: {e}", file=sys.stderr)
            print("  Continuing without canonical ID lookup...", file=sys.stderr)
        return None


def _build_promoted_records(promoted: list) -> list[PromotedEntityRecord]:
    """Builds a list of promoted entity records from a list of promoted entities."""
    promoted_records: list[PromotedEntityRecord] = []
    for entity in promoted:
        # Determine canonical source from canonical_ids
        canonical_source = "unknown"
        canonical_url = None
        if entity.canonical_ids:
            canonical_source = next(iter(entity.canonical_ids.keys()), "unknown")
        if entity.metadata.get("canonical_url"):
            canonical_url = entity.metadata["canonical_url"]

        promoted_records.append(
            PromotedEntityRecord(
                old_entity_id=entity.metadata.get("old_entity_id", entity.entity_id),
                new_entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.get_entity_type(),
                canonical_source=canonical_source,
                canonical_url=canonical_url,
            )
        )
    return promoted_records


async def run_promotion_phase(
    orchestrator: IngestionOrchestrator,
    entity_storage: EntityStorageInterface,
    cache_file: Path | None = None,
    use_ollama: bool = False,
    quiet: bool = False,
    trace_all: bool = False,  # noqa: ARG001 - reserved for future use
) -> tuple[CanonicalIdLookup | None, PromotionStageResult]:
    """Coordinates the entity promotion phase."""
    if not quiet:
        print("\n[3/5] Running entity promotion...", file=sys.stderr)

    embedding_generator = getattr(orchestrator, "embedding_generator", None)
    lookup = _initialize_lookup(use_ollama, cache_file, quiet, embedding_generator=embedding_generator)

    all_provisional = await entity_storage.list_all(status="provisional")
    candidates_count = len(all_provisional)
    if not quiet:
        print(f"      Found {candidates_count} provisional entities", file=sys.stderr)
        if all_provisional:
            print("      Sample entity stats:", file=sys.stderr)
            for entity in all_provisional[:5]:
                print(
                    f"        - {entity.name}: usage={entity.usage_count}, confidence={entity.confidence:.2f}",
                    file=sys.stderr,
                )
        config = orchestrator.domain.promotion_config
        print(
            f"      Promotion thresholds: usage>={config.min_usage_count}, confidence>={config.min_confidence}",
            file=sys.stderr,
        )

    promoted = await orchestrator.run_promotion(lookup=lookup)
    promoted_records = _build_promoted_records(promoted)

    if not quiet:
        print(f"      Promoted {len(promoted)} entities to canonical status:", file=sys.stderr)
        for entity in promoted:
            print(f"        - {entity.name} → {entity.entity_id}", file=sys.stderr)

    if lookup:
        try:
            lookup._save_cache(force=True)  # pylint: disable=protected-access
            cache_path = lookup.cache_file.absolute()
            if not quiet:
                print(f"  ✓ Canonical ID cache saved to: {cache_path}", file=sys.stderr)
        except Exception as e:
            if not quiet:
                print(f"  Warning: Failed to save cache: {e}", file=sys.stderr)

    final_canonical = len(await entity_storage.list_all(status="canonical"))
    final_provisional = len(await entity_storage.list_all(status="provisional"))

    stage_result = PromotionStageResult(
        stage="promotion",
        completed_at=datetime.now(timezone.utc),
        candidates_evaluated=candidates_count,
        entities_promoted=len(promoted),
        entities_skipped_no_canonical_id=candidates_count - len(promoted),
        entities_skipped_policy=0,
        entities_skipped_storage_failure=0,
        promoted_entities=tuple(promoted_records),
        total_canonical_entities=final_canonical,
        total_provisional_entities=final_provisional,
    )

    return lookup, stage_result


async def extract_relationships_phase(
    orchestrator: IngestionOrchestrator,
    input_files: list[tuple[Path, str]],
    max_workers: int = 1,
    progress_interval: float = 30.0,
    quiet: bool = False,
    trace_all: bool = False,  # noqa: ARG001 - traces written by MedLitRelationshipExtractor
) -> tuple[int, int, RelationshipExtractionStageResult]:
    """Coordinates the relationship extraction phase for all input files.

    This function manages the concurrent execution of the relationship
    extraction process, which runs after entities have been promoted. It uses
    a semaphore to limit parallelism and reports progress.

    Note: Relationship traces are always written by MedLitRelationshipExtractor
    to /tmp/kgraph-relationship-traces/{doc_id}.relationships.trace.json

    Args:
        orchestrator: The configured `IngestionOrchestrator` instance.
        input_files: A list of file paths and their content types to process.
        max_workers: The maximum number of concurrent extraction tasks.
        progress_interval: The interval in seconds for reporting progress.
        quiet: If True, suppress progress output.
        trace_all: Reserved for consistency; relationship traces are always written.

    Returns:
        A tuple containing:
        - The number of files for which relationships were extracted.
        - The number of files that resulted in errors.
        - RelationshipExtractionStageResult with detailed results.
    """
    workers_msg = f" (workers: {max_workers})" if max_workers > 1 else ""
    if not quiet:
        print(f"\n[4/5] Extracting relationships from all papers{workers_msg}...", file=sys.stderr)

    tracker = ProgressTracker(total=len(input_files), report_interval=progress_interval)
    results: list[tuple[str, int, int, str | None]] = []

    async def extract_with_progress(file_path: Path, content_type: str) -> tuple[str, int, int, str | None]:
        result = await extract_relationships_from_paper(orchestrator, file_path, content_type)
        tracker.increment()
        return (*result, str(file_path))

    if max_workers > 1:
        # Parallel extraction with semaphore
        semaphore = asyncio.Semaphore(max_workers)

        async def extract_with_limit(file_path: Path, content_type: str) -> tuple[str, int, int, str | None]:
            async with semaphore:
                return await extract_with_progress(file_path, content_type)

        results = await asyncio.gather(*[extract_with_limit(file_path, content_type) for file_path, content_type in input_files])
    else:
        # Sequential extraction
        for file_path, content_type in input_files:
            results.append(await extract_with_progress(file_path, content_type))

    # Final report
    if not quiet:
        tracker.report()

    processed_rels = 0
    errors_rels = 0
    paper_results: list[PaperRelationshipExtractionResult] = []
    total_relationships = 0

    for paper_id, _, relationships, source_uri in results:
        if relationships > 0:
            if not quiet:
                print(f"  {paper_id}: {relationships} relationships", file=sys.stderr)
            processed_rels += 1
            total_relationships += relationships
        else:
            errors_rels += 1

        paper_results.append(
            PaperRelationshipExtractionResult(
                document_id=paper_id,
                source_uri=source_uri,
                extracted_at=datetime.now(timezone.utc),
                relationships_extracted=relationships,
                relationships=(),  # Would need to collect from orchestrator
                errors=(),
            )
        )

    # Build predicate counts from storage
    predicate_counts: dict[str, int] = {}
    all_relationships = await orchestrator.relationship_storage.list_all()
    for rel in all_relationships:
        predicate_counts[rel.predicate] = predicate_counts.get(rel.predicate, 0) + 1

    stage_result = RelationshipExtractionStageResult(
        stage="relationships",
        completed_at=datetime.now(timezone.utc),
        papers_processed=len(results),
        papers_with_relationships=processed_rels,
        total_relationships_extracted=total_relationships,
        paper_results=tuple(paper_results),
        predicate_counts=predicate_counts,
    )

    return (processed_rels, errors_rels, stage_result)


async def print_summary(
    document_storage: DocumentStorageInterface,
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
    quiet: bool = False,
) -> None:
    """Prints a formatted summary of the knowledge graph's contents.

    This function queries the storage interfaces to get counts of documents,
    entities (total, canonical, and provisional), and relationships, then
    displays them in a table.

    Args:
        document_storage: The storage interface for documents.
        entity_storage: The storage interface for entities.
        relationship_storage: The storage interface for relationships.
        quiet: If True, suppress output.
    """
    if quiet:
        return

    print("\n[5/5] Summary", file=sys.stderr)
    doc_count = await document_storage.count()
    ent_count = await entity_storage.count()
    rel_count = await relationship_storage.count()

    # Show canonical vs provisional breakdown
    canonical_count = len(await entity_storage.list_all(status="canonical"))
    provisional_count = len(await entity_storage.list_all(status="provisional"))

    print(
        f"""
    ╔══════════════════════════════════════╗
    ║       Knowledge Graph Summary        ║
    ╠══════════════════════════════════════╣
    ║  Documents:               {doc_count:>10} ║
    ║  Entities (total):        {ent_count:>10} ║
    ║    - Canonical:           {canonical_count:>10} ║
    ║    - Provisional:         {provisional_count:>10} ║
    ║  Relationships:           {rel_count:>10} ║
    ╚══════════════════════════════════════╝
    """,
        file=sys.stderr,
    )


async def export_bundle(
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
    output_dir: Path,
    processed: int,
    errors: int,
    cache_file: Path | None = None,
) -> None:
    """Exports the final knowledge graph to a bundle directory.

    This function uses `kgraph.export.write_bundle` to serialize the entities
    and relationships from storage into JSONL files. It also creates a
    README, a manifest, and copies the canonical ID cache into the bundle.

    Args:
        entity_storage: The storage interface for entities.
        relationship_storage: The storage interface for relationships.
        output_dir: The path to the directory where the bundle will be written.
        processed: The number of papers successfully processed, for metadata.
        errors: The number of papers that failed, for metadata.
        cache_file: An optional path to the canonical ID cache file to be
                    included in the bundle.
    """
    print(
        f"""
[6/6] Exporting bundle...
      Processed: {processed} papers
      Errors/skipped: {errors} papers
""",
        file=sys.stderr,
    )

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
            print(f"  ✓ Copied canonical ID cache to bundle: {cache_file.name}", file=sys.stderr)

    cache_note = f"\n     - {cache_file.name}" if cache_file and cache_file.exists() else ""
    print(
        f"""
✓ Bundle exported to: {output_dir}
     - manifest.json
     - entities.jsonl
     - relationships.jsonl
     - doc_assets.jsonl (if docs provided)
     - docs/ (documentation){cache_note}
""",
        file=sys.stderr,
    )


def _handle_keyboard_interrupt(lookup: CanonicalIdLookup | None) -> None:
    """Handles graceful shutdown on KeyboardInterrupt (Ctrl+C).

    This function is registered as an exception handler to ensure that the
    canonical ID lookup cache is saved before the program exits, preventing
    loss of work.

    Args:
        lookup: The `CanonicalIdLookup` instance, which contains the cache
                to be saved.
    """
    print("\n  Interrupted by user (Ctrl+C)", file=sys.stderr)
    if not lookup:
        return

    try:
        # Debug: show cache state
        metrics = lookup._cache.get_metrics()  # pylint: disable=protected-access
        print(f"  Cache state: {metrics['total_entries']} total entries, {metrics['hits']} hits, {metrics['misses']} misses", file=sys.stderr)

        lookup._save_cache(force=True)  # pylint: disable=protected-access
        cache_path = lookup.cache_file.absolute()

        # Verify file exists
        if cache_path.exists():
            file_size = cache_path.stat().st_size
            print(f"  ✓ Canonical ID cache saved to: {cache_path} ({file_size} bytes)", file=sys.stderr)
        else:
            print(f"  ✗ Cache file not found at: {cache_path}", file=sys.stderr)
            print("     This should not happen - save may have failed silently", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ Warning: Failed to save cache: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        print(f"     Cache file location: {lookup.cache_file.absolute()}", file=sys.stderr)


async def _cleanup_lookup_service(lookup: CanonicalIdLookup | None) -> None:
    """Closes resources associated with the lookup service.

    This function is called in a `finally` block to ensure that the
    underlying HTTP client in the `CanonicalIdLookup` service is closed
    gracefully, regardless of whether the pipeline succeeded or failed.
    The cache is saved separately and not handled here.

    Args:
        lookup: The `CanonicalIdLookup` instance to clean up.
    """
    if not lookup:
        return

    try:
        await lookup.close()
    except Exception as e:
        print(f"  ✗ Warning: Failed to close lookup service: {e}", file=sys.stderr)


def _output_stage_result(result: BaseModel, stage_name: str, quiet: bool) -> None:
    """Output stage result as JSON to stdout."""
    if not quiet:
        print(f"\n[STOP] Pipeline stopped after '{stage_name}' stage. JSON output follows on stdout.\n", file=sys.stderr)

    # Output JSON to stdout (for piping to file or further processing)
    print(result.model_dump_json(indent=2))


def _initialize_pipeline(args: argparse.Namespace) -> tuple:
    """Initializes the pipeline and returns necessary components."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quiet = args.quiet or (args.stop_after is not None)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    input_files = find_input_files(input_dir, args.limit, args.input_papers)
    if not input_files:
        if args.input_papers:
            print(f"ERROR: No files matching '{args.input_papers}' found in {input_dir}", file=sys.stderr)
        else:
            print(f"ERROR: No JSON or XML files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    trace_collector = TraceCollector()
    if args.trace_all:
        trace_collector.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_collector.entity_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_collector.promotion_trace_dir.mkdir(parents=True, exist_ok=True)
        trace_collector.relationship_trace_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"  Trace mode enabled (run_id: {trace_collector.run_id})", file=sys.stderr)
            print(f"  Trace directory: {trace_collector.trace_dir}", file=sys.stderr)

    if not quiet:
        sep = "=" * 60
        lim = f"(Limited to {args.limit} papers)\n" if args.limit else ""
        filter_msg = f"(Filtered by: {args.input_papers})\n" if args.input_papers else ""
        json_count = sum(1 for _, ct in input_files if ct == "application/json")
        xml_count = sum(1 for _, ct in input_files if ct == "application/xml")
        print(
            f"""{sep}
Medical Literature Knowledge Graph - Ingestion Pipeline
{sep}

Input directory: {input_dir}
Found {len(input_files)} paper(s) to process ({json_count} JSON, {xml_count} XML)
{filter_msg}{lim}
[1/5] Initializing pipeline...""",
            file=sys.stderr,
        )

    pipeline_started_at = datetime.now(timezone.utc)
    cache_file = Path(args.cache_file) if args.cache_file else None
    embeddings_cache_file = output_dir / "embeddings_cache.json"

    orchestrator, lookup, cached_embedding_generator = build_orchestrator(
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        ollama_timeout=args.ollama_timeout,
        cache_file=cache_file,
        relationship_trace_dir=trace_collector.relationship_trace_dir if args.trace_all else None,
        embeddings_cache_file=embeddings_cache_file,
        evidence_validation_mode=args.evidence_validation_mode,
        evidence_similarity_threshold=args.evidence_similarity_threshold,
    )

    return (
        orchestrator,
        lookup,
        cached_embedding_generator,
        input_files,
        output_dir,
        quiet,
        trace_collector,
        pipeline_started_at,
        cache_file,
    )


async def main() -> None:  # pylint: disable=too-many-statements
    """Runs the main ingestion pipeline for medical literature."""
    args = parse_arguments()
    (
        orchestrator,
        lookup,
        cached_embedding_generator,
        input_files,
        output_dir,
        quiet,
        trace_collector,
        pipeline_started_at,
        cache_file,
    ) = _initialize_pipeline(args)

    if cached_embedding_generator is not None:
        await cached_embedding_generator.cache.load()
        if not quiet:
            stats = cached_embedding_generator.get_cache_stats()
            print(f"  Embeddings cache loaded ({stats.get('size', 0)} entries)", file=sys.stderr)

    entity_storage = orchestrator.entity_storage
    relationship_storage = orchestrator.relationship_storage
    document_storage = orchestrator.document_storage

    entity_stage_result: EntityExtractionStageResult | None = None
    promotion_stage_result: PromotionStageResult | None = None
    relationship_stage_result: RelationshipExtractionStageResult | None = None
    llm_timeout_abort = False

    try:
        processed, errors, entity_stage_result = await extract_entities_phase(
            orchestrator,
            input_files,
            max_workers=args.workers,
            progress_interval=args.progress_interval,
            quiet=quiet,
            trace_all=args.trace_all,
        )

        if args.trace_all:
            trace_collector.collect_from_directory(trace_collector.entity_trace_dir, "*.entities.trace.json")

        if args.stop_after == "entities":
            if args.trace_all:
                trace_collector.print_summary()
            pipeline_result = IngestionPipelineResult(
                started_at=pipeline_started_at,
                completed_at=datetime.now(timezone.utc),
                stopped_at_stage="entities",
                entity_extraction=entity_stage_result,
                promotion=None,
                relationship_extraction=None,
                total_documents=len(input_files),
                total_entities=await entity_storage.count(),
                total_relationships=0,
            )
            _output_stage_result(pipeline_result, "entities", quiet)
            return

        lookup, promotion_stage_result = await run_promotion_phase(
            orchestrator,
            entity_storage,
            cache_file=cache_file,
            use_ollama=args.use_ollama,
            quiet=quiet,
            trace_all=args.trace_all,
        )

        if args.trace_all:
            trace_collector.collect_from_directory(trace_collector.promotion_trace_dir, "*.trace.json")

        if args.stop_after == "promotion":
            if args.trace_all:
                trace_collector.print_summary()
            pipeline_result = IngestionPipelineResult(
                started_at=pipeline_started_at,
                completed_at=datetime.now(timezone.utc),
                stopped_at_stage="promotion",
                entity_extraction=entity_stage_result,
                promotion=promotion_stage_result,
                relationship_extraction=None,
                total_documents=len(input_files),
                total_entities=await entity_storage.count(),
                total_relationships=0,
            )
            _output_stage_result(pipeline_result, "promotion", quiet)
            return

        _, _, relationship_stage_result = await extract_relationships_phase(
            orchestrator,
            input_files,
            max_workers=args.workers,
            progress_interval=args.progress_interval,
            quiet=quiet,
            trace_all=args.trace_all,
        )

        if args.trace_all:
            trace_collector.collect_from_directory(trace_collector.relationship_trace_dir, "*.relationships.trace.json")

        if args.stop_after == "relationships":
            if args.trace_all:
                trace_collector.print_summary()
            pipeline_result = IngestionPipelineResult(
                started_at=pipeline_started_at,
                completed_at=datetime.now(timezone.utc),
                stopped_at_stage="relationships",
                entity_extraction=entity_stage_result,
                promotion=promotion_stage_result,
                relationship_extraction=relationship_stage_result,
                total_documents=len(input_files),
                total_entities=await entity_storage.count(),
                total_relationships=await relationship_storage.count(),
            )
            _output_stage_result(pipeline_result, "relationships", quiet)
            return

        await print_summary(document_storage, entity_storage, relationship_storage, quiet=quiet)
        cache_file_path = lookup.cache_file if lookup else cache_file
        await export_bundle(entity_storage, relationship_storage, output_dir, processed, errors, cache_file=cache_file_path)

        if args.trace_all:
            trace_collector.print_summary()

    except (LLMTimeoutError, TimeoutError) as e:
        llm_timeout_abort = True
        print(
            """
╔══════════════════════════════════════════════════════════════════════════════╗
║  LLM TIMEOUT — INGESTION ABORTED                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  The LLM (Ollama) request exceeded the timeout. This run is invalid.         ║
║  No bundle or caches will be written from this point. Exiting.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""",
            file=sys.stderr,
        )
        print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        _handle_keyboard_interrupt(lookup)
        raise
    finally:
        if not llm_timeout_abort and cached_embedding_generator is not None:
            await cached_embedding_generator.save_cache()
            if not quiet:
                stats = cached_embedding_generator.get_cache_stats()
                print(f"  Embeddings cache saved ({stats.get('size', 0)} entries)", file=sys.stderr)
        await _cleanup_lookup_service(lookup)


if __name__ == "__main__":
    asyncio.run(main())
