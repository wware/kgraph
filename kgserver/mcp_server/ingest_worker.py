"""
Background worker for paper ingestion jobs.

Processes jobs from a queue: fetch URL, run Pass 1/2/3 pipeline, load bundle incrementally.
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import httpx
from sqlmodel import Session

from query.storage_factory import get_engine
from storage.backends.postgres import PostgresStorage
from storage.backends.sqlite import SQLiteStorage
from storage.interfaces import StorageInterface

logger = logging.getLogger(__name__)

_job_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_tasks: list[asyncio.Task] = []
_shutdown = False

# PMC article URL pattern (e.g. https://www.ncbi.nlm.nih.gov/articles/PMC12345/)
_PMC_URL_RE = re.compile(r"PMC(\d+)", re.IGNORECASE)
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _storage_for_worker() -> tuple[StorageInterface, Callable[[], None]]:
    """Return (storage, close_fn) for use in the worker. Caller must call close_fn when done."""
    engine, db_url = get_engine()
    if db_url.startswith("postgresql://"):
        session = Session(engine)
        storage: StorageInterface = PostgresStorage(session)
        return storage, session.close
    db_path = db_url.replace("sqlite:///", "").strip() or "./test.db"
    if db_path == ":memory:" or not db_path:
        db_path = "./test.db"
    storage = SQLiteStorage(db_path)
    return storage, storage.close


async def start_worker(max_workers: int = 1) -> None:
    """Start background worker tasks that process the ingest job queue."""
    global _shutdown
    _shutdown = False
    for _ in range(max_workers):
        task = asyncio.create_task(_worker_loop())
        _worker_tasks.append(task)
    logger.info("Started %s ingest worker(s)", max_workers)


async def stop_worker() -> None:
    """Cancel worker tasks and stop processing new jobs."""
    global _shutdown
    _shutdown = True
    for task in _worker_tasks:
        task.cancel()
    for task in _worker_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    _worker_tasks.clear()
    logger.info("Stopped ingest workers")


async def enqueue_job(job_id: str) -> None:
    """Add a job id to the queue. Non-blocking."""
    await _job_queue.put(job_id)


async def _worker_loop() -> None:
    """Pull job IDs from the queue and run ingestion."""
    while not _shutdown:
        try:
            job_id = await _job_queue.get()
            try:
                await _run_ingest_job(job_id)
            finally:
                _job_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("Worker loop error: %s", e)


async def _run_ingest_job(job_id: str) -> None:
    """Fetch job, run pipeline (Pass 1 → 2 → 3), load bundle incrementally, update job."""
    storage, close_storage = _storage_for_worker()
    try:
        job = storage.get_ingest_job(job_id)
        if job is None:
            logger.warning("Ingest job not found: %s", job_id)
            return

        timeout_seconds = int(os.environ.get("INGEST_TIMEOUT", "600"))
        try:
            await asyncio.wait_for(_run_ingest_job_impl(job_id, storage, job), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=f"Job timed out after {timeout_seconds}s",
            )
            logger.error("Ingest job %s timed out", job_id)
        except Exception as e:
            logger.exception("Ingest job %s failed: %s", job_id, e)
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=str(e),
            )
    finally:
        close_storage()


async def _run_ingest_job_impl(job_id: str, storage: StorageInterface, job) -> None:
    """Core implementation: fetch URL, run pipeline, load bundle."""
    url = job.url
    logger.info("Ingest job %s starting: url=%s", job_id, url)
    storage.update_ingest_job(job_id, status="running", started_at=datetime.now(timezone.utc))

    pmcid = None
    paper_title = None
    entities_added = 0
    relationships_added = 0

    with TemporaryDirectory(prefix="ingest_") as tmp:
        root = Path(tmp)
        input_dir = root / "input"
        bundles_dir = root / "bundles"
        merged_dir = root / "merged"
        output_dir = root / "output"
        input_dir.mkdir()
        bundles_dir.mkdir()
        merged_dir.mkdir()
        output_dir.mkdir()

        # Fetch URL content
        t_fetch_start = time.perf_counter()
        match = _PMC_URL_RE.search(url)
        if match:
            pmcid_num = match.group(1)
            pmcid = f"PMC{pmcid_num}"
            efetch_url = f"{EFETCH_URL}?db=pmc&id={pmcid_num}&rettype=xml"
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(efetch_url)
                resp.raise_for_status()
                input_path = input_dir / "paper.xml"
                input_path.write_bytes(resp.content)
            logger.info("Ingest job %s: fetched PMC XML in %.1fs, %s bytes", job_id, time.perf_counter() - t_fetch_start, len(resp.content))
        else:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                suffix = ".xml" if "xml" in (resp.headers.get("content-type") or "").lower() else ".json"
                input_path = input_dir / f"paper{suffix}"
                input_path.write_bytes(resp.content)
            logger.info("Ingest job %s: fetched URL in %.1fs, %s bytes -> %s", job_id, time.perf_counter() - t_fetch_start, len(resp.content), input_path.name)

        # Pass 1
        from examples.medlit.scripts.pass1_extract import run_pass1

        llm_backend = os.environ.get("PASS1_LLM_BACKEND", "anthropic")
        logger.info("Ingest job %s: starting Pass 1 (backend=%s)", job_id, llm_backend)
        t0 = time.perf_counter()
        await run_pass1(input_dir, bundles_dir, llm_backend, limit=1)
        pass1_sec = time.perf_counter() - t0

        bundle_files = sorted(bundles_dir.glob("paper_*.json"))
        logger.info("Ingest job %s: Pass 1 done in %.1fs, bundle_files=%s", job_id, pass1_sec, [p.name for p in bundle_files])
        if not bundle_files:
            logger.warning("Ingest job %s: Pass 1 produced no bundle files", job_id)
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error="Pass 1 produced no bundle files (LLM may have failed, timed out, or returned invalid JSON).",
            )
            return

        # Optionally set paper_title from first Pass 1 bundle
        if bundle_files:
            try:
                import json
                data = json.loads(bundle_files[0].read_text(encoding="utf-8"))
                paper = data.get("paper") or {}
                if paper.get("title"):
                    paper_title = paper["title"]
            except Exception:
                pass

        # Pass 2 (sync — run in thread)
        from examples.medlit.pipeline.dedup import run_pass2

        logger.info("Ingest job %s: starting Pass 2", job_id)
        t0 = time.perf_counter()
        pass2_result = await asyncio.to_thread(
            run_pass2,
            bundle_dir=bundles_dir,
            output_dir=merged_dir,
            synonym_cache_path=merged_dir / "synonym_cache.json",
            canonical_id_cache_path=None,
        )
        pass2_sec = time.perf_counter() - t0
        logger.info("Ingest job %s: Pass 2 done in %.1fs, result: %s", job_id, pass2_sec, pass2_result)
        if isinstance(pass2_result, dict) and pass2_result.get("error"):
            logger.warning("Ingest job %s: Pass 2 returned error: %s", job_id, pass2_result.get("error"))
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=f"Pass 2 failed: {pass2_result['error']}",
            )
            return
        id_map_path = merged_dir / "id_map.json"
        if not id_map_path.exists():
            logger.warning("Ingest job %s: id_map.json missing after Pass 2", job_id)
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error="Pass 2 did not produce id_map.json (entity merge step failed or produced no output).",
            )
            return

        # Pass 3 (sync — run in thread)
        from examples.medlit.pipeline.bundle_builder import run_pass3

        logger.info("Ingest job %s: starting Pass 3", job_id)
        try:
            t0 = time.perf_counter()
            await asyncio.to_thread(run_pass3, merged_dir, bundles_dir, output_dir)
            pass3_sec = time.perf_counter() - t0
            logger.info("Ingest job %s: Pass 3 done in %.1fs", job_id, pass3_sec)
        except FileNotFoundError as e:
            logger.warning("Ingest job %s: Pass 3 FileNotFoundError: %s", job_id, e)
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=f"Pass 3 failed (missing input): {e}",
            )
            return

        # Load bundle incrementally with a fresh storage session
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("Ingest job %s: manifest.json missing after Pass 3", job_id)
            storage.update_ingest_job(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error="Pass 3 did not produce manifest.json",
            )
            return

        from kgbundle import BundleManifestV1

        manifest_data = manifest_path.read_text()
        manifest = BundleManifestV1.model_validate_json(manifest_data)
        logger.info("Ingest job %s: loading bundle incrementally (bundle_id=%s)", job_id, manifest.bundle_id)

        load_storage, load_close = _storage_for_worker()
        try:
            load_storage.load_bundle_incremental(manifest, str(output_dir))
        finally:
            load_close()
        logger.info("Ingest job %s: bundle loaded", job_id)

        if manifest.entities and manifest.entities.path:
            entities_file = output_dir / manifest.entities.path
            if entities_file.exists():
                entities_added = sum(1 for _ in entities_file.open())
        if manifest.relationships and manifest.relationships.path:
            rels_file = output_dir / manifest.relationships.path
            if rels_file.exists():
                relationships_added = sum(1 for _ in rels_file.open())

        storage.update_ingest_job(
            job_id,
            status="complete",
            completed_at=datetime.now(timezone.utc),
            paper_title=paper_title,
            pmcid=pmcid,
            entities_added=entities_added,
            relationships_added=relationships_added,
            error=None,
        )
        logger.info(
            "Ingest job %s complete: %s entities, %s relationships (Pass1=%.1fs Pass2=%.1fs Pass3=%.1fs)",
            job_id, entities_added, relationships_added, pass1_sec, pass2_sec, pass3_sec,
        )
