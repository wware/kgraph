# PLAN9: MCP tools for paper ingestion

Implement two MCP tools: `ingest_paper(url)` (async, enqueues job, returns job_id) and `check_ingest_status(job_id)` (sync, returns status and counts). Background jobs run the medlit Pass 1 → Pass 2 → Pass 3 pipeline in a temp directory, then load the resulting bundle into storage. Job state is persisted in the same DB as the graph (Postgres or SQLite) so status survives restarts.

---

## Prerequisites

- Repo root on `sys.path` or install in editable mode so `examples.medlit` and `kgserver` are importable from the process that runs the worker.
- Environment: `DATABASE_URL` (and optionally `PASS1_LLM_BACKEND`, `INGEST_MAX_WORKERS`, etc.) set where the MCP server runs.
- For PMC URLs, no extra auth; E-utilities are public.

---

## Step 1. Add `IngestJob` model

**New file:** `kgserver/storage/models/ingest_job.py`

- Define a single table model (SQLModel, `table=True`) with:
  - `__tablename__ = "ingest_jobs"`
  - `id: str` (primary key, UUID string)
  - `url: str`
  - `status: str = "queued"`  # one of: queued | running | complete | failed
  - `created_at: datetime`
  - `started_at: Optional[datetime] = None`
  - `completed_at: Optional[datetime] = None`
  - `paper_title: Optional[str] = None`
  - `pmcid: Optional[str] = None`
  - `entities_added: int = 0`
  - `relationships_added: int = 0`
  - `error: Optional[str] = None`
- Use `Field(primary_key=True)` for `id` and `Field(default=...)` where appropriate.

**Edit:** `kgserver/storage/models/__init__.py`

- Import `IngestJob` from `.ingest_job` and add `"IngestJob"` to `__all__`.

---

## Step 2. Extend storage interface with job CRUD

**Edit:** `kgserver/storage/interfaces.py`

- Add to `StorageInterface` three abstract methods:
  - `def create_ingest_job(self, url: str) -> IngestJob`
  - `def get_ingest_job(self, job_id: str) -> Optional[IngestJob]`
  - `def update_ingest_job(self, job_id: str, **fields: Any) -> Optional[IngestJob]`
- Use `TYPE_CHECKING` and forward reference or import `IngestJob` from `.models.ingest_job` (or from `.models` if exported there). `update_ingest_job` should accept keyword arguments that match `IngestJob` column names, update the row, commit/flush, and return the updated model or None if not found.

---

## Step 3. Implement job CRUD in Postgres and SQLite backends

**Edit:** `kgserver/storage/backends/postgres.py`

- Implement `create_ingest_job(url: str)`: build `IngestJob(id=uuid.uuid4().hex, url=url, status="queued", created_at=datetime.now(timezone.utc), ...)`, add to session, commit, return the model.
- Implement `get_ingest_job(job_id: str)`: query by primary key, return model or None.
- Implement `update_ingest_job(job_id: str, **fields)`: get row by id; if not found return None; for each key in `fields` that exists on the model, set the attribute; commit; return the updated model.

**Edit:** `kgserver/storage/backends/sqlite.py`

- Implement the same three methods for SQLite. Reuse the same semantics (create with UUID, get by id, update by allowed kwargs). Ensure the SQLite backend uses a session/connection that supports commits.

---

## Step 4. Incremental bundle load (no truncate, upsert)

Current `load_bundle` in Postgres (and SQLite) truncates entity/relationship tables then loads one bundle; that would wipe the graph when loading a second paper. For ingest we need an incremental load that does not truncate and that merges new data (e.g. entity usage_count accumulation).

**Relationship upsert constraint:** The `Relationship` model uses `id: UUID` as its primary key and has no unique constraint on the triple by default. A raw SQL `ON CONFLICT (subject_id, predicate, object_id)` upsert will fail at the DB level unless that constraint exists. Add it to the model (preferred) so the backend can use a single upsert; the alternative is a SELECT-then-INSERT-or-UPDATE pattern in `load_bundle_incremental`.

**Edit:** `kgserver/storage/models/relationship.py`

- Ensure the table has a unique constraint on the triple so `ON CONFLICT` can be used. Add (or align) `__table_args__`:
  - `from sqlalchemy import UniqueConstraint` (or `from sqlmodel import UniqueConstraint` if available).
  - `__table_args__ = (UniqueConstraint("subject_id", "predicate", "object_id", name="uq_relationship_triple"),)`.
- If the model already has a unique constraint on the triple (possibly with different column order or name), ensure the backend’s `ON CONFLICT (...)` clause uses the same columns in the same order as the constraint.

**Edit:** `kgserver/storage/interfaces.py`

- Add to `StorageInterface`: `def load_bundle_incremental(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None` with a short docstring: “Load a bundle into the graph without truncating; add/merge entities and relationships (e.g. upsert entities and accumulate usage_count).”

**Edit:** `kgserver/storage/backends/postgres.py`

- Implement `load_bundle_incremental(bundle_manifest, bundle_path)`:
  - Do **not** truncate. Do **not** check `is_bundle_loaded` to skip (incremental loads are additive).
  - Load entities: for each entity row, upsert by `entity_id` (e.g. `INSERT ... ON CONFLICT (entity_id) DO UPDATE SET usage_count = entity.usage_count + excluded.usage_count, ...`). Ensure all columns needed for the entity table are set in both INSERT and UPDATE. Use the same normalization as `_load_entities` (e.g. `_normalize_entity`) so that `canonical_url` and other fields are correct.
  - Load relationships: upsert by `(subject_id, predicate, object_id)` (or the actual primary key of the relationship table). If the schema has a single composite key, use `ON CONFLICT (subject_id, predicate, object_id) DO UPDATE SET ...` as needed; if no counters to accumulate, a simple DO NOTHING or DO UPDATE SET updated_at = excluded.updated_at is enough.
  - Load provenance (mentions, evidence) the same way as in `load_bundle` (append new rows; if tables have unique constraints, use ON CONFLICT as appropriate).
  - Call `record_bundle(bundle_manifest)` and commit.
- Keep existing `load_bundle` behavior unchanged for initial/full load (truncate + load).

**Edit:** `kgserver/storage/backends/sqlite.py`

- Implement `load_bundle_incremental` with the same semantics (no truncate; upsert entities with usage_count accumulation; upsert relationships; append/upsert provenance; record_bundle).

---

## Step 5. Background worker (queue + run_ingest_job)

**New file:** `kgserver/mcp_server/ingest_worker.py`

- Use a single asyncio queue and one or more worker tasks that pull job IDs and run ingestion.
- Public API:
  - `async def start_worker(max_workers: int = 1) -> None` — start `max_workers` background tasks that run `_worker_loop`.
  - `async def stop_worker() -> None` — cancel worker tasks and drain the queue (no new work; finish current job if desired).
  - `async def enqueue_job(job_id: str) -> None` — put `job_id` on the queue (non-blocking).
- Internal:
  - `_job_queue: asyncio.Queue[str]` (module-level).
  - `_worker_tasks: list[asyncio.Task]` (or a single task that runs a loop).
  - `async def _worker_loop() -> None`: in a loop, `job_id = await _job_queue.get()`, then `await _run_ingest_job(job_id)`; call `_job_queue.task_done()`.
  - `async def _run_ingest_job(job_id: str) -> None` — see Step 6.

Ensure the worker is started only once (e.g. from lifespan) and that the same queue is used for all enqueue calls.

---

## Step 6. Implement `_run_ingest_job` (pipeline + load)

**In:** `kgserver/mcp_server/ingest_worker.py`

- Get storage (or engine) via `get_engine()` from `query.storage_factory`. Open a new `Session(engine)` (and build `PostgresStorage(session)` or the appropriate backend) for the whole job — do not use the FastAPI request-scoped dependency.
- Fetch the job by `job_id`; if not found, log and return. Set `status="running"`, `started_at=now`, persist via `update_ingest_job`.
- **Fetch URL content:**
  - If URL is a PMC article URL (e.g. matches `r'/articles/(PMC\d+)/'` or similar), extract PMCID and fetch XML from `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&rettype=xml`. Use `httpx.AsyncClient` with a reasonable timeout (e.g. 60s). Save response body to a temp file (e.g. `input_dir/paper.xml`). Detect PMC URLs by a regex like `PMC\d+` in the path.
  - Otherwise treat as a direct URL: GET the URL, determine content type from Content-Type or extension (.xml / .json), save to `input_dir/paper.xml` or `paper.json`.
- **Temp directory:** Create a `TemporaryDirectory`. Use subdirs: e.g. `input_dir`, `bundles_dir`, `merged_dir`, `output_dir` inside it.
- **Pass 1:** Call `run_pass1(input_dir, bundles_dir, llm_backend, limit=1)`. Use `llm_backend = os.environ.get("PASS1_LLM_BACKEND", "anthropic")`. Get LLM via `from examples.medlit.pipeline.pass1_llm import get_pass1_llm` and `get_pass1_llm(backend=llm_backend)`. Pass 1 writes per-paper JSON into `bundles_dir`; for a single paper there will be one `paper_*.json`.
- **Pass 2:** Call `run_pass2(bundle_dir=bundles_dir, output_dir=merged_dir, synonym_cache_path=merged_dir / "synonym_cache.json", canonical_id_cache_path=...)`. Use a cache path under the temp dir (e.g. `merged_dir` or a dedicated cache file) so Pass 2 can persist synonym/canonical caches for this run. Optional: pass `canonical_id_cache_path` if you have a shared cache path; otherwise a temp path is fine.
- **Pass 3:** Call `run_pass3(merged_dir, bundles_dir, output_dir)`. This reads `merged_dir` (id_map.json, entities.json, relationships.json, etc.) and `bundles_dir` (paper_*.json), and writes kgbundle files into `output_dir` (entities.jsonl, relationships.jsonl, manifest.json, etc.).
- **Load bundle:** Load `BundleManifestV1` from `output_dir/manifest.json`. Get a fresh storage instance (new Session from `get_engine()`), then call `storage.load_bundle_incremental(manifest, str(output_dir))`. Count entities/relationships from the manifest or from the files to set `entities_added` and `relationships_added` (e.g. line count of entities.jsonl minus any header, or from manifest if available).
- **Update job:** Set `status="complete"`, `completed_at=now`, `paper_title=...` (from Pass 1 or Pass 3 output if available), `pmcid=...` (if PMC), `entities_added`, `relationships_added`, `error=None`. Persist via `update_ingest_job`.
- **On exception:** Set `status="failed"`, `error=str(e)`, `completed_at=now`; log traceback; persist.

Use `asyncio.to_thread` or a thread pool for any synchronous Pass 2/Pass 3 calls if they are blocking (so the event loop is not blocked). If `run_pass1` is async and `run_pass2`/`run_pass3` are sync, call them in the right order: await pass1, then run pass2 and pass3 in a thread or run pass2/pass3 in a thread after pass1.

---

## Step 7. MCP tools

**Edit:** `kgserver/mcp_server/server.py`

- **ingest_paper** (async tool):
  - Signature: `async def ingest_paper(url: str) -> dict`.
  - Docstring: describe that it ingests a paper from a URL, kicks off a background job, returns immediately with job_id; suggest polling `check_ingest_status(job_id)`.
  - Implementation: create a job with `storage.create_ingest_job(url)` (use `_get_storage()` to get storage), then call `ingest_worker.enqueue_job(job.id)`, return `{"job_id": job.id, "status": "queued", "url": url, "message": "Job queued. Use check_ingest_status(job_id) to track progress."}`.
  - Ensure storage is used only for the create and that the job is enqueued after the DB commit so the worker can see it.

- **check_ingest_status** (sync tool):
  - Signature: `def check_ingest_status(job_id: str) -> dict`.
  - Docstring: returns job_id, status, url, paper_title, pmcid, entities_added, relationships_added, error, created_at, started_at, completed_at.
  - Implementation: with `_get_storage()` get storage, call `storage.get_ingest_job(job_id)`. If None, return e.g. `{"job_id": job_id, "status": "not_found", "error": "No such job"}`. Otherwise return a dict with all fields from the model (serialize datetimes to ISO strings).

---

## Step 8. Lifespan: create table and start worker

**Edit:** `kgserver/mcp_main.py`

- Add `from contextlib import asynccontextmanager` and the app’s lifespan.
- Import `IngestJob` from `storage.models` and `get_engine` from `query.storage_factory`, and `ingest_worker` from `mcp_server.ingest_worker`.
- Define:
  - `@asynccontextmanager`
  - `async def lifespan(app: FastAPI):`
    - `engine, _ = get_engine()`
    - `SQLModel.metadata.create_all(engine, tables=[IngestJob.__table__])`
    - `max_workers = int(os.environ.get("INGEST_MAX_WORKERS", "1"))`
    - `await ingest_worker.start_worker(max_workers=max_workers)`
    - `yield`
    - `await ingest_worker.stop_worker()`
- Pass `lifespan=lifespan` into `FastAPI(title="Knowledge Graph MCP", version="0.1.0", lifespan=lifespan)` so the table is created and the worker started on startup.

---

## Step 9. Dependencies and env

- **Edit:** `kgserver/pyproject.toml` — ensure `httpx` is a dependency (for fetching PMC and other URLs in the worker). Add if missing.
- **Env (document in README or in this plan):**
  - `PASS1_LLM_BACKEND` — default `anthropic` (anthropic | openai | ollama).
  - `PASS1_LLM_MODEL` — optional model override.
  - `INGEST_MAX_WORKERS` — default `1`.
  - `INGEST_TIMEOUT` — optional; seconds per job before considering it failed (e.g. 600). If implemented, the worker should enforce it (e.g. `asyncio.wait_for(_run_ingest_job(job_id), timeout=...)`).

---

## Step 10. Tests (optional but recommended)

- **IngestJob model:** Create a row, read it back, update it; verify table schema.
- **Storage job CRUD:** For both Postgres and SQLite (or the one you use), test create_ingest_job, get_ingest_job, update_ingest_job.
- **load_bundle_incremental:** Load one bundle, then load_bundle_incremental a second bundle; assert entity and relationship counts and that usage_count for a repeated entity is increased (or merged correctly).
- **MCP tools:** With a test client or in-process, call ingest_paper with a stub URL; then check_ingest_status with the returned job_id; either mock the worker or run a real worker and stub the URL fetch to return a minimal XML so the pipeline runs.

---

## File change summary

| File | Action |
|------|--------|
| `kgserver/storage/models/ingest_job.py` | **New** — IngestJob SQLModel |
| `kgserver/storage/models/__init__.py` | Export IngestJob |
| `kgserver/storage/models/relationship.py` | Add (or align) UniqueConstraint on (subject_id, predicate, object_id) for relationship upsert |
| `kgserver/storage/interfaces.py` | Add create_ingest_job, get_ingest_job, update_ingest_job; add load_bundle_incremental |
| `kgserver/storage/backends/postgres.py` | Implement job CRUD + load_bundle_incremental (upsert entities/relationships) |
| `kgserver/storage/backends/sqlite.py` | Implement job CRUD + load_bundle_incremental |
| `kgserver/mcp_server/ingest_worker.py` | **New** — queue, start_worker, stop_worker, enqueue_job, _worker_loop, _run_ingest_job |
| `kgserver/mcp_server/server.py` | Add ingest_paper (async) and check_ingest_status tools |
| `kgserver/mcp_main.py` | Add lifespan: create IngestJob table, start_worker, stop_worker |
| `kgserver/pyproject.toml` | Add httpx if missing |

---

## Reference locations

- **run_pass1:** `examples/medlit/scripts/pass1_extract.py` — `async def run_pass1(input_dir, output_dir, llm_backend, limit=None, papers=None, system_prompt=None)`.
- **get_pass1_llm:** `examples/medlit/pipeline/pass1_llm.py` — `get_pass1_llm(backend, *, model=None, base_url=None, timeout=300.0, ollama_client=None)`.
- **run_pass2:** `examples/medlit/pipeline/dedup.py` — `run_pass2(bundle_dir, output_dir, synonym_cache_path=None, canonical_id_cache_path=None)`.
- **run_pass3:** `examples/medlit/pipeline/bundle_builder.py` — `run_pass3(merged_dir, bundles_dir, output_dir)`.
- **Storage factory:** `kgserver/query/storage_factory.py` — `get_engine()` returns `(engine, db_url)`; use `Session(engine)` for a new session in the worker.
- **PMC efetch:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&rettype=xml` (pmcid without “PMC” prefix, e.g. `12345` for PMC12345).

---

## What was done

Implementation completed per the plan above.

- **Step 1:** Added `IngestJob` in `kgserver/storage/models/ingest_job.py` and exported it from `storage/models/__init__.py`.
- **Step 2:** Extended `StorageInterface` with `create_ingest_job`, `get_ingest_job`, `update_ingest_job`, and `load_bundle_incremental`.
- **Step 3:** Implemented job CRUD in both Postgres and SQLite backends (create with UUID, get by id, update by allowed kwargs).
- **Step 4:** Set `Relationship.__table_args__` to `UniqueConstraint("subject_id", "predicate", "object_id", name="uq_relationship_triple")`. Postgres: `load_bundle_incremental` uses `INSERT ... ON CONFLICT (entity_id) DO UPDATE` for entities (with `usage_count` accumulation) and get-then-merge for relationships; provenance appended as in `load_bundle`. SQLite: same semantics via get-or-merge with usage_count accumulation.
- **Step 5–6:** Added `kgserver/mcp_server/ingest_worker.py`: module-level queue and worker tasks, `start_worker`, `stop_worker`, `enqueue_job`, `_worker_loop`, `_run_ingest_job`. Worker fetches URL (PMC regex → efetch XML, else GET), runs Pass 1 (async), Pass 2/3 via `asyncio.to_thread`, loads manifest and calls `load_bundle_incremental` with a fresh storage session, extracts `paper_title` from first Pass 1 bundle when present, updates job complete/failed; timeout via `INGEST_TIMEOUT` (default 600s).
- **Step 7:** Registered MCP tools `ingest_paper(url)` (async) and `check_ingest_status(job_id)` in `kgserver/mcp_server/server.py`.
- **Step 8:** Added lifespan in `kgserver/mcp_main.py`: create `IngestJob` table, start worker (`INGEST_MAX_WORKERS`), stop on shutdown.
- **Step 9:** `httpx` was already in `kgserver/pyproject.toml`; no change. Env: `PASS1_LLM_BACKEND`, `INGEST_MAX_WORKERS`, `INGEST_TIMEOUT` as documented.
- **Step 10:** No new tests added; existing kgserver test suite (108 tests) passes.
