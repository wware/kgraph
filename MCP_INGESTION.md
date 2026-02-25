# MCP tools for paper ingestion

There could be a MCP tool that ingests a paper given a URL. It would allow the graph to grow in new directions and pivot to meet changing needs. It is an async tool that kicks off a background job and returns a job ID, with a separate check_ingest_status(job_id) tool.


**The two tools:**
```python
ingest_paper(url: str) -> {"job_id": str, "status": "queued"}
check_ingest_status(job_id: str) -> {"job_id": str, "status": "queued|running|complete|failed", "paper_title": str, "entities_added": int, "relationships_added": int, "error": str | None}
```

**Background job implementation options** — since you're already in FastAPI/asyncio:
- Simplest: `asyncio.create_task()` + an in-memory dict of job states. Fine for a single-process server, lost on restart.
- Better: a lightweight job table in your existing database (Postgres/SQLite) so status survives restarts and you get a history of what's been ingested
- Overkill for now: Celery, ARQ, etc.

We will proceed with the "Better" approach. The storage implementation (by default) will be Postgres+pgvector, but it would be nice if the machinery remained DB-agnostic.

# Implementation thoughts

## Plan: `ingest_paper` + `check_ingest_status` MCP Tools

### 1. New SQLModel: `IngestJob`

**New file:** `kgserver/storage/models/ingest_job.py`

```python
class IngestJob(SQLModel, table=True):
    __tablename__ = "ingest_jobs"
    id: str = Field(primary_key=True)           # UUID
    url: str
    status: str = Field(default="queued")        # queued|running|complete|failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paper_title: Optional[str] = None
    pmcid: Optional[str] = None
    entities_added: int = Field(default=0)
    relationships_added: int = Field(default=0)
    error: Optional[str] = None
```

Register in `kgserver/storage/models/__init__.py` alongside the existing models.

---

### 2. StorageInterface Extensions

**Modified:** `kgserver/storage/interfaces.py`

Add three abstract methods:

```python
@abstractmethod
def create_ingest_job(self, url: str) -> IngestJob: ...

@abstractmethod
def get_ingest_job(self, job_id: str) -> Optional[IngestJob]: ...

@abstractmethod
def update_ingest_job(self, job_id: str, **fields) -> Optional[IngestJob]: ...
```

**Modified:** `kgserver/storage/backends/postgres.py` and `sqlite.py` — implement all three. `update_ingest_job` should accept arbitrary kwargs matching column names, fetch the row, apply updates, flush, and return the updated model.

---

### 3. Background Worker

**New file:** `kgserver/mcp_server/ingest_worker.py`

```python
# Module-level singletons — initialized in lifespan
_job_queue: asyncio.Queue[str] = asyncio.Queue()
_worker_task: Optional[asyncio.Task] = None

async def start_worker(max_workers: int = 1) -> None: ...
async def stop_worker() -> None: ...
async def enqueue_job(job_id: str) -> None: ...

async def _worker_loop() -> None:
    """Pull job IDs from queue, run ingestion, update DB."""
    while True:
        job_id = await _job_queue.get()
        await _run_ingest_job(job_id)
        _job_queue.task_done()

async def _run_ingest_job(job_id: str) -> None:
    # 1. Fetch job from DB, mark running
    # 2. Fetch URL content via httpx (handle PMC redirects/XML API)
    # 3. In a TemporaryDirectory:
    #    a. Write raw content to input_dir/paper.xml (or .json)
    #    b. run_pass1() → bundles_dir/paper_*.json
    #    c. run_pass2(bundles_dir, merged_dir)
    #    d. run_pass3(merged_dir, bundles_dir, output_dir) → bundle with manifest.json
    # 4. storage.load_bundle(manifest, output_dir)
    # 5. Update job: status=complete, entities_added, relationships_added, paper_title
    # 6. On any exception: status=failed, error=str(e), log traceback
```

**Key decisions for Cursor:**

- Use `get_pass1_llm(backend=os.environ.get("PASS1_LLM_BACKEND", "anthropic"))` from `examples/medlit/pipeline/pass1_llm.py` — the API key env vars are already in scope.
- For PMC URLs (`/articles/PMC\d+/`), fetch the XML via the E-utilities API: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&rettype=xml`. Extract the PMCID with a regex.
- `run_pass2` and `run_pass3` are already importable from `examples.medlit.pipeline.dedup` and `examples.medlit.pipeline.bundle_builder`.
- `storage.load_bundle()` needs to be called with a fresh session — use `get_engine()` from `query.storage_factory` and open a new `Session` directly (don't use the FastAPI dependency, since this runs in a background task outside a request).

---

### 4. MCP Tools

**Modified:** `kgserver/mcp_server/server.py` — add alongside existing tools:

```python
@mcp_server.tool()
async def ingest_paper(url: str) -> dict:
    """Ingest a medical paper from a URL into the knowledge graph.
    
    Kicks off a background job. Returns immediately with a job_id.
    Poll check_ingest_status(job_id) to track progress.
    Supports PMC full-text URLs and direct XML/JSON URLs.
    Returns: {job_id, status, url, message}
    """
    # 1. Create IngestJob in DB via _get_storage()
    # 2. Enqueue job_id via ingest_worker.enqueue_job(job.id)
    # 3. Return {job_id, status: "queued", url, message: "Job queued..."}

@mcp_server.tool()
def check_ingest_status(job_id: str) -> dict:
    """Check the status of a paper ingestion job.
    
    Returns: {job_id, status, url, paper_title, pmcid,
              entities_added, relationships_added, error,
              created_at, started_at, completed_at}
    Status values: queued | running | complete | failed
    """
    # Fetch from DB, return as dict. 404-style dict if not found.
```

Note: FastMCP supports `async def` tools — `ingest_paper` should be async so the queue `put` doesn't block. `check_ingest_status` can be sync.

---

### 5. Lifespan Integration

**Modified:** `kgserver/mcp_main.py`

Add a lifespan context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure IngestJob table exists
    engine, _ = get_engine()
    SQLModel.metadata.create_all(engine, tables=[IngestJob.__table__])
    
    # Start background worker(s)
    max_workers = int(os.environ.get("INGEST_MAX_WORKERS", "1"))
    await ingest_worker.start_worker(max_workers)
    
    yield
    
    await ingest_worker.stop_worker()

app = FastAPI(title="Knowledge Graph MCP", lifespan=lifespan, ...)
```

---

### 6. `load_bundle` Upsert Behavior — Critical

Check `PostgresStorage._load_entities()` to confirm whether it does `INSERT` or `INSERT ... ON CONFLICT DO UPDATE`. If it's plain INSERT, duplicate entity_ids from the new paper will fail when they collide with existing entities. Cursor should change the entity/relationship loading to use `ON CONFLICT (entity_id) DO UPDATE SET usage_count = usage_count + excluded.usage_count, ...` so incremental loads are truly additive. Same for relationships — upsert on `(subject_id, predicate, object_id)`.

---

### 7. New Environment Variables

```bash
PASS1_LLM_BACKEND=anthropic    # anthropic | openai | ollama
PASS1_LLM_MODEL=               # optional override
INGEST_MAX_WORKERS=1           # concurrent ingestion jobs
INGEST_TIMEOUT=600             # seconds per job before killing
```

---

### 8. File Change Summary

| File | Change |
|------|--------|
| `kgserver/storage/models/ingest_job.py` | **New** — `IngestJob` SQLModel |
| `kgserver/storage/models/__init__.py` | Add `IngestJob` to exports |
| `kgserver/storage/interfaces.py` | Add 3 abstract job CRUD methods |
| `kgserver/storage/backends/postgres.py` | Implement job CRUD + upsert fix for `_load_entities`/`_load_relationships` |
| `kgserver/storage/backends/sqlite.py` | Same |
| `kgserver/mcp_server/ingest_worker.py` | **New** — queue, worker loop, `_run_ingest_job` |
| `kgserver/mcp_server/server.py` | Add `ingest_paper` and `check_ingest_status` tools |
| `kgserver/mcp_main.py` | Add lifespan for table creation + worker start/stop |
| `kgserver/pyproject.toml` | Ensure `httpx` is listed (probably already there) |

The one genuinely tricky part is the `_run_ingest_job` function — it orchestrates the pass1/2/3 pipeline in a temp directory and then calls `load_bundle`. Cursor will need to look carefully at how `examples/medlit/scripts/pass1_extract.py` calls `run_pass1()` and replicate that call path, and similarly for pass2 and pass3. The existing scripts are the reference implementation — `_run_ingest_job` is essentially those three scripts composed in memory with a temp dir.