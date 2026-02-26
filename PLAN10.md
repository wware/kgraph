# PLAN10: Persistent ingest workspace + remove legacy ingest.py

A detailed implementation plan for two related changes: (1) refactoring `kgserver/mcp_server/ingest_worker.py` so ingestion uses a persistent workspace instead of a temp directory, and (2) deleting the legacy ingest script and cleaning up references to it.

---

## Part 1: Persistent workspace refactor (`ingest_worker.py`)

### Problem

`_run_ingest_job_impl` (line ~138) wraps the entire pipeline in `with TemporaryDirectory(prefix="ingest_") as tmp:`. This means `pass1_bundles/`, `medlit_merged/` (including `synonym_cache.json` and `id_map.json`), and `output_dir` are all destroyed after each job. Incremental ingestion is therefore impossible — every job rebuilds from scratch and throws away the state that makes subsequent jobs coherent.

### What must persist between jobs

| Directory | Contents | Why it must persist |
|-----------|----------|----------------------|
| `pass1_bundles/` | Per-paper Pass 1 JSON bundles | Pass 2 and Pass 3 read all of these |
| `medlit_merged/` | `entities.json`, `relationships.json`, `id_map.json`, `synonym_cache.json` | Pass 2 writes and reads synonym cache; entity synonyms from paper N affect deduplication of paper N+1 |
| `medlit_bundle/` | Pass 3 output (kgbundle) | Always fully rebuilt by Pass 3, but should persist so it can be reloaded or inspected |

`input_dir` (downloaded XML) is the only thing that should remain per-job temp.

### New environment variable

| Variable | Default | Description |
|----------|---------|-------------|
| `INGEST_WORKSPACE_ROOT` | `./ingest_workspace` | Persistent workspace root. Resolved with `Path(...).resolve()` so it is absolute when creating dirs and lock file. |

---

### 1. Workspace helpers (add at module level, after imports)

Add these functions and imports exactly as below. Use `Path` and `os` already in the file; add `fcntl` and `contextmanager` from `contextlib`.

**Imports to add:** `fcntl` (standard library), and `contextmanager` from `contextlib`.

```python
def _workspace_root() -> Path:
    """Persistent workspace root from env, defaulting to ./ingest_workspace."""
    root = Path(os.environ.get("INGEST_WORKSPACE_ROOT", "./ingest_workspace")).resolve()
    return root


def _ensure_workspace_dirs(root: Path) -> tuple[Path, Path, Path]:
    """Create and return (bundles_dir, merged_dir, output_dir). These persist across jobs.

    Note: output_dir (medlit_bundle/) is always fully rebuilt by Pass 3.
    The true incremental state is bundles_dir + merged_dir (especially synonym_cache.json).
    """
    bundles_dir = root / "pass1_bundles"
    merged_dir = root / "medlit_merged"
    output_dir = root / "medlit_bundle"
    for d in (bundles_dir, merged_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)
    return bundles_dir, merged_dir, output_dir


@contextmanager
def _workspace_lock(workspace_root: Path):
    """File lock serializing Pass 2 → Pass 3 → load_bundle_incremental across workers."""
    lock_path = workspace_root / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
```

**Platform note:** `fcntl` is Unix-only. On Windows this will fail at import or at first use. If Windows support is required, either (a) use the `filelock` package and wrap `FileLock(lock_path)` in the same way, or (b) document that the ingest worker is supported on Linux/macOS only. The plan assumes Unix for the initial implementation.

---

### 2. Pass 1 output filename convention (no-op guard)

**Verified in code:** `examples/medlit/scripts/pass1_extract.py` line 240:

```python
paper_id = paper_info.pmcid or path.stem
out_path = output_dir / f"paper_{paper_id}.json"
```

For PMC URLs the worker sets `pmcid = f"PMC{pmcid_num}"` (e.g. `PMC12345`). So the bundle file is **`paper_PMC12345.json`**. The no-op guard must check:

- **When** `pmcid` is not None (i.e. we parsed a PMC URL).
- **Check:** `(bundles_dir / f"paper_{pmcid}.json").exists()`.
- **If true:** Treat job as already complete: update job with `status="complete"`, `pmcid=pmcid`, `paper_title=` (optional: read from existing bundle or leave None), `entities_added=0`, `relationships_added=0`, `completed_at=now`, `error=None`. Then `return` without running Pass 1, 2, or 3.

**Placement:** Immediately after the fetch (so we have `pmcid`; the temp input dir already exists at that point). Run the guard before Pass 1. See §4 step 4 — the numbered flow is authoritative.

---

### 3. Lock scope and sync helper

Pass 1 writes one file per paper and does not touch `medlit_merged/`, so **Pass 1 runs outside the lock**. Pass 2, Pass 3, and `load_bundle_incremental` must run as a single critical section so that no other worker interleaves; the lock must be held **inside** the thread that runs these (since they are invoked via `asyncio.to_thread`).

**Approach:** Define a **sync** function that (1) acquires `_workspace_lock(workspace_root)`, (2) runs `run_pass2`, (3) runs `run_pass3`, (4) reads `manifest.json`, (5) calls `_storage_for_worker()`, then `load_bundle_incremental(manifest, str(output_dir))`, then `load_close()`. Any exception propagates out. The async code then calls `await asyncio.to_thread(this_sync_function, ...)` and, on success, computes `entities_added` / `relationships_added` and updates the job as today.

**Signature of the sync helper:**

```python
def _run_pass2_pass3_load(
    workspace_root: Path,
    bundles_dir: Path,
    merged_dir: Path,
    output_dir: Path,
) -> None:
    """Run Pass 2, Pass 3, and load_bundle_incremental under workspace lock. Raises on failure."""
```

Inside it: `with _workspace_lock(workspace_root):` then run Pass 2, Pass 3, then open manifest, get storage, `load_bundle_incremental(manifest, str(output_dir))`, close storage. Do not return the manifest; the async caller will re-read `output_dir / "manifest.json"` and the entity/relationship files after the thread returns (same as current logic).

**SQLite and dual connections:** The outer `_run_ingest_job_impl` holds a storage connection (for `get_ingest_job` / `update_ingest_job`) while `_run_pass2_pass3_load` opens a second connection via `_storage_for_worker()` for `load_bundle_incremental`. For Postgres this is fine. For SQLite, concurrent writers can cause "database is locked". Ensure SQLite is configured with WAL mode (`PRAGMA journal_mode=WAL`) or serialize the load (e.g. single worker or load outside the thread with one connection) to avoid conflicts.

---

### 4. Restructured `_run_ingest_job_impl` (drop-in replacement)

Replace the body of `_run_ingest_job_impl` from the line that sets `pmcid = None` through the end of the function with the following flow. Keep the same imports inside the function for `run_pass1`, `run_pass2`, `run_pass3`, `BundleManifestV1`, and `json` where used.

1. **Workspace setup (once per job):**
   - `workspace = _workspace_root()`
   - `bundles_dir, merged_dir, output_dir = _ensure_workspace_dirs(workspace)`

2. **Temp dir only for input:**
   - `with TemporaryDirectory(prefix="ingest_input_") as tmp_input:`
   - `input_dir = Path(tmp_input)` (no `input_dir / "input"` subdir unless the fetch code expects it; current code uses `root / "input"` and writes into it — so use `input_dir = Path(tmp_input) / "input"`, `input_dir.mkdir()` to keep the same structure, or keep a single flat dir and write `paper.xml` / `paper{suffix}` into `input_dir` directly). **Check current fetch:** it writes to `input_dir / "paper.xml"` or `input_dir / f"paper{suffix}"`. So `input_dir` is the dir that contains the single paper file. Use `input_dir = Path(tmp_input)` and write the fetched file there (e.g. `input_dir / "paper.xml"`). Pass 1 discovers files by glob; it will look for `*.xml` / `*.json` under `input_dir`. So ensure the fetched file is named so the parser finds it (e.g. `paper.xml`). Current code already uses `input_dir / "paper.xml"` for PMC and `input_dir / f"paper{suffix}"` for generic URL. So `input_dir = Path(tmp_input)` and create no subdir.

3. **Fetch URL** (unchanged): same logic as today to set `pmcid`, `paper_title` (left None for now), and write file into `input_dir`.

4. **No-op guard (immediately after fetch, before Pass 1):**
   - If `pmcid` is not None and `(bundles_dir / f"paper_{pmcid}.json").exists()`:
     - Update job: `status="complete"`, `completed_at=datetime.now(timezone.utc)`, `paper_title=paper_title or None`, `pmcid=pmcid`, `entities_added=0`, `relationships_added=0`, `error=None`
     - `return`

5. **Pass 1 (outside lock):**
   - Same as today: `await run_pass1(input_dir, bundles_dir, llm_backend, limit=1)`.
   - Same validation: if no `paper_*.json` in `bundles_dir`, update job failed and return.
   - Optional: set `paper_title` from first bundle as today.

6. **Pass 2 + Pass 3 + load (under lock, in thread):**
   - Call `await asyncio.to_thread(_run_pass2_pass3_load, workspace, bundles_dir, merged_dir, output_dir)`.
   - Catch exceptions: on failure, update job failed and return (same as current Pass 2/Pass 3/load error handling).
   - After success, read `manifest_path = output_dir / "manifest.json"`, parse manifest, compute `entities_added` and `relationships_added` from the entity/relationship file line counts (same as current code).
   - Update job: `status="complete"`, `paper_title`, `pmcid`, `entities_added`, `relationships_added`, `completed_at`, `error=None`.
   - Log completion with pass timings (pass1_sec, pass2_sec, pass3_sec — pass2_sec and pass3_sec can be summed from a single timer around the thread call if you do not need them separate).

**Exact directory names:** Use `bundles_dir` = `workspace / "pass1_bundles"`, `merged_dir` = `workspace / "medlit_merged"`, `output_dir` = `workspace / "medlit_bundle"` (already in `_ensure_workspace_dirs`). Pass 2 and Pass 3 are called with the same argument names as today: `run_pass2(bundle_dir=bundles_dir, output_dir=merged_dir, synonym_cache_path=merged_dir / "synonym_cache.json", canonical_id_cache_path=None)`, `run_pass3(merged_dir, bundles_dir, output_dir)`.

---

### 5. Test strategy

**Integration test (one test function or script):**

1. Set `INGEST_WORKSPACE_ROOT` to a temp directory (e.g. `tmp_path / "ingest_workspace"`).
2. Ingest **two** papers (e.g. two PMC IDs) via the worker (enqueue two jobs, wait for completion). Record entity count (or relationship count) after the second job.
3. Ingest a **third** paper via the worker. Assert:
   - Entity (or relationship) count **increases** from step 2 (monotonic).
   - The first two papers’ identifiers (e.g. PMC IDs or paper IDs in the bundle) are still present in the storage or in the workspace (e.g. `pass1_bundles/paper_PMC*.json` and merged output still reference them).
4. Optional: Re-enqueue the first paper (same PMC ID). Assert the job completes without re-running Pass 1 (no-op guard) and that entity counts do not double (idempotent).

Use the existing worker API: `enqueue_job(job_id)`, job with `url` pointing to PMC, and a way to read entity/relationship counts from storage or from the workspace manifest after each run. Prefer the existing test DB or a test storage so the test is deterministic.

---

## Part 2: Delete the legacy ingest script

### Changes

1. **Delete** the legacy ingest script file.

2. **Update Markdown docs** that reference the legacy ingest script or its CLI so they point to the three-pass pipeline and `run-ingest.sh` where appropriate.

### List of `.md` files to update (and what to do)

| File | Action |
|------|--------|
| `examples/medlit/README.md` | Replace all legacy ingest CLI examples with the equivalent three commands from `run-ingest.sh`: `pass1_extract`, `pass2_dedup`, `pass3_build_bundle` with the same dirs and options. Point to `INGESTION.md` and `run-ingest.sh` for the canonical flow. |
| `examples/medlit/CANONICAL_IDS.md` | Replace the sentence that names the legacy pipeline script with a sentence that the **two-pass pipeline** (Pass 1 → Pass 2) is canonical; the legacy promotion path is no longer available via a supported CLI. Remove or rephrase the line that says `scripts/ingest.py` is the CLI wiring. Update any legacy ingest command blocks to use `pass1_extract` / `pass2_dedup` / `pass3_build_bundle` as in `run-ingest.sh`. |
| `examples/medlit/INGESTION.md` | No change required for ingest.py (it already documents the three-pass scripts). If it mentions ingest.py anywhere, replace with “legacy script removed; use pass1_extract, pass2_dedup, pass3_build_bundle.” |
| `summary.md` | Generated manually using `uv run python summarize_codebase.py`, please ignore this file. |
| `SIMPLIFY_PROMOTION.md` | Update references to “scripts/ingest.py” to state that the script has been removed and the two-pass pipeline (pass1, pass2) is used; remove or stub promotion there, not in ingest.py. |
| `PLAN8a.md` | Update the comment about the legacy pipeline to “Legacy ingest.py removed; use pass1_extract, pass2_dedup, pass3_build_bundle (see run-ingest.sh).” |
| `PLAN8.md` | Replace the option “refactor scripts/ingest.py to have a --pass1-only mode” with “use pass1_extract (canonical).” |
| `PLAN6.md` | Replace the legacy ingest command with the equivalent pass1/pass2/pass3 commands. |
| `PLAN4.md` | Add at the top: “Superseded: ingest.py has been removed. The pipeline is now split into pass1_extract, pass2_dedup, pass3_build_bundle. See run-ingest.sh and examples/medlit/INGESTION.md.” Do not delete the rest of the plan (historical). |
| `PLAN3.md` | Replace “the legacy ingest script” with “examples/medlit/scripts/pass1_extract.py” (or the appropriate pass) for the entity extractor / config owner. |
| `PLAN2.md` | Replace references to “the legacy ingest script” with the appropriate pass script or “medlit pipeline scripts” and update any legacy ingest CLI examples to the three-pass invocations. |
| `PLAN10.md` | (This file.) No change. |

Do **not** delete PLAN4, PLAN6, PLAN8, PLAN8a, PLAN2, PLAN3, or SIMPLIFY_PROMOTION; only update text so that no doc implies ingest.py is a supported entrypoint.

### Narrative

- Any plan or doc that described “splitting ingest.py by stage” or “ingest.py as entrypoint” should state that the **three-pass scripts are now canonical** and ingest.py has been removed.
- The narrative about legacy promotion machinery can remain (e.g. in CANONICAL_IDS.md or SIMPLIFY_PROMOTION.md) but must not imply there is a supported CLI path through ingest.py.

---

## Execution order (for implementer)

1. **Part 1 in `ingest_worker.py`:**
   - Add `fcntl` and `contextmanager` imports; add `_workspace_root`, `_ensure_workspace_dirs`, `_workspace_lock`.
   - Add `_run_pass2_pass3_load` (sync) that holds the lock and runs Pass 2, Pass 3, load.
   - Rewrite `_run_ingest_job_impl`: workspace dirs, temp dir only for input, fetch, no-op guard for `paper_{pmcid}.json`, Pass 1 outside lock, then `asyncio.to_thread(_run_pass2_pass3_load, ...)`, then read manifest and update job.
   - Run existing tests; add the integration test described in §5.
2. **Part 2:**
   - Delete `the legacy ingest script`.
   - Edit each `.md` file in the table above per the “Action” column.
3. **Lint:** Run `./lint.sh` and fix any issues.

---

## Work done (implementation summary)

**Part 1 — Persistent workspace (`kgserver/mcp_server/ingest_worker.py`):**

- Added `_workspace_root()`, `_ensure_workspace_dirs()`, and `_workspace_lock()` (fcntl-based). New env var `INGEST_WORKSPACE_ROOT` (default `./ingest_workspace`).
- Added sync helper `_run_pass2_pass3_load(workspace_root, bundles_dir, merged_dir, output_dir)` that holds the lock and runs Pass 2, Pass 3, and `load_bundle_incremental`.
- Rewrote `_run_ingest_job_impl`: persistent workspace dirs; temp dir only for fetched input; no-op guard when `paper_{pmcid}.json` already exists in `bundles_dir`; Pass 1 outside lock; Pass 2+3+load via `asyncio.to_thread(_run_pass2_pass3_load, ...)`.
- Added `kgserver/tests/test_ingest_worker.py`: unit tests for workspace helpers and lock; integration test (two papers then three, entity count increases), skipped when `examples.medlit` is not importable.

**Part 2 — Remove legacy ingest and update docs:**

- Deleted the legacy ingest script.
- Updated `examples/medlit/README.md`, `CANONICAL_IDS.md`; `SIMPLIFY_PROMOTION.md`, `PLAN8a.md`, `PLAN8.md`, `PLAN6.md`, `PLAN4.md`, `PLAN3.md`, `PLAN2.md` per the plan table (three-pass pipeline canonical; ingest.py removed). `summary.md` is auto-generated; regenerate with `uv run python summarize_codebase.py` when desired.

**Follow-up:**

- **ProgressTracker:** Had been in the deleted ingest script. Added `examples/medlit/progress.py` with a Pydantic `BaseModel` `ProgressTracker` (same fields and behavior). Updated `examples/medlit/tests/test_progress_tracker.py` to import from `examples.medlit.progress`; all 13 tests pass.
- **Lint:** `lint.sh` now filters the Python file list to existing files only (`[ -f "$f" ]`) so deleted `ingest.py` is not passed to ruff.
