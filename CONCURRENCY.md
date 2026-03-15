# Concurrency Issues in `ingest_paper` at Scale

This document identifies concurrency hazards in the `ingest_paper` MCP tool and
proposes mitigations for each. The context is a future production system with
thousands of simultaneous users, each potentially triggering paper ingestion at
the same time.

## Background

The `ingest_paper` MCP tool (implemented in `kgserver/mcp_server/ingest_worker.py`)
runs a multi-pass pipeline for each paper:

- **Pass 1a:** Fast vocabulary extraction — reads and writes `pass1_vocab/vocab.json`
  and `pass1_vocab/seeded_synonym_cache.json` in a shared workspace directory.
- **Pass 1b:** Full LLM extraction — writes one JSON bundle per paper to
  `pass1_bundles/paper_{pmcid}.json` in the shared workspace.
- **Pass 2:** Deduplication — reads *all* files in `pass1_bundles/` and writes
  merged output to `medlit_merged/`.
- **Pass 3:** Bundle build — reads `medlit_merged/` and writes the final bundle.
- **Load:** Calls `load_bundle_incremental` to upsert into Postgres.

Pass 2 → Pass 3 → Load are already serialized by a `fcntl.flock` file lock
(`_workspace_lock`). Pass 1a and Pass 1b are currently unprotected.

The default worker count is 1 (`start_worker(max_workers=1)`), which serializes
all jobs end-to-end and masks these issues in the current demo. They will surface
if worker count is increased or multiple server replicas are deployed.

---

## Issue 1: Pass 1a Vocab File Race

**Problem:** Pass 1a reads `vocab.json` and `seeded_synonym_cache.json`, merges
in vocabulary from the new paper, and writes them back. With concurrent workers,
two workers can read the same stale file, independently merge their updates, and
the last writer silently discards the other's changes.

**Proposed fix — Redis distributed lock:**

Add Redis to the stack (one line in `docker-compose.yml`, official image, zero
configuration needed for basic use). Before Pass 1a runs, each worker acquires a
Redis lock keyed on the workspace path, e.g. `lock:vocab:{workspace_root}`. The
lock should have a TTL (e.g. 120s) so a crashed worker cannot leave it
permanently held.

```python
async with redis_lock("vocab:ingest_workspace"):
    await run_pass1a(input_dir, vocab_dir, llm_backend, papers=None, limit=1)
```

Redis is preferred over extending the existing `fcntl.flock` for two reasons:

1. `fcntl.flock` only works within a single host. Redis works correctly across
   multiple replicas of `mcpserver`.
2. Redis locks automatically expire on worker crash; file locks do not.

**Alternative — Postgres table for vocab:**

Replace `vocab.json` and `seeded_synonym_cache.json` with Postgres tables (JSONB
columns work fine for the existing data structures). Reads and writes become
atomic transactions with no explicit locking needed. This is the right long-term
answer if vocab data needs to be queryable, auditable, or part of the same
transactional boundary as entities and relationships. It requires schema changes
and a migration step, making it higher-effort than the Redis approach.

Redis and Postgres are not mutually exclusive: Redis for coordination, Postgres
for persistent vocab state, is a valid combination.

---

## Issue 2: Pass 1b Output Collision (Same PMC ID, Two Workers)

**Problem:** If two jobs are simultaneously ingesting the same PMC ID, both pass
the "already exists" guard and both write to `pass1_bundles/paper_{pmcid}.json`.
This is a classic TOCTOU (time-of-check/time-of-use) race. The guard currently
reads:

```python
if pmcid is not None and (bundles_dir / f"paper_{pmcid}.json").exists():
    # skip
```

Two workers can both evaluate the condition as False before either writes the
file.

**Proposed fix — Redis per-pmcid lock:**

Use Redis `SET NX EX` (set-if-not-exists with expiry) keyed on the pmcid, e.g.
`lock:pmcid:PMC12345`. The first worker acquires it and proceeds; the second
either waits or fast-fails with "already in progress." This is more explicit and
reliable than the file-existence check.

```python
async with redis_lock(f"pmcid:{pmcid}"):
    if not (bundles_dir / f"paper_{pmcid}.json").exists():
        await run_pass1b(...)
```

**Alternative — Postgres unique constraint:**

Add a `pmcid` unique constraint to the `ingest_jobs` table (or a separate
`ingest_lock` table). Use `INSERT ... ON CONFLICT DO NOTHING` or `SELECT FOR
UPDATE` to serialize access. Works without Redis but is slightly more verbose.

---

## Issue 3: Pass 2 Reading a Partial Bundle File

**Problem:** Pass 2 scans `pass1_bundles/` and reads every `paper_*.json` file
it finds. If worker A is mid-write to its bundle file at the moment worker B
acquires the workspace lock and starts Pass 2, Pass 2 may read a truncated or
partially-written JSON file, causing a parse error or silently dropping data.

**Proposed fix — atomic write-then-rename in Pass 1b:**

This is a standard POSIX idiom and requires no new infrastructure. Pass 1b writes
to a temp file in the same directory (same filesystem, so the rename is atomic),
then renames it into the final path:

```python
tmp_path = bundles_dir / f".tmp_{pmcid}_{os.getpid()}.json"
tmp_path.write_text(json.dumps(bundle_data))
tmp_path.rename(bundles_dir / f"paper_{pmcid}.json")
```

`os.rename()` is atomic on Linux/POSIX — Pass 2 either sees the complete file or
nothing. This fix is orthogonal to the Redis vs. Postgres choice and should be
implemented regardless.

The relevant write is in `examples/medlit/scripts/pass1_extract.py` around the
line that writes `output_dir / f"paper_{paper_id}.json"`.

---

## Summary

| Issue | Root cause | Recommended fix | Effort |
|---|---|---|---|
| Pass 1a vocab race | Concurrent read-modify-write of JSON files | Redis distributed lock (cross-replica safe) | Low |
| Pass 1b same-pmcid collision | TOCTOU on file-existence check | Redis `SET NX EX` per pmcid | Low |
| Pass 2 reads partial bundle | File written concurrently with Pass 2 scan | Atomic write-then-rename in Pass 1b | Very low |

All three fixes are independent and can be implemented incrementally. The
write-then-rename fix (Issue 3) is the cheapest and most impactful and should be
done first. The Redis fixes (Issues 1 and 2) depend on adding Redis to the stack,
which is a small infrastructure change but unlocks correct behavior across
multiple replicas.

## Infrastructure Change: Adding Redis

Add to `docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  restart: unless-stopped
  networks:
    - kgserver-network
  profiles: ["api"]
```

Add `REDIS_URL: redis://redis:6379` to the `mcpserver` environment. Use
`redis.asyncio` (included in the `redis` Python package) for async lock
acquisition in the worker.
