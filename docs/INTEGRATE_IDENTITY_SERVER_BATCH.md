# Integrating the Identity Server into Batch Ingestion (Option C)

This document summarizes the proposal and implementation guidance for integrating the Postgres-backed Identity Server into the existing batch ingestion (Pass 2 dedup) pipeline using Option C: keep batch file workflow but resolve entity IDs using the identity server during deduplication.

## Goals

- Use the Identity Server as the single source of truth for entity identity (canonical and provisional IDs).
- Preserve the existing batch workflow that produces inspectable merged JSON files.
- Avoid duplicating identity logic: replace ad-hoc name/type indexing and synonym lookups with calls to the Identity Server.
- Maintain debuggability, support offline runs, and minimize disruption to downstream steps.

## Summary of current state

- Pass 1 produces per-paper JSON bundles (paper_*.json) with extracted entities and relationships.
- Pass 2 (examples/medlit/pipeline/dedup.py) reads all bundles, builds an in-memory name/type index and a file-based synonym cache, performs authority lookups via CanonicalIdLookup (optional), assigns merge keys (prov- slugs or authoritative IDs), and writes merged entities.json and relationships.json.
- A Postgres-backed Identity Server implementation exists (kgserver/storage/backends/identity.py) and is used by incremental ingestion paths (MCP incremental ingestion). The batch pipeline does not currently call into the Identity Server.

## Option C (chosen approach)

- Keep Pass 2 as a bundle-driven deduplication script that reads all bundles and produces merged JSON files for inspection.
- Replace the ad-hoc assignment of merge keys and authority resolution with calls to the Identity Server's `resolve()` API for each unique entity (name + class + context).
- Keep other batch steps (SAME_AS high-confidence merges, embedding-based merges, relationship reconciliation) in Pass 2, but when a logical merge operation needs to change identity state across entities, call the Identity Server's `merge()` (or `promote()`) APIs as appropriate.

**Why Option C**

- Single authoritative source of entity IDs: the Identity Server centralizes identity state and idempotent behavior under Postgres, avoiding divergence between batch and incremental ingestion.
- Retains inspectable intermediate files (entities.json/relationships.json), which are helpful for debugging, QA, and auditing.
- Allows the batch pipeline to benefit from the Identity Server's authority caching (Redis) and locking semantics without fully streaming every entity into the DB at ingest time.
- Matches the natural evolution of the system where the Identity Server was designed as the canonical resolver.

## Integration plan — high level

1. **Dependency and bootstrap**
   - Ensure Pass 2 has access to the Identity Server client. This could be:
     - a local client library that constructs a sync/async client and a DB/connection/session, or
     - an HTTP/gRPC client if the identity server is run as a separate service.
   - The pipeline already optionally imports CanonicalIdLookup for authority lookups; re-use the same configuration approach for Identity Server access (env var toggles, connection URL, optional Redis URL).

2. **Replace get_or_assign_canonical with identity_server.resolve**
   - Where Pass 2 currently executes the sequence: (bundle authoritative ID check) → (external authority lookup) → (synonym cache lookups) → (prov slug), replace that path by calling `identity_server.resolve(mention, context)`.
   - Context should include at least: entity_class, paper_id, source (extracted), and any metadata used by authority lookups (e.g. UMLS hints).
   - Keep a local short-circuit: if the bundle row contains an authoritative ID (e.g. HGNC, MeSH, UniProt) prefer returning it directly to avoid an unnecessary round-trip, but still call `resolve()` when authoritative ID is missing or when you want the Identity Server to canonicalize it.

3. **SAME_AS merging and reconciliation**
   - High-confidence SAME_AS relationships that previously rewired local merge keys should result in calls to `identity_server.merge(...)` so the Postgres store reflects the merge decisions and redirects are available globally.
   - For reconciliations that only affect local merge keys and are not intended to be global merges, consider whether to call `merge` or to keep the local merge in the output files only. Prefer calling `merge` when the intent is to permanently collapse two identities.

4. **Writing merged files**
   - After resolve/merge calls, Pass 2 writes entities.json and relationships.json using the IDs returned by the Identity Server. These IDs will be authoritative Postgres-backed IDs (or preserved authoritative ontology IDs) rather than local prov- slugs.

5. **Transactional and performance considerations**
   - Choose between sync/async client depending on how Pass 2 is invoked (scripts are currently synchronous). If Identity Server client is async (its abstract API is async in the spec), a thin synchronous wrapper or an `asyncio.run` wrapper may be used for simplicity.
   - Batch size: do not flood the Identity Server / Postgres with one DB transaction per entity for very large runs. Use batching where appropriate (e.g. bulk-resolve groups of names, or run multiple resolves inside a larger transaction) while keeping the idempotency contract in mind.
   - Consider connection pool sizing and rate limits for external authority services (UMLS/ROR/ORCID). The existing CanonicalIdLookup cache logic is useful and can be subsumed by the Identity Server's Redis-backed cache.

6. **Offline / optional mode**
   - Preserve the existing canonical_id_cache.json and synonym_cache.json mechanisms for runs that should not contact external services or the Identity Server (e.g. offline development, CI tests). Make Identity Server usage optional via CLI flags (e.g. `--use-identity-server`) and fall back to the current in-process behavior when disabled.

7. **Logging and audit**
   - Ensure resolve/promote/merge calls are logged with structured metadata (mention, entity_class, chosen_id, reason) so batch runs have an audit trail if automatic merges or promotions are later questioned.
   - Leverage the Identity Server’s existing warnings for auto-merges to surface suspicious behavior.

8. **Tests and validation**
   - Add unit tests that mock the Identity Server client and verify Pass 2 behavior (IDs written match expected resolver responses).
   - End-to-end integration test: run a small bundle set against a test Postgres instance with Identity Server enabled and verify that entities.json contains Postgres-backed IDs and that merged relationships resolve correctly.
   - Add a regression test for the prior TOCTOU and advisory-lock issues the Identity Server fixes (e.g. resolve idempotency for two concurrent resolves of same mention).

## Detailed considerations and decisions

- **Where to short-circuit**
  - If the bundle row already contains a clearly authoritative external ID (CUI, MeSH, HGNC, UniProt, ROR, ORCID), Pass 2 may accept it directly and optionally call `identity_server.resolve()` with a context that includes the canonical_id to let the server record the external id mapping. This avoids unnecessary external lookups.

- **Same-as semantics vs local reconciliation**
  - Preserve the existing behavior of applying high-confidence SAME_AS merges during Pass 2, but make these persistent by calling `identity_server.merge` when the merge is a true entity collapse rather than a local-only consolidation.

- **Caching and authority lookups**
  - The Identity Server already supports a Redis-backed authority lookup cache (AuthorityCache). Configure Identity Server to reuse Redis used by other components to maximize cache hits.

- **Idempotency and retries**
  - The Identity Server APIs are designed to be idempotent. Pass 2 should treat calls as retriable and handle transient failures with retries/backoff. Keep local file outputs write-then-rename atomic to avoid leaving partial artifacts.

- **Performance tradeoffs**
  - Expect a performance hit compared to the pure file-based pass. Measure on representative data. If unacceptable, consider hybrid patterns like:
    - Bulk-resolve: group distinct (name, class) pairs and call a batch resolve endpoint (if implemented) or multiple concurrent resolve calls.
    - Two-phase approach: run the existing Pass 2 to produce provisional IDs, then run a short second pass that asks Identity Server to canonicalize only the prov- IDs created in the first pass (reduces DB churn).

## Suggested incremental implementation steps

1. Add Identity Server client wiring to the pipeline environment (config / CLI opt-in flag).
2. Implement a thin adapter in examples/medlit/pipeline that exposes a sync method `resolve_sync(name, context)` which calls the async `IdentityServer.resolve` under the hood (`asyncio.run` or equivalent).
3. Replace the authority lookup/lookup_entity flow in `get_or_assign_canonical` with a call to `resolve_sync`, keeping the existing local index short-circuits for authoritative IDs.
4. Update SAME_AS handling to call `identity_server.merge` when a durable merge is intended.
5. Add unit tests with a mocked Identity Server client and an integration test with a test Postgres instance.
6. Run perf tests on a representative batch, measure, and iterate (batching / concurrency / caching tuning).

## Open questions / tradeoffs to decide

- Do we want `merge()` calls for every automated reconciliation, or only for merges that pass a higher confidence threshold and are intended to be durable?
- Should we implement a batch-resolve endpoint on the Identity Server to reduce round-trips? (Nice optimization for large batches.)
- How to handle rollback semantics if a large batch partially fails after writing entities.json? Consider an approach where files are written with resolved IDs only after all DB resolves succeed for that run.

## References to existing code

- Current Pass 2 dedup implementation: `examples/medlit/pipeline/dedup.py`
- Pass 2 script entrypoint: `examples/medlit/scripts/pass2_dedup.py`
- Identity Server abstract interface: `kgschema/identity.py`
- Postgres-backed Identity Server: `kgserver/storage/backends/identity.py`
- Authority lookup helper used by Pass 2: `examples/medlit/pipeline/authority_lookup.py`
- Synonym cache utilities: `kgraph/pipeline/synonym_cache.py`
