# Feature request: Simplify / remove legacy promotion machinery

**Status:** For later consideration. Not part of PLAN8a.

**Context:** The new two-pass medlit pipeline (Pass 1 → Pass 2) does not use promotion. It does not call `run_promotion`, `PromotionPolicy`, or storage `promote()` / `find_provisional_for_promotion`. Canonical vs provisional is expressed only via Pass 2 output: `canonical_id` set when we have an authoritative ID, null otherwise. The legacy script `examples/medlit/scripts/ingest` has been removed; the canonical flow is the three-pass pipeline (pass1_extract, pass2_dedup, pass3_build_bundle). `run-ingest.sh` now uses that pipeline (usage/confidence thresholds, `MedLitPromotionPolicy`, `run_promotion`). This document describes a possible future change to remove or simplify that machinery. It is a larger refactor that touches kgschema and kgraph and should be decided with care.

---

## Goal

Remove or reduce code that exists only to support the old promotion workflow, once the team is comfortable retiring or not relying on the old ingest path.

---

## What would be removed or changed

### kgschema

- **PromotionConfig** (entity.py) — remove if no other use remains, or keep as an optional/deprecated type.
- **PromotionPolicy** (promotion.py) — remove the ABC and the module, or move to a deprecated/legacy package.
- **DomainSchema** — remove or make optional: `promotion_config`, `get_promotion_policy`. If kept for backward compatibility, document as deprecated and have default implementations return a no-op or raise "not supported."
- **EntityStorageInterface** (storage.py) — remove `find_provisional_for_promotion`. Remove or deprecate `promote()` if nothing else uses it.

### kgraph

- **ingest.py** — remove `run_promotion`, `_promote_single_entity`, `_lookup_canonical_ids_batch`, and any logic that calls `get_promotion_policy` / `find_provisional_for_promotion` / `promote()`. Alternatively, replace with a no-op (e.g. `run_promotion` returns `[]`) so callers do not break.
- **storage/memory.py** — remove `find_provisional_for_promotion` and `promote()` from `InMemoryEntityStorage` (and any other implementations of the storage interface).
- **promotion.py** — remove re-export of `PromotionPolicy` or keep only for a deprecation period; remove `TodoPromotionPolicy` if unused.

### examples/medlit

- **promotion.py** — remove `MedLitPromotionPolicy` (or move to a legacy module).
- **domain.py** — remove `promotion_config` and `get_promotion_policy`, or implement them as deprecated/no-op.
- **scripts/ingest.py** — removed (PLAN10). The two-pass pipeline (pass1_extract, pass2_dedup, pass3_build_bundle) is canonical; no promotion phase.

### examples/sherlock

- Same pattern as medlit: **promotion.py**, **domain** promotion wiring, **scripts/ingest** use of `run_promotion`.

### examples/medlit_schema

- **domain.py** — remove or stub `MedlitPromotionPolicy` and promotion-related methods if that schema is only used by the legacy path.

### tests

- **test_promotion.py**, **test_promotion_merge.py** — remove or rewrite to test only behavior that remains (e.g. entity status, storage listing).
- **test_export.py**, **test_pipeline_integration.py** — remove or adjust tests that call `run_promotion` or `storage.promote()`.
- **conftest.py** — remove `SimplePromotionPolicy` and promotion-related parts of `SimpleDomainSchema` (or keep minimal stubs if other tests still need a domain that implements the interface).

### docs

- **architecture.md**, **api.md**, **domains.md**, **pipeline.md**, **canonical_ids.md**, etc. — update or remove sections that describe promotion (PromotionConfig, PromotionPolicy, run_promotion, promote, find_provisional_for_promotion). Document the two-pass pipeline as the supported approach.

---

## What to keep

- **EntityStatus** (PROVISIONAL, CANONICAL) — still used by `BaseEntity`, storage, kgserver (filter/display), and some relationship logic. Do not remove.
- **BaseEntity** fields such as `status`, `usage_count`, `confidence` — part of the entity model and server data; keep unless a separate decision is made to change the schema.
- **CanonicalIdLookup** and authority lookup — used by Pass 2 (PLAN8a) and optionally by the old ingest; keep.

---

## Considerations before proceeding

1. **Old ingest and run-ingest.sh** — Confirm whether anything still depends on the old pipeline (ingest.py). If yes, decide: retire it (and then remove promotion) or keep it (and keep promotion, or stub run_promotion to no-op).
2. **Sherlock and other domains** — Same decision for any other domain that implements promotion; either remove promotion from those domains or leave them as legacy.
3. **Backward compatibility** — If any bundles or exports assume promotion has run, document the migration path (e.g. "use Pass 2 output; entity_id + canonical_id replace status for canonical/provisional").
4. **Tests and CI** — Ensure remaining tests and lint pass after removal; add or keep tests for the two-pass pipeline and Pass 2 output shape.

---

## Relationship to PLAN8a

PLAN8a includes **Phase 1**: document that the new two-pass pipeline does not use promotion and label the old ingest as legacy. No code is removed. This file (SIMPLIFY_PROMOTION.md) describes the optional **Phase 2** refactor for a later time.
