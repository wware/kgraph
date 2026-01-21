# Obsolete Components from med-lit-schema

This document identifies components in med-lit-schema that are now obsolete due to kgraph and kgserver.

## ✅ Obsolete (Can be removed)

### 1. Storage Layer (`storage/` directory)
**Status**: Completely replaced by kgraph/kgserver

- `storage/interfaces.py` - Replaced by `kgraph/kgraph/storage/interfaces.py`
- `storage/backends/sqlite.py` - Replaced by `kgraph/kgraph/storage/memory.py` (for ingestion)
- `storage/backends/postgres.py` - Replaced by `kgserver/storage/backends/postgres.py`
- `storage/models/` - SQLModel persistence models - No longer needed (kgraph uses bundles, kgserver has its own models)
- `mapper.py` - Domain-to-persistence mapping - No longer needed (no persistence layer in kgraph)

**Why obsolete**: kgraph uses bundle export (JSONL files), and kgserver loads bundles into its own storage. The dual-domain/persistence model separation is no longer needed.

### 2. Query/API Layer (`query/` directory)
**Status**: Completely replaced by kgserver

- `query/server.py` - Replaced by `kgserver/main.py`
- `query/graphql_schema.py` - Replaced by `kgserver/query/graphql_schema.py`
- `query/routers/rest_api.py` - Replaced by `kgserver/query/routers/rest_api.py`
- `query/routers/graphiql_custom.py` - Replaced by kgserver's GraphiQL
- `query/client.py` - May be reusable, but kgserver has its own client patterns
- `main.py` - Replaced by `kgserver/main.py`

**Why obsolete**: kgserver provides a complete, domain-agnostic query API that works with any kgraph bundle.

### 3. Entity Collection Interface
**Status**: Replaced by kgraph's EntityStorageInterface

- `ENTITY_COLLECTION_INTERFACE.md` - Documentation for obsolete interface
- `entity.py` has `EntityCollectionInterface` and `InMemoryEntityCollection` - Replaced by kgraph's `EntityStorageInterface` and `InMemoryEntityStorage`

**Why obsolete**: kgraph's storage interfaces are more general and work across domains.

### 4. Docker/Infrastructure (Potentially)
**Status**: May be obsolete if using kgserver's docker setup

- `docker-compose.yml` - Check if kgserver's docker-compose covers this
- `Dockerfile` - Check if kgserver's Dockerfile covers this

**Action**: Review kgserver's docker setup to see if med-lit-schema's docker config is still needed.

## ⚠️ Keep (But may need updates)

### 1. Domain Models (`entity.py`, `relationship.py`, `base.py`)
**Status**: Ported to kgraph, but keep for reference

- The domain models (Disease, Gene, Drug, etc.) have been ported to `kgraph/examples/medlit/`
- Keep the originals for reference during migration
- Can be removed once migration is complete and tested

### 2. Ingestion Pipeline (`ingest/` directory)
**Status**: Partially reusable, but architecture changed

- `ingest/pmc_parser.py` - Logic can be reused for PMC XML parsing in kgraph
- `ingest/ner_pipeline.py` - NER extraction logic can be adapted for `MedLitEntityExtractor`
- `ingest/claims_pipeline.py` - Relationship extraction logic can be adapted for `MedLitRelationshipExtractor`
- `ingest/embeddings_pipeline.py` - Embedding generation logic can be adapted

**Action**: Port useful extraction logic to kgraph's pipeline components, then can remove.

### 3. Output Data (`output/` directory)
**Status**: Keep - this is generated data

- `output/json_papers/` - Paper JSON files used by kgraph ingestion
- `output/entities.jsonl` - May be useful for reference
- `output/extraction_provenance.json` - Useful for reference

### 4. Tests (`tests/` directory)
**Status**: Keep for reference, but tests should be rewritten for kgraph

- Tests are tied to the old architecture
- Useful as reference for what to test in the new architecture
- Can be removed once kgraph tests are comprehensive

### 5. Documentation (`docs/` directory)
**Status**: Keep for reference

- Architecture docs may still be useful
- User guides may need updates for kgraph/kgserver
- Keep until migration is complete

## 📋 Migration Checklist

- [ ] Remove `storage/` directory (completely obsolete)
- [ ] Remove `query/` directory (completely obsolete)
- [ ] Remove `mapper.py` (no longer needed)
- [ ] Remove `ENTITY_COLLECTION_INTERFACE.md` (obsolete)
- [ ] Review and potentially remove `docker-compose.yml` and `Dockerfile`
- [ ] Port useful extraction logic from `ingest/` to kgraph
- [ ] Update tests to use kgraph architecture
- [ ] Update documentation to reflect kgraph/kgserver architecture
- [ ] Remove domain models from med-lit-schema once migration verified

## 🎯 Summary

**Completely Obsolete (Safe to Remove)**:
- `storage/` - All storage interfaces, backends, and models
- `query/` - All API/server code
- `mapper.py` - Domain-to-persistence mapping
- `ENTITY_COLLECTION_INTERFACE.md` - Obsolete interface docs

**Keep for Reference (Remove After Migration)**:
- `entity.py`, `relationship.py`, `base.py` - Domain models (ported to kgraph)
- `ingest/` - Extraction logic (port useful parts, then remove)
- `tests/` - Test patterns (rewrite for kgraph, then remove)
- `docs/` - Documentation (update for kgraph, then remove)

**Keep**:
- `output/` - Generated data files
