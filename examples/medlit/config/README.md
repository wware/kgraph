# Medlit Domain Configuration

Domain configuration for the medlit ingestion pipeline lives in **`domain_spec.py`** (one level up, at `examples/medlit/domain_spec.py`), not in this directory.

That module is the single source of truth for entity types, predicates, prompt instructions, and evidence/mentions specs. The extraction prompt, validation logic, and dedup rules all consume it. Edit `domain_spec.py` to change the schema; no separate YAML or markdown files.

This `config/` directory is retained for any future config that does not belong in the domain spec (e.g. paths, feature flags). For schema changes, edit `domain_spec.py`.
