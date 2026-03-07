# Medlit Schema Config

Schema configuration for the medlit ingestion pipeline. Domain experts edit these files; the pipeline loads them at runtime and injects content into extraction prompts.

- **entity_types.yaml** — Entity type taxonomy with descriptions
- **predicates.yaml** — Predicate definitions with type signatures
- **domain_instructions.md** — Domain-specific guidance (classification rules, counterexamples)

Do not edit Python or Jinja2 templates. Changes here enable collaborative iteration: edit config → re-run extraction → compare output.
