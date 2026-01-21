# GitHub Copilot Instructions for kgraph

This file provides guidance to GitHub Copilot when working with code in this repository.

## Project Overview

**kgraph** is a domain-agnostic framework for building knowledge graphs from documents. The system extracts entities and relationships across multiple knowledge domains (medical literature, legal documents, academic CS papers, etc.).

### Architecture

The system uses a two-pass ingestion pipeline:

1. **Pass 1 (Entity Extraction)**: Extract entities from documents, assign canonical IDs where appropriate (UMLS for medical, DBPedia URIs cross-domain, etc.)
2. **Pass 2 (Relationship Extraction)**: Identify edges/relationships between entities, produce per-document JSON with edges and provisional entities

### Key Concepts

- **Canonical entities**: Assigned stable IDs from authoritative sources
- **Provisional entities**: Mentions awaiting promotion based on usage count and confidence scores
- **Entity promotion**: Provisional → canonical when usage thresholds are met
- **Entity merging**: Combining canonical entities detected as duplicates via semantic vector similarity
- **Domain-agnostic design**: Each knowledge domain defines its own canonical ID source, schema, and edge types

## Technology Stack

- **Language**: Python 3.12+
- **Package Manager**: `uv` (modern Python package installer and resolver)
- **Testing**: pytest with pytest-asyncio for async support
- **Linting**: ruff, black, flake8, pylint
- **Data Models**: Pydantic v2 with `frozen=True` (immutable, thread-safe models)
- **Async**: All storage and pipeline interfaces use async/await

## Build & Test Commands

### Setup Environment

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (including dev dependencies)
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_entities.py

# Run single test with verbose output
uv run pytest tests/test_entities.py::test_canonical_promotion -v

# Run tests in parallel
uv run pytest -n auto
```

### Linting & Code Quality

Use the provided `lint.sh` script which runs all linters:

```bash
./lint.sh
```

Or run individual linters:

```bash
# Format check (does not modify files)
uv run black --check .

# Format code (modifies files)
uv run black .

# Ruff linter
uv run ruff check .

# Flake8
uv run flake8 . --count --show-source --statistics --exclude=.venv

# Pylint
uv run pylint $(find . -name "*.py" | grep -v venv)
```

## Code Style & Conventions

### General Guidelines

- **Line length**: 200 characters (configured in pyproject.toml for all tools)
- **Immutability**: All Pydantic models use `frozen=True` for thread safety
- **Type hints**: Always use type hints for function parameters and return values
- **Async/await**: Use async interfaces for all storage and pipeline operations
- **Abstract Base Classes**: Domain-specific implementations inherit from base interfaces

### Import Style

- Group imports: standard library, third-party, local
- Use absolute imports from `kgraph` package
- Avoid circular imports

### Testing Style

- Use pytest fixtures defined in `tests/conftest.py`
- Test files follow `test_*.py` naming convention
- Use async test functions with `@pytest.mark.asyncio` (auto-enabled via asyncio_mode)
- Test both success and error paths

### Model Design

- All models inherit from `pydantic.BaseModel`
- Use `frozen=True` for immutability
- Define `get_*` methods for polymorphic type identification
- Example:

```python
class MyEntity(BaseEntity):
    frozen: True
    
    def get_entity_type(self) -> str:
        return "my_type"
```

## Project Structure

```
kgraph/
├── entity.py           # BaseEntity, EntityStatus, EntityMention, PromotionConfig
├── relationship.py     # BaseRelationship
├── document.py         # BaseDocument, DocumentMetadata
├── domain.py           # DomainSchema ABC - domain-specific implementations
├── ingest.py           # IngestionOrchestrator - main pipeline coordinator
├── builders.py         # Builder pattern for creating entities/relationships
├── context.py          # ExecutionContext for orchestrator
├── export.py           # Export functionality for entities/relationships
├── clock.py            # ClockInterface for time operations
├── storage/
│   ├── interfaces.py   # Storage ABCs (EntityStorage, RelationshipStorage, etc.)
│   └── memory.py       # In-memory implementation for testing
├── pipeline/
│   ├── interfaces.py   # Parser, Extractor, Resolver ABCs
│   └── embedding.py    # EmbeddingGeneratorInterface
└── query/
    └── interfaces.py   # Query interface for graph operations
```

## What to Modify

✅ **Safe to modify**:
- Implementation files in `kgraph/`
- Test files in `tests/`
- Documentation files (`README.md`, `docs/`)
- Example code in `examples/`

⚠️ **Modify with caution**:
- `pyproject.toml` - only update if adding new dependencies
- `lint.sh` - only update if changing lint tools
- Configuration files (`.flake8`, `.pylintrc`, `pyproject.toml`)

❌ **Do not modify**:
- `uv.lock` - managed by uv
- `.git/` directory
- Generated files or build artifacts

## Common Tasks

### Adding a New Entity Type

1. Create a subclass of `BaseEntity` in the domain module
2. Implement required abstract methods (`get_entity_type()`, `get_canonical_id_source()`)
3. Add to domain's `entity_types` property
4. Add tests in `tests/test_entities.py` or create a new test file

### Adding a New Relationship Type

1. Create a subclass of `BaseRelationship` in the domain module
2. Implement required abstract methods (`get_edge_type()`)
3. Add to domain's `relationship_types` property
4. Add tests in `tests/test_relationships.py`

### Adding Storage Backend

1. Implement storage interfaces from `kgraph/storage/interfaces.py`
2. All methods must be async
3. Add integration tests
4. Document any external dependencies

### Adding Pipeline Component

1. Implement interfaces from `kgraph/pipeline/interfaces.py`
2. All processing methods must be async
3. Follow the pattern: input → process → output
4. Add unit tests and integration tests

## Dependencies

### Adding New Dependencies

Before adding a new dependency:
1. Check if functionality exists in Python standard library
2. Evaluate if it's truly necessary for the core functionality
3. Prefer well-maintained, popular packages
4. Add to `pyproject.toml` under `dependencies` or `dev` optional dependencies

Use `uv pip install <package>` to add new packages.

## Best Practices

1. **Always run tests** before committing changes: `uv run pytest`
2. **Run linters** to ensure code quality: `./lint.sh`
3. **Keep models immutable** - use `frozen=True` on Pydantic models
4. **Use async/await** for all I/O operations
5. **Write tests** for new functionality
6. **Update documentation** when changing public APIs
7. **Follow existing patterns** - check similar code before implementing new features
8. **Type everything** - use type hints for better IDE support and error catching

## Debugging Tips

- Use pytest's `-v` flag for verbose output
- Use `-s` flag to see print statements: `uv run pytest -s`
- Use `--pdb` to drop into debugger on failure: `uv run pytest --pdb`
- Check async context managers are properly awaited
- Verify Pydantic model validation errors with `ValidationError`

## Additional Resources

- Project README: `README.md`
- Architecture details: `docs/architecture.md`
- Domain implementation guide: `docs/domains.md`
- Storage backends: `docs/storage.md`
- Pipeline components: `docs/pipeline.md`

## Questions or Issues?

- Check existing tests for examples: `tests/`
- Review conftest.py for available fixtures: `tests/conftest.py`
- Look at example implementations: `examples/`
- Refer to the project VIBES.md for design philosophy
