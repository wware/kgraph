# Multi-stage build for KG server
FROM python:3.13-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy kgbundle first (local dependency)
COPY kgbundle ./kgbundle

# Copy dependency files from kgserver
COPY kgserver/pyproject.toml ./
COPY kgserver/uv.lock ./

# Install ALL dependencies using uv (not pip)
RUN apt update && \
    apt install -y git vim ripgrep
# Re-lock after path fix, then sync with relocatable venv
RUN uv venv --python /usr/local/bin/python3 --relocatable && \
    uv lock && uv sync && \
    uv pip install uvicorn fastapi zensical \
        mkdocs mkdocs-material pymdown-extensions

# Copy mkdocs config and build docs/ from repo markdown (kgserver index = site home)
COPY kgserver/mkdocs.yml ./
COPY kgserver/index.md ./docs/index.md
COPY docs/architecture.md ./docs/architecture.md
COPY docs/index.md ./docs/reference/index.md
COPY docs/api.md docs/bundle.md docs/canonical_ids.md docs/determinism.md docs/domains.md docs/pipeline.md docs/storage.md docs/graph_visualization.md docs/RELATIONSHIP_TRACING.md ./docs/reference/
COPY README.md CLAUDE.md LAMBDA_LABS.md MEDLIT_SCHEMA_SPEC.md IMPLEMENTATION_PLAN.md STATUS_20260212.md NEXT_STEPS.md JATS_PARSER_NOTES.md jupyter.md summary.md VIBES.md holmes_example_plan.md snapshot_semantics_v1.md RETHINK_INGESTION.md ./docs/project/
COPY PLAN1.md PLAN2.md PLAN3.md PLAN4.md PLAN5.md PLAN6.md PLAN7.md ./docs/project/plans/
COPY TODO1.md TODO2.md ./docs/project/todos/
COPY kgserver/index.md kgserver/DOCKER_SETUP.md kgserver/DOCKER_COMPOSE_GUIDE.md kgserver/LOCAL_DEV.md kgserver/MCP_CLIENT_SETUP.md kgserver/GRAPHQL_VIBES.md kgserver/MCP_GQL_WRAPPER.md ./docs/kgserver/
COPY kgserver/docs/architecture.md ./docs/kgserver/architecture.md
COPY kgserver/storage/README.md kgserver/storage/NEO4J_COMPATIBILITY.md ./docs/kgserver/storage/
COPY kgserver/storage/models/README.md ./docs/kgserver/storage/models/
COPY kgserver/storage/backends/README.md ./docs/kgserver/storage/backends/
COPY examples/medlit/README.md examples/medlit/TODO.md examples/medlit/CANONICAL_IDS.md ./docs/examples/medlit/
COPY examples/medlit_schema/README.md examples/medlit_schema/DEPTH_OF_FIELDS.md examples/medlit_schema/PROGRESS.md examples/medlit_schema/ONTOLOGY_GUIDE.md ./docs/examples/medlit_schema/
COPY examples/medlit_golden/README.md ./docs/examples/medlit_golden/
COPY examples/sherlock/README.md ./docs/examples/sherlock/
COPY .github/copilot-instructions.md ./docs/development/copilot-instructions.md

# Prebuild MkDocs site so runtime does not run zensical build (avoids hang on small droplets)
RUN uv run zensical build

# Final stage - use same base image to keep Python paths consistent
FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 unzip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from builder (symlinks will work since same Python path)
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/docs /app/docs
COPY --from=builder /app/site /app/site
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy kgbundle first, then kgserver (includes kgserver/chainlit for /chat UI)
COPY kgbundle ./kgbundle
COPY kgserver/ .
# Ingest worker (MCP ingest_paper) imports examples.medlit -> examples.medlit_schema -> kgschema
COPY kgschema ./kgschema
COPY kgraph ./kgraph
COPY medlit_bundle /bundle
COPY examples ./examples

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/kgbundle:/app/kgschema"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["tail", "-f", "/dev/null"]
