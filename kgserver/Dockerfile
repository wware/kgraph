# Multi-stage build for KG server
FROM python:3.13-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./

# Install ALL dependencies using uv (not pip)
RUN apt update && \
    apt install -y git vim ripgrep
RUN uv sync --frozen && \
    uv pip install uvicorn fastapi \
        mkdocs mkdocs-material pymdown-extensions

# Copy mkdocs config and docs for building
COPY mkdocs.yml ./
COPY docs ./docs
COPY README.md ./docs/index.md

# Final stage
FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/docs /app/docs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN mkdir -p /app/site
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/kgraph"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["tail", "-f", "/dev/null"]
