# Local Development Guide

This guide covers running kgserver locally while using Docker for PostgreSQL.

## Quick Start (Hybrid Setup)

The most common development setup: PostgreSQL in Docker, Python server running directly.

### 1. Start PostgreSQL

```bash
cd kgserver
docker compose up -d postgres
```

This creates a database named `kgserver` (as defined in `docker-compose.yml`).

### 2. Set Environment Variables

```bash
# Database connection (must match docker-compose.yml POSTGRES_DB)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/kgserver"

# Bundle to load
export BUNDLE_PATH="/path/to/your/bundle.zip"

# Optional: Local docs directory (avoids /app/docs permission issues)
export KGSERVER_DOCS_DIR="./docs_output"
export KGSERVER_APP_ROOT="."
```

### 3. Run the Server

```bash
cd kgserver/query
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify

```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}
```

## Common Issues

### "database does not exist"

The DATABASE_URL database name must match `POSTGRES_DB` in docker-compose.yml.

```yaml
# docker-compose.yml
postgres:
  environment:
    POSTGRES_DB: kgserver  # <-- This is the database name
```

```bash
# Must match:
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/kgserver"
```

If you need a different database name, either:
1. Update docker-compose.yml and restart postgres, OR
2. Create the database manually:

```bash
docker exec -it kgserver-postgres-1 psql -U postgres -c "CREATE DATABASE kgraph;"
```

### "Permission denied: /app/docs"

The `/app/docs` path is designed for Docker containers. For local development, override it:

```bash
export KGSERVER_DOCS_DIR="./docs_output"
export KGSERVER_APP_ROOT="."
mkdir -p docs_output
```

### PostgreSQL connection refused

Ensure postgres container is running and healthy:

```bash
docker compose ps
docker compose logs postgres
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `BUNDLE_PATH` | (required) | Path to bundle .zip file |
| `KGSERVER_DOCS_DIR` | `/app/docs` | Where to extract documentation assets |
| `KGSERVER_APP_ROOT` | `/app` | App root for mkdocs.yml placement |

## Full Docker Setup

To run everything in Docker (including the API server):

```bash
docker compose up --build
```

This is the production-like setup where all paths work as expected.

## Useful Commands

```bash
# View postgres logs
docker compose logs -f postgres

# Connect to postgres directly
docker exec -it kgserver-postgres-1 psql -U postgres -d kgserver

# List tables
docker exec -it kgserver-postgres-1 psql -U postgres -d kgserver -c '\dt'

# Stop postgres (preserves data)
docker compose down

# Stop postgres and DELETE all data
docker compose down -v
```
