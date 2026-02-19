# Plan: Gather all repo Markdown into MkDocs for droplet server

Collect all Markdown files from the repository into a single organized tree under the MkDocs `docs/` directory so the droplet server serves them with working navigation and search. Execute the steps in order from the **repository root**. Build context for Docker is the repo root (`docker build -f kgserver/Dockerfile .`).

**Exclusions (do not copy):**
- `medlit_bundle/docs/README.md` (generated at export)
- `kgserver/query/.ipynb_checkpoints/README-checkpoint.md` (checkpoint)
- `PLAN5.md` (this plan; not part of the doc set)

**Optional:** Include `.github/copilot-instructions.md` under `docs/development/`. If you omit it, remove its COPY and its nav entry.

---

## Step 0. Pre-flight: verify all source files exist

Run from repo root. If any path is missing, create the file or remove it from the plan (and from the COPY/nav in the steps below).

```bash
for f in \
  README.md CLAUDE.md LAMBDA_LABS.md MEDLIT_SCHEMA_SPEC.md IMPLEMENTATION_PLAN.md STATUS_20260212.md NEXT_STEPS.md JATS_PARSER_NOTES.md jupyter.md summary.md VIBES.md holmes_example_plan.md snapshot_semantics_v1.md \
  PLAN1.md PLAN2.md PLAN3.md PLAN4.md \
  TODO1.md TODO2.md \
  docs/index.md docs/architecture.md docs/api.md docs/bundle.md docs/canonical_ids.md docs/determinism.md docs/domains.md docs/pipeline.md docs/storage.md docs/graph_visualization.md docs/RELATIONSHIP_TRACING.md \
  kgserver/index.md kgserver/DOCKER_SETUP.md kgserver/DOCKER_COMPOSE_GUIDE.md kgserver/LOCAL_DEV.md kgserver/GRAPHQL_VIBES.md kgserver/MCP_GQL_WRAPPER.md kgserver/docs/architecture.md \
  kgserver/storage/README.md kgserver/storage/NEO4J_COMPATIBILITY.md kgserver/storage/models/README.md kgserver/storage/backends/README.md \
  examples/medlit/README.md examples/medlit/TODO.md examples/medlit/CANONICAL_IDS.md \
  examples/medlit_schema/README.md examples/medlit_schema/DEPTH_OF_FIELDS.md examples/medlit_schema/PROGRESS.md examples/medlit_schema/ONTOLOGY_GUIDE.md \
  examples/medlit_golden/README.md examples/sherlock/README.md \
  .github/copilot-instructions.md; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done
```

Fix any MISSING before proceeding. If you exclude development, omit `.github/copilot-instructions.md` from the loop and from Step 1 and Step 2.

---

## Step 1. Update `kgserver/mkdocs.yml` nav

**File:** `kgserver/mkdocs.yml`

**Action:** Replace the existing `nav:` block (lines 23–25) with the block below. Leave everything above `nav:` (site_name through `toc:`) unchanged.

**New `nav` block (paste exactly, same indentation as current nav):**

```yaml
nav:
  - Home: index.md
  - Architecture: architecture.md
  - Project:
    - Overview: project/README.md
    - CLAUDE: project/CLAUDE.md
    - Lambda Labs: project/LAMBDA_LABS.md
    - Medlit schema spec: project/MEDLIT_SCHEMA_SPEC.md
    - Implementation plan: project/IMPLEMENTATION_PLAN.md
    - Status: project/STATUS_20260212.md
    - Next steps: project/NEXT_STEPS.md
    - JATS parser notes: project/JATS_PARSER_NOTES.md
    - Jupyter: project/jupyter.md
    - Summary: project/summary.md
    - VIBES: project/VIBES.md
    - Holmes example plan: project/holmes_example_plan.md
    - Snapshot semantics: project/snapshot_semantics_v1.md
    - Plans:
      - PLAN1: project/plans/PLAN1.md
      - PLAN2: project/plans/PLAN2.md
      - PLAN3: project/plans/PLAN3.md
      - PLAN4: project/plans/PLAN4.md
    - TODOs:
      - TODO1: project/todos/TODO1.md
      - TODO2: project/todos/TODO2.md
  - Reference:
    - Framework overview: reference/index.md
    - API: reference/api.md
    - Architecture: reference/architecture.md
    - Bundle: reference/bundle.md
    - Canonical IDs: reference/canonical_ids.md
    - Determinism: reference/determinism.md
    - Domains: reference/domains.md
    - Pipeline: reference/pipeline.md
    - Storage: reference/storage.md
    - Graph visualization: reference/graph_visualization.md
    - Relationship tracing: reference/RELATIONSHIP_TRACING.md
  - KGServer:
    - Server home: kgserver/index.md
    - Docker setup: kgserver/DOCKER_SETUP.md
    - Docker Compose guide: kgserver/DOCKER_COMPOSE_GUIDE.md
    - Local dev: kgserver/LOCAL_DEV.md
    - GraphQL VIBES: kgserver/GRAPHQL_VIBES.md
    - MCP GraphQL wrapper: kgserver/MCP_GQL_WRAPPER.md
    - Architecture: kgserver/architecture.md
    - Storage:
      - Overview: kgserver/storage/README.md
      - Neo4j compatibility: kgserver/storage/NEO4J_COMPATIBILITY.md
      - Models: kgserver/storage/models/README.md
      - Backends: kgserver/storage/backends/README.md
  - Examples:
    - Medlit: examples/medlit/README.md
    - Medlit TODO: examples/medlit/TODO.md
    - Medlit canonical IDs: examples/medlit/CANONICAL_IDS.md
    - Medlit schema: examples/medlit_schema/README.md
    - Medlit schema depth: examples/medlit_schema/DEPTH_OF_FIELDS.md
    - Medlit schema progress: examples/medlit_schema/PROGRESS.md
    - Medlit schema ontology: examples/medlit_schema/ONTOLOGY_GUIDE.md
    - Medlit golden: examples/medlit_golden/README.md
    - Sherlock: examples/sherlock/README.md
  - Development:
    - Copilot instructions: development/copilot-instructions.md
```

If you excluded development: remove the entire `- Development:` block (last 3 lines).

---

## Step 2. Update Dockerfile builder stage (docs + MkDocs build)

**File:** `kgserver/Dockerfile`

**Action:** Replace the three lines that copy docs and build the site with the block below.

**Current lines to remove (builder stage only):**
```dockerfile
# Copy mkdocs config and docs for building
COPY kgserver/mkdocs.yml ./
COPY kgserver/docs ./docs
COPY kgserver/index.md ./docs/index.md

# Prebuild MkDocs site so runtime does not run zensical build (avoids hang on small droplets)
RUN uv run zensical build
```

**New block to insert in their place (same location, after uv pip install ...):**

```dockerfile
# Copy mkdocs config and build docs/ from repo markdown (framework = site home)
COPY kgserver/mkdocs.yml ./
COPY docs/index.md ./docs/index.md
COPY docs/architecture.md ./docs/architecture.md
COPY docs/index.md ./docs/reference/index.md
COPY docs/api.md docs/bundle.md docs/canonical_ids.md docs/determinism.md docs/domains.md docs/pipeline.md docs/storage.md docs/graph_visualization.md docs/RELATIONSHIP_TRACING.md ./docs/reference/
COPY README.md CLAUDE.md LAMBDA_LABS.md MEDLIT_SCHEMA_SPEC.md IMPLEMENTATION_PLAN.md STATUS_20260212.md NEXT_STEPS.md JATS_PARSER_NOTES.md jupyter.md summary.md VIBES.md holmes_example_plan.md snapshot_semantics_v1.md ./docs/project/
COPY PLAN1.md PLAN2.md PLAN3.md PLAN4.md ./docs/project/plans/
COPY TODO1.md TODO2.md ./docs/project/todos/
COPY kgserver/index.md kgserver/DOCKER_SETUP.md kgserver/DOCKER_COMPOSE_GUIDE.md kgserver/LOCAL_DEV.md kgserver/GRAPHQL_VIBES.md kgserver/MCP_GQL_WRAPPER.md ./docs/kgserver/
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
```

If you excluded development: remove the line `COPY .github/copilot-instructions.md ./docs/development/copilot-instructions.md`.

---

## Step 3. Update Dockerfile final stage (stop overwriting docs)

**File:** `kgserver/Dockerfile`

**Action:** Remove the line that copies kgserver index over docs in the final stage so `/app/docs` in the image matches what the builder produced.

**Line to remove (final stage, after `COPY kgserver/ .`):**
```dockerfile
COPY kgserver/index.md ./docs/index.md
```

Delete only that one line. The comment above it ("Ensure docs has index.md...") can also be removed.

---

## Step 4. Verification

Run from **repository root**.

```bash
# Build (must succeed; no MISSING files)
docker build -f kgserver/Dockerfile -t kgserver:test .

# Run
docker run --rm -p 8000:8000 kgserver:test
```

Then in a browser:

1. Open `http://localhost:8000/` — site home (framework index) loads.
2. Open `http://localhost:8000/kgserver/` — kgserver index loads.
3. Use the left nav: open at least one page from Project, Reference, KGServer, Examples (and Development if included).
4. Use the search box — query e.g. "docker" or "pipeline"; results from different sections appear.

If anything fails: fix the failing step (missing file, typo in path, or nav key) and re-run from that step.

---

## Reference: source → destination map

All paths relative to repo root. Used only if you need to adjust COPY or add a file later.

| Destination under `docs/` | Source |
|---------------------------|--------|
| `index.md` | `docs/index.md` |
| `architecture.md` | `docs/architecture.md` |
| `reference/index.md` | `docs/index.md` |
| `reference/api.md` | `docs/api.md` |
| `reference/architecture.md` | `docs/architecture.md` |
| `reference/bundle.md` | `docs/bundle.md` |
| `reference/canonical_ids.md` | `docs/canonical_ids.md` |
| `reference/determinism.md` | `docs/determinism.md` |
| `reference/domains.md` | `docs/domains.md` |
| `reference/pipeline.md` | `docs/pipeline.md` |
| `reference/storage.md` | `docs/storage.md` |
| `reference/graph_visualization.md` | `docs/graph_visualization.md` |
| `reference/RELATIONSHIP_TRACING.md` | `docs/RELATIONSHIP_TRACING.md` |
| `project/README.md` … `project/snapshot_semantics_v1.md` | root `README.md` … `snapshot_semantics_v1.md` |
| `project/plans/PLAN1.md` … `PLAN4.md` | `PLAN1.md` … `PLAN4.md` |
| `project/todos/TODO1.md`, `TODO2.md` | `TODO1.md`, `TODO2.md` |
| `kgserver/index.md` … `kgserver/MCP_GQL_WRAPPER.md` | `kgserver/` same names |
| `kgserver/architecture.md` | `kgserver/docs/architecture.md` |
| `kgserver/storage/README.md` … `backends/README.md` | `kgserver/storage/` same layout |
| `examples/medlit/README.md` … `CANONICAL_IDS.md` | `examples/medlit/` same names |
| `examples/medlit_schema/` … | `examples/medlit_schema/` same names |
| `examples/medlit_golden/README.md` | `examples/medlit_golden/README.md` |
| `examples/sherlock/README.md` | `examples/sherlock/README.md` |
| `development/copilot-instructions.md` | `.github/copilot-instructions.md` |
