# Plan: MCP server in separate container with SSE (PLAN7)

Execute steps in order from the **repository root**. Reference: **mcp_work.md**.

**Scope:** Move the MCP server from the FastAPI container into its own Docker service on port 8001, using SSE (not stdio). Ensure the MCP container receives the same configuration (e.g. `DATABASE_URL`) and add an nginx snippet for proxying MCP with SSE-friendly settings. Document how to use the MCP server from Cursor IDE or Claude Code on Linux (locally and in the cloud) and how to confirm it works by having a conversation with the graph.

---

## Step 0. Pre-flight

From repo root:

```bash
(./lint.sh && uv run pytest kgserver/ -v --tb=short) 2>&1 | tail -40
```

Note any failures. After each step below, re-run the same and fix regressions.

---

## Step 1. Standalone MCP entrypoint (port 8001, SSE only)

**Goal:** Add a runnable ASGI app that serves only the MCP server over SSE on port 8001, so it can be run in a separate container.

**New file:** `kgserver/mcp_main.py`

**Content:**

- Import the existing `mcp_server` from `mcp_server` (same as in `query/server.py`).
- Build the SSE app with `mcp_server.http_app(path="/sse", transport="sse")` and assign it to a variable (e.g. `sse_app`).
- Create a minimal Starlette/FastAPI app that:
  - Mounts the SSE app at `/sse` (so the MCP SSE endpoint is at `/sse`; nginx will forward `/mcp/` to this server and strip the prefix, so `/mcp/sse` becomes `/sse`).
  - Adds a `GET /health` that returns `{"status": "ok"}` for container healthchecks.
- Assign the top-level app to `app` so uvicorn can run it: `uvicorn mcp_main:app --host 0.0.0.0 --port 8001`.

**Details:**

- Use a single top-level app. If using FastAPI: `app = FastAPI()`, then `app.mount("/sse", sse_app)` and `@app.get("/health")` for health. The root `/` can return a short JSON or 404.
- No bundle loading or lifespan in this entrypoint: the MCP tools call `get_engine()` / `get_storage()` from `query.storage_factory`, which read `DATABASE_URL` from the environment. The API container is responsible for loading the bundle; the MCP container only needs `DATABASE_URL` to query the same Postgres.

**Verification:**

From repo root (with `DATABASE_URL` set to a valid Postgres or SQLite URL):

```bash
cd kgserver && uv run uvicorn mcp_main:app --host 0.0.0.0 --port 8001
```

In another terminal:

- `curl -s http://localhost:8001/health` → `{"status":"ok"}` or equivalent.
- `curl -s -N -H "Accept: text/event-stream" http://localhost:8001/sse` → SSE stream (may require MCP client; at least no connection error).

Stop the server and re-run tests from Step 0.

---

## Step 2. Docker Compose: add MCP service and wire env

**Goal:** Add a second service that runs the MCP server on port 8001, depending on the API service, and pass configuration (e.g. `DATABASE_URL`) so the MCP server can use the same Postgres.

**File:** `docker-compose.yml`

**Changes:**

1. **New service `mcpserver`:**
   - `build`: same as `api` (context `.`, dockerfile `kgserver/Dockerfile`).
   - `command`: run the MCP entrypoint, e.g. `["python", "-m", "uvicorn", "mcp_main:app", "--host", "0.0.0.0", "--port", "8001"]`. Ensure `WORKDIR` in the Dockerfile is `/app` and `kgserver/` is copied into `/app`, so the module path is `mcp_main:app` (i.e. `kgserver/mcp_main.py` is at `/app/mcp_main.py`). If the image keeps app code under a subdirectory, use the appropriate module path (e.g. if code is at `/app` and `mcp_main.py` is there, the command above is correct; the existing Dockerfile copies `COPY kgserver/ .` so files end up at `/app/mcp_main.py` if we add it under kgserver — so the copy is `kgserver/` → `/app/`, meaning `kgserver/mcp_main.py` → `/app/mcp_main.py`; hence `mcp_main:app` is correct).
   - `environment`: pass at least `DATABASE_URL` with the same value as `api` (e.g. `postgresql://postgres:postgres@postgres:5432/kgserver`). Do **not** pass `BUNDLE_PATH` unless you want the MCP container to be able to load bundles (current design: only API loads the bundle; MCP only reads DB).
   - `ports`: expose `8001:8001`.
   - `depends_on`: `api` with `condition: service_healthy` (so MCP starts after API and its healthcheck pass; API is the one that loads the bundle into Postgres).
   - `networks`: `kgserver-network`.
   - `restart`: `unless-stopped`.
   - `healthcheck`: `curl` or `python -c "urllib.request.urlopen('http://localhost:8001/health')"` every 30s, timeout 10s, retries 3, start_period 10s.

2. **Profiles:** Add `mcpserver` to the same profile(s) as `api` (e.g. `profiles: ["api"]`) so `docker compose --profile api up` brings up both API and MCP.

**Verification:**

```bash
docker compose --profile api build
docker compose --profile api up -d
curl -s http://localhost:8001/health
docker compose --profile api down
```

---

## Step 3. Remove MCP from FastAPI app

**Goal:** Stop mounting the MCP server on the API app so MCP is only served by the new container.

**File:** `kgserver/query/server.py`

**Changes:**

- Remove the import of `mcp_server` and the four lines that create and mount `mcp_sse_app` and `mcp_http_app` (the block that mounts `/mcp/sse` and `/mcp`).

**Verification:**

- Run tests (Step 0).
- After `docker compose --profile api up -d`, `curl -s http://localhost:8000/health` still works; `curl -s http://localhost:8000/mcp/sse` should 404 (or no longer serve MCP). MCP is only on port 8001.

---

## Step 4. Nginx snippet for MCP (SSE)

**Goal:** Add a reference nginx location block so that when nginx is used in front of the stack, `/mcp/` can be proxied to the MCP container with SSE-safe settings.

**New file:** `kgserver/nginx-mcp.conf` (or `deploy/nginx-mcp.conf`; choose one and document it)

**Content:** The exact snippet from **mcp_work.md**:

```nginx
# FastMCP server (SSE transport)
location /mcp/ {
    proxy_pass http://mcpserver:8001/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # SSE requires these — disable buffering so events stream immediately
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;  # SSE connections are long-lived
}
```

**Note:** The trailing slash on `proxy_pass http://mcpserver:8001/` ensures nginx strips the `/mcp/` prefix, so the MCP server sees paths like `/sse` when the client requests `/mcp/sse`.

Add a short comment at the top of the file: include this in your nginx server block when using docker-compose with the `mcpserver` service; `proxy_buffering off` and `proxy_read_timeout` are required for SSE.

**Verification:** No automated check; file is for reference. Optionally run `nginx -t` if you have a full nginx config that includes this file.

---

## Step 5. Docs: Dockerfile, service overview, and MCP client setup

**Goal:** Ensure the Dockerfile includes the new entrypoint, docs mention the MCP service, and users have clear instructions to use the MCP server from Cursor IDE or Claude Code (Linux) both locally and in the cloud, including how to confirm it works and have a conversation with it.

**File:** `kgserver/Dockerfile`

- Confirm that `COPY kgserver/ .` copies the whole `kgserver/` directory (including `mcp_main.py` once added). No change needed if `mcp_main.py` is under `kgserver/` and the existing `COPY kgserver/ .` is in place.

**File:** `kgserver/DOCKER_COMPOSE_GUIDE.md` or `kgserver/DOCKER_SETUP.md` (whichever describes services)

- Add a short subsection for the MCP service: runs on port 8001, SSE only; receives `DATABASE_URL`; nginx snippet is in `kgserver/nginx-mcp.conf` (or the path you chose). Point to the client setup doc for connecting from an IDE.

**New file:** `kgserver/MCP_CLIENT_SETUP.md`

This doc must give **clear, copy-paste-friendly instructions** so a user can connect Cursor IDE or Claude Code (on Linux) to the MCP server and confirm it works by having a conversation that uses the graph. Include the following sections.

1. **Prerequisites**
   - API and MCP servers must be running (e.g. `docker compose --profile api up -d` or local uvicorn for both).
   - A bundle must be loaded (API loads it at startup when `BUNDLE_PATH` is set); MCP only reads from the same database.
   - The MCP SSE endpoint: **local** = `http://localhost:8001/sse`, **cloud** (behind nginx) = `https://YOUR_HOST/mcp/sse` (or `http://` if no TLS). Use the same host/port as your API if nginx is in front.

2. **Running locally (no nginx)**
   - With Docker: `docker compose --profile api up -d`; MCP is at `http://localhost:8001/sse`.
   - Without Docker: start API (e.g. `uvicorn main:app --port 8000`) and MCP (e.g. `uvicorn mcp_main:app --port 8001`) from the kgserver app directory, with `DATABASE_URL` (and `BUNDLE_PATH` for the API) set. MCP URL is again `http://localhost:8001/sse`.

3. **Running in the cloud**
   - Ensure nginx is configured with the snippet in `kgserver/nginx-mcp.conf` so `/mcp/` proxies to the `mcpserver` container. External SSE URL is then `https://YOUR_DOMAIN/mcp/sse` (or `http://` if not using TLS). Replace `YOUR_DOMAIN` with the host users actually use (e.g. the same host as the API).

4. **Cursor IDE (Linux)**
   - Config file: project root `.cursor/mcp.json` or user-level Cursor MCP config.
   - Example for **local**:
     ```json
     {
       "mcpServers": {
         "knowledge-graph": {
           "type": "url",
           "url": "http://localhost:8001/sse"
         }
       }
     }
     ```
   - Example for **cloud**: same structure with `"url": "https://YOUR_DOMAIN/mcp/sse"` (or `http://` as appropriate).
   - Restart Cursor or reload MCP servers so it picks up the config. The server should appear as "knowledge-graph" in Cursor’s MCP / tools list.

5. **Claude Code (Linux)**
   - Config can live in `~/.claude/mcp_servers.json`, `~/.claude/settings.json`, or project `.claude/settings.local.json`. Use the format documented by Claude Code for an SSE/URL MCP server (e.g. a server entry with the SSE URL).
   - **Local:** set the MCP server URL to `http://localhost:8001/sse`.
   - **Cloud:** set the MCP server URL to `https://YOUR_DOMAIN/mcp/sse` (or `http://` as appropriate).
   - (If the exact JSON key names differ from Cursor, use the official Claude Code MCP docs and provide the minimal working example for the SSE URL.)

6. **Confirming it works and having a conversation**
   - **Check the server is available:** In Cursor, open MCP/tools settings and confirm "knowledge-graph" (or the name you used) is listed and connected. In Claude Code, confirm the MCP server appears and is enabled.
   - **Confirm with a tool call:** In a new chat/composer session, ask the AI to use the knowledge graph, for example:
     - “Use the knowledge graph MCP to list some entities” or “Call list_entities with limit 5.”
     - “Get entity with id …” (use a real entity ID from your bundle) via `get_entity`.
   - The model should invoke tools such as `list_entities`, `get_entity`, `search_entities`, or `find_relationships` and return real data from your graph. That confirms the MCP server is working; you can then ask follow-up questions (e.g. “What relationships does that entity have?”) to have a conversation with the graph.

**Verification:** Read through the edited docs. Manually follow the “Confirming it works” steps once with either Cursor or Claude Code (local or cloud) and confirm you can see the server and get a successful tool response in a conversation.

---

## Step 6. Final verification

From repo root:

1. **Tests:** `./lint.sh && uv run pytest kgserver/ -v --tb=short`
2. **Compose:** `docker compose --profile api up -d`
3. **API:** `curl -s http://localhost:8000/health` → `{"status":"ok"}`
4. **MCP:** `curl -s http://localhost:8001/health` → `{"status":"ok"}`
5. **No MCP on API:** `curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/mcp/sse` → expect 404 (or 405) so MCP is not served on port 8000.
6. **Compose down:** `docker compose --profile api down`

---

## Summary

| Step | Action |
|------|--------|
| 1 | Add `kgserver/mcp_main.py`: SSE app at `/sse`, `/health`, runnable with uvicorn on 8001 |
| 2 | Add `mcpserver` service in `docker-compose.yml` (build, command, env, port 8001, depends_on api, healthcheck, profile) |
| 3 | Remove MCP mount and import from `kgserver/query/server.py` |
| 4 | Add `kgserver/nginx-mcp.conf` (or `deploy/nginx-mcp.conf`) with the SSE proxy snippet |
| 5 | Ensure Dockerfile copies `mcp_main.py`; add MCP subsection to docker docs; add `kgserver/MCP_CLIENT_SETUP.md` with local/cloud URLs, Cursor and Claude Code config examples, and steps to confirm it works and have a conversation |
| 6 | Run tests and full compose smoke check |
