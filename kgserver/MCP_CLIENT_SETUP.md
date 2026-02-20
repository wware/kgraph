# MCP client setup (Cursor IDE & Claude Code)

This guide explains how to connect **Cursor IDE** or **Claude Code** (on Linux) to the Knowledge Graph MCP server so you can confirm it works and have a conversation with the graph.

## Prerequisites

- **API and MCP servers** must be running (e.g. `docker compose --profile api up -d`, or run API and MCP with uvicorn locally).
- A **bundle must be loaded**: the API loads it at startup when `BUNDLE_PATH` is set. The MCP server only reads from the same database; it does not load bundles.
- **MCP SSE endpoint:**
  - **Local:** `http://localhost:8001/sse`
  - **Cloud (behind nginx):** `https://YOUR_HOST/mcp/sse` (or `http://` if no TLS). Use the same host as your API when nginx is in front.

## Running locally (no nginx)

**With Docker:**

```bash
docker compose --profile api up -d
```

MCP is at **`http://localhost:8001/sse`**.

**Without Docker:** Start the API and MCP from the kgserver app directory with `DATABASE_URL` (and `BUNDLE_PATH` for the API) set:

```bash
# Terminal 1 – API
cd kgserver && uv run uvicorn main:app --port 8000

# Terminal 2 – MCP
cd kgserver && uv run uvicorn mcp_main:app --port 8001
```

MCP URL is again **`http://localhost:8001/sse`**.

## Running in the cloud

1. Configure nginx with the snippet in **`kgserver/nginx-mcp.conf`** so `/mcp/` proxies to the MCP server (see below for step-by-step on a droplet).
2. The external SSE URL is **`https://YOUR_DOMAIN/mcp/sse`** (or `http://` if not using TLS). Replace `YOUR_DOMAIN` with the host you use (e.g. the same host as the API).

---

## Adding MCP to nginx on a droplet (all steps)

These steps assume you already have nginx on the droplet and the kgserver stack (API, mcpserver, postgres) running via Docker with ports 8000 and 8001 exposed on the host.

**1. Ensure the MCP container is running**

From your project directory on the droplet:

```bash
docker compose --profile api up -d
docker compose --profile api ps
```

You should see `api` and `mcpserver` (and `postgres`) up. Check MCP health:

```bash
curl -s http://localhost:8001/health
```

Expect `{"status":"ok"}`.

**2. Open your nginx site config**

If you use a single site file (e.g. from jupyter.md):

```bash
sudo vim /etc/nginx/sites-available/kgraph
```

If you use a different path (e.g. `sites-available/default` or `conf.d/`), open that file instead.

**3. Add the MCP location blocks (two blocks required)**

Inside the `server { ... }` block (the same block where you have `location /` for the API), add **two** `location` blocks. Place both **before** the catch-all `location /` so they are matched first.

The MCP client GETs `/mcp/sse` to open the stream, then POSTs to `/messages/?session_id=...` (the URL is sent in the SSE event). If nginx does not proxy `/messages/` to the MCP server, those POSTs hit the API and you get **405 Method Not Allowed** on the api container. So you need both blocks.

Add exactly (use `127.0.0.1` when nginx is on the host; use `mcpserver` if nginx is in Docker on the same network):

```nginx
    # MCP: messages endpoint (client POSTs here; must go to mcpserver, not API)
    location /messages/ {
        proxy_pass http://127.0.0.1:8001/messages/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 300s;
    }

    # MCP: SSE endpoint
    location /mcp/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
    }
```

Notes:

- Use **`http://127.0.0.1:8001/`** when nginx is on the **host** and Docker exposes 8001 on localhost. If nginx runs in a container on the same Docker network as `mcpserver`, use **`http://mcpserver:8001/`** in both blocks.
- The trailing slash in `proxy_pass http://127.0.0.1:8001/` makes nginx strip `/mcp/` so the MCP server sees `/sse` for a request to `/mcp/sse`.
- **`proxy_buffering off`** and **`proxy_read_timeout 3600s`** are required for SSE; without them the stream can hang or not reach the client.

**4. Test and reload nginx**

```bash
sudo nginx -t
sudo systemctl reload nginx
```

If `nginx -t` reports an error, fix the config (e.g. missing semicolon, wrong brace) and try again.

**5. (Optional) If you use HTTPS**

If your site is already behind SSL (e.g. Let’s Encrypt), the same `location /mcp/` block works inside your `listen 443 ssl` server block; no extra SSL config is needed for MCP. Just add the block in step 3 to that server block and reload.

**6. Verify from outside**

From your laptop (replace with your droplet hostname or IP):

```bash
# might be http rather than https...
curl -s https://YOUR_DROPLET/mcp/sse -N -H "Accept: text/event-stream" -m 3
```

You should see SSE output (e.g. `event: endpoint` and a `data:` line). Then in Cursor or Claude Code set the MCP URL to **`https://YOUR_DROPLET/mcp/sse`** (or `http://` if you are not using TLS).

## Cursor IDE (Linux)

**Config file:** Project root **`.cursor/mcp.json`** or your user-level Cursor MCP config.

**Local:**

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

**Cloud:** Same structure with the public URL:

```json
{
  "mcpServers": {
    "knowledge-graph": {
      "type": "url",
      "url": "https://YOUR_DOMAIN/mcp/sse"
    }
  }
}
```

Restart Cursor or reload MCP servers so the config is picked up. The server should appear as **"knowledge-graph"** in Cursor’s MCP / tools list.

## Claude Code (Linux)

Config can live in **`~/.claude/mcp_servers.json`**, **`~/.claude/settings.json`**, or project **`.claude/settings.local.json`**. Use the format documented by Claude Code for an SSE/URL MCP server.

- **Local:** set the MCP server URL to **`http://localhost:8001/sse`**.
- **Cloud:** set the MCP server URL to **`https://YOUR_DOMAIN/mcp/sse`** (or `http://` as appropriate).

Refer to [Claude Code MCP documentation](https://docs.anthropic.com/en/docs/claude-code/mcp) for the exact JSON keys if they differ from Cursor.

## Debugging the MCP server (localhost:8001)

If the IDE shows the server as disconnected or tool calls fail, check:

1. **Server running:** `curl -s http://localhost:8001/health` → `{"status":"ok"}`.
2. **SSE endpoint:** Use `http://localhost:8001/sse` in your MCP client config.
3. **SSE responds:** `curl -s -N -H "Accept: text/event-stream" -m 3 http://localhost:8001/sse` should stream at least one event (e.g. `event: endpoint` and a `data:` line with a messages URL).
4. **Cursor/Claude config:** Ensure `mcp.json` (or Claude config) uses exactly `http://localhost:8001/sse` for local. Restart the IDE or reload MCP after changing the URL.

## Confirming it works and having a conversation

1. **Check the server is available**
   - **Cursor:** Open MCP/tools settings and confirm **"knowledge-graph"** (or the name you used) is listed and connected.
   - **Claude Code:** Confirm the MCP server appears and is enabled.

2. **Confirm with a tool call**  
   In a new chat/composer session, ask the AI to use the knowledge graph, for example:
   - “Use the knowledge graph MCP to list some entities” or “Call `list_entities` with limit 5.”
   - “Get entity with id …” (use a real entity ID from your bundle) via `get_entity`.

3. The model should invoke tools such as **`list_entities`**, **`get_entity`**, **`search_entities`**, or **`find_relationships`** and return real data from your graph. That confirms the MCP server is working.

4. You can then ask follow-up questions (e.g. “What relationships does that entity have?”) to have a conversation with the graph.
