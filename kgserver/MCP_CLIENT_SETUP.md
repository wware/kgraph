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

1. Configure nginx with the snippet in **`kgserver/nginx-mcp.conf`** so `/mcp/` proxies to the `mcpserver` container.
2. The external SSE URL is **`https://YOUR_DOMAIN/mcp/sse`** (or `http://` if not using TLS). Replace `YOUR_DOMAIN` with the host you use (e.g. the same host as the API).

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
