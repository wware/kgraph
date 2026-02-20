"""
Standalone MCP server entrypoint (SSE on port 8001).

Run with: uvicorn mcp_main:app --host 0.0.0.0 --port 8001

SSE endpoint: http://localhost:8001/sse (mount at root so path stays /sse).
"""

from fastapi import FastAPI

from mcp_server import mcp_server

sse_app = mcp_server.http_app(path="/sse", transport="sse")

app = FastAPI(title="Knowledge Graph MCP", version="0.1.0")


# Register /health before the mount so it takes precedence
@app.get("/health")
async def health():
    """Health check for container/orchestration."""
    return {"status": "ok"}


# Mount at root so FastMCP's /sse route is exposed as /sse (not /sse/sse)
app.mount("/", sse_app)
