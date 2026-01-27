"""
MCP (Model Context Protocol) server for Knowledge Graph GraphQL API.

This module provides an MCP server that wraps the GraphQL API, making it
accessible to AI agents like Claude or Cursor IDE.

The server can run in two modes:
1. HTTP/SSE mode: Mounted as FastAPI routes for remote access
2. STDIO mode: Standalone server for local subprocess communication
"""

from .server import mcp_server

__all__ = ["mcp_server"]
