"""
Tests for the MCP graphql_query tool.

Verifies that the tool executes GraphQL against the same schema and returns
the standard { data, errors } shape. Uses patched storage so we don't depend
on bundle/env.
"""

import pytest
from unittest.mock import patch

from mcp_server.server import graphql_query


@pytest.fixture
def mock_storage(populated_storage):
    """Provide populated storage to the MCP tool via _get_storage."""
    from contextlib import contextmanager

    @contextmanager
    def _get_storage():
        yield populated_storage

    return _get_storage


def _call_tool(query: str, variables=None):
    """Call the underlying MCP tool function (FastMCP wraps it in FunctionTool)."""
    return graphql_query.fn(query=query, variables=variables)


def test_graphql_query_returns_data_and_errors_shape(mock_storage):
    """graphql_query returns dict with 'data' and 'errors' keys."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("{ bundle { bundleId } }")
    assert "data" in result
    assert "errors" in result
    assert result["errors"] is None
    assert result["data"] is not None


def test_graphql_query_entity(mock_storage):
    """graphql_query can fetch an entity by id."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool('{ entity(id: "test:entity:1") { entityId name status } }')
    assert result["errors"] is None
    assert result["data"]["entity"]["entityId"] == "test:entity:1"
    assert result["data"]["entity"]["name"] == "Test Character 1"
    assert result["data"]["entity"]["status"] == "canonical"


def test_graphql_query_entities_paginated(mock_storage):
    """graphql_query can list entities with pagination."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("{ entities(limit: 2, offset: 0) { items { name } total limit offset } }")
    assert result["errors"] is None
    data = result["data"]["entities"]
    assert len(data["items"]) == 2
    assert data["total"] == 3
    assert data["limit"] == 2
    assert data["offset"] == 0


def test_graphql_query_returns_errors_on_invalid_query(mock_storage):
    """graphql_query returns errors in standard shape for invalid GraphQL."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("{ unknownField }")
    assert result["data"] is None
    assert result["errors"] is not None
    assert len(result["errors"]) >= 1
    assert "message" in result["errors"][0]
