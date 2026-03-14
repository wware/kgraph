"""
Tests for the MCP bfs_subgraph tool.

Uses populated_storage fixture. Verifies response structure and stub vs full.
"""

import pytest
from unittest.mock import patch

from mcp_server.server import bfs_subgraph


@pytest.fixture
def mock_storage(populated_storage):
    """Provide populated storage to the MCP tool via _get_storage."""
    from contextlib import contextmanager

    @contextmanager
    def _get_storage():
        yield populated_storage

    return _get_storage


def _call_tool(seeds, max_hops=2, node_filter=None, edge_filter=None):
    """Call the underlying MCP tool function."""
    return bfs_subgraph.fn(
        seeds=seeds,
        max_hops=max_hops,
        node_filter=node_filter,
        edge_filter=edge_filter,
    )


def test_bfs_subgraph_structure(mock_storage, populated_storage):
    """Result has seeds, max_hops, node_count, edge_count, truncated, nodes, edges."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool(["test:entity:1"], max_hops=2)

    assert "seeds" in result
    assert result["seeds"] == ["test:entity:1"]
    assert "max_hops" in result
    assert "node_count" in result
    assert "edge_count" in result
    assert "truncated" in result
    assert "nodes" in result
    assert "edges" in result


def test_bfs_subgraph_unknown_seed_raises(mock_storage, populated_storage):
    """Unknown seed ID raises ValueError."""
    with patch("mcp_server.server._get_storage", mock_storage):
        with pytest.raises(ValueError, match="Unknown seed ID"):
            _call_tool(["nonexistent:entity"])


def test_bfs_subgraph_node_filter_stub(mock_storage, populated_storage):
    """node_filter: non-matching nodes are stubs."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool(
            ["test:entity:1"],
            max_hops=2,
            node_filter={"entity_types": ["character"]},
        )

    for node in result["nodes"]:
        if node["entity_type"] == "character":
            assert "name" in node
        else:
            assert set(node.keys()) == {"id", "entity_type"}
