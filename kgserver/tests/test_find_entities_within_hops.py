"""
Tests for the MCP find_entities_within_hops tool.

Uses the same populated_storage fixture as other MCP tests (entities 1,2,3
and relationships 1->2, 1->3, 2->3). Verifies BFS result structure and
hop_distance consistency.
"""

import pytest
from unittest.mock import patch

from mcp_server.server import find_entities_within_hops


@pytest.fixture
def mock_storage(populated_storage):
    """Provide populated storage to the MCP tool via _get_storage."""
    from contextlib import contextmanager

    @contextmanager
    def _get_storage():
        yield populated_storage

    return _get_storage


def _call_tool(start_id: str, max_hops: int = 3, entity_type=None):
    """Call the underlying MCP tool function."""
    return find_entities_within_hops.fn(
        start_id=start_id,
        max_hops=max_hops,
        entity_type=entity_type,
    )


def test_find_entities_within_hops_structure(mock_storage):
    """Result has start_id, results_by_hop dict, and hop_distance matches key."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("test:entity:1", max_hops=2)

    assert "start_id" in result
    assert result["start_id"] == "test:entity:1"
    assert "results_by_hop" in result
    assert isinstance(result["results_by_hop"], dict)
    assert "total_entities_found" in result

    for hop_str, entities in result["results_by_hop"].items():
        hop = int(hop_str)
        for ent in entities:
            assert ent["hop_distance"] == hop


def test_find_entities_within_hops_from_entity_1(mock_storage):
    """From test:entity:1, one hop gives entity 2 and 3 (via edges 1->2, 1->3)."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("test:entity:1", max_hops=2)

    assert result["total_entities_found"] == 2
    assert "1" in result["results_by_hop"]
    hop1 = result["results_by_hop"]["1"]
    ids = {e["entity_id"] for e in hop1}
    assert ids == {"test:entity:2", "test:entity:3"}


def test_find_entities_within_hops_entity_type_filter(mock_storage):
    """Filter by entity_type returns only matching entities."""
    with patch("mcp_server.server._get_storage", mock_storage):
        result = _call_tool("test:entity:1", max_hops=2, entity_type="location")

    # Only test:entity:3 is type "location"
    assert result["total_entities_found"] == 1
    assert result["entity_type_filter"] == "location"
    assert result["results_by_hop"]["1"][0]["entity_id"] == "test:entity:3"
    assert result["results_by_hop"]["1"][0]["entity_type"] == "location"
