"""
Unit tests for extract_subgraph_bfs in graph_traversal.py.

Uses populated_storage fixture (entities 1,2,3 and relationships 1->2, 1->3, 2->3).
"""

from query.graph_traversal import extract_subgraph_bfs


def test_extract_subgraph_bfs_no_filters(populated_storage):
    """With no filters, all nodes and edges are full."""
    nodes, edges, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1"],
        hops=2,
    )
    assert not truncated
    assert len(nodes) == 3
    assert len(edges) == 3

    for node in nodes:
        assert "id" in node
        assert "entity_type" in node
        assert "name" in node
        assert "confidence" in node

    for edge in edges:
        assert "subject" in edge
        assert "predicate" in edge
        assert "object" in edge
        assert "confidence" in edge
        assert "source_documents" in edge


def test_extract_subgraph_bfs_node_filter(populated_storage):
    """node_filter.entity_types: matching nodes full, others stub."""
    nodes, _, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1"],
        hops=2,
        node_filter={"entity_types": ["character"]},
    )
    assert not truncated

    for node in nodes:
        if node["entity_type"] == "character":
            assert "name" in node
            assert "confidence" in node
        else:
            assert node["entity_type"] == "location"
            assert set(node.keys()) == {"id", "entity_type"}


def test_extract_subgraph_bfs_edge_filter(populated_storage):
    """edge_filter.predicates: matching edges full, others stub."""
    _, edges, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1"],
        hops=2,
        edge_filter={"predicates": ["co_occurs_with"]},
    )
    assert not truncated

    for edge in edges:
        if edge["predicate"] == "co_occurs_with":
            assert "confidence" in edge
            assert "source_documents" in edge
        else:
            assert edge["predicate"] == "appears_in"
            assert set(edge.keys()) == {"subject", "predicate", "object"}


def test_extract_subgraph_bfs_topology_min_confidence(populated_storage):
    """topology_filter.min_confidence: edges below threshold skipped."""
    nodes, edges, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1"],
        hops=2,
        min_confidence=0.80,
    )
    assert not truncated
    # Edge 2->3 has confidence 0.75, should be skipped
    assert len(nodes) == 3
    assert len(edges) == 2
    preds = {e["predicate"] for e in edges}
    assert "co_occurs_with" in preds
    assert "appears_in" in preds
    # 2->3 co_occurs_with (0.75) should not be traversed
    edge_triples = {(e["subject"], e["predicate"], e["object"]) for e in edges}
    assert ("test:entity:2", "co_occurs_with", "test:entity:3") not in edge_triples


def test_extract_subgraph_bfs_multi_seed(populated_storage):
    """Multi-seed returns union of neighborhoods."""
    nodes, _, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1", "test:entity:2"],
        hops=1,
    )
    assert not truncated
    node_ids = {n["id"] for n in nodes}
    assert node_ids == {"test:entity:1", "test:entity:2", "test:entity:3"}


def test_extract_subgraph_bfs_truncation(populated_storage):
    """When max_nodes hit, truncated is True."""
    nodes, _, truncated = extract_subgraph_bfs(
        populated_storage,
        seed_ids=["test:entity:1"],
        hops=2,
        max_nodes=2,
    )
    assert truncated
    assert len(nodes) == 2
