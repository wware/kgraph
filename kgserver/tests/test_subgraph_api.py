"""
Tests for query/routers/subgraph_api.py REST API endpoints.

Covers GET (legacy) and POST (BFS JSON body) subgraph endpoints.
"""

# pylint: disable=protected-access
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from query.routers import subgraph_api


@pytest.fixture
def app():
    """Create FastAPI app with subgraph router."""
    app = FastAPI()
    app.include_router(subgraph_api.router)
    return app


@pytest.fixture
def file_storage(tmp_path, sample_entities, sample_relationships):
    """Create SQLite storage for testing."""
    from storage.backends.sqlite import SQLiteStorage

    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(str(db_path), check_same_thread=False)
    for entity in sample_entities:
        storage._session.add(entity)
    for rel in sample_relationships:
        storage._session.add(rel)
    storage._session.commit()
    try:
        yield storage
    finally:
        storage.close()


@pytest.fixture
def client(app, file_storage):
    """Create test client with storage dependency override."""
    from query.storage_factory import get_storage

    def override_get_storage():
        yield file_storage

    app.dependency_overrides[get_storage] = override_get_storage
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestPostBfsSubgraph:
    """Test POST /api/v1/subgraph (BFS JSON body)."""

    def test_post_valid_returns_200(self, client):
        """POST with valid JSON body returns 200 and BfsSubgraphResponse shape."""
        response = client.post(
            "/api/v1/subgraph",
            json={
                "seeds": ["test:entity:1"],
                "max_hops": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "seeds" in data
        assert data["seeds"] == ["test:entity:1"]
        assert "max_hops" in data
        assert "node_count" in data
        assert "edge_count" in data
        assert "truncated" in data
        assert "nodes" in data
        assert "edges" in data

    def test_post_unknown_seed_returns_400(self, client):
        """POST with unknown seed ID returns 400 with error identifying unknown ID(s)."""
        response = client.post(
            "/api/v1/subgraph",
            json={
                "seeds": ["test:entity:1", "nonexistent:entity"],
                "max_hops": 2,
            },
        )
        assert response.status_code == 400
        assert "nonexistent:entity" in response.json()["detail"]


class TestGetSubgraph:
    """Test GET /api/v1/subgraph (legacy query params)."""

    def test_get_returns_legacy_format(self, client):
        """GET without JSON returns legacy format (entities, relationships, query)."""
        response = client.get(
            "/api/v1/subgraph",
            params={"entity": "test:entity:1", "hops": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relationships" in data
        assert "query" in data
        assert "seeds" in data["query"]
