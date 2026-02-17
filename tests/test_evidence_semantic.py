"""Tests for semantic evidence validation (_evidence_contains_both_entities_semantic)."""

import pytest
from datetime import datetime, timezone

from kgschema.entity import BaseEntity, EntityStatus


class _MockEntity(BaseEntity):
    """Minimal entity for testing."""

    def get_entity_type(self) -> str:
        return "disease"


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator: same text -> same vector; different texts -> orthogonal."""

    class MockGen:
        def __init__(self):
            self._cache: dict[str, tuple[float, ...]] = {}
            self._dim = 4

        async def generate(self, text: str) -> tuple[float, ...]:
            key = text.strip().lower()
            if key not in self._cache:
                # Deterministic: hash text to a seed, then make a simple vector
                h = hash(key) % 10000
                # Same key -> same vector; different keys -> different vectors
                self._cache[key] = tuple(
                    float((h + i) % 7 - 3) for i in range(self._dim)
                )
            return self._cache[key]

    return MockGen()


@pytest.fixture
def entity_with_embedding():
    """Entity with pre-set embedding (so we don't need to generate)."""
    return _MockEntity(
        entity_id="test:subj",
        name="Subject",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        created_at=datetime.now(timezone.utc),
        embedding=(1.0, 0.0, 0.0, 0.0),
    )


@pytest.fixture
def entity_without_embedding():
    """Entity without embedding (will be generated via cache)."""
    return _MockEntity(
        entity_id="test:obj",
        name="Object",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        created_at=datetime.now(timezone.utc),
        embedding=None,
    )


@pytest.mark.asyncio
async def test_evidence_empty_rejected(
    mock_embedding_generator, entity_with_embedding, entity_without_embedding
):
    """Empty evidence is rejected without calling embedding generator."""
    from examples.medlit.pipeline.relationships import (
        _evidence_contains_both_entities_semantic,
    )

    evidence_cache: dict[str, tuple[float, ...]] = {}
    entity_name_cache: dict[str, tuple[float, ...]] = {}

    ok, reason, detail = await _evidence_contains_both_entities_semantic(
        "",
        entity_with_embedding,
        entity_without_embedding,
        mock_embedding_generator,
        0.5,
        evidence_cache,
        entity_name_cache,
    )
    assert ok is False
    assert reason == "evidence_empty"
    assert detail["subject_in_evidence"] is False
    assert detail["object_in_evidence"] is False
    assert len(evidence_cache) == 0


@pytest.mark.asyncio
async def test_evidence_semantic_returns_detail_shape(
    mock_embedding_generator, entity_with_embedding, entity_without_embedding
):
    """Semantic helper returns (ok, reason, detail) with expected keys."""
    from examples.medlit.pipeline.relationships import (
        _evidence_contains_both_entities_semantic,
    )

    evidence_cache: dict[str, tuple[float, ...]] = {}
    entity_name_cache: dict[str, tuple[float, ...]] = {}

    ok, reason, detail = await _evidence_contains_both_entities_semantic(
        "Some evidence text here.",
        entity_with_embedding,
        entity_without_embedding,
        mock_embedding_generator,
        0.0,  # permissive threshold so we get a pass or predictable fail
        evidence_cache,
        entity_name_cache,
    )
    assert "subject_in_evidence" in detail
    assert "object_in_evidence" in detail
    assert "subject_similarity" in detail
    assert "object_similarity" in detail
    assert isinstance(detail["subject_similarity"], (int, float))
    assert isinstance(detail["object_similarity"], (int, float))


@pytest.mark.asyncio
async def test_evidence_embedding_cached(
    mock_embedding_generator, entity_with_embedding, entity_without_embedding
):
    """Evidence string is cached so same evidence does not call generate twice."""
    from examples.medlit.pipeline.relationships import (
        _evidence_contains_both_entities_semantic,
    )

    evidence_cache: dict[str, tuple[float, ...]] = {}
    entity_name_cache: dict[str, tuple[float, ...]] = {}
    evidence = "Same evidence text twice."

    await _evidence_contains_both_entities_semantic(
        evidence,
        entity_with_embedding,
        entity_without_embedding,
        mock_embedding_generator,
        0.0,
        evidence_cache,
        entity_name_cache,
    )
    assert evidence in evidence_cache
    first_emb = evidence_cache[evidence]

    # Call again with same evidence
    evidence_cache2 = dict(evidence_cache)
    entity_name_cache2 = dict(entity_name_cache)
    await _evidence_contains_both_entities_semantic(
        evidence,
        entity_with_embedding,
        entity_without_embedding,
        mock_embedding_generator,
        0.0,
        evidence_cache2,
        entity_name_cache2,
    )
    assert evidence_cache2[evidence] == first_emb
