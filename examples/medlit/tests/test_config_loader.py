"""Tests for config_loader module."""

import tempfile
from pathlib import Path


from examples.medlit.pipeline.config_loader import (
    get_schema_version,
    load_domain_instructions,
    load_entity_types,
    load_predicates,
)


def test_load_entity_types() -> None:
    """Load entity_types.yaml returns types dict."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    types = load_entity_types(config_dir)
    assert "disease" in types
    assert types["disease"]["bundle_class"] == "Disease"
    assert "description" in types["disease"]


def test_load_predicates() -> None:
    """Load predicates.yaml returns predicates dict."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    preds = load_predicates(config_dir)
    assert "TREATS" in preds
    assert "SAME_AS" in preds
    assert "description" in preds["TREATS"]


def test_load_domain_instructions() -> None:
    """Load domain_instructions.md returns string."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    content = load_domain_instructions(config_dir)
    assert "Entity type classification" in content
    assert "Biomarker" in content


def test_get_schema_version_deterministic() -> None:
    """Same config produces same schema version."""
    config_dir = Path(__file__).resolve().parents[1] / "config"
    v1 = get_schema_version(config_dir)
    v2 = get_schema_version(config_dir)
    assert v1 == v2
    assert len(v1) == 8
    assert all(c in "0123456789abcdef" for c in v1)


def test_get_schema_version_changes_with_config() -> None:
    """Schema version changes when config changes."""
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp)
        (config_dir / "entity_types.yaml").write_text("types:\n  x: {}\n")
        (config_dir / "predicates.yaml").write_text("predicates:\n  X: {}\n")
        (config_dir / "domain_instructions.md").write_text("# Foo\n")
        v1 = get_schema_version(config_dir)
        (config_dir / "domain_instructions.md").write_text("# Bar\n")
        v2 = get_schema_version(config_dir)
        assert v1 != v2
