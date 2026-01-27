"""
Tests for query/bundle_loader.py bundle loading logic.

Tests cover:
- _find_manifest() function
- Bundle loading from directory
- Bundle loading from ZIP
- Bundle-specific graphql_examples.yml override
- Error handling
"""

import pytest
import json
import zipfile
from datetime import datetime
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from query.bundle_loader import _load_from_directory, _load_from_zip
from query.graphql_examples import load_examples, get_examples, get_default_query
from storage.backends.sqlite import SQLiteStorage


@pytest.fixture
def sample_manifest_data():
    """Create sample manifest data."""
    return {
        "bundle_id": "test-bundle-123",
        "domain": "test",
        "created_at": datetime.now().isoformat(),
        "bundle_version": "v1",
        "entities": {"path": "entities.jsonl", "format": "jsonl"},
        "relationships": {"path": "relationships.jsonl", "format": "jsonl"},
    }


@pytest.fixture
def bundle_directory(sample_manifest_data, tmp_path):
    """Create a temporary bundle directory with manifest and data files."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    # Create manifest.json
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest_data))

    # Create entities.jsonl
    entities_file = bundle_dir / "entities.jsonl"
    entities_file.write_text(json.dumps({"entity_id": "test:1", "entity_type": "test", "name": "Test 1"}) + "\n")

    # Create relationships.jsonl
    relationships_file = bundle_dir / "relationships.jsonl"
    relationships_file.write_text(
        json.dumps(
            {
                "subject_id": "test:1",
                "predicate": "test",
                "object_id": "test:2",
            }
        )
        + "\n"
    )

    return bundle_dir


@pytest.fixture
def bundle_zip(bundle_directory, tmp_path):
    """Create a ZIP file from bundle directory."""
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in bundle_directory.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(bundle_directory)
                zf.write(file_path, arcname)
    return zip_path


@pytest.fixture
def test_engine():
    """Create a test SQLAlchemy engine."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


class TestLoadFromDirectory:
    """Test _load_from_directory() function."""

    def test_load_from_directory_success(self, bundle_directory, test_engine):
        """Test successfully loading bundle from directory."""
        db_url = "sqlite:///:memory:"

        # Should not raise
        _load_from_directory(test_engine, db_url, bundle_directory)

        # Verify data was loaded
        storage = SQLiteStorage(":memory:")
        # Note: SQLiteStorage creates its own engine, so we can't easily verify
        # the data was loaded. This test mainly verifies no exceptions are raised.
        storage.close()

    def test_load_from_directory_no_manifest(self, tmp_path, test_engine):
        """Test loading from directory without manifest."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        db_url = "sqlite:///:memory:"
        # Should handle gracefully (prints error but doesn't raise)
        _load_from_directory(test_engine, db_url, empty_dir)


class TestLoadFromZip:
    """Test _load_from_zip() function."""

    def test_load_from_zip_success(self, bundle_zip, test_engine):
        """Test successfully loading bundle from ZIP."""
        db_url = "sqlite:///:memory:"

        # Should not raise
        _load_from_zip(test_engine, db_url, bundle_zip)

    def test_load_from_zip_no_manifest(self, tmp_path, test_engine):
        """Test loading ZIP without manifest."""
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            # Create empty ZIP
            pass

        db_url = "sqlite:///:memory:"
        # Should handle gracefully (prints error but doesn't raise)
        _load_from_zip(test_engine, db_url, zip_path)


class TestBundleGraphqlExamples:
    """Test that a bundle's graphql_examples.yml replaces the default examples."""

    def test_bundle_examples_override(self, bundle_directory, test_engine):
        """When a bundle contains graphql_examples.yml, those examples
        should replace the built-in defaults after loading."""
        # Reset to built-in defaults first
        load_examples()
        default_keys = set(get_examples().keys())
        assert "Search Entities" in default_keys

        # Write a bundle-specific graphql_examples.yml
        bundle_examples = {
            "Custom Query": "query { entities(limit: 1) { items { name } } }",
            "Search Entities": "query { entities(limit: 99) { items { entityId } } }",
        }
        import yaml

        (bundle_directory / "graphql_examples.yml").write_text(yaml.dump(bundle_examples))

        db_url = "sqlite:///:memory:"
        _load_from_directory(test_engine, db_url, bundle_directory)

        # The examples should now match the bundle file
        current = get_examples()
        assert set(current.keys()) == {"Custom Query", "Search Entities"}
        assert "limit: 99" in current["Search Entities"]
        assert get_default_query() == current["Search Entities"]

        # Restore built-in defaults so other tests are unaffected
        load_examples()

    def test_no_bundle_examples_keeps_defaults(self, bundle_directory, test_engine):
        """When a bundle does NOT contain graphql_examples.yml, the built-in
        defaults should remain unchanged."""
        load_examples()
        expected_keys = set(get_examples().keys())

        db_url = "sqlite:///:memory:"
        _load_from_directory(test_engine, db_url, bundle_directory)

        assert set(get_examples().keys()) == expected_keys
