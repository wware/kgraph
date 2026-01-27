# query/bundle_loader.py
"""
Bundle loading utilities for the KG server.
Handles loading bundles from directories or ZIP files at startup.
"""

import json
import os
import sys
import logging
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from sqlmodel import Session, SQLModel, delete

from kgbundle import BundleManifestV1
from storage.interfaces import StorageInterface
from storage.backends.sqlite import SQLiteStorage
from storage.backends.postgres import PostgresStorage

# from storage.backends.sqlite import SQLiteStorage
from storage.models.bundle import Bundle
from storage.models.entity import Entity
from storage.models.relationship import Relationship

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
FORMAT = "%(levelname)s:     %(asctime)s - %(pathname)s:%(lineno)d - %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_bundle_at_startup(engine, db_url: str) -> None:
    """
    Load a bundle at server startup if BUNDLE_PATH is set.

    Environment variables:
        BUNDLE_PATH: Path to a bundle directory or ZIP file
    """
    bundle_path_str = os.getenv("BUNDLE_PATH")
    logger.info("bundle_path=%s", bundle_path_str)
    if not bundle_path_str:
        print("BUNDLE_PATH not set, skipping bundle load.")
        return

    bundle_path = Path(bundle_path_str)
    if not bundle_path.exists():
        print(f"Warning: BUNDLE_PATH '{bundle_path}' does not exist, skipping bundle load.")
        return

    logger.info("Loading bundle from: %s", bundle_path)

    # Ensure tables exist
    SQLModel.metadata.create_all(engine)

    # Handle ZIP file vs directory
    if bundle_path.suffix == ".zip":
        _load_from_zip(engine, db_url, bundle_path)
    elif bundle_path.is_dir():
        _load_from_directory(engine, db_url, bundle_path)
    else:
        print(f"Warning: BUNDLE_PATH '{bundle_path}' is not a directory or ZIP file.")


def _load_from_zip(engine, db_url: str, zip_path: Path) -> None:
    """Extract and load a bundle from a ZIP file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find the manifest - could be at root or in a subdirectory
        tmpdir_path = Path(tmpdir)
        manifest_path = _find_manifest(tmpdir_path)
        if not manifest_path:
            print(f"Error: No manifest.json found in ZIP file {zip_path}")
            return

        bundle_dir = manifest_path.parent
        _do_load(engine, db_url, bundle_dir, manifest_path)


def _load_from_directory(engine, db_url: str, bundle_dir: Path) -> None:
    """Load a bundle from a directory."""
    manifest_path = _find_manifest(bundle_dir)
    if not manifest_path:
        print(f"Error: No manifest.json found in {bundle_dir}")
        return

    bundle_dir = manifest_path.parent
    _do_load(engine, db_url, bundle_dir, manifest_path)


def _find_manifest(search_dir: Path) -> Path | None:
    """Find manifest.json in a directory (possibly in a subdirectory)."""
    # Check directly in the directory
    direct = search_dir / "manifest.json"
    if direct.exists():
        return direct

    # Check one level of subdirectories
    for subdir in search_dir.iterdir():
        if subdir.is_dir():
            manifest = subdir / "manifest.json"
            if manifest.exists():
                return manifest

    return None


def _load_document_assets(bundle_dir: Path, manifest: BundleManifestV1) -> None:
    """Load document assets from documents.jsonl into /app/docs.

    Reads the documents.jsonl file (if present) and copies all listed assets
    to /app/docs, preserving directory structure. Special handling for mkdocs.yml
    which is moved to /app/mkdocs.yml.
    """
    if not manifest.documents:
        return

    documents_file = bundle_dir / manifest.documents.path
    if not documents_file.exists():
        logger.warning("Documents file %s not found, skipping document asset loading", documents_file)
        return

    app_docs = Path("/app/docs")
    app_docs.mkdir(parents=True, exist_ok=True)

    asset_count = 0
    with open(documents_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                asset = json.loads(line)
                asset_path = asset.get("path")
                if not asset_path:
                    logger.warning("Skipping asset entry without path: %s", line)
                    continue

                # Source file in bundle
                source_file = bundle_dir / asset_path
                if not source_file.exists():
                    logger.warning("Asset file not found: %s", source_file)
                    continue

                # Destination in /app/docs (strip "docs/" prefix if present)
                if asset_path.startswith("docs/"):
                    rel_path = asset_path[5:]  # Remove "docs/" prefix
                else:
                    rel_path = asset_path

                # Special handling for mkdocs.yml - move to /app root
                if rel_path == "mkdocs.yml" or asset_path.endswith("/mkdocs.yml"):
                    dest_path = Path("/app/mkdocs.yml")
                    shutil.copy2(source_file, dest_path)
                    logger.info("Copied %s to %s", source_file, dest_path)
                    asset_count += 1
                    continue

                # Regular file - copy to /app/docs preserving structure
                dest_path = app_docs / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, dest_path)
                asset_count += 1

            except json.JSONDecodeError as e:
                logger.warning("Failed to parse asset entry: %s... Error: %s", line[:100], e)
                continue

    if asset_count > 0:
        logger.info("Loaded %s document assets to /app/docs", asset_count)

        # Build mkdocs if mkdocs.yml exists
        mkdocs_yml = Path("/app/mkdocs.yml")
        if mkdocs_yml.exists():
            logger.info("Building MkDocs documentation...")
            result = subprocess.run(["uv", "run", "mkdocs", "build"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info("MkDocs build completed successfully")
            else:
                logger.warning("MkDocs build failed: %s", result.stderr)


def _do_load(engine, db_url: str, bundle_dir: Path, manifest_path: Path) -> None:
    """Actually load the bundle into storage."""
    # Parse manifest
    manifest = BundleManifestV1.model_validate_json(manifest_path.read_text())
    logger.info("Loaded manifest for bundle: %s (domain: %s)", manifest.bundle_id, manifest.domain)

    # Load document assets if present
    _load_document_assets(bundle_dir, manifest)

    # Use bundle-specific GraphQL examples if provided
    bundle_examples = bundle_dir / "graphql_examples.yml"
    if bundle_examples.exists():
        from query.graphql_examples import load_examples

        load_examples(bundle_examples)
        logger.info("Loaded bundle-specific GraphQL examples from %s", bundle_examples)

    with Session(engine) as session:
        storage: StorageInterface
        if db_url.startswith("postgres"):
            storage = PostgresStorage(session)
        else:
            db_path = db_url.replace("sqlite:///", "")
            if not db_path:
                db_path = ":memory:" if db_url == "sqlite:///:memory:" else "./test.db"
            storage = SQLiteStorage(db_path)
        force = os.getenv("BUNDLE_FORCE_RELOAD", "").lower() in {"1", "true", "yes"}
        if storage.is_bundle_loaded(manifest.bundle_id) and not force:
            logger.info("Bundle %s already loaded. Skipping.", manifest.bundle_id)
            return

        if force:
            logger.info("Force reload enabled: clearing Bundle, Relationship, and Entity tables...")
            session.exec(delete(Bundle))
            session.exec(delete(Relationship))
            session.exec(delete(Entity))
            session.commit()

        storage.load_bundle(manifest, str(bundle_dir))
        logger.info("Bundle %s loaded successfully.", manifest.bundle_id)
