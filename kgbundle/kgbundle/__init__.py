"""
Knowledge Graph Bundle Models

Lightweight Pydantic models defining the contract between bundle producers
(kgraph ingestion framework) and consumers (kgserver query server).

This package has minimal dependencies (only pydantic) and is designed to be
importable by both sides without pulling in heavy ML or web frameworks.

Example:
    # In kgraph (producer) - export bundle
    from kgbundle import BundleManifestV1, EntityRow, RelationshipRow

    entity = EntityRow(
        entity_id="char:123",
        entity_type="character",
        name="Sherlock Holmes",
        status="canonical",
        usage_count=1,
        created_at="2024-01-15T10:30:00Z",
        source="sherlock:curated"
    )

    # In kgserver (consumer) - load bundle
    from kgbundle import BundleManifestV1

    with open("manifest.json") as f:
        manifest = BundleManifestV1.model_validate_json(f.read())
"""

from .models import (
    BundleFile,
    BundleManifestV1,
    DocAssetRow,
    EntityRow,
    RelationshipRow,
)

__all__ = [
    "EntityRow",
    "RelationshipRow",
    "BundleFile",
    "DocAssetRow",
    "BundleManifestV1",
]

__version__ = "0.1.0"
