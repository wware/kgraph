import shutil
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Dict, Any, Optional, List
import uuid

from kgraph.query.bundle import BundleManifestV1, EntityRow, RelationshipRow, BundleFile, DocumentAssetRow
from kgraph.storage.interfaces import EntityStorageInterface, RelationshipStorageInterface


def _collect_document_assets(docs_source: Path, bundle_path: Path) -> List[DocumentAssetRow]:
    """Collect document assets from a source directory and copy them to the bundle.

    Args:
        docs_source: Source directory containing document assets
        bundle_path: Bundle root directory where assets will be copied

    Returns:
        List of DocumentAssetRow entries for the documents.jsonl file
    """
    if not docs_source.exists() or not docs_source.is_dir():
        return []

    asset_rows = []
    docs_dir = bundle_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Walk through the source directory and copy files
    for source_file in docs_source.rglob("*"):
        if source_file.is_file():
            # Calculate relative path from source root
            rel_path = source_file.relative_to(docs_source)
            # Destination path in bundle
            dest_path = docs_dir / rel_path
            # Ensure parent directories exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy the file
            shutil.copy2(source_file, dest_path)

            # Determine content type
            content_type, _ = mimetypes.guess_type(str(source_file))
            if content_type is None:
                # Default based on extension
                if source_file.suffix == ".md":
                    content_type = "text/markdown"
                elif source_file.suffix in (".yml", ".yaml"):
                    content_type = "text/yaml"
                else:
                    content_type = "application/octet-stream"

            # Create asset row with path relative to bundle root
            bundle_relative_path = f"docs/{rel_path.as_posix()}"
            asset_rows.append(DocumentAssetRow(path=bundle_relative_path, content_type=content_type))

    return asset_rows


class GraphBundleExporter(Protocol):
    async def export_graph_bundle(
        self,
        entity_storage: EntityStorageInterface,
        relationship_storage: RelationshipStorageInterface,
        bundle_path: Path,
        domain: str,
        label: Optional[str] = None,
        docs: Optional[Path] = None,
        description: Optional[str] = None,
    ) -> None: ...  # type: ignore


class JsonlGraphBundleExporter:
    async def export_graph_bundle(
        self,
        entity_storage: EntityStorageInterface,
        relationship_storage: RelationshipStorageInterface,
        bundle_path: Path,
        domain: str,
        label: Optional[str] = None,
        docs: Optional[Path] = None,
        description: Optional[str] = None,
    ) -> None:
        bundle_path.mkdir(parents=True, exist_ok=True)

        entities_file = bundle_path / "entities.jsonl"
        relationships_file = bundle_path / "relationships.jsonl"
        manifest_file = bundle_path / "manifest.json"

        bundle_id = str(uuid.uuid4())

        entity_count = 0
        with open(entities_file, "w") as f_entities:
            for entity in await entity_storage.list_all():
                entity_row = EntityRow(
                    entity_id=entity.entity_id,
                    entity_type=entity.get_entity_type(),
                    name=entity.name,
                    status=entity.status.value,
                    confidence=entity.confidence,
                    usage_count=entity.usage_count,
                    created_at=entity.created_at.isoformat(),
                    source=entity.source,
                    properties=entity.metadata,
                )
                f_entities.write(entity_row.model_dump_json() + "\n")
                entity_count += 1

        relationship_count = 0
        with open(relationships_file, "w") as f_relationships:
            for relationship in await relationship_storage.list_all():
                relationship_row = RelationshipRow(
                    subject_id=relationship.subject_id,
                    object_id=relationship.object_id,
                    predicate=relationship.predicate,
                    confidence=relationship.confidence,
                    source_documents=list(relationship.source_documents),
                    created_at=relationship.created_at.isoformat(),
                    properties=relationship.metadata,
                )
                f_relationships.write(relationship_row.model_dump_json() + "\n")
                relationship_count += 1

        # Build manifest metadata
        manifest_metadata: Dict[str, Any] = {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
        }
        if description:
            manifest_metadata["description"] = description

        # Handle document assets
        documents_file = None
        if docs is not None:
            documents_file = bundle_path / "documents.jsonl"
            asset_rows = _collect_document_assets(docs, bundle_path)
            if asset_rows:
                with open(documents_file, "w") as f_docs:
                    for asset_row in asset_rows:
                        f_docs.write(asset_row.model_dump_json() + "\n")

        manifest = BundleManifestV1(
            bundle_id=bundle_id,
            domain=domain,
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(),
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
            documents=BundleFile(path="documents.jsonl", format="jsonl") if documents_file and documents_file.exists() else None,
            metadata=manifest_metadata,
        )
        with open(manifest_file, "w") as f_manifest:
            f_manifest.write(manifest.model_dump_json(indent=2))

        print(f"Bundle '{label or domain}' exported to {bundle_path}")
        print(f"  Bundle ID: {bundle_id}")
        print(f"  Domain: {domain}")
        print(f"  Entities: {entity_count}")
        print(f"  Relationships: {relationship_count}")


# Default exporter instance
default_exporter = JsonlGraphBundleExporter()


async def write_bundle(
    entity_storage: EntityStorageInterface,
    relationship_storage: RelationshipStorageInterface,
    bundle_path: Path,
    domain: str,
    label: Optional[str] = None,
    docs: Optional[Path] = None,
    description: Optional[str] = None,
) -> None:
    """Write a graph bundle to disk in JSONL format.

    Args:
        entity_storage: Storage interface for entities
        relationship_storage: Storage interface for relationships
        bundle_path: Directory path for the bundle output
        domain: Knowledge domain identifier (e.g., "sherlock", "medical")
        label: Optional human-readable bundle label
        description: Optional description for bundle metadata
    """
    await default_exporter.export_graph_bundle(
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        bundle_path=bundle_path,
        domain=domain,
        label=label,
        docs=docs,
        description=description,
    )
