from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Dict, Any, Optional
import uuid

from kgraph.query.bundle import BundleManifestV1, EntityRow, RelationshipRow, BundleFile
from kgraph.storage.interfaces import EntityStorageInterface, RelationshipStorageInterface


class GraphBundleExporter(Protocol):
    async def export_graph_bundle(
        self,
        entity_storage: EntityStorageInterface,
        relationship_storage: RelationshipStorageInterface,
        bundle_path: Path,
        domain: str,
        label: Optional[str] = None,
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

        manifest = BundleManifestV1(
            bundle_id=bundle_id,
            domain=domain,
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(),
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
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
        description=description,
    )
