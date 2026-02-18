import shutil
import mimetypes
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import uuid

from kgbundle import BundleFile, BundleManifestV1, DocAssetRow, EntityRow, RelationshipRow
from kgschema.storage import EntityStorageInterface, RelationshipStorageInterface

from kgraph.provenance import ProvenanceAccumulator


def get_git_hash() -> Optional[str]:
    """Gets the current git commit hash in short format.

    This is used to version-stamp exported bundles, providing a precise
    reference to the codebase state at the time of export.

    Returns:
        The short git commit hash (e.g., "6b50d25") as a string, or `None`
        if the git command fails (e.g., not in a git repository).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _collect_doc_assets(docs_source: Path, bundle_path: Path) -> List[DocAssetRow]:
    """Copies documentation assets from a source directory into the bundle.

    This function recursively walks the `docs_source` directory, copies each
    file to a `docs/` subdirectory within the `bundle_path`, and generates
    a `DocAssetRow` for each copied file to be included in the bundle's
    `doc_assets.jsonl`.

    Note: These are human-readable documentation files (markdown, images, etc.),
    NOT source documents (papers, articles) used for entity extraction.

    Args:
        docs_source: The source directory containing the documentation assets
                     (e.g., Markdown files).
        bundle_path: The root directory of the bundle being created.

    Returns:
        A list of `DocAssetRow` objects, one for each file copied.
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
            asset_rows.append(DocAssetRow(path=bundle_relative_path, content_type=content_type))

    return asset_rows


def _entity_provenance_summary(accumulator: ProvenanceAccumulator) -> Dict[str, Dict[str, Any]]:
    """Build per-entity provenance summary from accumulator mentions (first_seen, total_mentions, supporting_documents)."""
    by_entity: Dict[str, List[Any]] = {}
    for m in accumulator.mentions:
        by_entity.setdefault(m.entity_id, []).append(m)
    summary: Dict[str, Dict[str, Any]] = {}
    for entity_id, mention_list in by_entity.items():
        sorted_mentions = sorted(mention_list, key=lambda x: x.created_at)
        first = sorted_mentions[0]
        doc_ids = list({m.document_id for m in mention_list})
        summary[entity_id] = {
            "first_seen_document": first.document_id,
            "first_seen_section": first.section,
            "total_mentions": len(mention_list),
            "supporting_documents": doc_ids,
        }
    return summary


def _relationship_evidence_summary(accumulator: ProvenanceAccumulator) -> Dict[str, Dict[str, Any]]:
    """Build per-relationship evidence summary from accumulator evidence."""
    by_key: Dict[str, List[Any]] = {}
    for e in accumulator.evidence:
        by_key.setdefault(e.relationship_key, []).append(e)
    summary: Dict[str, Dict[str, Any]] = {}
    for rel_key, ev_list in by_key.items():
        if not ev_list:
            continue
        best = max(ev_list, key=lambda x: x.confidence)
        avg_conf = sum(x.confidence for x in ev_list) / len(ev_list)
        summary[rel_key] = {
            "evidence_count": len(ev_list),
            "strongest_evidence_quote": best.text_span,
            "evidence_confidence_avg": round(avg_conf, 4),
        }
    return summary


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
        provenance_accumulator: Optional[ProvenanceAccumulator] = None,
    ) -> None: ...  # type: ignore


class JsonlGraphBundleExporter:
    async def export_graph_bundle(  # pylint: disable=too-many-statements
        self,
        entity_storage: EntityStorageInterface,
        relationship_storage: RelationshipStorageInterface,
        bundle_path: Path,
        domain: str,
        label: Optional[str] = None,
        docs: Optional[Path] = None,
        description: Optional[str] = None,
        provenance_accumulator: Optional[ProvenanceAccumulator] = None,
    ) -> None:
        """Exports the graph content into a standardized JSONL bundle format.

        This method orchestrates the entire bundle creation process. It reads
        all entities and relationships from the provided storage interfaces,
        serializes them into JSONL files (`entities.jsonl`, `relationships.jsonl`),
        copies any associated document assets, and generates a `manifest.json`
        file that describes the bundle's contents.

        Args:
            entity_storage: The storage backend containing the entities.
            relationship_storage: The storage backend containing the relationships.
            bundle_path: The root directory where the bundle will be written.
            domain: The knowledge domain of the graph (e.g., "medlit").
            label: An optional human-readable label for the bundle.
            docs: An optional path to a directory of document assets to include.
            description: An optional description to include in the manifest.
        """
        bundle_path.mkdir(parents=True, exist_ok=True)

        entities_file = bundle_path / "entities.jsonl"
        relationships_file = bundle_path / "relationships.jsonl"
        manifest_file = bundle_path / "manifest.json"

        bundle_id = str(uuid.uuid4())

        entity_provenance = _entity_provenance_summary(provenance_accumulator) if provenance_accumulator else {}
        rel_evidence = _relationship_evidence_summary(provenance_accumulator) if provenance_accumulator else {}

        entity_count = 0
        with open(entities_file, "w") as f_entities:
            for entity in await entity_storage.list_all():
                # Extract canonical_url from metadata if present
                metadata = entity.metadata.copy() if entity.metadata else {}
                canonical_url = metadata.pop("canonical_url", None) or metadata.pop("canonicalUrl", None)
                prov = entity_provenance.get(entity.entity_id, {})
                entity_row = EntityRow(
                    entity_id=entity.entity_id,
                    entity_type=entity.get_entity_type(),
                    name=entity.name,
                    status=entity.status.value,
                    confidence=entity.confidence,
                    usage_count=entity.usage_count,
                    created_at=entity.created_at.isoformat(),
                    source=entity.source,
                    canonical_url=canonical_url,
                    properties=metadata,  # Remaining metadata without canonical_url
                    first_seen_document=prov.get("first_seen_document"),
                    first_seen_section=prov.get("first_seen_section"),
                    total_mentions=prov.get("total_mentions", 0),
                    supporting_documents=prov.get("supporting_documents", []),
                )
                f_entities.write(entity_row.model_dump_json() + "\n")
                entity_count += 1

        relationship_count = 0
        with open(relationships_file, "w") as f_relationships:
            for relationship in await relationship_storage.list_all():
                rel_key = f"{relationship.subject_id}:{relationship.predicate}:{relationship.object_id}"
                ev = rel_evidence.get(rel_key, {})
                relationship_row = RelationshipRow(
                    subject_id=relationship.subject_id,
                    object_id=relationship.object_id,
                    predicate=relationship.predicate,
                    confidence=relationship.confidence,
                    source_documents=list(relationship.source_documents),
                    created_at=relationship.created_at.isoformat(),
                    properties=relationship.metadata,
                    evidence_count=ev.get("evidence_count", 0),
                    strongest_evidence_quote=ev.get("strongest_evidence_quote"),
                    evidence_confidence_avg=ev.get("evidence_confidence_avg"),
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

        # Add git hash for version tracking
        kgraph_version = get_git_hash()
        if kgraph_version:
            manifest_metadata["kgraph_version"] = kgraph_version

        # Handle documentation assets (NOT source documents - these are for MkDocs)
        doc_assets_file = None
        if docs is not None:
            doc_assets_file = bundle_path / "doc_assets.jsonl"
            asset_rows = _collect_doc_assets(docs, bundle_path)
            if asset_rows:
                with open(doc_assets_file, "w") as f_docs:
                    for asset_row in asset_rows:
                        f_docs.write(asset_row.model_dump_json() + "\n")

        # Write provenance files when accumulator has data
        wrote_mentions = False
        wrote_evidence = False
        if provenance_accumulator and provenance_accumulator.mention_count() > 0:
            mentions_file = bundle_path / "mentions.jsonl"
            with open(mentions_file, "w") as f_mentions:
                for mention_row in provenance_accumulator.mentions:
                    f_mentions.write(mention_row.model_dump_json() + "\n")
            wrote_mentions = True
        if provenance_accumulator and provenance_accumulator.evidence_count() > 0:
            evidence_file = bundle_path / "evidence.jsonl"
            with open(evidence_file, "w") as f_evidence:
                for evidence_row in provenance_accumulator.evidence:
                    f_evidence.write(evidence_row.model_dump_json() + "\n")
            wrote_evidence = True

        manifest = BundleManifestV1(
            bundle_version="v1",
            bundle_id=bundle_id,
            domain=domain,
            label=label,
            created_at=datetime.now(timezone.utc).isoformat(),
            entities=BundleFile(path="entities.jsonl", format="jsonl"),
            relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
            doc_assets=BundleFile(path="doc_assets.jsonl", format="jsonl") if doc_assets_file and doc_assets_file.exists() else None,
            mentions=BundleFile(path="mentions.jsonl", format="jsonl") if wrote_mentions else None,
            evidence=BundleFile(path="evidence.jsonl", format="jsonl") if wrote_evidence else None,
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
    provenance_accumulator: Optional[ProvenanceAccumulator] = None,
) -> None:
    """Writes a knowledge graph bundle to disk using the default exporter.

    This function is a convenient wrapper around the `JsonlGraphBundleExporter`
    that serializes entities and relationships into JSONL files and creates a
    bundle manifest.

    Args:
        entity_storage: The storage backend for retrieving entities.
        relationship_storage: The storage backend for retrieving relationships.
        bundle_path: The root directory where the bundle will be created.
        domain: The knowledge domain identifier for the graph (e.g., "medlit").
        label: An optional human-readable label for the bundle.
        docs: An optional path to a directory of document assets to copy
              into the bundle.
        description: An optional description to be included in the bundle's
                     manifest metadata.
    """
    await default_exporter.export_graph_bundle(
        entity_storage=entity_storage,
        relationship_storage=relationship_storage,
        bundle_path=bundle_path,
        domain=domain,
        label=label,
        docs=docs,
        description=description,
        provenance_accumulator=provenance_accumulator,
    )
