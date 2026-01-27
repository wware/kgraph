"""
PostgreSQL implementation of the storage interface.
"""

import json
from datetime import datetime
from typing import Optional, Sequence
from sqlalchemy import func
from sqlmodel import Session, select
from storage.interfaces import StorageInterface
from storage.models import Bundle, Entity, Relationship
from kgbundle import BundleManifestV1, EntityRow, RelationshipRow
from pydantic import ValidationError


class PostgresStorage(StorageInterface):
    """
    PostgreSQL implementation of the storage interface.
    """

    def __init__(self, session: Session):
        self._session = session

    def load_bundle(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None:
        """
        Load a data bundle into the storage.
        This is an idempotent operation. If the bundle is already loaded, it will do nothing.
        """
        if self.is_bundle_loaded(bundle_manifest.bundle_id):
            print(f"Bundle {bundle_manifest.bundle_id} already loaded. Skipping.")
            return

        print(f"Loading bundle {bundle_manifest.bundle_id} from {bundle_path}")

        # Load entities
        if bundle_manifest.entities:
            entities_file = f"{bundle_path}/{bundle_manifest.entities.path}"
            self._debug_print_sample_entities(entities_file)
            self._load_entities(entities_file)

        # Load relationships
        if bundle_manifest.relationships:
            relationships_file = f"{bundle_path}/{bundle_manifest.relationships.path}"
            self._load_relationships(relationships_file)

        self.record_bundle(bundle_manifest)
        self._session.commit()

    def _debug_print_sample_entities(self, entities_file: str) -> None:
        """Print first few entities for debugging."""
        print("  Debug: Checking first 3 entities from bundle...")
        with open(entities_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                entity_data = json.loads(line)
                entity_id = entity_data.get("entity_id", "unknown")
                status = entity_data.get("status", "unknown")
                props = entity_data.get("properties", {})
                print(f"    Entity {i+1}: {entity_id} (status: {status})")
                print(f"      Properties type: {type(props).__name__}")
                if isinstance(props, dict):
                    print(f"      Properties keys: {list(props.keys())}")
                    if props:
                        print(f"      Properties sample: {str(props)[:300]}")
                else:
                    print(f"      Properties value: {str(props)[:300]}")

    def _load_entities(self, entities_file: str) -> None:
        """Load entities from JSONL file."""
        canonical_url_count = 0
        total_entities = 0
        canonical_entities = 0
        sample_with_url = None
        sample_without_url = None
        sample_entity_raw = None
        sample_canonical_entity = None

        with open(entities_file, "r") as f:
            for line in f:
                entity_data = json.loads(line)
                try:
                    entity_row = EntityRow.model_validate(entity_data)
                    entity_data = entity_row.model_dump(exclude_unset=True)
                except ValidationError as e:
                    print(f"  ⚠ Validation error for entity: {e}")
                    print(f"    Problematic data: {line.strip()}")
                    continue  # Skip this entity

                total_entities += 1
                entity_id = entity_data.get("entity_id", "unknown")
                status = entity_data.get("status", "unknown")

                # Track canonical entities and capture samples
                is_canonical = status == "canonical" or (not entity_id.startswith("prov:") and status != "provisional")
                if is_canonical:
                    canonical_entities += 1
                    if sample_canonical_entity is None:
                        sample_canonical_entity = self._capture_entity_sample(entity_data, entity_id, status)

                if sample_entity_raw is None:
                    sample_entity_raw = self._capture_entity_sample(entity_data, entity_id, status)

                # Check if canonical_url exists in properties before normalization
                has_url_in_props, props = self._check_canonical_url_in_props(entity_data, entity_id, status)
                if has_url_in_props and sample_with_url is None:
                    sample_with_url = {
                        "entity_id": entity_id,
                        "status": status,
                        "properties": props,
                        "canonical_url_in_props": props.get("canonical_url") or props.get("canonicalUrl"),
                    }

                # Flatten metadata into top-level fields if present
                normalized_data = self._normalize_entity(entity_data)

                # Verify canonical_url was extracted
                if normalized_data.get("canonical_url"):
                    canonical_url_count += 1
                elif has_url_in_props and sample_without_url is None:
                    sample_without_url = {
                        "entity_id": entity_id,
                        "status": status,
                        "properties_before": props,
                        "properties_after": normalized_data.get("properties"),
                        "canonical_url_after": normalized_data.get("canonical_url"),
                    }

                entity = Entity(**normalized_data)
                self._session.merge(entity)

        self._print_entity_loading_summary(canonical_url_count, canonical_entities, total_entities, sample_canonical_entity, sample_entity_raw, sample_with_url, sample_without_url)

    def _capture_entity_sample(self, entity_data: dict, entity_id: str, status: str) -> dict:
        """Capture a sample entity for debugging."""
        return {
            "entity_id": entity_id,
            "status": status,
            "has_properties": "properties" in entity_data,
            "properties_type": type(entity_data.get("properties")).__name__ if "properties" in entity_data else None,
            "properties_keys": list(entity_data.get("properties", {}).keys()) if isinstance(entity_data.get("properties"), dict) else None,
            "properties_value": str(entity_data.get("properties"))[:200] if "properties" in entity_data else None,
        }

    def _check_canonical_url_in_props(self, entity_data: dict, entity_id: str, status: str) -> tuple[bool, dict]:
        """Check if canonical_url exists in properties and return it along with props."""
        has_url_in_props = False
        props = {}
        if "properties" in entity_data:
            props = entity_data["properties"]
            if isinstance(props, dict):
                if "canonical_url" in props or "canonicalUrl" in props:
                    has_url_in_props = True
        return has_url_in_props, props

    def _print_entity_loading_summary(
        self,
        canonical_url_count: int,
        canonical_entities: int,
        total_entities: int,
        sample_canonical_entity: Optional[dict],
        sample_entity_raw: Optional[dict],
        sample_with_url: Optional[dict],
        sample_without_url: Optional[dict],
    ) -> None:
        """Print summary of entity loading with debug information."""
        if canonical_url_count > 0:
            print(f"  ✓ Extracted canonical_url for {canonical_url_count} of {canonical_entities} canonical entities ({total_entities} total)")
        else:
            print("  ⚠ Warning: No canonical_url found in bundle properties")
            print(f"     Total entities: {total_entities}, Canonical entities: {canonical_entities}")
            if sample_canonical_entity:
                self._print_entity_sample("Sample CANONICAL entity structure:", sample_canonical_entity)
            if sample_entity_raw:
                self._print_entity_sample("Sample entity (any status):", sample_entity_raw)
            if sample_with_url:
                print(f"     Sample entity with URL in properties: {sample_with_url['entity_id']} (status: {sample_with_url['status']})")
                print(f"     URL value: {sample_with_url['canonical_url_in_props']}")
            if sample_without_url:
                print(f"     Sample entity where extraction failed: {sample_without_url['entity_id']} (status: {sample_without_url['status']})")
                print(f"     Properties before: {sample_without_url['properties_before']}")
                print(f"     Properties after: {sample_without_url['properties_after']}")
                print(f"     Canonical URL after: {sample_without_url['canonical_url_after']}")

    def _print_entity_sample(self, title: str, sample: dict) -> None:
        """Print a sample entity structure."""
        print(f"     {title}")
        print(f"       entity_id: {sample['entity_id']}")
        print(f"       status: {sample['status']}")
        print(f"       has_properties: {sample['has_properties']}")
        print(f"       properties_type: {sample['properties_type']}")
        print(f"       properties_keys: {sample['properties_keys']}")
        if "properties_value" in sample:
            print(f"       properties_value (first 200 chars): {sample['properties_value']}")

    def _load_relationships(self, relationships_file: str) -> None:
        """Load relationships from JSONL file."""
        with open(relationships_file, "r") as f:
            for line in f:
                relationship_data = json.loads(line)
                try:
                    relationship_row = RelationshipRow.model_validate(relationship_data)
                    relationship_data = relationship_row.model_dump(exclude_unset=True)
                except ValidationError as e:
                    print(f"  ⚠ Validation error for relationship: {e}")
                    print(f"    Problematic data: {line.strip()}")
                    continue
                # Map source_entity_id/target_entity_id to subject_id/object_id
                relationship_data = self._normalize_relationship(relationship_data)
                relationship = Relationship(**relationship_data)
                self._session.merge(relationship)

    def _normalize_entity(self, data: dict) -> dict:
        """Normalize entity data, flattening metadata fields."""
        result = dict(data)
        # If metadata contains fields that should be top-level, extract them
        if "metadata" in result and isinstance(result["metadata"], dict):
            meta = result.pop("metadata")
            for key in ["status", "usage_count", "source", "created_at", "canonical_url"]:
                if key in meta and key not in result:
                    result[key] = meta[key]
            # Store remaining metadata in properties
            if meta:
                result.setdefault("properties", {}).update(meta)

        # Ensure properties is a dict (handle case where it might be a JSON string or None)
        if "properties" not in result:
            result["properties"] = {}
        elif result["properties"] is None:
            result["properties"] = {}
        elif not isinstance(result["properties"], dict):
            if isinstance(result["properties"], str):
                try:
                    result["properties"] = json.loads(result["properties"])
                except (json.JSONDecodeError, TypeError):
                    result["properties"] = {}
            else:
                result["properties"] = {}

        # Extract canonical_url from properties if not already set
        # Check both canonical_url and canonicalUrl (camelCase variant)
        if "canonical_url" not in result or not result.get("canonical_url"):
            props = result.get("properties", {})
            if isinstance(props, dict):
                # Try canonical_url first
                if "canonical_url" in props:
                    result["canonical_url"] = props.pop("canonical_url")
                # Also try canonicalUrl (camelCase)
                elif "canonicalUrl" in props:
                    result["canonical_url"] = props.pop("canonicalUrl")

        return result

    def _normalize_relationship(self, data: dict) -> dict:
        """Normalize relationship data, mapping field names."""
        result = dict(data)
        # Map source_entity_id -> subject_id
        if "source_entity_id" in result and "subject_id" not in result:
            result["subject_id"] = result.pop("source_entity_id")
        # Map target_entity_id -> object_id
        if "target_entity_id" in result and "object_id" not in result:
            result["object_id"] = result.pop("target_entity_id")
        # Handle metadata -> properties
        if "metadata" in result and isinstance(result["metadata"], dict):
            meta = result.pop("metadata")
            if "source_documents" in meta and "source_documents" not in result:
                result["source_documents"] = meta.pop("source_documents", [])
            result.setdefault("properties", {}).update(meta)
        # Remove fields not in the model
        result.pop("relationship_id", None)
        result.pop("evidence_document_id", None)
        return result

    def is_bundle_loaded(self, bundle_id: str) -> bool:
        """
        Check if a bundle with the given ID is already loaded.
        """
        bundle = self._session.get(Bundle, bundle_id)
        return bundle is not None

    def record_bundle(self, bundle_manifest: BundleManifestV1) -> None:
        """
        Record that a bundle has been loaded.
        """
        bundle = Bundle(
            bundle_id=bundle_manifest.bundle_id,
            domain=bundle_manifest.domain,
            created_at=datetime.fromisoformat(bundle_manifest.created_at),
            bundle_version=bundle_manifest.bundle_version,
        )
        self._session.add(bundle)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID.
        """
        return self._session.get(Entity, entity_id)

    def get_entities(
        self,
        limit: int = 100,
        offset: int = 0,
        entity_type: Optional[str] = None,
        name: Optional[str] = None,
        name_contains: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Sequence[Entity]:
        """
        List entities with optional filtering.
        """
        statement = select(Entity)
        if entity_type:
            statement = statement.where(Entity.entity_type == entity_type)
        if name:
            statement = statement.where(Entity.name == name)
        if name_contains:
            statement = statement.where(Entity.name.ilike(f"%{name_contains}%"))  # type: ignore[union-attr]
        if source:
            statement = statement.where(Entity.source == source)
        if status:
            statement = statement.where(Entity.status == status)
        statement = statement.limit(limit).offset(offset)
        return self._session.exec(statement).all()

    def count_entities(
        self,
        entity_type: Optional[str] = None,
        name: Optional[str] = None,
        name_contains: Optional[str] = None,
        source: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """
        Count entities matching filter criteria.
        """
        statement = select(func.count(Entity.entity_id))  # type: ignore[arg-type] # pylint: disable=not-callable
        if entity_type:
            statement = statement.where(Entity.entity_type == entity_type)
        if name:
            statement = statement.where(Entity.name == name)
        if name_contains:
            statement = statement.where(Entity.name.ilike(f"%{name_contains}%"))  # type: ignore[union-attr]
        if source:
            statement = statement.where(Entity.source == source)
        if status:
            statement = statement.where(Entity.status == status)
        return self._session.exec(statement).one()

    def find_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Sequence[Relationship]:
        """
        Find relationships matching criteria.
        """
        statement = select(Relationship)
        if subject_id:
            statement = statement.where(Relationship.subject_id == subject_id)
        if predicate:
            statement = statement.where(Relationship.predicate == predicate)
        if object_id:
            statement = statement.where(Relationship.object_id == object_id)
        if limit:
            statement = statement.limit(limit)
        if offset:
            statement = statement.offset(offset)
        return self._session.exec(statement).all()

    def count_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> int:
        """
        Count relationships matching filter criteria.
        """
        statement = select(func.count(Relationship.id))  # type: ignore[arg-type] # pylint: disable=not-callable
        if subject_id:
            statement = statement.where(Relationship.subject_id == subject_id)
        if predicate:
            statement = statement.where(Relationship.predicate == predicate)
        if object_id:
            statement = statement.where(Relationship.object_id == object_id)
        return self._session.exec(statement).one()

    def get_relationship(self, subject_id: str, predicate: str, object_id: str) -> Optional[Relationship]:
        """
        Get a relationship by its canonical triple (subject_id, predicate, object_id).
        """
        statement = select(Relationship).where(
            Relationship.subject_id == subject_id,
            Relationship.predicate == predicate,
            Relationship.object_id == object_id,
        )
        return self._session.exec(statement).first()

    def get_relationships(self, limit: int = 100, offset: int = 0) -> Sequence[Relationship]:
        """
        List all relationships.
        """
        statement = select(Relationship).limit(limit).offset(offset)
        return self._session.exec(statement).all()

    def get_bundle_info(self):
        """
        Get bundle metadata (latest bundle).
        Returns None if no bundle is loaded.
        """
        statement = select(Bundle).order_by(Bundle.created_at.desc()).limit(1)
        return self._session.exec(statement).first()

    def close(self) -> None:
        """
        Close connections and clean up resources.
        """
        self._session.close()
