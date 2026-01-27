"""
SQLite implementation of the storage interface.
"""

import json
from datetime import datetime
from typing import Optional, Sequence
from sqlalchemy import func
from sqlmodel import Session, SQLModel, create_engine, select
from storage.interfaces import StorageInterface
from storage.models import Bundle, Entity, Relationship
from kgbundle import BundleManifestV1


class SQLiteStorage(StorageInterface):
    """
    SQLite implementation of the storage interface.
    """

    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)
        self._session = Session(self.engine)

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
            with open(entities_file, "r") as f:
                for line in f:
                    entity_data = json.loads(line)
                    entity_data = self._normalize_entity(entity_data)
                    entity = Entity(**entity_data)
                    self._session.merge(entity)

        # Load relationships
        if bundle_manifest.relationships:
            relationships_file = f"{bundle_path}/{bundle_manifest.relationships.path}"
            with open(relationships_file, "r") as f:
                for line in f:
                    relationship_data = json.loads(line)
                    relationship_data = self._normalize_relationship(relationship_data)
                    relationship = Relationship(**relationship_data)
                    self._session.merge(relationship)

        self.record_bundle(bundle_manifest)
        self._session.commit()

    def _normalize_entity(self, data: dict) -> dict:
        """Normalize entity data, flattening metadata fields."""
        result = dict(data)
        if "metadata" in result and isinstance(result["metadata"], dict):
            meta = result.pop("metadata")
            for key in ["status", "usage_count", "source", "created_at", "canonical_url"]:
                if key in meta and key not in result:
                    result[key] = meta[key]
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
        if "source_entity_id" in result and "subject_id" not in result:
            result["subject_id"] = result.pop("source_entity_id")
        if "target_entity_id" in result and "object_id" not in result:
            result["object_id"] = result.pop("target_entity_id")
        if "metadata" in result and isinstance(result["metadata"], dict):
            meta = result.pop("metadata")
            if "source_documents" in meta and "source_documents" not in result:
                result["source_documents"] = meta.pop("source_documents", [])
            result.setdefault("properties", {}).update(meta)
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
            # SQLite doesn't have ILIKE, but LIKE is case-insensitive for ASCII
            # Use ilike() which SQLAlchemy will translate appropriately
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
