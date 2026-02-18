"""
Storage interfaces for the Knowledge Graph Server.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Sequence
from .models.entity import Entity
from .models.relationship import Relationship
from kgbundle import BundleManifestV1

if TYPE_CHECKING:
    from kgbundle import EvidenceRow, MentionRow


class StorageInterface(ABC):
    """
    Abstract interface for a knowledge graph storage backend.
    """

    @abstractmethod
    def load_bundle(self, bundle_manifest: BundleManifestV1, bundle_path: str) -> None:
        """
        Load a data bundle into the storage.
        This should be an idempotent operation.
        """
        pass

    @abstractmethod
    def is_bundle_loaded(self, bundle_id: str) -> bool:
        """
        Check if a bundle with the given ID is already loaded.
        """
        pass

    @abstractmethod
    def record_bundle(self, bundle_manifest: BundleManifestV1) -> None:
        """
        Record that a bundle has been loaded.
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def count_relationships(
        self,
        subject_id: Optional[str] = None,
        predicate: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> int:
        """
        Count relationships matching filter criteria.
        """
        pass

    @abstractmethod
    def get_relationship(self, subject_id: str, predicate: str, object_id: str) -> Optional[Relationship]:
        """
        Get a relationship by its canonical triple (subject_id, predicate, object_id).
        """
        pass

    @abstractmethod
    def get_relationships(self, limit: int = 100, offset: int = 0) -> Sequence[Relationship]:
        """
        List all relationships.
        """
        pass

    @abstractmethod
    def get_bundle_info(self):
        """
        Get bundle metadata (latest bundle).
        Returns None if no bundle is loaded.
        """
        pass

    def get_mentions_for_entity(self, entity_id: str) -> Sequence["MentionRow"]:
        """
        Return all mention rows for the given entity (bundle provenance).
        Returns empty list if no mentions or provenance not loaded.
        """
        return []

    def get_evidence_for_relationship(self, subject_id: str, predicate: str, object_id: str) -> Sequence["EvidenceRow"]:
        """
        Return all evidence rows for the given relationship triple (bundle provenance).
        Returns empty list if no evidence or provenance not loaded.
        """
        return []

    @abstractmethod
    def close(self) -> None:
        """
        Close connections and clean up resources.
        """
        pass
