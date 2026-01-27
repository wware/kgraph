"""
Storage interfaces for the Knowledge Graph Server.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, TYPE_CHECKING
from .models.entity import Entity
from .models.relationship import Relationship
from kgbundle import BundleManifestV1

if TYPE_CHECKING:
    pass


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

    @abstractmethod
    def close(self) -> None:
        """
        Close connections and clean up resources.
        """
        pass
