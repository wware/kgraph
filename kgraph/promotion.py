from abc import ABC, abstractmethod
from typing import Optional

from kgraph.canonical_id import CanonicalId
from .entity import BaseEntity, PromotionConfig
from kgraph.entity import EntityStatus


class PromotionPolicy(ABC):
    """Abstract base for domain-specific entity promotion policies."""

    def __init__(self, config: PromotionConfig):
        self.config = config

    def should_promote(self, entity: BaseEntity) -> bool:
        """Check if entity meets promotion thresholds."""
        if entity.status != EntityStatus.PROVISIONAL:
            return False
        return entity.usage_count >= self.config.min_usage_count and entity.confidence >= self.config.min_confidence and (not self.config.require_embedding or entity.embedding is not None)

    @abstractmethod
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        """Return canonical ID for this entity, or None if not found.

        This method is async to support external API lookups.
        Returns a CanonicalId object which includes the ID, URL, and synonyms.
        """
        pass


class TodoPromotionPolicy(PromotionPolicy):
    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        raise NotImplementedError("Need a proper PromotionPolicy")
