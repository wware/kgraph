from abc import ABC, abstractmethod
from typing import Optional
from .entity import BaseEntity, PromotionConfig


class PromotionPolicy(ABC):
    """Abstract base for domain-specific entity promotion policies."""

    def __init__(self, config: PromotionConfig):
        self.config = config

    def should_promote(self, entity: BaseEntity) -> bool:
        """Check if entity meets promotion thresholds."""
        if entity.status != "provisional":
            return False
        return entity.usage_count >= self.config.min_usage_count and entity.confidence >= self.config.min_confidence and (not self.config.require_embedding or entity.embedding is not None)

    @abstractmethod
    def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        """Return canonical ID for this entity, or None if not found."""
        pass


class TodoPromotionPolicy(PromotionPolicy):
    def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        raise NotImplementedError("Need a proper PromotionPolicy")
