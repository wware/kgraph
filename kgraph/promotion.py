"""Legacy promotion utilities.

This module is kept for backwards compatibility. The PromotionPolicy ABC
has been moved to kgschema.promotion.
"""

from typing import Optional

from kgschema.canonical_id import CanonicalId
from kgschema.entity import BaseEntity
from kgschema.promotion import PromotionPolicy


class TodoPromotionPolicy(PromotionPolicy):
    """Placeholder promotion policy that raises NotImplementedError."""

    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        raise NotImplementedError("Need a proper PromotionPolicy")
