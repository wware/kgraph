from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone
import uuid
from typing import Sequence

from kgraph.pipeline.interfaces import EntityResolverInterface
from kgraph.entity import BaseEntity, EntityStatus, EntityMention
from kgraph.domain import DomainSchema
from kgraph.storage.interfaces import EntityStorageInterface


class SherlockEntityResolver(BaseModel, EntityResolverInterface):
    """
    Resolve Sherlock entity mentions to canonical or provisional entities.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    domain: DomainSchema

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:

        entity_type = mention.entity_type
        entity_cls = self.domain.entity_types.get(entity_type)

        if entity_cls is None:
            raise ValueError(f"Unknown entity_type {entity_type!r}")

        # First, try to find an existing entity by name. This is the most reliable way
        # to ensure we group mentions of the same name together.
        existing = await existing_storage.find_by_name(mention.text, entity_type, limit=1)
        if existing:
            return existing[0], 0.95  # Return with high confidence since we found a match

        # If no entity found by name, create a new one.
        # Use the canonical_id_hint as the provisional ID if available.
        provisional_id: str = (mention.metadata.get("canonical_id_hint") if mention.metadata else None) or f"prov:{uuid.uuid4().hex}"

        entity = entity_cls(
            entity_id=provisional_id,
            name=mention.text,
            status=EntityStatus.PROVISIONAL,
            confidence=mention.confidence,
            usage_count=1,  # Initial usage
            source="sherlock:curated",
            created_at=datetime.now(timezone.utc),
            metadata={"canonical_id_hint": provisional_id} if mention.metadata else {},
        )
        return entity, mention.confidence

    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]:
        # Simple default: sequential resolution
        return [await self.resolve(m, existing_storage) for m in mentions]
