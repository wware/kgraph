# examples/sherlock/extractors.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from kgraph.entity import BaseEntity, EntityMention, EntityStatus
from kgraph.pipeline.interfaces import (
    EntityResolverInterface,
    EntityStorageInterface,
)

from ..data import ADVENTURES_STORIES, KNOWN_CHARACTERS, KNOWN_LOCATIONS
from ..domain import (
    SherlockCharacter,
    SherlockLocation,
    SherlockStory,
)


# ============================================================================
# Entity resolver
# ============================================================================


class SherlockEntityResolver(EntityResolverInterface):
    """Resolve mentions to canonical entities when a hint is present; else provisional."""

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        canonical_id = (mention.metadata or {}).get("canonical_id_hint") if hasattr(mention, "metadata") else None

        if canonical_id:
            existing = await existing_storage.get(canonical_id)
            if existing:
                return existing, float(getattr(mention, "confidence", 1.0))

            # Create the right concrete entity type
            if mention.entity_type == "character":
                data = KNOWN_CHARACTERS.get(canonical_id, {})
                ent = SherlockCharacter(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=data.get("name", mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    role=data.get("role"),
                )
                return ent, ent.confidence

            if mention.entity_type == "location":
                data = KNOWN_LOCATIONS.get(canonical_id, {})
                ent = SherlockLocation(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=data.get("name", mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    location_type=data.get("location_type"),
                )
                return ent, ent.confidence

            if mention.entity_type == "story":
                # Look up story data from ADVENTURES_STORIES if available
                meta = next((s for s in ADVENTURES_STORIES if s["canonical_id"] == canonical_id), None)
                ent = SherlockStory(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=(meta["title"] if meta else mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    collection=(meta.get("collection") if meta else None),
                    publication_year=(meta.get("year") if meta else None),
                )
                return ent, ent.confidence

        # Provisional fallback
        prov_id = f"prov:{uuid.uuid4().hex}"
        ent = BaseEntity(
            entity_id=prov_id,
            status=EntityStatus.PROVISIONAL,
            name=mention.text,
            confidence=float(getattr(mention, "confidence", 0.5)) * 0.5,
            usage_count=1,
            created_at=datetime.now(timezone.utc),
            source="sherlock:extracted",
            metadata={},
        )
        return ent, ent.confidence

    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]:
        out: list[tuple[BaseEntity, float]] = []
        for m in mentions:
            out.append(await self.resolve(m, existing_storage))
        return out
