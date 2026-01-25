# examples/sherlock/domain.py
from __future__ import annotations

from typing import Optional

from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema
from kgraph.entity import BaseEntity, PromotionConfig
from kgraph.relationship import BaseRelationship
from kgraph.promotion import PromotionPolicy

from .promotion import SherlockPromotionPolicy

# -----------------------
# Entities
# -----------------------


class SherlockCharacter(BaseEntity):
    """A character in the Sherlock Holmes stories."""

    role: Optional[str] = None

    def get_entity_type(self) -> str:
        return "character"


class SherlockLocation(BaseEntity):
    """A location mentioned in the stories."""

    location_type: Optional[str] = None

    def get_entity_type(self) -> str:
        return "location"


class SherlockStory(BaseEntity):
    """A story or novel in the Holmes canon."""

    collection: Optional[str] = None
    publication_year: Optional[int] = None

    def get_entity_type(self) -> str:
        return "story"


# -----------------------
# Relationships
# -----------------------


class AppearsInRelationship(BaseRelationship):
    """Character appears in a story."""

    def get_edge_type(self) -> str:
        return "appears_in"


class CoOccursWithRelationship(BaseRelationship):
    """Two characters co-occur within the same textual context."""

    def get_edge_type(self) -> str:
        return "co_occurs_with"


class LivesAtRelationship(BaseRelationship):
    """Character lives at a location."""

    def get_edge_type(self) -> str:
        return "lives_at"


class AntagonistOfRelationship(BaseRelationship):
    """Character is an antagonist of another character."""

    def get_edge_type(self) -> str:
        return "antagonist_of"


class AllyOfRelationship(BaseRelationship):
    """Character is an ally of another character."""

    def get_edge_type(self) -> str:
        return "ally_of"


# -----------------------
# Documents
# -----------------------


class SherlockDocument(BaseDocument):
    """A Sherlock Holmes story document."""

    story_id: Optional[str] = None
    collection: Optional[str] = None

    def get_document_type(self) -> str:
        return "sherlock_story"

    def get_sections(self) -> list[tuple[str, str]]:
        return [("body", self.content)]


# -----------------------
# Domain
# -----------------------


class SherlockDomainSchema(DomainSchema):
    @property
    def name(self) -> str:
        return "sherlock"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {
            "character": SherlockCharacter,
            "location": SherlockLocation,
            "story": SherlockStory,
        }

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {
            "appears_in": AppearsInRelationship,
            "co_occurs_with": CoOccursWithRelationship,
            "lives_at": LivesAtRelationship,
            "antagonist_of": AntagonistOfRelationship,
            "ally_of": AllyOfRelationship,
        }

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        return {"sherlock_story": SherlockDocument}

    @property
    def promotion_config(self) -> PromotionConfig:
        # Sherlock is a demo; keep it simple.
        return PromotionConfig(
            min_usage_count=2,
            min_confidence=0.7,
            require_embedding=False,
        )

    def get_promotion_policy(self, lookup=None) -> PromotionPolicy:
        return SherlockPromotionPolicy(self.promotion_config)

    def validate_entity(self, entity: BaseEntity) -> bool:
        return entity.get_entity_type() in self.entity_types

    def validate_relationship(self, relationship: BaseRelationship) -> bool:
        return relationship.predicate in self.relationship_types
