"""Relationship system for the knowledge graph framework."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field


class BaseRelationship(ABC, BaseModel):
    """Abstract base for all domain relationships.

    Represents an edge between two entities in the knowledge graph.
    Subclasses must implement get_edge_type() to define domain-specific behavior.
    """

    model_config = {"frozen": True}

    subject_id: str = Field(
        description="Entity ID of the relationship subject."
    )
    predicate: str = Field(
        description="Relationship type (domain defines valid predicates)."
    )
    object_id: str = Field(
        description="Entity ID of the relationship object."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this relationship.",
    )
    source_documents: tuple[str, ...] = Field(
        default=(),
        description="Document IDs where this relationship was found.",
    )
    created_at: datetime = Field(
        description="Timestamp when relationship was first created."
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Timestamp of most recent update.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific metadata (evidence, provenance, etc.).",
    )

    @abstractmethod
    def get_edge_type(self) -> str:
        """Return domain-specific edge type category.

        Examples: 'treats', 'causes', 'interacts_with' for medical domain;
        'cites', 'overrules', 'interprets' for legal domain.
        """
