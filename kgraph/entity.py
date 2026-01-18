"""Entity system for the knowledge graph framework."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EntityStatus(str, Enum):
    """Status of an entity in the knowledge graph."""

    CANONICAL = "canonical"
    """Entity has been assigned a stable ID from an authoritative source."""

    PROVISIONAL = "provisional"
    """Entity is awaiting promotion based on usage count and confidence scores."""


class PromotionConfig(BaseModel, frozen=True):
    """Configuration for entity promotion thresholds.

    Controls when provisional entities get promoted to canonical status.
    Domains can override these defaults via DomainSchema.promotion_config.
    """

    min_usage_count: int = Field(
        default=3,
        ge=1,
        description="Minimum number of appearances before promotion is considered.",
    )
    min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required for promotion.",
    )
    require_embedding: bool = Field(
        default=True,
        description="Whether an embedding must be present for promotion.",
    )


class BaseEntity(ABC, BaseModel):
    """Abstract base for all domain entities.

    Subclasses must implement get_entity_type() and get_canonical_id_source()
    to define domain-specific behavior.
    """

    model_config = {"frozen": True}

    entity_id: str = Field(
        description="Domain-specific canonical ID or provisional UUID."
    )
    status: EntityStatus = Field(
        description="Whether entity is canonical or provisional."
    )
    name: str = Field(description="Primary name/label for the entity.")
    synonyms: tuple[str, ...] = Field(
        default=(),
        description="Alternative names or aliases for this entity.",
    )
    embedding: tuple[float, ...] | None = Field(
        default=None,
        description="Semantic vector embedding for similarity operations.",
    )
    dbpedia_uri: str | None = Field(
        default=None,
        description="Cross-domain linking via DBPedia URI.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the canonical ID assignment.",
    )
    usage_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this entity has been referenced.",
    )
    created_at: datetime = Field(
        description="Timestamp when the entity was first created."
    )
    source: str = Field(
        description="Origin indicator (e.g., document ID, extraction pipeline)."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific metadata.",
    )

    @abstractmethod
    def get_entity_type(self) -> str:
        """Return domain-specific entity type identifier.

        Examples: 'drug', 'disease', 'gene' for medical domain;
        'case', 'statute', 'court' for legal domain.
        """

    @abstractmethod
    def get_canonical_id_source(self) -> str | None:
        """Return the authoritative source for this entity's ID.

        Examples: 'UMLS' for medical entities, 'DBPedia' for cross-domain,
        'court_id' for legal cases. Returns None if no canonical source applies.
        """


class EntityMention(BaseModel, frozen=True):
    """A mention of an entity extracted from a document.

    Represents a raw extraction before resolution to a canonical or
    provisional entity.
    """

    text: str = Field(
        description="The exact text span mentioning the entity."
    )
    entity_type: str = Field(
        description="Domain-specific type classification."
    )
    start_offset: int = Field(
        ge=0,
        description="Character offset where mention starts in source text.",
    )
    end_offset: int = Field(
        ge=0,
        description="Character offset where mention ends in source text.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score.",
    )
    context: str | None = Field(
        default=None,
        description="Surrounding text for disambiguation.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific extraction metadata.",
    )
