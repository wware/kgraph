"""Relationship system for the knowledge graph framework.

This module defines `BaseRelationship`, the abstract base class for all
relationships (edges) in the knowledge graph. Relationships connect entities
via typed predicates, forming the graph structure.

Each relationship is a triple: (subject_entity, predicate, object_entity)

- **Subject**: The entity performing or originating the action
- **Predicate**: The relationship type (domain-specific vocabulary)
- **Object**: The entity receiving or being affected by the action

For example:
    - ("Aspirin", "treats", "Headache")
    - ("Paper A", "cites", "Paper B")
    - ("Court Case X", "overrules", "Court Case Y")

Relationships also track:
    - **Confidence**: How certain we are about this relationship
    - **Source documents**: Which documents support this relationship
    - **Metadata**: Domain-specific evidence and provenance

Relationships are immutable (frozen Pydantic models) and are typically
extracted during Pass 2 of the ingestion pipeline, after entities have
been resolved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kgraph.domain import Evidence


class BaseRelationship(ABC, BaseModel):
    """Abstract base class for relationships (edges) in the knowledge graph.

    Relationships connect two entities via a typed predicate, representing
    facts extracted from source documents. The relationship model supports
    aggregating evidence from multiple documents that assert the same fact.

    Relationships are frozen (immutable) Pydantic models. To update a
    relationship (e.g., to add a new source document), use
    `rel.model_copy(update={...})` to create a new instance.

    Subclasses must implement:
        - `get_edge_type()`: Return the domain-specific edge type category

    Key fields:
        - `subject_id`: Entity ID of the relationship source (the "doer")
        - `predicate`: Relationship type from the domain vocabulary
        - `object_id`: Entity ID of the relationship target (the "receiver")
        - `confidence`: How certain we are about this relationship
        - `source_documents`: Documents that support this relationship

    Example:
        ```python
        class TreatsRelationship(BaseRelationship):
            mechanism: str | None = None  # How the treatment works
            evidence_level: str = "observational"

            def get_edge_type(self) -> str:
                return "treats"
        ```
    """

    model_config = {"frozen": True}

    subject_id: str = Field(description="Entity ID of the relationship subject.")
    predicate: str = Field(description="Relationship type (domain defines valid predicates).")
    object_id: str = Field(description="Entity ID of the relationship object.")
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
    evidence: Any = Field(
        default=None,
        description="Structured evidence including provenance (document, section, paragraph, offsets). Type: Evidence | None",
    )
    created_at: datetime = Field(description="Timestamp when relationship was first created.")
    last_updated: datetime | None = Field(
        default=None,
        description="Timestamp of most recent update.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific metadata. NOTE: Prefer using 'evidence' field for provenance tracking.",
    )

    @abstractmethod
    def get_edge_type(self) -> str:
        """Return domain-specific edge type category.

        Examples: 'treats', 'causes', 'interacts_with' for medical domain;
        'cites', 'overrules', 'interprets' for legal domain.
        """
