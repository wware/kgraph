"""
Generic Entity model for the Knowledge Graph Server.
"""

from typing import Optional, List, Any
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel, JSON, Column


class Entity(SQLModel, table=True):
    """
    A generic entity in the knowledge graph.
    """

    __table_args__ = (UniqueConstraint("name", "entity_type", name="uq_entity_name_type"),)

    entity_id: str = Field(primary_key=True)
    entity_type: str = Field(index=True)
    name: Optional[str] = Field(default=None, index=True)
    status: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)
    usage_count: Optional[int] = Field(default=None)
    source: Optional[str] = Field(default=None)
    canonical_url: Optional[str] = Field(default=None, description="URL to the authoritative source for this entity")
    synonyms: List[str] = Field(default=[], sa_column=Column(JSON))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    merged_into: Optional[str] = Field(
        default=None,
        description="If status='merged', the entity_id of the survivor. NULL otherwise.",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Semantic embedding vector for pgvector similarity search.",
    )
