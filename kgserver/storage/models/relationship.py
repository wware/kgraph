"""
Generic Relationship model for the Knowledge Graph Server.
"""

from typing import Optional, List, Any
from uuid import UUID, uuid4
from sqlmodel import Field, SQLModel, JSON, Column, UniqueConstraint


class Relationship(SQLModel, table=True):
    """
    A generic relationship in the knowledge graph.
    """

    __table_args__ = (UniqueConstraint("subject_id", "object_id", "predicate", name="uq_relationship"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    subject_id: str = Field(index=True)
    predicate: str = Field(index=True)
    object_id: str = Field(index=True)
    confidence: Optional[float] = Field(default=None)
    source_documents: List[str] = Field(default=[], sa_column=Column(JSON))
    properties: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
