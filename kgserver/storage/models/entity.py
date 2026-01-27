"""
Generic Entity model for the Knowledge Graph Server.
"""

from typing import Optional, List, Any
from sqlmodel import Field, SQLModel, JSON, Column


class Entity(SQLModel, table=True):
    """
    A generic entity in the knowledge graph.
    """

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
