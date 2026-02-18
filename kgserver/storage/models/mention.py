"""
Bundle provenance: one row per entity mention (mentions.jsonl).
"""

from typing import Optional

from sqlmodel import Field, SQLModel


class Mention(SQLModel, table=True):
    """One entity mention from a loaded bundle (mentions.jsonl)."""

    __tablename__ = "bundle_mention"

    id: Optional[int] = Field(default=None, primary_key=True)
    entity_id: str = Field(index=True)
    document_id: str = Field(index=True)
    section: Optional[str] = Field(default=None)
    start_offset: int = Field()
    end_offset: int = Field()
    text_span: str = Field()
    context: Optional[str] = Field(default=None)
    confidence: float = Field()
    extraction_method: str = Field()
    created_at: str = Field()
