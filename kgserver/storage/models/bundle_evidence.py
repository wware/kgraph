"""
Bundle provenance: one row per relationship evidence span (evidence.jsonl).
"""

from typing import Optional

from sqlmodel import Field, SQLModel


class BundleEvidence(SQLModel, table=True):
    """One evidence span for a relationship from a loaded bundle (evidence.jsonl)."""

    __tablename__ = "bundle_evidence"

    id: Optional[int] = Field(default=None, primary_key=True)
    relationship_key: str = Field(index=True)
    document_id: str = Field(index=True)
    section: Optional[str] = Field(default=None)
    start_offset: int = Field()
    end_offset: int = Field()
    text_span: str = Field()
    confidence: float = Field()
    supports: bool = Field(default=True)
