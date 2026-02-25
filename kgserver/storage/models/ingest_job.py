"""
Ingest job model for tracking paper ingestion background jobs.
"""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class IngestJob(SQLModel, table=True):
    """
    A single paper ingestion job (queued, running, complete, or failed).
    """

    __tablename__ = "ingest_jobs"

    id: str = Field(primary_key=True, description="UUID string for the job")
    url: str = Field(description="URL of the paper to ingest")
    status: str = Field(default="queued", description="queued | running | complete | failed")
    created_at: datetime = Field(description="When the job was created")
    started_at: Optional[datetime] = Field(default=None, description="When the job started running")
    completed_at: Optional[datetime] = Field(default=None, description="When the job completed or failed")
    paper_title: Optional[str] = Field(default=None, description="Title of the paper once extracted")
    pmcid: Optional[str] = Field(default=None, description="PMC ID if the paper is from PMC")
    entities_added: int = Field(default=0, description="Number of entities added from this paper")
    relationships_added: int = Field(default=0, description="Number of relationships added")
    error: Optional[str] = Field(default=None, description="Error message if status is failed")
