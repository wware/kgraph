# storage/models/bundle.py

from datetime import datetime
from sqlmodel import Field, SQLModel


class Bundle(SQLModel, table=True):
    """
    Represents a loaded data bundle's metadata for idempotent tracking.
    """

    bundle_id: str = Field(primary_key=True)
    domain: str
    created_at: datetime
    bundle_version: str

    # Add any other metadata from the manifest that might be useful
    # for auditing or display purposes, e.g., generator info, checksum.
