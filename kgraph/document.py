"""Document representation for the knowledge graph framework."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field


class BaseDocument(ABC, BaseModel):
    """Abstract base for documents processed by the knowledge graph.

    Represents a parsed document ready for entity and relationship extraction.
    Subclasses define domain-specific document structure.
    """

    model_config = {"frozen": True}

    document_id: str = Field(
        description="Unique identifier for this document."
    )
    title: str | None = Field(
        default=None,
        description="Document title if available."
    )
    content: str = Field(
        description="Full text content of the document."
    )
    content_type: str = Field(
        description="MIME type or format indicator (e.g., 'text/plain', 'application/pdf')."
    )
    source_uri: str | None = Field(
        default=None,
        description="Original source location (URL, file path, etc.)."
    )
    created_at: datetime = Field(
        description="When the document was added to the system."
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific document metadata.",
    )

    @abstractmethod
    def get_document_type(self) -> str:
        """Return domain-specific document type.

        Examples: 'journal_article', 'clinical_trial' for medical;
        'case_opinion', 'statute' for legal; 'conference_paper' for CS.
        """

    @abstractmethod
    def get_sections(self) -> list[tuple[str, str]]:
        """Return document sections as (section_name, content) tuples.

        Allows domain-specific document structure. For unstructured documents,
        return a single section like [('body', self.content)].
        """
