"""Document representation for the knowledge graph framework.

This module defines `BaseDocument`, the abstract base class for all documents
processed by the knowledge graph ingestion pipeline. Documents represent the
source material from which entities and relationships are extracted.

A document contains:
    - **Content**: The full text of the document
    - **Metadata**: Title, source URI, content type, creation timestamp
    - **Structure**: Domain-specific sections via `get_sections()`

Domain implementations subclass `BaseDocument` to add domain-specific fields
and structure. For example:
    - `JournalArticle` might add fields for authors, abstract, and citations
    - `LegalDocument` might add fields for court, case number, and parties
    - `ConferencePaper` might add fields for venue, year, and keywords

Documents are immutable (frozen Pydantic models) to ensure consistency
throughout the extraction pipeline.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field


class BaseDocument(ABC, BaseModel):
    """Abstract base class for documents in the knowledge graph.

    Represents a parsed document ready for entity and relationship extraction.
    All documents share common fields (ID, content, metadata) while subclasses
    add domain-specific structure and fields.

    Documents are frozen (immutable) Pydantic models, ensuring they cannot be
    modified after creation. This immutability guarantees consistency when
    documents are referenced by multiple entities and relationships.

    Subclasses must implement:
        - `get_document_type()`: Return the domain-specific document type
        - `get_sections()`: Return document structure as (name, content) tuples

    Example:
        ```python
        class JournalArticle(BaseDocument):
            authors: tuple[str, ...] = Field(default=())
            abstract: str | None = None

            def get_document_type(self) -> str:
                return "journal_article"

            def get_sections(self) -> list[tuple[str, str]]:
                sections = []
                if self.abstract:
                    sections.append(("abstract", self.abstract))
                sections.append(("body", self.content))
                return sections
        ```
    """

    model_config = {"frozen": True}

    document_id: str = Field(description="Unique identifier for this document.")
    title: str | None = Field(default=None, description="Document title if available.")
    content: str = Field(description="Full text content of the document.")
    content_type: str = Field(description="MIME type or format indicator (e.g., 'text/plain', 'application/pdf').")
    source_uri: str | None = Field(default=None, description="Original source location (URL, file path, etc.).")
    created_at: datetime = Field(description="When the document was added to the system.")
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
