"""
Knowledge Graph Bundle Models

Lightweight Pydantic models defining the contract between bundle producers (kgraph)
and consumers (kgserver).

This module has minimal dependencies (only pydantic) and is designed to be
importable by both sides without pulling in heavy ML or web frameworks.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EntityRow(BaseModel):
    """Entity row format for bundle JSONL files.

    Matches the server bundle contract with proper field names and types.
    """

    entity_id: str = Field(..., description="Unique entity identifier (namespaced)")
    entity_type: str = Field(..., description="Type of entity (e.g., character, location)")
    name: Optional[str] = Field(None, description="Primary display name")
    status: str = Field(..., description="Entity status (e.g., canonical, provisional)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    usage_count: int = Field(..., description="Number of times entity has been mentioned")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    source: str = Field(..., description="Source of the entity (e.g., sherlock:curated)")
    canonical_url: Optional[str] = Field(None, description="URL to the authoritative source for this entity")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity properties",
    )
    first_seen_document: Optional[str] = Field(None, description="Document ID of earliest mention")
    first_seen_section: Optional[str] = Field(None, description="Section of earliest mention")
    total_mentions: int = Field(0, description="Total number of mention rows for this entity")
    supporting_documents: List[str] = Field(
        default_factory=list,
        description="Distinct document IDs where this entity is mentioned",
    )


class RelationshipRow(BaseModel):
    """Relationship row format for bundle JSONL files.

    Matches the server bundle contract with proper field names and types.
    """

    subject_id: str = Field(..., description="Source/subject entity ID")
    object_id: str = Field(..., description="Target/object entity ID")
    predicate: str = Field(..., description="Relationship type (e.g., appears_in)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    source_documents: List[str] = Field(..., description="List of document IDs providing evidence")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties (co_occurrence_count, etc.)",
    )
    evidence_count: int = Field(0, description="Number of evidence spans for this relationship")
    strongest_evidence_quote: Optional[str] = Field(None, description="Text span of highest-confidence evidence")
    evidence_confidence_avg: Optional[float] = Field(None, description="Mean confidence of evidence spans")


class MentionRow(BaseModel):
    """One entity mention occurrence (one line in mentions.jsonl)."""

    entity_id: str = Field(..., description="Entity ID this mention resolves to")
    document_id: str = Field(..., description="Document where the mention appears")
    section: Optional[str] = Field(None, description="Section within the document")
    start_offset: int = Field(..., description="Start character offset in document")
    end_offset: int = Field(..., description="End character offset in document")
    text_span: str = Field(..., description="Mention text")
    context: Optional[str] = Field(None, description="Surrounding context")
    confidence: float = Field(..., description="Extraction confidence")
    extraction_method: str = Field(
        ...,
        description="How the mention was extracted (e.g. llm, rule_based, canonical_lookup)",
    )
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class EvidenceRow(BaseModel):
    """One evidence span supporting a relationship (one line in evidence.jsonl)."""

    relationship_key: str = Field(
        ...,
        description="Composite key: subject_id:predicate:object_id",
    )
    document_id: str = Field(..., description="Document where the evidence appears")
    section: Optional[str] = Field(None, description="Section within the document")
    start_offset: int = Field(..., description="Start character offset")
    end_offset: int = Field(..., description="End character offset")
    text_span: str = Field(..., description="Evidence quote text")
    confidence: float = Field(..., description="Confidence for this evidence")
    supports: bool = Field(True, description="True=supports, False=contradicts")


class BundleFile(BaseModel):
    """Reference to a file within the bundle."""

    path: str = Field(..., description="Path to the file relative to the manifest")
    format: str = Field(..., description="File format (e.g., jsonl)")


class DocAssetRow(BaseModel):
    """Documentation asset row format for bundle doc_assets.jsonl files.

    Lists static assets (markdown files, images, etc.) that should be
    copied from the bundle to provide documentation for the knowledge domain.

    Note: This is for human-readable documentation, NOT source documents
    (papers, articles) used for entity/relationship extraction.
    """

    path: str = Field(..., description="Path to the asset file relative to the bundle root")
    content_type: str = Field(..., description="MIME type of the asset (e.g., text/markdown, image/png)")


class BundleManifestV1(BaseModel):
    """Bundle manifest format matching the server contract.

    Contains bundle identification, file references, and metadata.
    """

    bundle_version: str = Field("v1", frozen=True, description="Bundle format version")
    bundle_id: str = Field(..., description="Unique bundle identifier (UUID)")
    domain: str = Field(..., description="Knowledge domain (e.g., sherlock, medical)")
    label: Optional[str] = Field(None, description="Human-readable bundle label")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    entities: BundleFile = Field(..., description="Entities file information")
    relationships: BundleFile = Field(..., description="Relationships file information")
    doc_assets: Optional[BundleFile] = Field(None, description="Optional doc_assets.jsonl file listing documentation assets")
    mentions: Optional[BundleFile] = Field(None, description="Optional mentions.jsonl (entity provenance)")
    evidence: Optional[BundleFile] = Field(None, description="Optional evidence.jsonl (relationship evidence)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional bundle metadata (description, counts, etc.)",
    )
