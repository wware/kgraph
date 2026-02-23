"""Pydantic models for the per-paper bundle JSON (Pass 1 output / Pass 2 input).

Matches the structure in INGESTION_REFACTOR.md. Entity type is stored with
Field(alias="class") because "class" is a Python reserved word; use
model_dump(by_alias=True) for JSON and populate_by_name=True for parsing.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from examples.medlit_schema.base import ExtractionProvenance

# -----------------------------------------------------------------------------
# Paper metadata (top-level "paper" in bundle)
# -----------------------------------------------------------------------------


class PaperInfo(BaseModel):
    """Paper metadata in the per-paper bundle."""

    doi: Optional[str] = None
    pmcid: Optional[str] = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    study_type: Optional[str] = None
    eco_type: Optional[str] = None


# -----------------------------------------------------------------------------
# Provenance (re-use schema types; all optional at top level)
# -----------------------------------------------------------------------------

# ExtractionProvenance, ExtractionPipelineInfo, PromptInfo, ExecutionInfo,
# EntityResolutionInfo, ModelInfo are imported from base and used as-is.
# ExtractionProvenance already has optional extraction_pipeline, prompt,
# execution, entity_resolution.


# -----------------------------------------------------------------------------
# Entity and evidence rows (use entity_class + alias "class" for JSON)
# -----------------------------------------------------------------------------


class ExtractedEntityRow(BaseModel):
    """Minimal entity record in the bundle. JSON key "class" via alias."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    entity_class: str = Field(alias="class", description="Entity type, e.g. Disease, Gene, Drug")
    name: str
    synonyms: list[str] = Field(default_factory=list)
    symbol: Optional[str] = None  # Gene
    brand_names: list[str] = Field(default_factory=list)  # Drug
    source: Literal["extracted", "umls", "hgnc", "rxnorm", "loinc", "uniprot"] = "extracted"
    canonical_id: Optional[str] = None
    umls_id: Optional[str] = None
    hgnc_id: Optional[str] = None
    rxnorm_id: Optional[str] = None
    loinc_code: Optional[str] = None
    uniprot_id: Optional[str] = None


class EvidenceEntityRow(BaseModel):
    """Evidence entity in the bundle. id format: {paper_id}:{section}:{paragraph_idx}:{method}."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    entity_class: Literal["Evidence"] = Field(default="Evidence", alias="class")
    entity_id: Optional[str] = None  # same as id typically; for compatibility
    paper_id: str
    text_span_id: str
    text: Optional[str] = None
    confidence: float = 0.5
    extraction_method: str = "llm"
    study_type: Optional[str] = None
    eco_type: Optional[str] = None
    source: Literal["extracted"] = "extracted"


class RelationshipRow(BaseModel):
    """One relationship in the bundle. evidence_ids optional for SAME_AS."""

    model_config = ConfigDict(populate_by_name=True)

    subject: str
    predicate: str
    object_id: str = Field(alias="object", description="Object entity ID (bundle-local or canonical)")
    evidence_ids: list[str] = Field(default_factory=list)
    source_papers: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    properties: dict[str, Any] = Field(default_factory=dict)
    section: Optional[str] = None
    asserted_by: str = "llm"
    resolution: Optional[Literal["merged", "distinct"]] = None  # SAME_AS
    note: Optional[str] = None  # SAME_AS and others


# -----------------------------------------------------------------------------
# Top-level per-paper bundle
# -----------------------------------------------------------------------------


class PerPaperBundle(BaseModel):
    """Per-paper bundle: Pass 1 output and Pass 2 input. Immutable after Pass 1."""

    paper: PaperInfo
    extraction_provenance: Optional[ExtractionProvenance] = None
    entities: list[ExtractedEntityRow] = Field(default_factory=list)
    evidence_entities: list[EvidenceEntityRow] = Field(default_factory=list)
    relationships: list[RelationshipRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_bundle_dict(self) -> dict:
        """Serialize for JSON with alias 'class' used for entity type."""
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_bundle_dict(cls, data: dict) -> "PerPaperBundle":
        """Load from dict/JSON (accepts key 'class' for entity type)."""
        return cls.model_validate(data)
