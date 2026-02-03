"""Tests for the Paper model."""
from datetime import datetime
from examples.medlit_schema.entity import Paper, PaperMetadata
from examples.medlit_schema.base import ExtractionProvenance, ModelInfo

def test_paper_with_full_metadata_validates():
    """Test that a Paper with full metadata validates."""
    Paper(
        entity_id="PMC123",
        paper_id="PMC123",
        pmid="12345",
        doi="10.1234/5678",
        title="Test Paper",
        abstract="This is a test paper.",
        authors=["Author One", "Author Two"],
                    publication_date=datetime.now(),
                    journal="Journal of Testing",
                    paper_metadata=PaperMetadata(
                        study_type="RCT",
                        sample_size=100,            study_population="Test Population",
            primary_outcome="Test Outcome",
            clinical_phase="Phase 3",
            mesh_terms=["term1", "term2"],
        ),
        extraction_provenance=ExtractionProvenance(
            model_info=ModelInfo(name="test_model", version="1.0")
        ),
        created_at=datetime.now(),
        name="Test Paper",
        source="extracted",
    )

def test_papermetadata_with_study_type_validates():
    """Test that PaperMetadata with a study_type validates."""
    PaperMetadata(study_type="Observational")

def test_extractionprovenance_serializes_correctly():
    """Test that ExtractionProvenance serializes correctly."""
    provenance = ExtractionProvenance(
        model_info=ModelInfo(name="test_model", version="1.0")
    )
    assert provenance.model_dump_json() == '{"extraction_pipeline":null,"models":{},"prompt":null,"execution":null,"entity_resolution":null,"model_info":{"name":"test_model","version":"1.0"}}'
