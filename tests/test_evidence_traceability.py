"""Tests for evidence traceability."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from examples.medlit_schema.entity import Evidence, TextSpan
from examples.medlit_schema.base import ExtractionMethod, StudyType
from kgschema.entity import EntityStatus

def test_evidence_with_ids_validates():
    """Test that an Evidence entity with paper_id and text_span_id validates."""
    Evidence(
        entity_id="evidence:1",
        paper_id="PMC123",
        text_span_id="PMC123:abstract:p1",
        confidence=0.9,
        extraction_method=ExtractionMethod.LLM,
        study_type=StudyType.RCT,
        created_at=datetime.now(),
        name="evidence",
        source="extracted"
    )

def test_evidence_without_paper_id_fails():
    """Test that an Evidence entity without a paper_id fails."""
    with pytest.raises(ValidationError):
        Evidence(
            entity_id="evidence:1",
            text_span_id="PMC123:abstract:p1",
            confidence=0.9,
            extraction_method=ExtractionMethod.LLM,
            study_type=StudyType.RCT,
            created_at=datetime.now(),
            name="evidence",
            source="extracted"
        )

def test_evidence_canonical_id_format():
    """Test that an Evidence entity with a canonical ID format validates."""
    Evidence(
        entity_id="PMC123:abstract:p1:llm",
        paper_id="PMC123",
        text_span_id="PMC123:abstract:p1",
        confidence=0.9,
        extraction_method=ExtractionMethod.LLM,
        study_type=StudyType.RCT,
        created_at=datetime.now(),
        name="evidence",
        source="extracted"
    )

def test_conceptual_navigation():
    """Test the conceptual navigation from Relationship to Paper."""
    # This is a conceptual test to show the traceability path.
    # It does not perform actual navigation in a graph.
    from examples.medlit_schema.relationship import Treats
    from examples.medlit_schema.entity import Paper
    from examples.medlit_schema.base import TextSpanRef, SectionType

    paper = Paper(
        entity_id="PMC123",
        paper_id="PMC123",
        name="Test Paper",
        source="extracted",
        created_at=datetime.now(),
    )

    text_span_ref = TextSpanRef(
        paper_id=paper.entity_id,
        section_type=SectionType.ABSTRACT,
        paragraph_idx=1,
    )

    evidence = Evidence(
        entity_id="PMC123:abstract:1:llm",
        paper_id=paper.entity_id,
        text_span_id="PMC123:abstract:1", # This would be the ID of a TextSpan entity if it were persisted
        confidence=0.9,
        extraction_method=ExtractionMethod.LLM,
        study_type=StudyType.RCT,
        created_at=datetime.now(),
        name="evidence",
        source="extracted"
    )

    relationship = Treats(
        subject_id="drug:1",
        object_id="disease:1",
        predicate="TREATS",
        evidence_ids=[evidence.entity_id],
        created_at=datetime.now(),
    )

    # Conceptual navigation
    assert relationship.evidence_ids[0] == evidence.entity_id
    assert evidence.paper_id == paper.entity_id


# TextSpan entity tests
def test_textspan_is_canonical_only():
    """Test that TextSpan entities are always canonical (not promotable)."""
    span = TextSpan(
        entity_id="PMC123:abstract:0:100",
        paper_id="PMC123",
        section="abstract",
        start_offset=0,
        end_offset=100,
        name="test span",
        source="extracted",
        created_at=datetime.now(),
    )
    assert span.promotable is False
    assert span.status == EntityStatus.CANONICAL


def test_textspan_cannot_be_provisional():
    """Test that TextSpan cannot be created with provisional status."""
    with pytest.raises(ValidationError):
        TextSpan(
            entity_id="PMC123:abstract:0:100",
            paper_id="PMC123",
            section="abstract",
            start_offset=0,
            end_offset=100,
            name="test span",
            source="extracted",
            created_at=datetime.now(),
            status=EntityStatus.PROVISIONAL,  # Should fail
        )
    # ValidationError raised is sufficient - don't assert on message text


def test_textspan_requires_offsets():
    """Test that TextSpan requires start_offset and end_offset."""
    with pytest.raises(ValidationError):
        TextSpan(
            entity_id="PMC123:abstract:0:100",
            paper_id="PMC123",
            section="abstract",
            # Missing start_offset and end_offset
            name="test span",
            source="extracted",
            created_at=datetime.now(),
        )


def test_textspan_validates_offset_order():
    """Test that end_offset must be greater than start_offset."""
    with pytest.raises(ValidationError) as exc_info:
        TextSpan(
            entity_id="PMC123:abstract:50:50",
            paper_id="PMC123",
            section="abstract",
            start_offset=50,
            end_offset=50,  # Equal to start - should fail
            name="test span",
            source="extracted",
            created_at=datetime.now(),
        )
    # Check that end_offset is the failing field
    error_locs = [e["loc"] for e in exc_info.value.errors()]
    assert any("end_offset" in loc for loc in error_locs)

    with pytest.raises(ValidationError) as exc_info:
        TextSpan(
            entity_id="PMC123:abstract:100:50",
            paper_id="PMC123",
            section="abstract",
            start_offset=100,
            end_offset=50,  # Less than start - should fail
            name="test span",
            source="extracted",
            created_at=datetime.now(),
        )
    # Check that end_offset is the failing field
    error_locs = [e["loc"] for e in exc_info.value.errors()]
    assert any("end_offset" in loc for loc in error_locs)


def test_textspan_valid_creation():
    """Test that a valid TextSpan can be created."""
    span = TextSpan(
        entity_id="PMC123:abstract:0:150",
        paper_id="PMC123",
        section="abstract",
        start_offset=0,
        end_offset=150,
        text_content="This is the text content of the span.",
        name="Test text span",
        source="extracted",
        created_at=datetime.now(),
    )
    assert span.paper_id == "PMC123"
    assert span.section == "abstract"
    assert span.start_offset == 0
    assert span.end_offset == 150
    assert span.text_content == "This is the text content of the span."
    assert span.get_entity_type() == "text_span"
