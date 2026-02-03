"""Tests for evidence traceability."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from examples.medlit_schema.entity import Evidence
from examples.medlit_schema.base import ExtractionMethod, StudyType

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
    from examples.medlit_schema.base import TextSpan, SectionType

    paper = Paper(
        entity_id="PMC123",
        paper_id="PMC123",
        name="Test Paper",
        source="extracted",
        created_at=datetime.now(),
    )

    text_span = TextSpan(
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
