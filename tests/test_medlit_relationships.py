"""Tests for medlit relationship validation."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from examples.medlit_schema.relationship import Treats, AuthoredBy, AssociatedWith, PartOf

def test_treats_with_evidence_validates():
    """Test that a Treats relationship with evidence validates."""
    Treats(
        subject_id="drug:1",
        object_id="disease:1",
        predicate="TREATS",
        evidence_ids=["evidence:1"],
        created_at=datetime.now(),
    )

def test_treats_without_evidence_fails():
    """Test that a Treats relationship without evidence fails."""
    with pytest.raises(ValidationError):
        Treats(
            subject_id="drug:1",
            object_id="disease:1",
            predicate="TREATS",
            evidence_ids=[],
            created_at=datetime.now(),
        )

def test_bibliographic_relationship_without_evidence_validates():
    """Test that a bibliographic relationship without evidence validates."""
    AuthoredBy(
        subject_id="paper:1",
        object_id="author:1",
        predicate="AUTHORED_BY",
        created_at=datetime.now(),
    )

def test_associated_with_with_evidence_validates():
    """Test that an AssociatedWith relationship with evidence validates."""
    AssociatedWith(
        subject_id="disease:1",
        object_id="disease:2",
        predicate="ASSOCIATED_WITH",
        evidence_ids=["evidence:1"],
        created_at=datetime.now(),
    )

def test_part_of_without_evidence_validates():
    """Test that a PartOf relationship without evidence validates."""
    PartOf(
        subject_id="paper:1",
        object_id="clinical_trial:1",
        predicate="PART_OF",
        created_at=datetime.now(),
    )
