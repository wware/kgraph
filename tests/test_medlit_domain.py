"""Tests for the MedlitDomain."""
from examples.medlit_schema.domain import MedlitDomain

def test_medlit_domain_instantiates():
    """Test that MedlitDomain can be instantiated."""
    domain = MedlitDomain()
    assert domain.name == "medlit"

def test_medlit_domain_entity_types():
    """Test that all entity types are registered."""
    domain = MedlitDomain()
    expected_entities = {
        "disease",
        "gene",
        "drug",
        "protein",
        "mutation",
        "symptom",
        "biomarker",
        "pathway",
        "procedure",
        "paper",
        "author",
        "clinical_trial",
        "institution",
        "hypothesis",
        "study_design",
        "statistical_method",
        "evidence_line",
        "evidence",
    }
    assert expected_entities == set(domain.entity_types.keys())

def test_medlit_domain_relationship_types():
    """Test that all relationship types are registered."""
    domain = MedlitDomain()
    expected_relationships = {
        "TREATS",
        "PREVENTS",
        "CONTRAINDICATED_FOR",
        "SIDE_EFFECT",
        "CAUSES",
        "INCREASES_RISK",
        "ASSOCIATED_WITH",
        "INTERACTS_WITH",
        "DIAGNOSED_BY",
        "PARTICIPATES_IN",
        "ENCODES",
        "BINDS_TO",
        "INHIBITS",
        "UPREGULATES",
        "DOWNREGULATES",
        "CITES",
        "STUDIED_IN",
        "AUTHORED_BY",
        "PART_OF",
        "PREDICTS",
        "REFUTES",
        "TESTED_BY",
        "SUPPORTS",
        "SUBTYPE_OF",
        "GENERATES",
    }
    assert expected_relationships == set(domain.relationship_types.keys())
