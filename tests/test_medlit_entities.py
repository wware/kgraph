"""Tests for medlit entity validation."""
import pytest
from datetime import datetime
from pydantic import ValidationError
from examples.medlit_schema.entity import Disease, Gene, Drug, Protein, Procedure, Institution, Evidence
from kgschema.entity import EntityStatus

def test_disease_with_umls_id_validates():
    """Test that a Disease entity with a UMLS ID validates."""
    Disease(
        entity_id="UMLS:C0006142",
        name="Breast Cancer",
        source="umls",
        umls_id="C0006142",
        created_at=datetime.now(),
    )

def test_gene_with_hgnc_id_validates():
    """Test that a Gene entity with an HGNC ID validates."""
    Gene(
        entity_id="HGNC:1100",
        name="BRCA1",
        source="hgnc",
        hgnc_id="HGNC:1100",
        created_at=datetime.now(),
    )

def test_drug_with_rxnorm_id_validates():
    """Test that a Drug entity with an RxNorm ID validates."""
    Drug(
        entity_id="RxNorm:1187832",
        name="Olaparib",
        source="rxnorm",
        rxnorm_id="1187832",
        created_at=datetime.now(),
    )

def test_protein_with_uniprot_id_validates():
    """Test that a Protein entity with a UniProt ID validates."""
    Protein(
        entity_id="UniProt:P38398",
        name="BRCA1",
        source="uniprot",
        uniprot_id="P38398",
        created_at=datetime.now(),
    )

def test_procedure_validates():
    """Test that a Procedure entity validates."""
    Procedure(
        entity_id="procedure-123",
        name="Biopsy",
        source="extracted",
        created_at=datetime.now(),
    )

def test_institution_validates():
    """Test that an Institution entity validates."""
    Institution(
        entity_id="institution-123",
        name="Mayo Clinic",
        source="extracted",
        created_at=datetime.now(),
    )

def test_provisional_entity_validates():
    """Test that a provisional entity (no ontology ID) validates."""
    Disease(
        entity_id="provisional-123",
        name="some disease",
        source="extracted",
        created_at=datetime.now(),
    )

def test_canonical_entity_without_ontology_id_fails():
    """Test that a canonical entity without an ontology ID fails."""
    with pytest.raises(ValidationError):
        Disease(
            entity_id="UMLS:C0006142",
            name="Breast Cancer",
            source="umls",
            created_at=datetime.now(),
        )

def test_evidence_cannot_be_provisional():
    """Test that Evidence entities cannot be created with PROVISIONAL status."""
    with pytest.raises(ValueError, match="Entities that are not promotable must be created with CANONICAL status."):
        Evidence(
            entity_id="PMC999888:results:3:llm",
            name="Evidence from Olaparib RCT results",
            paper_id="PMC999888",
            text_span_id="PMC999888:results:3",
            confidence=0.92,
            extraction_method="llm", # Assuming this is a valid ExtractionMethod value
            study_type="rct", # Assuming this is a valid StudyType value
            source="extracted",
            created_at=datetime.now(),
            promotable=False,
            status=EntityStatus.PROVISIONAL,
        )
