"""Tests for automatic subject/object swapping when LLM gets the order wrong."""

import pytest
from datetime import datetime, timezone

from kgschema.domain import DomainSchema, PredicateConstraint, ValidationIssue
from kgschema.entity import BaseEntity, EntityStatus
from kgschema.relationship import BaseRelationship


class DrugEntity(BaseEntity):
    """Drug entity for testing."""

    def get_entity_type(self) -> str:
        return "drug"


class DiseaseEntity(BaseEntity):
    """Disease entity for testing."""

    def get_entity_type(self) -> str:
        return "disease"


class TreatsRelationship(BaseRelationship):
    """Treats relationship for testing."""

    def get_edge_type(self) -> str:
        return "treats"


class TestDomainSchema(DomainSchema):
    """Test domain schema with predicate constraints."""

    @property
    def name(self) -> str:
        return "test_domain"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {
            "drug": DrugEntity,
            "disease": DiseaseEntity,
        }

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {"treats": TreatsRelationship}

    @property
    def predicate_constraints(self) -> dict[str, PredicateConstraint]:
        return {
            "treats": PredicateConstraint(
                subject_types={"drug"},
                object_types={"disease"},
            )
        }

    @property
    def document_types(self) -> dict[str, type]:
        return {}

    def validate_entity(self, entity: BaseEntity) -> list[ValidationIssue]:
        """Validate entity is of a registered type."""
        entity_type = entity.get_entity_type()
        if entity_type not in self.entity_types:
            return [ValidationIssue(
                field="entity_type",
                message=f"Unknown entity type: {entity_type}",
                value=entity_type,
                code="UNKNOWN_TYPE",
            )]
        return []

    def get_promotion_policy(self, lookup=None):
        """Not needed for these tests."""
        return None


@pytest.fixture
def domain():
    """Create test domain schema."""
    return TestDomainSchema()


@pytest.fixture
def drug_entity():
    """Create a drug entity."""
    return DrugEntity(
        entity_id="drug:aspirin",
        name="Aspirin",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        canonical_ids={"test": "drug:aspirin"},
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )


@pytest.fixture
def disease_entity():
    """Create a disease entity."""
    return DiseaseEntity(
        entity_id="disease:headache",
        name="Headache",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        canonical_ids={"test": "disease:headache"},
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )


@pytest.mark.asyncio
async def test_validate_correct_order(domain, drug_entity, disease_entity):
    """Test that correctly ordered relationship passes validation."""
    rel = TreatsRelationship(
        subject_id=drug_entity.entity_id,  # drug
        predicate="treats",
        object_id=disease_entity.entity_id,  # disease
        confidence=0.9,
        source_documents=("doc1",),
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )

    # Add entities to mock storage
    from kgraph.storage.memory import InMemoryEntityStorage

    storage = InMemoryEntityStorage()
    await storage.add(drug_entity)
    await storage.add(disease_entity)

    # Should pass validation
    assert await domain.validate_relationship(rel, entity_storage=storage)


@pytest.mark.asyncio
async def test_validate_reversed_order_detected(domain, drug_entity, disease_entity):
    """Test that reversed relationship is detected and rejected with helpful message."""
    rel = TreatsRelationship(
        subject_id=disease_entity.entity_id,  # disease (WRONG - should be object)
        predicate="treats",
        object_id=drug_entity.entity_id,  # drug (WRONG - should be subject)
        confidence=0.9,
        source_documents=("doc1",),
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )

    # Add entities to mock storage
    from kgraph.storage.memory import InMemoryEntityStorage

    storage = InMemoryEntityStorage()
    await storage.add(drug_entity)
    await storage.add(disease_entity)

    # Should fail validation (relationship is reversed)
    # The validation will detect that swapping would fix it
    assert not await domain.validate_relationship(rel, entity_storage=storage)


def test_should_swap_detection():
    """Test the swap detection logic in the medlit extractor."""
    from examples.medlit.domain import MedLitDomainSchema
    from examples.medlit.pipeline.relationships import MedLitRelationshipExtractor

    domain = MedLitDomainSchema()
    extractor = MedLitRelationshipExtractor(domain=domain)

    # Create test entities
    drug = DrugEntity(
        entity_id="drug:test",
        name="Test Drug",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        canonical_ids={"test": "drug:test"},
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )

    disease = DiseaseEntity(
        entity_id="disease:test",
        name="Test Disease",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        canonical_ids={"test": "disease:test"},
        created_at=datetime.now(timezone.utc),
        last_updated=None,
    )

    # Correct order: drug treats disease - no swap needed
    # pylint: disable=protected-access
    assert not extractor._should_swap_subject_object("treats", drug, disease)

    # Reversed order: disease treats drug - should detect swap is needed
    # pylint: disable=protected-access
    assert extractor._should_swap_subject_object("treats", disease, drug)


def test_evidence_contains_both_entities_both_present():
    """Evidence containing both subject and object is accepted."""
    from examples.medlit.pipeline.relationships import _evidence_contains_both_entities

    ok, drop_reason, detail = _evidence_contains_both_entities(
        evidence="Aspirin treats headache in this study.",
        subject_name="Aspirin",
        object_name="headache",
        subject_entity=None,
        object_entity=None,
    )
    assert ok is True
    assert drop_reason is None
    assert detail["subject_in_evidence"] is True
    assert detail["object_in_evidence"] is True


def test_evidence_contains_both_entities_missing_subject():
    """Evidence missing subject is rejected with evidence_missing_subject."""
    from examples.medlit.pipeline.relationships import _evidence_contains_both_entities

    ok, drop_reason, detail = _evidence_contains_both_entities(
        evidence="headache was improved.",
        subject_name="Aspirin",
        object_name="headache",
        subject_entity=None,
        object_entity=None,
    )
    assert ok is False
    assert drop_reason == "evidence_missing_subject"
    assert detail["subject_in_evidence"] is False
    assert detail["object_in_evidence"] is True


def test_evidence_contains_both_entities_empty_evidence():
    """Empty evidence is rejected with evidence_empty."""
    from examples.medlit.pipeline.relationships import _evidence_contains_both_entities

    ok, drop_reason, detail = _evidence_contains_both_entities(
        evidence="",
        subject_name="Aspirin",
        object_name="headache",
        subject_entity=None,
        object_entity=None,
    )
    assert ok is False
    assert drop_reason == "evidence_empty"
    assert detail["subject_in_evidence"] is False
    assert detail["object_in_evidence"] is False


def test_evidence_contains_both_entities_synonym_match():
    """Entity synonym appearing in evidence counts as match."""
    from examples.medlit.pipeline.relationships import _evidence_contains_both_entities

    drug = DrugEntity(
        entity_id="drug:1",
        name="Acetylsalicylic acid",
        status=EntityStatus.CANONICAL,
        confidence=1.0,
        usage_count=1,
        source="test",
        canonical_ids={},
        created_at=datetime.now(timezone.utc),
        last_updated=None,
        synonyms=("Aspirin",),
    )

    ok, drop_reason, _ = _evidence_contains_both_entities(
        evidence="Aspirin reduces pain.",
        subject_name="Acetylsalicylic acid",
        object_name="pain",
        subject_entity=drug,
        object_entity=None,
    )
    assert ok is True
    assert drop_reason is None
