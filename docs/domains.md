# Implementing a Domain

Each knowledge domain (medical, legal, CS papers, etc.) defines its own entity types, relationship types, and validation rules by implementing `DomainSchema`.

## Step 1: Define Entity Types

Create entity classes by extending `BaseEntity`:

```python
from datetime import datetime
from kgraph import BaseEntity, EntityStatus

class PersonEntity(BaseEntity):
    """A person in the legal domain."""

    # Additional domain-specific fields can be added
    bar_number: str | None = None

    def get_entity_type(self) -> str:
        return "person"

    def get_canonical_id_source(self) -> str | None:
        if self.status == EntityStatus.CANONICAL:
            return "bar_registry"
        return None


class CaseEntity(BaseEntity):
    """A legal case."""

    court: str | None = None
    year: int | None = None

    def get_entity_type(self) -> str:
        return "case"

    def get_canonical_id_source(self) -> str | None:
        if self.status == EntityStatus.CANONICAL:
            return "case_law_db"
        return None
```

## Step 2: Define Relationship Types

Create relationship classes by extending `BaseRelationship`:

```python
from kgraph import BaseRelationship

class CitesRelationship(BaseRelationship):
    """One case cites another."""

    def get_edge_type(self) -> str:
        return "cites"


class RepresentsRelationship(BaseRelationship):
    """An attorney represents a party."""

    def get_edge_type(self) -> str:
        return "represents"
```

## Step 3: Define Document Types

Create document classes by extending `BaseDocument`:

```python
from kgraph.document import BaseDocument

class CaseOpinion(BaseDocument):
    """A court opinion document."""

    court: str | None = None
    judge: str | None = None

    def get_document_type(self) -> str:
        return "case_opinion"

    def get_sections(self) -> list[tuple[str, str]]:
        # Parse document into sections
        # Real implementation would parse the opinion structure
        return [
            ("syllabus", "..."),
            ("opinion", self.content),
            ("dissent", "..."),
        ]
```

## Step 4: Implement Domain Schema

Bring everything together in a `DomainSchema`:

```python
from kgraph import DomainSchema, PromotionConfig, BaseEntity, BaseRelationship
from kgraph.document import BaseDocument

class LegalDomain(DomainSchema):

    @property
    def name(self) -> str:
        return "legal"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {
            "person": PersonEntity,
            "case": CaseEntity,
        }

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        return {
            "cites": CitesRelationship,
            "represents": RepresentsRelationship,
        }

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        return {
            "case_opinion": CaseOpinion,
        }

    @property
    def canonical_id_sources(self) -> dict[str, str]:
        return {
            "person": "bar_registry",
            "case": "case_law_db",
        }

    @property
    def promotion_config(self) -> PromotionConfig:
        # Legal domain: require more appearances before promotion
        return PromotionConfig(
            min_usage_count=5,
            min_confidence=0.85,
            require_embedding=True,
        )

    def validate_entity(self, entity: BaseEntity) -> bool:
        # Check entity type is registered
        if entity.get_entity_type() not in self.entity_types:
            return False
        # Add domain-specific validation
        if isinstance(entity, CaseEntity):
            # Cases should have a year if canonical
            if entity.status == EntityStatus.CANONICAL and entity.year is None:
                return False
        return True

    def validate_relationship(self, rel: BaseRelationship) -> bool:
        # Check predicate is registered
        return rel.predicate in self.relationship_types

    def get_valid_predicates(
        self, subject_type: str, object_type: str
    ) -> list[str]:
        # Define which predicates are valid between entity types
        valid = {
            ("case", "case"): ["cites", "overrules", "distinguishes"],
            ("person", "case"): ["represents", "authored"],
        }
        return valid.get((subject_type, object_type), [])
```

## Step 5: Implement Pipeline Components

For a working system, implement the pipeline interfaces:

```python
from kgraph.pipeline import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    RelationshipExtractorInterface,
)
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
```

See [Pipeline Components](pipeline.md) for details on implementing these interfaces.

## Domain-Specific Metadata

Use the `metadata` field on entities and relationships to store domain-specific data that doesn't fit the base schema:

```python
# Entity with domain metadata
entity = PersonEntity(
    entity_id="...",
    status=EntityStatus.PROVISIONAL,
    name="Jane Doe",
    created_at=datetime.now(timezone.utc),
    source="case_parser",
    metadata={
        "firm": "Smith & Associates",
        "practice_areas": ["corporate", "securities"],
    },
)

# Relationship with domain metadata
rel = CitesRelationship(
    subject_id="case-123",
    predicate="cites",
    object_id="case-456",
    created_at=datetime.now(timezone.utc),
    metadata={
        "citation_type": "positive",
        "page_reference": "123 F.3d at 456",
    },
)
```
