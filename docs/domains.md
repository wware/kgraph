# Implementing a Domain

Each knowledge domain (medical, legal, CS papers, etc.) defines its own entity types, relationship types, and validation rules by implementing `DomainSchema`.

## Step 1: Define Entity Types

Create entity classes by extending `BaseEntity`:

```python
from datetime import datetime
from kgschema.entity import BaseEntity, EntityStatus

class PersonEntity(BaseEntity):
    """A person in the legal domain."""

    # Additional domain-specific fields can be added
    bar_number: str | None = None

    def get_entity_type(self) -> str:
        return "person"


class CaseEntity(BaseEntity):
    """A legal case."""

    court: str | None = None
    year: int | None = None

    def get_entity_type(self) -> str:
        return "case"
```

## Step 2: Define Relationship Types

Create relationship classes by extending `BaseRelationship`:

```python
from kgschema.relationship import BaseRelationship

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
from kgschema.document import BaseDocument

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
from kgschema.domain import DomainSchema
from kgschema.entity import PromotionConfig, BaseEntity
from kgschema.relationship import BaseRelationship
from kgschema.document import BaseDocument

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

    def get_promotion_policy(self, lookup=None) -> PromotionPolicy:
        """Return the promotion policy for this domain.

        Args:
            lookup: Optional canonical ID lookup service (CanonicalIdLookupInterface).
                   Domains that support external lookups can use this to pass the
                   service to the policy.
        """
        from kgraph.promotion import PromotionPolicy
        # Return a domain-specific promotion policy
        # See examples/medlit/promotion.py or examples/sherlock/promotion.py
        raise NotImplementedError("Subclass must implement get_promotion_policy")

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

## Step 5: Implement Promotion Policy

Implement a promotion policy that assigns canonical IDs to entities:

```python
from kgraph import PromotionPolicy, CanonicalId
from kgraph.canonical_id import (
    CanonicalIdLookupInterface,
    check_entity_id_format,
    extract_canonical_id_from_entity,
)

class MyPromotionPolicy(PromotionPolicy):
    def __init__(self, config, lookup: CanonicalIdLookupInterface | None = None):
        super().__init__(config)
        self.lookup = lookup  # Optional lookup service

    async def assign_canonical_id(self, entity: BaseEntity) -> CanonicalId | None:
        # Strategy 1: Check entity.canonical_ids
        cid = extract_canonical_id_from_entity(entity, priority_sources=["my_source"])
        if cid:
            return cid

        # Strategy 2: Check entity_id format
        format_patterns = {"person": ("my_prefix:",)}
        cid = check_entity_id_format(entity, format_patterns)
        if cid:
            return cid

        # Strategy 3: External lookup (if available)
        if self.lookup:
            return await self.lookup.lookup(entity.name, entity.get_entity_type())

        return None
```

Then implement `get_promotion_policy` in your domain:

```python
def get_promotion_policy(self, lookup=None) -> PromotionPolicy:
    return MyPromotionPolicy(self.promotion_config, lookup=lookup)
```

See [Canonical IDs](canonical_ids.md) for more details on the canonical ID system.

## Step 6: Implement Pipeline Components

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
