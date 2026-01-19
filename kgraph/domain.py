"""Domain schema definition for the knowledge graph framework.

A domain schema defines the vocabulary and rules for a specific knowledge
domain (medical literature, legal documents, academic CS papers, etc.).
Each domain specifies:

- **Entity types**: The kinds of entities that can exist (drugs, diseases,
  legal cases, algorithms, etc.) and their concrete class implementations.

- **Relationship types**: Valid predicates between entities (treats, cites,
  implements, etc.) and their class implementations.

- **Document types**: Source document formats the domain processes (journal
  articles, court filings, conference papers, etc.).

- **Validation rules**: Domain-specific constraints on entities and
  relationships beyond basic type checking.

- **Promotion configuration**: Thresholds for promoting provisional entities
  to canonical status, which may vary by domain based on data quality and
  external authority availability.

The domain schema serves as the central configuration point for domain-specific
behavior, allowing the core knowledge graph framework to remain domain-agnostic
while supporting specialized use cases.

Example usage:
    ```python
    class MedicalDomainSchema(DomainSchema):
        @property
        def name(self) -> str:
            return "medical"

        @property
        def entity_types(self) -> dict[str, type[BaseEntity]]:
            return {"drug": DrugEntity, "disease": DiseaseEntity, "gene": GeneEntity}

        @property
        def relationship_types(self) -> dict[str, type[BaseRelationship]]:
            return {"treats": TreatsRelationship, "causes": CausesRelationship}
        # ... etc
    ```
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity, PromotionConfig
from kgraph.relationship import BaseRelationship


class Provenance(BaseModel, frozen=True):
    document_id: str
    source_uri: str | None = None
    section: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class Evidence(BaseModel, frozen=True):
    kind: str
    source_documents: tuple[str, ...] = Field(min_length=1)
    primary: Provenance | None = None
    mentions: tuple[Provenance, ...] = ()
    notes: dict[str, object] = Field(default_factory=dict)


class DomainSchema(ABC):
    """Abstract schema definition for a knowledge domain.

    Each domain (medical, legal, CS papers, etc.) implements this interface
    to define its vocabulary of types and validation rules. The schema is
    used throughout the ingestion pipeline to:

    - Validate extracted entities and relationships before storage
    - Configure entity promotion thresholds
    - Determine valid predicates between entity type pairs
    - Deserialize domain-specific entity/relationship subclasses

    Implementations should be stateless and thread-safe, as the same schema
    instance may be used across multiple concurrent ingestion operations.

    Required methods to implement:
        - name: Unique domain identifier
        - entity_types: Registry of entity type names to classes
        - relationship_types: Registry of predicate names to classes
        - document_types: Registry of document format names to classes
        - validate_entity: Domain-specific entity validation
        - validate_relationship: Domain-specific relationship validation

    Optional methods to override:
        - promotion_config: Customize promotion thresholds
        - get_valid_predicates: Restrict predicates by entity type pair
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this domain.

        The domain name is used for:
            - Namespacing entities in multi-domain deployments
            - Selecting the correct deserializer for stored data
            - Logging and debugging

        Returns:
            A short, lowercase identifier (e.g., 'medical', 'legal', 'cs_papers').
            Should contain only alphanumeric characters and underscores.
        """

    @property
    @abstractmethod
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        """Return the registry of entity types for this domain.

        Maps type name strings to concrete BaseEntity subclasses. The type
        names are used in entity extraction and must match the values
        returned by each entity's `get_entity_type()` method.

        Returns:
            Dictionary mapping entity type names to their implementing classes.

        Example:
            ```python
            return {
                'drug': DrugEntity,
                'disease': DiseaseEntity,
                'gene': GeneEntity,
            }
            ```
        """

    @property
    @abstractmethod
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        """Return the registry of relationship types for this domain.

        Maps predicate name strings to concrete BaseRelationship subclasses.
        The predicate names define the vocabulary of edges in the knowledge
        graph and must match the values used in relationship extraction.

        Returns:
            Dictionary mapping predicate names to their implementing classes.

        Example:
            ```python
            return {
                'treats': TreatsRelationship,
                'causes': CausesRelationship,
                'interacts_with': InteractionRelationship,
            }
            ```
        """

    @property
    @abstractmethod
    def document_types(self) -> dict[str, type[BaseDocument]]:
        """Return the registry of document types for this domain.

        Maps document format names to concrete BaseDocument subclasses.
        Different document types may have different structures and metadata
        fields relevant to the domain.

        Returns:
            Dictionary mapping document type names to their implementing classes.

        Example:
            ```python
            return {
                'journal_article': JournalArticle,
                'clinical_trial': ClinicalTrialDocument,
                'drug_label': DrugLabelDocument,
            }
            ```
        """

    @property
    def promotion_config(self) -> PromotionConfig:
        """Return the configuration for promoting provisional entities.

        Promotion configuration controls when provisional entities (newly
        discovered mentions without canonical IDs) are promoted to canonical
        status. The thresholds should be tuned based on:

        - Data quality: Noisy extraction requires higher thresholds
        - External authority availability: Domains with good authorities
          (UMLS, DBPedia) can use higher confidence requirements
        - Entity importance: Critical domains may require more evidence

        Override this property to customize thresholds for your domain.
        The default configuration uses framework defaults.

        Returns:
            PromotionConfig with min_usage_count, min_confidence, and
            require_embedding settings appropriate for this domain.
        """
        return PromotionConfig()

    @abstractmethod
    def validate_entity(self, entity: BaseEntity) -> bool:
        """Validate an entity against domain-specific rules.

        Called by the ingestion pipeline before storing an entity. Use this
        to enforce constraints beyond basic type checking, such as:

        - Required fields for specific entity types
        - Value constraints (e.g., confidence thresholds)
        - Cross-field validation (e.g., canonical entities must have IDs)

        Args:
            entity: The entity to validate.

        Returns:
            True if the entity is valid for this domain, False otherwise.

        Note:
            At minimum, implementations should verify that the entity's type
            (from `get_entity_type()`) is registered in `entity_types`.
        """

    @abstractmethod
    def validate_relationship(self, relationship: BaseRelationship) -> bool:
        """Validate a relationship against domain-specific rules.

        Called by the ingestion pipeline before storing a relationship.
        Use this to enforce constraints such as:

        - Predicate must be registered in `relationship_types`
        - Subject/object entity types must be compatible with the predicate
        - Confidence must meet domain thresholds

        Args:
            relationship: The relationship to validate.

        Returns:
            True if the relationship is valid for this domain, False otherwise.

        Note:
            At minimum, implementations should verify that the relationship's
            predicate is registered in `relationship_types`.
        """

    def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]:
        """Return predicates valid between two entity types.

        Override this method to enforce type-specific relationship constraints.
        For example, in a medical domain, "treats" might only be valid from
        Drug to Disease, not from Disease to Drug.

        The default implementation allows any predicate registered in
        `relationship_types` between any entity type pair.

        Args:
            subject_type: The entity type of the relationship subject.
            object_type: The entity type of the relationship object.

        Returns:
            List of predicate names that are valid for this entity type pair.
            Returns an empty list if no predicates are valid.

        Example:
            ```python
            def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]:
                if subject_type == "drug" and object_type == "disease":
                    return ["treats", "prevents", "exacerbates"]
                if subject_type == "gene" and object_type == "disease":
                    return ["associated_with", "causes"]
                return []  # No other combinations allowed
            ```
        """
        return list(self.relationship_types.keys())

    @property
    def evidence_model(self) -> type[Evidence]:
        """Return the domain's version of Evidence

        The domain can add stuff to the evidence:
            - A predicate might be supported (or counter-argued) by lab data or test results
            - Or by whether some paper from the past was retracted or not

        Returns:
            A type that is, or is a subclass of, the Evidence type
        """
        return Evidence

    @property
    def provenance_model(self) -> type[Provenance]:
        """Return the domain's version of Provenance

        The domain can add stuff to the provenance, much like Evidence

        Returns:
            A type that is, or is a subclass of, the Provenance type
        """
        return Provenance
