"""Domain schema definition for the knowledge graph framework."""

from abc import ABC, abstractmethod

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity, PromotionConfig
from kgraph.relationship import BaseRelationship


class DomainSchema(ABC):
    """Abstract schema definition for a knowledge domain.

    Each domain (medical, legal, CS papers, etc.) implements this to define:
    - What entity types exist
    - What relationship types are valid
    - How canonical IDs are assigned
    - Custom validation rules
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain identifier (e.g., 'medical', 'legal', 'cs_papers')."""

    @property
    @abstractmethod
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        """Map of entity type names to their classes.

        Example: {'drug': DrugEntity, 'disease': DiseaseEntity}
        """

    @property
    @abstractmethod
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        """Map of relationship type names to their classes.

        Example: {'treats': TreatsRelationship, 'causes': CausesRelationship}
        """

    @property
    @abstractmethod
    def document_types(self) -> dict[str, type[BaseDocument]]:
        """Map of document type names to their classes.

        Example: {'journal_article': JournalArticle, 'clinical_trial': ClinicalTrial}
        """

    @property
    def promotion_config(self) -> PromotionConfig:
        """Override to customize promotion thresholds for this domain.

        Returns framework defaults if not overridden.
        """
        return PromotionConfig()

    @abstractmethod
    def validate_entity(self, entity: BaseEntity) -> bool:
        """Domain-specific entity validation.

        Called before storing an entity. Return True if valid.
        Implementations should check that the entity type is registered
        and that any domain-specific constraints are met.
        """

    @abstractmethod
    def validate_relationship(self, relationship: BaseRelationship) -> bool:
        """Domain-specific relationship validation.

        Called before storing a relationship. Return True if valid.
        Implementations should check that:
        - The predicate is valid for this domain
        - Subject/object entity types are compatible with the predicate
        """

    def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]:
        """Return valid predicates between two entity types.

        Override to enforce type-specific relationship constraints.
        Default allows any predicate registered in relationship_types.
        """
        return list(self.relationship_types.keys())
