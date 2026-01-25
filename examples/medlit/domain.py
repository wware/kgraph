"""Domain schema for medical literature knowledge graph."""

from kgraph.document import BaseDocument
from kgraph.domain import DomainSchema
from kgraph.entity import BaseEntity, PromotionConfig
from kgraph.relationship import BaseRelationship
from kgraph.promotion import PromotionPolicy

from .documents import JournalArticle
from .pipeline.authority_lookup import CanonicalIdLookup
from .promotion import MedLitPromotionPolicy
from .entities import (
    BiomarkerEntity,
    DiseaseEntity,
    DrugEntity,
    GeneEntity,
    PathwayEntity,
    ProcedureEntity,
    ProteinEntity,
    SymptomEntity,
)
from .relationships import MedicalClaimRelationship
from .vocab import ALL_PREDICATES, get_valid_predicates


class MedLitDomainSchema(DomainSchema):
    """Domain schema for medical literature extraction.

    Defines the vocabulary and validation rules for extracting medical knowledge
    from journal articles. Uses canonical IDs (UMLS, HGNC, RxNorm, UniProt) for
    entity identification and supports rich relationship metadata with evidence
    and provenance tracking.
    """

    @property
    def name(self) -> str:
        return "medlit"

    @property
    def entity_types(self) -> dict[str, type[BaseEntity]]:
        return {
            "disease": DiseaseEntity,
            "gene": GeneEntity,
            "drug": DrugEntity,
            "protein": ProteinEntity,
            "symptom": SymptomEntity,
            "procedure": ProcedureEntity,
            "biomarker": BiomarkerEntity,
            "pathway": PathwayEntity,
        }

    @property
    def relationship_types(self) -> dict[str, type[BaseRelationship]]:
        # Pattern A: All predicates map to the same relationship class
        # The predicate field distinguishes the relationship type
        return {predicate: MedicalClaimRelationship for predicate in ALL_PREDICATES}

    @property
    def document_types(self) -> dict[str, type[BaseDocument]]:
        return {"journal_article": JournalArticle}

    @property
    def promotion_config(self) -> PromotionConfig:
        """Medical domain promotion configuration.

        Lowered thresholds to match LLM extraction characteristics:
        - min_usage_count=1: Entities appear once per paper
        - min_confidence=0.4: LLM typically returns ~0.47 confidence
        - require_embedding=False: Don't block promotion if embeddings not ready
        """
        return PromotionConfig(
            min_usage_count=1,  # Appear in at least 1 paper
            min_confidence=0.4,  # 40% confidence threshold (LLM typically ~0.47)
            require_embedding=False,  # Don't require embeddings (we have them but don't block)
        )

    def validate_entity(self, entity: BaseEntity) -> bool:
        """Validate an entity against medical domain rules.

        Rules:
        - Entity type must be registered
        - Canonical entities should have canonical IDs in entity_id or canonical_ids
        - Provisional entities are allowed (they'll be promoted later)
        """
        if entity.get_entity_type() not in self.entity_types:
            return False

        # Canonical entities should have meaningful IDs
        if entity.status.value == "canonical":
            # Allow IDs like "C0006142" (UMLS), "HGNC:1100", "RxNorm:1187832", etc.
            if not entity.entity_id or entity.entity_id.startswith("prov:"):
                # Provisional prefix suggests this should be provisional
                return False

        return True

    def validate_relationship(self, relationship: BaseRelationship) -> bool:
        """Validate a relationship against medical domain rules.

        Rules:
        - Predicate must be registered
        - Subject and object entity types must be compatible with predicate
        - Confidence must be in valid range (enforced by BaseRelationship)
        """
        if relationship.predicate not in self.relationship_types:
            return False

        # Check if predicate is valid for the entity type pair
        # Note: We'd need to look up entity types from storage, so this is
        # a simplified check. Full validation would require entity lookups.
        # For now, we just check that the predicate is registered.
        return True

    def get_valid_predicates(self, subject_type: str, object_type: str) -> list[str]:
        """Return predicates valid between two entity types.

        Uses the vocabulary validation function to enforce domain-specific
        constraints on which relationships are semantically valid.
        """
        return get_valid_predicates(subject_type, object_type)

    def get_promotion_policy(self, lookup: CanonicalIdLookup | None = None) -> PromotionPolicy:
        """Return the promotion policy for medical literature domain.

        Uses MedLitPromotionPolicy which assigns canonical IDs based on
        authoritative medical ontologies (UMLS, HGNC, RxNorm, UniProt).

        Args:
            lookup: Optional canonical ID lookup service. If None, a new
                    instance will be created (without UMLS API key unless
                    set in environment).
        """
        return MedLitPromotionPolicy(config=self.promotion_config, lookup=lookup)
