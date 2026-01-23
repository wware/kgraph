"""Promotion policy for medical literature domain.

Promotes provisional entities to canonical status when they have authoritative
identifiers (UMLS, HGNC, RxNorm, UniProt) or meet usage/confidence thresholds.
"""

from typing import Optional

from kgraph.entity import BaseEntity, EntityStatus
from kgraph.promotion import PromotionPolicy


class MedLitPromotionPolicy(PromotionPolicy):
    """Promotion policy for medical literature domain.

    Assigns canonical IDs based on authoritative medical ontologies:
    - Diseases: UMLS IDs (e.g., "C0006142")
    - Genes: HGNC IDs (e.g., "HGNC:1100")
    - Drugs: RxNorm IDs (e.g., "RxNorm:1187832")
    - Proteins: UniProt IDs (e.g., "P38398")

    Promotion strategy:
    1. If entity already has canonical_id in canonical_ids dict, use that
    2. If entity_id is already a canonical ID format, use it directly
    3. Otherwise, return None (wait for more evidence or external lookup)
    """

    def should_promote(self, entity: BaseEntity) -> bool:
        """Check if entity meets promotion thresholds."""
        if entity.status != EntityStatus.PROVISIONAL:
            return False
        # return entity.usage_count >= self.config.min_usage_count and entity.confidence >= self.config.min_confidence and (not self.config.require_embedding or entity.embedding is not None)
        return entity.confidence >= self.config.min_confidence

    def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        """Assign canonical ID for a provisional entity.

        Args:
            entity: The provisional entity to promote.

        Returns:
            Canonical ID string if available, None otherwise.
        """
        # Strategy 1: Check if entity already has canonical_ids with authoritative ID
        # Priority order: umls > hgnc > rxnorm > uniprot
        if entity.canonical_ids:
            # Check for UMLS (diseases, symptoms, procedures)
            if "umls" in entity.canonical_ids:
                return entity.canonical_ids["umls"]
            # Check for HGNC (genes)
            if "hgnc" in entity.canonical_ids:
                return entity.canonical_ids["hgnc"]
            # Check for RxNorm (drugs)
            if "rxnorm" in entity.canonical_ids:
                return entity.canonical_ids["rxnorm"]
            # Check for UniProt (proteins)
            if "uniprot" in entity.canonical_ids:
                return entity.canonical_ids["uniprot"]

        # Strategy 2: Check if entity_id is already in canonical format
        entity_id = entity.entity_id

        # UMLS format: C followed by digits (e.g., "C0006142")
        if entity_id.startswith("C") and len(entity_id) > 1 and entity_id[1:].isdigit():
            return entity_id

        # HGNC format: "HGNC:XXXX" or just the number
        if entity_id.startswith("HGNC:"):
            return entity_id
        # If it's just a number and entity type is gene, assume HGNC
        if entity.get_entity_type() == "gene" and entity_id.isdigit():
            return f"HGNC:{entity_id}"

        # RxNorm format: "RxNorm:XXXX" or just the number
        if entity_id.startswith("RxNorm:"):
            return entity_id
        # If it's just a number and entity type is drug, assume RxNorm
        if entity.get_entity_type() == "drug" and entity_id.isdigit():
            return f"RxNorm:{entity_id}"

        # UniProt format: P or Q followed by alphanumeric (e.g., "P38398", "Q9Y6K9")
        if (entity_id.startswith("P") or entity_id.startswith("Q")) and len(entity_id) == 6:
            # Check if it looks like a UniProt ID (P/Q + 5 alphanumeric)
            if entity_id[1:].isalnum():
                return entity_id

        # Strategy 3: No canonical ID available yet
        # Return None to wait for more evidence or external lookup
        return None
