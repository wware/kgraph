"""Promotion policy for medical literature domain.

Promotes provisional entities to canonical status when they have authoritative
identifiers (UMLS, HGNC, RxNorm, UniProt) or meet usage/confidence thresholds.
"""

from typing import Optional

from kgraph.entity import BaseEntity, EntityStatus
from kgraph.logging import setup_logging
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
        logger = setup_logging()

        if entity.status != EntityStatus.PROVISIONAL:
            logger.debug(
                {
                    "message": f"Entity {entity.name} ({entity.entity_id}) is not provisional (status: {entity.status})",
                    "entity": entity,
                },
                pprint=True,
            )
            return False

        # Check confidence threshold
        confidence_ok = entity.confidence >= self.config.min_confidence
        # Note: usage_count and embedding checks are commented out in current implementation
        # Original line: return entity.usage_count >= self.config.min_usage_count and entity.confidence >= self.config.min_confidence and (not self.config.require_embedding or entity.embedding is not None)
        usage_ok = True  # Not currently checked
        embedding_ok = True  # Not currently checked

        result = confidence_ok

        logger.debug(
            {
                "message": f"Promotion check for {entity.name} ({entity.entity_id})",
                "entity": entity,
                "config": self.config,
                "confidence_check": {
                    "value": entity.confidence,
                    "threshold": self.config.min_confidence,
                    "passed": confidence_ok,
                },
                "usage_check": {
                    "value": entity.usage_count,
                    "threshold": self.config.min_usage_count,
                    "passed": usage_ok,
                    "note": "not currently checked",
                },
                "embedding_check": {
                    "present": entity.embedding is not None,
                    "required": self.config.require_embedding,
                    "passed": embedding_ok,
                    "note": "not currently checked",
                },
                "result": result,
            },
            pprint=True,
        )

        return result

    def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        """Assign canonical ID for a provisional entity.

        Args:
            entity: The provisional entity to promote.

        Returns:
            Canonical ID string if available, None otherwise.
        """
        logger = setup_logging()

        logger.debug(
            f"Attempting to assign canonical ID for {entity.name} ({entity.entity_id}, type: {entity.get_entity_type()})",
            extra={"entity": entity},
            pprint=True,
        )

        # Strategy 1: Check if entity already has canonical_ids with authoritative ID
        # Priority order: umls > hgnc > rxnorm > uniprot
        if entity.canonical_ids:
            logger.debug(
                {
                    "message": "Entity has canonical_ids dict",
                    "canonical_ids": entity.canonical_ids,
                },
                pprint=True,
            )

            # Check for UMLS (diseases, symptoms, procedures)
            if "umls" in entity.canonical_ids:
                canonical_id = entity.canonical_ids["umls"]
                logger.info(
                    {
                        "message": f"Found UMLS canonical ID for {entity.name}: {canonical_id}",
                        "entity": entity,
                        "canonical_id": canonical_id,
                        "source": "umls",
                    },
                    pprint=True,
                )
                return canonical_id

            # Check for HGNC (genes)
            if "hgnc" in entity.canonical_ids:
                canonical_id = entity.canonical_ids["hgnc"]
                logger.info(
                    {
                        "message": f"Found HGNC canonical ID for {entity.name}: {canonical_id}",
                        "entity": entity,
                        "canonical_id": canonical_id,
                        "source": "hgnc",
                    },
                    pprint=True,
                )
                return canonical_id

            # Check for RxNorm (drugs)
            if "rxnorm" in entity.canonical_ids:
                canonical_id = entity.canonical_ids["rxnorm"]
                logger.info(
                    f"Found RxNorm canonical ID for {entity.name}: {canonical_id}",
                    extra={"entity": entity, "canonical_id": canonical_id},
                    pprint=True,
                )
                return canonical_id

            # Check for UniProt (proteins)
            if "uniprot" in entity.canonical_ids:
                canonical_id = entity.canonical_ids["uniprot"]
                logger.info(
                    {
                        "message": f"Found UniProt canonical ID for {entity.name}: {canonical_id}",
                        "entity": entity,
                        "canonical_id": canonical_id,
                        "source": "uniprot",
                    },
                    pprint=True,
                )
                return canonical_id

            logger.debug(
                {
                    "message": "Entity has canonical_ids but none match authoritative sources (umls, hgnc, rxnorm, uniprot)",
                    "canonical_ids": entity.canonical_ids,
                },
                pprint=True,
            )

        # Strategy 2: Check if entity_id is already in canonical format
        entity_id = entity.entity_id
        entity_type = entity.get_entity_type()

        logger.debug(
            {
                "message": f"Checking if entity_id '{entity_id}' is in canonical format for type '{entity_type}'",
            },
            pprint=True,
        )

        # UMLS format: C followed by digits (e.g., "C0006142")
        if entity_id.startswith("C") and len(entity_id) > 1 and entity_id[1:].isdigit():
            logger.info(
                {
                    "message": f"Entity ID '{entity_id}' matches UMLS format for {entity.name}",
                    "entity": entity,
                    "canonical_id": entity_id,
                    "source": "umls_format",
                },
                pprint=True,
            )
            return entity_id

        # HGNC format: "HGNC:XXXX" or just the number
        if entity_id.startswith("HGNC:"):
            logger.info(
                {
                    "message": f"Entity ID '{entity_id}' matches HGNC format for {entity.name}",
                    "entity": entity,
                    "canonical_id": entity_id,
                    "source": "hgnc_format",
                },
                pprint=True,
            )
            return entity_id
        # If it's just a number and entity type is gene, assume HGNC
        if entity_type == "gene" and entity_id.isdigit():
            canonical_id = f"HGNC:{entity_id}"
            logger.info(
                {
                    "message": f"Entity ID '{entity_id}' inferred as HGNC for gene {entity.name}: {canonical_id}",
                    "entity": entity,
                    "canonical_id": canonical_id,
                    "source": "hgnc_inferred",
                },
                pprint=True,
            )
            return canonical_id

        # RxNorm format: "RxNorm:XXXX" or just the number
        if entity_id.startswith("RxNorm:"):
            logger.info(
                {
                    "message": f"Entity ID '{entity_id}' matches RxNorm format for {entity.name}",
                    "entity": entity,
                    "canonical_id": entity_id,
                    "source": "rxnorm_format",
                },
                pprint=True,
            )
            return entity_id
        # If it's just a number and entity type is drug, assume RxNorm
        if entity_type == "drug" and entity_id.isdigit():
            canonical_id = f"RxNorm:{entity_id}"
            logger.info(
                {
                    "message": f"Entity ID '{entity_id}' inferred as RxNorm for drug {entity.name}: {canonical_id}",
                    "entity": entity,
                    "canonical_id": canonical_id,
                    "source": "rxnorm_inferred",
                },
                pprint=True,
            )
            return canonical_id

        # UniProt format: P or Q followed by alphanumeric (e.g., "P38398", "Q9Y6K9")
        if (entity_id.startswith("P") or entity_id.startswith("Q")) and len(entity_id) == 6:
            # Check if it looks like a UniProt ID (P/Q + 5 alphanumeric)
            if entity_id[1:].isalnum():
                logger.info(
                    {
                        "message": f"Entity ID '{entity_id}' matches UniProt format for {entity.name}",
                        "entity": entity,
                        "canonical_id": entity_id,
                        "source": "uniprot_format",
                    },
                    pprint=True,
                )
                return entity_id

        # Strategy 3: No canonical ID available yet
        # Return None to wait for more evidence or external lookup
        logger.debug(
            {
                "message": f"Cannot assign canonical ID for {entity.name}",
                "entity": entity,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "reason": "No canonical_ids dict entry found, and entity_id does not match any known format",
                "note": f"External lookup would be needed (e.g., HGNC API for '{entity.name}')",
            },
            pprint=True,
        )
        return None
