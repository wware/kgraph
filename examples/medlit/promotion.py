"""Promotion policy for medical literature domain.

Promotes provisional entities to canonical status when they have authoritative
identifiers (UMLS, HGNC, RxNorm, UniProt) or meet usage/confidence thresholds.
"""

from typing import Optional

from kgraph.entity import BaseEntity, EntityStatus
from kgraph.logging import setup_logging
from kgraph.promotion import PromotionPolicy

from .pipeline.authority_lookup import CanonicalIdLookup


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
    3. Otherwise, look up canonical ID from authority APIs (UMLS, HGNC, RxNorm, UniProt)
    """

    def __init__(self, config, lookup: Optional[CanonicalIdLookup] = None):
        """Initialize promotion policy.

        Args:
            config: Promotion configuration with thresholds.
            lookup: Optional canonical ID lookup service. If None, will create
                    a new instance (without UMLS API key unless set in env).
        """
        super().__init__(config)
        self.lookup = lookup or CanonicalIdLookup()

    def should_promote(self, entity: BaseEntity) -> bool:
        """Check if entity meets promotion thresholds.

        Force-promote rules (bypass standard thresholds):
        - If confidence >= 0.7, ignore usage count requirement
        - If canonical ID is found (checked in run_promotion), promote regardless

        Standard thresholds:
        - usage_count >= min_usage_count (default: 1)
        - confidence >= min_confidence (default: 0.4)
        - embedding required only if require_embedding=True (default: False)
        """
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

        # Force-promote: High confidence bypasses usage requirement
        if entity.confidence >= 0.7:
            logger.debug(
                {
                    "message": f"Force-promoting {entity.name} - high confidence ({entity.confidence} >= 0.7)",
                    "entity": entity,
                    "reason": "high_confidence",
                },
                pprint=True,
            )
            return True

        # Standard threshold checks
        confidence_ok = entity.confidence >= self.config.min_confidence
        usage_ok = entity.usage_count >= self.config.min_usage_count
        embedding_ok = not self.config.require_embedding or entity.embedding is not None

        result = confidence_ok and usage_ok and embedding_ok

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
                },
                "embedding_check": {
                    "present": entity.embedding is not None,
                    "required": self.config.require_embedding,
                    "passed": embedding_ok,
                },
                "result": result,
            },
            pprint=True,
        )

        return result

    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        """Assign canonical ID for a provisional entity.

        Args:
            entity: The provisional entity to promote.

        Returns:
            Canonical ID string if available, None otherwise.
        """
        logger = setup_logging()

        logger.debug(
            {
                "message": f"Attempting to assign canonical ID for {entity.name}",
                "entity": entity,
                "entity_id": entity.entity_id,
                "entity_type": entity.get_entity_type(),
            },
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
                    {
                        "message": f"Found RxNorm canonical ID for {entity.name}: {canonical_id}",
                        "entity": entity,
                        "canonical_id": canonical_id,
                        "source": "rxnorm",
                    },
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

        # Strategy 3: Look up from authority APIs
        logger.debug(
            {
                "message": f"Attempting external lookup for canonical ID for {entity.name}",
                "entity": entity,
                "entity_id": entity_id,
                "entity_type": entity_type,
            },
            pprint=True,
        )

        lookup_result = await self.lookup.lookup_canonical_id(term=entity.name, entity_type=entity_type)

        if lookup_result:
            logger.info(
                {
                    "message": f"Found canonical ID via external lookup for {entity.name}: {lookup_result}",
                    "entity": entity,
                    "canonical_id": lookup_result,
                    "source": "external_api",
                },
                pprint=True,
            )
            return lookup_result

        # No canonical ID found via any strategy
        logger.debug(
            {
                "message": f"Cannot assign canonical ID for {entity.name}",
                "entity": entity,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "reason": "No canonical_ids dict entry found, entity_id does not match any known format, and external lookup returned None",
            },
            pprint=True,
        )
        return None
