"""Promotion policy for Sherlock Holmes domain.

Promotes provisional entities to canonical status using curated DBPedia URI mappings.
Uses the shared canonical ID helper functions for consistency with other domains.
"""

from typing import Optional

from kgraph.canonical_helpers import extract_canonical_id_from_entity
from kgraph.canonical_id import CanonicalId
from kgraph.entity import BaseEntity
from kgraph.promotion import PromotionPolicy

# Hand-curated mapping of provisional IDs to DBPedia URIs
SHERLOCK_CANONICAL_IDS = {
    "holmes:char:SherlockHolmes": "http://dbpedia.org/resource/Sherlock_Holmes",
    "holmes:char:JohnWatson": "http://dbpedia.org/resource/Dr._Watson",
    "holmes:char:MrsHudson": "http://dbpedia.org/resource/Mrs._Hudson",
    "holmes:char:IreneAdler": "http://dbpedia.org/resource/Irene_Adler",
    "holmes:loc:BakerStreet221B": "http://dbpedia.org/resource/221B_Baker_Street",
    "holmes:story:AScandalInBohemia": "http://dbpedia.org/resource/A_Scandal_in_Bohemia",
    # Add more as needed
}


class SherlockPromotionPolicy(PromotionPolicy):
    """Promotion policy for Sherlock Holmes domain using curated DBPedia mappings.

    Promotion strategy:
    1. If entity already has canonical_ids dict, use that
    2. If entity_id is already a DBPedia URI, use it directly
    3. Otherwise, look up from curated mapping
    """

    async def assign_canonical_id(self, entity: BaseEntity) -> Optional[CanonicalId]:
        """Assign canonical ID for a provisional entity.

        Args:
            entity: The provisional entity to promote.

        Returns:
            CanonicalId if available, None otherwise.
        """
        # Strategy 1: Check if entity already has canonical_ids
        canonical_id = extract_canonical_id_from_entity(entity)
        if canonical_id:
            return canonical_id

        # Strategy 2: Check if entity_id is already a DBPedia URI
        if entity.entity_id.startswith("http://dbpedia.org/resource/"):
            return CanonicalId(id=entity.entity_id, url=entity.entity_id, synonyms=(entity.name,))

        # Strategy 3: Look up from curated mapping
        canonical_id_str = SHERLOCK_CANONICAL_IDS.get(entity.entity_id)
        if canonical_id_str:
            return CanonicalId(id=canonical_id_str, url=canonical_id_str, synonyms=(entity.name,))

        return None
