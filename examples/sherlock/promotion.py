from kgraph.promotion import PromotionPolicy
from kgraph.entity import BaseEntity
from typing import Optional

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
    """Promotion policy for Sherlock Holmes domain using curated DBPedia mappings."""

    def assign_canonical_id(self, entity: BaseEntity) -> Optional[str]:
        """Look up canonical ID from curated mapping."""
        return SHERLOCK_CANONICAL_IDS.get(entity.entity_id)
