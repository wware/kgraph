# examples/sherlock/extractors.py
from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone
from typing import Sequence

from kgraph.document import BaseDocument
from kgraph.relationship import BaseRelationship
from kgraph.entity import BaseEntity
from kgraph.pipeline.interfaces import RelationshipExtractorInterface

from ..domain import (
    AppearsInRelationship,
    CoOccursWithRelationship,
)

# ============================================================================
# Relationship extractor
# ============================================================================


class SherlockRelationshipExtractor(RelationshipExtractorInterface):
    """Extract relationships from resolved entities + document text.

    Strategies:
    - appears_in: character -> story for each character seen in doc
    - co_occurs_with: character pairs co-mentioned within same paragraph
    """

    async def extract(self, document: BaseDocument, entities: Sequence[BaseEntity]) -> list[BaseRelationship]:
        relationships: list[BaseRelationship] = []

        story = next((e for e in entities if e.get_entity_type() == "story"), None)
        if story is None:
            # If the story entity wasn't emitted/resolved, skip story-based edges.
            return relationships

        dc = {}
        for e in entities:
            if e.get_entity_type() == "character":
                # uniqueness
                dc[e.entity_id] = e
        characters = list(dc.values())

        for c in characters:
            relationships.append(
                AppearsInRelationship(
                    subject_id=c.entity_id,
                    predicate="appears_in",
                    object_id=story.entity_id,
                    confidence=0.95,
                    source_documents=(document.document_id,),
                    created_at=datetime.now(timezone.utc),
                    last_updated=None,
                    metadata={},
                )
            )

        def _names_for(c) -> tuple[str, ...]:
            syns = getattr(c, "synonyms", ()) or ()
            # include canonical name plus synonyms/aliases
            return tuple(n for n in (c.name, *syns) if n and n.strip())

        def _present(para_lc: str, names: tuple[str, ...]) -> bool:
            for n in names:
                n_lc = n.lower().strip()
                if not n_lc:
                    continue
                # basic "word boundary" match; handles punctuation reasonably
                if re.search(rf"\b{re.escape(n_lc)}\b", para_lc):
                    return True
            return False

        # If I try co-occurrence by individual paragraph, there are almost none
        # so let's take paragraphs N-at-a-time and find co-occurrence that way.
        # If we were using an LLM for this, we could do smarter things, like
        # observe that all action is from Watson's point of view so anybody
        # present is always co-occurring with Watson.
        N = 5
        paragraphs = [p for p in document.content.split("\n\n") if p.strip()]
        counts: dict[tuple[str, str], int] = {}

        for j in range(len(paragraphs) - (N - 1)):
            para = "\n".join(paragraphs[j : j + N])
            para_lc = para.lower()
            if len(para_lc) < 40:
                continue
            present = sorted({c.entity_id for c in characters if _present(para_lc, _names_for(c))})
            for a, b in itertools.combinations(present, 2):
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1

        for (a, b), n in counts.items():
            conf = min(0.95, 0.60 + 0.10 * n)
            relationships.append(
                CoOccursWithRelationship(
                    subject_id=a,
                    predicate="co_occurs_with",
                    object_id=b,
                    confidence=conf,
                    source_documents=(document.document_id,),
                    created_at=datetime.now(timezone.utc),
                    last_updated=None,
                    metadata={"co_occurrence_count": n},
                )
            )

        return relationships
