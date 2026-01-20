# examples/sherlock/extractors.py
from __future__ import annotations

import re

from kgraph.document import BaseDocument
from kgraph.entity import EntityMention
from kgraph.pipeline.interfaces import EntityExtractorInterface

from ..data import KNOWN_CHARACTERS, KNOWN_LOCATIONS

# ============================================================================
# Entity extractor (mentions)
# ============================================================================


class SherlockEntityExtractor(EntityExtractorInterface):
    """Extract character, location, and story mentions using curated alias lists."""

    def __init__(self) -> None:
        self._patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[re.Pattern, str, str, float]]:
        patterns: list[tuple[re.Pattern, str, str, float]] = []

        def add_alias(entity_type: str, canonical_id: str, alias: str, base_conf: float) -> None:
            alias = alias.strip()
            if not alias:
                return
            # Word-boundary match; escape all user-provided aliases.
            pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            patterns.append((pat, canonical_id, entity_type, base_conf))

        for canonical_id, data in KNOWN_CHARACTERS.items():
            add_alias("character", canonical_id, data["name"], 1.0)
            for alias in data.get("aliases", ()):
                # very short aliases are ambiguous; multi-word is more specific
                conf = 0.95 if len(alias.split()) >= 2 else 0.80
                add_alias("character", canonical_id, alias, conf)

        for canonical_id, data in KNOWN_LOCATIONS.items():
            add_alias("location", canonical_id, data["name"], 1.0)
            for alias in data.get("aliases", ()):
                conf = 0.95 if len(alias.split()) >= 2 else 0.80
                add_alias("location", canonical_id, alias, conf)

        # Sort longer patterns first to reduce weird overlaps (optional but helps)
        patterns.sort(key=lambda x: len(x[0].pattern), reverse=True)
        return patterns

    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        mentions: list[EntityMention] = []

        # Emit a single "story" mention so the story entity exists.
        # Prefer document.story_id if parser found it.
        story_id = getattr(document, "story_id", None)
        if story_id:
            mentions.append(
                EntityMention(
                    text=getattr(document, "title", "") or "",
                    entity_type="story",
                    start_offset=0,
                    end_offset=0,
                    confidence=1.0,
                    context=None,
                    metadata={
                        "canonical_id_hint": getattr(document, "story_id", None),
                        "document_id": document.document_id,
                    },
                )
            )

        # Emit character/location mentions from content
        for pattern, canonical_id, entity_type, confidence in self._patterns:
            for match in pattern.finditer(document.content):
                start, end = match.start(), match.end()
                mentions.append(
                    EntityMention(
                        text=match.group(),
                        entity_type=entity_type,
                        start_offset=start,
                        end_offset=end,
                        confidence=confidence,
                        context=document.content[max(0, start - 60) : min(len(document.content), end + 60)],
                        metadata={"canonical_id_hint": canonical_id},
                    )
                )

        return mentions
