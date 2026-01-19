# examples/sherlock/extractors.py
from __future__ import annotations

import os
import hashlib
import itertools
import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from kgraph.document import BaseDocument
from kgraph.entity import BaseEntity, EntityMention, EntityStatus
from kgraph.pipeline.embedding import EmbeddingGeneratorInterface
from kgraph.pipeline.interfaces import (
    DocumentParserInterface,
    EntityExtractorInterface,
    EntityResolverInterface,
    EntityStorageInterface,
    RelationshipExtractorInterface,
)

from ..data import ADVENTURES_STORIES, KNOWN_CHARACTERS, KNOWN_LOCATIONS, find_story_by_title
from ..domain import (
    SherlockCharacter,
    SherlockDocument,
    SherlockLocation,
    SherlockStory,
    AppearsInRelationship,
    CoOccursWithRelationship,
    SherlockDomainSchema,
)


_ROMAN_HEADER = re.compile(r"^\s*(?P<roman>I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s+(?P<title>.+?)\s*$", re.I)

def _clean_story_header(s: str) -> str:
    m = _ROMAN_HEADER.match(s.strip())
    if m:
        # Convert "A SCANDAL IN BOHEMIA" to "A Scandal In Bohemia"-ish
        return m.group("title").strip().title()
    return s.strip()


# ============================================================================
# Document parser
# ============================================================================

class SherlockDocumentParser(DocumentParserInterface):
    """Parse plain text Sherlock Holmes stories into SherlockDocument objects."""

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> SherlockDocument:
        if content_type != "text/plain":
            raise ValueError("SherlockDocumentParser only supports text/plain")

        text = raw_content.decode("utf-8", errors="replace")

        title = (source_uri or "").strip() or self._extract_title(text, source_uri)
        meta = find_story_by_title(title)
        if not meta:
            raise ValueError(f"Unknown story title {title!r} (source_uri={source_uri!r})")
        story_id = meta["canonical_id"] if meta else None
        collection = meta.get("collection") if meta else None
        year = meta.get("year") if meta else None

        return SherlockDocument(
            document_id=str(uuid.uuid4()),
            title=title,
            content=text,
            content_type=content_type,
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
            story_id=story_id,
            collection=collection,
            metadata={"publication_year": year} if year else {},
        )

    def _extract_title(self, text: str, source_uri: str | None) -> str:
        if source_uri and source_uri.strip():
            return source_uri.strip()
        return "Untitled Sherlock Story"

    def ____extract_title(self, text: str, source_uri: str | None) -> str:
        # Try first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line:
                # Gutenberg headers can be noisy; if the first non-empty line looks like
                # an "ADVENTURE ..." marker, fall back to source_uri.
                if line.upper().startswith("ADVENTURE "):
                    break
                return _clean_story_header(line)

        # Fallback
        if source_uri and source_uri.strip():
            return source_uri.strip()

        return "Untitled Sherlock Story"


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
            # mentions.append(
            #     EntityMention(
            #         text=document.title or story_id,
            #         entity_type="story",
            #         start_offset=0,
            #         end_offset=0,
            #         confidence=1.0,
            #         context=None,
            #         metadata={"canonical_id_hint": story_id},
            #     )
            # )
            mentions.append(EntityMention(
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
            ))

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


# ============================================================================
# Entity resolver
# ============================================================================

class SherlockEntityResolver(EntityResolverInterface):
    """Resolve mentions to canonical entities when a hint is present; else provisional."""

    async def resolve(
        self,
        mention: EntityMention,
        existing_storage: EntityStorageInterface,
    ) -> tuple[BaseEntity, float]:
        canonical_id = (mention.metadata or {}).get("canonical_id_hint") if hasattr(mention, "metadata") else None

        if canonical_id:
            existing = await existing_storage.get(canonical_id)
            if existing:
                return existing, float(getattr(mention, "confidence", 1.0))

            # Create the right concrete entity type
            if mention.entity_type == "character":
                data = KNOWN_CHARACTERS.get(canonical_id, {})
                ent = SherlockCharacter(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=data.get("name", mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    role=data.get("role"),
                )
                return ent, ent.confidence

            if mention.entity_type == "location":
                data = KNOWN_LOCATIONS.get(canonical_id, {})
                ent = SherlockLocation(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=data.get("name", mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    location_type=data.get("location_type"),
                )
                return ent, ent.confidence

            if mention.entity_type == "story":
                # Look up story data from ADVENTURES_STORIES if available
                meta = next((s for s in ADVENTURES_STORIES if s["canonical_id"] == canonical_id), None)
                ent = SherlockStory(
                    entity_id=canonical_id,
                    status=EntityStatus.CANONICAL,
                    name=(meta["title"] if meta else mention.text),
                    confidence=float(getattr(mention, "confidence", 1.0)),
                    usage_count=1,
                    created_at=datetime.now(timezone.utc),
                    source="sherlock:curated",
                    metadata={},
                    collection=(meta.get("collection") if meta else None),
                    publication_year=(meta.get("year") if meta else None),
                )
                return ent, ent.confidence

        # Provisional fallback
        prov_id = f"prov:{uuid.uuid4().hex}"
        ent = BaseEntity(
            entity_id=prov_id,
            status=EntityStatus.PROVISIONAL,
            name=mention.text,
            confidence=float(getattr(mention, "confidence", 0.5)) * 0.5,
            usage_count=1,
            created_at=datetime.now(timezone.utc),
            source="sherlock:extracted",
            metadata={},
        )
        return ent, ent.confidence

    async def resolve_batch(
        self,
        mentions: Sequence[EntityMention],
        existing_storage: EntityStorageInterface,
    ) -> list[tuple[BaseEntity, float]]:
        out: list[tuple[BaseEntity, float]] = []
        for m in mentions:
            out.append(await self.resolve(m, existing_storage))
        return out


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
            para = "\n".join(paragraphs[j:j+N])
            para_lc = para.lower()
            if len(para_lc) < 40:
                continue
            present = sorted({
                c.entity_id
                for c in characters
                if _present(para_lc, _names_for(c))
            })
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


# ============================================================================
# Embedding generator (toy)
# ============================================================================

class SimpleEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Deterministic hash-based embedding generator (demo only)."""

    @property
    def dimension(self) -> int:
        return 32

    async def generate(self, text: str) -> tuple[float, ...]:
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        values = [b / 255.0 for b in h[: self.dimension]]
        mag = sum(v * v for v in values) ** 0.5
        if mag == 0:
            return tuple(0.0 for _ in values)
        return tuple(v / mag for v in values)

    async def generate_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [await self.generate(t) for t in texts]
