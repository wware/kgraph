# examples/sherlock/extractors.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from kgraph.pipeline.interfaces import DocumentParserInterface

from ..data import find_story_by_title
from ..domain import SherlockDocument


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
