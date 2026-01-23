"""Document parser for journal articles.

Converts raw paper input (PMC XML, JSON, etc.) into JournalArticle documents.
"""

from datetime import datetime, timezone
from typing import Any

from kgraph.pipeline.interfaces import DocumentParserInterface

from ..documents import JournalArticle


class JournalArticleParser(DocumentParserInterface):
    """Parse raw journal article content into JournalArticle documents.

    This parser handles various input formats (PMC XML, JSON from med-lit-schema,
    etc.) and converts them to kgraph's JournalArticle format.

    For now, this is a minimal implementation. A full implementation would:
    1. Parse PMC XML (using existing med-lit-schema parser logic)
    2. Parse JSON from med-lit-schema's Paper format
    3. Extract metadata and map to JournalArticle fields
    """

    async def parse(
        self,
        raw_content: bytes,
        content_type: str,
        source_uri: str | None = None,
    ) -> JournalArticle:
        """Parse raw content into a JournalArticle.

        Args:
            raw_content: Raw document bytes (may be JSON, XML, etc.)
            content_type: MIME type or format indicator
            source_uri: Optional URI identifying the document's origin

        Returns:
            A JournalArticle instance ready for entity and relationship extraction.

        Raises:
            ValueError: If content_type is unsupported or content is malformed.
        """
        if content_type == "application/json":
            # Parse JSON (e.g., from med-lit-schema's Paper format)
            import json

            data = json.loads(raw_content.decode("utf-8"))
            return self._parse_from_dict(data, source_uri)

        elif content_type in {"application/xml", "text/xml"}:
            # Parse PMC XML
            # TODO: Implement full PMC XML parsing (see med-lit-schema's pmc_parser.py)
            raise ValueError("PMC XML parsing not yet implemented. Use JSON format for now.")

        else:
            raise ValueError(f"Unsupported content_type: {content_type}")

    def _parse_from_dict(self, data: dict[str, Any], source_uri: str | None) -> JournalArticle:
        """Parse from a dictionary (e.g., med-lit-schema's Paper format).

        Maps Paper fields to JournalArticle:
        - paper_id → document_id (prefer doi:, else pmid:, else paper_id)
        - title → title
        - abstract → abstract
        - abstract (+ optional full_text) → content
        - authors → authors
        - publication_date → publication_date
        - journal → journal
        - doi → doi
        - pmid → pmid
        - metadata → metadata (study_type, sample_size, mesh_terms, etc.)
        - extraction_provenance → metadata["extraction"]
        """
        # Determine document_id (prefer DOI, else PMID, else paper_id)
        paper_id = data.get("paper_id", "")
        doi = data.get("doi")
        pmid = data.get("pmid")

        if doi:
            document_id = f"doi:{doi}"
        elif pmid:
            document_id = f"pmid:{pmid}"
        elif paper_id:
            document_id = paper_id
        else:
            raise ValueError("No valid identifier found (paper_id, doi, or pmid required)")

        ## Extract content (abstract + optional full text)
        # abstract = data.get("abstract", "")
        # full_text = data.get("full_text", "")  # May not be present
        # content = abstract
        # if full_text:
        #    content = f"{abstract}\n\n{full_text}" if abstract else full_text

        # NEW CODE (fixed):
        # Extract content (abstract + optional full text)
        abstract_obj = data.get("abstract", "")
        # Handle both dict {"text": "..."} and plain string formats
        if isinstance(abstract_obj, dict):
            abstract = abstract_obj.get("text", "")
        else:
            abstract = str(abstract_obj) if abstract_obj else ""
        full_text = data.get("full_text", "")
        content = abstract
        if full_text:
            content = f"{abstract}\n\n{full_text}" if abstract else full_text

        # Extract metadata
        metadata: dict[str, Any] = {}

        # Map PaperMetadata fields
        paper_metadata = data.get("metadata", {})
        if isinstance(paper_metadata, dict):
            metadata.update(paper_metadata)
        else:
            # If metadata is a PaperMetadata object (already parsed), extract its fields
            if hasattr(paper_metadata, "study_type"):
                metadata["study_type"] = paper_metadata.study_type
            if hasattr(paper_metadata, "sample_size"):
                metadata["sample_size"] = paper_metadata.sample_size
            if hasattr(paper_metadata, "mesh_terms"):
                metadata["mesh_terms"] = paper_metadata.mesh_terms

        # Store extraction provenance if present
        extraction_provenance = data.get("extraction_provenance")
        if extraction_provenance:
            metadata["extraction"] = extraction_provenance

        # Store entities and relationships in metadata so extractors can find them
        # (they're at top level in Paper JSON, but we put them in metadata for extractors)
        if "entities" in data:
            metadata["entities"] = data["entities"]
        if "relationships" in data:
            metadata["relationships"] = data["relationships"]

        # Extract authors (handle both list and string formats)
        authors = data.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        elif not isinstance(authors, list):
            authors = []

        return JournalArticle(
            document_id=document_id,
            title=data.get("title", ""),
            content=content,
            content_type="text/plain",  # Processed text
            source_uri=source_uri,
            created_at=datetime.now(timezone.utc),
            authors=tuple(authors),
            abstract=abstract,
            publication_date=data.get("publication_date"),
            journal=data.get("journal"),
            doi=doi,
            pmid=pmid,
            metadata=metadata,
        )
