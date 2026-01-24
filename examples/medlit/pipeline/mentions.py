"""Entity mention extraction from journal articles.

Extracts entity mentions from Paper JSON format (from med-lit-schema).
Since the papers already have extracted entities, we convert those to EntityMention objects.
Can also use Ollama LLM for NER extraction from text with tool calling for canonical ID lookup.
"""

from typing import Optional

from kgraph.document import BaseDocument
from kgraph.entity import EntityMention
from kgraph.pipeline.interfaces import EntityExtractorInterface

from .authority_lookup import CanonicalIdLookup
from .llm_client import LLMClientInterface


class MedLitEntityExtractor(EntityExtractorInterface):
    """Extract entity mentions from journal articles.

    This extractor works with Paper JSON format from med-lit-schema, which
    already contains extracted entities. We convert those to EntityMention objects.

    Can also use Ollama LLM to extract entities directly from text if llm_client is provided.
    When a lookup service is provided, the LLM can use tool calling to look up canonical IDs.
    """

    OLLAMA_NER_PROMPT = """Extract medical entities from the following text. Return a JSON array.

Rules:
1. Find all diseases, genes, drugs, and proteins
2. Format: [{{"entity": "name", "type": "disease|gene|drug|protein", "confidence": 0.95}}]
3. Return ONLY the JSON array

Text:
{text}

JSON:"""

    OLLAMA_NER_PROMPT_WITH_TOOLS = """Extract medical entities from the following text.

For each entity found, use the lookup_canonical_id tool to find its canonical ID.
The tool takes two arguments: term (the entity name) and entity_type (disease, gene, drug, or protein).

Return a JSON array with entities including their canonical IDs.
Format: [{{"entity": "name", "type": "disease|gene|drug|protein", "confidence": 0.95, "canonical_id": "ID or null"}}]

Text:
{text}

JSON:"""

    def __init__(
        self,
        llm_client: Optional[LLMClientInterface] = None,
        lookup: Optional[CanonicalIdLookup] = None,
    ):
        """Initialize entity extractor.

        Args:
            llm_client: Optional LLM client for extracting entities from text.
                        If None, only uses pre-extracted entities from Paper JSON.
            lookup: Optional canonical ID lookup service. When provided along with
                   an LLM client that supports tools, enables tool calling for
                   canonical ID lookup during extraction.
        """
        self._llm = llm_client
        self._lookup = lookup

    async def extract(self, document: BaseDocument) -> list[EntityMention]:
        """Extract entity mentions from a journal article.

        If the document metadata contains pre-extracted entities (from med-lit-schema),
        we convert those to EntityMention objects. Otherwise, if llm_client is provided,
        extracts entities from document text using LLM.

        Args:
            document: The journal article document.

        Returns:
            List of EntityMention objects representing entities mentioned in the paper.
        """
        mentions: list[EntityMention] = []

        # Check if document has pre-extracted entities in metadata
        # (from med-lit-schema's Paper format)
        entities_data = document.metadata.get("entities", [])

        if entities_data:
            # Convert pre-extracted entities to EntityMention objects
            for entity_ref in entities_data:
                # Handle both dict format and EntityReference format
                if isinstance(entity_ref, dict):
                    entity_id = entity_ref.get("id", "")
                    entity_name = entity_ref.get("name", "")
                    entity_type = entity_ref.get("type", "")

                    # Map med-lit-schema entity types to kgraph entity types
                    # (they should match, but normalize just in case)
                    entity_type = entity_type.lower() if entity_type else ""

                    # Create mention (we don't have exact text spans, so use 0 offsets)
                    # The text is the name as it appeared in the paper
                    mention = EntityMention(
                        text=entity_name,
                        entity_type=entity_type,
                        start_offset=0,  # Unknown from pre-extracted format
                        end_offset=0,  # Unknown from pre-extracted format
                        confidence=0.9,  # Assume high confidence for pre-extracted
                        context=None,
                        metadata={
                            "canonical_id_hint": entity_id,
                            "document_id": document.document_id,
                            "pre_extracted": True,
                        },
                    )
                    mentions.append(mention)
            return mentions

        # No pre-extracted entities - try LLM extraction if available
        if self._llm and document.content:
            try:
                # Use abstract for extraction (shorter, more focused)
                text = document.abstract if hasattr(document, "abstract") and document.abstract else document.content
                text_sample = text[:2000]  # Limit text size for faster processing

                # Check if we can use tool calling
                use_tools = self._lookup is not None and hasattr(self._llm, "generate_json_with_tools")

                if use_tools:
                    print(f"  Extracting entities with LLM + tools from {len(text_sample)} chars...")
                    prompt = self.OLLAMA_NER_PROMPT_WITH_TOOLS.format(text=text_sample)

                    # Create the tool function that wraps our lookup
                    def lookup_canonical_id(term: str, entity_type: str) -> str | None:
                        """Look up canonical ID for a medical entity.

                        Args:
                            term: The entity name (e.g., "BRCA1", "breast cancer")
                            entity_type: Type of entity - one of: disease, gene, drug, protein

                        Returns:
                            Canonical ID if found (e.g., "HGNC:1100", "C0006142"), or null if not found
                        """
                        assert self._lookup is not None  # Guarded by use_tools check
                        return self._lookup.lookup_canonical_id_sync(term, entity_type)

                    response = await self._llm.generate_json_with_tools(
                        prompt,
                        tools=[lookup_canonical_id],
                    )
                else:
                    print(f"  Extracting entities with LLM from {len(text_sample)} chars...")
                    prompt = self.OLLAMA_NER_PROMPT.format(text=text_sample)
                    response = await self._llm.generate_json(prompt)

                print(f"  LLM returned {len(response) if isinstance(response, list) else 0} entities")

                if isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict):
                            # Handle both "entity" and "name" keys
                            entity_name = item.get("entity") or item.get("name", "")
                            entity_name = entity_name.strip()

                            entity_type = item.get("type", "disease").lower()
                            confidence = float(item.get("confidence", 0.5))

                            # Get canonical_id if present (from tool calling)
                            canonical_id = item.get("canonical_id")
                            if canonical_id in ("null", ""):
                                canonical_id = None

                            if len(entity_name) < 3 or confidence < 0.5:
                                continue

                            metadata = {
                                "document_id": document.document_id,
                                "extraction_method": "llm_with_tools" if use_tools else "llm",
                            }

                            # Add canonical_id_hint if we got one from tool calling
                            if canonical_id:
                                metadata["canonical_id_hint"] = canonical_id

                            mention = EntityMention(
                                text=entity_name,
                                entity_type=entity_type,
                                start_offset=0,  # LLM doesn't provide offsets
                                end_offset=0,
                                confidence=confidence,
                                context=None,
                                metadata=metadata,
                            )
                            mentions.append(mention)
            except Exception as e:
                print(f"Warning: LLM entity extraction failed: {e}")
                import traceback

                traceback.print_exc()

        return mentions
