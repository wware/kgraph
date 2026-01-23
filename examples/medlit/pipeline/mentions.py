"""Entity mention extraction from journal articles.

Extracts entity mentions from Paper JSON format (from med-lit-schema).
Since the papers already have extracted entities, we convert those to EntityMention objects.
Can also use Ollama LLM for NER extraction from text.
"""

from typing import Optional

from kgraph.document import BaseDocument
from kgraph.entity import EntityMention
from kgraph.pipeline.interfaces import EntityExtractorInterface

from .llm_client import LLMClientInterface


class MedLitEntityExtractor(EntityExtractorInterface):
    """Extract entity mentions from journal articles.

    This extractor works with Paper JSON format from med-lit-schema, which
    already contains extracted entities. We convert those to EntityMention objects.

    Can also use Ollama LLM to extract entities directly from text if llm_client is provided.
    """

    OLLAMA_NER_PROMPT = """Extract all medical entities (diseases, drugs, genes, proteins) from the following text.
Return ONLY a JSON array of objects with "entity" (the entity name), "type" (disease, drug, gene, or protein), and "confidence" (0.0-1.0) fields.
Do not include any explanation, just the JSON array.

Example output:
[{{"entity": "diabetes", "type": "disease", "confidence": 0.95}}, {{"entity": "insulin", "type": "drug", "confidence": 0.90}}]

If no entities are found, return an empty array: []

Text to analyze:
{text}

JSON output:"""

    def __init__(self, llm_client: Optional[LLMClientInterface] = None):
        """Initialize entity extractor.

        Args:
            llm_client: Optional LLM client for extracting entities from text.
                        If None, only uses pre-extracted entities from Paper JSON.
        """
        self._llm = llm_client

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

                print(f"  Extracting entities with LLM from {len(text_sample)} chars...")
                prompt = self.OLLAMA_NER_PROMPT.format(text=text_sample)
                response = await self._llm.generate_json(prompt)
                print(f"  LLM returned {len(response) if isinstance(response, list) else 0} entities")

                if isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict):
                            entity_name = item.get("entity", "").strip()
                            entity_type = item.get("type", "disease").lower()
                            confidence = float(item.get("confidence", 0.5))

                            if len(entity_name) < 3 or confidence < 0.5:
                                continue

                            mention = EntityMention(
                                text=entity_name,
                                entity_type=entity_type,
                                start_offset=0,  # LLM doesn't provide offsets
                                end_offset=0,
                                confidence=confidence,
                                context=None,
                                metadata={
                                    "document_id": document.document_id,
                                    "extraction_method": "llm",
                                },
                            )
                            mentions.append(mention)
            except Exception as e:
                print(f"Warning: LLM entity extraction failed: {e}")

        return mentions
