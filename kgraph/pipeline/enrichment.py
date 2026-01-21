"""Entity enrichment interfaces and implementations.

This module provides interfaces and implementations for enriching entities
with external data sources after entity resolution. Enrichers augment
resolved entities with canonical identifiers and metadata from external
knowledge bases such as DBPedia, Wikidata, UMLS, etc.

The enrichment process runs after entity resolution but before storage,
allowing entities to be augmented with cross-domain identifiers without
interfering with the core extraction pipeline.

Key components:
    - EntityEnricherInterface: Abstract base class for enrichers
    - DBPediaEnricher: Concrete implementation for DBPedia Lookup API
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict
import asyncio
import json
import logging
import urllib.parse
import urllib.request

from kgraph.entity import BaseEntity

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s")
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)


class EntityEnricherInterface(ABC):
    """Interface for enriching entities with external data sources.

    Enrichers augment resolved entities with information from external
    knowledge bases (DBPedia, Wikidata, UMLS, etc.), populating the
    canonical_ids field and optionally adding metadata.

    Enrichers run after entity resolution but before storage, allowing
    them to add cross-domain canonical identifiers and supplementary
    information without interfering with the core extraction pipeline.

    Example:
        ```python
        class CustomEnricher(EntityEnricherInterface):
            async def enrich_entity(self, entity: BaseEntity) -> BaseEntity:
                # Query external API
                external_id = await lookup_external_id(entity.name)
                if external_id:
                    # Add to canonical_ids
                    ids = dict(entity.canonical_ids)
                    ids['custom_source'] = external_id
                    return entity.model_copy(update={'canonical_ids': ids})
                return entity
        ```
    """

    @abstractmethod
    async def enrich_entity(self, entity: BaseEntity) -> BaseEntity:
        """Enrich a single entity with external data.

        Args:
            entity: The entity to enrich (canonical or provisional)

        Returns:
            Updated entity with enriched canonical_ids and/or metadata.
            If enrichment fails or no match found, returns original entity.

        Note:
            Implementations should handle errors gracefully and return the
            original entity rather than raising exceptions, to avoid breaking
            the ingestion pipeline.
        """

    async def enrich_batch(self, entities: list[BaseEntity]) -> list[BaseEntity]:
        """Enrich multiple entities in a batch (optional optimization).

        Default implementation calls enrich_entity for each entity,
        but implementations may override for batch API calls.

        Args:
            entities: List of entities to enrich

        Returns:
            List of enriched entities in the same order as input
        """
        return [await self.enrich_entity(entity) for entity in entities]


class DBPediaEnricher(EntityEnricherInterface):
    """Enriches entities with DBPedia URIs and linked data.

    Queries DBPedia Lookup API to find matching resources and adds
    DBPedia URIs to entity.canonical_ids['dbpedia'].

    The enricher uses a disambiguation strategy based on:
    1. Exact label matching with entity name
    2. Synonym matching against DBPedia labels/redirects
    3. Entity type filtering (if supported by domain)
    4. Confidence score thresholding

    Configuration:
        - entity_types_to_enrich: Whitelist of entity types (None = all types)
        - confidence_threshold: Minimum entity confidence to attempt enrichment
        - min_lookup_score: Minimum DBPedia lookup score to accept match
        - match_similarity_score: Score assigned to non-exact matches (default: 0.5)
        - cache_results: Whether to cache lookup results
        - timeout: HTTP request timeout in seconds

    Example:
        ```python
        enricher = DBPediaEnricher(
            entity_types_to_enrich={'person', 'location'},
            confidence_threshold=0.8,
            min_lookup_score=0.6,
        )

        enriched = await enricher.enrich_entity(entity)
        if 'dbpedia' in enriched.canonical_ids:
            print(f"Found DBPedia URI: {enriched.canonical_ids['dbpedia']}")
        ```
    """

    def __init__(
        self,
        entity_types_to_enrich: set[str] | None = None,
        confidence_threshold: float = 0.7,
        min_lookup_score: float = 0.5,
        match_similarity_score: float = 0.5,
        cache_results: bool = True,
        timeout: float = 5.0,
    ):
        """Initialize DBPedia enricher.

        Args:
            entity_types_to_enrich: Set of entity types to enrich (None = all)
            confidence_threshold: Only enrich entities with confidence >= this
            min_lookup_score: Minimum DBPedia score to accept match
            match_similarity_score: Score for partial/non-exact matches (0.0-1.0)
            cache_results: Enable in-memory caching of lookups
            timeout: HTTP timeout for DBPedia API calls
        """
        self.entity_types_to_enrich = entity_types_to_enrich
        self.confidence_threshold = confidence_threshold
        self.min_lookup_score = min_lookup_score
        self.match_similarity_score = match_similarity_score
        self.cache_results = cache_results
        self.timeout = timeout
        self._cache: dict[tuple[str, str], str | None] = {}

    async def enrich_entity(self, entity: BaseEntity) -> BaseEntity:
        """Enrich entity with DBPedia URI if found.

        Process:
        1. Check if entity type should be enriched
        2. Check if entity confidence meets threshold
        3. Query DBPedia Lookup API with entity name
        4. Disambiguate results using entity type and synonyms
        5. Add best match URI to canonical_ids['dbpedia']
        6. Handle errors gracefully (return original entity on failure)

        Note: this is BUGGY AS HELL for the Sherlock example
        * Holmes -> http://dbpedia.org/resource/Minor_Sherlock_Holmes_characters
        * Watson -> http://dbpedia.org/resource/Reg_Watson
        * Mrs. Hudson -> http://dbpedia.org/resource/Landlord
        but it's a start. TODO: make this work better

        Args:
            entity: The entity to enrich

        Returns:
            Entity with DBPedia URI added to canonical_ids, or original entity
            if no match found or enrichment fails.
        """
        # Check if entity type should be enriched
        if self.entity_types_to_enrich is not None:
            if entity.get_entity_type() not in self.entity_types_to_enrich:
                logger.debug(
                    "Skipping enrichment for entity '%s' (type '%s' not in whitelist)",
                    entity.name,
                    entity.get_entity_type(),
                )
                return entity

        # Check confidence threshold
        if entity.confidence < self.confidence_threshold:
            logger.debug(
                "Skipping enrichment for entity '%s' (confidence %.2f < %.2f)",
                entity.name,
                entity.confidence,
                self.confidence_threshold,
            )
            return entity

        # Check cache
        cache_key = (entity.name, entity.get_entity_type())
        if self.cache_results and cache_key in self._cache:
            cached_uri = self._cache[cache_key]
            if cached_uri:
                logger.debug("Using cached DBPedia URI for '%s': %s", entity.name, cached_uri)
                ids = dict(entity.canonical_ids)
                ids["dbpedia"] = cached_uri
                return entity.model_copy(update={"canonical_ids": ids})
            else:
                logger.debug("Using cached negative result for '%s'", entity.name)
                return entity

        # Query DBPedia
        try:
            results = await self._query_dbpedia(entity.name, entity.get_entity_type())
            dbpedia_uri = self._disambiguate_results(results, entity)

            # Cache result
            if self.cache_results:
                self._cache[cache_key] = dbpedia_uri

            # Add to canonical_ids if found
            if dbpedia_uri:
                logger.info("Enriched entity '%s' with DBPedia URI: %s", entity.name, dbpedia_uri)
                ids = dict(entity.canonical_ids)
                ids["dbpedia"] = dbpedia_uri
                return entity.model_copy(update={"canonical_ids": ids})
            else:
                logger.debug("No DBPedia match found for entity '%s'", entity.name)
                return entity

        except Exception:
            logger.exception("DBPedia enrichment failed for entity '%s'", entity.name)
            # Cache negative result to avoid repeated failures
            if self.cache_results:
                self._cache[cache_key] = None
            return entity

    async def _query_dbpedia(self, query: str, entity_type: str, retries: int = 2) -> List[Dict[str, Any]]:
        """Query DBPedia Lookup API with retry logic.

        Args:
            query: Entity name to search for
            entity_type: Entity type (for logging/future filtering)
            retries: Number of retry attempts on failure

        Returns:
            List of DBPedia resource dictionaries

        Raises:
            Exception: On HTTP errors, timeouts, or JSON parse errors after all retries
        """
        url = f"https://lookup.dbpedia.org/api/search?query={urllib.parse.quote(query)}&format=json&maxResults=5"
        for attempt in range(retries):
            try:
                # Add format=json parameter and increase maxResults

                def sync_request():
                    # Add User-Agent header - some APIs require it
                    req = urllib.request.Request(url, headers={"User-Agent": "kgraph-entity-enricher/1.0"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as response:
                        # Check response status
                        if response.status != 200:
                            logger.warning("DBPedia returned status %d for query: %s", response.status, query)
                            return {"docs": []}

                        # Read and validate content before parsing
                        content = response.read().decode("utf-8")
                        if not content or not content.strip():
                            logger.warning("Empty response from DBPedia for query: %s", query)
                            return {"docs": []}

                        return json.loads(content)

                data = await asyncio.to_thread(sync_request)

                # Extract docs from response
                if isinstance(data, dict) and "docs" in data:
                    return data["docs"]
                return []

            except json.JSONDecodeError as e:
                logger.warning("JSON decode error for '%s' (attempt %d/%d): %s", query, attempt + 1, retries, e)
                if attempt < retries - 1:
                    await asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    # Return empty results instead of raising on final attempt
                    logger.debug("All retries exhausted for '%s', returning no results", query)
                    return []
            except Exception as e:
                logger.debug("DBPedia API request failed for '%s' (attempt %d/%d): %s", query, attempt + 1, retries, e)
                if attempt < retries - 1:
                    await asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    raise

        return []

    def _disambiguate_results(self, results: list[dict[str, Any]], entity: BaseEntity) -> str | None:
        """Select best DBPedia result based on entity attributes.

        Disambiguation strategy:
        1. Prefer exact label matches with entity.name
        2. Check entity synonyms against DBPedia labels
        3. Select highest-scoring result above threshold
        4. Return None if no suitable match found

        Args:
            results: List of DBPedia resource dictionaries
            entity: The entity being enriched

        Returns:
            DBPedia resource URI, or None if no suitable match
        """
        if not results:
            return None

        best_uri = None
        best_score = 0.0

        entity_names = {entity.name.lower()} | {syn.lower() for syn in entity.synonyms}

        for result in results:
            # Extract label and URI
            label = result.get("label", [""])[0] if isinstance(result.get("label"), list) else result.get("label", "")
            uri = result.get("resource", [""])[0] if isinstance(result.get("resource"), list) else result.get("resource", "")

            if not uri:
                continue

            # Calculate match score
            score = 0.0

            # Exact match gets highest score
            if label and label.lower() in entity_names:
                score = 1.0
            else:
                # Partial match - use configurable similarity score
                score = self.match_similarity_score

            # Consider DBPedia's own relevance score if available
            if "score" in result:
                dbpedia_score = float(result["score"][0])
                # Combine our score with DBPedia's score
                score = (score + dbpedia_score) / 2.0

            logger.debug("DBPedia match for '%s': %s (%s) - score: %.2f", entity.name, label, uri, score)

            # Update best match if score is higher
            if score > best_score and score >= self.min_lookup_score:
                best_score = score
                best_uri = uri

        logger.info("DBPedia match for '%s' (%s) - score: %.2f", entity.name, best_uri, best_score)
        return best_uri
