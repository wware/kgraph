"""Canonical ID lookup from medical ontology authorities.

Provides lookup functionality for canonical IDs from various medical ontology
sources: UMLS, HGNC, RxNorm, and UniProt.

Features persistent caching to avoid repeated API calls across runs.
"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment, misc]

from kgraph.logging import setup_logging

# Module-level logger for sync methods
logger = setup_logging()


class CanonicalIdLookup:
    """Look up canonical IDs from various medical ontology authorities.

    Supports lookup from:
    - UMLS (diseases, symptoms, procedures)
    - HGNC (genes)
    - RxNorm (drugs)
    - UniProt (proteins)

    Features persistent caching to disk to avoid repeated API calls across runs.
    """

    def __init__(
        self,
        umls_api_key: Optional[str] = None,
        cache_file: Optional[Path] = None,
    ):
        """Initialize the canonical ID lookup service.

        Args:
            umls_api_key: Optional UMLS API key. If not provided, will try to
                         read from UMLS_API_KEY environment variable.
            cache_file: Optional path to cache file. If not provided, defaults
                       to "authority_lookup_cache.json" in current directory.
        """
        if httpx is None:
            raise ImportError("httpx is required for authority lookup. Install with: uv add httpx")

        self.client = httpx.AsyncClient(timeout=10.0)
        self.umls_api_key = umls_api_key or os.getenv("UMLS_API_KEY")
        self.cache_file = cache_file or Path("authority_lookup_cache.json")
        self._cache: dict[str, str] = {}
        self._cache_dirty = False
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
                logger = setup_logging()
                logger.debug(
                    {
                        "message": f"Loaded {len(self._cache)} cached lookups from {self.cache_file}",
                        "cache_file": str(self.cache_file),
                        "cache_size": len(self._cache),
                    },
                    pprint=True,
                )
            except Exception as e:
                logger = setup_logging()
                logger.warning(
                    {
                        "message": f"Failed to load cache from {self.cache_file}",
                        "cache_file": str(self.cache_file),
                        "error": str(e),
                    },
                    pprint=True,
                )
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self._cache_dirty:
            return

        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
            self._cache_dirty = False
            logger = setup_logging()
            logger.debug(
                {
                    "message": f"Saved {len(self._cache)} cached lookups to {self.cache_file}",
                    "cache_file": str(self.cache_file),
                    "cache_size": len(self._cache),
                },
                pprint=True,
            )
        except Exception as e:
            logger = setup_logging()
            logger.warning(
                {
                    "message": f"Failed to save cache to {self.cache_file}",
                    "cache_file": str(self.cache_file),
                    "error": str(e),
                },
                pprint=True,
            )

    async def lookup_canonical_id(self, term: str, entity_type: str) -> Optional[str]:
        """Look up canonical ID for a medical term.

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (disease, gene, drug, protein, etc.)

        Returns:
            Canonical ID string if found, None otherwise
        """
        logger = setup_logging()

        # Normalize cache key (lowercase, strip whitespace, use string format for JSON)
        cache_key = f"{entity_type}:{term.lower().strip()}"

        # Check cache first
        if cache_key in self._cache:
            cached_value = self._cache[cache_key]
            # Handle cached None values (stored as "NULL" string for JSON compatibility)
            result = None if cached_value == "NULL" else cached_value
            logger.debug(
                {
                    "message": f"Cache hit for {term} ({entity_type})",
                    "cached_id": result,
                },
                pprint=True,
            )
            return result

        logger.debug(
            {
                "message": f"Looking up canonical ID for '{term}' (type: {entity_type})",
                "term": term,
                "entity_type": entity_type,
            },
            pprint=True,
        )

        # Route to appropriate authority based on entity type
        canonical_id: Optional[str] = None

        if entity_type in ("disease", "symptom", "procedure"):
            canonical_id = await self._lookup_umls(term)
        elif entity_type == "gene":
            canonical_id = await self._lookup_hgnc(term)
        elif entity_type == "drug":
            canonical_id = await self._lookup_rxnorm(term)
        elif entity_type == "protein":
            canonical_id = await self._lookup_uniprot(term)

        # Cache result (store "NULL" for None to distinguish from "not yet looked up" in JSON)
        self._cache[cache_key] = "NULL" if canonical_id is None else canonical_id
        self._cache_dirty = True

        # Periodically save cache (every 100 new entries to avoid losing work)
        if len(self._cache) % 100 == 0:
            self._save_cache()

        if canonical_id:
            logger.info(
                {
                    "message": f"Found canonical ID for '{term}': {canonical_id}",
                    "term": term,
                    "entity_type": entity_type,
                    "canonical_id": canonical_id,
                },
                pprint=True,
            )
        else:
            logger.debug(
                {
                    "message": f"No canonical ID found for '{term}' (type: {entity_type})",
                    "term": term,
                    "entity_type": entity_type,
                },
                pprint=True,
            )

        return canonical_id

    async def _lookup_umls(self, term: str) -> Optional[str]:
        """Look up UMLS CUI for a disease/symptom term.

        Falls back to MeSH lookup if UMLS API key is not available.
        """
        # Try UMLS first if API key is available
        if self.umls_api_key:
            try:
                # UMLS REST API search endpoint
                url = "https://uts-ws.nlm.nih.gov/rest/search/current"
                params = {
                    "string": term,
                    "apiKey": self.umls_api_key,
                    "searchType": "exact",
                }
                response = await self.client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", {}).get("results", [])
                    if results:
                        # Return first match CUI
                        cui = results[0].get("ui")
                        if cui:
                            return cui  # e.g., "C0006142"
            except Exception as e:
                logger.warning(
                    {
                        "message": f"UMLS lookup failed for '{term}', trying MeSH fallback",
                        "term": term,
                        "error": str(e),
                    },
                    pprint=True,
                )

        # Fall back to MeSH (no API key required)
        return await self._lookup_mesh(term)

    async def _lookup_mesh(self, term: str) -> Optional[str]:
        """Look up MeSH descriptor ID for a disease/symptom term.

        MeSH (Medical Subject Headings) is freely accessible without API key.
        Returns MeSH descriptor IDs like "MeSH:D001943" (breast neoplasms).

        Strategy:
        1. Try descriptor lookup with original term
        2. Try with medical term normalization (cancer → neoplasms)
        3. Try exact match on descriptor label
        """
        # Terms to try: original and normalized versions
        search_terms = [term]

        # Common medical term normalizations (MeSH uses formal terminology)
        term_lower = term.lower()
        if "cancer" in term_lower:
            # "breast cancer" → "breast neoplasms"
            normalized = term_lower.replace("cancer", "neoplasms")
            search_terms.append(normalized.title())
        if "tumor" in term_lower or "tumour" in term_lower:
            normalized = term_lower.replace("tumor", "neoplasms").replace("tumour", "neoplasms")
            search_terms.append(normalized.title())

        for search_term in search_terms:
            result = await self._try_mesh_descriptor_lookup(search_term)
            if result:
                return result

        return None

    def _extract_mesh_id_from_results(self, data: list, term: str) -> Optional[str]:
        """Extract MeSH descriptor ID from API results if a good match is found."""
        term_lower = term.lower()
        term_words = set(term_lower.split())

        for result in data:
            label = result.get("label", "").lower()
            label_words = set(label.split())

            # Accept if most words match (handles word order differences)
            common_words = term_words & label_words
            if len(common_words) < len(term_words) - 1 or len(common_words) < 1:
                continue

            resource_uri = result.get("resource", "")
            if "/mesh/D" in resource_uri:
                mesh_id = resource_uri.split("/mesh/")[-1]
                return f"MeSH:{mesh_id}"

        return None

    async def _try_mesh_descriptor_lookup(self, term: str) -> Optional[str]:
        """Try to find a MeSH descriptor for a term."""
        try:
            url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
            params = {"label": term, "match": "contains", "limit": "10"}
            response = await self.client.get(url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            return self._extract_mesh_id_from_results(data, term)
        except Exception as e:
            logger.warning(
                {
                    "message": f"MeSH lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    async def _lookup_hgnc(self, term: str) -> Optional[str]:
        """Look up HGNC ID for a gene.

        Tries official symbol first, then falls back to alias search.
        This handles cases like "p53" which is an alias for "TP53".
        """
        headers = {"Accept": "application/json"}

        try:
            # Strategy 1: Try exact symbol match
            url = f"https://rest.genenames.org/fetch/symbol/{term}"
            response = await self.client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                if docs:
                    hgnc_id = docs[0].get("hgnc_id")
                    if hgnc_id:
                        return hgnc_id  # Already includes "HGNC:" prefix

            # Strategy 2: Try alias search (handles "p53" → "TP53")
            url = f"https://rest.genenames.org/search/alias_symbol/{term}"
            response = await self.client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                docs = data.get("response", {}).get("docs", [])
                if docs:
                    hgnc_id = docs[0].get("hgnc_id")
                    if hgnc_id:
                        return hgnc_id

            return None
        except Exception as e:
            logger.warning(
                {
                    "message": f"HGNC lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    async def _lookup_rxnorm(self, term: str) -> Optional[str]:
        """Look up RxNorm ID for a drug."""
        logger = setup_logging()

        try:
            # RxNorm API (no authentication required!)
            url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={term}"
            response = await self.client.get(url)

            if response.status_code == 200:
                data = response.json()
                id_group = data.get("idGroup", {})
                rxnorm_ids = id_group.get("rxnormId", [])
                if rxnorm_ids:
                    return f"RxNorm:{rxnorm_ids[0]}"
            return None
        except Exception as e:
            logger.warning(
                {
                    "message": f"RxNorm lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    async def _lookup_uniprot(self, term: str) -> Optional[str]:
        """Look up UniProt ID for a protein."""
        logger = setup_logging()

        try:
            # UniProt REST API (no authentication required!)
            url = "https://rest.uniprot.org/uniprotkb/search"
            params: dict[str, str] = {
                "query": f"protein_name:{term}",
                "format": "json",
                "size": "1",
            }
            response = await self.client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    return results[0].get("primaryAccession")  # e.g., "P38398"
            return None
        except Exception as e:
            logger.warning(
                {
                    "message": f"UniProt lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    def lookup_canonical_id_sync(self, term: str, entity_type: str) -> Optional[str]:
        """Synchronous wrapper for use as Ollama tool.

        This is needed because Ollama tool functions must be synchronous.
        Uses the cache first, then makes synchronous HTTP calls if needed.

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (disease, gene, drug, protein, etc.)

        Returns:
            Canonical ID string if found, None otherwise
        """
        import httpx as httpx_sync

        # Normalize inputs
        term_normalized = term.strip().lower()
        entity_type_normalized = entity_type.lower()

        # Check cache first (this is synchronous)
        cache_key = f"{entity_type_normalized}:{term_normalized}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Invalidate stale NULL entries that might succeed with improved lookup logic:
            # - diseases: MeSH fallback was added (when no UMLS key)
            # - genes: HGNC alias search was added
            should_retry = False
            if cached == "NULL":
                if entity_type_normalized in ("disease", "symptom", "procedure") and not os.environ.get("UMLS_API_KEY"):
                    should_retry = True  # MeSH fallback might find it now
                elif entity_type_normalized == "gene":
                    should_retry = True  # HGNC alias search might find it now

            if should_retry:
                logger.debug(
                    {
                        "message": f"Invalidating stale NULL cache for '{term}' - will retry lookup",
                        "term": term,
                        "entity_type": entity_type,
                    },
                    pprint=True,
                )
                del self._cache[cache_key]
                # Fall through to do fresh lookup
            elif cached == "NULL":
                return None
            else:
                result = None if cached == "NULL" else cached
                logger.debug(
                    {
                        "message": f"Sync cache hit for '{term}' ({entity_type})",
                        "term": term,
                        "entity_type": entity_type,
                        "result": result,
                    },
                    pprint=True,
                )
                return result

        # Cache miss - make synchronous HTTP request
        logger.debug(
            {
                "message": f"Sync cache miss, making HTTP request for '{term}' ({entity_type})",
                "term": term,
                "entity_type": entity_type,
            },
            pprint=True,
        )

        try:
            canonical_id: Optional[str] = None

            with httpx_sync.Client(timeout=10.0) as sync_client:
                if entity_type_normalized in ("disease", "symptom", "procedure"):
                    canonical_id = self._lookup_umls_sync(sync_client, term)
                elif entity_type_normalized == "gene":
                    canonical_id = self._lookup_hgnc_sync(sync_client, term)
                elif entity_type_normalized == "drug":
                    canonical_id = self._lookup_rxnorm_sync(sync_client, term)
                elif entity_type_normalized == "protein":
                    canonical_id = self._lookup_uniprot_sync(sync_client, term)

            # Cache the result
            self._cache[cache_key] = canonical_id if canonical_id else "NULL"
            self._save_cache()

            if canonical_id:
                logger.info(
                    {
                        "message": f"Found canonical ID for '{term}': {canonical_id}",
                        "term": term,
                        "entity_type": entity_type,
                        "canonical_id": canonical_id,
                    },
                    pprint=True,
                )

            # Sanity check: common terms should always resolve
            # This helps catch lookup bugs early
            if term_normalized == "breast cancer" and entity_type_normalized == "disease":
                assert canonical_id is not None, "'breast cancer' should always get a MeSH ID! " "Got None. Check MeSH API or normalization logic."

            return canonical_id

        except Exception as e:
            logger.warning(
                {
                    "message": f"Sync lookup failed for '{term}' ({entity_type})",
                    "term": term,
                    "entity_type": entity_type,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    def _lookup_umls_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous UMLS lookup with MeSH fallback."""
        api_key = os.environ.get("UMLS_API_KEY")

        # Try UMLS first if API key is available
        if api_key:
            try:
                url = "https://uts-ws.nlm.nih.gov/rest/search/current"
                params = {"string": term, "apiKey": api_key, "returnIdType": "concept"}
                response = client.get(url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", {}).get("results", [])
                    if results:
                        cui = results[0].get("ui")
                        if cui:
                            return cui
            except Exception as e:
                logger.warning(
                    {
                        "message": f"UMLS sync lookup failed for '{term}', trying MeSH fallback",
                        "term": term,
                        "error": str(e),
                    },
                    pprint=True,
                )

        # Fall back to MeSH (no API key required)
        return self._lookup_mesh_sync(client, term)

    def _lookup_mesh_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous MeSH lookup (no API key required).

        Same strategy as async version:
        1. Try descriptor lookup with original term
        2. Try with medical term normalization (cancer → neoplasms)
        """
        # Terms to try: original and normalized versions
        search_terms = [term]

        term_lower = term.lower()
        if "cancer" in term_lower:
            normalized = term_lower.replace("cancer", "neoplasms")
            search_terms.append(normalized.title())
        if "tumor" in term_lower or "tumour" in term_lower:
            normalized = term_lower.replace("tumor", "neoplasms").replace("tumour", "neoplasms")
            search_terms.append(normalized.title())

        for search_term in search_terms:
            result = self._try_mesh_descriptor_lookup_sync(client, search_term)
            if result:
                return result

        return None

    def _try_mesh_descriptor_lookup_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Try to find a MeSH descriptor for a term (sync version)."""
        try:
            url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
            params = {"label": term, "match": "contains", "limit": "10"}
            response = client.get(url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            return self._extract_mesh_id_from_results(data, term)
        except Exception as e:
            logger.warning(
                {
                    "message": f"MeSH sync lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    def _lookup_hgnc_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous HGNC lookup with alias fallback."""
        headers = {"Accept": "application/json"}

        # Strategy 1: Try exact symbol match
        url = f"https://rest.genenames.org/fetch/symbol/{term}"
        response = client.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if docs:
                hgnc_id = docs[0].get("hgnc_id")
                if hgnc_id:
                    return hgnc_id

        # Strategy 2: Try alias search (handles "p53" → "TP53")
        url = f"https://rest.genenames.org/search/alias_symbol/{term}"
        response = client.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            if docs:
                hgnc_id = docs[0].get("hgnc_id")
                if hgnc_id:
                    return hgnc_id

        return None

    def _lookup_rxnorm_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous RxNorm lookup."""
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={term}"
        response = client.get(url)

        if response.status_code == 200:
            data = response.json()
            id_group = data.get("idGroup", {})
            rxnorm_ids = id_group.get("rxnormId", [])
            if rxnorm_ids:
                return f"RxNorm:{rxnorm_ids[0]}"
        return None

    def _lookup_uniprot_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous UniProt lookup."""
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": f"protein_name:{term}", "format": "json", "size": "1"}
        response = client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession")
        return None

    async def close(self) -> None:
        """Close the HTTP client and save cache."""
        self._save_cache()
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - saves cache and closes client."""
        await self.close()
