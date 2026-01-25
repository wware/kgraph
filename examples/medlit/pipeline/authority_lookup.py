"""Canonical ID lookup from medical ontology authorities.

Provides lookup functionality for canonical IDs from various medical ontology
sources: UMLS, HGNC, RxNorm, and UniProt.

Features persistent caching to avoid repeated API calls across runs.
"""

import os
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment, misc]

from kgraph.canonical_id import (
    CanonicalId,
    CanonicalIdLookupInterface,
    JsonFileCanonicalIdCache,
)
from kgraph.logging import setup_logging

from .canonical_urls import build_canonical_url

# Module-level logger for sync methods
logger = setup_logging()


class CanonicalIdLookup(CanonicalIdLookupInterface):
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
                       to "canonical_id_cache.json" in current directory.
        """
        if httpx is None:
            raise ImportError("httpx is required for authority lookup. Install with: uv add httpx")

        self.client = httpx.AsyncClient(timeout=10.0)
        self.umls_api_key = umls_api_key or os.getenv("UMLS_API_KEY")
        self.cache_file = cache_file or Path("canonical_id_cache.json")
        self._cache = JsonFileCanonicalIdCache(cache_file=self.cache_file)
        self._cache.load(str(self.cache_file))
        self._lookup_count = 0  # Track lookups for periodic saves

    def _save_cache(self, force: bool = False) -> None:
        """Save cache to disk.

        Args:
            force: If True, save even if cache is not marked dirty (for emergency saves).
        """
        self._cache.save(str(self.cache_file))

    async def lookup(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        """Look up canonical ID for a medical term (interface method).

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (disease, gene, drug, protein, etc.)

        Returns:
            CanonicalId if found, None otherwise
        """
        canonical_id_str = await self.lookup_canonical_id(term, entity_type)
        if canonical_id_str:
            url = build_canonical_url(canonical_id_str, entity_type=entity_type)
            return CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
        return None

    async def lookup_canonical_id(self, term: str, entity_type: str) -> Optional[str]:
        """Look up canonical ID for a medical term.

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (disease, gene, drug, protein, etc.)

        Returns:
            Canonical ID string if found, None otherwise
        """
        logger = setup_logging()

        # Check cache first
        cached = self._cache.fetch(term, entity_type)
        if cached is not None:
            logger.debug(
                {
                    "message": f"Cache hit for {term} ({entity_type})",
                    "cached_id": cached.id,
                },
                pprint=True,
            )
            return cached.id

        # Check if known bad
        if self._cache.is_known_bad(term, entity_type):
            logger.debug(
                {
                    "message": f"Known bad entry for '{term}' ({entity_type}), skipping lookup",
                    "term": term,
                    "entity_type": entity_type,
                },
                pprint=True,
            )
            return None

        logger.debug(
            {
                "message": f"Cache miss for '{term}' (type: {entity_type}), looking up",
                "term": term,
                "entity_type": entity_type,
            },
            pprint=True,
        )

        # Route to appropriate authority based on entity type
        canonical_id_str: Optional[str] = None

        if entity_type in ("disease", "symptom", "procedure"):
            canonical_id_str = await self._lookup_umls(term)
        elif entity_type == "gene":
            canonical_id_str = await self._lookup_hgnc(term)
        elif entity_type == "drug":
            canonical_id_str = await self._lookup_rxnorm(term)
        elif entity_type == "protein":
            canonical_id_str = await self._lookup_uniprot(term)

        # Fallback to DBPedia for any entity type if specialized lookup failed
        if canonical_id_str is None:
            canonical_id_str = await self._lookup_dbpedia(term)

        # Build CanonicalId object with URL
        if canonical_id_str:
            url = build_canonical_url(canonical_id_str, entity_type=entity_type)
            canonical_id = CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
            self._cache.store(term, entity_type, canonical_id)
            logger.info(
                {
                    "message": f"Found canonical ID for '{term}': {canonical_id_str}",
                    "term": term,
                    "entity_type": entity_type,
                    "canonical_id": canonical_id_str,
                    "url": url,
                },
                pprint=True,
            )
        else:
            # Mark as known bad
            self._cache.mark_known_bad(term, entity_type)
            logger.debug(
                {
                    "message": f"No canonical ID found for '{term}' (type: {entity_type})",
                    "term": term,
                    "entity_type": entity_type,
                },
                pprint=True,
            )

        # Periodically save cache (every 100 new entries to avoid losing work)
        self._lookup_count += 1
        if self._lookup_count % 100 == 0:
            self._save_cache()

        return canonical_id_str

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

    def _normalize_mesh_search_terms(self, term: str) -> list[str]:
        """Generate normalized search terms for MeSH lookup.

        MeSH uses formal terminology, so we normalize common informal terms.
        Returns a list of search terms to try, in order of preference.

        Args:
            term: Original search term

        Returns:
            List of normalized search terms (original first, then normalized variants)
        """
        search_terms = [term]
        term_lower = term.lower()

        # Common medical term normalizations (MeSH uses formal terminology)
        if "cancer" in term_lower:
            # "breast cancer" → "breast neoplasms"
            normalized = term_lower.replace("cancer", "neoplasms")
            search_terms.append(normalized.title())
        if "tumor" in term_lower or "tumour" in term_lower:
            normalized = term_lower.replace("tumor", "neoplasms").replace("tumour", "neoplasms")
            search_terms.append(normalized.title())

        return search_terms

    async def _lookup_mesh(self, term: str) -> Optional[str]:
        """Look up MeSH descriptor ID for a disease/symptom term.

        MeSH (Medical Subject Headings) is freely accessible without API key.
        Returns MeSH descriptor IDs like "MeSH:D001943" (breast neoplasms).

        Strategy:
        1. Try descriptor lookup with original term and normalized variants
        2. Collect all results and score them together
        3. Return the best match across all search terms
        """
        # Get all normalized search terms
        search_terms = self._normalize_mesh_search_terms(term)

        # Collect results from all search terms
        all_results = []
        seen_mesh_ids = set()

        for search_term in search_terms:
            results = await self._try_mesh_descriptor_lookup_all(search_term)
            for result in results:
                mesh_id = result.get("resource", "").split("/mesh/")[-1] if "/mesh/D" in result.get("resource", "") else None
                if mesh_id and mesh_id not in seen_mesh_ids:
                    all_results.append(result)
                    seen_mesh_ids.add(mesh_id)

        if not all_results:
            return None

        # Score all results together, considering all search terms
        return self._extract_mesh_id_from_results(all_results, search_terms)

    def _extract_mesh_id_from_results(self, data: list, search_terms: str | list[str]) -> Optional[str]:
        """Extract MeSH descriptor ID from API results, preferring best matches.

        Scores results based on how well they match any of the provided search terms.
        This allows normalized terms (e.g., "breast neoplasms") to score well even
        when the original search was "breast cancer".

        Scoring strategy:
        1. Exact match (case-insensitive) gets highest score
        2. Exact word match (all words present) gets high score
        3. Prefer shorter labels (more general terms) over longer ones (complications)
        4. Prefer matches where term is at the start of the label
        5. Penalize matches that are much longer than the search term (likely complications)
        6. Prefer matches to earlier search terms (original > normalized)

        Args:
            data: List of result dictionaries from MeSH API
            search_terms: Single search term (str) or list of search terms tried
                         (original first, then normalized variants). If a single string
                         is provided, it's treated as the only search term.
        """
        # Normalize input - accept either string or list
        if isinstance(search_terms, str):
            # If single string provided, generate normalized variants automatically
            search_terms_list = self._normalize_mesh_search_terms(search_terms)
        else:
            search_terms_list = search_terms

        # Normalize all search terms for comparison
        normalized_search_terms = [s.lower().strip() for s in search_terms_list]
        search_term_word_sets = [set(s.split()) for s in normalized_search_terms]

        scored_results = []

        for result in data:
            label = result.get("label", "").strip()
            label_lower = label.lower()
            label_words = set(label_lower.split())
            label_word_count = len(label_words)

            resource_uri = result.get("resource", "")
            if "/mesh/D" not in resource_uri:
                continue

            mesh_id = resource_uri.split("/mesh/")[-1]

            # Calculate match score - try each search term and take the best match
            best_score = 0.0
            best_term_index = len(search_terms_list)  # Prefer earlier terms
            best_term_lower = ""
            best_term_word_count = 0
            original_term_words = search_term_word_sets[0] if search_term_word_sets else set()

            for term_index, (term_lower, term_words) in enumerate(zip(normalized_search_terms, search_term_word_sets)):
                term_word_count = len(term_words)
                score = 0.0

                # 1. Exact match (case-insensitive) - highest priority
                if term_lower == label_lower:
                    score = 1000.0 - term_index  # Slight penalty for normalized terms
                    # For exact matches, apply additional scoring
                    # Prefer shorter labels (exact match on shorter = more general)
                    if label_word_count <= 2:
                        score += 50.0  # Bonus for short, general terms
                    if score > best_score:
                        best_score = score
                        best_term_index = term_index
                        best_term_lower = term_lower
                        best_term_word_count = term_word_count
                    continue  # This is the best possible match for this term

                # 2. All words from term are in label (exact word match)
                common_words = term_words & label_words

                # Require strong word overlap to prevent false matches
                # For single-word terms, require exact match
                if term_word_count == 1:
                    if len(common_words) == 1:
                        score += 500.0 - term_index * 10
                    else:
                        continue  # No match
                elif len(common_words) == term_word_count:
                    score += 500.0 - term_index * 10  # Bonus for earlier search terms
                elif len(common_words) >= term_word_count - 1:
                    # All words except one match - but check if it's a valid match
                    # For normalized terms (index > 0), require that original term words also match
                    if term_index > 0:
                        # Check if original term's words (non-normalized) match the label
                        original_common = original_term_words & label_words
                        if len(original_common) < len(original_term_words) - 1:
                            # Original term doesn't match well - likely false match
                            continue
                    score += 200.0 - term_index * 5
                else:
                    # For 2+ word terms, require at least term_word_count - 1 words to match
                    continue  # Skip if not enough words match

                # Update best score if this term matches better
                if score > best_score or (score == best_score and term_index < best_term_index):
                    best_score = score
                    best_term_index = term_index
                    best_term_lower = term_lower
                    best_term_word_count = term_word_count

            if best_score == 0.0:
                continue  # No match found for any search term

            # Apply additional scoring based on the best matching term
            # 3. Prefer shorter labels (more general terms)
            # Shorter labels indicate more general concepts
            length_ratio = len(best_term_lower) / max(len(label_lower), 1)
            if length_ratio > 0.8:  # Term is almost as long as label
                best_score += 100.0
            elif length_ratio > 0.6:
                best_score += 50.0
            elif length_ratio < 0.4:  # Label is much longer (likely a complication)
                best_score -= 100.0

            # 4. Prefer matches where term appears at the start
            if label_lower.startswith(best_term_lower):
                best_score += 150.0
            elif best_term_lower in label_lower:
                best_score += 50.0

            # 5. Penalize results with many extra words (likely complications)
            extra_words = label_word_count - best_term_word_count
            if extra_words > 1:
                best_score -= 100.0 * extra_words  # Heavy penalty for extra words (complications)
            elif extra_words == 1:
                best_score -= 50.0  # Moderate penalty for one extra word

            # 6. Special handling: if label is much longer, it's likely a complication
            # Penalize labels that are >50% longer than the search term
            if len(label_lower) > len(best_term_lower) * 1.5:
                best_score -= 150.0

            # 7. Bonus for word count similarity (exact match preferred)
            if label_word_count == best_term_word_count:
                best_score += 100.0
            elif label_word_count == best_term_word_count + 1:
                best_score += 25.0  # Slight bonus for one extra word (handles plurals)

            if best_score > 0:
                scored_results.append((best_score, mesh_id, label))

        if not scored_results:
            return None

        # Sort by score (descending) and return the best match
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_mesh_id, best_label = scored_results[0]

        logger.debug(
            {
                "message": f"MeSH match selected for '{search_terms_list[0]}'",
                "search_terms": search_terms_list,
                "selected": best_label,
                "mesh_id": best_mesh_id,
                "score": best_score,
                "alternatives": [(score, label) for score, _, label in scored_results[:3]],
            },
            pprint=True,
        )

        return f"MeSH:{best_mesh_id}"

    async def _try_mesh_descriptor_lookup_all(self, term: str) -> list[dict]:
        """Try to find MeSH descriptors for a term, returning all results.

        Args:
            term: Search term

        Returns:
            List of result dictionaries from MeSH API
        """
        try:
            url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
            params = {"label": term, "match": "contains", "limit": "10"}
            response = await self.client.get(url, params=params)

            if response.status_code != 200:
                return []

            data = response.json()
            if not data:
                return []

            return data
        except Exception as e:
            logger.warning(
                {
                    "message": f"MeSH lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return []

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
                    uniprot_id = results[0].get("primaryAccession")  # e.g., "P38398"
                    return f"UniProt:{uniprot_id}" if uniprot_id else None
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

    def _dbpedia_label_matches(self, term: str, label: str) -> bool:
        """Check if a DBPedia label is a good match for the search term."""
        import re

        # Strip HTML tags from label (DBPedia returns <B>...</B> highlighting)
        label_clean = re.sub(r"<[^>]+>", "", label).lower().strip()
        term_lower = term.lower().strip()

        # Extract first word (for prefix matching)
        term_first = term_lower.replace("-", " ").split()[0] if term_lower else ""
        label_first = label_clean.replace("-", " ").split()[0] if label_clean else ""

        # Accept if:
        # 1. Term is contained in label (or vice versa)
        # 2. First words share a common prefix of 6+ chars (handles mitochondria/mitochondrion)
        # 3. Label starts with the term (e.g., "breast cancer" matches "breast cancer syndrome")
        common_prefix = len(term_first) >= 6 and len(label_first) >= 6 and term_first[:6] == label_first[:6]

        return term_lower in label_clean or label_clean in term_lower or label_clean.startswith(term_lower) or common_prefix

    async def _lookup_dbpedia(self, term: str) -> Optional[str]:
        """Look up DBPedia URI as fallback for any entity type.

        DBPedia is a general knowledge base extracted from Wikipedia.
        Used as a fallback when specialized medical ontologies don't find a match.

        Only accepts results where the label closely matches the search term
        to avoid garbage matches like "HER2-enriched" → "Insect".
        """
        try:
            # DBPedia Lookup API (no authentication required!)
            url = "https://lookup.dbpedia.org/api/search"
            params = {"query": term.lower(), "format": "json", "maxResults": "5"}
            response = await self.client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])

                for doc in docs:
                    # Get label (returned as list)
                    label_list = doc.get("label", [])
                    label = label_list[0] if label_list else ""

                    if self._dbpedia_label_matches(term, label):
                        resource = doc.get("resource", [])
                        if resource:
                            uri = resource[0] if isinstance(resource, list) else resource
                            return f"DBPedia:{uri.split('/')[-1]}"

            return None
        except Exception as e:
            logger.warning(
                {
                    "message": f"DBPedia lookup failed for '{term}'",
                    "term": term,
                    "error": str(e),
                },
                pprint=True,
            )
            return None

    def lookup_sync(self, term: str, entity_type: str) -> Optional[CanonicalId]:
        """Synchronous lookup (interface method).

        Args:
            term: The entity name/mention text
            entity_type: Type of entity (disease, gene, drug, protein, etc.)

        Returns:
            CanonicalId if found, None otherwise
        """
        canonical_id_str = self.lookup_canonical_id_sync(term, entity_type)
        if canonical_id_str:
            url = build_canonical_url(canonical_id_str, entity_type=entity_type)
            return CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
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
        cached = self._cache.fetch(term, entity_type)
        if cached is not None:
            logger.debug(
                {
                    "message": f"Sync cache hit for '{term}' ({entity_type})",
                    "term": term,
                    "entity_type": entity_type,
                    "result": cached.id,
                },
                pprint=True,
            )
            return cached.id

        # Check if known bad
        if self._cache.is_known_bad(term, entity_type):
            return None

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
            canonical_id_str: Optional[str] = None

            with httpx_sync.Client(timeout=10.0) as sync_client:
                if entity_type_normalized in ("disease", "symptom", "procedure"):
                    canonical_id_str = self._lookup_umls_sync(sync_client, term)
                elif entity_type_normalized == "gene":
                    canonical_id_str = self._lookup_hgnc_sync(sync_client, term)
                elif entity_type_normalized == "drug":
                    canonical_id_str = self._lookup_rxnorm_sync(sync_client, term)
                elif entity_type_normalized == "protein":
                    canonical_id_str = self._lookup_uniprot_sync(sync_client, term)

                # Fallback to DBPedia if specialized lookup failed
                if canonical_id_str is None:
                    canonical_id_str = self._lookup_dbpedia_sync(sync_client, term)

            # Build CanonicalId object with URL and cache it
            if canonical_id_str:
                url = build_canonical_url(canonical_id_str, entity_type=entity_type)
                canonical_id = CanonicalId(id=canonical_id_str, url=url, synonyms=(term,))
                self._cache.store(term, entity_type, canonical_id)
            else:
                # Mark as known bad
                self._cache.mark_known_bad(term, entity_type)

            self._save_cache()

            # Sanity check: common terms should always resolve
            # This helps catch lookup bugs early
            if term_normalized == "breast cancer" and entity_type_normalized == "disease":
                assert canonical_id_str is not None, "'breast cancer' should always get a MeSH ID! " "Got None. Check MeSH API or normalization logic."

            return canonical_id_str

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
        """Synchronous MeSH lookup with term normalization.

        Uses the same multi-term approach as async version.
        """
        # Get all normalized search terms
        search_terms = self._normalize_mesh_search_terms(term)

        # Collect results from all search terms
        all_results = []
        seen_mesh_ids = set()

        try:
            for search_term in search_terms:
                url = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
                params = {"label": search_term, "match": "contains", "limit": "10"}
                response = client.get(url, params=params)

                if response.status_code != 200:
                    continue

                data = response.json()
                if not data:
                    continue

                for result in data:
                    mesh_id = result.get("resource", "").split("/mesh/")[-1] if "/mesh/D" in result.get("resource", "") else None
                    if mesh_id and mesh_id not in seen_mesh_ids:
                        all_results.append(result)
                        seen_mesh_ids.add(mesh_id)
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

        if not all_results:
            return None

        # Score all results together, considering all search terms
        return self._extract_mesh_id_from_results(all_results, search_terms)

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
                uniprot_id = results[0].get("primaryAccession")
                return f"UniProt:{uniprot_id}" if uniprot_id else None
        return None

    def _lookup_dbpedia_sync(self, client: "httpx.Client", term: str) -> Optional[str]:
        """Synchronous DBPedia lookup as fallback with validation."""
        try:
            url = "https://lookup.dbpedia.org/api/search"
            params = {"query": term.lower(), "format": "json", "maxResults": "5"}
            response = client.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])

                for doc in docs:
                    label_list = doc.get("label", [])
                    label = label_list[0] if label_list else ""

                    if self._dbpedia_label_matches(term, label):
                        resource = doc.get("resource", [])
                        if resource:
                            uri = resource[0] if isinstance(resource, list) else resource
                            return f"DBPedia:{uri.split('/')[-1]}"
            return None
        except Exception:
            return None

    async def close(self) -> None:
        """Close the HTTP client and save cache."""
        self._save_cache(force=True)
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - saves cache and closes client."""
        await self.close()
