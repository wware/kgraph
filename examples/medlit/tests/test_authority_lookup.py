"""Tests for canonical ID authority lookup.

Tests the matching logic for DBPedia and other ontology lookups.
"""

import pytest

from examples.medlit.pipeline.authority_lookup import CanonicalIdLookup


class TestDBPediaLabelMatching:
    """Test the DBPedia label matching logic."""

    @pytest.fixture
    def lookup(self):
        """Create a CanonicalIdLookup instance for testing."""
        return CanonicalIdLookup()

    # --- Tests for matches that SHOULD succeed ---

    def test_exact_match(self, lookup):
        """Exact match should succeed."""
        assert lookup._dbpedia_label_matches("breast cancer", "Breast Cancer")

    def test_term_contained_in_label(self, lookup):
        """Term contained in label should succeed."""
        assert lookup._dbpedia_label_matches("mitochondria", "<B>Mitochondria</B> (song)")

    def test_label_contained_in_term(self, lookup):
        """Label contained in term should succeed."""
        assert lookup._dbpedia_label_matches("breast cancer syndrome", "Breast Cancer")

    def test_label_starts_with_term(self, lookup):
        """Label starting with term should succeed."""
        assert lookup._dbpedia_label_matches("breast", "Breast cancer")

    def test_common_prefix_singular_plural(self, lookup):
        """Common 6-char prefix should succeed (handles singular/plural)."""
        # mitochondria vs mitochondrion - same first 6 chars "mitoch"
        assert lookup._dbpedia_label_matches("mitochondria", "<B>Mitochondrion</B>")

    def test_html_tags_stripped(self, lookup):
        """HTML bold tags should be stripped from labels."""
        assert lookup._dbpedia_label_matches("mitochondria", "<B>Mitochondria</B>")
        assert lookup._dbpedia_label_matches("cancer", "<B>Cancer</B>")

    def test_case_insensitive(self, lookup):
        """Matching should be case-insensitive."""
        assert lookup._dbpedia_label_matches("BRCA1", "brca1")
        assert lookup._dbpedia_label_matches("brca1", "BRCA1")

    # --- Tests for matches that SHOULD fail ---

    def test_garbage_match_insect(self, lookup):
        """Garbage match 'HER2-enriched' → 'Insect' should fail."""
        assert not lookup._dbpedia_label_matches("HER2-enriched", "Insect")

    def test_garbage_match_animal(self, lookup):
        """Garbage match 'basal-like' → 'Animal' should fail."""
        assert not lookup._dbpedia_label_matches("basal-like", "Animal")

    def test_unrelated_terms(self, lookup):
        """Completely unrelated terms should fail."""
        assert not lookup._dbpedia_label_matches("diabetes", "Python programming")
        assert not lookup._dbpedia_label_matches("gene", "Music")

    def test_substring_match_allowed(self, lookup):
        """Substring matching is allowed (term in label)."""
        # "test" is contained in "testing" - this is accepted
        assert lookup._dbpedia_label_matches("test", "Testing framework")
        # "car" is contained in "cardinal" - this is accepted
        # (for medical terms, coincidental substrings are rare)
        assert lookup._dbpedia_label_matches("car", "Cardinal")

    def test_no_overlap_fails(self, lookup):
        """Terms with no overlap should fail."""
        # No substring match, no common prefix
        assert not lookup._dbpedia_label_matches("xyz", "abc")
        assert not lookup._dbpedia_label_matches("hello", "world")


class TestMeSHTermNormalization:
    """Test MeSH term normalization (cancer → neoplasms)."""

    @pytest.fixture
    def lookup(self):
        """Create a CanonicalIdLookup instance for testing."""
        return CanonicalIdLookup()

    def test_mesh_id_extraction(self, lookup):
        """Test extracting MeSH ID from API results."""
        # Simulated API response
        data = [
            {"resource": "http://id.nlm.nih.gov/mesh/D001943", "label": "Breast Neoplasms"},
            {"resource": "http://id.nlm.nih.gov/mesh/D018567", "label": "Breast Neoplasms, Male"},
        ]

        # Should find D001943 for "breast neoplasms"
        result = lookup._extract_mesh_id_from_results(data, "breast neoplasms")
        assert result == "MeSH:D001943"

    def test_mesh_id_extraction_word_order(self, lookup):
        """Test MeSH extraction handles word order differences."""
        data = [
            {"resource": "http://id.nlm.nih.gov/mesh/D001943", "label": "Breast Neoplasms"},
        ]

        # "neoplasms breast" should still match "Breast Neoplasms"
        result = lookup._extract_mesh_id_from_results(data, "neoplasms breast")
        assert result == "MeSH:D001943"

    def test_mesh_id_extraction_no_match(self, lookup):
        """Test MeSH extraction returns None for no match."""
        data = [
            {"resource": "http://id.nlm.nih.gov/mesh/D001943", "label": "Breast Neoplasms"},
        ]

        # "lung cancer" should not match "Breast Neoplasms"
        result = lookup._extract_mesh_id_from_results(data, "lung cancer")
        assert result is None

    def test_mesh_id_extraction_empty_data(self, lookup):
        """Test MeSH extraction handles empty data."""
        result = lookup._extract_mesh_id_from_results([], "anything")
        assert result is None
