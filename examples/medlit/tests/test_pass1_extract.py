"""Unit tests for Pass 1 extract helpers: vocab-in-prompt and type normalization."""

from pathlib import Path

from examples.medlit.pipeline.config_loader import load_entity_types
from examples.medlit.scripts.pass1_extract import (
    _default_system_prompt,
    _normalized_to_bundle_class,
    normalize_entity_type,
)


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "config"


def _normalized_to_bundle():
    return _normalized_to_bundle_class(load_entity_types(_config_dir()))


class TestNormalizeEntityType:
    """Test raw LLM type string -> bundle class (PascalCase)."""

    def test_biological_process_to_biological_process(self):
        """'biological process' (with space) normalizes to BiologicalProcess."""
        assert normalize_entity_type("biological process", _normalized_to_bundle()) == "BiologicalProcess"

    def test_gene_to_gene(self):
        """'gene' -> Gene."""
        assert normalize_entity_type("gene", _normalized_to_bundle()) == "Gene"

    def test_disease_lowercase(self):
        """'disease' -> Disease."""
        assert normalize_entity_type("disease", _normalized_to_bundle()) == "Disease"

    def test_hormone_and_enzyme(self):
        """PLAN3: hormone -> Hormone, enzyme -> Enzyme."""
        ntb = _normalized_to_bundle()
        assert normalize_entity_type("hormone", ntb) == "Hormone"
        assert normalize_entity_type("enzyme", ntb) == "Enzyme"

    def test_biological_process_underscore(self):
        """'biological_process' -> BiologicalProcess."""
        assert normalize_entity_type("biological_process", _normalized_to_bundle()) == "BiologicalProcess"

    def test_unknown_maps_to_other(self):
        """Unknown type maps to Other."""
        ntb = _normalized_to_bundle()
        assert normalize_entity_type("foo", ntb) == "Other"
        assert normalize_entity_type("", ntb) == "Other"

    def test_whitespace_stripped(self):
        """Whitespace is stripped before mapping."""
        assert normalize_entity_type("  gene  ", _normalized_to_bundle()) == "Gene"


class TestBuildSystemPromptWithVocab:
    """Test that vocab section is appended when vocab_entries provided."""

    def test_empty_vocab_returns_base(self):
        """None or empty vocab returns base prompt without vocab section."""
        base = _default_system_prompt(_config_dir(), None)
        base_empty = _default_system_prompt(_config_dir(), [])
        assert base == base_empty
        assert "pasireotide" not in base

    def test_vocab_entries_included_in_prompt(self):
        """Short vocab list is included in the prompt with names and types."""
        vocab = [
            {"name": "pasireotide", "type": "drug"},
            {"name": "Cushing syndrome", "type": "disease"},
        ]
        out = _default_system_prompt(_config_dir(), vocab)
        assert "pasireotide" in out
        assert "drug" in out
        assert "Cushing syndrome" in out
        assert "disease" in out
        assert "already been identified" in out or "exact names and types" in out
