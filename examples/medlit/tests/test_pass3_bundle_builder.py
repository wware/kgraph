"""Tests for Pass 3 bundle builder (medlit_merged + pass1_bundles -> kgbundle)."""

import json

import pytest

from examples.medlit.bundle_models import PerPaperBundle
from examples.medlit.pipeline.bundle_builder import (
    load_merged_output,
    load_pass1_bundles,
    run_pass3,
)
from kgbundle import BundleManifestV1


@pytest.fixture
def minimal_merged_dir(tmp_path):
    """Minimal merged dir: entities.json, relationships.json, id_map.json, synonym_cache.json."""
    entities = [
        {
            "entity_id": "HGNC:1100",
            "canonical_id": "HGNC:1100",
            "class": "Gene",
            "name": "BRCA2",
            "synonyms": [],
            "source": "extracted",
            "source_papers": ["PMC12756687"],
        },
        {
            "entity_id": "C0006142",
            "canonical_id": "C0006142",
            "class": "Disease",
            "name": "breast cancer",
            "synonyms": [],
            "source": "extracted",
            "source_papers": ["PMC12756687"],
        },
    ]
    relationships = [
        {
            "subject": "HGNC:1100",
            "predicate": "INCREASES_RISK",
            "object": "C0006142",
            "evidence_ids": ["PMC12756687:abstract:0:llm"],
            "source_papers": ["PMC12756687"],
            "confidence": 0.55,
        },
    ]
    id_map = {
        "PMC12756687": {
            "g01": "HGNC:1100",
            "e01": "C0006142",
        },
    }
    (tmp_path / "entities.json").write_text(json.dumps(entities, indent=2), encoding="utf-8")
    (tmp_path / "relationships.json").write_text(json.dumps(relationships, indent=2), encoding="utf-8")
    (tmp_path / "id_map.json").write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    (tmp_path / "synonym_cache.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def minimal_bundles_dir(tmp_path):
    """Minimal bundles dir: one paper_*.json with one relationship and matching evidence_entity."""
    bundle_data = {
        "paper": {"pmcid": "PMC12756687", "title": "Test", "authors": []},
        "entities": [
            {"id": "g01", "class": "Gene", "name": "BRCA2", "synonyms": [], "source": "extracted"},
            {"id": "e01", "class": "Disease", "name": "breast cancer", "synonyms": [], "source": "extracted"},
        ],
        "evidence_entities": [
            {
                "id": "PMC12756687:abstract:0:llm",
                "class": "Evidence",
                "paper_id": "PMC12756687",
                "text_span_id": "PMC12756687:abstract:0",
                "text": "BRCA2 increases risk of breast cancer",
                "confidence": 0.95,
                "extraction_method": "llm",
                "source": "extracted",
            },
        ],
        "relationships": [
            {
                "subject": "g01",
                "predicate": "INCREASES_RISK",
                "object": "e01",
                "evidence_ids": ["PMC12756687:abstract:0:llm"],
                "source_papers": ["PMC12756687"],
                "confidence": 0.55,
            },
        ],
        "notes": [],
    }
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    (bundles_dir / "paper_PMC12756687.json").write_text(json.dumps(bundle_data, indent=2), encoding="utf-8")
    return bundles_dir


def test_run_pass3_produces_bundle_files(minimal_merged_dir, minimal_bundles_dir, tmp_path):
    """run_pass3 writes entities.jsonl, relationships.jsonl, evidence.jsonl, mentions.jsonl, manifest.json."""
    output_dir = tmp_path / "out"
    run_pass3(minimal_merged_dir, minimal_bundles_dir, output_dir)

    assert (output_dir / "entities.jsonl").exists()
    assert (output_dir / "relationships.jsonl").exists()
    assert (output_dir / "evidence.jsonl").exists()
    assert (output_dir / "mentions.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "doc_assets.jsonl").exists()
    assert (output_dir / "docs" / "README.md").exists()

    with open(output_dir / "entities.jsonl", encoding="utf-8") as f:
        entity_lines = [line for line in f if line.strip()]
    assert len(entity_lines) >= 1

    with open(output_dir / "relationships.jsonl", encoding="utf-8") as f:
        rel_lines = [line for line in f if line.strip()]
    assert len(rel_lines) >= 1

    manifest_data = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = BundleManifestV1.model_validate(manifest_data)
    assert manifest.entities.path == "entities.jsonl"
    assert manifest.domain == "medlit"


def test_entity_row_has_usage_and_status(minimal_merged_dir, minimal_bundles_dir, tmp_path):
    """EntityRow in entities.jsonl has entity_id, status, usage_count/total_mentions from bundle scan."""
    output_dir = tmp_path / "out"
    run_pass3(minimal_merged_dir, minimal_bundles_dir, output_dir)

    with open(output_dir / "entities.jsonl", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 1
    ent = json.loads(lines[0])
    assert "entity_id" in ent
    assert ent["status"] in ("canonical", "provisional")
    assert "usage_count" in ent
    assert "total_mentions" in ent


def test_evidence_row_relationship_key_uses_merge_keys(minimal_merged_dir, minimal_bundles_dir, tmp_path):
    """EvidenceRow relationship_key uses merge keys (from id_map), not local ids."""
    output_dir = tmp_path / "out"
    run_pass3(minimal_merged_dir, minimal_bundles_dir, output_dir)

    with open(output_dir / "evidence.jsonl", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) >= 1
    ev = json.loads(lines[0])
    # relationship_key must be subject:predicate:object with merge keys
    rk = ev["relationship_key"]
    parts = rk.split(":")
    assert len(parts) >= 3
    # Merge keys from id_map: g01 -> HGNC:1100, e01 -> C0006142
    assert "g01" not in rk and "e01" not in rk
    assert "HGNC:1100" in rk or "C0006142" in rk


def test_run_pass3_raises_when_id_map_missing(tmp_path, minimal_bundles_dir):
    """If id_map.json is missing in merged_dir, run_pass3 raises FileNotFoundError."""
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "entities.json").write_text("[]", encoding="utf-8")
    (merged_dir / "relationships.json").write_text("[]", encoding="utf-8")
    # No id_map.json

    output_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError) as exc_info:
        run_pass3(merged_dir, minimal_bundles_dir, output_dir)
    assert "id_map.json" in str(exc_info.value)


def test_load_merged_output_requires_id_map(tmp_path):
    """load_merged_output raises FileNotFoundError when id_map.json is missing."""
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    (merged_dir / "entities.json").write_text("[]", encoding="utf-8")
    (merged_dir / "relationships.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_merged_output(merged_dir)
    assert "id_map.json" in str(exc_info.value)


def test_load_pass1_bundles(minimal_bundles_dir):
    """load_pass1_bundles returns list of (paper_id, PerPaperBundle)."""
    bundles = load_pass1_bundles(minimal_bundles_dir)
    assert len(bundles) == 1
    paper_id, bundle = bundles[0]
    assert paper_id == "PMC12756687"
    assert isinstance(bundle, PerPaperBundle)
    assert len(bundle.relationships) == 1
    assert len(bundle.evidence_entities) == 1
