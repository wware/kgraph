"""
Pass 3: Build kgbundle from medlit_merged and pass1_bundles.

Reads merged output (entities.json, relationships.json, id_map.json, synonym_cache.json)
and Pass 1 paper_*.json bundles; writes a kgbundle directory loadable by kgserver.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kgbundle import (
    BundleFile,
    BundleManifestV1,
    DocAssetRow,
    EntityRow,
    EvidenceRow,
    MentionRow,
    RelationshipRow,
)

from examples.medlit.bundle_models import PerPaperBundle
from examples.medlit.pipeline.canonical_urls import build_canonical_url


def load_merged_output(merged_dir: Path) -> tuple[list[dict], list[dict], dict, dict]:
    """Load merged Pass 2 output and id_map.

    Returns (entities, relationships, id_map, synonym_cache).
    Raises FileNotFoundError if id_map.json is missing.
    """
    entities_path = merged_dir / "entities.json"
    relationships_path = merged_dir / "relationships.json"
    id_map_path = merged_dir / "id_map.json"
    synonym_cache_path = merged_dir / "synonym_cache.json"

    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map.json not found in {merged_dir}. Run Pass 2 so that merged_dir contains id_map.json.")

    entities: list[dict] = []
    if entities_path.exists():
        with open(entities_path, encoding="utf-8") as f:
            entities = json.load(f)

    relationships: list[dict] = []
    if relationships_path.exists():
        with open(relationships_path, encoding="utf-8") as f:
            relationships = json.load(f)

    with open(id_map_path, encoding="utf-8") as f:
        id_map: dict[str, dict[str, str]] = json.load(f)

    synonym_cache: dict = {}
    if synonym_cache_path.exists():
        with open(synonym_cache_path, encoding="utf-8") as f:
            synonym_cache = json.load(f)

    return (entities, relationships, id_map, synonym_cache)


def load_pass1_bundles(bundles_dir: Path) -> list[tuple[str, PerPaperBundle]]:
    """Load all paper_*.json bundles from bundles_dir. Returns list of (paper_id, bundle)."""
    result: list[tuple[str, PerPaperBundle]] = []
    for path in sorted(bundles_dir.glob("paper_*.json")):
        paper_id = path.stem.replace("paper_", "")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        bundle = PerPaperBundle.from_bundle_dict(data)
        if bundle.paper.pmcid:
            paper_id = bundle.paper.pmcid
        result.append((paper_id, bundle))
    return result


def _entity_usage_from_bundles(
    bundles: list[tuple[str, PerPaperBundle]],
    id_map: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Compute usage_count, total_mentions, supporting_documents, first_seen_* per merge_key."""
    # merge_key -> usage_count, total_mentions, supporting_documents, first_seen_document, first_seen_section
    by_key: dict[str, dict[str, Any]] = {}

    for paper_id, bundle in bundles:
        paper_map = id_map.get(paper_id) or {}
        for rel in bundle.relationships:
            sub_merge = paper_map.get(rel.subject)
            obj_merge = paper_map.get(rel.object_id)
            evidence_ids = rel.evidence_ids or []
            for _ in evidence_ids:
                if sub_merge:
                    by_key.setdefault(
                        sub_merge,
                        {
                            "usage_count": 0,
                            "total_mentions": 0,
                            "supporting_documents": [],
                            "first_seen_document": None,
                            "first_seen_section": None,
                        },
                    )
                    rec = by_key[sub_merge]
                    rec["total_mentions"] += 1
                    if paper_id not in rec["supporting_documents"]:
                        rec["supporting_documents"].append(paper_id)
                    if rec["first_seen_document"] is None:
                        rec["first_seen_document"] = paper_id
                        rec["first_seen_section"] = None  # EvidenceEntityRow has no section
                if obj_merge:
                    by_key.setdefault(
                        obj_merge,
                        {
                            "usage_count": 0,
                            "total_mentions": 0,
                            "supporting_documents": [],
                            "first_seen_document": None,
                            "first_seen_section": None,
                        },
                    )
                    rec = by_key[obj_merge]
                    rec["total_mentions"] += 1
                    if paper_id not in rec["supporting_documents"]:
                        rec["supporting_documents"].append(paper_id)
                    if rec["first_seen_document"] is None:
                        rec["first_seen_document"] = paper_id
                        rec["first_seen_section"] = None

    for rec in by_key.values():
        rec["usage_count"] = len(rec["supporting_documents"])

    return by_key


def _merged_entity_to_entity_row(ent: dict, usage: dict[str, Any], created_at: str) -> EntityRow:
    """Convert merged entity dict to EntityRow."""
    entity_id = ent["entity_id"]
    entity_type = (ent.get("class") or "unknown").lower()
    name = ent.get("name")
    has_canonical = bool(ent.get("canonical_id"))
    status = "canonical" if has_canonical else "provisional"
    confidence = 0.8 if has_canonical else 0.5
    usage_count = usage.get("usage_count", 0)
    total_mentions = usage.get("total_mentions", 0)
    supporting_documents = usage.get("supporting_documents", [])
    first_seen_document = usage.get("first_seen_document")
    first_seen_section = usage.get("first_seen_section")
    canonical_url = None
    if ent.get("canonical_id"):
        canonical_url = build_canonical_url(ent["canonical_id"], entity_type)
    properties: dict[str, Any] = {}
    synonyms = ent.get("synonyms")
    if synonyms is not None:
        properties["synonyms"] = list(synonyms)

    return EntityRow(
        entity_id=entity_id,
        entity_type=entity_type,
        name=name,
        status=status,
        confidence=confidence,
        usage_count=usage_count,
        created_at=created_at,
        source="medlit:llm",
        canonical_url=canonical_url,
        properties=properties,
        first_seen_document=first_seen_document,
        first_seen_section=first_seen_section,
        total_mentions=total_mentions,
        supporting_documents=supporting_documents,
    )


def _relationship_evidence_stats(
    merged_rels: list[dict],
    bundles: list[tuple[str, PerPaperBundle]],
    id_map: dict[str, dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """For each (sub, pred, obj) merge key, compute evidence_count, strongest_evidence_quote, evidence_confidence_avg."""
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rel in merged_rels:
        key = (rel["subject"], rel["predicate"], rel["object"])
        result[key] = {
            "evidence_count": 0,
            "strongest_evidence_quote": None,
            "evidence_confidence_avg": None,
        }

    # Collect evidence text+confidence per (sub, pred, obj) from bundles
    key_to_evidence: dict[tuple[str, str, str], list[tuple[str, float]]] = {(rel["subject"], rel["predicate"], rel["object"]): [] for rel in merged_rels}

    for paper_id, bundle in bundles:
        paper_map = id_map.get(paper_id) or {}
        ev_by_id = {ev.id: ev for ev in bundle.evidence_entities}
        for bundle_rel in bundle.relationships:
            sub_merge = paper_map.get(bundle_rel.subject)
            obj_merge = paper_map.get(bundle_rel.object_id)
            if not sub_merge or not obj_merge:
                continue
            key = (sub_merge, bundle_rel.predicate, obj_merge)
            if key not in key_to_evidence:
                continue
            for eid in bundle_rel.evidence_ids or []:
                ev = ev_by_id.get(eid)
                if ev is not None:
                    text = ev.text or ""
                    key_to_evidence[key].append((text, ev.confidence))

    for key, pairs in key_to_evidence.items():
        if not pairs:
            continue
        result[key]["evidence_count"] = len(pairs)
        # Strongest = text from pair with max confidence
        best = max(pairs, key=lambda p: p[1])
        result[key]["strongest_evidence_quote"] = best[0] or None
        avg = sum(p[1] for p in pairs) / len(pairs)
        result[key]["evidence_confidence_avg"] = round(avg, 4)

    return result


def _merged_rel_to_relationship_row(
    rel: dict,
    stats: dict[tuple[str, str, str], dict[str, Any]],
    created_at: str,
) -> RelationshipRow:
    """Convert merged relationship dict to RelationshipRow."""
    key = (rel["subject"], rel["predicate"], rel["object"])
    s = stats.get(key, {})
    return RelationshipRow(
        subject_id=rel["subject"],
        object_id=rel["object"],
        predicate=rel["predicate"],
        confidence=rel.get("confidence"),
        source_documents=rel.get("source_papers", []),
        created_at=created_at,
        properties=rel.get("properties", {}),
        evidence_count=s.get("evidence_count", 0),
        strongest_evidence_quote=s.get("strongest_evidence_quote"),
        evidence_confidence_avg=s.get("evidence_confidence_avg"),
    )


def _build_evidence_rows(
    bundles: list[tuple[str, PerPaperBundle]],
    id_map: dict[str, dict[str, str]],
    merged_relationships: list[dict],
) -> list[EvidenceRow]:
    """Build EvidenceRow list from bundles; relationship_key uses merge keys. Offsets stubbed (0, len(text))."""
    merged_keys = {(r["subject"], r["predicate"], r["object"]) for r in merged_relationships}
    rows: list[EvidenceRow] = []

    for paper_id, bundle in bundles:
        paper_map = id_map.get(paper_id) or {}
        ev_by_id = {ev.id: ev for ev in bundle.evidence_entities}
        for rel in bundle.relationships:
            sub_merge = paper_map.get(rel.subject)
            obj_merge = paper_map.get(rel.object_id)
            if not sub_merge or not obj_merge:
                continue
            key = (sub_merge, rel.predicate, obj_merge)
            if key not in merged_keys:
                continue
            relationship_key = f"{sub_merge}:{rel.predicate}:{obj_merge}"
            for eid in rel.evidence_ids or []:
                ev = ev_by_id.get(eid)
                if ev is None:
                    continue
                text = ev.text or ""
                # EvidenceEntityRow has no character offsets; stub 0 and len(text_span).
                start_offset = 0
                end_offset = len(text)
                rows.append(
                    EvidenceRow(
                        relationship_key=relationship_key,
                        document_id=paper_id,
                        section=None,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        text_span=text,
                        confidence=ev.confidence,
                        supports=True,
                    )
                )
    return rows


def _build_mention_rows(
    bundles: list[tuple[str, PerPaperBundle]],
    id_map: dict[str, dict[str, str]],
    created_at: str,
) -> list[MentionRow]:
    """Build MentionRow list from bundles; entity_id is merge_key. Offsets stubbed (0, len(text_span))."""
    rows: list[MentionRow] = []
    for paper_id, bundle in bundles:
        paper_map = id_map.get(paper_id) or {}
        ev_by_id = {ev.id: ev for ev in bundle.evidence_entities}
        for rel in bundle.relationships:
            sub_merge = paper_map.get(rel.subject)
            obj_merge = paper_map.get(rel.object_id)
            for eid in rel.evidence_ids or []:
                ev = ev_by_id.get(eid)
                if ev is None:
                    continue
                text = ev.text or ""
                # EvidenceEntityRow has no character offsets; stub 0 and len(text_span).
                start_offset = 0
                end_offset = len(text)
                if sub_merge:
                    rows.append(
                        MentionRow(
                            entity_id=sub_merge,
                            document_id=paper_id,
                            section=None,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            text_span=text,
                            confidence=ev.confidence,
                            extraction_method=ev.extraction_method,
                            created_at=created_at,
                        )
                    )
                if obj_merge:
                    rows.append(
                        MentionRow(
                            entity_id=obj_merge,
                            document_id=paper_id,
                            section=None,
                            start_offset=start_offset,
                            end_offset=end_offset,
                            text_span=text,
                            confidence=ev.confidence,
                            extraction_method=ev.extraction_method,
                            created_at=created_at,
                        )
                    )
    return rows


def run_pass3(merged_dir: Path, bundles_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build kgbundle from merged Pass 2 output and Pass 1 bundles. Writes all bundle files."""
    entities_list, relationships_list, id_map, _ = load_merged_output(merged_dir)
    bundles = load_pass1_bundles(bundles_dir)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    usage = _entity_usage_from_bundles(bundles, id_map)
    evidence_stats = _relationship_evidence_stats(relationships_list, bundles, id_map)

    entity_rows = [_merged_entity_to_entity_row(ent, usage.get(ent["entity_id"], {}), created_at) for ent in entities_list]
    relationship_rows = [_merged_rel_to_relationship_row(rel, evidence_stats, created_at) for rel in relationships_list]
    evidence_rows = _build_evidence_rows(bundles, id_map, relationships_list)
    mention_rows = _build_mention_rows(bundles, id_map, created_at)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "docs").mkdir(exist_ok=True)

    with open(output_dir / "entities.jsonl", "w", encoding="utf-8") as f:
        for entity_row in entity_rows:
            f.write(entity_row.model_dump_json() + "\n")

    with open(output_dir / "relationships.jsonl", "w", encoding="utf-8") as f:
        for rel_row in relationship_rows:
            f.write(rel_row.model_dump_json() + "\n")

    with open(output_dir / "evidence.jsonl", "w", encoding="utf-8") as f:
        for ev_row in evidence_rows:
            f.write(ev_row.model_dump_json() + "\n")

    with open(output_dir / "mentions.jsonl", "w", encoding="utf-8") as f:
        for mention_row in mention_rows:
            f.write(mention_row.model_dump_json() + "\n")

    doc_asset = DocAssetRow(path="docs/README.md", content_type="text/markdown")
    with open(output_dir / "doc_assets.jsonl", "w", encoding="utf-8") as f:
        f.write(doc_asset.model_dump_json() + "\n")

    manifest = BundleManifestV1(
        bundle_version="v1",
        bundle_id=uuid.uuid4().hex,
        domain="medlit",
        label="medical-literature",
        created_at=created_at,
        entities=BundleFile(path="entities.jsonl", format="jsonl"),
        relationships=BundleFile(path="relationships.jsonl", format="jsonl"),
        doc_assets=BundleFile(path="doc_assets.jsonl", format="jsonl"),
        mentions=BundleFile(path="mentions.jsonl", format="jsonl"),
        evidence=BundleFile(path="evidence.jsonl", format="jsonl"),
        metadata={
            "entity_count": len(entity_rows),
            "relationship_count": len(relationship_rows),
            "description": "Knowledge graph bundle from two-pass medlit pipeline",
        },
    )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    synonym_src = merged_dir / "synonym_cache.json"
    if synonym_src.exists():
        shutil.copy(synonym_src, output_dir / "canonical_id_cache.json")

    readme = output_dir / "docs" / "README.md"
    readme.write_text(
        "Medlit bundle built from Pass 1 + Pass 2 output.\n",
        encoding="utf-8",
    )

    return {
        "entity_count": len(entity_rows),
        "relationship_count": len(relationship_rows),
        "evidence_count": len(evidence_rows),
        "mention_count": len(mention_rows),
        "manifest_path": str(output_dir / "manifest.json"),
    }
