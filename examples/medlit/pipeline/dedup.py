"""Pass 2: Deduplication and promotion over per-paper bundles.

Reads all PerPaperBundle JSONs from a directory, builds name/type index (with
synonym cache), resolves high-confidence SAME_AS, assigns canonical IDs,
updates relationship refs, accumulates triples, and saves the synonym cache.
Original bundle files are never modified; output is written to a separate
directory (overlay or merged graph).
"""

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from examples.medlit.bundle_models import PerPaperBundle
from examples.medlit.pipeline.synonym_cache import (
    add_same_as_to_cache,
    load_synonym_cache,
    lookup_entity,
    save_synonym_cache,
)


def _canonical_id_slug() -> str:
    """Generate a short canonical ID (e.g. for entities)."""
    return "canon-" + uuid.uuid4().hex[:12]


def run_pass2(  # pylint: disable=too-many-statements
    bundle_dir: Path,
    output_dir: Path,
    synonym_cache_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Pass 2: dedup and promotion. Reads bundles from bundle_dir, writes to output_dir.

    Original bundle files in bundle_dir are never modified.
    Returns summary dict (entities_count, relationships_count, etc.).
    """
    synonym_cache_path = synonym_cache_path or output_dir / "synonym_cache.json"
    cache = load_synonym_cache(synonym_cache_path)

    # Load all bundles
    bundle_files = sorted(bundle_dir.glob("paper_*.json"))
    if not bundle_files:
        return {"error": "no bundle files", "entities_count": 0, "relationships_count": 0}

    bundles: list[tuple[str, PerPaperBundle]] = []
    for path in bundle_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        bundle = PerPaperBundle.from_bundle_dict(data)
        paper_id = bundle.paper.pmcid or bundle.paper.doi or path.stem.replace("paper_", "")
        bundles.append((paper_id, bundle))

    # (name_lower, entity_class) -> canonical_id
    name_type_to_canonical: dict[tuple[str, str], str] = {}
    # (paper_id, bundle_local_id) -> canonical_id
    local_to_canonical: dict[tuple[str, str], str] = {}

    def get_or_assign_canonical(paper_id: str, local_id: str, name: str, entity_class: str) -> str:
        key_nt = (name.lower(), entity_class)
        key_local = (paper_id, local_id)
        if key_local in local_to_canonical:
            return local_to_canonical[key_local]
        existing, _ = lookup_entity(cache, name, entity_class)
        if existing:
            name_type_to_canonical[key_nt] = existing
            local_to_canonical[key_local] = existing
            return existing
        if key_nt in name_type_to_canonical:
            cid = name_type_to_canonical[key_nt]
            local_to_canonical[key_local] = cid
            return cid
        cid = _canonical_id_slug()
        name_type_to_canonical[key_nt] = cid
        local_to_canonical[key_local] = cid
        return cid

    # 1) Build initial name/type index from all entities (use cache when available)
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            name = e.name
            entity_class = e.entity_class
            cid = get_or_assign_canonical(paper_id, e.id, name, entity_class)
            # store back so we can use it
            local_to_canonical[(paper_id, e.id)] = cid

    # 2) Auto-resolve high-confidence SAME_AS: merge entities
    def _entity_name_class(bundle: PerPaperBundle, local_id: str) -> tuple[str, str]:
        for e in bundle.entities:
            if e.id == local_id:
                return (e.name, e.entity_class)
        return (local_id, "?")

    for paper_id, bundle in bundles:
        for rel in bundle.relationships:
            if rel.predicate != "SAME_AS" or rel.confidence < 0.85:
                continue
            sub_id = local_to_canonical.get((paper_id, rel.subject))
            obj_id = local_to_canonical.get((paper_id, rel.object_id))
            if sub_id and obj_id and sub_id != obj_id:
                for (pid, bid), cid in list(local_to_canonical.items()):
                    if cid == obj_id:
                        local_to_canonical[(pid, bid)] = sub_id
                for key_nt, cid in list(name_type_to_canonical.items()):
                    if cid == obj_id:
                        name_type_to_canonical[key_nt] = sub_id
                name_a, class_a = _entity_name_class(bundle, rel.subject)
                name_b, class_b = _entity_name_class(bundle, rel.object_id)
                add_same_as_to_cache(
                    cache,
                    {"name": name_a, "class": class_a, "canonical_id": sub_id},
                    {"name": name_b, "class": class_b, "canonical_id": sub_id},
                    rel.confidence,
                    "automated",
                    "merged",
                    list(rel.source_papers) if rel.source_papers else [paper_id],
                )

    # 3) Assign canonical IDs to any entity not yet assigned
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            if (paper_id, e.id) not in local_to_canonical:
                cid = get_or_assign_canonical(paper_id, e.id, e.name, e.entity_class)
                local_to_canonical[(paper_id, e.id)] = cid

    # 4) Ontology stub: leave all source="extracted", no ontology id changes

    # 5) Build merged entities: one row per canonical_id (with representative name, etc.)
    canonical_entities: dict[str, dict[str, Any]] = {}
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            cid_opt = local_to_canonical.get((paper_id, e.id))
            if cid_opt is None:
                continue
            canon_id = cid_opt
            if canon_id not in canonical_entities:
                canonical_entities[canon_id] = {
                    "canonical_id": canon_id,
                    "class": e.entity_class,
                    "name": e.name,
                    "synonyms": list(e.synonyms),
                    "source": e.source,
                    "source_papers": [],
                }
            if paper_id not in canonical_entities[canon_id]["source_papers"]:
                canonical_entities[canon_id]["source_papers"].append(paper_id)

    # 6) Accumulate relationships: (sub_canonical, predicate, obj_canonical) -> union source_papers, evidence_ids, max confidence
    triple_to_rel: dict[tuple[str, str, str], dict[str, Any]] = {}
    for paper_id, bundle in bundles:
        for rel in bundle.relationships:
            if rel.predicate == "SAME_AS" and rel.confidence >= 0.85:
                continue  # already merged
            sub_c = local_to_canonical.get((paper_id, rel.subject))
            obj_c = local_to_canonical.get((paper_id, rel.object_id))
            if not sub_c or not obj_c:
                continue
            key = (sub_c, rel.predicate, obj_c)
            if key not in triple_to_rel:
                triple_to_rel[key] = {
                    "subject": sub_c,
                    "predicate": rel.predicate,
                    "object": obj_c,
                    "evidence_ids": [],
                    "source_papers": [],
                    "confidence": rel.confidence,
                }
            r = triple_to_rel[key]
            for eid in rel.evidence_ids or []:
                if eid not in r["evidence_ids"]:
                    r["evidence_ids"].append(eid)
            for sp in rel.source_papers or [paper_id]:
                if sp not in r["source_papers"]:
                    r["source_papers"].append(sp)
            if rel.confidence > r["confidence"]:
                r["confidence"] = rel.confidence

    # 7) Add all SAME_AS to synonym cache for persistence
    for paper_id, bundle in bundles:
        for rel in bundle.relationships:
            if rel.predicate != "SAME_AS":
                continue
            sub_c = local_to_canonical.get((paper_id, rel.subject))
            obj_c = local_to_canonical.get((paper_id, rel.object_id))
            if sub_c and obj_c:
                name_a, class_a = _entity_name_class(bundle, rel.subject)
                name_b, class_b = _entity_name_class(bundle, rel.object_id)
                add_same_as_to_cache(
                    cache,
                    {"name": name_a, "class": class_a, "canonical_id": sub_c},
                    {"name": name_b, "class": class_b, "canonical_id": obj_c},
                    rel.confidence,
                    "llm" if rel.asserted_by == "llm" else "automated",
                    rel.resolution,
                    list(rel.source_papers) if rel.source_papers else [paper_id],
                )

    # 8) Save synonym cache
    output_dir.mkdir(parents=True, exist_ok=True)
    save_synonym_cache(synonym_cache_path, cache)

    # 9) Write merged output (do not touch original bundles)
    entities_path = output_dir / "entities.json"
    relationships_path = output_dir / "relationships.json"
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(list(canonical_entities.values()), f, indent=2)
    with open(relationships_path, "w", encoding="utf-8") as f:
        json.dump(list(triple_to_rel.values()), f, indent=2)

    return {
        "entities_count": len(canonical_entities),
        "relationships_count": len(triple_to_rel),
        "bundles_processed": len(bundles),
        "entities_path": str(entities_path),
        "relationships_path": str(relationships_path),
    }
