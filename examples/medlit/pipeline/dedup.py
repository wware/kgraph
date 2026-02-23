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

from examples.medlit.bundle_models import ExtractedEntityRow, PerPaperBundle
from examples.medlit.pipeline.synonym_cache import (
    add_same_as_to_cache,
    load_synonym_cache,
    lookup_entity,
    save_synonym_cache,
)


def _is_authoritative_id(s: str) -> bool:
    """Return True if s looks like an authoritative ontology ID, not a synthetic slug."""
    if not s or not s.strip():
        return False
    if s.startswith("canon-"):
        return False
    # MeSH (D + digits or MeSH:...)
    if s.startswith("MeSH:"):
        return True
    if len(s) > 1 and s[0] == "D" and s[1:].isdigit():
        return True
    # UMLS (C + digits)
    if s.startswith("C") and len(s) > 1 and s[1:].isdigit():
        return True
    # HGNC
    if s.startswith("HGNC:"):
        return True
    if s.isdigit():
        return True  # numeric HGNC
    # RxNorm
    if s.startswith("RxNorm:"):
        return True
    # UniProt (P/Q + alphanumeric)
    if (s.startswith("P") or s.startswith("Q")) and len(s) >= 6 and s[1:].isalnum():
        return True
    if s.startswith("UniProt:"):
        return True
    # DBPedia
    if s.startswith("DBPedia:"):
        return True
    return False


def _authoritative_id_from_entity(e: ExtractedEntityRow) -> Optional[str]:
    """Return the best authoritative ID from bundle entity row, or None."""
    for val in (e.canonical_id, e.umls_id, e.hgnc_id, e.rxnorm_id, e.uniprot_id):
        if val and val.strip() and _is_authoritative_id(val):
            return val.strip()
    return None


def _entity_class_to_lookup_type(entity_class: str) -> Optional[str]:
    """Map bundle entity_class to CanonicalIdLookup entity_type (lowercase)."""
    m = {
        "Disease": "disease",
        "Gene": "gene",
        "Drug": "drug",
        "Protein": "protein",
        "Biomarker": "disease",
    }
    return m.get(entity_class)


def _canonical_id_slug() -> str:
    """Generate a short synthetic merge key for entities without authoritative ID."""
    return "canon-" + uuid.uuid4().hex[:12]


def run_pass2(  # pylint: disable=too-many-statements
    bundle_dir: Path,
    output_dir: Path,
    synonym_cache_path: Optional[Path] = None,
    canonical_id_cache_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Pass 2: dedup and promotion. Reads bundles from bundle_dir, writes to output_dir.

    Original bundle files in bundle_dir are never modified.
    Returns summary dict (entities_count, relationships_count, etc.).
    """
    synonym_cache_path = synonym_cache_path or output_dir / "synonym_cache.json"
    cache = load_synonym_cache(synonym_cache_path)

    # Optional authority lookup for resolving canonical IDs (save cache at end)
    lookup = None
    if canonical_id_cache_path is not None:
        from examples.medlit.pipeline.authority_lookup import CanonicalIdLookup

        lookup = CanonicalIdLookup(cache_file=canonical_id_cache_path)

    try:
        return _run_pass2_impl(
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            synonym_cache_path=synonym_cache_path,
            cache=cache,
            lookup=lookup,
        )
    finally:
        if lookup is not None:
            lookup._save_cache(force=True)  # Save lookup cache so results persist across runs.  # pylint: disable=protected-access


def _run_pass2_impl(  # pylint: disable=too-many-statements
    bundle_dir: Path,
    output_dir: Path,
    synonym_cache_path: Path,
    cache: dict,
    lookup: Any,
) -> dict[str, Any]:
    """Inner Pass 2 implementation (lookup created and saved by caller)."""
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

    # (name_lower, entity_class) -> merge key (authoritative or synthetic)
    name_type_to_canonical: dict[tuple[str, str], str] = {}
    # (paper_id, bundle_local_id) -> merge key
    local_to_canonical: dict[tuple[str, str], str] = {}

    def get_or_assign_canonical(
        paper_id: str,
        local_id: str,
        name: str,
        entity_class: str,
        entity_row: Optional[ExtractedEntityRow] = None,
    ) -> str:
        key_nt = (name.lower(), entity_class)
        key_local = (paper_id, local_id)
        if key_local in local_to_canonical:
            return local_to_canonical[key_local]
        # Prefer authoritative ID from bundle entity
        if entity_row is not None:
            auth = _authoritative_id_from_entity(entity_row)
            if auth:
                name_type_to_canonical[key_nt] = auth
                local_to_canonical[key_local] = auth
                return auth
        existing, _ = lookup_entity(cache, name, entity_class)
        if existing:
            name_type_to_canonical[key_nt] = existing
            local_to_canonical[key_local] = existing
            return existing
        if key_nt in name_type_to_canonical:
            cid = name_type_to_canonical[key_nt]
            local_to_canonical[key_local] = cid
            return cid
        # Optional authority lookup
        if lookup is not None:
            lookup_type = _entity_class_to_lookup_type(entity_class)
            if lookup_type:
                resolved = lookup.lookup_canonical_id_sync(name, lookup_type)
                if resolved:
                    name_type_to_canonical[key_nt] = resolved
                    local_to_canonical[key_local] = resolved
                    return resolved
        cid = _canonical_id_slug()
        name_type_to_canonical[key_nt] = cid
        local_to_canonical[key_local] = cid
        return cid

    # 1) Build initial name/type index from all entities (use cache when available)
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            name = e.name
            entity_class = e.entity_class
            cid = get_or_assign_canonical(paper_id, e.id, name, entity_class, entity_row=e)
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
                # Prefer authoritative ID as winner so merged entity has canonical_id when possible
                winner = sub_id
                if _is_authoritative_id(obj_id) and not _is_authoritative_id(sub_id):
                    winner = obj_id
                for (pid, bid), cid in list(local_to_canonical.items()):
                    if cid in (obj_id, sub_id):
                        local_to_canonical[(pid, bid)] = winner
                for key_nt, cid in list(name_type_to_canonical.items()):
                    if cid in (obj_id, sub_id):
                        name_type_to_canonical[key_nt] = winner
                name_a, class_a = _entity_name_class(bundle, rel.subject)
                name_b, class_b = _entity_name_class(bundle, rel.object_id)
                add_same_as_to_cache(
                    cache,
                    {"name": name_a, "class": class_a, "canonical_id": winner},
                    {"name": name_b, "class": class_b, "canonical_id": winner},
                    rel.confidence,
                    "automated",
                    "merged",
                    list(rel.source_papers) if rel.source_papers else [paper_id],
                )

    # 3) Assign merge keys to any entity not yet assigned
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            if (paper_id, e.id) not in local_to_canonical:
                cid = get_or_assign_canonical(paper_id, e.id, e.name, e.entity_class, entity_row=e)
                local_to_canonical[(paper_id, e.id)] = cid

    # 4) Ontology stub: leave all source="extracted", no ontology id changes

    # 5) Build merged entities: one row per merge key (entity_id); canonical_id only when authoritative
    canonical_entities: dict[str, dict[str, Any]] = {}
    for paper_id, bundle in bundles:
        for e in bundle.entities:
            merge_key = local_to_canonical.get((paper_id, e.id))
            if merge_key is None:
                continue
            if merge_key not in canonical_entities:
                canonical_entities[merge_key] = {
                    "entity_id": merge_key,
                    "canonical_id": merge_key if _is_authoritative_id(merge_key) else None,
                    "class": e.entity_class,
                    "name": e.name,
                    "synonyms": list(e.synonyms),
                    "source": e.source,
                    "source_papers": [],
                }
            if paper_id not in canonical_entities[merge_key]["source_papers"]:
                canonical_entities[merge_key]["source_papers"].append(paper_id)

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
