#!/usr/bin/env python3
"""Pass 1: Extract entities and relationships from papers via LLM → per-paper bundle JSON.

Reads papers from --input-dir (JATS XML or JSON), calls the configured LLM once per paper,
and writes one JSON file per paper to --output-dir. These bundles are immutable;
Pass 2 (dedup) reads them and writes overlays or a merged graph elsewhere.

Requires an LLM backend: --llm-backend anthropic | openai | ollama.
Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or run Ollama locally. See LLM_SETUP.md.

Usage:
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend anthropic
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --llm-backend ollama --limit 1
  python -m examples.medlit.scripts.pass1_extract --input-dir pmc_xmls/ --output-dir bundles/ --papers "PMC127*.xml,PMC128*.json"
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from repo root so ANTHROPIC_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from examples.medlit.bundle_models import (  # noqa: E402  # pylint: disable=wrong-import-position
    EvidenceEntityRow,
    ExtractedEntityRow,
    PaperInfo,
    PerPaperBundle,
    RelationshipRow,
)
from examples.medlit_schema.base import (  # noqa: E402  # pylint: disable=wrong-import-position
    ExecutionInfo,
    ExtractionPipelineInfo,
    ExtractionProvenance,
    ModelInfo,
    PromptInfo,
)


def _git_info() -> dict:
    """Return git_commit, git_commit_short, git_branch, git_dirty, repo_url."""
    out = {
        "git_commit": "",
        "git_commit_short": "",
        "git_branch": "",
        "git_dirty": False,
        "repo_url": "https://github.com/org/medlit",
    }
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        out["git_commit_short"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        out["git_branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5.0,
            cwd=REPO_ROOT,
            check=False,
        )
        out["git_dirty"] = bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return out


def build_provenance(
    llm_name: str,
    llm_version: str,
    prompt_version: str = "v1",
    prompt_template: str = "medlit_extraction_v1",
    prompt_checksum: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> ExtractionProvenance:
    """Build extraction_provenance for Pass 1 output."""
    git = _git_info()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return ExtractionProvenance(
        models={"llm": ModelInfo(name=llm_name, version=llm_version)},
        extraction_pipeline=ExtractionPipelineInfo(
            name="medlit_pass1_extract",
            version="0.1.0",
            git_commit=git["git_commit"] or "unknown",
            git_commit_short=git["git_commit_short"] or "unknown",
            git_branch=git["git_branch"] or "unknown",
            git_dirty=git["git_dirty"],
            repo_url=git["repo_url"],
        ),
        prompt=PromptInfo(
            version=prompt_version,
            template=prompt_template,
            checksum=prompt_checksum,
        ),
        execution=ExecutionInfo(
            timestamp=now,
            hostname=os.environ.get("HOSTNAME", "unknown"),
            python_version=sys.version.split()[0],
            duration_seconds=duration_seconds,
        ),
        entity_resolution=None,
    )


def _default_system_prompt() -> str:
    """Minimal system prompt asking for per-paper bundle JSON."""
    return """You are a biomedical knowledge extraction expert. Extract entities and relationships from the given paper and return a single JSON object with this structure (use the exact keys):

- "paper": { "pmcid", "title", "authors", "journal", "year", "study_type", "eco_type" (optional) }
- "entities": array of { "id", "class", "name", "synonyms" (array), "source": "extracted", "canonical_id": null, "umls_id"/"hgnc_id"/"rxnorm_id" as needed }
- "evidence_entities": array of { "id" (format paper_id:section:paragraph_idx:method), "class": "Evidence", "paper_id", "text_span_id", "text", "confidence", "extraction_method", "study_type", "eco_type", "source": "extracted" }
- "relationships": array of { "subject", "predicate", "object", "evidence_ids", "source_papers", "confidence", "properties", "section", "asserted_by": "llm" }. For SAME_AS use "resolution": null, "note".
- "notes": array of strings (optional clarifications)

Return ONLY valid JSON, no markdown or commentary. Use "class" for entity type (Disease, Gene, Drug, Evidence, etc.). Predicates: TREATS, INCREASES_RISK, INDICATES, ASSOCIATED_WITH, SAME_AS, SUBTYPE_OF, etc. Evidence id format: {paper_id}:{section}:{paragraph_idx}:llm."""


async def _paper_content_from_parser(raw_content: bytes, content_type: str, source_uri: str) -> tuple[str, Optional[PaperInfo]]:
    """Extract text and minimal PaperInfo from raw content using existing parser."""
    from examples.medlit.pipeline.parser import JournalArticleParser

    parser = JournalArticleParser()
    doc = await parser.parse(raw_content, content_type, source_uri)
    # Build content for LLM: title + abstract + body
    parts = [doc.title or ""]
    if getattr(doc, "content", None):
        parts.append(str(doc.content))
    text = "\n\n".join(p for p in parts if p)
    paper_id = doc.document_id or Path(source_uri).stem
    info = PaperInfo(
        pmcid=paper_id if paper_id.startswith("PMC") else None,
        title=doc.title or "",
        authors=list(getattr(doc, "authors", [])) or [],
    )
    return text or "(no content)", info


def _paper_content_fallback(raw_content: bytes, source_uri: str) -> tuple[str, PaperInfo]:
    """Fallback: use raw text and filename for paper id."""
    text = raw_content.decode("utf-8", errors="replace")
    stem = Path(source_uri).stem
    return text or "(no content)", PaperInfo(pmcid=stem if stem.startswith("PMC") else None, title=stem, authors=[])


async def run_pass1(
    input_dir: Path,
    output_dir: Path,
    llm_backend: str,
    limit: Optional[int] = None,
    papers: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Run Pass 1: for each paper in input_dir, call LLM and write bundle JSON to output_dir."""
    from examples.medlit.pipeline.pass1_llm import get_pass1_llm

    llm = get_pass1_llm(llm_backend)
    prompt = system_prompt or _default_system_prompt()
    prompt_checksum = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # Discover input files: by glob patterns or all *.xml/*.json
    files: list[Path] = []
    if papers is not None:
        for pattern in papers:
            pattern = pattern.strip()
            if not pattern:
                continue
            files.extend(input_dir.glob(pattern))
        files = sorted(set(files))
    else:
        for ext in ("*.xml", "*.json"):
            files.extend(input_dir.glob(ext))
        files = sorted(set(files))
    if limit is not None:
        files = files[:limit]

    if not files:
        print(f"No XML/JSON files in {input_dir}", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pass 1: {len(files)} paper(s), backend={llm_backend}, output={output_dir}", file=sys.stderr)

    for _, path in enumerate(files):
        content_type = "application/xml" if path.suffix.lower() in (".xml",) else "application/json"
        with open(path, "rb") as f:
            raw = f.read()
        try:
            content, paper_info = await _paper_content_from_parser(raw, content_type, str(path))
        except Exception:
            content, paper_info = _paper_content_fallback(raw, str(path))
        if paper_info is None:
            content, paper_info = _paper_content_fallback(raw, str(path))

        paper_id = paper_info.pmcid or path.stem
        out_path = output_dir / f"paper_{paper_id}.json"
        if out_path.exists():
            print(f"  Skip (exists): {out_path.name}", file=sys.stderr)
            continue

        start = time.perf_counter()
        try:
            raw_bundle = await llm.generate_json(
                system_prompt=prompt,
                user_message=content[:500000],  # cap size
                temperature=0.1,
                max_tokens=16384,
            )
        except Exception as e:
            print(f"  ERROR {path.name}: {e}", file=sys.stderr)
            continue
        duration = time.perf_counter() - start

        # Parse rows (JSON may use "class" / "object"; models use alias)
        entities = [ExtractedEntityRow.model_validate(e) for e in raw_bundle.get("entities", [])]
        evidence_entities = [EvidenceEntityRow.model_validate(ev) for ev in raw_bundle.get("evidence_entities", [])]
        relationships = [RelationshipRow.model_validate(r) for r in raw_bundle.get("relationships", [])]

        # Use paper from LLM if present and valid, else parser
        raw_paper = raw_bundle.get("paper") or {}
        raw_authors = raw_paper.get("authors", paper_info.authors)
        if isinstance(raw_authors, list):
            authors = raw_authors
        elif isinstance(raw_authors, str):
            authors = [raw_authors] if raw_authors.strip() else []
        else:
            authors = list(paper_info.authors) if paper_info.authors else []
        paper = PaperInfo(
            doi=raw_paper.get("doi") or paper_info.doi,
            pmcid=raw_paper.get("pmcid") or paper_info.pmcid or paper_id,
            title=raw_paper.get("title") or paper_info.title,
            authors=authors,
            journal=raw_paper.get("journal"),
            year=raw_paper.get("year"),
            study_type=raw_paper.get("study_type"),
            eco_type=raw_paper.get("eco_type"),
        )
        llm_name = getattr(llm, "model", llm_backend)
        provenance = build_provenance(
            llm_name=llm_name,
            llm_version="",
            prompt_checksum=f"sha256:{prompt_checksum}",
            duration_seconds=round(duration, 2),
        )
        bundle = PerPaperBundle(
            paper=paper,
            extraction_provenance=provenance,
            entities=entities,
            evidence_entities=evidence_entities,
            relationships=relationships,
            notes=raw_bundle.get("notes", []),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            import json

            json.dump(bundle.to_bundle_dict(), f, indent=2)
        print(f"  Wrote {out_path.name} ({len(entities)} entities, {len(relationships)} rels)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 1: Extract per-paper bundle JSON via LLM (immutable output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing paper XML/JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("pass1_bundles"), help="Output directory for bundle JSONs")
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=("anthropic", "openai", "ollama"),
        default=os.environ.get("LLM_BACKEND", "ollama"),
        help="LLM backend (default: LLM_BACKEND or ollama)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers to process")
    parser.add_argument(
        "--papers",
        type=str,
        default=None,
        metavar="GLOB[,GLOB,...]",
        help="Comma-separated glob patterns for input files (e.g. PMC127*.xml,PMC128*.json). Resolved under --input-dir. If omitted, all *.xml and *.json are used.",
    )
    args = parser.parse_args()
    import asyncio

    papers_list: Optional[list[str]] = None
    if args.papers is not None:
        papers_list = [p.strip() for p in args.papers.split(",") if p.strip()]

    asyncio.run(run_pass1(args.input_dir, args.output_dir, args.llm_backend, args.limit, papers_list))


if __name__ == "__main__":
    main()
