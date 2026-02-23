#!/usr/bin/env python3
"""Pass 2: Deduplication and promotion over per-paper bundles.

Reads all paper_*.json bundles from --bundle-dir (output of Pass 1), builds
name/type index and synonym cache, resolves SAME_AS, assigns canonical IDs,
and writes merged entities and relationships to --output-dir. Original
bundle files are never modified.

Usage:
  python -m examples.medlit.scripts.pass2_dedup --bundle-dir pass1_bundles/ --output-dir merged/
"""

import argparse
import sys
from pathlib import Path

# Repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.medlit.pipeline.dedup import run_pass2  # noqa: E402  # pylint: disable=wrong-import-position


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 2: Deduplicate and promote entities/relationships from per-paper bundles.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Directory containing paper_*.json bundle files (Pass 1 output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pass2_merged"),
        help="Output directory for merged entities.json, relationships.json, synonym_cache.json",
    )
    parser.add_argument(
        "--synonym-cache",
        type=Path,
        default=None,
        help="Path to synonym cache file (default: <output-dir>/synonym_cache.json)",
    )
    args = parser.parse_args()

    if not args.bundle_dir.exists():
        print(f"Error: bundle dir not found: {args.bundle_dir}", file=sys.stderr)
        sys.exit(1)

    result = run_pass2(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        synonym_cache_path=args.synonym_cache,
    )

    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)
    print(
        f"Pass 2 done: {result['entities_count']} entities, {result['relationships_count']} relationships",
        file=sys.stderr,
    )
    print(f"  entities: {result['entities_path']}", file=sys.stderr)
    print(f"  relationships: {result['relationships_path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
