#!/usr/bin/env python3
"""Pass 3: Build kgbundle from medlit_merged and pass1_bundles.

Reads merged_dir (entities.json, relationships.json, id_map.json, synonym_cache.json)
and bundles_dir (paper_*.json), writes output_dir in kgbundle format for kgserver.
"""

import sys
from pathlib import Path

# Add repo root for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from examples.medlit.pipeline.bundle_builder import run_pass3  # noqa: E402  # pylint: disable=wrong-import-position


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pass 3: Build kgbundle from medlit_merged and pass1_bundles.",
    )
    parser.add_argument("--merged-dir", type=Path, required=True, help="Pass 2 merged output directory")
    parser.add_argument("--bundles-dir", type=Path, required=True, help="Pass 1 paper_*.json bundles directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("medlit_bundle"),
        help="Output kgbundle directory (default: medlit_bundle)",
    )
    args = parser.parse_args()

    merged_dir = args.merged_dir.resolve()
    bundles_dir = args.bundles_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not merged_dir.is_dir():
        print(f"Error: merged-dir is not a directory: {merged_dir}", file=sys.stderr)
        return 1
    if not (merged_dir / "id_map.json").exists():
        print(
            f"Error: id_map.json not found in {merged_dir}. Run Pass 2 so that merged_dir contains id_map.json.",
            file=sys.stderr,
        )
        return 1
    if not bundles_dir.is_dir():
        print(f"Error: bundles-dir is not a directory: {bundles_dir}", file=sys.stderr)
        return 1

    try:
        summary = run_pass3(merged_dir, bundles_dir, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Pass 3 complete: {summary['entity_count']} entities, {summary['relationship_count']} relationships, " f"{summary['evidence_count']} evidence, {summary['mention_count']} mentions",
        file=sys.stderr,
    )
    print(f"Manifest: {summary['manifest_path']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
