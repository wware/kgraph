#!/usr/bin/env python3
"""
summarize_codebase.py

Produces a compact structural summary of a Python codebase suitable for
sharing with an LLM. Outputs only:
  - Package/module hierarchy
  - Imports (collapsed to one line per module)
  - Class definitions with base classes
  - Method and function signatures (no bodies)
  - Docstrings (optional, truncated)

Usage:
    python summarize_codebase.py [root_dir] [options]

Options:
    --no-docs        Omit all docstrings
    --no-imports     Omit import summaries
    --no-private     Omit names starting with _
    --max-doc N      Truncate docstrings to N chars (default: 120)
    --exclude GLOB   Exclude paths matching glob (repeatable)
    --out FILE       Write output to FILE instead of stdout

Examples:
    # Basic — signatures + truncated docstrings
    python summarize_codebase.py /path/to/project --out summary.md

    # Most compact — no docs, no imports, no private names
    python summarize_codebase.py /path/to/project --no-docs --no-imports --no-private --out summary.md

    # Exclude test dirs and migrations
    python summarize_codebase.py . --exclude "tests/*" --exclude "*/migrations/*" --out summary.md
"""

import ast
import argparse
import fnmatch
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default=".", help="Root directory (default: .)")
    p.add_argument("--no-docs", action="store_true", help="Omit docstrings")
    p.add_argument("--no-imports", action="store_true", help="Omit import lines")
    p.add_argument("--no-private", action="store_true", help="Omit _private names")
    p.add_argument("--max-doc", type=int, default=120, help="Truncate docstrings to N chars")
    p.add_argument("--exclude", action="append", default=[], metavar="GLOB", help="Exclude path globs")
    p.add_argument("--out", default=None, help="Output file (default: stdout)")
    p.add_argument("--files", nargs="*", metavar="FILE",
                   help="Explicit list of files (overrides root dir scan)")
    return p.parse_args()


def get_docstring(node: ast.AST, max_len: int) -> str | None:
    doc = ast.get_docstring(node)
    if not doc:
        return None
    doc = doc.strip().split("\n")[0]  # first line only
    if len(doc) > max_len:
        doc = doc[:max_len] + "…"
    return doc


def format_annotation(node) -> str:
    if node is None:
        return ""
    return f": {ast.unparse(node)}"


def format_default(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def format_args(args: ast.arguments) -> str:
    parts = []
    # positional-only
    n_posonlyargs = len(args.posonlyargs)
    all_args = args.posonlyargs + args.args
    n_defaults_offset = len(all_args) - len(args.defaults)

    for i, arg in enumerate(all_args):
        ann = format_annotation(arg.annotation)
        default_idx = i - n_defaults_offset
        if default_idx >= 0:
            default = f"={format_default(args.defaults[default_idx])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{ann}{default}")
        if i == n_posonlyargs - 1 and n_posonlyargs:
            parts.append("/")

    if args.vararg:
        ann = format_annotation(args.vararg.annotation)
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        ann = format_annotation(arg.annotation)
        default = ""
        if args.kw_defaults[i] is not None:
            default = f"={format_default(args.kw_defaults[i])}"
        parts.append(f"{arg.arg}{ann}{default}")

    if args.kwarg:
        ann = format_annotation(args.kwarg.annotation)
        parts.append(f"**{args.kwarg.arg}{ann}")

    return ", ".join(parts)


def format_func(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str, args: argparse.Namespace) -> list[str]:
    if args.no_private and node.name.startswith("_") and node.name != "__init__":
        return []
    lines = []
    decorators = [f"{indent}@{ast.unparse(d)}" for d in node.decorator_list]
    lines.extend(decorators)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = ""
    if node.returns:
        ret = f" -> {ast.unparse(node.returns)}"
    sig = f"{indent}{prefix} {node.name}({format_args(node.args)}){ret}"
    if not args.no_docs:
        doc = get_docstring(node, args.max_doc)
        if doc:
            sig += f'  # "{doc}"'
    lines.append(sig)
    return lines


def format_class(node: ast.ClassDef, indent: str, args: argparse.Namespace) -> list[str]:
    if args.no_private and node.name.startswith("_"):
        return []
    lines = []
    decorators = [f"{indent}@{ast.unparse(d)}" for d in node.decorator_list]
    lines.extend(decorators)
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    header = f"{indent}class {node.name}({bases}):" if bases else f"{indent}class {node.name}:"
    if not args.no_docs:
        doc = get_docstring(node, args.max_doc)
        if doc:
            header += f'  # "{doc}"'
    lines.append(header)

    child_indent = indent + "    "
    # class-level assignments (typed attributes)
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, ast.ClassDef):
            break  # don't recurse into nested here
    
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.extend(format_func(child, child_indent, args))
        elif isinstance(child, ast.ClassDef):
            lines.extend(format_class(child, child_indent, args))
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            if args.no_private and child.target.id.startswith("_"):
                continue
            ann = ast.unparse(child.annotation)
            val = f" = {ast.unparse(child.value)}" if child.value else ""
            lines.append(f"{child_indent}{child.target.id}: {ann}{val}")

    if len(lines) == len(decorators) + 1:
        lines.append(f"{child_indent}...")
    return lines


def summarize_module(path: Path, rel: Path, args: argparse.Namespace) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [f"# PARSE ERROR: {e}"]

    lines = []

    if not args.no_imports:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = ", ".join(a.name for a in node.names)
                imports.append(f"{'.' * node.level}{mod}.{names}" if mod else f"{'.' * node.level}{names}")
        if imports:
            lines.append(f"# imports: {', '.join(sorted(set(imports)))}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if args.no_private and node.name.startswith("_"):
                continue
            lines.extend(format_func(node, "", args))
        elif isinstance(node, ast.ClassDef):
            lines.extend(format_class(node, "", args))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if args.no_private and target.id.startswith("_"):
                        continue
                    try:
                        lines.append(f"{target.id} = {ast.unparse(node.value)}")
                    except Exception:
                        pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if args.no_private and node.target.id.startswith("_"):
                continue
            ann = ast.unparse(node.annotation)
            val = f" = {ast.unparse(node.value)}" if node.value else ""
            lines.append(f"{node.target.id}: {ann}{val}")

    return lines


def is_excluded(path: Path, root: Path, excludes: list[str]) -> bool:
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    for pattern in excludes:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout

    default_excludes = ["*.pyc", "__pycache__", ".git", ".venv", "venv", "node_modules", "*.egg-info"]
    all_excludes = default_excludes + args.exclude
    if args.files:
        py_files = [Path(f) for f in args.files]
    else:
        py_files = sorted(root.rglob("*.py"))

    total_chars = 0
    for path in py_files:
        if is_excluded(path, root, all_excludes):
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        header = f"\n{'='*60}\n# {rel}\n{'='*60}"
        body_lines = summarize_module(path, rel, args)
        body = "\n".join(body_lines)
        block = f"{header}\n{body}\n"
        total_chars += len(block)
        out.write(block)

    summary = f"\n# Total: {len(py_files)} files, ~{total_chars:,} chars\n"
    out.write(summary)
    if args.out:
        out.close()
        print(f"Written to {args.out} ({total_chars:,} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
