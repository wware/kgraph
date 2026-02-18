#!/usr/bin/env -S uv run python
"""
Extract documentation from Python source files into Markdown.

Walks the AST to find docstrings, standalone strings, and creates
formatted signatures for classes, methods, and functions.

Includes a portion of each *.md, *.yml, Dockerfile, and shell script
to add more context.

$ uv run python extract_summary.py   # writes summary.md
"""

import ast
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

skip_markdown = True
sys.stdout = open("summary.md", "w")


class DocExtractor(ast.NodeVisitor):
    def __init__(self):
        self.sections: List[Tuple[str, str, str, str]] = []  # (type, signature, doc, fields)
        self.current_class = None

    def visit_Module(self, node: ast.Module) -> None:
        """Extract module-level docstring."""
        docstring = ast.get_docstring(node)
        if docstring:
            self.sections.append(("module", "", docstring, ""))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class signature and docstring."""
        bases = ", ".join(self._format_expr(base) for base in node.bases)
        sig = f"class {node.name}({bases})" if bases else f"class {node.name}"

        docstring = ast.get_docstring(node)

        # Check if this is a Pydantic model (inherits from BaseModel)
        is_pydantic = self._is_pydantic_model(node)
        fields_md = ""
        if is_pydantic:
            fields_md = self._extract_pydantic_fields(node)

        if docstring or fields_md:
            self.sections.append(("class", sig, docstring or "", fields_md))

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function/method signature and docstring."""
        args = self._format_arguments(node.args)
        returns = f" -> {self._format_expr(node.returns)}" if node.returns else ""

        if self.current_class:
            sig = f"def {self.current_class}.{node.name}({args}){returns}"
            doc_type = "method"
        else:
            sig = f"def {node.name}({args}){returns}"
            doc_type = "function"

        docstring = ast.get_docstring(node)
        if docstring:
            self.sections.append((doc_type, sig, docstring, ""))

        # Also look for standalone string literals in the function body
        for stmt in node.body[1:]:  # Skip docstring
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    self.sections.append(("note", "", stmt.value.value, ""))

        self.generic_visit(node)

    def _format_arguments(self, args: ast.arguments) -> str:
        """Format function arguments."""
        parts = []

        # Regular args
        for i, arg in enumerate(args.args):
            annotation = f": {self._format_expr(arg.annotation)}" if arg.annotation else ""
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                default = self._format_expr(args.defaults[i - default_offset])
                parts.append(f"{arg.arg}{annotation} = {default}")
            else:
                parts.append(f"{arg.arg}{annotation}")

        # *args
        if args.vararg:
            annotation = f": {self._format_expr(args.vararg.annotation)}" if args.vararg.annotation else ""
            parts.append(f"*{args.vararg.arg}{annotation}")

        # **kwargs
        if args.kwarg:
            annotation = f": {self._format_expr(args.kwarg.annotation)}" if args.kwarg.annotation else ""
            parts.append(f"**{args.kwarg.arg}{annotation}")

        return ", ".join(parts)

    def _format_expr(self, node) -> str:
        """Format an expression node as a string."""
        if node is None:
            return ""
        return ast.unparse(node)

    def _is_pydantic_model(self, node: ast.ClassDef) -> bool:
        """Check if a class inherits from BaseModel (directly or indirectly)."""
        for base in node.bases:
            base_str = self._format_expr(base)
            # Check for BaseModel in bases (could be pydantic.BaseModel, BaseModel, etc.)
            if "BaseModel" in base_str:
                return True
            # Also check for common Pydantic model base class names
            # This handles cases like BaseMedicalEntity which inherits from BaseModel
            if "Entity" in base_str and "Base" in base_str:
                return True

        # Also check if class has annotated assignments (field definitions)
        # which is a strong indicator of a Pydantic model
        has_annotated_fields = any(isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and not stmt.target.id.startswith("_") for stmt in node.body)

        # If it has annotated fields and inherits from something with "Model" or "Entity" in name
        if has_annotated_fields:
            base_names = [self._format_expr(base) for base in node.bases]
            if any("Model" in name or "Entity" in name for name in base_names):
                return True

        return False

    def _extract_pydantic_fields(self, node: ast.ClassDef) -> str:
        """Extract field type hints from a Pydantic model class."""
        fields = []

        for stmt in node.body:
            # Look for annotated assignments (field definitions)
            if isinstance(stmt, ast.AnnAssign):
                field_name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                if field_name:
                    # Skip private attributes and special methods
                    if field_name.startswith("_") and field_name != "__init__":
                        continue

                    # Format type annotation
                    type_hint = self._format_expr(stmt.annotation) if stmt.annotation else "Any"

                    # Format default value if present
                    default = ""
                    if stmt.value:
                        default_str = self._format_expr(stmt.value)
                        # Simplify Field() calls to show just the default
                        if default_str.startswith("Field("):
                            # Try to extract default value from Field()
                            default = self._extract_field_default(stmt.value)
                        else:
                            default = default_str

                    # Format the field line
                    if default:
                        fields.append(f"{field_name}: {type_hint} = {default}")
                    else:
                        fields.append(f"{field_name}: {type_hint}")

        if fields:
            return "**Fields:**\n\n```python\n" + "\n".join(fields) + "\n```\n"
        return ""

    def _extract_field_default(self, field_node: ast.Call) -> str:
        """Extract default value from Field() call."""
        # Look for default or default_factory in Field() arguments
        for keyword in field_node.keywords:
            if keyword.arg == "default":
                return self._format_expr(keyword.value)
            elif keyword.arg == "default_factory":
                factory = self._format_expr(keyword.value)
                # Simplify common factories
                if factory == "list":
                    return "Field(default_factory=list)"
                elif factory == "dict":
                    return "Field(default_factory=dict)"
                elif factory == "datetime.now":
                    return "Field(default_factory=datetime.now)"
                else:
                    return f"Field(default_factory={factory})"

        # If no default found, check if first positional arg is a default
        if field_node.args:
            return self._format_expr(field_node.args[0])

        return "Field(...)"


def extract_docs(source_path: Path) -> str:
    """Extract documentation from a Python file and return as Markdown."""
    source_code = source_path.read_text()
    tree = ast.parse(source_code)

    extractor = DocExtractor()
    extractor.visit(tree)

    # Section header is emitted by main(); we return only the body
    lines = []

    for section in extractor.sections:
        doc_type, signature, content, fields = section
        if doc_type == "module":
            lines.append(f"{content}\n")
        elif doc_type == "class":
            lines.append(f"## `{signature}`\n")
            if content:
                lines.append(f"{content}\n")
            if fields:
                lines.append(f"{fields}\n")
        elif doc_type in ("function", "method"):
            lines.append(f"### `{signature}`\n")
            lines.append(f"{content}\n")
        elif doc_type == "note":
            lines.append(f"> {content}\n")

    return "\n".join(lines)


# Files that contain prompt/response conversation rather than formal docs
CONVERSATION_FILES = {"TODO1.md", "TODO2.md", "VIBES.md"}


def _heading_anchor(path: str) -> str:
    """Generate GFM-style anchor from section heading (path). Matches GitHub: lowercase,
    spaces to hyphens, strip other punctuation, collapse hyphens. E.g. 'CLAUDE.md' -> 'claudemd'."""
    s = path.lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "section"


def _closing_fence_for_truncated_md(body: str) -> str:
    """If body ends with an unclosed fenced block, return closing fence (e.g. '\\n```\\n' or '\\n````\\n').
    CommonMark/GFM require the closing fence to have at least as many backticks as the opening."""
    fence_lines = [ln for ln in body.splitlines() if ln.strip().startswith("```")]
    if len(fence_lines) % 2 == 0:
        return ""
    # Last fence is an opener; count its backticks so we close with at least that many (GFM)
    stripped = fence_lines[-1].strip()
    n = 0
    for c in stripped:
        if c == "`":
            n += 1
        else:
            break
    n = max(n, 3)
    return "\n" + "`" * n + "\n"


INTRO = """\
# Project summary (generated by extract_summary.py)

This file is a concatenated dump of project documentation, config snippets, and \
extracted docstrings from Python modules. It is intended for search and context \
(e.g. one place to grep or to feed as context to an AI assistant). Use it to \
orient and find code; for full text of any file, open the canonical source in \
the repository. Markdown and config snippets below are truncated (first ~20 lines).

**Start here for an overview:** [CLAUDE.md](CLAUDE.md), [docs/architecture.md](docs/architecture.md), [docs/bundle.md](docs/bundle.md).

---

## Contents

"""


def main():
    sections: List[Tuple[str, str]] = []  # (path, content)

    for path in os.popen("git ls-files | sort").readlines():
        path = path.rstrip()
        if path == "summary.md":
            continue  # avoid embedding previous run / self-reference
        if ("Dockerfile" in path or
            path.endswith(".md") or
            path.endswith(".yml") or
            path.endswith(".sh")):
            with open(path) as f:
                head = "".join(f.readline() for _ in range(20))
            body = head.rstrip()
            # Avoid duplicate "# path" when file starts with that heading (for .md)
            lines = body.split("\n")
            if lines and path.endswith(".md") and lines[0].strip() == "# " + path:
                body = "\n".join(lines[1:]).rstrip()
            # Wrap Dockerfile, .yml, .sh in fenced code blocks so "#" and other
            # syntax are not interpreted as markdown (e.g. Dockerfile comments)
            if "Dockerfile" in path:
                content = "```dockerfile\n" + body + "\n\n    ...\n```\n"
            elif path.endswith(".yml") or path.endswith(".yaml"):
                content = "```yaml\n" + body + "\n\n    ...\n```\n"
            elif path.endswith(".sh"):
                content = "```sh\n" + body + "\n\n    ...\n```\n"
            else:
                # If we truncated inside a fenced block (e.g. ````plaintext), close it
                # so the next section (e.g. VIBES.md) is not rendered inside the block.
                # Use same number of backticks as opener (GFM requires >= opener).
                closing = _closing_fence_for_truncated_md(body)
                suffix = (closing + "\n    ...\n") if closing else "\n\n    ...\n"
                content = body + suffix
            sections.append((path, content))
        elif path.endswith(".py"):
            if path.endswith("extract_summary.py"):
                continue
            content = extract_docs(Path(path))
            sections.append((path, content))

    # Intro and TOC (no indent so markdown renders; links jump to sections below)
    # GitHub prefixes heading IDs with "user-content-"; use that so TOC links work on GitHub blob view.
    sys.stdout.write(INTRO)
    for path, _ in sections:
        anchor = _heading_anchor(path)
        sys.stdout.write(f"- [{path}](#user-content-{anchor})\n")
    sys.stdout.write("\n---\n\n")

    # Body: each section with optional conversation label, no indent
    # Explicit anchor so #user-content-{anchor} works in Cursor (GitHub already assigns that id to the heading).
    for path, content in sections:
        anchor = _heading_anchor(path)
        sys.stdout.write(f'<span id="user-content-{anchor}"></span>\n\n')
        sys.stdout.write(f"# {path}\n\n")
        if os.path.basename(path) in CONVERSATION_FILES:
            sys.stdout.write("*Design / conversation notes.*\n\n")
        sys.stdout.write(content)
        sys.stdout.write("\n\n")


if __name__ == "__main__":
    main()
