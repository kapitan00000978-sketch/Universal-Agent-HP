"""Bounded, non-executing Python repository map and import-dependency analysis."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

MAX_FILES = 250
MAX_TOTAL_BYTES = 5_000_000
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist",
    "out", "target", "coverage", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}


def _module_name(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "<root>"


def _local_dependencies(tree: ast.Module, module_name: str, is_package: bool, known: set[str]) -> set[str]:
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module_name.split(".") if is_package else module_name.split(".")[:-1]
                ascend = node.level - 1
                if ascend > len(base):
                    continue
                base = base[:len(base) - ascend] if ascend else base
                prefix = ".".join(base)
                imported_module = ".".join(part for part in (prefix, node.module or "") if part)
            else:
                imported_module = node.module or ""
            if imported_module:
                candidates.add(imported_module)
            for alias in node.names:
                if alias.name != "*" and imported_module:
                    candidates.add(f"{imported_module}.{alias.name}")

    dependencies: set[str] = set()
    for candidate in candidates:
        parts = candidate.split(".")
        for end in range(len(parts), 0, -1):
            possible = ".".join(parts[:end])
            if possible in known and possible != module_name:
                dependencies.add(possible)
                break
    return dependencies


def analyze_python_repository(root: Path | str, max_files: int = 100) -> dict[str, Any]:
    """Inventory Python modules and local imports; never imports or executes them."""
    root = Path(root).resolve()
    if not root.is_dir():
        return {"ok": False, "error": "Workspace root is not a directory."}
    max_files = max(1, min(int(max_files), MAX_FILES))
    candidates: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root, topdown=True, followlinks=False):
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in EXCLUDED_DIRS and not name.startswith(".")
        )
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(directory) / filename
            try:
                resolved = path.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue  # Ignore symlinks escaping the selected workspace.
            if resolved.is_file():
                candidates.append(resolved)
    def priority(path: Path) -> tuple[int, str]:
        relative = path.relative_to(root)
        parts = relative.parts
        if parts[0] == "titan_agent":
            rank = 0  # Core product code is most useful when a scan is capped.
        elif len(parts) == 1:
            rank = 1  # Root launchers/config code next.
        elif parts[0] == "tests" or any(part.startswith("test_") for part in parts):
            rank = 3  # Keep test files, but don't let them crowd out implementation.
        else:
            rank = 2
        return rank, relative.as_posix()

    candidates.sort(key=priority)

    modules: dict[str, dict[str, Any]] = {}
    trees: dict[str, ast.Module] = {}
    used_bytes = 0
    skipped: list[str] = []
    for path in candidates:
        relative = path.relative_to(root)
        module_name = _module_name(relative)
        if len(modules) >= max_files:
            skipped.append(relative.as_posix())
            continue
        try:
            size = path.stat().st_size
            if used_bytes + size > MAX_TOTAL_BYTES:
                skipped.append(relative.as_posix())
                continue
            source = path.read_text(encoding="utf-8")
            used_bytes += size
            tree = ast.parse(source, filename=relative.as_posix(), type_comments=True)
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
            modules[module_name] = {
                "path": relative.as_posix(),
                "syntax_error": f"{type(exc).__name__}: {exc}",
                "docstring": "",
                "symbols": [],
                "dependencies": [],
            }
            continue

        symbols: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(("async " if isinstance(node, ast.AsyncFunctionDef) else "") + node.name)
            elif isinstance(node, ast.ClassDef):
                symbols.append(f"class {node.name}")
                symbols.extend(
                    f"{node.name}.{child.name}"
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
        modules[module_name] = {
            "path": relative.as_posix(),
            "syntax_error": None,
            "docstring": ast.get_docstring(tree) or "",
            "symbols": symbols[:60],
            "dependencies": [],
        }
        trees[module_name] = tree

    known = set(modules)
    for module_name, tree in trees.items():
        modules[module_name]["dependencies"] = sorted(
            _local_dependencies(
                tree,
                module_name,
                Path(modules[module_name]["path"]).name == "__init__.py",
                known,
            )
        )

    dependents: dict[str, list[str]] = {name: [] for name in modules}
    for source_name, data in modules.items():
        for dependency in data["dependencies"]:
            dependents[dependency].append(source_name)

    return {
        "ok": True,
        "root": str(root),
        "module_count": len(modules),
        "total_source_bytes": used_bytes,
        "truncated": bool(skipped),
        "skipped_count": len(skipped),
        "syntax_error_count": sum(item["syntax_error"] is not None for item in modules.values()),
        "modules": [
            {
                "name": name,
                **data,
                "imported_by_count": len(dependents[name]),
            }
            for name, data in sorted(modules.items())
        ],
        "notice": "Static Python AST/import map only: modules were not imported or executed; dynamic imports and runtime behavior are not represented.",
    }


def format_repository_analysis(report: dict[str, Any]) -> str:
    """Render a bounded repository overview for an agent's context window."""
    if not report.get("ok"):
        return f"### PYTHON REPOSITORY ANALYSIS FAILED\n{report.get('error', 'Unknown analysis error')}"
    lines = [
        "### STATIC PYTHON REPOSITORY MAP",
        f"- Modules inspected: {report['module_count']}; syntax errors: {report['syntax_error_count']}; source bytes: {report['total_source_bytes']}",
    ]
    for module in report["modules"]:
        line = f"- `{module['name']}` — `{module['path']}`"
        if module["syntax_error"]:
            line += f" [SYNTAX ERROR: {module['syntax_error']}]"
        lines.append(line)
        if module["docstring"]:
            lines.append(f"  - Purpose: {module['docstring'].replace(chr(10), ' ')[:180]}")
        if module["symbols"]:
            lines.append("  - Symbols: " + ", ".join(module["symbols"]))
        if module["dependencies"]:
            lines.append("  - Local imports: " + ", ".join(module["dependencies"]))
        if module["imported_by_count"]:
            lines.append(f"  - Imported by {module['imported_by_count']} local module(s).")
    if report["truncated"]:
        lines.append(f"- Scan truncated by limits; {report['skipped_count']} Python file(s) skipped.")
    lines.append(f"\n- Limit: {report['notice']}")
    return "\n".join(lines)
