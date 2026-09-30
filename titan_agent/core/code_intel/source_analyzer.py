"""Static, non-executing structural analysis for Python source files."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

MAX_SOURCE_BYTES = 1_000_000
MAX_ITEMS = 40


@dataclass
class _CallableInfo:
    name: str
    signature: str
    line: int
    end_line: int
    docstring: str
    calls: list[str]
    constructs: list[str]
    return_count: int
    raise_count: int
    async_function: bool


def _display(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return "?"


def _callable_info(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str = "") -> _CallableInfo:
    name = f"{prefix}.{node.name}" if prefix else node.name
    args = _display(node.args)
    signature = f"{name}({args})"
    if node.returns is not None:
        signature += f" -> {_display(node.returns)}"
    calls: list[str] = []
    constructs: set[str] = set()
    return_count = 0
    raise_count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            called = _display(child.func)
            if called and called not in calls:
                calls.append(called)
        if isinstance(child, ast.Return):
            return_count += 1
        elif isinstance(child, ast.Raise):
            raise_count += 1
        elif isinstance(child, ast.If):
            constructs.add("if/elif")
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            constructs.add("for loop")
        elif isinstance(child, ast.While):
            constructs.add("while loop")
        elif isinstance(child, ast.Try):
            constructs.add("exception handling")
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            constructs.add("context manager")
        elif isinstance(child, ast.Await):
            constructs.add("awaits asynchronous work")
        elif isinstance(child, (ast.Yield, ast.YieldFrom)):
            constructs.add("generator/yield")
    return _CallableInfo(
        name=name,
        signature=signature,
        line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        docstring=ast.get_docstring(node) or "",
        calls=calls[:MAX_ITEMS],
        constructs=sorted(constructs),
        return_count=return_count,
        raise_count=raise_count,
        async_function=isinstance(node, ast.AsyncFunctionDef),
    )


def analyze_python_source(source: str, filename: str = "<memory>") -> dict[str, Any]:
    """Summarize Python structure and behavior clues without importing/executing it."""
    if not isinstance(source, str):
        return {"ok": False, "error": "Source must be text."}
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return {"ok": False, "error": f"Source exceeds the {MAX_SOURCE_BYTES}-byte analysis limit."}
    try:
        tree = ast.parse(source, filename=filename, type_comments=True)
    except (SyntaxError, ValueError, RecursionError) as exc:
        location = f" at line {exc.lineno}" if isinstance(exc, SyntaxError) and exc.lineno else ""
        return {"ok": False, "error": f"{type(exc).__name__}{location}: {exc}"}

    imports: list[str] = []
    symbols: list[_CallableInfo] = []
    constants: list[str] = []
    module_calls: list[str] = []
    executable_lines: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            imported = ", ".join(
                alias.name + (f" as {alias.asname}" if alias.asname else "")
                for alias in node.names
            )
            imports.append(f"from {dots}{node.module or ''} import {imported}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_callable_info(node))
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_display(base) for base in node.bases)
            class_prefix = f"class {node.name}({bases})" if bases else f"class {node.name}"
            constants.append(f"{class_prefix} [lines {node.lineno}-{getattr(node, 'end_lineno', node.lineno)}]")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(_callable_info(child, node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants.append(f"{target.id} = {_display(node.value)[:120]}")
                else:
                    constants.append(f"module-level assignment at line {node.lineno}")
            if isinstance(node.value, ast.Call):
                module_calls.append(_display(node.value.func))
                executable_lines.append(node.lineno)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue  # module docstring or a standalone string literal
            if isinstance(node.value, ast.Call):
                module_calls.append(_display(node.value.func))
            executable_lines.append(node.lineno)
        elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
            executable_lines.append(node.lineno)
            module_calls.extend(
                _display(child.func)
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
            )

    defined_names = {symbol.name.rsplit(".", 1)[-1] for symbol in symbols}
    for symbol in symbols:
        symbol_calls = symbol.calls
        symbol.calls = [
            f"{call} [local]" if call in defined_names or call.rsplit(".", 1)[-1] in defined_names else call
            for call in symbol_calls
        ]

    return {
        "ok": True,
        "filename": filename,
        "line_count": len(source.splitlines()),
        "module_docstring": ast.get_docstring(tree) or "",
        "imports": imports[:MAX_ITEMS],
        "symbols": [symbol.__dict__ for symbol in symbols[:MAX_ITEMS]],
        "module_assignments_and_classes": constants[:MAX_ITEMS],
        "module_level_calls": list(dict.fromkeys(module_calls))[:MAX_ITEMS],
        "module_executable_lines": sorted(set(executable_lines))[:MAX_ITEMS],
        "truncated": len(symbols) > MAX_ITEMS or len(imports) > MAX_ITEMS or len(constants) > MAX_ITEMS,
        "notice": "Static AST summary only: code was not executed; dynamic dispatch, decorators, imports, and runtime data may change behavior.",
    }


def format_analysis(report: dict[str, Any]) -> str:
    """Format analysis data as concise guidance for the coding agent."""
    if not report.get("ok"):
        return f"### PYTHON SOURCE ANALYSIS FAILED\n{report.get('error', 'Unknown analysis error')}"
    lines = [f"### STATIC PYTHON ANALYSIS: {report['filename']}", f"- Lines: {report['line_count']}"]
    if report["module_docstring"]:
        lines.append(f"- Module purpose: {report['module_docstring']}")
    if report["imports"]:
        lines.append("- Imports: " + "; ".join(report["imports"]))
    if report["module_assignments_and_classes"]:
        lines.append("- Module declarations: " + "; ".join(report["module_assignments_and_classes"]))
    if report["module_level_calls"]:
        lines.append("- Calls at module load time (not executed by this analyzer): " + ", ".join(report["module_level_calls"]))
    if report["module_executable_lines"]:
        lines.append("- Module-level executable/control-flow lines: " + ", ".join(map(str, report["module_executable_lines"])))
    if report["symbols"]:
        lines.append("\n#### Functions and methods")
        for symbol in report["symbols"]:
            lines.append(f"- `{symbol['signature']}` (lines {symbol['line']}-{symbol['end_line']})")
            if symbol["docstring"]:
                lines.append(f"  - Purpose: {symbol['docstring'].replace(chr(10), ' ')[:240]}")
            else:
                lines.append("  - Purpose: no docstring; behavior inferred from syntax only")
            details = []
            if symbol["constructs"]:
                details.append(", ".join(symbol["constructs"]))
            details.append(f"{symbol['return_count']} return(s), {symbol['raise_count']} raise(s)")
            lines.append("  - Flow: " + "; ".join(details))
            if symbol["calls"]:
                lines.append("  - Calls: " + ", ".join(symbol["calls"]))
    else:
        lines.append("- No top-level functions or classes found.")
    if report["truncated"]:
        lines.append(f"- Output capped at {MAX_ITEMS} declarations/imports.")
    lines.append(f"\n- Limit: {report['notice']}")
    return "\n".join(lines)
