from titan_agent.core.code_intel.source_analyzer import analyze_python_source, format_analysis


def test_source_analysis_explains_module_functions_and_internal_calls():
    source = '''"""Normalize incoming names."""
from collections.abc import Iterable


def normalize(value: str) -> str:
    """Trim outer whitespace without changing case."""
    return value.strip()


class NameStore:
    def save_all(self, values: Iterable[str]) -> list[str]:
        return [self.save(value) for value in values]

    def save(self, value: str) -> str:
        return normalize(value)
'''

    report = analyze_python_source(source, "names.py")
    formatted = format_analysis(report)

    assert report["ok"] is True
    assert report["module_docstring"] == "Normalize incoming names."
    assert report["imports"] == ["from collections.abc import Iterable"]
    symbols = {item["name"]: item for item in report["symbols"]}
    assert "normalize(value: str) -> str" == symbols["normalize"]["signature"]
    assert symbols["normalize"]["docstring"] == "Trim outer whitespace without changing case."
    assert "self.save [local]" in symbols["NameStore.save_all"]["calls"]
    assert "normalize [local]" in symbols["NameStore.save"]["calls"]
    assert "not executed" in report["notice"]
    assert "STATIC PYTHON ANALYSIS: names.py" in formatted
    assert "Trim outer whitespace" in formatted


def test_source_analysis_reports_syntax_error_without_execution():
    report = analyze_python_source("def broken(:\n    pass\n", "broken.py")

    assert report["ok"] is False
    assert "SyntaxError at line 1" in report["error"]
    assert "FAILED" in format_analysis(report)


def test_source_analysis_enforces_size_limit():
    report = analyze_python_source("x = 1\n" * 200_000)

    assert report["ok"] is False
    assert "analysis limit" in report["error"]


def test_analyze_python_file_reads_only_workspace_python_source(tmp_path, monkeypatch):
    from titan_agent.tools import ToolRegistry

    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "worker.py"
    source.write_text("def work(item):\n    return str(item).strip()\n", encoding="utf-8")
    registry = ToolRegistry(workspace)

    result = registry.tool_analyze_python_file("worker.py")

    assert "STATIC PYTHON ANALYSIS: worker.py" in result
    assert "work(item)" in result
    assert "not executed" in result
    assert "Error" in registry.tool_analyze_python_file("../outside.py")
    assert "Python (.py) files only" in registry.tool_analyze_python_file("README.md")
