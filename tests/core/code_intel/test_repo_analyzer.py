from titan_agent.core.code_intel.repo_analyzer import (
    analyze_python_repository,
    format_repository_analysis,
)


def test_repository_map_reports_modules_local_imports_and_dependents(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text('"""Package docs."""\n', encoding="utf-8")
    (package / "models.py").write_text(
        '"""Domain model."""\nclass Item:\n    pass\n', encoding="utf-8"
    )
    (package / "service.py").write_text(
        "from .models import Item\n\ndef load() -> Item:\n    return Item()\n",
        encoding="utf-8",
    )
    (tmp_path / "runner.py").write_text(
        "from pkg.service import load\n\nload()\n", encoding="utf-8"
    )
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "ignored.py").write_text("secret = True\n", encoding="utf-8")

    report = analyze_python_repository(tmp_path)

    assert report["ok"] is True
    assert report["module_count"] == 4
    modules = {module["name"]: module for module in report["modules"]}
    assert modules["pkg.service"]["dependencies"] == ["pkg.models"]
    assert modules["pkg.models"]["imported_by_count"] == 1
    assert modules["runner"]["dependencies"] == ["pkg.service"]
    assert ".venv" not in format_repository_analysis(report)
    assert "not imported or executed" in report["notice"]


def test_repository_map_surfaces_syntax_errors_and_obeys_file_limit(tmp_path):
    (tmp_path / "a.py").write_text("def good(): return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("value = 3\n", encoding="utf-8")

    report = analyze_python_repository(tmp_path, max_files=2)

    assert report["module_count"] == 2
    assert report["syntax_error_count"] == 1
    assert report["truncated"] is True
    assert report["skipped_count"] == 1
    assert "SYNTAX ERROR" in format_repository_analysis(report)
    assert "Scan truncated" in format_repository_analysis(report)


def test_capped_scan_prioritizes_product_source_over_tests(tmp_path):
    (tmp_path / "titan_agent").mkdir()
    (tmp_path / "titan_agent" / "core.py").write_text("def core(): pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def main(): pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_many.py").write_text("def test_many(): pass\n", encoding="utf-8")

    report = analyze_python_repository(tmp_path, max_files=2)

    assert {module["path"] for module in report["modules"]} == {
        "titan_agent/core.py", "main.py"
    }
    assert report["truncated"] is True


def test_repository_map_rejects_invalid_workspace(tmp_path):
    report = analyze_python_repository(tmp_path / "missing")

    assert report["ok"] is False
    assert "not a directory" in report["error"]


def test_tool_repository_map_is_workspace_scoped_and_available(tmp_path, monkeypatch):
    from titan_agent.tools import ToolRegistry

    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "entry.py").write_text("def main(): return 0\n", encoding="utf-8")
    registry = ToolRegistry(workspace)

    result = registry.tool_analyze_python_repository(max_files=10)

    assert "STATIC PYTHON REPOSITORY MAP" in result
    assert "`entry`" in result
    names = {item["function"]["name"] for item in registry.get_tool_definitions()}
    assert "analyze_python_repository" in names
