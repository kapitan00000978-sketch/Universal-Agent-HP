import asyncio
from pathlib import Path

from titan_agent.deep_coder import DeepCoderEngine


class FakeTools:
    def __init__(self, workspace: Path, *, fail_write_for: str | None = None):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.fail_write_for = fail_write_for
        self.executed_commands = []

    def _resolve_path(self, relative_path):
        path = (self.workspace / relative_path).resolve()
        path.relative_to(self.workspace.resolve())
        return path

    def tool_write_file(self, path, content):
        if Path(path).name == self.fail_write_for:
            return "Error: simulated write failure"
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {target}."

    async def tool_execute_command(self, command, **_kwargs):
        self.executed_commands.append(command)
        return "### DOCKER SANDBOX (Exit 0)"


def test_invalid_python_is_not_marked_success_or_tested(tmp_path):
    tools = FakeTools(tmp_path)
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "bad_syntax",
        {"solution.py": "def broken(:\n    pass\n"},
        "assert True",
    ))

    assert result["status"] == "syntax_error"
    assert result["syntax_checks"]["solution.py"]["valid"] is False
    assert result["test_passed"] is None
    assert tools.executed_commands == []


def test_failed_implementation_write_cannot_be_hidden_by_passing_test(tmp_path):
    tools = FakeTools(tmp_path, fail_write_for="solution.py")
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "write_failure",
        {"solution.py": "def answer(): return 42"},
        "assert True",
    ))

    assert result["status"] == "write_error"
    assert result["test_passed"] is None
    assert tools.executed_commands == []


def test_test_file_write_failure_is_reported_without_execution(tmp_path):
    tools = FakeTools(tmp_path, fail_write_for="test_test_failure.py")
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "test_failure",
        {"solution.py": "def answer(): return 42"},
        "assert answer() == 42",
    ))

    assert result["status"] == "test_write_error"
    assert result["test_passed"] is None
    assert tools.executed_commands == []


def test_valid_cycle_reports_sandbox_test_result(tmp_path):
    tools = FakeTools(tmp_path)
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "valid_cycle",
        {"solution.py": "def answer(): return 42"},
        "assert answer() == 42",
    ))

    assert result["status"] == "success"
    assert result["syntax_checks"]["solution.py"]["valid"] is True
    assert result["test_passed"] is True
    assert tools.executed_commands


def test_no_test_script_is_reported_as_syntax_only_not_success(tmp_path):
    tools = FakeTools(tmp_path)
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "syntax_only",
        {"solution.py": "def answer(): return 42"},
    ))

    assert result["status"] == "syntax_only"
    assert result["test_passed"] is None
    assert tools.executed_commands == []


def test_syntax_verification_never_runs_generated_module(tmp_path):
    tools = FakeTools(tmp_path)
    engine = DeepCoderEngine(tools)
    sentinel = tmp_path / "executed.txt"
    source = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
    source_path = tmp_path / "candidate.py"
    source_path.write_text(source, encoding="utf-8")

    ok, message = asyncio.run(engine.verify_python_code(source_path))

    assert ok is True
    assert "without execution" in message
    assert sentinel.exists() is False


def test_task_name_rejects_path_separators_before_writing(tmp_path):
    tools = FakeTools(tmp_path)
    engine = DeepCoderEngine(tools)

    result = asyncio.run(engine.execute_coding_cycle(
        "../escape",
        {"solution.py": "answer = 1"},
        "assert True",
    ))

    assert result["status"] == "invalid_input"
    assert not (tmp_path / "solution.py").exists()
    assert tools.executed_commands == []
