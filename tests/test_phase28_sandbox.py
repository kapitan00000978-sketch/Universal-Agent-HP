"""Tests for Phase 28: Execution Sandbox & Isolated Environment."""

import tempfile
from pathlib import Path

import pytest

from titan_agent.core.sandbox import (
    FilesystemSnapshot,
    SafeScriptRunner,
    SandboxEnvironment,
)
from titan_agent.tools import ToolRegistry


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        w_path = Path(tmpdir)
        (w_path / "hello.py").write_text("print('original')", encoding="utf-8")
        (w_path / "config.json").write_text('{"version": 1}', encoding="utf-8")
        subdir = w_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("def run(): pass\n", encoding="utf-8")
        yield w_path


class TestFilesystemSnapshot:
    def test_snapshot_capture_and_diff(self, temp_workspace: Path):
        snap1 = FilesystemSnapshot.capture("v1", temp_workspace)
        assert snap1.name == "v1"
        assert len(snap1.file_hashes) >= 3
        assert "hello.py" in snap1.file_hashes

        # Modify a file, add a file, delete a file
        (temp_workspace / "hello.py").write_text("print('modified')", encoding="utf-8")
        (temp_workspace / "new_file.txt").write_text("created", encoding="utf-8")
        (temp_workspace / "config.json").unlink()

        snap2 = FilesystemSnapshot.capture("v2", temp_workspace)
        diff = snap1.diff(snap2)

        assert "hello.py" in diff["modified"]
        assert "new_file.txt" in diff["added"]
        assert "config.json" in diff["removed"]


class TestSandboxEnvironment:
    def test_snapshot_and_rollback(self, temp_workspace: Path):
        env = SandboxEnvironment(temp_workspace)
        snap = env.create_snapshot("clean_state")
        assert snap.name == "clean_state"

        # Make mutations
        (temp_workspace / "hello.py").write_text("print('corrupted')", encoding="utf-8")
        (temp_workspace / "dangerous_artifact.sh").write_text("rm -rf /", encoding="utf-8")
        (temp_workspace / "src" / "main.py").unlink()

        assert not (temp_workspace / "src" / "main.py").exists()
        assert (temp_workspace / "dangerous_artifact.sh").exists()

        # Perform rollback
        report = env.rollback("clean_state")
        assert report["success"] is True
        # hello.py is restored (modified), src/main.py is recreated (deleted)
        assert len(report["restored"]) + len(report["recreated"]) >= 2
        assert "dangerous_artifact.sh" in report["deleted_new"]

        # Verify state is fully restored
        assert (temp_workspace / "hello.py").read_text(encoding="utf-8") == "print('original')"
        assert (temp_workspace / "src" / "main.py").exists()
        assert not (temp_workspace / "dangerous_artifact.sh").exists()

    def test_rollback_nonexistent_snapshot(self, temp_workspace: Path):
        env = SandboxEnvironment(temp_workspace)
        with pytest.raises(KeyError, match="not found"):
            env.rollback("nonexistent")


class TestSafeScriptRunner:
    def test_run_python_success(self, temp_workspace: Path):
        runner = SafeScriptRunner(temp_workspace)
        res = runner.run("print('Hello from safe runner!')", language="python")
        assert res.success is True
        assert res.exit_code == 0
        assert "Hello from safe runner!" in res.stdout
        assert res.rolled_back is False

    def test_run_dangerous_pattern_blocked(self, temp_workspace: Path):
        runner = SafeScriptRunner(temp_workspace)
        dangerous_code = "import os\nos.system(':(){ :|:& };:')"
        res = runner.run(dangerous_code, language="python")
        assert res.success is False
        assert res.exit_code == -1
        assert "Security alert" in (res.error or "")

    def test_run_exception_with_auto_rollback(self, temp_workspace: Path):
        env = SandboxEnvironment(temp_workspace)
        runner = SafeScriptRunner(temp_workspace, sandbox_env=env)

        # Code that mutates a file and then crashes
        bad_code = (
            "from pathlib import Path\n"
            "Path('hello.py').write_text('mutated before crash')\n"
            "raise RuntimeError('Boom!')\n"
        )
        res = runner.run(bad_code, language="python", auto_rollback=True)
        assert res.success is False
        assert res.exit_code != 0
        assert res.rolled_back is True

        # Check hello.py was rolled back
        assert (temp_workspace / "hello.py").read_text(encoding="utf-8") == "print('original')"

    def test_run_timeout(self, temp_workspace: Path):
        runner = SafeScriptRunner(temp_workspace)
        infinite_code = "import time\ntime.sleep(10)\n"
        res = runner.run(infinite_code, language="python", timeout=1.0)
        assert res.success is False
        assert res.exit_code == -1
        assert "timeout" in (res.error or "").lower()


class TestToolRegistrySandboxIntegration:
    def test_tool_sandbox_definitions(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        defs = tools.get_tool_definitions()
        names = [d.get("function", {}).get("name") for d in defs]
        assert "sandbox_execute" in names
        assert "sandbox_snapshot_create" in names
        assert "sandbox_snapshot_rollback" in names

    def test_tool_sandbox_snapshot_and_rollback(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        create_res = tools.tool_sandbox_snapshot_create("checkpoint_1")
        assert "WORKSPACE SNAPSHOT CREATED" in create_res
        assert "checkpoint_1" in create_res

        # Mutate
        (temp_workspace / "hello.py").write_text("mutated")
        (temp_workspace / "test_tmp.txt").write_text("tmp")

        rollback_res = tools.tool_sandbox_snapshot_rollback("checkpoint_1")
        assert "WORKSPACE ROLLBACK EXECUTED" in rollback_res
        assert "SUCCESS" in rollback_res

        assert (temp_workspace / "hello.py").read_text(encoding="utf-8") == "print('original')"
        assert not (temp_workspace / "test_tmp.txt").exists()

    def test_tool_sandbox_execute_python(self, temp_workspace: Path, monkeypatch):
        import asyncio

        tools = ToolRegistry(temp_workspace)
        captured = {}

        async def fake_sandbox_command(command, cwd="", _timeout_override=None):
            captured["command"] = command
            script = command.split(" ", 2)[-1]
            captured["script"] = script
            return "### DOCKER COMMAND SANDBOX (Exit 0)\\nSTDOUT:\\n42"

        monkeypatch.setattr(tools, "tool_execute_command", fake_sandbox_command)
        exec_res = asyncio.run(tools.tool_sandbox_execute("print(40 + 2)", language="python"))
        assert "SANDBOX EXECUTION RESULT" in exec_res
        assert "42" in exec_res
        assert "PASSED" in exec_res
        assert captured["command"].startswith("python -I /workspace/")
        assert not list(temp_workspace.glob(".titan_python_exec_*.py"))

    def test_tool_sandbox_execute_empty(self, temp_workspace: Path):
        import asyncio

        tools = ToolRegistry(temp_workspace)
        assert "Error: code is required" in asyncio.run(tools.tool_sandbox_execute(""))
        assert "Error: snapshot name is required" in tools.tool_sandbox_snapshot_create("")
        assert "Error: snapshot name is required" in tools.tool_sandbox_snapshot_rollback("")
