"""Security boundary tests for default command and file access."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from titan_agent.tools import ToolRegistry


class _CompletedProcess:
    returncode = 0

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(b"sandbox-ok")
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture(autouse=True)
def _clear_full_access(monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    monkeypatch.delenv("TITAN_ABSOLUTE_ACCESS", raising=False)


def test_missing_docker_fails_closed_without_host_fallback(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setattr("titan_agent.tools.shutil.which", lambda _name: None)

    async def host_fallback_must_not_run(*_args, **_kwargs):
        pytest.fail("host fallback must never run when Docker is unavailable")

    monkeypatch.setattr(registry, "_tool_execute_command_host", host_fallback_must_not_run)
    result = asyncio.run(registry.tool_execute_command("echo unsafe"))

    assert "Docker was not found" in result
    assert "No host-shell fallback" in result


def test_default_command_uses_hardened_docker_workspace_sandbox(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setattr("titan_agent.tools.shutil.which", lambda _name: "/usr/bin/docker")
    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _CompletedProcess()

    monkeypatch.setattr("titan_agent.tools.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    result = asyncio.run(registry.tool_execute_command("printf sandbox-ok", cwd="."))
    command = captured["args"]

    assert "sandbox-ok" in result
    assert command[0:2] == ("/usr/bin/docker", "run")
    assert "--name" in command
    assert command[command.index("--name") + 1].startswith("titan-command-")
    assert "--rm" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--pids-limit=128" in command
    assert "--memory=512m" in command
    assert "--cpus=1.0" in command
    assert "type=bind" in command[command.index("--mount") + 1]
    assert str(tmp_path.resolve()) in command[command.index("--mount") + 1]
    assert "--pull=never" in command
    assert command[-4:-1] == ("titan-agent-sandbox:local", "sh", "-lc")


def test_command_cwd_cannot_escape_workspace(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path / "workspace")
    monkeypatch.setattr("titan_agent.tools.shutil.which", lambda _name: "/usr/bin/docker")
    called = False

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        nonlocal called
        called = True
        return _CompletedProcess()

    monkeypatch.setattr("titan_agent.tools.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    result = asyncio.run(registry.tool_execute_command("pwd", cwd="../outside"))

    assert "must be inside the workspace" in result
    assert not called


def test_full_access_is_explicit_host_execution_bypass(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setenv("TITAN_FULL_ACCESS", "1")
    monkeypatch.setattr("titan_agent.tools.shutil.which", lambda _name: None)

    async def fake_host(command, cwd, timeout):
        assert command == "echo trusted"
        assert cwd == ""
        assert timeout >= 45
        return "explicit-host"

    monkeypatch.setattr(registry, "_tool_execute_command_host", fake_host)
    assert asyncio.run(registry.tool_execute_command("echo trusted")) == "explicit-host"


def test_dynamic_tool_verification_uses_container_and_no_host_fallback(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path)
    monkeypatch.setenv("TITAN_DYNAMIC_TOOLS_ENABLED", "true")
    captured = {}

    async def fake_docker(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        captured["test_source"] = (Path(kwargs["_mount_path"]) / "test_synthesized.py").read_text()
        return "### DOCKER SANDBOX [titan-agent-sandbox:local] (Exit 0)\\nSTDOUT:\\n___SYNTHESIS_TEST_PASSED___"

    monkeypatch.setattr(registry, "tool_docker_sandbox_run", fake_docker)
    result = asyncio.run(registry.tool_synthesize_tool(
        name="verified_helper",
        description="test only",
        python_code="def verified_helper(value): return value + 1",
        test_code="assert synthesized_module.verified_helper(1) == 2",
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
    ))

    assert "synthesized and verified" in result
    assert captured["command"] == "python -E -s /workspace/test_synthesized.py"
    assert captured["network"] == "none"
    assert captured["timeout"] <= 300
    assert "assert synthesized_module.verified_helper(1) == 2" in captured["test_source"]
    assert registry.tool_verified_helper.__name__ == "verified_helper"


def test_dynamic_tool_invocation_uses_fresh_network_disabled_container(tmp_path, monkeypatch):
    import base64
    import json
    import re

    registry = ToolRegistry(tmp_path)
    captured = {}

    async def fake_docker(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        root = Path(kwargs["_mount_path"])
        captured["source"] = (root / "synthesized_module.py").read_text()
        captured["arguments"] = json.loads((root / "arguments.json").read_text())
        captured["runner"] = (root / "run_dynamic_tool.py").read_text()
        marker = re.search(r"(__TITAN_DYNAMIC_RESULT_[0-9a-f]+__:)", captured["runner"]).group(1)
        payload = base64.b64encode(json.dumps({"ok": True, "value": "isolated-result"}).encode()).decode()
        return f"### DOCKER SANDBOX [titan-agent-sandbox:local] (Exit 0)\\nSTDOUT:\\n{marker}{payload}"

    monkeypatch.setattr(registry, "tool_docker_sandbox_run", fake_docker)
    marker_path = tmp_path / "generated-source-ran-on-host"
    code = f"from pathlib import Path\nPath({str(marker_path)!r}).write_text('bad')\ndef safe_helper(value): return value\n"

    result = asyncio.run(registry._run_dynamic_tool_container(
        "safe_helper", code, {"value": "x"}, 10.0
    ))

    assert result == "isolated-result"
    assert captured["command"] == "python -I -s /workspace/run_dynamic_tool.py"
    assert captured["network"] == "none"
    assert captured["arguments"] == {"value": "x"}
    assert "sys.path.insert(0" in captured["runner"]
    assert not marker_path.exists()


def test_file_tool_paths_are_confined_to_workspace(tmp_path):
    registry = ToolRegistry(tmp_path / "workspace")
    with pytest.raises(PermissionError, match="outside the configured workspace"):
        registry._resolve_path(tmp_path / "outside.txt")


def test_full_access_retains_outside_workspace_path_bypass(tmp_path, monkeypatch):
    registry = ToolRegistry(tmp_path / "workspace")
    monkeypatch.setenv("TITAN_FULL_ACCESS", "1")
    assert registry._resolve_path(tmp_path / "outside.txt") == (tmp_path / "outside.txt").resolve()
