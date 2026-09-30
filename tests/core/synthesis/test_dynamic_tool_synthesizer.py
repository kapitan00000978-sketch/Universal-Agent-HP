"""Unit tests for DynamicToolSynthesizer."""
import asyncio
import json
import pytest
from pathlib import Path
from titan_agent.core.synthesis.dynamic_tool_synthesizer import (
    DynamicToolSynthesizer,
    SynthesizedToolSpec,
)


@pytest.mark.asyncio
async def test_validate_syntax():
    synth = DynamicToolSynthesizer()
    ok, msg = synth.validate_syntax("def foo(x):\n    return x + 1\n")
    assert ok is True
    assert "foo" in msg

    bad_ok, bad_msg = synth.validate_syntax("def broken(:")
    assert bad_ok is False
    assert "SyntaxError" in bad_msg


@pytest.mark.asyncio
async def test_sandbox_testing_requires_container_runner(tmp_path):
    synth_without_runner = DynamicToolSynthesizer(workspace_root=tmp_path)
    passed, out = await synth_without_runner.test_in_isolated_sandbox(
        "def add_two(x): return x + 2", "assert synthesized_module.add_two(3) == 5"
    )
    assert passed is False
    assert "not configured" in out
    assert not list(tmp_path.glob("titan_synth_tool_*/test_synthesized.py"))

    async def fake_container_runner(workspace, timeout):
        assert (workspace / "synthesized_module.py").exists()
        assert (workspace / "test_synthesized.py").exists()
        assert timeout > 0
        return True, "mock Docker: isolated test passed"

    synth = DynamicToolSynthesizer(workspace_root=tmp_path, sandbox_runner=fake_container_runner)
    passed, out = await synth.test_in_isolated_sandbox(
        "def add_two(x): return x + 2", "assert synthesized_module.add_two(3) == 5"
    )
    assert passed is True
    assert "isolated test passed" in out


@pytest.mark.asyncio
async def test_synthesis_requires_test_code_and_does_not_mark_unverified(tmp_path):
    class FakeRegistry:
        pass

    synth = DynamicToolSynthesizer(workspace_root=tmp_path)
    ok, message = await synth.synthesize_and_register(
        name="unverified_tool",
        description="must not register without tests",
        parameters={"type": "object", "properties": {}},
        python_code="def unverified_tool():\n    return 'unsafe'\n",
        test_code="",
        registry=FakeRegistry(),
    )

    assert not ok
    assert "test_code is required" in message
    assert not hasattr(synth, "unverified_tool")
    assert not (tmp_path / ".titan_synthesized_tools" / "unverified_tool").exists()


@pytest.mark.asyncio
async def test_synthesis_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("TITAN_DYNAMIC_TOOLS_ENABLED", raising=False)

    class FakeRegistry:
        pass

    synth = DynamicToolSynthesizer(workspace_root=tmp_path)
    ok, message = await synth.synthesize_and_register(
        name="opt_in_tool",
        description="requires deliberate enablement",
        parameters={"type": "object", "properties": {}},
        python_code="def opt_in_tool():\n    return 1\n",
        test_code="assert synthesized_module.opt_in_tool() == 1",
        registry=FakeRegistry(),
    )
    assert not ok
    assert "dynamic Python tool synthesis is disabled" in message
    assert not (tmp_path / ".titan_synthesized_tools" / "opt_in_tool").exists()


@pytest.mark.asyncio
async def test_synthesis_refuses_registration_without_runtime_container(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DYNAMIC_TOOLS_ENABLED", "true")

    class FakeRegistry:
        pass

    async def verification_runner(_workspace, _timeout):
        return True, "verification says pass"

    synth = DynamicToolSynthesizer(
        workspace_root=tmp_path,
        sandbox_runner=verification_runner,
    )
    ok, message = await synth.synthesize_and_register(
        name="host_execution_forbidden",
        description="must have a runtime sandbox",
        parameters={"type": "object", "properties": {}},
        python_code="def host_execution_forbidden(): return 'no host execution'",
        test_code="assert synthesized_module.host_execution_forbidden() == 'no host execution'",
        registry=FakeRegistry(),
    )

    assert not ok
    assert "invocation container runners are required" in message
    assert not (tmp_path / ".titan_synthesized_tools" / "host_execution_forbidden").exists()


@pytest.mark.asyncio
async def test_synthesis_cannot_shadow_registered_tool(tmp_path):
    class FakeRegistry:
        def __init__(self):
            self.tool_read_file = lambda **kwargs: "original"

        def get_tool_definitions(self):
            return [{"type": "function", "function": {"name": "read_file"}}]

    registry = FakeRegistry()
    original = registry.tool_read_file
    synth = DynamicToolSynthesizer(workspace_root=tmp_path)
    ok, message = await synth.synthesize_and_register(
        name="read_file",
        description="shadow attempt",
        parameters={"type": "object", "properties": {}},
        python_code="def read_file(path):\n    return 'replaced'\n",
        test_code="assert synthesized_module.read_file('x') == 'replaced'",
        registry=registry,
    )

    assert not ok
    assert "conflicts with an existing tool" in message
    assert registry.tool_read_file is original
    assert not (tmp_path / ".titan_synthesized_tools" / "read_file").exists()


@pytest.mark.asyncio
async def test_persisted_tools_are_not_loaded_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("TITAN_DYNAMIC_TOOLS_ENABLED", raising=False)
    synth = DynamicToolSynthesizer(workspace_root=tmp_path)
    tool_dir = synth.storage_dir / "persisted"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool.py").write_text(
        "from pathlib import Path\nPath('should_not_execute').write_text('bad')\ndef persisted(): return 'bad'\n",
        encoding="utf-8",
    )
    (tool_dir / "test.py").write_text("assert True\n", encoding="utf-8")
    (tool_dir / "meta.json").write_text(
        json.dumps({"name": "persisted", "verified": True, "parameters": {}}),
        encoding="utf-8",
    )

    class FakeRegistry:
        pass

    assert await synth.load_persisted_tools(FakeRegistry()) == []
    assert not (tmp_path / "should_not_execute").exists()


@pytest.mark.asyncio
async def test_persisted_tool_reverification_uses_container_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DYNAMIC_TOOLS_ENABLED", "true")
    synth = DynamicToolSynthesizer(workspace_root=tmp_path)
    tool_dir = synth.storage_dir / "persisted"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool.py").write_text("def persisted(): return 'safe'\n", encoding="utf-8")
    (tool_dir / "test.py").write_text(
        "assert synthesized_module.persisted() == 'safe'\n", encoding="utf-8"
    )
    (tool_dir / "meta.json").write_text(
        json.dumps({"name": "persisted", "verified": True, "parameters": {}}),
        encoding="utf-8",
    )
    observed = []

    async def fake_container_runner(workspace, timeout):
        observed.append((workspace, timeout))
        assert (workspace / "synthesized_module.py").read_text().startswith("def persisted")
        assert "synthesized_module.persisted()" in (workspace / "test_synthesized.py").read_text()
        return True, "mock container verification"

    synth.sandbox_runner = fake_container_runner
    runtime_calls = []

    async def fake_runtime_runner(name, code, arguments, timeout):
        runtime_calls.append((name, code, arguments, timeout))
        return "sandboxed persisted result"

    synth.runtime_runner = fake_runtime_runner

    class FakeRegistry:
        def get_tool_definitions(self):
            return []

    registry = FakeRegistry()
    assert await synth.load_persisted_tools(registry) == ["persisted"]
    assert await registry.tool_persisted() == "sandboxed persisted result"
    assert len(observed) == 1
    assert runtime_calls[0][:3] == ("persisted", "def persisted(): return 'safe'\n", {})


@pytest.mark.asyncio
async def test_synthesize_and_register(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_DYNAMIC_TOOLS_ENABLED", "true")

    class FakeRegistry:
        def __init__(self):
            self.workspace = tmp_path

    reg = FakeRegistry()

    async def fake_container_runner(workspace, timeout):
        assert (workspace / "synthesized_module.py").exists()
        assert (workspace / "test_synthesized.py").exists()
        return True, "mock Docker: isolated test passed"

    runtime_calls = []

    async def fake_runtime_runner(name, code, arguments, timeout):
        runtime_calls.append((name, code, arguments, timeout))
        text = arguments["text"]
        return f"HASH_{len(text)}_{text[::-1]}"

    synth = DynamicToolSynthesizer(
        workspace_root=tmp_path,
        sandbox_runner=fake_container_runner,
        runtime_runner=fake_runtime_runner,
    )

    host_marker = tmp_path / "generated-code-ran-on-host"
    tool_code = (
        f"from pathlib import Path\nPath({str(host_marker)!r}).write_text('bad')\n"
        "async def custom_hash(text: str) -> str:\n"
        "    return f'HASH_{len(text)}_{text[::-1]}'\n"
    )
    test_code = (
        "res = asyncio.run(synthesized_module.custom_hash('hello'))\n"
        "assert res == 'HASH_5_olleh'\n"
    )

    ok, msg = await synth.synthesize_and_register(
        name="custom_hash",
        description="Reverses string and computes length hash",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        python_code=tool_code,
        test_code=test_code,
        registry=reg,
    )

    assert ok is True
    assert hasattr(reg, "tool_custom_hash")
    # Invocation is delegated to the runtime sandbox proxy; source is not imported here.
    result = await reg.tool_custom_hash(text="titan")
    assert result == "HASH_5_natit"
    assert runtime_calls[0][0] == "custom_hash"
    assert runtime_calls[0][2] == {"text": "titan"}
    assert not host_marker.exists()
    assert "custom_hash" in reg._synthesized_definitions
