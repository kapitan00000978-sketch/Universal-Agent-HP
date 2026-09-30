"""
Phase 40 — Autonomous Dynamic Tool Synthesizer (Avtonom Vosita Sintezi).

Allows Titan Agent to synthesize, test, verify, and register new tools
on-the-fly during a live execution session when existing tools are insufficient.
"""
from __future__ import annotations

import ast
import asyncio
import json
import keyword
import logging
import os
import re
import tempfile
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class SynthesizedToolSpec:
    """Specification of an autonomously created tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    python_code: str
    test_code: str = ""
    is_async: bool = True
    verified: bool = False
    verification_notes: str = ""
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0)


class DynamicToolSynthesizer:
    """Synthesizes, tests in sandbox, and hot-injects Python tools into live registries."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        sandbox_runner: Callable[[Path, float], Awaitable[tuple[bool, str]]] | None = None,
        runtime_runner: Callable[[str, str, dict[str, Any], float], Awaitable[str]] | None = None,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.storage_dir = self.workspace_root / ".titan_synthesized_tools"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.synthesized_registry: dict[str, SynthesizedToolSpec] = {}
        self.sandbox_runner = sandbox_runner
        self.runtime_runner = runtime_runner

    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """Validates that the provided code is syntactically sound Python."""
        try:
            tree = ast.parse(code)
            # Ensure no syntax errors and contains at least one function definition
            funcs = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not funcs:
                return False, "Code must define at least one function or async function."
            return True, f"Syntax valid. Functions defined: {', '.join(funcs)}"
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"AST parsing failed: {e!s}"

    async def test_in_isolated_sandbox(
        self,
        tool_code: str,
        test_code: str,
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """Verify generated code only through an injected OS/container runner.

        No host-subprocess fallback is permitted. Without a sandbox runner the
        generated source is not executed and cannot be registered.
        """
        if self.sandbox_runner is None:
            return False, "Docker security sandbox runner is not configured; generated code was not executed."
        with tempfile.TemporaryDirectory(prefix="titan_synth_tool_") as tmpdir:
            tmppath = Path(tmpdir)
            module_file = tmppath / "synthesized_module.py"
            test_file = tmppath / "test_synthesized.py"
            module_file.write_text(tool_code, encoding="utf-8")
            runner_script = (
                "import asyncio\n"
                "import synthesized_module\n\n"
                f"{test_code}\n\n"
                "print('___SYNTHESIS_TEST_PASSED___')\n"
            )
            test_file.write_text(runner_script, encoding="utf-8")
            try:
                return await self.sandbox_runner(tmppath, max(0.5, min(float(timeout), 300.0)))
            except Exception as exc:  # noqa: BLE001 - never fall back to host execution
                return False, f"Docker sandbox execution failed; generated code was not registered: {exc!s}"

    @staticmethod
    def _name_is_available(name: str, registry: Any) -> bool:
        """Do not allow synthesized definitions to shadow live registry tools."""
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or keyword.iskeyword(name):
            return False
        if hasattr(registry, f"tool_{name}"):
            return False
        try:
            definitions = registry.get_tool_definitions()
        except (AttributeError, TypeError):
            definitions = []
        for definition in definitions or []:
            function = definition.get("function", definition)
            if str(function.get("name", "")).lower() == name:
                return False
        synthesized = getattr(registry, "_synthesized_definitions", {})
        return name not in synthesized

    def _make_runtime_handler(self, name: str, code: str):
        """Create a proxy that delegates every invocation to the OS sandbox."""
        if self.runtime_runner is None:
            raise RuntimeError("dynamic tool runtime sandbox is not configured")

        async def _handler(**arguments):
            try:
                return await self.runtime_runner(name, code, arguments, 30.0)
            except Exception as exc:  # noqa: BLE001 - never execute generated code on the host
                log.warning("isolated dynamic tool '%s' failed (%s)", name, type(exc).__name__)
                return f"Error: isolated dynamic tool execution failed ({type(exc).__name__})."

        _handler.__name__ = name
        return _handler

    async def synthesize_and_register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        python_code: str,
        test_code: str,
        registry: Any,
    ) -> tuple[bool, str]:
        """Validate and test in containers; register only an isolated runtime proxy."""
        norm_name = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()
        if not norm_name or not self._name_is_available(norm_name, registry):
            return False, (
                f"Error: Tool name '{norm_name or name}' is invalid or conflicts with "
                "an existing tool; synthesized tools cannot replace registered tools."
            )
        if not test_code or not test_code.strip():
            return False, "Error: test_code is required; untested tools cannot be registered or marked verified."
        if os.getenv("TITAN_DYNAMIC_TOOLS_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return False, "Error: dynamic Python tool synthesis is disabled; explicitly enable it only in a trusted deployment."
        if self.sandbox_runner is None or self.runtime_runner is None:
            return False, (
                "Error: both verification and invocation container runners are required; "
                "generated code will not be executed or registered on the host."
            )
        if not isinstance(python_code, str) or len(python_code.encode("utf-8")) > 128 * 1024:
            return False, "Error: generated Python source must be at most 128 KiB."
        if not isinstance(test_code, str) or len(test_code.encode("utf-8")) > 128 * 1024:
            return False, "Error: generated test source must be at most 128 KiB."
        if not isinstance(description, str) or len(description) > 1000:
            return False, "Error: synthesized tool descriptions must be text of at most 1000 characters."
        if not isinstance(parameters, dict):
            return False, "Error: tool parameters must be a JSON object schema."
        try:
            serialized_parameters = json.dumps(parameters, ensure_ascii=False)
        except (TypeError, ValueError):
            return False, "Error: tool parameters must be JSON-serializable."
        if len(serialized_parameters.encode("utf-8")) > 64 * 1024:
            return False, "Error: tool parameter schema must be at most 64 KiB."

        # 1. Syntax and named entry-point validation without importing code.
        valid_syntax, syn_msg = self.validate_syntax(python_code)
        if not valid_syntax:
            return False, f"Validation Error: {syn_msg}"
        function_names = {
            node.name for node in ast.walk(ast.parse(python_code))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if norm_name not in function_names and f"tool_{norm_name}" not in function_names:
            return False, f"Validation Error: source must define '{norm_name}' or 'tool_{norm_name}'."

        # 2. Isolated sandbox test.
        passed, test_msg = await self.test_in_isolated_sandbox(python_code, test_code)
        if not passed:
            return False, f"Sandbox Verification FAILED:\n{test_msg}"
        test_output = test_msg

        # 3. Register only a proxy: generated source is never imported or run here.
        handler_fn = self._make_runtime_handler(norm_name, python_code)
        setattr(registry, f"tool_{norm_name}", handler_fn)

        # 5. Build standard OpenAI tool schema definition
        tool_def = {
            "type": "function",
            "function": {
                "name": norm_name,
                "description": f"[SYNTHESIZED TOOL] {description}",
                "parameters": parameters if isinstance(parameters, dict) and "type" in parameters else {
                    "type": "object",
                    "properties": parameters if isinstance(parameters, dict) else {},
                },
            },
        }

        # Store in dynamic definitions if registry supports it
        if hasattr(registry, "_synthesized_definitions"):
            registry._synthesized_definitions[norm_name] = tool_def
        else:
            registry._synthesized_definitions = {norm_name: tool_def}

        spec = SynthesizedToolSpec(
            name=norm_name,
            description=description,
            parameters=parameters,
            python_code=python_code,
            test_code=test_code,
            is_async=True,
            verified=True,
            verification_notes=test_output,
        )
        self.synthesized_registry[norm_name] = spec
        self.persist_tool(spec)

        log.info("Successfully synthesized, verified and registered dynamic tool: %s", norm_name)
        return True, (
            f"✅ Tool '{norm_name}' synthesized and verified successfully!\n"
            f"- Registered name: {norm_name}\n"
            "- Async: True (container proxy)\n"
            f"- Test: {test_output.splitlines()[0] if test_output else 'OK'}\n"
            "The tool is immediately available for invocation."
        )

    def persist_tool(self, spec: SynthesizedToolSpec) -> Path:
        """Persists the tool specification and source code to disk."""
        target_dir = self.storage_dir / spec.name
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "tool.py").write_text(spec.python_code, encoding="utf-8")
        if spec.test_code:
            (target_dir / "test.py").write_text(spec.test_code, encoding="utf-8")
        meta = {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "is_async": spec.is_async,
            "verified": spec.verified,
            "created_at": spec.created_at,
        }
        (target_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target_dir

    async def _verify_persisted_tool(self, code: str, test_code: str, timeout: float = 30.0) -> bool:
        """Re-run persisted tests only through the configured container runner."""
        passed, _ = await self.test_in_isolated_sandbox(code, test_code, timeout=timeout)
        return passed

    async def load_persisted_tools(self, registry: Any) -> list[str]:
        """Loads persisted definitions after container-backed re-verification.

        The source is never imported into the agent process. Invocation proxies
        send it to the configured container runner each time.
        """
        loaded = []
        if os.getenv("TITAN_DYNAMIC_TOOLS_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            log.warning("Skipping persisted dynamic tools: TITAN_DYNAMIC_TOOLS_ENABLED is not enabled")
            return loaded
        if self.sandbox_runner is None or self.runtime_runner is None:
            log.warning("Skipping persisted dynamic tools: container runners are not configured")
            return loaded
        if not self.storage_dir.exists():
            return loaded

        for tool_dir in self.storage_dir.iterdir():
            if not tool_dir.is_dir():
                continue
            meta_file = tool_dir / "meta.json"
            code_file = tool_dir / "tool.py"
            if not meta_file.exists() or not code_file.exists():
                continue
            try:
                if meta_file.stat().st_size > 64 * 1024 or code_file.stat().st_size > 128 * 1024:
                    log.warning("Skipping oversized persisted synthesized tool in %s", tool_dir)
                    continue
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                code = code_file.read_text(encoding="utf-8")
                name = meta.get("name")
                test_file = tool_dir / "test.py"
                if (
                    not isinstance(name, str)
                    or not isinstance(meta.get("parameters", {"type": "object", "properties": {}}), dict)
                    or not meta.get("verified")
                    or not test_file.is_file()
                    or test_file.stat().st_size > 128 * 1024
                    or not self._name_is_available(name, registry)
                ):
                    log.warning("Skipping unverified or conflicting synthesized tool in %s", tool_dir)
                    continue
                test_code = test_file.read_text(encoding="utf-8")
                valid_syntax, _ = self.validate_syntax(code)
                function_names = {
                    node.name for node in ast.walk(ast.parse(code))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if not valid_syntax or (name not in function_names and f"tool_{name}" not in function_names):
                    log.warning("Skipping malformed persisted synthesized tool %s", name)
                    continue
                if not await self._verify_persisted_tool(code, test_code):
                    log.warning("Persisted synthesized tool %s failed re-verification", name)
                    continue
                handler_fn = self._make_runtime_handler(name, code)
                setattr(registry, f"tool_{name}", handler_fn)
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"[SYNTHESIZED TOOL] {meta.get('description', '')}",
                        "parameters": meta.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
                if not hasattr(registry, "_synthesized_definitions"):
                    registry._synthesized_definitions = {}
                registry._synthesized_definitions[name] = tool_def
                loaded.append(name)
            except Exception as e:
                log.warning("Could not load persisted synthesized tool from %s: %s", tool_dir, e)
        return loaded
