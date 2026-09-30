import re
import shlex
from pathlib import Path
from typing import Any

from .config import full_access_enabled
from .tools import ToolRegistry


class DeepCoderEngine:
    """Write generated files, validate Python syntax, and run tests via sandbox.

    This engine verifies the supplied implementation; it does not yet generate
    automatic repair attempts or claim behavioral correctness without tests.
    """
    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    async def verify_python_code(self, filepath: Path) -> tuple[bool, str]:
        """Parse Python syntax without executing generated code or spawning a host process."""
        try:
            source = filepath.read_text(encoding="utf-8")
            compile(source, str(filepath), "exec")
            return True, "Syntax OK (compiled without execution)"
        except Exception as exc:  # noqa: BLE001 - verification errors must fail closed
            return False, f"{type(exc).__name__}: {exc!s}"

    async def run_test_script(self, test_filepath: Path) -> tuple[bool, str]:
        """Executes generated tests inside the agent's Docker command sandbox."""
        try:
            relative = Path(test_filepath).resolve().relative_to(self.tools.workspace.resolve()).as_posix()
        except ValueError:
            return False, "Test file is outside the configured workspace; refusing host execution."
        script_path = (
            str(Path(test_filepath).resolve())
            if full_access_enabled()
            else f"/workspace/{relative}"
        )
        result = await self.tools.tool_execute_command(
            f"python -E -s {shlex.quote(script_path)}",
            cwd=".",
            _timeout_override=30.0,
        )
        match = re.search(r"\(Exit (-?\d+)\)", result)
        if match and int(match.group(1)) == 0:
            return True, result
        return False, f"Test failed or was not run in the sandbox:\n{result}"

    async def execute_coding_cycle(
        self,
        task_name: str,
        files_to_create: dict[str, str],
        test_script_content: str | None = None
    ) -> dict[str, Any]:
        """
        Executes a deep coding cycle:
        1. Writes files
        2. Validates syntax
        3. Runs tests
        4. Reports verification status
        """
        results: dict[str, Any] = {
            "task": task_name,
            "created_files": [],
            "syntax_checks": {},
            "test_passed": None,
            "test_output": "",
            "errors": [],
            "status": "in_progress",
        }
        if not isinstance(task_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", task_name):
            results["status"] = "invalid_input"
            results["errors"].append("task_name must be 1-64 letters, digits, underscores, or hyphens and start with a letter or digit.")
            return results
        if not isinstance(files_to_create, dict) or not files_to_create:
            results["status"] = "invalid_input"
            results["errors"].append("files_to_create must be a non-empty object of workspace-relative paths to text.")
            return results
        if any(not isinstance(path, str) or not isinstance(content, str) for path, content in files_to_create.items()):
            results["status"] = "invalid_input"
            results["errors"].append("Every generated file path and content must be a string.")
            return results

        write_failed = False
        syntax_failed = False
        # Step 1: Write all implementation files. A failed write is not a
        # successful coding cycle, even if an unrelated test happens to pass.
        for rel_path, code in files_to_create.items():
            try:
                write_result = self.tools.tool_write_file(rel_path, code)
            except Exception as exc:  # noqa: BLE001 - report tool failures in the cycle
                write_result = f"Error writing file: {type(exc).__name__}: {exc!s}"
            wrote_file = isinstance(write_result, str) and write_result.startswith("Successfully wrote ")
            results["created_files"].append({
                "path": rel_path,
                "status": write_result,
                "written": wrote_file,
            })
            if not wrote_file:
                write_failed = True
                results["errors"].append(f"Could not write {rel_path}: {write_result}")
                continue

            if rel_path.endswith(".py"):
                try:
                    fpath = self.tools._resolve_path(rel_path)
                    ok, msg = await self.verify_python_code(fpath)
                except Exception as exc:  # noqa: BLE001 - path/inspection failures are verification failures
                    ok, msg = False, f"{type(exc).__name__}: {exc!s}"
                results["syntax_checks"][rel_path] = {"valid": ok, "details": msg}
                if not ok:
                    syntax_failed = True

        if write_failed:
            results["status"] = "write_error"
            return results
        if syntax_failed:
            results["status"] = "syntax_error"
            return results

        # Step 2: Run tests only after all writes and Python syntax checks pass.
        if test_script_content:
            try:
                test_path = self.tools._resolve_path(f"test_{task_name}.py")
                test_write = self.tools.tool_write_file(test_path.name, test_script_content)
                if not isinstance(test_write, str) or not test_write.startswith("Successfully wrote "):
                    results["status"] = "test_write_error"
                    results["test_output"] = str(test_write)
                    results["errors"].append(f"Could not write test file: {test_write}")
                    return results
                test_ok, test_out = await self.run_test_script(test_path)
            except Exception as exc:  # noqa: BLE001 - sandbox/setup errors are failed verification
                test_ok, test_out = False, f"Test execution setup failed: {type(exc).__name__}: {exc!s}"
            results["test_passed"] = test_ok
            results["test_output"] = test_out
            results["status"] = "success" if test_ok else "failed_tests"
        else:
            # Parsing proves syntax only; it does not establish behavior.
            results["status"] = "syntax_only"

        return results
