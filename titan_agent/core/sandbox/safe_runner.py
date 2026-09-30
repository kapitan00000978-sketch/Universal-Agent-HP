"""Legacy host-subprocess helper for trusted tests and internal utilities.

This module provides interpreter/process separation, timeouts, static pattern
checks, and workspace rollback. It is NOT an OS/container security sandbox:
code retains the host process user's privileges. Agent-facing code execution
must use ToolRegistry's Docker-backed command path instead.
"""
from __future__ import annotations

import asyncio
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .environment import SandboxEnvironment


@dataclass
class SandboxResult:
    """Outcome of a process-isolated execution, not a host sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    rolled_back: bool = False
    error: str | None = None
    language: str = "python"

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def duration_sec(self) -> float:
        return round(self.duration_ms / 1000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "duration_sec": self.duration_sec,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "language": self.language,
        }


class SafeScriptRunner:
    """Runs code in a host subprocess; not an OS/container isolation boundary."""

    DANGEROUS_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:", re.IGNORECASE),  # bash fork bomb
        re.compile(r"os\.system\s*\(\s*['\"].*?(?:rm\s+-rf\s+/[*]?|format\s+[a-z]:).*?['\"]\)", re.IGNORECASE),
        re.compile(r"shutil\.rmtree\s*\(\s*['\"](?:/[*]?|[a-zA-Z]:\\?)['\"]\)", re.IGNORECASE),
        re.compile(r"format\s+[a-z]:\s+/fs", re.IGNORECASE),  # Windows drive format
    ]

    def __init__(
        self,
        workspace_or_sandbox: Any = None,
        sandbox_env: SandboxEnvironment | None = None,
    ):
        if isinstance(workspace_or_sandbox, SandboxEnvironment):
            self.sandbox_env = workspace_or_sandbox
            self.workspace_root = workspace_or_sandbox.workspace_root
        elif workspace_or_sandbox is not None:
            self.workspace_root = Path(workspace_or_sandbox).resolve()
            self.sandbox_env = sandbox_env or SandboxEnvironment(self.workspace_root)
        elif sandbox_env is not None:
            self.sandbox_env = sandbox_env
            self.workspace_root = sandbox_env.workspace_root
        else:
            self.workspace_root = Path.cwd()
            self.sandbox_env = SandboxEnvironment(self.workspace_root)

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """Scans code for known destructive patterns."""
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(code):
                return False, f"Dangerous command pattern detected matching {pattern.pattern}"
        return True, ""

    def run(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        cwd: Path | str | None = None,
        auto_rollback: bool = False,
        rollback_on_failure: bool = False,
    ) -> SandboxResult:
        """Synchronous runner helper for running code in a sandbox."""
        rb = auto_rollback or rollback_on_failure
        coro = self.execute(
            code=code,
            language=language,
            timeout=timeout,
            cwd=cwd,
            rollback_on_failure=rb,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_container: list[SandboxResult] = []
            exception_container: list[BaseException] = []

            def _thread_target():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    res = new_loop.run_until_complete(coro)
                    result_container.append(res)
                    new_loop.close()
                except BaseException as e:  # noqa: BLE001
                    exception_container.append(e)

            t = threading.Thread(target=_thread_target)
            t.start()
            t.join(timeout=timeout + 5.0)
            if exception_container:
                raise exception_container[0]
            if result_container:
                return result_container[0]
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="Thread execution timed out",
                duration_ms=timeout * 1000,
                error="Thread timeout",
                language=language,
            )
        else:
            return asyncio.run(coro)

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        cwd: Path | str | None = None,
        rollback_on_failure: bool = False,
    ) -> SandboxResult:
        """
        Executes code in an isolated subprocess.
        If rollback_on_failure is enabled, restores workspace if execution fails.
        """
        lang = str(language).lower().strip()
        t = max(0.5, min(float(timeout or 30.0), 300.0))

        # 1. Validate safety
        is_safe, reason = self.validate_code_safety(code)
        if not is_safe:
            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr="",
                duration_ms=0.0,
                rolled_back=False,
                error=f"Security alert: Execution blocked by sandbox safety guard: {reason}",
                language=lang,
            )

        # 2. Snapshot if rollback requested
        snap_name = ""
        if rollback_on_failure and self.sandbox_env:
            snap_name = f"auto_pre_exec_{int(time.time() * 1000)}"
            self.sandbox_env.create_snapshot(snap_name)

        # 3. Prepare execution directory and script file
        work_dir = Path(cwd) if cwd else self.workspace_root
        work_dir.mkdir(parents=True, exist_ok=True)

        script_file = work_dir / f"__sandbox_run_{int(time.time() * 1000)}.py"
        script_file.write_text(code, encoding="utf-8")

        start_time = time.monotonic()
        try:
            if lang == "python":
                cmd = [sys.executable, str(script_file)]
            else:
                cmd = ["python", str(script_file)]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=t)
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 1)

            out_str = stdout_b.decode("utf-8", errors="ignore").strip()
            err_str = stderr_b.decode("utf-8", errors="ignore").strip()
            exit_code = proc.returncode if proc.returncode is not None else 0

            # Clean up the temp script file if not rolled back
            script_file.unlink(missing_ok=True)

            rolled_back = False
            if exit_code != 0 and rollback_on_failure and self.sandbox_env and snap_name:
                self.sandbox_env.rollback(snap_name)
                rolled_back = True

            return SandboxResult(
                exit_code=exit_code,
                stdout=out_str,
                stderr=err_str,
                duration_ms=duration_ms,
                rolled_back=rolled_back,
                error=None if exit_code == 0 else f"Process exited with status {exit_code}",
                language=lang,
            )

        except asyncio.TimeoutError:
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 1)
            try:
                proc.kill()
                await proc.wait()
            except Exception:  # noqa: BLE001, S110
                pass
            script_file.unlink(missing_ok=True)

            rolled_back = False
            if rollback_on_failure and self.sandbox_env and snap_name:
                self.sandbox_env.rollback(snap_name)
                rolled_back = True

            return SandboxResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {t:.1f} seconds",
                duration_ms=duration_ms,
                rolled_back=rolled_back,
                error=f"Timeout after {t:.1f}s",
                language=lang,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 1)
            script_file.unlink(missing_ok=True)
            rolled_back = False
            if rollback_on_failure and self.sandbox_env and snap_name:
                self.sandbox_env.rollback(snap_name)
                rolled_back = True

            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                duration_ms=duration_ms,
                rolled_back=rolled_back,
                error=f"Subprocess failure: {exc!s}",
                language=lang,
            )
