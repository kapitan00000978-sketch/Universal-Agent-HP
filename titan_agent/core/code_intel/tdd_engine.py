"""
Phase 46 — Autonomous TDD Engine (Red-Green-Refactor Loop).

Guarantees high-integrity software engineering by enforcing strict Test-Driven Development:
1. RED: Writes a test first and executes it in isolation to prove it fails (asserting the requirement/bug).
2. GREEN: Writes the minimal implementation code and proves the test now passes.
3. REFACTOR & SYMBOLIC: Statically checks code invariants (no infinite loops, no shell injections)
   and verifies clean code structure before finalizing.
"""
from __future__ import annotations

import asyncio
import logging
import re
import tempfile
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .symbolic_checker import SymbolicInvariantChecker

log = logging.getLogger(__name__)


@dataclass
class TDDStageResult:
    """Outcome of one TDD phase (RED, GREEN, or REFACTOR)."""

    stage: str  # "RED", "GREEN", "REFACTOR"
    success: bool
    output: str
    exit_code: int = 0
    duration_sec: float = 0.0


@dataclass
class TDDExecutionReport:
    """Full lifecycle report of an autonomous Red-Green-Refactor cycle."""

    test_name: str
    red_stage: TDDStageResult
    green_stage: TDDStageResult
    refactor_stage: TDDStageResult
    full_cycle_success: bool
    summary: str = ""

    def format_text(self) -> str:
        status_icon = "✅ PASSED" if self.full_cycle_success else "❌ FAILED"
        lines = [
            f"### AUTONOMOUS TDD CYCLE: {status_icon} [{self.test_name}]",
            f"1. **RED Phase (Failing Test)**: {'✓ Confirmed Failed' if self.red_stage.success else '✗ Unexpectedly Passed/Errored'}",
            f"   - Exit Code: {self.red_stage.exit_code} | Duration: {self.red_stage.duration_sec:.2f}s",
            f"2. **GREEN Phase (Implementation Fix)**: {'✓ Successfully Passed' if self.green_stage.success else '✗ Still Failing'}",
            f"   - Exit Code: {self.green_stage.exit_code} | Duration: {self.green_stage.duration_sec:.2f}s",
            f"3. **REFACTOR Phase (Symbolic Invariant)**: {'✓ Invariants Sound' if self.refactor_stage.success else '✗ Violations Detected'}",
            f"   - Invariant Status:\n{self.refactor_stage.output.strip()}",
        ]
        return "\n".join(lines)


class AutonomousTDDEngine:
    """Executes rigorous Red-Green-Refactor cycles in hermetic temp sandboxes."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        sandbox_runner: Callable[[Path, Path, float], Awaitable[tuple[int, str, float]]] | None = None,
    ):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.sandbox_runner = sandbox_runner

    async def _run_pytest_in_dir(self, test_path: Path, cwd: Path, timeout: float = 30.0) -> tuple[int, str, float]:
        """Run generated tests in the configured OS/container sandbox only."""
        if self.sandbox_runner is None:
            return 126, "Docker sandbox runner is not configured; generated tests were not executed.", 0.0
        started = asyncio.get_running_loop().time()
        try:
            return await self.sandbox_runner(test_path, cwd, timeout)
        except Exception as exc:
            duration = asyncio.get_running_loop().time() - started
            log.warning("TDD sandbox runner failed: %s", exc)
            return 125, f"TDD sandbox runner failed: {exc!s}", duration

    @staticmethod
    def _red_failure_is_expected(exit_code: int, output: str, module_name: str) -> bool:
        """Count RED only when a test failed or collection found the missing API.

        Syntax errors, missing third-party dependencies, timeouts, sandbox
        failures, and truncated output are not evidence that the feature test
        correctly detects the requested bug.
        """
        if exit_code in {0, 124, 125, 126}:
            return False
        missing_api = re.search(r"cannot import name '([^']+)' from '([^']+)'", output)
        if missing_api is None:
            missing_api = re.search(r'cannot import name "([^"]+)" from "([^"]+)"', output)
        if missing_api and missing_api.group(2) == module_name:
            return True
        missing_attribute = re.search(r"module '([^']+)' has no attribute '([^']+)'", output)
        if missing_attribute is None:
            missing_attribute = re.search(r'module "([^"]+)" has no attribute "([^"]+)"', output)
        if missing_attribute and missing_attribute.group(1) == module_name:
            return True
        # A failed test is not a meaningful RED result if collection/runtime
        # broke for unrelated reasons, such as a dependency or malformed test.
        if re.search(r"ModuleNotFoundError|No module named|SyntaxError|ERROR collecting|pytest: error:", output):
            return False
        return bool(re.search(r"(?m)^FAILED\s+[^\s:]+::", output))

    async def execute_tdd_cycle(
        self,
        test_code: str,
        implementation_code: str,
        test_filename: str = "test_feature.py",
        code_filename: str = "feature.py",
        timeout: float = 30.0,
    ) -> TDDExecutionReport:
        """Execute RED/GREEN tests through the configured OS/container sandbox."""
        for filename in (test_filename, code_filename):
            if not filename or Path(filename).name != filename or filename in {".", ".."}:
                raise ValueError("TDD filenames must be simple basenames inside the temporary workspace")
        with tempfile.TemporaryDirectory(prefix="titan_tdd_") as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / test_filename
            code_file = tmppath / code_filename

            # -------------------------------------------------------------
            # Stage 1: RED PHASE (Test written, implementation absent/dummy)
            # -------------------------------------------------------------
            # Write dummy implementation or empty stub
            code_file.write_text("# Initial empty stub\n", encoding="utf-8")
            test_file.write_text(test_code, encoding="utf-8")

            red_code, red_out, red_dur = await self._run_pytest_in_dir(test_file, tmppath, timeout=timeout)
            # RED is valid only when pytest reports a failed test or an import/
            # attribute failure for the intentionally empty feature module.
            red_success = self._red_failure_is_expected(
                red_code, red_out, Path(code_filename).stem
            )
            red_stage = TDDStageResult(
                stage="RED",
                success=red_success,
                output=red_out,
                exit_code=red_code,
                duration_sec=red_dur,
            )

            # -------------------------------------------------------------
            # Stage 2: GREEN PHASE (Implementation written, test must pass)
            # -------------------------------------------------------------
            code_file.write_text(implementation_code, encoding="utf-8")
            if not red_success:
                green_code = 126
                green_out = "Not run because RED did not demonstrate a valid failing test."
                green_dur = 0.0
            else:
                green_code, green_out, green_dur = await self._run_pytest_in_dir(test_file, tmppath, timeout=timeout)
            green_success = green_code == 0
            green_stage = TDDStageResult(
                stage="GREEN",
                success=green_success,
                output=green_out,
                exit_code=green_code,
                duration_sec=green_dur,
            )

            # -------------------------------------------------------------
            # Stage 3: REFACTOR PHASE (Symbolic AST Invariant Verification)
            # -------------------------------------------------------------
            symbolic_report = SymbolicInvariantChecker.check_code(implementation_code)
            refactor_success = symbolic_report.is_safe
            refactor_stage = TDDStageResult(
                stage="REFACTOR",
                success=refactor_success,
                output=symbolic_report.summary(),
                exit_code=0 if refactor_success else 1,
                duration_sec=0.01,
            )

            full_success = red_success and green_success and refactor_success
            report = TDDExecutionReport(
                test_name=test_filename,
                red_stage=red_stage,
                green_stage=green_stage,
                refactor_stage=refactor_stage,
                full_cycle_success=full_success,
            )
            report.summary = report.format_text()
            return report
