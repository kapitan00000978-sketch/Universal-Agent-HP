import asyncio
import sys
import time

import pytest
from titan_agent.core.code_intel.tdd_engine import AutonomousTDDEngine


async def _trusted_local_pytest(test_path, cwd, timeout):
    """Test-only runner for fixed benign fixtures; subprocess is not a sandbox."""
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", str(test_path), "-v", "-s",
        cwd=str(cwd), stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(errors="replace"), time.monotonic() - started
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "Test-only subprocess timed out.", time.monotonic() - started


@pytest.mark.asyncio
async def test_tdd_engine_full_successful_cycle(tmp_path):
    engine = AutonomousTDDEngine(workspace_root=tmp_path, sandbox_runner=_trusted_local_pytest)

    test_code = """
import pytest
from feature import calculate_discount

def test_calculate_discount():
    assert calculate_discount(100.0, 0.2) == 80.0
    assert calculate_discount(50.0, 0.5) == 25.0
"""

    implementation_code = """
def calculate_discount(price: float, discount: float) -> float:
    return price * (1.0 - discount)
"""

    report = await engine.execute_tdd_cycle(
        test_code=test_code,
        implementation_code=implementation_code,
        test_filename="test_feature.py",
        code_filename="feature.py",
        timeout=15.0,
    )

    assert report.full_cycle_success is True
    assert report.red_stage.success is True  # Failed on initial stub
    assert report.green_stage.success is True  # Passed on implementation
    assert report.refactor_stage.success is True  # Symbolic invariants sound
    assert "PASSED" in report.format_text()


@pytest.mark.asyncio
async def test_tdd_engine_fails_on_broken_implementation(tmp_path):
    engine = AutonomousTDDEngine(workspace_root=tmp_path, sandbox_runner=_trusted_local_pytest)

    test_code = """
def test_multiply():
    from feature import multiply
    assert multiply(3, 4) == 12
"""

    # Flawed implementation
    implementation_code = """
def multiply(a: int, b: int) -> int:
    return a + b  # Bug: returns addition instead of multiplication
"""

    report = await engine.execute_tdd_cycle(
        test_code=test_code,
        implementation_code=implementation_code,
        test_filename="test_feature.py",
        code_filename="feature.py",
        timeout=15.0,
    )

    assert report.full_cycle_success is False
    assert report.red_stage.success is True  # Failed initially
    assert report.green_stage.success is False  # Implementation also failed


@pytest.mark.parametrize(
    ("exit_code", "output", "expected"),
    [
        (1, "FAILED test_feature.py::test_feature - AssertionError", True),
        (2, "ImportError: cannot import name 'feature_api' from 'feature'", True),
        (1, "AttributeError: module 'feature' has no attribute 'feature_api'", True),
        (2, "SyntaxError: invalid syntax", False),
        (2, "ModuleNotFoundError: No module named 'third_party_dep'", False),
        (1, "FAILED test_feature.py::test_feature - ModuleNotFoundError: No module named 'third_party_dep'", False),
        (1, "pytest: error: unrecognized arguments: --bad", False),
        (124, "Timed out while running pytest", False),
        (125, "Docker runtime failed", False),
        (126, "Sandbox runner unavailable", False),
        (0, "2 passed", False),
    ],
)
def test_red_requires_expected_test_failure(exit_code, output, expected):
    assert AutonomousTDDEngine._red_failure_is_expected(
        exit_code, output, "feature"
    ) is expected


@pytest.mark.asyncio
async def test_tdd_engine_does_not_run_green_after_invalid_red(tmp_path):
    calls = []

    async def malformed_test_runner(test_path, cwd, timeout):
        calls.append(test_path)
        return 2, "ERROR collecting test_feature.py: SyntaxError: invalid syntax", 0.01

    engine = AutonomousTDDEngine(workspace_root=tmp_path, sandbox_runner=malformed_test_runner)
    report = await engine.execute_tdd_cycle(
        test_code="def test_example(: pass",
        implementation_code="def example(): return True",
        test_filename="test_feature.py",
        code_filename="feature.py",
        timeout=1.0,
    )

    assert report.red_stage.success is False
    assert report.green_stage.success is False
    assert "Not run" in report.green_stage.output
    assert len(calls) == 1
    assert report.full_cycle_success is False


@pytest.mark.asyncio
async def test_tdd_engine_sandbox_runner_exception_fails_closed(tmp_path):
    async def broken_runner(test_path, cwd, timeout):
        raise RuntimeError("container runtime unavailable")

    engine = AutonomousTDDEngine(workspace_root=tmp_path, sandbox_runner=broken_runner)
    report = await engine.execute_tdd_cycle(
        test_code="def test_example(): assert True",
        implementation_code="def example(): return True",
        timeout=1.0,
    )

    assert report.red_stage.success is False
    assert report.red_stage.exit_code == 125
    assert report.green_stage.success is False
    assert report.green_stage.exit_code == 126


@pytest.mark.asyncio
async def test_tdd_engine_without_sandbox_fails_closed(tmp_path):
    engine = AutonomousTDDEngine(workspace_root=tmp_path)
    report = await engine.execute_tdd_cycle(
        test_code="def test_example(): assert True",
        implementation_code="def example(): return True",
        timeout=1.0,
    )
    assert report.full_cycle_success is False
    assert report.red_stage.success is False
    assert "not executed" in report.red_stage.output
    assert report.green_stage.success is False


@pytest.mark.asyncio
async def test_tdd_engine_fails_on_symbolic_invariant_hazard(tmp_path):
    engine = AutonomousTDDEngine(workspace_root=tmp_path, sandbox_runner=_trusted_local_pytest)

    test_code = """
def test_dangerous():
    from feature import do_danger
    assert do_danger("ls") == 0
"""

    # Has shell injection hazard
    implementation_code = """
import subprocess
def do_danger(arg):
    return subprocess.run(f"echo {arg}", shell=True).returncode
"""

    report = await engine.execute_tdd_cycle(
        test_code=test_code,
        implementation_code=implementation_code,
        test_filename="test_feature.py",
        code_filename="feature.py",
        timeout=15.0,
    )

    # Even if pytest passed or failed, refactor phase must detect symbolic violation
    assert report.refactor_stage.success is False
    assert report.full_cycle_success is False
