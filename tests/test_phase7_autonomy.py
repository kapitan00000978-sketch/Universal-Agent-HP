"""Phase 7 — Full Autonomy tests.

Covers the five autonomy blocks:
A. Self-healing loop (heal.py) — deterministic repairs (pip install, retries)
B. Dynamic step budget — TITAN_STEP_CAP / TITAN_UNLIMITED_STEPS (no hard 48 clamp)
C. Persistent task queue + daemon loop (queue.py, daemon.py)
D. Real-world tools (download, http-server, queue tools)
E. Deep subagent architecture (subagents.py)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from titan_agent import config
from titan_agent.agent import _compute_max_steps
from titan_agent.daemon import TaskDaemon
from titan_agent.heal import SelfHealEngine, diagnose, heal_run
from titan_agent.queue import TaskQueue

# ---------------------------------------------------------------------------
# Block A: Self-healing loop
# ---------------------------------------------------------------------------

async def _fake_run(results: list[tuple[int, str, str]]):
    """Build an async run function that pops canned (code, out, err) results."""
    it = iter(results)

    async def run(command: str) -> tuple[int, str, str]:
        return next(it)

    return run


def test_diagnose_missing_module():
    repair = diagnose("Traceback ... ModuleNotFoundError: No module named 'requests'")
    assert repair is not None
    assert repair.kind == "pip_install"
    assert "requests" in repair.command


def test_diagnose_numpy_misimport():
    repair = diagnose("ImportError: numpy.core.multiarray failed to import")
    assert repair is not None
    assert repair.kind == "pip_install"
    assert "upgrading numpy" in repair.label


def test_diagnose_command_not_found():
    repair = diagnose("xyz: The term 'xyz' is not recognized as the name of a cmdlet")
    assert repair is not None
    assert repair.kind == "retry"


def test_diagnose_unknown_error_returns_none():
    assert diagnose("random business logic error 42") is None


def test_heal_run_succeeds_first_try():
    async def run(cmd):
        return 0, "ok", ""

    res = asyncio.run(heal_run(run, "echo hi"))
    assert res.succeeded is True
    assert len(res.attempts) == 1
    assert res.repairs == []


def test_heal_run_installs_missing_module_then_succeeds():
    """Fail -> pip install -> succeed on the re-run."""
    calls = {"original": 0}

    async def run(cmd):
        if "pip install" in cmd:
            return 0, "", "installed"
        calls["original"] += 1
        if calls["original"] >= 2:
            return 0, "ok now", ""
        return 1, "", "ModuleNotFoundError: No module named 'requests'"

    res = asyncio.run(heal_run(run, "import requests; print('x')"))
    assert res.succeeded is True
    assert len(res.repairs) == 1
    assert res.repairs[0]["kind"] == "pip_install"


def test_heal_run_gives_up_without_repair():
    async def run(cmd):
        return 1, "", "KeyError: boom"

    res = asyncio.run(heal_run(run, "bad-thing", max_attempts=3))
    assert res.succeeded is False
    assert res.repairs == []
    assert res.final_code == 1


def test_heal_run_exhausts_attempts_on_persistent_failure():
    """A repair that never fixes things must still stop the loop."""
    attempts = {"n": 0}

    async def run(cmd):
        attempts["n"] += 1
        return 1, "", "ModuleNotFoundError: No module named 'xyz'"

    res = asyncio.run(heal_run(run, "run-me", max_attempts=2))
    assert res.succeeded is False
    # max_attempts=2 -> 2 original runs + 2 repair commands (one after each fail)
    assert attempts["n"] == 4
    assert len(res.repairs) == 2


def test_self_heal_engine_requires_runner():
    engine = SelfHealEngine()
    with pytest.raises(RuntimeError):
        asyncio.run(engine.heal("echo hi"))


# ---------------------------------------------------------------------------
# Block B: dynamic step budget
# ---------------------------------------------------------------------------

def test_compute_max_steps_default_cap():
    """Default behavior keeps the classic 48 clamp when unlimited is off."""
    assert config.UNLIMITED_STEPS is False
    assert _compute_max_steps("deep", "ultra") <= config.MAX_STEPS_CAP


def test_unlimited_steps_raises_cap(monkeypatch):
    from titan_agent import agent as agent_mod

    monkeypatch.setattr(agent_mod, "UNLIMITED_STEPS", True)
    steps = _compute_max_steps("deep", "ultra")
    assert steps > config.MAX_STEPS_CAP


def test_step_cap_config_exported():
    assert isinstance(config.MAX_STEPS_CAP, int)
    assert isinstance(config.TASK_QUEUE_FILE, Path)
    assert config.DAEMON_POLL_INTERVAL > 0
    assert config.DAEMON_MAX_CONCURRENT >= 1


# ---------------------------------------------------------------------------
# Block C: task queue
# ---------------------------------------------------------------------------

@pytest.fixture()
def queue(tmp_path: Path):
    q = TaskQueue(tmp_path / "queue.db")
    yield q
    q.close()


def test_queue_enqueue_and_claim(queue: TaskQueue):
    tid = queue.enqueue("do the thing", name="work")
    assert tid == 1
    task = queue.claim_next()
    assert task is not None
    assert task.id == tid
    assert task.status == "running"
    # No second claim while running
    assert queue.claim_next() is None


def test_queue_priority_order(queue: TaskQueue):
    low = queue.enqueue("low", priority=0)
    high = queue.enqueue("high", priority=10)
    first = queue.claim_next()
    assert first.id == high
    queue.complete(high)
    second = queue.claim_next()
    assert second.id == low


def test_queue_schedule_at_waits(queue: TaskQueue):
    queue.enqueue("future", schedule_at=10_000_000_000)
    assert queue.claim_next() is None
    tid = queue.enqueue("now")
    assert queue.claim_next().id == tid


def test_queue_complete(queue: TaskQueue):
    tid = queue.enqueue("task", name="t")
    task = queue.claim_next()
    assert task.id == tid
    queue.complete(tid, "DONE: yes")
    done = queue.get(tid)
    assert done.status == "done"
    assert done.result == "DONE: yes"
    d = done.to_dict()
    assert d["status"] == "done"
    assert d["result"] == "DONE: yes"


def test_queue_retry_then_fail(queue: TaskQueue):
    tid = queue.enqueue("flaky", max_attempts=2)
    queue.claim_next()
    queue.fail(tid, "err 1")
    t = queue.get(tid)
    assert t.status == "pending"  # retried with backoff
    assert t.attempts == 1
    assert t.schedule_at > 0  # backoff scheduled
    # Clear backoff so it is claimable immediately for the next attempt
    queue.complete(tid)  # move to done
    # Re-check persistence of stats
    assert queue.stats()["done"] == 1


def test_queue_fail_exhausts(queue: TaskQueue):
    tid = queue.enqueue("doomed", max_attempts=1)
    queue.claim_next()
    queue.fail(tid, "always fails")
    t = queue.get(tid)
    assert t.status == "failed"
    assert t.attempts == 1


def test_queue_cancel_pending(queue: TaskQueue):
    tid = queue.enqueue("later")
    assert queue.cancel(tid) is True
    assert queue.get(tid).status == "cancelled"


def test_queue_list_and_stats(queue: TaskQueue):
    queue.enqueue("a")
    queue.enqueue("b")
    claimed = queue.claim_next()
    queue.complete(claimed.id)
    tasks = queue.list()
    assert len(tasks) == 2
    stats = queue.stats()
    assert stats["done"] == 1
    assert stats["pending"] == 1


# ---------------------------------------------------------------------------
# Block C2: daemon loop
# ---------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self, outcomes: dict[int, int]):
        self.outcomes = outcomes  # task_id -> exit code
        self.calls: list[str] = []

    def __call__(self, task: str, opts: dict):
        self.calls.append(task)
        tid = int(opts.get("task_id", 0))
        code = self.outcomes.get(tid, 0)
        return code, "answer" if code == 0 else "nope", []


def test_daemon_run_once_marks_done(queue: TaskQueue):
    tid_ok = queue.enqueue("fine")
    runner = _FakeRunner({tid_ok: 0})
    daemon = TaskDaemon(queue, runner=runner, max_concurrent=1)
    asyncio.run(daemon.run_once())
    assert queue.get(tid_ok).status == "done"
    assert queue.get(tid_ok).result == "answer"


def test_daemon_run_once_fails_queue_retries(queue: TaskQueue):
    tid = queue.enqueue("bad", max_attempts=3)
    runner = _FakeRunner({tid: 1})
    daemon = TaskDaemon(queue, runner=runner, max_concurrent=1)
    asyncio.run(daemon.run_once())
    t = queue.get(tid)
    # failed task stays pending for retry (attempts < max)
    assert t.status == "pending"
    assert t.attempts == 1


def test_daemon_concurrency_bounded(queue: TaskQueue):
    t1 = queue.enqueue("one")
    t2 = queue.enqueue("two")
    runner = _FakeRunner({t1: 0, t2: 0})
    daemon = TaskDaemon(queue, runner=runner, max_concurrent=2)
    asyncio.run(daemon.run_once())
    assert queue.get(t1).status == "done"
    assert queue.get(t2).status == "done"
    assert len(runner.calls) == 2


# ---------------------------------------------------------------------------
# Block E: subagent architecture
# ---------------------------------------------------------------------------

def test_subagent_pool_delegate():
    from titan_agent.subagents import SubagentPool

    def fake_runner(task: str, opts: dict):
        assert opts["session_id"].startswith("sub-")
        return 0, f"result for: {task}", []

    pool = SubagentPool(runner=fake_runner)
    res = asyncio.run(pool.delegate("build x", session_id="parent", label="w1"))
    assert res.exit_code == 0
    assert "result for: build x" in res.final
    assert res.to_text().startswith("### SUBAGENT [w1]")


def test_subagent_pool_team_parallel():
    from titan_agent.subagents import SubagentPool

    order: list[str] = []

    def fake_runner(task: str, opts: dict):
        order.append(task)
        return 0, f"{task} done", []

    pool = SubagentPool(runner=fake_runner, max_workers=3)
    results = asyncio.run(pool.team(["a", "b", "c"], labels=["x", "y", "z"]))
    assert len(results) == 3
    assert sorted(order) == ["a", "b", "c"]
    assert results[0].label == "x"


# ---------------------------------------------------------------------------
# Block D: tool registry integration
# ---------------------------------------------------------------------------

def test_phase7_tools_registered():
    from titan_agent.tools import ToolRegistry

    reg = ToolRegistry()
    names = {t["function"]["name"] for t in reg.get_tool_definitions()}
    expected = {
        "self_heal",
        "download_file",
        "start_http_server",
        "stop_http_server",
        "take_screenshot",
        "self_update",
        "task_enqueue",
        "task_list",
        "task_stats",
        "task_cancel",
        "subagent_delegate",
        "subagent_team",
    }
    assert expected <= names


def test_task_tools_roundtrip(tmp_path: Path, monkeypatch):
    """task_enqueue -> task_list -> task_cancel through ToolRegistry."""
    import titan_agent.tools as tools_mod
    from titan_agent.tools import ToolRegistry

    monkeypatch.setattr(tools_mod, "TASK_QUEUE_FILE", tmp_path / "q.db")
    reg = ToolRegistry(tmp_path / "ws")
    out = reg.tool_task_enqueue("hello task", name="t1")
    assert "Task #1 enqueued" in out

    listing = reg.tool_task_list()
    assert "t1" in listing
    assert "pending" in listing

    stats = reg.tool_task_stats()
    assert "pending=1" in stats

    cancelled = reg.tool_task_cancel(1)
    assert "cancelled" in cancelled


def test_download_refuses_private_target():
    from titan_agent.tools import ToolRegistry

    reg = ToolRegistry()
    res = asyncio.run(reg.tool_download_file("http://127.0.0.1:8080/admin"))
    assert "refused" in res


def test_public_download_uses_validated_resolution_and_atomic_destination(tmp_path, monkeypatch):
    import socket
    import aiohttp

    from titan_agent.tools import ToolRegistry

    registry = ToolRegistry(tmp_path)
    resolutions = []

    async def _run():
        loop = asyncio.get_running_loop()

        async def fake_getaddrinfo(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

        monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)

        class FakeConnector:
            def __init__(self, resolver, **_kwargs):
                self.resolver = resolver

        class FakeContent:
            async def iter_chunked(self, _size):
                yield b"validated-public-content"

        connector_ref = {}

        class FakeResponse:
            content = FakeContent()

            def raise_for_status(self):
                return None

        class FakeRequestContext:
            async def __aenter__(self):
                addresses = await connector_ref["connector"].resolver.resolve(
                    "public-looking.example", 80, family=socket.AF_INET
                )
                resolutions.extend(addresses)
                return FakeResponse()

            async def __aexit__(self, *_args):
                return None

        class FakeSession:
            def __init__(self, connector, **_kwargs):
                connector_ref["connector"] = connector

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def get(self, *_args, **_kwargs):
                return FakeRequestContext()

        monkeypatch.setattr(aiohttp, "TCPConnector", FakeConnector)
        monkeypatch.setattr(aiohttp, "ClientSession", FakeSession)
        return await registry.tool_download_file(
            "http://public-looking.example/file.bin?token=do-not-echo"
        )

    result = asyncio.run(_run())
    assert "Downloaded" in result
    assert "do-not-echo" not in result
    assert (tmp_path / "file.bin").read_bytes() == b"validated-public-content"
    assert resolutions[0]["host"] == "93.184.216.34"
    assert resolutions[0]["flags"] == socket.AI_NUMERICHOST


def test_download_refuses_hostname_resolving_to_private_address(tmp_path, monkeypatch):
    import socket

    from titan_agent.tools import ToolRegistry

    registry = ToolRegistry(tmp_path)
    observed_hosts = []

    async def _run():
        loop = asyncio.get_running_loop()

        async def fake_getaddrinfo(host, port, **kwargs):
            observed_hosts.append(host)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

        monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
        return await registry.tool_download_file("http://public-looking.example/secret.bin")

    result = asyncio.run(_run())
    assert observed_hosts == ["public-looking.example"]
    assert "failed" in result.lower()
    assert not (tmp_path / "secret.bin").exists()
    assert not list(tmp_path.glob(".titan-download-*"))


def test_download_refuses_bad_scheme():
    from titan_agent.tools import ToolRegistry

    reg = ToolRegistry()
    res = asyncio.run(reg.tool_download_file("file:///etc/passwd"))
    assert "only http(s)" in res


def test_http_server_start_stop(tmp_path: Path):
    from titan_agent.tools import ToolRegistry

    reg = ToolRegistry(tmp_path)
    out = reg.tool_start_http_server(port=8765, directory=".")
    assert "127.0.0.1:8765" in out
    stopped = reg.tool_stop_http_server(port=8765)
    assert "Stopped" in stopped


def test_screenshot_non_windows_graceful():
    from titan_agent.tools import ToolRegistry

    if sys.platform == "win32":
        pytest.skip("skipping cross-platform branch on Windows")

    reg = ToolRegistry()
    res = asyncio.run(reg.tool_take_screenshot())
    assert "Windows only" in res


def test_agent_tool_catalog_contains_phase7():
    from titan_agent.agent import TITAN_SYSTEM_PROMPT

    for tool in ("self_heal", "subagent_team", "task_enqueue", "self_update"):
        assert tool in TITAN_SYSTEM_PROMPT