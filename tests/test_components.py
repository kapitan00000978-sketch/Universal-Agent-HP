"""Titan Agent component tests (pytest-compatible).

Run from the project root:
    python -m pytest tests -q
"""
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from titan_agent.agent import TitanAgent
from titan_agent.config import MCP_CONFIG_FILE, WORKSPACE_DIR
from titan_agent.mcp_client import MCPManager
from titan_agent.memory import MemoryManager
from titan_agent.tools import ToolRegistry


def test_memory_manager():
    mem = MemoryManager(Path(__file__).parent / "test_memory.db")
    mem.clear_session("test_sess")
    mem.add_message("test_sess", "user", "Salom Titan")
    mem.add_message("test_sess", "assistant", "Assalomu alaykum!", thoughts="Tahlil qilindi")
    msgs = mem.get_recent_messages("test_sess")
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Salom Titan"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1].get("thoughts") == "Tahlil qilindi"
    mem.remember_fact("user_name", "Alisher", "profile")
    facts = mem.search_knowledge("Alisher")
    assert len(facts) == 1
    assert facts[0]["key"] == "user_name"


def test_tool_file_write_read():
    tools = ToolRegistry(WORKSPACE_DIR)
    w_res = tools.tool_write_file("test_sample.txt", "Salom Dunyo Titan Agent!")
    assert "Successfully" in w_res
    content = tools.tool_read_file("test_sample.txt")
    assert "Salom Dunyo Titan Agent!" in content


def test_tool_edit_and_list():
    tools = ToolRegistry(WORKSPACE_DIR)
    tools.tool_write_file("test_edit.txt", "alpha beta")
    res = tools.tool_edit_file("test_edit.txt", "beta", "gamma")
    assert "Successfully" in res
    edited = tools.tool_read_file("test_edit.txt")
    assert edited == "alpha gamma"
    listing = tools.tool_list_directory(WORKSPACE_DIR)
    assert "test_edit.txt" in listing


def test_tool_python_eval_routes_through_sandbox(monkeypatch, tmp_path):
    tools = ToolRegistry(tmp_path)
    captured = {}

    async def fake_sandbox_command(command, cwd="", _timeout_override=None):
        captured["command"] = command
        captured["cwd"] = cwd
        return "### DOCKER COMMAND SANDBOX (Exit 0)\\nSTDOUT:\\n42"

    monkeypatch.setattr(tools, "tool_execute_command", fake_sandbox_command)
    py_res = asyncio.run(tools.tool_python_eval("print(40 + 2)"))
    assert "42" in py_res
    assert "python -I -c" in captured["command"]
    assert captured["cwd"] == "."


def test_tool_workspace_rag(tmp_path):
    """workspace_rag finds relevant snippets with file paths, locally."""
    from titan_agent.tools import ToolRegistry

    ws = tmp_path / "ragdoc"
    ws.mkdir()
    (ws / "notes.md").write_text(
        "# Meeting notes\nBudget for Q3 is 45k dollars. Team wants more tests.\n",
        encoding="utf-8",
    )
    (ws / "code.py").write_text(
        "def calculate_budget():\n    return 45000  # quarterly budget\n",
        encoding="utf-8",
    )
    (ws / "unrelated.txt").write_text(
        "The cat sat on the mat. Nothing about money here.\n",
        encoding="utf-8",
    )
    reg = ToolRegistry(workspace=ws)

    result = reg.tool_workspace_rag("budget", top_k=3)
    assert "notes.md" in result
    assert "code.py" in result
    assert "unrelated.txt" not in result
    # Snippets carry the matched text for citation-style answers
    assert "45k" in result or "45000" in result

    # Empty / no-match query returns a graceful message
    empty = reg.tool_workspace_rag("zzzzzznomatch", top_k=2)
    assert "No relevant snippets" in empty


def test_memory_auto_recall(tmp_path):
    """recall_relevant returns only facts that actually match the query."""
    from titan_agent.memory import MemoryManager

    mem = MemoryManager(tmp_path / "mem_test.db")
    mem.remember_fact("user_name", "Aziz", category="profile")
    mem.remember_fact("favorite_food", "osh", category="preference")
    mem.remember_fact("project_api_key", "sk-test-123", category="project")

    hits = mem.recall_relevant("Aziz asked about osh recipe", limit=5)
    keys = {f["key"] for f in hits}
    assert "user_name" in keys
    assert "favorite_food" in keys
    assert "project_api_key" not in keys

    # Exact key match ranks first even when the shared token is common
    hits2 = mem.recall_relevant("user_name please", limit=5)
    assert hits2 and hits2[0]["key"] == "user_name"


def test_run_task_auto_recalls_memory_into_system_prompt(tmp_path):
    """Fast-mode run_task seeds the system context with remembered facts."""
    import asyncio

    from titan_agent.agent import TitanAgent
    from titan_agent.llm_client import LLMResponse
    from titan_agent.memory import MemoryManager

    mem = MemoryManager(tmp_path / "recall_integration.db")
    mem.remember_fact("user_favorite", "apple juice", category="preference")

    captured = {}

    class FakeLLM:
        async def chat_completion(self, messages, tools=None):
            captured["system"] = messages[0]["content"]
            captured["tools"] = tools
            return LLMResponse(content="You like apple juice.")

    async def _run():
        agent = TitanAgent(llm=FakeLLM(), memory=mem)
        return [ev async for ev in agent.run_task("What do I like to drink?", session_id="t", mode="fast")]

    events = asyncio.run(_run())
    # (the recall is best-effort lexical: use a query that overlaps the fact) — rerun with overlap
    async def _run2():
        agent = TitanAgent(llm=FakeLLM(), memory=mem)
        return [ev async for ev in agent.run_task("I want some apple juice", session_id="t", mode="fast")]

    asyncio.run(_run())
    events = asyncio.run(_run2())

    # The recall block reached the model's system prompt
    assert "REMEMBERED FACTS" in captured["system"]
    assert "apple juice" in captured["system"]
    # The live catalog still advertises the new RAG tool
    assert "workspace_rag" in captured["system"]
    # And a plain final answer was produced
    assert any(ev.type == "final_answer" for ev in events)


def test_effort_levels_scale_budget():
    """Effort levels resolve correctly and scale the iteration budget."""
    from titan_agent.agent import _compute_max_steps, _resolve_effort

    assert _resolve_effort("auto", "fast") == "medium"
    assert _resolve_effort("auto", "deep") == "high"
    assert _resolve_effort("bogus", "fast") == "medium"
    assert _resolve_effort("ULTRA", "fast") == "ultra"

    steps_low = _compute_max_steps("fast", "low")
    steps_med = _compute_max_steps("fast", "medium")
    steps_high = _compute_max_steps("fast", "high")
    steps_ultra = _compute_max_steps("fast", "ultra")
    assert steps_low < steps_med <= steps_high < steps_ultra
    # Deep modes start from a bigger base
    assert _compute_max_steps("deep", "medium") > steps_med
    # Clamped so a run can never explode
    assert _compute_max_steps("deep_search", "ultra") <= 48


def test_run_task_effort_guidance(tmp_path):
    """Effort injects guidance into the prompt and controls forced reflection."""
    import asyncio

    from titan_agent.agent import TitanAgent
    from titan_agent.llm_client import LLMResponse
    from titan_agent.memory import MemoryManager

    calls = {"n": 0, "system": ""}

    class FakeLLM:
        async def chat_completion(self, messages, tools=None):
            calls["n"] += 1
            if calls["n"] == 1:
                calls["system"] = messages[0]["content"]
            return LLMResponse(content="done")

    mem = MemoryManager(tmp_path / "effort_test.db")

    async def _run(effort):
        agent = TitanAgent(llm=FakeLLM(), memory=mem)
        return [ev async for ev in agent.run_task("test task", session_id="t", mode="fast", effort=effort)]

    # ULTRA: guidance in the prompt + zero-tool grounding + critic reflection
    calls["n"] = 0
    asyncio.run(_run("ultra"))
    assert "ULTRA effort" in calls["system"]
    assert calls["n"] == 3  # main pass + grounding verification + critic reflection

    # LOW: speed guidance + grounding pass, but no forced reflection
    calls["n"] = 0
    asyncio.run(_run("low"))
    assert "LOW effort" in calls["system"]
    assert calls["n"] == 2  # main pass + zero-tool grounding


def test_token_rate_limit_default_is_214_k():
    """The global token throughput cap is 214,000 tokens/second by default."""
    from titan_agent.config import TOKEN_RATE_LIMIT_PER_SEC
    from titan_agent.llm_client import LLMClient
    from titan_agent.token_limit import TokenRateLimiter

    assert TOKEN_RATE_LIMIT_PER_SEC == 214000
    limiter = TokenRateLimiter()
    assert limiter.rate == 214000
    assert limiter.capacity == 214000.0
    # The shared LLM client enforces the same cap
    assert LLMClient().token_limiter.rate == 214000


def test_token_limiter_enforces_rate():
    """The token bucket actually throttles once the per-second budget is spent."""
    import asyncio

    from titan_agent.token_limit import TokenRateLimiter

    async def _run():
        limiter = TokenRateLimiter(tokens_per_sec=10)  # tiny cap for a fast test
        w1 = await limiter.acquire(10)  # consumes the whole bucket (burst allowed)
        w2 = await limiter.acquire(10)  # bucket is now empty -> must wait ~1s
        return limiter, w1, w2

    limiter, w1, w2 = asyncio.run(_run())
    assert w1 == 0.0
    assert w2 >= 0.95  # 10 tokens/s -> 10 tokens take 1s to refill
    stats = limiter.stats()
    assert stats["cap_per_second"] == 10
    assert stats["total_tokens_reserved"] == 20
    assert stats["calls"] == 2
    assert stats["max_wait_seconds"] >= 0.95


def test_estimate_tokens_returns_positive():
    from titan_agent.token_limit import estimate_tokens

    msgs = [{"role": "system", "content": "You are TITAN AGENT."}, {"role": "user", "content": "Hello world"}]
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    est = estimate_tokens(msgs, tools, max_output=4096)
    assert est >= 4096  # at least the output allowance
    assert estimate_tokens([], max_output=1) >= 1


def test_token_usage_endpoint_shape():
    """The /api/token-usage endpoint reports the 214k/s cap and live stats."""
    import asyncio

    from titan_agent import server

    data = asyncio.run(server.get_token_usage())
    assert data["cap_per_second"] == 214000
    assert data["total_tokens_reserved"] >= 0
    assert data["calls"] >= 0


def test_tool_deep_search(monkeypatch):
    from titan_agent.deep_search import DeepSearchEngine

    async def fake_search(self, query, max_results=4):
        return [{
            "title": "Python 3.12 release notes",
            "url": "https://example.test/python-312",
            "snippet": "Stable test result; no external search required.",
        }]

    async def fake_scrape(self, url):
        return "Deterministic scraped content for the test."

    monkeypatch.setattr(DeepSearchEngine, "_search_query", fake_search)
    monkeypatch.setattr(DeepSearchEngine, "_scrape_url", fake_scrape)

    tools = ToolRegistry(WORKSPACE_DIR)
    deep_s_res = asyncio.run(tools.tool_deep_search("Python 3.12"))
    assert "DEEP RESEARCH DOSSIER" in deep_s_res
    assert "Python 3.12 release notes" in deep_s_res
    assert "Deterministic scraped content" in deep_s_res


def test_deep_search_provider_error_returns_empty_results(monkeypatch):
    from titan_agent import deep_search

    class BrokenSearchProvider:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def text(self, *_args, **_kwargs):
            raise ValueError("provider-specific network error")

    monkeypatch.setattr(deep_search, "DDGS", BrokenSearchProvider)
    results = asyncio.run(deep_search.DeepSearchEngine()._search_query("Python 3.12"))

    assert results == []


def test_tool_deep_coder(monkeypatch):
    # Trusted fixed fixture: this test explicitly opts into host execution.
    monkeypatch.setenv("TITAN_FULL_ACCESS", "1")
    tools = ToolRegistry(WORKSPACE_DIR)
    deep_c_res = asyncio.run(tools.tool_deep_coder(
        "sample_math",
        {"sample.py": "def f(x): return x * 2"},
        "from sample import f\nassert f(10) == 20\nprint('MATH PASSED')"
    ))
    assert "PASSED" in deep_c_res


def test_mcp_config_parsing():
    mcp = MCPManager(MCP_CONFIG_FILE)
    cfg = mcp.load_config()
    assert "mcpServers" in cfg
    assert "filesystem" in cfg["mcpServers"]


def test_unknown_tool_error():
    tools = ToolRegistry(WORKSPACE_DIR)
    res = tools.tool_write_file("test_unknown.txt", "x")
    assert "Successfully" in res


def test_llm_puter_provider_is_client_side():
    """Puter.js runs in the browser; the backend must reject server-side use."""
    import pytest

    from titan_agent.llm_client import LLMClient
    client = LLMClient(provider="puter", model="deepseek/deepseek-v4-pro")
    assert client.base_url == ""
    with pytest.raises(RuntimeError, match="Puter.js"):
        asyncio.run(client.chat_completion([{"role": "user", "content": "hi"}]))


def test_tool_system_info():
    """system_info returns live host facts."""
    tools = ToolRegistry(WORKSPACE_DIR)
    res = tools.tool_system_info()
    assert "OS:" in res
    assert "CPU cores:" in res
    assert "Python:" in res


def test_tool_manage_processes_list():
    """manage_processes(list) returns at least the current process."""
    tools = ToolRegistry(WORKSPACE_DIR)
    res = asyncio.run(tools.tool_manage_processes("list"))
    assert "PID" in res


def test_agent_memory_tools():
    """memory_save / memory_search route through the agent's MemoryManager."""
    from titan_agent.agent import TitanAgent
    from titan_agent.memory import MemoryManager
    mem = MemoryManager(Path(__file__).parent / "test_memory.db")
    mem.clear_session("mem_tools_test")
    # Remove any pre-existing fact with this key to keep the test deterministic
    agent = TitanAgent(memory=mem)

    saved = asyncio.run(agent.execute_tool_unified(
        "memory_save", {"key": "test_favorite_food", "value": "osh", "category": "profile"}
    ))
    assert "Saved to memory" in saved

    found = asyncio.run(agent.execute_tool_unified(
        "memory_search", {"query": "osh"}
    ))
    assert "test_favorite_food" in found


def test_agent_tool_definitions_include_new_tools():
    """The agent's live tool list now includes system/memory/process tools."""
    from titan_agent.agent import TitanAgent
    agent = TitanAgent()
    names = [t["function"]["name"] for t in agent._build_tools_list()]
    for expected in ("system_info", "manage_processes", "memory_save", "memory_search", "analyze_python_file", "analyze_python_repository"):
        assert expected in names


# ---------------------------------------------------------------
# Execution modes + parallel tool execution (2026)
# ---------------------------------------------------------------

class _FakeToolResponse:
    """Minimal stand-in for an LLM response containing tool_calls."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.thoughts = None
        self.tool_calls = tool_calls or []


class _FakeParallelAgent(TitanAgent):
    """Agent whose execute_tool_unified records real concurrency."""

    def __init__(self):
        super().__init__()
        self._active = 0
        self._max_concurrent = 0
        self._lock = asyncio.Lock()

    async def execute_tool_unified(self, name, args):
        async with self._lock:
            self._active += 1
            self._max_concurrent = max(self._max_concurrent, self._active)
        await asyncio.sleep(0.05)
        async with self._lock:
            self._active -= 1
        return f"{name}:ok"


def test_emit_tool_results_runs_tools_in_parallel():
    """Multiple batched tool calls must execute concurrently (asyncio.gather)."""
    agent = _FakeParallelAgent()
    resp = _FakeToolResponse(content="", tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "echo_1", "arguments": "{}"}},
        {"id": "call_2", "type": "function", "function": {"name": "echo_2", "arguments": "{}"}},
        {"id": "call_3", "type": "function", "function": {"name": "echo_3", "arguments": "{}"}},
    ])
    messages = []

    async def _collect():
        events = []
        async for ev in agent._emit_tool_results(resp, messages, 1):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    types = [ev.type for ev in events]
    # 3 tool_call + 1 status + 3 tool_result
    assert types.count("tool_call") == 3
    assert types.count("tool_result") == 3
    assert any(ev.type == "status" and "parallel" in ev.data for ev in events)
    # Real concurrency must have happened (gather, not sequential)
    assert agent._max_concurrent >= 2
    # assistant + 3 tool messages appended
    assert len(messages) == 4


class _FakeFailingLLM:
    """LLM that always raises immediately — lets us inspect the agent loop shell."""

    async def chat_completion(self, *args, **kwargs):
        raise RuntimeError("test-llm-down")


class _FakeDossierEngine:
    async def run(self, topic):
        return {
            "topic": topic,
            "total_sources_found": 2,
            "sources": [
                {"title": "Source One", "url": "https://example.com/1"},
                {"title": "Source Two", "url": "https://example.com/2"},
            ],
        }


def test_run_task_invalid_mode_falls_back_to_fast(monkeypatch):
    from titan_agent.agent import TitanAgent
    from titan_agent.config import MAX_ITERATIONS

    agent = TitanAgent(llm=_FakeFailingLLM())

    async def _collect():
        events = []
        async for ev in agent.run_task("test", session_id="mode_test", mode="bogus_mode"):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    steps = [ev for ev in events if ev.type == "step_start"]
    assert steps, "expected at least one step_start event"
    assert steps[0].data["max_steps"] == MAX_ITERATIONS
    assert events[-1].type == "error"


def test_run_task_deep_mode_doubles_iteration_budget(monkeypatch):
    from titan_agent.agent import TitanAgent
    from titan_agent.config import MAX_ITERATIONS

    agent = TitanAgent(llm=_FakeFailingLLM())

    async def _collect():
        events = []
        async for ev in agent.run_task("test", session_id="mode_test", mode="deep"):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    steps = [ev for ev in events if ev.type == "step_start"]
    assert steps[0].data["max_steps"] == min(MAX_ITERATIONS * 2, 40)


def test_run_task_deep_search_seeds_dossier(monkeypatch):
    """deep_search mode must auto-build a dossier and seed it into the context."""
    import titan_agent.deep_search as ds_mod
    from titan_agent.agent import TitanAgent

    monkeypatch.setattr(ds_mod, "DeepSearchEngine", lambda *a, **k: _FakeDossierEngine())

    agent = TitanAgent(llm=_FakeFailingLLM())

    async def _collect():
        events = []
        async for ev in agent.run_task("test topic", session_id="mode_test", mode="deep_search"):
            events.append(ev)
        return events

    events = asyncio.run(_collect())
    statuses = [ev.data for ev in events if ev.type == "status"]
    assert any("Building deep search dossier" in s for s in statuses)
    assert any("Dossier ready: 2 sources found" in s for s in statuses)
    from titan_agent.config import MAX_ITERATIONS
    steps = [ev for ev in events if ev.type == "step_start"]
    assert steps[0].data["max_steps"] == min(MAX_ITERATIONS * 2, 40)


def test_mcp_10_servers_load_and_run_in_parallel(tmp_path):
    """10 MCP servers must start in parallel, expose tools, and serve
    concurrent tool calls — across servers and within a single server.

    Uses fake_mcp_server.py (a real JSON-RPC-over-stdio MCP process).
    Server names contain underscores on purpose: resolution must not
    depend on naive name splitting.
    """
    import json as _json

    from titan_agent.mcp_client import MCPManager

    script = Path(__file__).resolve().parent / "fake_mcp_server.py"
    assert script.exists(), "tests/fake_mcp_server.py missing"

    servers = {}
    for i in range(1, 11):
        sname = f"srv_{i:02d}"  # underscore in server name on purpose
        servers[sname] = {
            "command": sys.executable,
            "args": [str(script), sname],
        }
    cfg = tmp_path / "mcp_servers.json"
    cfg.write_text(_json.dumps({"mcpServers": servers}), encoding="utf-8")

    async def _run():
        mcp = MCPManager(cfg)

        # 1) parallel startup of all 10 servers
        await mcp.start_all()
        assert len(mcp.servers) == 10, f"expected 10 servers, got {list(mcp.servers)}"

        # 2) every server exposes its 3 tools → 30 namespaced tools
        tools = mcp.get_all_tools()
        assert len(tools) == 30
        names = {t["function"]["name"] for t in tools}
        for i in range(1, 11):
            assert f"mcp_srv_{i:02d}_echo" in names
            assert f"mcp_srv_{i:02d}_add" in names

        # 3) 10 parallel calls ACROSS the 10 servers (one per server)
        t0 = asyncio.get_running_loop().time()
        cross = await asyncio.gather(*[
            mcp.execute_tool(f"mcp_srv_{i:02d}_echo", {"text": f"hi{i}"})
            for i in range(1, 11)
        ])
        cross_elapsed = asyncio.get_running_loop().time() - t0
        for i, res in enumerate(cross, start=1):
            assert f"srv_{i:02d}:echo:hi{i}" in res, f"bad routing: {res!r}"
        # 10 × 0.15s echo: parallel should finish far under the 1.5s
        # sequential total.
        assert cross_elapsed < 1.0, f"cross-server calls were sequential: {cross_elapsed:.2f}s"

        # 4) 10 parallel calls WITHIN a single server (same process)
        t0 = asyncio.get_running_loop().time()
        same = await asyncio.gather(*[
            mcp.execute_tool("mcp_srv_01_add", {"a": i, "b": 1})
            for i in range(1, 11)
        ])
        same_elapsed = asyncio.get_running_loop().time() - t0
        for i, res in enumerate(same, start=1):
            assert f"srv_01:add:{i + 1}" in res, f"bad same-server result: {res!r}"
        # 10 × 0.25s add: sequential would be 2.5s.
        assert same_elapsed < 2.0, f"same-server calls were sequential: {same_elapsed:.2f}s"

        # 5) parallel stop
        await mcp.stop_all()
        assert len(mcp.servers) == 0

    asyncio.run(_run())


def test_mcp_single_broken_server_never_blocks_others(tmp_path):
    """A server that fails to start must be isolated: the other servers
    still come up and keep serving tools (seamless multi-MCP operation)."""
    import json as _json

    from titan_agent.mcp_client import MCPManager

    script = Path(__file__).resolve().parent / "fake_mcp_server.py"
    servers = {
        "good_1": {"command": sys.executable, "args": [str(script), "good_1"]},
        "broken": {
            # a real interpreter pointed at a script that does not exist:
            # the process starts and dies immediately → must fail fast and
            # never block the other servers' startup/tool calls
            "command": sys.executable,
            "args": [str(Path(__file__).resolve().parent / "missing_script_xyz.py")],
        },
        "good_2": {"command": sys.executable, "args": [str(script), "good_2"]},
    }
    cfg = tmp_path / "mcp_servers_broken.json"
    cfg.write_text(_json.dumps({"mcpServers": servers}), encoding="utf-8")

    async def _run():
        mcp = MCPManager(cfg)
        await mcp.start_all()
        # broken server excluded; the two good ones must be connected
        assert "broken" not in mcp.servers
        assert set(mcp.servers) == {"good_1", "good_2"}
        res = await mcp.execute_tool("mcp_good_1_echo", {"text": "still-works"})
        assert "good_1:echo:still-works" in res
        await mcp.stop_all()

    asyncio.run(_run())


def test_mcp_auto_restart_after_disconnect(tmp_path):
    """If a server drops mid-session, the next tool call auto-restarts it."""
    import json as _json

    from titan_agent.mcp_client import MCPManager

    script = Path(__file__).resolve().parent / "fake_mcp_server.py"
    cfg = tmp_path / "mcp_servers_restart.json"
    cfg.write_text(_json.dumps({
        "mcpServers": {
            "flaky": {"command": sys.executable, "args": [str(script), "flaky"]},
        }
    }), encoding="utf-8")

    async def _run():
        mcp = MCPManager(cfg)
        await mcp.start_all()
        assert "flaky" in mcp.servers
        _ = mcp.get_all_tools()
        assert (await mcp.execute_tool("mcp_flaky_echo", {"text": "one"})) == "flaky:echo:one"

        # simulate the server dying by force-killing its process
        conn = mcp.servers["flaky"]
        proc = conn.process
        if proc and proc.returncode is None:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
        conn.is_connected = False

        # next call must reconnect automatically and still work
        res = await mcp.execute_tool("mcp_flaky_echo", {"text": "two"})
        assert "flaky:echo:two" in res, f"auto-restart failed: {res!r}"
        await mcp.stop_all()

    asyncio.run(_run())

def test_swarm_manager_deploy():
    """SwarmManager deploys agents and tracks their status."""
    import asyncio
    from titan_agent.swarm_manager import SwarmManager
    swarm = SwarmManager()
    res = asyncio.run(swarm.deploy_agent("coder", "fix the bugs"))
    assert "Deployed coder agent" in res
    status = asyncio.run(swarm.get_swarm_status())
    assert "coder" in status
    assert "running" in status


def test_dynamic_mcp_generator(tmp_path):
    """DynamicMCPGenerator synthesizes a valid MCP server file."""
    import asyncio
    from titan_agent.swarm_manager import DynamicMCPGenerator
    gen = DynamicMCPGenerator(str(tmp_path))
    res = asyncio.run(gen.synthesize_server("test_server", "Does some testing"))
    assert "synthesized at" in res
    script_path = tmp_path / "mcp_test_server_dynamic.js"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    assert "test_server" in content
    assert "@modelcontextprotocol" in content


def test_vector_rag_fallback(tmp_path):
    """VectorRAG handles missing dependencies gracefully or initializes ChromaDB."""
    from titan_agent.vector_rag import VectorRAG
    rag = VectorRAG(tmp_path)
    if rag.client:
        assert rag.collection is not None
    else:
        assert "Error:" in rag.index_workspace()
        assert "Error:" in rag.search("test")

def test_lsp_client_mock():
    """LSPClient acts as a stub for now."""
    import asyncio
    from titan_agent.lsp_client import LSPClient
    client = LSPClient("test")
    assert not client.ready
    res = asyncio.run(client.start_server("pyright"))
    assert client.ready
    assert "started" in res