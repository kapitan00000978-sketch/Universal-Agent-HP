"""
Phase 14 — Human-in-the-loop: server HTTP API + live agent approval gate.

Drives the real FastAPI app through TestClient (like test_server_auth.py) so the
endpoints exercise the actual dependency tree incl. the phase-12 Bearer auth,
plus unit tests for the two wiring points that make approvals *live*:

- agent.execute_tool_unified gate (single approval funnel for every loop)
- core ToolRegistry require-approval gate (guarded wrapper parity)
- ToolBridge defer_approval (structured strategy defers to the registry gate)

All deterministic: no network, no real LLM, no real tools executed.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Deterministic auth key — set before importing the app (auth reads env per request).
_TEST_KEY = "sk-hitl-" + uuid.uuid4().hex
os.environ["TITAN_API_KEY"] = _TEST_KEY

from titan_agent.agent import TitanAgent
from titan_agent.core.guardrails.hitl import HumanInTheLoop
from titan_agent.core.guardrails.policy import PolicyEngine, Rule
from titan_agent.core.tools.registry import ToolRegistry as GuardedToolRegistry
from titan_agent.llm_client import LLMClient
from titan_agent.server import app  # (must import after env setup)
from titan_agent.structured import ToolBridge
from titan_agent.checkpoint import CheckpointStore, RunCheckpoint
import titan_agent.server as server_module


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def auth() -> dict[str, str]:
    # Read the live env key at call time (test_server_auth also sets this env at
    # import; the last import wins) so this module stays order-independent.
    return {"Authorization": f"Bearer {os.environ.get('TITAN_API_KEY', _TEST_KEY)}"}


def _create(client: TestClient, action: str, resource: str = "*", reason: str = "test") -> dict:
    res = client.post(
        "/api/hitl/request",
        json={"action": action, "resource": resource, "reason": reason},
        headers=auth(),
    )
    assert res.status_code == 200, res.text
    return res.json()["request"]


# ---------------------------------------------------------------------------
# Auth: every /api/hitl route is protected by the phase-12 Bearer token.
# ---------------------------------------------------------------------------

def test_hitl_routes_require_auth(client: TestClient) -> None:
    assert client.get("/api/hitl").status_code == 401
    assert client.get("/api/hitl/pending").status_code == 401
    assert client.get("/api/hitl/pending", headers=auth()).status_code == 200


# ---------------------------------------------------------------------------
# HTTP lifecycle: request -> pending -> decide (approve/deny/cancel) -> audit
# ---------------------------------------------------------------------------

def test_hitl_create_appears_in_pending(client: TestClient) -> None:
    req = _create(client, "delete_file", resource="C:/x.txt")
    assert req["status"] == "pending"
    assert req["action"] == "delete_file"
    assert req["resource"] == "C:/x.txt"

    res = client.get("/api/hitl/pending", headers=auth()).json()
    ids = [r["request_id"] for r in res["pending"]]
    assert req["request_id"] in ids
    assert res["count"] >= 1
    assert res["counts"]["pending"] >= 1


def test_hitl_get_status(client: TestClient) -> None:
    req = _create(client, "screenshot")
    res = client.get(f"/api/hitl/{req['request_id']}", headers=auth())
    assert res.status_code == 200
    assert res.json()["request"]["status"] == "pending"


def test_hitl_approve_flow(client: TestClient) -> None:
    req = _create(client, "delete_file", resource="C:/secret.txt")
    res = client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "approve", "by": "tester"},
        headers=auth(),
    )
    assert res.status_code == 200, res.text
    body = res.json()["request"]
    assert body["status"] == "approved"
    assert body["decided_by"] == "tester"
    assert body["decided_at"] is not None

    # GET reflects the decision
    detail = client.get(f"/api/hitl/{req['request_id']}", headers=auth()).json()["request"]
    assert detail["status"] == "approved"

    # Request no longer pending
    pending_ids = [p["request_id"] for p in client.get("/api/hitl/pending", headers=auth()).json()["pending"]]
    assert req["request_id"] not in pending_ids


def test_hitl_deny_flow(client: TestClient) -> None:
    req = _create(client, "screenshot")
    res = client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "deny"},
        headers=auth(),
    )
    assert res.status_code == 200
    assert res.json()["request"]["status"] == "denied"


def test_hitl_cancel_flow(client: TestClient) -> None:
    req = _create(client, "delete_file")
    res = client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "cancel"},
        headers=auth(),
    )
    assert res.status_code == 200
    assert res.json()["request"]["status"] == "cancelled"


def test_hitl_decide_idempotent_after_resolution(client: TestClient) -> None:
    req = _create(client, "delete_file")
    first = client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "approve"},
        headers=auth(),
    ).json()["request"]
    second = client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "deny"},
        headers=auth(),
    ).json()["request"]
    # A resolved request is never re-decided — first decision stands.
    assert first["status"] == "approved"
    assert second["status"] == "approved"


def test_hitl_decide_invalid_and_unknown(client: TestClient) -> None:
    res = client.post(
        "/api/hitl/decide",
        json={"request_id": "nope", "decision": "maybe"},
        headers=auth(),
    )
    assert res.status_code == 422

    res = client.post(
        "/api/hitl/decide",
        json={"request_id": uuid.uuid4().hex[:12], "decision": "approve"},
        headers=auth(),
    )
    assert res.status_code == 404

    assert client.get("/api/hitl/definitely-missing", headers=auth()).status_code == 404


def test_hitl_audit_trail_lists_recent(client: TestClient) -> None:
    marker = f"audit-{uuid.uuid4().hex[:8]}"
    req = _create(client, marker)
    client.post(
        "/api/hitl/decide",
        json={"request_id": req["request_id"], "decision": "approve"},
        headers=auth(),
    )
    res = client.get("/api/hitl", headers=auth()).json()
    ids = [r["request_id"] for r in res["requests"]]
    assert req["request_id"] in ids
    # Newest first: our marker (just created) must be at the top.
    assert res["requests"][0]["request_id"] == req["request_id"]


# ---------------------------------------------------------------------------
# Live approval gate: agent.execute_tool_unified (the single funnel).
# ---------------------------------------------------------------------------

def test_agent_hitl_wired_onto_tool_registry() -> None:
    hitl = HumanInTheLoop()
    reg = GuardedToolRegistry(_FakeDelegate(), policy=PolicyEngine())
    agent = TitanAgent(llm=LLMClient(), tools=reg, hitl=hitl, hitl_timeout=2.0)
    assert agent.hitl is hitl
    assert agent.tools.hitl is hitl


def test_agent_gate_waits_and_executes_after_approval() -> None:
    hitl = HumanInTheLoop()
    agent = TitanAgent(llm=LLMClient(), hitl=hitl, hitl_timeout=5.0)

    async def scenario() -> tuple[bool | None, object]:
        task = asyncio.create_task(agent._approval_gate("delete_file", {"path": "C:/x.txt"}))
        for _ in range(200):
            if hitl.pending():
                break
            await asyncio.sleep(0.01)
        pending = hitl.pending()
        assert pending, "the gate must create a pending approval request"
        req = pending[0]
        assert req.action == "delete_file"
        hitl.approve(req.request_id, by="tester")
        granted = await task
        return granted, req

    granted, req = asyncio.run(scenario())
    assert granted is True
    assert req.status.value == "approved"


def test_agent_gate_denied_blocks_execution() -> None:
    hitl = HumanInTheLoop()
    agent = TitanAgent(llm=LLMClient(), hitl=hitl, hitl_timeout=5.0)

    async def scenario() -> bool | None:
        task = asyncio.create_task(agent._approval_gate("delete_file", {"path": "C:/y.txt"}))
        for _ in range(200):
            if hitl.pending():
                break
            await asyncio.sleep(0.01)
        hitl.deny(hitl.pending()[0].request_id, by="tester")
        return await task

    assert asyncio.run(scenario()) is False


def test_agent_without_hitl_fails_closed_for_approval_required_tools(monkeypatch) -> None:
    import titan_agent.config as config

    monkeypatch.setattr(config, "_FULL_ACCESS_OVERRIDE", False)
    agent = TitanAgent(llm=LLMClient(), hitl=None, hitl_timeout=2.0)
    calls = []

    async def execute_tool(name, args):
        calls.append((name, args))
        return "deleted"

    monkeypatch.setattr(agent.tools, "execute_tool", execute_tool)
    result = asyncio.run(agent.execute_tool_unified("delete_file", {"path": "C:/z.txt"}))
    assert result.startswith("Error: approval required")
    assert calls == []


def test_agent_approval_gate_exception_fails_closed(monkeypatch) -> None:
    import titan_agent.config as config

    class BrokenHITL:
        def request(self, *args, **kwargs):
            raise RuntimeError("approval service unavailable")

    monkeypatch.setattr(config, "_FULL_ACCESS_OVERRIDE", False)
    agent = TitanAgent(llm=LLMClient(), hitl=BrokenHITL(), hitl_timeout=2.0)
    calls = []

    async def execute_tool(name, args):
        calls.append((name, args))
        return "deleted"

    monkeypatch.setattr(agent.tools, "execute_tool", execute_tool)
    result = asyncio.run(agent.execute_tool_unified("delete_file", {"path": "C:/z.txt"}))
    assert result.startswith("Error: approval required")
    assert calls == []


# ---------------------------------------------------------------------------
# Guarded ToolRegistry: require-approval gate honours the attached HITL.
# ---------------------------------------------------------------------------

class _FakeDelegate:
    def get_tool_definitions(self):
        return [
            {"type": "function", "function": {
                "name": "delete_file", "description": "delete a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {
                "name": "read_file", "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        ]

    def tool_delete_file(self, path: str):
        return f"deleted {path}"

    def tool_read_file(self, path: str):
        return f"read {path}"


def test_guarded_registry_hitl_executes_after_approval() -> None:
    hitl = HumanInTheLoop()
    policy = PolicyEngine(rules=[Rule("delete_file", "*", "require_approval", "needs consent")])
    reg = GuardedToolRegistry(_FakeDelegate(), policy=policy, hitl=hitl, hitl_timeout=5.0)

    async def scenario() -> str:
        task = asyncio.create_task(reg.execute("delete_file", {"path": "x.txt"}))
        for _ in range(200):
            if hitl.pending():
                break
            await asyncio.sleep(0.01)
        assert hitl.pending(), "registry gate must create a pending request"
        req = hitl.pending()[0]
        assert req.details == {"args": {"path": "x.txt"}}
        hitl.approve(req.request_id)
        return await task

    assert asyncio.run(scenario()) == "deleted x.txt"


def test_guarded_registry_hitl_denied_raises() -> None:
    hitl = HumanInTheLoop()
    policy = PolicyEngine(rules=[Rule("delete_file", "*", "require_approval", "needs consent")])
    reg = GuardedToolRegistry(_FakeDelegate(), policy=policy, hitl=hitl, hitl_timeout=5.0)

    async def scenario():
        task = asyncio.create_task(reg.execute("delete_file", {"path": "x.txt"}))
        for _ in range(200):
            if hitl.pending():
                break
            await asyncio.sleep(0.01)
        hitl.deny(hitl.pending()[0].request_id)
        await task

    with pytest.raises(PermissionError):
        asyncio.run(scenario())


def test_guarded_registry_no_hitl_keeps_denial() -> None:
    policy = PolicyEngine(rules=[Rule("delete_file", "*", "require_approval", "needs consent")])
    reg = GuardedToolRegistry(_FakeDelegate(), policy=policy)  # hitl not wired
    with pytest.raises(PermissionError):
        asyncio.run(reg.execute("delete_file", {"path": "x.txt"}))


# ---------------------------------------------------------------------------
# ToolBridge defer_approval: structured strategy defers to the registry gate.
# ---------------------------------------------------------------------------

def test_tool_bridge_defer_approval_passes_through() -> None:
    policy = PolicyEngine(rules=[Rule("delete_file", "*", "require_approval", "needs consent")])
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return "deleted"

    bridge = ToolBridge(execute, lambda: [_SIMPLE_TOOL_DEF], policy_engine=policy, hitl=None, defer_approval=True)
    res = asyncio.run(bridge.execute("delete_file", {"path": "C:/x.txt"}))
    assert res == "deleted"
    assert calls == [("delete_file", {"path": "C:/x.txt"})]


def test_tool_bridge_default_still_denies_without_responder() -> None:
    policy = PolicyEngine(rules=[Rule("delete_file", "*", "require_approval", "needs consent")])

    async def execute(name, args):
        raise AssertionError("must not run")

    bridge = ToolBridge(execute, lambda: [_SIMPLE_TOOL_DEF], policy_engine=policy, hitl=None)
    result = asyncio.run(bridge.execute("delete_file", {"path": "C:/x.txt"}))
    assert "approval required" in result


def test_checkpoint_reconciliation_endpoint_is_authenticated_and_does_not_replay(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "reconcile.db")
    store.save(
        RunCheckpoint(
            session_id="api-reconcile",
            user_input="send message",
            messages=[{"role": "assistant", "tool_calls": [
                {"id": "m1", "function": {"name": "telegram_send", "arguments": "{}"}}
            ]}],
            status="tool_in_progress",
        )
    )
    monkeypatch.setattr(server_module.agent, "_checkpoint", store)
    monkeypatch.setattr(server_module.agent, "_checkpoint_loaded", True)

    unauthorized = TestClient(app).post(
        "/api/checkpoints/api-reconcile/reconcile",
        json={"operator": "tester", "outcomes": {"m1": "sent"}},
    )
    assert unauthorized.status_code == 401

    incomplete = TestClient(app).post(
        "/api/checkpoints/api-reconcile/reconcile",
        json={"operator": "tester", "outcomes": {}},
        headers=auth(),
    )
    assert incomplete.status_code in {409, 422}

    response = TestClient(app).post(
        "/api/checkpoints/api-reconcile/reconcile",
        json={"operator": "tester", "outcomes": {"m1": "delivery receipt verified"}},
        headers=auth(),
    )
    assert response.status_code == 200, response.text
    assert response.json()["checkpoint_status"] == "running"
    assert store.load("api-reconcile").status == "running"


_SIMPLE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "delete a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}