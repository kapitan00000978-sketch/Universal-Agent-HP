"""The agent-wide approval funnel must block policy denies and unsafe integrations."""
from __future__ import annotations

import asyncio

from titan_agent.agent import TitanAgent


class _LLM:
    async def chat_completion(self, *_args, **_kwargs):
        raise AssertionError("approval tests must not call the model")


def _gate(agent: TitanAgent, name: str, args: dict):
    return asyncio.run(agent._approval_gate(name, args))


def test_policy_deny_cannot_fall_through_to_tool_execution(monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    agent = TitanAgent(llm=_LLM())

    assert _gate(agent, "execute_command", {"command": "rm -rf /tmp/example"}) is False
    assert _gate(agent, "tool_execute_command", {"command": "rm -rf /tmp/example"}) is False
    assert _gate(agent, "self_heal", {"command": "rm -rf /tmp/example"}) is False


def test_mcp_calls_fail_closed_without_human_approval(monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    agent = TitanAgent(llm=_LLM(), hitl=None)

    assert _gate(agent, "mcp_filesystem_write_file", {"path": "report.txt"}) is False


def test_file_downloads_and_telegram_logout_fail_closed_without_approval(monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    agent = TitanAgent(llm=_LLM(), hitl=None)

    assert _gate(agent, "download_file", {"url": "https://example.com/file.bin"}) is False
    assert _gate(agent, "tool_download_file", {"url": "https://example.com/file.bin"}) is False
    assert _gate(agent, "telegram_logout", {"label": "primary", "delete": True}) is False


def test_synthesized_tools_and_sandbox_expansion_need_human_approval(monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    agent = TitanAgent(llm=_LLM(), hitl=None)
    agent.tools._synthesized_definitions = {"generated_helper": {}}

    assert _gate(agent, "generated_helper", {"value": "x"}) is False
    assert _gate(agent, "docker_sandbox_run", {"command": "curl example", "network": "bridge"}) is False
    assert _gate(agent, "docker_sandbox_run", {"command": "python -c pass", "network": "none"}) is None


def test_explicit_full_access_preserves_approval_bypass(monkeypatch):
    monkeypatch.setenv("TITAN_FULL_ACCESS", "1")
    agent = TitanAgent(llm=_LLM(), hitl=None)

    assert _gate(agent, "mcp_external_action", {"value": "confirmed by operator"}) is None
