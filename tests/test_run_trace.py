"""Privacy and lifecycle tests for local run traces."""
from __future__ import annotations

import asyncio
import json

from titan_agent.agent import TitanAgent
from titan_agent.llm_client import LLMResponse
from titan_agent.memory import MemoryManager
from titan_agent.run_trace import RunTraceStore, bind_run_id, reset_run_id
from titan_agent.tool_stats import ToolStatsCollector


def test_trace_store_allowlists_metadata_and_fingerprints_session(tmp_path):
    store = RunTraceStore(tmp_path / "traces.db")
    token = bind_run_id("run-test-001")
    try:
        store.record(
            "run_started",
            details={
                "mode": "fast",
                "prompt": "PRIVATE_PROMPT_SENTINEL",
                "api_key": "PRIVATE_KEY_SENTINEL",
            },
        )
    finally:
        reset_run_id(token)

    event = store.recent_events()[0]
    serialized = json.dumps(event)
    assert event["event_type"] == "run_started"
    assert "session_fingerprint" not in event
    assert "session_id" not in event
    assert event["details"] == {"mode": "fast"}
    assert "PRIVATE_PROMPT_SENTINEL" not in serialized
    assert "PRIVATE_KEY_SENTINEL" not in serialized
    assert store.stats() == {"events": 1, "runs": 1}


def test_contextvars_keep_concurrent_runs_separate(tmp_path):
    store = RunTraceStore(tmp_path / "traces.db")

    async def record_for_run(run_id):
        token = bind_run_id(run_id)
        try:
            await asyncio.sleep(0)
            store.record("run_started", component="run")
            await asyncio.sleep(0)
            store.record("run_finished", component="run", status="finished")
        finally:
            reset_run_id(token)

    async def run_both():
        await asyncio.gather(record_for_run("run-a"), record_for_run("run-b"))

    asyncio.run(run_both())
    events = store.recent_events(limit=10)
    by_run = {}
    for event in events:
        by_run.setdefault(event["run_id"], []).append(event["event_type"])

    assert set(by_run) == {"run-a", "run-b"}
    assert all(set(kinds) == {"run_started", "run_finished"} for kinds in by_run.values())


def test_run_task_persists_model_lifecycle_without_prompt_or_answer(tmp_path):
    secret_marker = "PRIVATE_RUN_CONTENT_SENTINEL"

    class FakeLLM:
        provider = "fake-provider"
        model = "fake-model"

        async def chat_completion(self, messages, tools=None):
            return LLMResponse(content=f"Answer contains {secret_marker}")

    store = RunTraceStore(tmp_path / "traces.db")
    agent = TitanAgent(
        llm=FakeLLM(),
        trace_store=store,
        checkpoint_path=tmp_path / "checkpoints.db",
        memory=MemoryManager(tmp_path / "memory.db"),
        git_root=tmp_path,
        tool_stats=ToolStatsCollector(),
    )

    async def run():
        return [
            event
            async for event in agent.run_task(
                f"prompt {secret_marker}", session_id="trace-test", mode="fast"
            )
        ]

    events = asyncio.run(run())
    traces = list(reversed(store.recent_events(limit=100)))
    event_types = [trace["event_type"] for trace in traces]
    serialized = json.dumps(traces)

    assert any(event.type == "final_answer" for event in events)
    assert event_types[0] == "run_started"
    assert "model_call_started" in event_types
    assert "model_call_finished" in event_types
    assert event_types[-1] == "run_finished"
    assert traces[0]["details"]["model"] == "fake-model"
    assert secret_marker not in serialized


def test_run_task_traces_tool_names_but_not_arguments(tmp_path):
    sensitive_args = "PRIVATE_TOOL_ARGUMENT_SENTINEL"

    class FakeLLM:
        provider = "fake-provider"
        model = "fake-model"

        def __init__(self):
            self.responses = [
                LLMResponse(
                    tool_calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "memory_save",
                                "arguments": json.dumps({"key": "k", "value": sensitive_args}),
                            },
                        }
                    ]
                ),
                LLMResponse(content="Saved."),
            ]

        async def chat_completion(self, messages, tools=None):
            if self.responses:
                return self.responses.pop(0)
            return LLMResponse(content="No further changes required.")

    store = RunTraceStore(tmp_path / "traces.db")
    agent = TitanAgent(
        llm=FakeLLM(),
        trace_store=store,
        checkpoint_path=tmp_path / "checkpoints.db",
        memory=MemoryManager(tmp_path / "memory.db"),
        git_root=tmp_path,
        tool_stats=ToolStatsCollector(),
    )

    async def run():
        return [event async for event in agent.run_task("save a private value", session_id="trace-tool")]

    events = asyncio.run(run())
    traces = store.recent_events(limit=100)
    tool_events = [event for event in traces if event["component"] == "tool"]
    serialized = json.dumps(traces)

    assert any(event.type == "tool_call" for event in events)
    assert [event["event_type"] for event in tool_events] == ["tool_finished", "tool_started"]
    assert all(event["details"]["tool_name"] == "memory_save" for event in tool_events)
    assert sensitive_args not in serialized
    assert "arguments" not in serialized
