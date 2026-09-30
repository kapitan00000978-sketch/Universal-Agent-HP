"""Phase 5 — Devin-style session checkpoints / resume tests."""

import asyncio

import pytest

from titan_agent.agent import TitanAgent
from titan_agent.checkpoint import CheckpointStore, RunCheckpoint
from titan_agent.llm_client import LLMResponse

# ---------- store primitives ----------


def test_checkpoint_database_is_owner_only_on_posix(tmp_path):
    import os
    import stat

    path = tmp_path / "private-checkpoints.db"
    CheckpointStore(path)
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_store_round_trip(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    cp = RunCheckpoint(
        session_id="s1",
        user_input="fix login",
        mode="fast",
        effort="medium",
        strategy="react",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        steps_done=3,
        tools_used=["web_search", "write_file"],
        status="running",
    )
    store.save(cp)

    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.session_id == "s1"
    assert loaded.user_input == "fix login"
    assert loaded.strategy == "react"
    assert loaded.steps_done == 3
    assert loaded.tools_used == ["web_search", "write_file"]
    assert loaded.messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert loaded.status == "running"


def test_store_save_overwrites_same_session(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(RunCheckpoint(session_id="s1", user_input="a", steps_done=1))
    store.save(RunCheckpoint(session_id="s1", user_input="b", steps_done=5, status="done", final_answer="ans"))
    loaded = store.load("s1")
    assert loaded.user_input == "b"
    assert loaded.steps_done == 5
    assert loaded.status == "done"
    assert loaded.final_answer == "ans"
    assert store.stats() == {"total": 1, "done": 1}


def test_store_list_newest_first_and_delete(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(RunCheckpoint(session_id="old", user_input="first"))
    store.save(RunCheckpoint(session_id="new", user_input="second"))
    sessions = [cp.session_id for cp in store.list()]
    assert sessions == ["new", "old"]
    assert store.delete("old") is True
    assert store.load("old") is None
    assert store.delete("old") is False


def test_store_load_missing_returns_none(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    assert store.load("nope") is None


def test_reconcile_tool_batch_records_exact_operator_outcomes(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="interrupted",
            user_input="do two operations",
            messages=[
                {"role": "user", "content": "do two operations"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "telegram_send", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "already returned"},
            ],
            status="tool_in_progress",
        )
    )

    reconciled = store.reconcile_tool_batch(
        "interrupted",
        {"c2": "Verified in Telegram sent-items; message was delivered."},
        operator="operator-7",
    )

    assert reconciled.status == "running"
    assert [m.get("tool_call_id") for m in reconciled.messages if m.get("role") == "tool"] == ["c1", "c2"]
    result = reconciled.messages[-1]["content"]
    assert "operator-7" in result
    assert "NOT replayed" in result
    assert "message was delivered" in result
    assert store.load("interrupted").status == "running"


def test_reconcile_rejects_missing_extra_and_structured_outcomes(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="classic",
            user_input="perform action",
            messages=[{"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": "{}"}}
            ]}],
            status="tool_in_progress",
        )
    )
    with pytest.raises(ValueError, match="match pending"):
        store.reconcile_tool_batch("classic", {"other": "done"}, operator="operator")
    with pytest.raises(ValueError, match="match pending"):
        store.reconcile_tool_batch("classic", {"c1": "done", "c2": "extra"}, operator="operator")

    store.save(RunCheckpoint(session_id="structured", user_input="task", status="structured_in_progress"))
    with pytest.raises(ValueError, match="classic tool_in_progress"):
        store.reconcile_tool_batch("structured", {"c1": "done"}, operator="operator")


def test_reconciled_resume_continues_without_replaying_tool_calls(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="reconcile-resume",
            user_input="send a status message",
            messages=[
                {"role": "user", "content": "send a status message"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "send-1", "type": "function", "function": {
                        "name": "telegram_send", "arguments": "{}"
                    }}
                ]},
            ],
            status="tool_in_progress",
        )
    )
    store.reconcile_tool_batch(
        "reconcile-resume",
        {"send-1": "Operator verified delivery receipt."},
        operator="tester",
    )
    fake = FakeLLM([
        LLMResponse(content="NO_TOOLS_NEEDED Resumed after operator verification."),
        LLMResponse(content="NO_TOOLS_NEEDED Operator verification confirms the action was not replayed."),
    ])
    agent = TitanAgent(llm=fake, checkpoint_path=tmp_path / "cp.db")

    events = _run(agent, "ignored on resume", session_id="reconcile-resume", resume=True)

    assert any(ev.type == "final_answer" for ev in events)
    assert fake.calls == 2
    reconciled_results = [m for m in fake.seen_messages if m.get("role") == "tool"]
    assert len(reconciled_results) == 1
    assert "NOT replayed" in reconciled_results[0]["content"]
    assert CheckpointStore(tmp_path / "cp.db").load("reconcile-resume").status == "done"


# ---------- agent integration ----------


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0
        self.seen_messages = None

    async def chat_completion(self, messages, tools=None):
        self.calls += 1
        self.seen_messages = list(messages)
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(content="Done: yes.")


def _run(agent, task, **kwargs):
    async def _go():
        return [ev async for ev in agent.run_task(task, session_id=kwargs.pop("session_id", "s"), **kwargs)]

    return asyncio.run(_go())


def test_run_task_always_on_checkpoint_marks_done(tmp_path):
    agent = TitanAgent(llm=FakeLLM(), checkpoint_path=tmp_path / "cp.db")
    events = _run(agent, "summarize the docs", mode="fast")
    assert any(ev.type == "final_answer" for ev in events)

    store = CheckpointStore(tmp_path / "cp.db")
    cp = store.load("s")
    assert cp is not None and cp.status == "done"
    assert cp.final_answer == "Done: yes."
    assert cp.user_input == "summarize the docs"


def test_run_task_records_tools_used(tmp_path):
    def _resp_with_tool():
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "memory_save", "arguments": '{"key":"k","value":"v"}'},
                }
            ],
        )

    fake = FakeLLM(responses=[_resp_with_tool(), LLMResponse(content="saved the fact")])
    agent = TitanAgent(llm=fake, checkpoint_path=tmp_path / "cp.db")
    events = _run(agent, "remember something", mode="fast")
    assert any(ev.type == "tool_call" for ev in events)

    cp = CheckpointStore(tmp_path / "cp.db").load("s")
    assert cp.tools_used == ["memory_save"]
    assert cp.steps_done >= 1


def test_resume_restores_messages_and_emits_status(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="s",
            user_input="finish the login work",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "finish the login work"},
                {"role": "assistant", "content": "PARTIAL WORK DONE: auth module wired"},
            ],
            steps_done=4,
            status="running",
        )
    )

    fake = FakeLLM()
    agent = TitanAgent(llm=fake, checkpoint=store)
    events = _run(agent, "finish the login work", resume=True, mode="fast")

    assert any(ev.type == "status" and "Resuming session 's'" in str(ev.data) for ev in events)
    assert any(ev.type == "final_answer" for ev in events)
    # The LLM saw the restored checkpoint context, not a fresh conversation
    roles = [m["role"] for m in fake.seen_messages]
    assert "system" in roles
    assert any("PARTIAL WORK DONE" in str(m.get("content", "")) for m in fake.seen_messages)

    cp = CheckpointStore(tmp_path / "cp.db").load("s")
    assert cp.status == "done"
    assert cp.steps_done >= 4


def test_resume_without_checkpoint_is_normal_run(tmp_path):
    fake = FakeLLM()
    agent = TitanAgent(llm=fake, checkpoint_path=tmp_path / "cp.db")
    events = _run(agent, "brand new task", resume=True, mode="fast")
    assert not any(ev.type == "status" and "Resuming" in str(ev.data) for ev in events)
    assert any(ev.type == "final_answer" for ev in events)


def test_run_checkpoints_tool_intent_before_side_effect_and_resume_blocks_replay(tmp_path):
    tool_response = LLMResponse(
        tool_calls=[
            {
                "id": "call-risky-1",
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "arguments": '{"path":"artifact.txt"}',
                },
            }
        ]
    )
    fake = FakeLLM(responses=[tool_response])
    agent = TitanAgent(llm=fake, checkpoint_path=tmp_path / "cp.db")
    attempted = []

    async def interrupted_after_possible_side_effect(name, args):
        attempted.append((name, args))
        raise asyncio.CancelledError()

    agent.execute_tool_unified = interrupted_after_possible_side_effect

    with pytest.raises(asyncio.CancelledError):
        _run(agent, "remove the generated artifact", mode="fast")

    store = CheckpointStore(tmp_path / "cp.db")
    cp = store.load("s")
    assert cp is not None
    assert cp.status == "tool_in_progress"
    assert cp.messages[-1]["role"] == "assistant"
    assert cp.messages[-1]["tool_calls"][0]["id"] == "call-risky-1"
    assert attempted and attempted[0][0] == "delete_file"

    class NoReplayLLM:
        async def chat_completion(self, messages, tools=None):
            raise AssertionError("an ambiguous side effect must not be replayed through the model")

    resumed = TitanAgent(llm=NoReplayLLM(), checkpoint=store)
    events = _run(resumed, "remove the generated artifact", resume=True, mode="fast")
    final = next(event.data for event in events if event.type == "final_answer")
    assert "Paused safely" in final
    assert "delete_file" in final
    assert "no pending action was replayed" in " ".join(str(event.data) for event in events).lower()
    assert store.load("s").status == "tool_in_progress"


def test_checkpoint_failure_prevents_tool_execution(tmp_path, monkeypatch):
    class FakeToolLLM(FakeLLM):
        async def chat_completion(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[{
                        "id": "call-no-checkpoint",
                        "type": "function",
                        "function": {"name": "delete_file", "arguments": '{"path":"x"}'},
                    }]
                )
            return LLMResponse(content="The action was blocked before execution.")

    agent = TitanAgent(llm=FakeToolLLM(), checkpoint_path=tmp_path / "cp.db")
    calls = []

    async def should_never_run(name, args):
        calls.append((name, args))
        return "deleted"

    agent.execute_tool_unified = should_never_run
    monkeypatch.setattr(agent, "_checkpoint_save", lambda **_kwargs: False)
    events = _run(agent, "remove x", mode="fast")

    assert calls == []
    assert any(
        event.type == "error" and "pre-action checkpoint" in str(event.data)
        for event in events
    )


def test_structured_execution_has_durable_in_progress_marker(tmp_path):
    class CancelledLLM:
        async def chat_completion(self, messages, tools=None, **kwargs):
            raise asyncio.CancelledError()

    agent = TitanAgent(llm=CancelledLLM(), checkpoint_path=tmp_path / "cp.db")
    with pytest.raises(asyncio.CancelledError):
        _run(agent, "structured task", strategy="react", mode="fast")

    cp = CheckpointStore(tmp_path / "cp.db").load("s")
    assert cp is not None
    assert cp.status == "structured_in_progress"


def test_resume_preserves_openai_tool_message_structure(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="s",
            user_input="continue after tool work",
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "continue after tool work"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call-7",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call-7", "name": "read_file", "content": "file text"},
            ],
            steps_done=2,
            tools_used=["read_file"],
            status="running",
        )
    )

    fake = FakeLLM()
    agent = TitanAgent(llm=fake, checkpoint=store)
    events = _run(agent, "continue after tool work", resume=True)
    assert any(event.type == "final_answer" for event in events)
    assistant = next(message for message in fake.seen_messages if message["role"] == "assistant")
    tool_message = next(message for message in fake.seen_messages if message["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == "call-7"
    assert tool_message["tool_call_id"] == "call-7"


def test_resume_done_session_returns_saved_result_without_llm(tmp_path):
    store = CheckpointStore(tmp_path / "cp.db")
    store.save(
        RunCheckpoint(
            session_id="s",
            user_input="already done task",
            status="done",
            final_answer="the saved answer",
            messages=[{"role": "user", "content": "already done task"}],
        )
    )

    class _NoCallLLM:
        async def chat_completion(self, messages, tools=None):
            raise AssertionError("LLM must not be called for a completed session")

    agent = TitanAgent(llm=_NoCallLLM(), checkpoint=store)
    events = _run(agent, "already done task", resume=True, mode="fast")

    finals = [ev.data for ev in events if ev.type == "final_answer"]
    assert finals == ["the saved answer"]
    assert any(ev.type == "status" and "already completed" in str(ev.data) for ev in events)


def test_llm_error_records_error_status(tmp_path):
    class _FailingLLM:
        async def chat_completion(self, messages, tools=None):
            raise RuntimeError("test-llm-down")

    agent = TitanAgent(llm=_FailingLLM(), checkpoint_path=tmp_path / "cp.db")
    events = _run(agent, "will explode", mode="fast")
    assert any(ev.type == "error" for ev in events)
    cp = CheckpointStore(tmp_path / "cp.db").load("s")
    assert cp is not None and cp.status == "error"


def test_structured_success_marks_checkpoint_done(tmp_path):
    """A plan/react/tot run saves the checkpoint as done with its final answer."""
    agent = TitanAgent(llm=FakeLLM(), checkpoint_path=tmp_path / "cp.db")
    events = _run(agent, "structured task", strategy="react", mode="fast")
    assert any(ev.type == "final_answer" for ev in events)
    cp = CheckpointStore(tmp_path / "cp.db").load("s")
    assert cp is not None and cp.status == "done"


# ---------- pass-through ----------


def test_headless_and_server_expose_resume():
    import inspect

    from titan_agent.headless import run_headless
    from titan_agent.server import ChatRequest

    assert "resume" in inspect.signature(run_headless).parameters
    assert ChatRequest(message="x").resume is False
    assert ChatRequest(message="x", resume=True).resume is True