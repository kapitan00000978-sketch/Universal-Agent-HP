import asyncio
import json
from types import SimpleNamespace

import pytest

from titan_agent import laya_decisions


QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Choose the request category.",
        "criteria": ["billing", "technical", "other"],
    }
}


def test_keyless_decision_call_uses_local_runtime_and_returns_structured_result(monkeypatch):
    calls = []

    def fake_predict(state, questions, backend, model):
        calls.append((state, questions, backend, model))
        return {"backend": "laya", "result": {"answers": {"intent": {"choice": "billing"}}}}

    monkeypatch.setattr(laya_decisions, "_predict_sync", fake_predict)
    output = asyncio.run(laya_decisions.predict_decisions(
        "I was charged twice.", QUESTIONS, backend="laya"
    ))

    result = json.loads(output)
    assert calls == [("I was charged twice.", QUESTIONS, "laya", "")]
    assert result["backend"] == "laya"
    assert result["result"]["answers"]["intent"]["choice"] == "billing"


def test_upstream_laya_model_selection_is_forwarded_to_router(monkeypatch):
    calls = []
    runtime = SimpleNamespace(predict=lambda *args, **kwargs: calls.append((args, kwargs)) or {"ok": True})
    monkeypatch.setattr(laya_decisions, "_load_runtime", lambda *_args: runtime)

    result = laya_decisions._predict_sync("text", QUESTIONS, "laya", "multilingual")

    assert result["result"]["ok"] is True
    assert calls[0][1] == {"model": "multilingual"}


def test_auto_backend_falls_back_from_mlx_to_upstream_laya(monkeypatch):
    calls = []

    def fake_predict(state, questions, backend, model):
        calls.append(backend)
        if backend == "laya-mlx":
            raise RuntimeError("MLX runtime unavailable")
        return {"backend": backend, "result": {"ok": True}}

    monkeypatch.setattr(laya_decisions, "_default_backend", lambda: "laya-mlx")
    monkeypatch.setattr(laya_decisions, "_predict_sync", fake_predict)
    result = json.loads(asyncio.run(laya_decisions.predict_decisions(
        "state", QUESTIONS, backend="auto"
    )))

    assert calls == ["laya-mlx", "laya"]
    assert result["backend"] == "laya"
    assert result["fallback_from"] == "laya-mlx"


def test_decision_call_validates_inputs_before_loading_models(monkeypatch):
    def must_not_run(*_args, **_kwargs):
        raise AssertionError("invalid inputs must be rejected before inference")

    monkeypatch.setattr(laya_decisions, "_predict_sync", must_not_run)

    assert asyncio.run(laya_decisions.predict_decisions("", QUESTIONS)).startswith("Error:")
    assert asyncio.run(laya_decisions.predict_decisions("state", {})).startswith("Error:")
    assert asyncio.run(laya_decisions.predict_decisions("state", QUESTIONS, backend="unknown")).startswith("Error:")


def test_auto_backend_prefers_mlx_only_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(laya_decisions.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(laya_decisions.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(laya_decisions.importlib.util, "find_spec", lambda name: object())
    assert laya_decisions._default_backend() == "laya-mlx"

    monkeypatch.setattr(laya_decisions.platform, "system", lambda: "Linux")
    assert laya_decisions._default_backend() == "laya"


def test_laya_is_available_as_a_regular_tool_without_api_credentials(tmp_path, monkeypatch):
    from titan_agent.tools import ToolRegistry

    async def fake_predict(state, questions, backend="auto", model=""):
        return json.dumps({"backend": backend, "result": {"answer": "yes"}})

    monkeypatch.setattr("titan_agent.laya_decisions.predict_decisions", fake_predict)
    registry = ToolRegistry(tmp_path)
    names = {item["function"]["name"] for item in registry.get_tool_definitions()}
    result = asyncio.run(registry.tool_laya_decide("state", QUESTIONS, backend="laya"))

    assert "laya_decide" in names
    assert json.loads(result)["result"]["answer"] == "yes"


def test_laya_mlx_cannot_masquerade_as_a_chat_model():
    from titan_agent.llm_client import LLMClient

    client = LLMClient(provider="laya-mlx", model="local")
    with pytest.raises(RuntimeError, match="not a chat/text-generation provider"):
        asyncio.run(client._chat_completion_once([{"role": "user", "content": "hello"}]))


def test_keyless_cli_loads_questions_and_runs_laya(tmp_path, monkeypatch, capsys):
    question_file = tmp_path / "questions.json"
    question_file.write_text(json.dumps(QUESTIONS), encoding="utf-8")
    seen = {}

    async def fake_predict(state, questions, backend, model):
        seen.update(state=state, questions=questions, backend=backend, model=model)
        return '{"backend":"laya","result":{}}'

    monkeypatch.setattr(laya_decisions, "predict_decisions", fake_predict)
    status = laya_decisions.main([
        "--state", "example", "--questions", str(question_file), "--backend", "laya"
    ])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "laya"
    assert seen["state"] == "example"
    assert seen["backend"] == "laya"
