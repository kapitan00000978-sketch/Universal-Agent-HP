import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.catalog import load_catalog
from benchmarks.run import main, run_case, select_cases


def test_default_selection_skips_only_critical_safety_probe():
    cases = load_catalog()["cases"]

    runnable, skipped = select_cases(cases)

    assert len(runnable) == 7
    assert [case["id"] for case in skipped] == ["safety-destructive-command"]


def test_critical_probe_requires_explicit_opt_in():
    cases = load_catalog()["cases"]
    selected = {"safety-destructive-command"}

    runnable, skipped = select_cases(cases, selected_ids=selected)
    assert runnable == []
    assert len(skipped) == 1

    runnable, skipped = select_cases(
        cases,
        selected_ids=selected,
        include_safety_probes=True,
    )
    assert [case["id"] for case in runnable] == ["safety-destructive-command"]
    assert skipped == []


def test_selection_rejects_unknown_case_id():
    with pytest.raises(ValueError, match="Unknown benchmark case"):
        select_cases(load_catalog()["cases"], selected_ids={"not-a-real-case"})


def test_list_mode_does_not_run_models(capsys):
    assert main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "coding-small-change" in output
    assert "safety-destructive-command" in output


def test_run_case_records_tool_risk_without_saving_raw_arguments(tmp_path, monkeypatch):
    case = load_catalog()["cases"][0]

    class FakeAgent:
        llm = SimpleNamespace(provider="fake-provider", model="fake-model")

        async def run_task(self, *_args, **_kwargs):
            yield SimpleNamespace(
                type="tool_call",
                data={
                    "name": "execute_command",
                    "arguments": {"command": "rm -rf /important/data", "api_key": "secret"},
                },
            )
            yield SimpleNamespace(type="final_answer", data="I will not run that command.")

    monkeypatch.setattr("benchmarks.run._new_agent", lambda *_args: FakeAgent())
    monkeypatch.setattr("titan_agent.config.full_access_enabled", lambda: False)

    result = asyncio.run(
        run_case(
            case,
            workspace=tmp_path / "workspace",
            provider=None,
            model=None,
            effort="auto",
            timeout_seconds=5,
        )
    )

    assert result["status"] == "completed"
    assert result["provider"] == "fake-provider"
    assert result["tool_calls"] == [{
        "name": "execute_command",
        "dangerous": True,
        "risk_level": "CRITICAL",
        "category": "recursive_file_deletion",
    }]
    assert "rm -rf" not in str(result["tool_calls"])
    assert "secret" not in str(result)


def test_run_case_refuses_full_access_configuration(tmp_path, monkeypatch):
    case = load_catalog()["cases"][0]

    def agent_must_not_be_created(*_args):
        raise AssertionError("benchmark should refuse before constructing agent")

    monkeypatch.setattr("benchmarks.run._new_agent", agent_must_not_be_created)
    monkeypatch.setattr("titan_agent.config.full_access_enabled", lambda: True)

    result = asyncio.run(
        run_case(
            case,
            workspace=tmp_path / "workspace",
            provider=None,
            model=None,
            effort="auto",
            timeout_seconds=5,
        )
    )

    assert result["status"] == "failed"
    assert "Refusing to run benchmark" in result["errors"][0]
