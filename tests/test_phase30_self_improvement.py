"""Tests for Phase 30: Self-Improvement Loop & Eval Benchmark Suite."""

import tempfile
from pathlib import Path

import pytest

from titan_agent.core.self_improvement import (
    EvalCase,
    EvalSuite,
    ImprovementLesson,
    SelfImprovementLoop,
)
from titan_agent.tools import ToolRegistry


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestEvalSuite:
    def test_default_suite_does_not_fabricate_success_without_runner(self):
        suite = EvalSuite()
        res = suite.run_suite()
        assert res["total_cases"] >= 4
        assert res["attempted"] == 0
        assert res["not_run"] == res["total_cases"]
        assert res["passed"] == 0
        assert res["failed"] == 0
        assert res["pass_rate"] is None
        assert res["average_score"] is None
        assert all(result["status"] == "not_run" for result in res["results"])

    def test_live_runner_results_are_counted_as_executed(self):
        case = EvalCase(
            id="actual-run",
            description="Runner output must satisfy the rubric",
            expected_keywords=["checked", "passed"],
        )
        result = EvalSuite([case]).run_suite(runner_fn=lambda _prompt: "Checked and passed.")
        assert result["attempted"] == 1
        assert result["not_run"] == 0
        assert result["passed"] == 1
        assert result["pass_rate"] == 100.0
        assert result["results"][0]["status"] == "completed"

    def test_eval_case_with_custom_runner_and_failure(self):
        suite = EvalSuite([
            EvalCase(
                id="test_greeting",
                description="Checks polite greeting",
                expected_keywords=["hello", "welcome"],
                forbidden_keywords=["error"],
            )
        ])
        # Runner output that misses 'welcome'
        runner = lambda prompt: "Hello there user!"
        res = suite.run_suite(runner_fn=runner)
        assert res["total_cases"] == 1
        assert res["passed"] == 0
        assert "welcome" in (res["results"][0]["error"] or "")

    def test_category_filtering(self):
        suite = EvalSuite()
        coding_res = suite.run_suite(category="coding")
        assert coding_res["total_cases"] >= 1
        for case_res in coding_res["results"]:
            assert "python" in case_res["case_id"] or "coding" in case_res["case_id"]


class TestSelfImprovementLoop:
    def test_analyze_syntax_error(self, temp_workspace: Path):
        loop = SelfImprovementLoop(workspace_root=temp_workspace)
        lesson = loop.analyze_failure(
            task_id="task_fail_1",
            prompt="Refactor user model",
            failure_log="SyntaxError: invalid syntax in models.py line 42",
            category="coding",
        )
        assert isinstance(lesson, ImprovementLesson)
        assert "syntax" in lesson.root_cause.lower()
        assert "ast_patch_file" in lesson.rule_text.lower() or "ast" in lesson.guidance.lower()

    def test_analyze_timeout_error(self, temp_workspace: Path):
        loop = SelfImprovementLoop(workspace_root=temp_workspace)
        lesson = loop.analyze_failure(
            task_id="task_fail_2",
            prompt="Process large dataset",
            failure_log="Timeout after 60.0s waiting for process",
            category="data_processing",
        )
        assert "timeout" in lesson.root_cause.lower()
        assert "dag" in lesson.guidance.lower() or "dag" in lesson.rule_text.lower()

    def test_analyze_tool_failure(self, temp_workspace: Path):
        loop = SelfImprovementLoop(workspace_root=temp_workspace)
        lesson = loop.analyze_failure(
            task_id="task_fail_3",
            prompt="Deploy to staging",
            failure_log="Execution failed with exit code 1",
            failed_tools=["docker_sandbox_run"],
            category="devops",
        )
        assert "docker_sandbox_run" in lesson.root_cause
        assert "docker_sandbox_run" in lesson.rule_text

    def test_propose_prompt_refinement(self, temp_workspace: Path):
        loop = SelfImprovementLoop(workspace_root=temp_workspace)
        lesson = loop.analyze_failure(
            task_id="task_refine",
            prompt="Fix bug",
            failure_log="SyntaxError: unexpected EOF",
            category="coding",
        )
        diff = loop.propose_prompt_refinement("coding", "Original system prompt", [lesson])
        assert "--- system_prompt_coding.txt" in diff
        assert "+++ system_prompt_coding.txt" in diff
        assert "+ [LESSON LEARNED CODING]:" in diff

    def test_crystallize_lesson(self, temp_workspace: Path):
        loop = SelfImprovementLoop(workspace_root=temp_workspace)
        lesson = ImprovementLesson(
            task_id="task_crystallize",
            category="testing",
            symptom="Tests failed due to missing mock",
            root_cause="Unmocked network request in unit test",
            guidance="Always mock external HTTP requests in test suite",
            rule_text="RULE [TESTING]: Never make real HTTP calls in unit tests",
        )
        from titan_agent.core.memory import KnowledgeGraph
        kg = KnowledgeGraph()
        status = loop.crystallize_lesson(lesson, knowledge_graph=kg)
        assert status["skill_saved"] is True
        assert status["knowledge_fact_added"] is True


class TestToolRegistrySelfImprovementIntegration:
    def test_self_improvement_tools_in_definitions(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        defs = tools.get_tool_definitions()
        names = [d.get("function", {}).get("name") for d in defs]
        assert "self_improve_analyze_failure" in names
        assert "self_improve_eval_run" in names
        assert "self_improve_crystallize_lesson" in names

    def test_tool_self_improve_analyze_failure(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        res = tools.tool_self_improve_analyze_failure(
            task_id="task_demo_err",
            prompt="Update config",
            failure_log="SyntaxError: invalid token",
            category="coding",
        )
        assert "FAILURE ANALYSIS & LESSON LEARNED" in res
        assert "task_demo_err" in res
        assert "Extracted Rule" in res

    def test_tool_self_improve_eval_run_reports_not_run_without_evaluator(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        res = tools.tool_self_improve_eval_run()
        assert "NOT RUN" in res
        assert "no live evaluation runner is configured" in res
        assert "100.0%" not in res
        assert "Average Quality Score" not in res

    def test_tool_self_improve_crystallize_lesson(self, temp_workspace: Path):
        tools = ToolRegistry(temp_workspace)
        res = tools.tool_self_improve_crystallize_lesson(
            lesson_title="Always run linter before commit",
            guidance="Run ruff check before staging git changes.",
            category="quality",
        )
        assert "LESSON CRYSTALLIZED SUCCESSFULLY" in res
        assert "Saved as Skill Playbook" in res
