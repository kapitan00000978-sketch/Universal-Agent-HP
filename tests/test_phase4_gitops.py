"""Phase 4 — Git-first (Aider-style) and core-memory integration tests."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

GIT = shutil.which("git")
if GIT is None:
    pytest.skip("git executable not found — skipping Phase 4 git tests", allow_module_level=True)

from titan_agent.agent import TitanAgent
from titan_agent.core.memory.memory_system import MemorySystem
from titan_agent.gitops import (
    auto_commit,
    find_repo_root,
    git_commit,
    git_diff,
    git_status,
)
from titan_agent.llm_client import LLMResponse


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=False
    )
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def _init_repo(tmp_path: Path, configure: bool = True) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    if configure:
        _git(repo, "config", "user.email", "a@b.c")
        _git(repo, "config", "user.name", "Test Agent")
    return repo


class FakeLLM:
    async def chat_completion(self, messages, tools=None):
        return LLMResponse(content="Done: yes.")


# ---------- gitops primitives ----------


def test_find_repo_root_detects_from_nested_dir(tmp_path):
    repo = _init_repo(tmp_path)
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == repo.resolve()
    assert find_repo_root(repo) == repo.resolve()


def test_find_repo_root_returns_none_for_plain_dir(tmp_path):
    assert find_repo_root(tmp_path / "no_git_here") is None


def test_git_status_clean_then_dirty(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("one")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    assert git_status(repo) == "(working tree clean)"

    (repo / "a.txt").write_text("two")
    (repo / "b.txt").write_text("new")
    status = git_status(repo)
    assert "a.txt" in status  # modified
    assert "b.txt" in status  # untracked


def test_git_diff_shows_untracked_hint(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "b.txt").write_text("new")
    out = git_diff(repo)
    assert "b.txt" in out


def test_git_commit_stages_and_commits(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("one")
    (repo / "b.txt").write_text("new")
    res = git_commit(repo, "feat: add files")
    assert res.startswith("Committed: feat: add files")
    assert git_status(repo) == "(working tree clean)"
    log = _git(repo, "log", "--oneline", "-1")
    assert "feat: add files" in log


def test_git_commit_returns_repo_num_message(tmp_path):
    repo = _init_repo(tmp_path)
    res = git_commit(repo, "nothing here")
    assert "nothing to commit" in res


def test_git_commit_sets_local_author_when_missing(tmp_path, monkeypatch):
    # Isolate from any global git identity so the local-author fallback is exercised.
    empty = tmp_path / "empty_config"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    repo = _init_repo(tmp_path, configure=False)
    (repo / "a.txt").write_text("one")
    res = git_commit(repo, "init work")
    assert res.startswith("Committed:")
    author = _git(repo, "log", "-1", "--format=%an <%ae>")
    assert "Titan Agent <titan@localhost>" in author


def test_git_status_not_a_repo(tmp_path):
    assert "not a git repository" in git_status(tmp_path)


def test_auto_commit_derives_subject(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("one")
    res = auto_commit(repo, "update the readme file")
    assert "agent: update the readme" in res


# ---------- agent tool integration ----------


def test_agent_tool_catalog_includes_git_tools():
    agent = TitanAgent(llm=FakeLLM())
    names = {t["function"]["name"] for t in agent._build_tools_list()}
    assert {"git_status", "git_diff", "git_commit"} <= names


def test_dispatch_git_tools_through_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_FULL_ACCESS", "true")
    repo = _init_repo(tmp_path)
    (repo / "build.txt").write_text("v2")
    agent = TitanAgent(llm=FakeLLM(), git_root=repo)

    async def _run():
        out_status = await agent.execute_tool_unified("git_status", {})
        out_commit = await agent.execute_tool_unified("git_commit", {"message": "chore: build"})
        out_final = await agent.execute_tool_unified("git_status", {})
        return out_status, out_commit, out_final

    out_status, out_commit, out_final = asyncio.run(_run())
    assert "build.txt" in out_status
    assert out_commit.startswith("Committed: chore: build")
    assert "working tree clean" in out_final


def test_git_commit_tool_requires_message(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_FULL_ACCESS", "true")
    repo = _init_repo(tmp_path)
    agent = TitanAgent(llm=FakeLLM(), git_root=repo)
    out = asyncio.run(agent.execute_tool_unified("git_commit", {}))
    assert "requires a 'message'" in out


# ---------- end-to-end: auto-commit + core memory ----------


def test_run_task_auto_commit_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_FULL_ACCESS", "true")
    repo = _init_repo(tmp_path)
    (repo / "build.txt").write_text("v1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    (repo / "build.txt").write_text("v2")  # dirty workspace

    agent = TitanAgent(
        llm=FakeLLM(),
        git_root=repo,
        auto_commit=True,
        core_memory_path=tmp_path / "core.db",
    )

    async def _run():
        return [ev async for ev in agent.run_task("update the build file", session_id="web", mode="fast")]

    events = asyncio.run(_run())
    assert any(ev.type == "final_answer" for ev in events)
    assert "agent: update the build file" in _git(repo, "log", "--oneline", "-1")
    assert git_status(repo) == "(working tree clean)"


def test_auto_commit_fails_closed_without_hitl(tmp_path, monkeypatch):
    monkeypatch.delenv("TITAN_FULL_ACCESS", raising=False)
    monkeypatch.delenv("TITAN_ABSOLUTE_ACCESS", raising=False)
    repo = _init_repo(tmp_path)
    (repo / "build.txt").write_text("v1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")
    (repo / "build.txt").write_text("v2")

    agent = TitanAgent(llm=FakeLLM(), git_root=repo, auto_commit=True)

    async def _run():
        return [ev async for ev in agent.run_task("update the build file", session_id="auto-no-hitl", mode="fast")]

    asyncio.run(_run())
    assert "seed" in _git(repo, "log", "--oneline", "-1")
    assert "build.txt" in git_status(repo)


def test_run_task_writes_core_memory_and_recalls(tmp_path):
    db_path = tmp_path / "core.db"
    agent = TitanAgent(llm=FakeLLM(), core_memory_path=db_path)

    async def _run():
        return [ev async for ev in agent.run_task("fix the login flow", session_id="s7", mode="fast")]

    events = asyncio.run(_run())
    assert any(ev.type == "final_answer" for ev in events)

    # Episodic record persisted and recallable by the same store
    store = MemorySystem(db_path)
    hits = store.recall("login flow", limit=10)
    assert hits, "no memories recorded after a completed run"
    assert any(r.content.startswith("task[") for r in hits)
    assert any("login flow" in r.content for r in hits)

    # Live agent pulls past runs back into context after (only when records exist)
    block = agent._core_recall_block("login flow")
    assert block.startswith("\n\n### PAST RUNS & LESSONS")
    assert "login flow" in block


def test_core_recall_block_empty_for_fresh_store(tmp_path):
    agent = TitanAgent(llm=FakeLLM(), core_memory_path=tmp_path / "fresh.db")
    assert agent._core_recall_block("anything") == ""