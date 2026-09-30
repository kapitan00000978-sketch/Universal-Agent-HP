import subprocess
from pathlib import Path

import pytest

from titan_agent.core.git.pr_engine import GitPREngine


def _init_repo(path: Path) -> Path:
    repo = path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "agent@test.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=str(repo), check=True)
    # create initial commit
    init_file = repo / "README.md"
    init_file.write_text("# Test Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(repo), check=True)
    return repo


def test_git_pr_engine_branch_creation(tmp_path):
    repo = _init_repo(tmp_path)
    engine = GitPREngine(workspace_root=repo)

    ok, msg = engine.create_feature_branch("Add OAuth2 Google Login", prefix="agent/feature-")
    assert ok is True
    assert "agent/feature-add-oauth2-google-login" in msg

    curr = engine.get_current_branch()
    assert curr == "agent/feature-add-oauth2-google-login"


def test_git_pr_engine_test_gated_commit(tmp_path):
    repo = _init_repo(tmp_path)
    engine = GitPREngine(workspace_root=repo)
    engine.create_feature_branch("Auth fix")

    # Add code change
    (repo / "auth.py").write_text("def auth(): return True\n", encoding="utf-8")

    # Passing test check (python -c "exit(0)")
    ok, msg = engine.run_tests_and_commit(
        commit_message="feat: add auth helper",
        test_command='python -c "exit(0)"',
    )
    assert ok is True
    assert "Committed" in msg

    # Failing test check prevents commit
    (repo / "bug.py").write_text("syntax bug\n", encoding="utf-8")
    ok2, msg2 = engine.run_tests_and_commit(
        commit_message="buggy commit",
        test_command='python -c "exit(1)"',
    )
    assert ok2 is False
    assert "Tests failed before commit" in msg2


def _mock_push(monkeypatch, engine, repo):
    monkeypatch.setattr(engine, "_get_repo", lambda: (repo, ""))
    monkeypatch.setattr(engine, "get_current_branch", lambda: "agent/feature-test")
    monkeypatch.setattr(
        "titan_agent.core.git.pr_engine._run",
        lambda *args, **kwargs: (0, "pushed"),
    )


def test_pr_body_does_not_claim_unrun_checks_passed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    engine = GitPREngine(workspace_root=repo)
    _mock_push(monkeypatch, engine, repo)
    monkeypatch.setattr("titan_agent.core.git.pr_engine.shutil.which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        "titan_agent.core.git.pr_engine.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "https://example.test/pr/1\n", ""),
    )

    result = engine.create_pull_request("Example change")

    assert result.success is True
    assert result.pr_url == "https://example.test/pr/1"
    assert "not run by the PR engine" in result.pr_body
    assert "100% passing" not in result.pr_body


def test_pr_creation_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    engine = GitPREngine(workspace_root=repo)
    _mock_push(monkeypatch, engine, repo)
    monkeypatch.setattr("titan_agent.core.git.pr_engine.shutil.which", lambda _: "/usr/bin/gh")
    monkeypatch.setattr(
        "titan_agent.core.git.pr_engine.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "permission denied"),
    )

    result = engine.create_pull_request("Example change")

    assert result.success is False
    assert result.pr_url == ""
    assert "Branch pushed to origin" in result.message
    assert "creation failed" in result.message


def test_missing_gh_does_not_return_fake_pr_url(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    engine = GitPREngine(workspace_root=repo)
    _mock_push(monkeypatch, engine, repo)
    monkeypatch.setattr("titan_agent.core.git.pr_engine.shutil.which", lambda _: None)

    result = engine.create_pull_request("Example change")

    assert result.success is False
    assert result.pr_url == ""
    assert "no Pull Request was created" in result.message
