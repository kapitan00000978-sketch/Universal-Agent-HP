"""
Autonomous Git Branching & Pull Request (PR) Engine.

Enforces corporate GitOps best practices:
1. Isolates feature/bugfix development into dedicated branches (agent/feature-*)
2. Verifies test suite passes cleanly before committing
3. Formats professional Pull Requests with changelog and test results
4. Automatically opens PR on GitHub using the gh CLI with fallback.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from titan_agent.gitops import _run, find_repo_root, git_commit

log = logging.getLogger(__name__)


@dataclass
class PullRequestResult:
    """Outcome of automated Pull Request creation."""

    success: bool
    branch: str
    base_branch: str
    pr_url: str
    message: str
    pr_body: str


class GitPREngine:
    """Manages autonomous feature branches, test-gated commits, and PR creation."""

    def __init__(self, workspace_root: Path | str | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

    def _get_repo(self) -> tuple[Path | None, str]:
        root = find_repo_root(self.workspace_root)
        if not root:
            return None, "Not a valid git repository (no .git folder found)."
        return root, ""

    def get_current_branch(self) -> str:
        root, err = self._get_repo()
        if not root:
            return ""
        code, out = _run(root, "branch", "--show-current")
        return out.strip() if code == 0 else ""

    def create_feature_branch(self, task_name: str, prefix: str = "agent/feature-") -> tuple[bool, str]:
        """Creates and switches to a dedicated feature branch for the task."""
        root, err = self._get_repo()
        if not root:
            return False, err

        # Slugify task name (e.g., 'Add OAuth2 login support' -> 'oauth2-login-support')
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", task_name.lower()).strip("-")[:36]
        if not slug:
            slug = "task-work"
        branch_name = f"{prefix}{slug}"

        # Check if branch exists
        code, out = _run(root, "checkout", "-b", branch_name)
        if code != 0:
            # If already exists, switch to it
            code, out = _run(root, "checkout", branch_name)
            if code != 0:
                return False, f"Failed to switch to branch '{branch_name}': {out}"

        return True, f"Successfully created and checked out branch: '{branch_name}'"

    def run_tests_and_commit(
        self,
        commit_message: str,
        test_command: str = "python -m pytest tests/ -q",
    ) -> tuple[bool, str]:
        """Runs test command to guarantee stability before committing."""
        root, err = self._get_repo()
        if not root:
            return False, err

        # Run test verification
        if test_command:
            try:
                proc = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=180.0,
                )
                if proc.returncode != 0:
                    err_snippet = (proc.stderr or proc.stdout)[-400:]
                    return False, f"Tests failed before commit. Aborting commit.\nErrors:\n{err_snippet}"
            except Exception as exc:
                return False, f"Test execution failed: {exc!s}"

        # Commit changes
        commit_res = git_commit(root, commit_message, add_all=True)
        return True, commit_res

    def create_pull_request(
        self,
        title: str,
        body: str = "",
        base_branch: str = "main",
        draft: bool = False,
    ) -> PullRequestResult:
        """Pushes branch to remote origin and opens a GitHub Pull Request."""
        root, err = self._get_repo()
        if not root:
            return PullRequestResult(False, "", base_branch, "", err, body)

        current_branch = self.get_current_branch()
        if not current_branch or current_branch == base_branch:
            return PullRequestResult(
                False,
                current_branch,
                base_branch,
                "",
                f"Cannot create PR from base branch '{base_branch}'. Please create a feature branch first.",
                body,
            )

        # 1. Push branch to origin
        push_code, push_out = _run(root, "push", "-u", "origin", current_branch)
        if push_code != 0:
            return PullRequestResult(
                False,
                current_branch,
                base_branch,
                "",
                f"Failed to push branch '{current_branch}' to origin: {push_out}",
                body,
            )

        # 2. Build a truthful PR body if none provided. This method pushes and
        # opens the PR, but does not run tests or the symbolic checker itself.
        if not body:
            body = (
                f"## 🤖 Automated Pull Request: {title}\n\n"
                f"### Description\nAutonomous changes generated by Titan Agent on branch `{current_branch}`.\n\n"
                f"### Verification\n"
                f"- [ ] Tests: not run by the PR engine; add verified results before merge.\n"
                f"- [ ] Symbolic safety checks: not run by the PR engine.\n"
            )

        # 3. Use gh CLI if available
        gh_bin = shutil.which("gh")
        if gh_bin:
            cmd = [gh_bin, "pr", "create", "--base", base_branch, "--title", title, "--body", body]
            if draft:
                cmd.append("--draft")

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                )
                if proc.returncode == 0:
                    pr_url = proc.stdout.strip()
                    return PullRequestResult(
                        True,
                        current_branch,
                        base_branch,
                        pr_url,
                        f"Pull Request successfully opened: {pr_url}",
                        body,
                    )
                else:
                    return PullRequestResult(
                        False,
                        current_branch,
                        base_branch,
                        "",
                        f"Branch pushed to origin, but Pull Request creation failed: {proc.stderr.strip() or proc.stdout.strip() or 'gh returned a non-zero exit code.'}",
                        body,
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("gh CLI call failed: %s", exc)
                return PullRequestResult(
                    False,
                    current_branch,
                    base_branch,
                    "",
                    f"Branch pushed to origin, but Pull Request creation failed: {exc!s}",
                    body,
                )

        return PullRequestResult(
            False,
            current_branch,
            base_branch,
            "",
            f"Branch '{current_branch}' was pushed to origin, but no Pull Request was created because the GitHub CLI ('gh') is not installed. Install gh and open the PR manually.",
            body,
        )
