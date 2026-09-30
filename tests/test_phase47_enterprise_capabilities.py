import pytest
from pathlib import Path
from titan_agent.tools import ToolRegistry
from titan_agent.agent import TitanAgent


@pytest.mark.asyncio
async def test_tool_registry_mcp_presets(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    # 1. List presets
    presets_output = registry.tool_mcp_list_presets()
    assert "AVAILABLE 1-LINE MCP PRESETS" in presets_output
    assert "postgres" in presets_output
    assert "github" in presets_output
    assert "slack" in presets_output

    # 2. Connect preset configuration
    res = await registry.tool_mcp_connect_preset(
        preset_id="postgres",
        server_name="pg_test",
        env_overrides={"POSTGRES_URL": "postgresql://localhost:5432/test"},
    )
    assert "Successfully configured and saved MCP preset 'pg_test'" in res
    assert (tmp_path / "mcp_servers.json").exists()
    # Standalone registries can only save a credential-free template; they
    # must not claim a live connection when no MCP manager is attached.
    assert "not connected" in res
    assert "postgresql://localhost:5432/test" not in (tmp_path / "mcp_servers.json").read_text()


@pytest.mark.asyncio
async def test_agent_wires_preset_connect_to_live_mcp_manager(tmp_path):
    class FakeMCPManager:
        def __init__(self):
            self.calls = []

        async def connect_preset(self, **kwargs):
            self.calls.append(kwargs)
            return True, "connected in test"

    manager = FakeMCPManager()
    registry = ToolRegistry(workspace=tmp_path)
    _agent = TitanAgent(tools=registry, mcp=manager)

    result = await registry.tool_mcp_connect_preset(
        preset_id="postgres",
        server_name="pg_live",
        env_overrides={"POSTGRES_URL": "{POSTGRES_URL}"},
    )

    assert result == "connected in test"
    assert manager.calls[0]["preset_id"] == "postgres"
    assert manager.calls[0]["server_name"] == "pg_live"
    assert manager.calls[0]["env_overrides"] == {"POSTGRES_URL": "{POSTGRES_URL}"}


@pytest.mark.asyncio
async def test_tool_registry_hitl_approval_prompt(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    # Generate approval prompt for sensitive action
    prompt_res = await registry.tool_hitl_request_approval(
        action="execute_command",
        resource="git",
        reason="git push origin main --force",
    )
    assert "Do you authorize this? [Yes / No]" in prompt_res


def test_tool_registry_git_branch_and_pr(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "agent@test.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=str(repo), check=True)
    (repo / "README.md").write_text("# Test\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), check=True)

    registry = ToolRegistry(workspace=repo)

    # Branch creation
    branch_msg = registry.tool_git_create_branch("feature user auth login")
    assert "Successfully created and checked out branch" in branch_msg
    assert "agent/feature-feature-user-auth-login" in branch_msg


def test_tool_registry_semantic_cache(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    # Query miss
    miss = registry.tool_semantic_cache_query("How to implement binary search in Python?")
    assert "Semantic Cache Miss" in miss

    # Seed cache
    registry._semantic_cache.set(
        "How to implement binary search in Python?",
        "Use bisect module or while low <= high with mid = (low + high) // 2.",
        estimated_tokens=80,
    )

    # Query hit
    hit = registry.tool_semantic_cache_query("How to implement binary search in Python?")
    assert "SEMANTIC CACHE HIT" in hit
    assert "bisect module" in hit

    stats = registry.tool_semantic_cache_stats()
    assert "SEMANTIC CACHE STATS" in stats


def test_tool_registry_experience_replay(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    # Query miss
    miss = registry.tool_experience_replay_query("ModuleNotFoundError: No module named 'jwt'")
    assert "No prior experience found" in miss

    # Record experience
    rec = registry.tool_experience_replay_record(
        error_text="ModuleNotFoundError: No module named 'jwt'",
        resolution="Run 'pip install PyJWT' to resolve import.",
        diagnosis="Package is PyJWT on PyPI, not jwt.",
    )
    assert "Successfully recorded experience episode" in rec

    # Query hit
    hit = registry.tool_experience_replay_query("ModuleNotFoundError: No module named 'jwt'")
    assert "EPISODIC EXPERIENCE MATCH" in hit
    assert "pip install PyJWT" in hit
