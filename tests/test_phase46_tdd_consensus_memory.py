import pytest
from pathlib import Path
from titan_agent.tools import ToolRegistry
from titan_agent.agent import TitanAgent


@pytest.mark.asyncio
async def test_tool_registry_tdd_cycle(tmp_path, monkeypatch):
    # Trusted fixture exercises the explicit FULL_ACCESS host-execution path.
    monkeypatch.setenv("TITAN_FULL_ACCESS", "1")
    registry = ToolRegistry(workspace=tmp_path)

    test_code = """
def test_square():
    from feature import square
    assert square(4) == 16
"""
    impl_code = """
def square(x: int) -> int:
    return x * x
"""
    result = await registry.execute_tool(
        "tdd_cycle",
        {
            "test_code": test_code,
            "implementation_code": impl_code,
            "test_filename": "test_sq.py",
            "code_filename": "feature.py",
        },
    )

    assert "AUTONOMOUS TDD CYCLE: ✅ PASSED" in result
    assert "RED Phase" in result
    assert "GREEN Phase" in result
    assert "REFACTOR Phase" in result


def test_tool_registry_consensus_deliberation(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    result = registry.tool_consensus_deliberation(
        proposal="Safely migrate user configuration to SQLite with parameterized queries.",
        context="Local single-process daemon.",
    )

    assert "FORMAL CONSENSUS MEMO" in result
    assert "Architect" in result
    assert "SecurityOfficer" in result
    assert "Pragmatist" in result


def test_tool_registry_working_memory_update(tmp_path):
    registry = ToolRegistry(workspace=tmp_path)

    result = registry.tool_working_memory_update(
        confirmed_fact="PostgreSQL port 5432 is responding",
        dead_end="Do not use default postgres password",
        subtask="Apply alembic migrations",
        todo="Run integration tests",
    )

    assert "### WORKING MEMORY HUD" in result
    assert "PostgreSQL port 5432 is responding" in result
    assert "Do not use default postgres password" in result
    assert "Apply alembic migrations" in result
    assert "Run integration tests" in result


def test_titan_agent_working_memory_hud_integration(tmp_path):
    agent = TitanAgent(tools=ToolRegistry(workspace=tmp_path))
    messages = [
        {"role": "system", "content": "You are Titan Agent."},
        {"role": "user", "content": "Check database status"},
    ]

    # Confirm facts and render HUD
    agent._working_memory.set_goal("Database migration")
    agent._working_memory.set_subtask("Check schema version")
    agent._working_memory.confirm_fact("Schema version is v2.1")
    agent._inject_or_update_working_memory_hud(messages)

    # Check that HUD was injected
    hud_msg = next((m for m in messages if "### WORKING MEMORY HUD" in m.get("content", "")), None)
    assert hud_msg is not None
    assert "Database migration" in hud_msg["content"]
    assert "Schema version is v2.1" in hud_msg["content"]

    # Updating and re-injecting updates in-place without duplicating
    agent._working_memory.confirm_fact("Schema version updated to v2.2")
    agent._inject_or_update_working_memory_hud(messages)
    hud_msgs = [m for m in messages if "### WORKING MEMORY HUD" in m.get("content", "")]
    assert len(hud_msgs) == 1
    assert "Schema version updated to v2.2" in hud_msgs[0]["content"]
