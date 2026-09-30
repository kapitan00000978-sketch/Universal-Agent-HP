"""Extended MCP config tests (Block 5: Hermes-class MCP server baseline)."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from titan_agent.config import MCP_CONFIG_FILE
from titan_agent.mcp_client import MCPManager


def test_extended_servers_present_in_config():
    cfg = json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
    servers = cfg.get("mcpServers", {})
    for needed in ("filesystem", "memory", "sequential-thinking", "everything",
                   "github", "fetch", "context7", "chrome-devtools", "obsidian"):
        assert needed in servers, f"missing MCP server: {needed}"


def test_every_server_has_command_and_args():
    cfg = json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
    for name, details in cfg.get("mcpServers", {}).items():
        assert details.get("command"), name
        assert isinstance(details.get("args"), list), name


def test_env_placeholder_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_123")
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps({
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "{GITHUB_TOKEN}"}
            }
        }
    }), encoding="utf-8")
    mgr = MCPManager(cfg_file)
    details = mgr.load_config()["mcpServers"]["github"]
    cmd, _args, env = mgr._resolve_server_command(details, tmp_path)
    assert cmd == "npx"
    assert env["GITHUB_TOKEN"] == "ghp_test_token_123"


def test_missing_env_var_becomes_empty_not_crash(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps({
        "mcpServers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": "{GITHUB_TOKEN}"}
            }
        }
    }), encoding="utf-8")
    mgr = MCPManager(cfg_file)
    details = mgr.load_config()["mcpServers"]["github"]
    _, _, env = mgr._resolve_server_command(details, tmp_path)
    assert env["GITHUB_TOKEN"] == ""
    assert mgr.servers == {}  # nothing started, nothing crashed


def test_arg_env_placeholder_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT", r"C:\Users\me\Documents\MyVault")
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps({
        "mcpServers": {
            "obsidian": {
                "command": "npx",
                "args": ["-y", "obsidian-mcp@2", "serve", "--vault", "notes={OBSIDIAN_VAULT}"]
            }
        }
    }), encoding="utf-8")
    mgr = MCPManager(cfg_file)
    details = mgr.load_config()["mcpServers"]["obsidian"]
    cmd, args, _ = mgr._resolve_server_command(details, tmp_path)
    assert cmd == "npx"
    assert args == ["-y", "obsidian-mcp@2", "serve", "--vault", r"notes=C:\Users\me\Documents\MyVault"]


def test_missing_arg_env_var_becomes_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("OBSIDIAN_VAULT", raising=False)
    cfg_file = tmp_path / "mcp.json"
    cfg_file.write_text(json.dumps({
        "mcpServers": {
            "obsidian": {
                "command": "npx",
                "args": ["-y", "obsidian-mcp@2", "serve", "--vault", "notes={OBSIDIAN_VAULT}"]
            }
        }
    }), encoding="utf-8")
    mgr = MCPManager(cfg_file)
    details = mgr.load_config()["mcpServers"]["obsidian"]
    _, args, _ = mgr._resolve_server_command(details, tmp_path)
    assert args[-1] == "notes="


def test_obsidian_config_uses_env_placeholder_for_vault():
    cfg = json.loads(MCP_CONFIG_FILE.read_text(encoding="utf-8"))
    obs = cfg["mcpServers"]["obsidian"]
    assert obs["command"] == "npx"
    assert "--vault" in obs["args"]
    assert any("OBSIDIAN_VAULT" in a for a in obs["args"])


def test_dynamic_preset_connect_uses_live_manager_and_does_not_persist_secret(tmp_path, monkeypatch):
    secret_url = "postgresql://private-user:private-password@db.invalid/private"
    monkeypatch.setenv("POSTGRES_URL", secret_url)
    config_file = tmp_path / "mcp_servers.json"
    mgr = MCPManager(config_file)

    async def fake_start_one(conn):
        conn.is_connected = True
        conn.tools = [{"name": "query", "inputSchema": {"type": "object"}}]
        mgr.servers[conn.name] = conn
        return True

    monkeypatch.setattr(mgr, "_start_one", fake_start_one)

    ok, message = asyncio.run(
        mgr.connect_preset(
            "postgres",
            server_name="pg_live",
            env_overrides={"POSTGRES_URL": "{POSTGRES_URL}"},
            workspace_dir=tmp_path,
        )
    )

    assert ok is True
    assert "connected" in message
    assert "pg_live" in mgr.servers
    saved_config = config_file.read_text(encoding="utf-8")
    assert secret_url not in saved_config
    assert "private-password" not in saved_config


def test_dynamic_preset_refuses_inline_credentials(tmp_path, monkeypatch):
    mgr = MCPManager(tmp_path / "mcp_servers.json")
    called = False

    async def should_not_start(_conn):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(mgr, "_start_one", should_not_start)
    ok, message = asyncio.run(
        mgr.connect_preset(
            "postgres",
            env_overrides={"POSTGRES_URL": "postgresql://user:secret@localhost/db"},
            workspace_dir=tmp_path,
        )
    )

    assert ok is False
    assert "Refusing inline value" in message
    assert called is False
    assert not (tmp_path / "mcp_servers.json").exists()