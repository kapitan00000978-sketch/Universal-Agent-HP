"""
Phase 16 — Docker packaging (deterministic, no daemon needed).

Validates the Dockerfile / compose recipe structurally: image base, exposed
port, uvicorn command, .dockerignore hygiene (no .env/workspace baked in), and
that the smoke-import target exists. Building the image is a manual step.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_has_valid_base_and_entrypoint() -> None:
    dockerfile = _read("Dockerfile")
    assert "FROM python:3.12-slim" in dockerfile
    assert 'EXPOSE 7860' in dockerfile
    assert 'CMD ["uvicorn", "titan_agent.server:app"' in dockerfile
    # Smoke import must gate the build (catches broken server wiring early).
    assert "import titan_agent.server" in dockerfile


def test_dockerfile_installs_system_git_and_mcp_node_runtime() -> None:
    dockerfile = _read("Dockerfile")
    assert "git" in dockerfile
    assert "ca-certificates" in dockerfile
    assert "nodejs" in dockerfile
    assert "npm" in dockerfile
    assert "uv>=" in _read("requirements.txt")


def test_compose_mounts_workspace_and_healthchecks() -> None:
    compose = _read("docker-compose.yml")
    assert "titan-workspace:/app/workspace" in compose
    assert "./mcp_servers.json:/app/mcp_servers.json:ro" in compose
    assert "healthcheck" in compose
    assert "/health" in compose
    assert 'restart: unless-stopped' in compose


def test_dockerignore_never_bakes_secrets_or_workspace() -> None:
    ignore = _read(".dockerignore")
    for secret in (".env", "workspace/", ".server_key", "*.log"):
        assert secret in ignore, f".dockerignore must exclude {secret}"
    # The shipable config template must NOT be excluded.
    assert "!.env.example" in ignore


def test_server_module_imports_cleanly() -> None:
    """The exact smoke check the Dockerfile runs — must pass offline."""
    import titan_agent.server  # noqa: F401 - the import itself is the assertion


def test_ui_static_dir_exists_for_container() -> None:
    assert (ROOT / "titan_agent" / "web_ui").is_dir()