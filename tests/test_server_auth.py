"""
Phase 12 — Server authentication (Bearer token on every /api route).

Every non-public route requires `Authorization: Bearer <TITAN_API_KEY>`.
The liveness check (`/health`) and the web-UI shell (`/`) stay public.

We drive the live FastAPI app through its own middleware/stack via
``fastapi.testclient.TestClient`` (httpx), which means these tests exercise the
real dependency tree — including the in-place ``_apply_auth_dependency()``
route patch — exactly as a browser or curl client would.
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Deterministic key for this test process. ``auth.get_server_api_key`` reads the
# env var on every request, so setting it here makes the 401/200 checks stable
# without needing to know or mock the persisted workspace key.
_TEST_KEY = "sk-test-" + uuid.uuid4().hex
os.environ["TITAN_API_KEY"] = _TEST_KEY

from titan_agent.server import app  # (must import after env setup)

PUBLIC_PATHS = {"/", "/health", "/metrics", "/api/metrics"}

# A representative sample of protected routes, one per auth surface (GET, POST,
# and the SSE/streaming endpoint both are APIRoutes and must be guarded).
_PROTECTED_SAMPLES = [
    ("GET", "/api/config"),
    ("GET", "/api/token-usage"),
    ("POST", "/api/queue/tasks"),
    ("GET", "/api/cron/jobs"),
    ("POST", "/api/chat/stream"),
    ("GET", "/api/mcp/tools"),
    ("GET", "/api/traces/recent"),
]


@pytest.fixture()
def client() -> TestClient:
    # Not used as a context manager on purpose: TestClient.__enter__ drives the
    # app lifespan (starts MCP servers + cron daemon) which we do not want in a
    # unit test. Without the context manager no lifespan code runs.
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    # Multiple server test modules (e.g. test_server_hitl) also set TITAN_API_KEY
    # at import time, and the LAST import wins the env var. Reading it at call
    # time keeps every module consistent regardless of pytest collection order.
    return {"Authorization": f"Bearer {os.environ.get('TITAN_API_KEY', _TEST_KEY)}"}


# ---------------------------------------------------------------------------
# /health is public — the only liveness check that never needs a token.
# ---------------------------------------------------------------------------

def test_health_public(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_mcp_status_lists_configured_servers_even_when_offline(client: TestClient) -> None:
    res = client.get("/api/mcp/tools", headers=_auth_headers())
    assert res.status_code == 200
    data = res.json()
    assert "servers" in data
    assert "configured_count" in data
    assert "connected_count" in data
    assert isinstance(data["servers"], dict)


def test_root_public_shell(client: TestClient) -> None:
    # The web-UI shell must stay reachable without a token so a browser can load
    # the page and only then be asked for the key (the UI prompts via JS).
    res = client.get("/")
    assert res.status_code in (200, 404)  # 404 when web_ui/index.html absent


# ---------------------------------------------------------------------------
# Every protected route rejects missing / wrong tokens with 401.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", _PROTECTED_SAMPLES)
def test_missing_token_rejected(client: TestClient, method: str, path: str) -> None:
    res = client.request(method, path)
    assert res.status_code == 401, f"{method} {path} should 401 (no token)"
    assert "WWW-Authenticate" in res.headers


@pytest.mark.parametrize("method,path", _PROTECTED_SAMPLES)
def test_wrong_token_rejected(client: TestClient, method: str, path: str) -> None:
    res = client.request(
        method, path, headers={"Authorization": "Bearer sk-titan-wrong-token"}
    )
    assert res.status_code == 401, f"{method} {path} should 401 (bad token)"


@pytest.mark.parametrize("method,path", _PROTECTED_SAMPLES)
def test_valid_token_reaches_handler(client: TestClient, method: str, path: str) -> None:
    res = client.request(method, path, headers=_auth_headers())
    # We are not asserting 200 (some handlers 404/400 on bad input) — we assert
    # the request got PAST the auth gate, i.e. never 401/500-shape auth errors.
    assert res.status_code != 401
    assert res.status_code not in (500,), f"{method} {path} must not crash with auth on"


# ---------------------------------------------------------------------------
# The auth dependency is attached to routes, not just the app-level lifespan.
# ---------------------------------------------------------------------------

def test_every_protected_route_has_auth_dependency() -> None:
    from fastapi.routing import APIRoute

    from titan_agent.auth import require_api_key

    guarded = [r for r in app.routes if isinstance(r, APIRoute) and r.path not in PUBLIC_PATHS]
    assert guarded, "expected at least one protected API route"

    for route in guarded:
        resolved = {
            getattr(d, "dependency", None) or getattr(d, "call", None)
            for d in route.dependant.dependencies
        }
        assert require_api_key in resolved, f"{route.path} missing require_api_key dep"


def test_public_routes_have_no_auth_dependency() -> None:
    from fastapi.routing import APIRoute

    from titan_agent.auth import require_api_key

    for path in PUBLIC_PATHS:
        route = next(r for r in app.routes if isinstance(r, APIRoute) and r.path == path)
        resolved = {
            getattr(d, "dependency", None) or getattr(d, "call", None)
            for d in route.dependant.dependencies
        }
        assert require_api_key not in resolved, f"{path} must stay public"
