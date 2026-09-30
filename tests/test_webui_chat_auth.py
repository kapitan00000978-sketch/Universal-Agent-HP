from pathlib import Path
import re


APP_JS = Path(__file__).resolve().parents[1] / "titan_agent" / "web_ui" / "app.js"


def test_chat_stream_uses_bearer_authenticated_fetch_wrapper():
    source = APP_JS.read_text(encoding="utf-8")
    send_message = re.search(
        r"async function sendMessage\(prompt\) \{(?P<body>.*?)\n\}\n\nasync function sendPuterMessage",
        source,
        flags=re.DOTALL,
    )
    assert send_message is not None
    body = send_message.group("body")

    assert 'apiFetch("/api/chat/stream"' in body
    assert 'fetch("/api/chat/stream"' not in body
    assert "if (!response.ok)" in body
    assert "response.body.getReader()" in body


def test_api_fetch_attaches_stored_bearer_token():
    source = APP_JS.read_text(encoding="utf-8")
    assert 'headers["Authorization"] = `Bearer ${token}`' in source
    assert "const response = await apiFetch(\"/api/chat/stream\"" in source


def test_dashboard_api_calls_use_authenticated_wrapper():
    source = APP_JS.read_text(encoding="utf-8")
    assert not re.search(r"\bfetch\(\s*[\"']/api/", source)
    assert 'apiFetch("/api/mcp/tools")' in source
    assert "connected}/${configured} MCP servers connected" in source
