"""Telegram account manager tests (Block 6: consent-gated Telegram control).

Deterministic — never touches the network. The consent gate, allowlist parsing
and PII sanitizers are tested via module-level constant monkeypatching.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import titan_agent.telegram as tg_mod
from titan_agent.telegram import (
    TelegramError,
    TelegramManager,
    _mask_phone,
    _mask_username,
)


def _manager(tmp_path, enabled=True, allowlist=""):
    m = TelegramManager(session_dir=tmp_path / "sessions")
    # Patch the module-level config the manager reads at call time.
    tg_mod.TITAN_TELEGRAM_ENABLED = enabled
    tg_mod.TITAN_TELEGRAM_API_ID = "12345"
    tg_mod.TITAN_TELEGRAM_API_HASH = "abcdef0123456789abcdef0123456789"
    tg_mod.TITAN_TELEGRAM_SEND_ALLOWLIST = allowlist
    return m


def _clear():
    tg_mod.TITAN_TELEGRAM_ENABLED = False
    tg_mod.TITAN_TELEGRAM_API_ID = ""
    tg_mod.TITAN_TELEGRAM_API_HASH = ""
    tg_mod.TITAN_TELEGRAM_SEND_ALLOWLIST = ""


def test_mask_phone_hides_middle(tmp_path):
    assert _mask_phone("+998901234567") == "+998 ** *** ** 67"


def test_mask_phone_short(tmp_path):
    assert _mask_phone("12") == "***"


def test_mask_username(tmp_path):
    out = _mask_username("titan_user")
    assert out != "titan_user"
    assert "*" in out
    assert _mask_username("") == "(no username)"


def test_disabled_gate_blocks_everything(tmp_path):
    _clear()
    m = TelegramManager(session_dir=tmp_path / "s")
    assert m.status()["enabled"] is False
    for fn in (
        lambda: m.list_accounts(),
        lambda: asyncio.run(m.login_start("x", "+998901234567")),
        lambda: asyncio.run(m.send_message("x", "someone", "hi")),
        lambda: asyncio.run(m.recent_messages("x")),
        lambda: asyncio.run(m.whoami("x")),
        lambda: asyncio.run(m.logout("x")),
    ):
        try:
            fn()
            assert False, "should have raised"
        except TelegramError as e:
            assert "disabled" in str(e).lower() or "TITAN_TELEGRAM_ENABLED" in str(e)


def test_status_reflects_config(tmp_path):
    _clear()
    m = TelegramManager(session_dir=tmp_path / "s")
    s = m.status()
    assert set(s) >= {"enabled", "credentials_set", "sessions", "send_allowlist"}
    assert s["sessions"] == 0


def test_send_refused_when_allowlist_empty(tmp_path):
    m = _manager(tmp_path, enabled=True, allowlist="")
    try:
        asyncio.run(m.send_message("work", "titan_bot", "hello"))
        assert False, "should have raised"
    except TelegramError as e:
        assert "TITAN_TELEGRAM_SEND_ALLOWLIST" in str(e)


def test_send_refused_for_non_allowlisted_target_before_network(tmp_path):
    m = _manager(tmp_path, enabled=True, allowlist="allowed_user, another_one")
    try:
        asyncio.run(m.send_message("work", "stranger", "hello"))
        assert False, "should have raised"
    except TelegramError as e:
        assert "not allowed" in str(e)
        assert "allowed_user" in str(e)


def test_send_allowlist_parsing(tmp_path):
    m = _manager(tmp_path, enabled=True, allowlist=" @UserA, userB , 123456789")
    allow = m.send_allowlist()
    assert allow == ["usera", "userb", "123456789"]


def test_agent_dispatch_telegram_status(tmp_path):
    """The agent tool path surfaces telegram_status without any network."""
    import asyncio

    from titan_agent.agent import TitanAgent
    _clear()
    agent = TitanAgent()  # construction does not start MCP/network
    try:
        res = asyncio.run(agent.execute_tool_unified("telegram_status", {}))
    finally:
        try:
            asyncio.run(agent.mcp.stop_all())
        except Exception:  # noqa: BLE001, S110 - cleanup must never fail the test
            pass
    assert "Telegram control:" in res
    assert "DISABLED" in res


def test_agent_dispatch_telegram_send_refusal(tmp_path):
    import asyncio

    from titan_agent.agent import TitanAgent
    m = _manager(tmp_path, enabled=True, allowlist="ok_user")
    # Reuse the same manager via a lightweight agent with overridden telegram
    import titan_agent.agent as ag
    orig = ag.TelegramManager
    ag.TelegramManager = lambda: m  # type: ignore[assignment]
    agent = None
    try:
        from types import SimpleNamespace
        from titan_agent.core.guardrails.hitl import ApprovalStatus

        class _ApprovedHITL:
            def request(self, *_args, **_kwargs):
                return SimpleNamespace(status=ApprovalStatus.APPROVED)

            async def wait(self, request, timeout=None):
                return request

        agent = TitanAgent(hitl=_ApprovedHITL())
        res = asyncio.run(
            agent.execute_tool_unified("telegram_send", {"label": "w", "target": "stranger", "text": "hi"})
        )
    finally:
        ag.TelegramManager = orig
        try:
            if agent is not None:
                asyncio.run(agent.mcp.stop_all())
        except Exception:  # noqa: BLE001, S110 - cleanup must never fail the test
            pass
    assert "Telegram:" in res
    assert "not allowed" in res