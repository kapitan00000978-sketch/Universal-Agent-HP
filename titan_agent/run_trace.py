"""Privacy-minimizing SQLite event journal for agent run observability.

The journal intentionally stores operational metadata only: no user prompts,
model messages, tool arguments, tool output, or exception text. It is a local
trace aid, not an OpenTelemetry exporter or a proof of task correctness.
"""
from __future__ import annotations

import contextvars
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_CURRENT_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "titan_current_run_id", default=None
)

EVENT_TYPES = {
    "run_started",
    "run_finished",
    "run_failed",
    "run_cancelled",
    "run_interrupted",
    "tool_started",
    "tool_finished",
    "tool_failed",
    "model_call_started",
    "model_call_finished",
    "model_call_failed",
    "approval_requested",
    "approval_result",
}
_ALLOWED_DETAIL_KEYS = {
    "mode",
    "effort",
    "strategy",
    "provider",
    "model",
    "tool_name",
    "call_role",
    "attempt",
    "error_type",
    "decision",
}


def current_run_id() -> str | None:
    """Return the run ID bound to this async context, if any."""
    return _CURRENT_RUN_ID.get()


def bind_run_id(run_id: str) -> contextvars.Token[str | None]:
    """Bind a run ID to the current task and its child async work."""
    return _CURRENT_RUN_ID.set(run_id)


def reset_run_id(token: contextvars.Token[str | None]) -> None:
    _CURRENT_RUN_ID.reset(token)


class RunTraceStore:
    """Append-only local run event store with strict metadata allowlisting."""

    def __init__(self, db_path: Path | str, *, max_events: int = 100_000):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = max(100, int(max_events))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_trace_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    session_fingerprint TEXT,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT,
                    duration_ms REAL,
                    occurred_at TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_trace_run_id "
                "ON run_trace_events(run_id, event_id)"
            )

    @staticmethod
    def _clean_details(details: dict[str, Any] | None) -> str:
        """Serialize only known scalar operational fields; discard unknown keys."""
        safe: dict[str, str | int | float | bool | None] = {}
        for key, value in (details or {}).items():
            if key not in _ALLOWED_DETAIL_KEYS:
                continue
            if value is None or isinstance(value, (bool, int, float)):
                safe[key] = value
            elif isinstance(value, str):
                safe[key] = value[:128]
        return json.dumps(safe, ensure_ascii=True, separators=(",", ":"))

    def record(
        self,
        event_type: str,
        *,
        run_id: str | None = None,
        component: str = "agent",
        status: str | None = None,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a sanitized event; return False when no run is currently bound."""
        effective_run_id = run_id or current_run_id()
        if not effective_run_id:
            return False
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported run trace event type: {event_type}")
        safe_duration = None
        if duration_ms is not None:
            safe_duration = min(86_400_000.0, max(0.0, float(duration_ms)))
        safe_status = str(status)[:32] if status is not None else None
        safe_component = str(component)[:64]
        with self._connection() as connection:
            cursor = connection.execute(
                """INSERT INTO run_trace_events
                   (run_id, session_fingerprint, event_type, component, status,
                    duration_ms, occurred_at, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(effective_run_id)[:64],
                    None,
                    event_type,
                    safe_component,
                    safe_status,
                    safe_duration,
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    self._clean_details(details),
                ),
            )
            if cursor.lastrowid and cursor.lastrowid % 1000 == 0:
                connection.execute(
                    "DELETE FROM run_trace_events WHERE event_id <= "
                    "(SELECT COALESCE(MAX(event_id), 0) - ? FROM run_trace_events)",
                    (self.max_events,),
                )
        return True

    def recent_events(self, limit: int = 100, *, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return newest sanitized events; never includes prompts or action payloads."""
        bounded_limit = max(1, min(1000, int(limit)))
        with self._connection() as connection:
            if run_id:
                rows = connection.execute(
                    """SELECT event_id, run_id, session_fingerprint, event_type,
                              component, status, duration_ms, occurred_at, details_json
                       FROM run_trace_events WHERE run_id = ?
                       ORDER BY event_id DESC LIMIT ?""",
                    (str(run_id)[:64], bounded_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT event_id, run_id, session_fingerprint, event_type,
                              component, status, duration_ms, occurred_at, details_json
                       FROM run_trace_events ORDER BY event_id DESC LIMIT ?""",
                    (bounded_limit,),
                ).fetchall()
        events = []
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except (TypeError, ValueError):
                details = {}
            events.append(
                {
                    "event_id": row["event_id"],
                    "run_id": row["run_id"],
                    "event_type": row["event_type"],
                    "component": row["component"],
                    "status": row["status"],
                    "duration_ms": row["duration_ms"],
                    "occurred_at": row["occurred_at"],
                    "details": details,
                }
            )
        return events

    def stats(self) -> dict[str, int]:
        with self._connection() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM run_trace_events").fetchone()[0])
            runs = int(connection.execute("SELECT COUNT(DISTINCT run_id) FROM run_trace_events").fetchone()[0])
        return {"events": total, "runs": runs}


__all__ = ["RunTraceStore", "bind_run_id", "current_run_id", "reset_run_id"]
