"""Phase 5 — Devin-style session checkpoints and resume.

A checkpoint captures a run's live state (conversation messages + metadata) so an
interrupted or long-running session can be resumed later — the same continuity
Devin's long-horizon sessions and Claude Code's ``--continue`` rely on.

Design rules:

- **SQLite-backed**: checkpoint rows save run state between completed steps.
- **Pre-action barrier**: classic tool batches and structured runs are marked
  in progress durably before tools may cause side effects. If recovery finds an
  ambiguous in-progress action, it stops rather than replaying it. For classic
  tool batches, an operator can verify each external result and atomically
  reconcile every pending tool-call ID before resuming. Structured runs remain
  paused and require a fresh run; this is not exactly-once execution.
- **Resume** restores message structure and step metadata for ordinary
  interrupted runs. A ``status == \"done\"`` session returns its saved final
  answer instead of re-running.
- **Sensitive state**: checkpoint messages can include user text, model output,
  and tool arguments/results. Protect the local SQLite file accordingly; unlike
  the separate run trace journal, checkpoint contents are not metadata-only.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_MESSAGES = 40  # messages persisted per checkpoint (kept when resuming)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class RunCheckpoint:
    """Serializable snapshot of a run's live state."""

    session_id: str
    user_input: str
    mode: str = "fast"
    effort: str = "auto"
    strategy: str = "auto"
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps_done: int = 0
    tools_used: list[str] = field(default_factory=list)
    status: str = "running"  # running | done | error
    final_answer: str | None = None
    created_at: str = ""
    updated_at: str = ""


class CheckpointStore:
    """SQLite store for run checkpoints (one row per session_id)."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else Path.cwd() / "titan_checkpoints.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Checkpoints contain prompts and tool payloads. Create/restrict the DB
        # to the owner on POSIX rather than inheriting a permissive umask.
        fd = os.open(self.db_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        if os.name != "nt":
            os.chmod(self.db_path, 0o600)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    session_id TEXT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'fast',
                    effort TEXT NOT NULL DEFAULT 'auto',
                    strategy TEXT NOT NULL DEFAULT 'auto',
                    messages TEXT NOT NULL DEFAULT '[]',
                    steps_done INTEGER NOT NULL DEFAULT 0,
                    tools_used TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'running',
                    final_answer TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    # ---------- persistence ----------

    def save(self, cp: RunCheckpoint) -> None:
        """Upsert a checkpoint (replaces any existing one for the session)."""
        now = _now()
        created = cp.created_at or now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                    (session_id, user_input, mode, effort, strategy, messages,
                     steps_done, tools_used, status, final_answer, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_input=excluded.user_input,
                    mode=excluded.mode,
                    effort=excluded.effort,
                    strategy=excluded.strategy,
                    messages=excluded.messages,
                    steps_done=excluded.steps_done,
                    tools_used=excluded.tools_used,
                    status=excluded.status,
                    final_answer=excluded.final_answer,
                    updated_at=excluded.updated_at
                """,
                (
                    cp.session_id,
                    cp.user_input[:2000],
                    cp.mode,
                    cp.effort,
                    cp.strategy,
                    json.dumps(cp.messages, ensure_ascii=False),
                    int(cp.steps_done),
                    json.dumps(cp.tools_used, ensure_ascii=False),
                    cp.status,
                    cp.final_answer[:4000] if cp.final_answer else None,
                    created,
                    now,
                ),
            )

    def load(self, session_id: str) -> RunCheckpoint | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._row_to_cp(row) if row else None

    def list(self, limit: int = 20) -> list[RunCheckpoint]:
        """Recent checkpoints, newest first."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                (max(1, min(500, int(limit))),),
            ).fetchall()
        return [self._row_to_cp(r) for r in rows]

    def delete(self, session_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0

    def reconcile_tool_batch(
        self,
        session_id: str,
        outcomes: dict[str, str],
        *,
        operator: str,
    ) -> RunCheckpoint:
        """Record operator-verified results for an interrupted classic tool batch.

        This does not retry tools. The operator must provide exactly one
        non-empty outcome for every still-pending tool_call_id. Reconciliation
        is atomic and only applies to ``tool_in_progress`` checkpoints;
        structured runs remain paused because their internal engine state is
        not represented as resumable tool-call messages.
        """
        who = str(operator or "").strip()[:120]
        if not who:
            raise ValueError("operator identity is required")
        if not isinstance(outcomes, dict) or not outcomes:
            raise ValueError("at least one operator-verified outcome is required")
        normalized = {
            str(call_id): str(result).strip()
            for call_id, result in outcomes.items()
        }
        if any(not result for result in normalized.values()):
            raise ValueError("every reconciled tool call needs a non-empty outcome")
        if any(len(result) > 4000 for result in normalized.values()):
            raise ValueError("reconciled tool outcomes must be at most 4000 characters")

        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"checkpoint '{session_id}' was not found")
            cp = self._row_to_cp(row)
            if cp.status != "tool_in_progress":
                raise ValueError(
                    "manual tool reconciliation is available only for classic "
                    "tool_in_progress checkpoints"
                )

            calls: dict[str, str] = {}
            completed_ids = {
                str(message.get("tool_call_id"))
                for message in cp.messages
                if message.get("role") == "tool" and message.get("tool_call_id")
            }
            for message in cp.messages:
                if message.get("role") != "assistant":
                    continue
                for call in message.get("tool_calls") or []:
                    call_id = str(call.get("id") or "").strip()
                    function = call.get("function") or {}
                    if call_id and call_id not in completed_ids:
                        if call_id in calls:
                            raise ValueError(f"duplicate pending tool_call_id in checkpoint: {call_id}")
                        calls[call_id] = str(function.get("name") or "unknown_tool")
            pending_ids = set(calls)
            supplied_ids = set(normalized)
            if not pending_ids:
                raise ValueError("checkpoint contains no pending classic tool calls")
            if supplied_ids != pending_ids:
                missing = sorted(pending_ids - supplied_ids)
                extra = sorted(supplied_ids - pending_ids)
                raise ValueError(
                    f"outcomes must match pending tool_call_ids exactly; missing={missing}, extra={extra}"
                )

            for call_id, tool_name in calls.items():
                cp.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": (
                        f"[Operator-reported reconciliation by {who}; this tool was NOT replayed. "
                        f"Verify source evidence before relying on this result.]\n{normalized[call_id]}"
                    ),
                })
            cp.status = "running"
            cp.final_answer = None
            cp.updated_at = _now()
            updated = conn.execute(
                """
                UPDATE checkpoints
                SET messages = ?, status = ?, final_answer = NULL, updated_at = ?
                WHERE session_id = ? AND status = 'tool_in_progress'
                """,
                (
                    json.dumps(cp.messages, ensure_ascii=False),
                    cp.status,
                    cp.updated_at,
                    session_id,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("checkpoint changed during reconciliation")
        return cp

    def stats(self) -> dict[str, int]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            done = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE status = 'done'"
            ).fetchone()[0]
        return {"total": int(total), "done": int(done)}

    # ---------- helpers ----------

    @staticmethod
    def _row_to_cp(row: sqlite3.Row) -> RunCheckpoint:
        try:
            messages = json.loads(row["messages"] or "[]")
        except (ValueError, TypeError):
            messages = []
        try:
            tools = json.loads(row["tools_used"] or "[]")
        except (ValueError, TypeError):
            tools = []
        return RunCheckpoint(
            session_id=row["session_id"],
            user_input=row["user_input"] or "",
            mode=row["mode"] or "fast",
            effort=row["effort"] or "auto",
            strategy=row["strategy"] or "auto",
            messages=messages if isinstance(messages, list) else [],
            steps_done=int(row["steps_done"] or 0),
            tools_used=tools if isinstance(tools, list) else [],
            status=row["status"] or "running",
            final_answer=row["final_answer"],
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )