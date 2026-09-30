import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import TitanAgent
from .auth import require_api_key
from .config import MCP_CONFIG_FILE, WORKSPACE_DIR, full_access_enabled, set_full_access
from .core.guardrails.hitl import HumanInTheLoop
from .llm_client import LLMClient
from .logging_setup import LOG_RING, setup_logging
from .mcp_client import MCPManager
from .memory import MemoryManager
from .scheduler import CronScheduler
from .telegram import TelegramError, TelegramManager
from .tool_stats import TOOL_STATS
from .tools import ToolRegistry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: attempt to start MCP servers (mcp_manager is defined at module
    # load, before the app ever starts serving, so this lookup is always valid).
    asyncio.create_task(mcp_manager.start_all())
    # Startup: begin the cron scheduler loop (runs due jobs from cron/jobs.json).
    asyncio.create_task(cron_scheduler.start())
    yield
    # Shutdown
    await cron_scheduler.stop()
    await mcp_manager.stop_all()

app = FastAPI(title="Titan Agent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 15: structured JSON logging — one stream handler + the in-memory ring
# that backs GET /api/logs/recent (idempotent, no-op if already configured).
setup_logging()

# Global instances
mcp_manager = MCPManager(MCP_CONFIG_FILE)
tool_registry = ToolRegistry(WORKSPACE_DIR)
memory_manager = MemoryManager()
telegram_manager = TelegramManager()
llm_client = LLMClient()
# Phase 14: Human-in-the-loop. One global approval manager, wired into BOTH the
# tool registry (guarded wrappers) and the live agent (single gate in
# execute_tool_unified). Decisions are audited to workspace/hitl_audit.jsonl.
hitl_timeout = float(os.getenv("TITAN_HITL_TIMEOUT", "120"))
hitl_manager = HumanInTheLoop(
    default_timeout=hitl_timeout,
    audit_path=WORKSPACE_DIR / "hitl_audit.jsonl",
)
tool_registry.attach_hitl(hitl_manager, hitl_timeout)
agent = TitanAgent(
    llm=llm_client,
    tools=tool_registry,
    mcp=mcp_manager,
    memory=memory_manager,
    telegram=telegram_manager,
    hitl=hitl_manager,
    hitl_timeout=hitl_timeout,
)

async def _cron_runner(prompt: str, session_id: str, mode: str, effort: str, strategy: str = "auto", auto_commit: bool = False, resume: bool = False) -> str:
    """Runner used by the cron scheduler: execute a prompt with the live agent
    and return the final answer text (errors raise so the job is marked failed)."""
    final = ""
    async for ev in agent.run_task(prompt, session_id=session_id, mode=mode, effort=effort, strategy=strategy, auto_commit=auto_commit, resume=resume):
        if ev.type == "final_answer":
            final = (final + "\n\n" + ev.data).strip() if final else ev.data
        elif ev.type == "error":
            raise RuntimeError(ev.data)
    return final or "(no answer produced)"

cron_scheduler = CronScheduler(runner=_cron_runner)


def _cron_runner_as_runner(
    task: str, opts: dict[str, Any]
) -> tuple[int, str, list[dict[str, Any]]]:
    """Adapt the live-agent cron runner to the daemon Runner signature.

    Runs one task through the full live agent loop in this process (shared
    LLM/agent wiring) and returns (exit_code, final_answer, events). Used by
    POST /api/queue/process-once so queue tasks flow through the SAME agent
    instance the web UI uses.
    """
    import asyncio as _asyncio

    final = ""
    err = ""

    async def _run() -> None:
        nonlocal final, err
        try:
            async for ev in agent.run_task(
                task,
                session_id=str(opts.get("session_id") or "queued-task"),
                mode=str(opts.get("mode", "fast")),
                effort=str(opts.get("effort", "auto")),
                strategy=str(opts.get("strategy", "auto")),
            ):
                if ev.type == "final_answer":
                    final = (final + "\n\n" + ev.data).strip() if final else ev.data
                elif ev.type == "error":
                    err = err or str(ev.data)
        except Exception as exc:  # noqa: BLE001
            err = err or str(exc)

    try:
        _asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - nested-loop safety
        err = err or str(exc)
    return (0 if (final and not err) else 1), final or err or "(no answer produced)", []

WEB_UI_DIR = Path(__file__).resolve().parent / "web_ui"
WEB_UI_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(WEB_UI_DIR)), name="static")

@app.get("/")
async def root():
    index_file = WEB_UI_DIR / "index.html"
    if index_file.exists():
        async with aiofiles.open(index_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return HTMLResponse(content)
    return HTMLResponse("<h1>Titan Agent Web UI loading...</h1>")

@app.get("/health")
async def health():
    """Public liveness check — the only route that needs no Bearer token."""
    return {"status": "ok"}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "web_session"
    mode: str = "fast"
    effort: str = "auto"
    strategy: str = "auto"
    auto_commit: bool = False
    resume: bool = False

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        try:
            async for ev in agent.run_task(req.message, session_id=req.session_id, mode=req.mode, effort=req.effort, strategy=req.strategy, auto_commit=req.auto_commit, resume=req.resume):
                payload = json.dumps(ev.to_dict())
                yield f"data: {payload}\n\n"
        except (RuntimeError, OSError, ValueError) as e:
            err_payload = json.dumps({"type": "error", "data": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class ToolExecuteRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

@app.post("/api/tools/execute")
async def execute_tool_endpoint(req: ToolExecuteRequest):
    res = await agent.execute_tool_unified(req.tool_name, req.arguments)
    return {"status": "success", "tool_name": req.tool_name, "result": res}

@app.get("/api/local-models")
async def get_local_models():
    models = await LLMClient.detect_local_models()
    return models

@app.get("/api/token-usage")
async def get_token_usage():
    """Live token throughput guardrail stats (cap enforced on every LLM call)."""
    return llm_client.token_limiter.stats()

@app.get("/api/config")
async def get_config():
    return {
        "provider": llm_client.provider,
        "model": llm_client.model,
        "base_url": llm_client.base_url,
        "has_key": bool(llm_client.api_key),
        "workspace": str(WORKSPACE_DIR),
        # Phase 8: Full Access — all capability boundaries removed.
        "full_access": full_access_enabled(),
    }

class ConfigUpdateRequest(BaseModel):
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""
    full_access: bool | None = None

@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    llm_client.set_model(
        provider=req.provider,
        model=req.model,
        api_key=req.api_key if req.api_key else None,
        base_url=req.base_url if req.base_url else None
    )
    # Phase 8: runtime Full Access toggle (no restart needed).
    if req.full_access is not None:
        set_full_access(req.full_access)
    return {
        "status": "success",
        "provider": llm_client.provider,
        "model": llm_client.model,
        "full_access": full_access_enabled(),
    }

@app.get("/api/free-providers")
async def get_free_providers_endpoint():
    """Return catalog of free API key providers and models."""
    from titan_agent.free_providers import get_free_providers
    return {"providers": get_free_providers()}

@app.get("/api/mcp/tools")
async def get_mcp_tools():
    # Include configured-but-offline servers as well as connected ones so an
    # empty MCP result does not look like there is no MCP configuration.
    configured = mcp_manager.load_config().get("mcpServers", {})
    servers_info = {
        name: {
            "connected": False,
            "status": "not_connected",
            "tools_count": 0,
            "tools": [],
        }
        for name in configured
    }
    for s_name, conn in mcp_manager.servers.items():
        servers_info[s_name] = {
            "connected": conn.is_connected,
            "status": "connected" if conn.is_connected else "disconnected",
            "tools_count": len(conn.tools),
            "tools": [t.get("name") for t in conn.tools],
        }
    return {
        "servers": servers_info,
        "connected_count": sum(1 for server in servers_info.values() if server["connected"]),
        "configured_count": len(servers_info),
        "builtin_tools": [t["function"]["name"] for t in tool_registry.get_tool_definitions()],
    }

@app.get("/api/workspace/files")
async def list_workspace_files():
    files = []
    if WORKSPACE_DIR.exists():
        for p in WORKSPACE_DIR.rglob("*"):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "rel_path": str(p.relative_to(WORKSPACE_DIR)),
                    "size": p.stat().st_size
                })
    return {"files": files}

@app.get("/api/memory")
async def get_memory(query: str = "", scope: str = ""):
    if query:
        return {"knowledge": memory_manager.search_knowledge(query, scope=scope or None)}
    return {"knowledge": memory_manager.get_all_knowledge()}

@app.get("/api/memory/vault")
async def get_memory_vault(scope: str = ""):
    return {"vault": memory_manager.vault_list(scope=scope or None, limit=200)}

@app.get("/api/memory/handoffs")
async def get_handoffs(status: str = "open"):
    return {"handoffs": memory_manager.list_handoffs(status=status or None)}

# ---- Telegram account manager (consent-gated) ----
class TelegramLoginStartRequest(BaseModel):
    label: str
    phone: str

class TelegramLoginConfirmRequest(BaseModel):
    label: str
    code: str

class TelegramSendRequest(BaseModel):
    label: str
    target: str
    text: str

class TelegramLogoutRequest(BaseModel):
    label: str
    delete: bool = False

async def _tg(resp_fn):
    """Run a telegram handler, converting TelegramError to a JSON-safe error."""
    try:
        return await resp_fn()
    except TelegramError as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/telegram/status")
async def telegram_status():
    return {"status": "success", "data": telegram_manager.status()}

@app.get("/api/telegram/accounts")
async def telegram_accounts():
    return await _tg(lambda: asyncio.to_thread(telegram_manager.list_accounts))

@app.post("/api/telegram/login/start")
async def telegram_login_start(req: TelegramLoginStartRequest):
    return await _tg(lambda: telegram_manager.login_start(req.label, req.phone))

@app.post("/api/telegram/login/confirm")
async def telegram_login_confirm(req: TelegramLoginConfirmRequest):
    return await _tg(lambda: telegram_manager.login_confirm(req.label, req.code))

@app.post("/api/telegram/send")
async def telegram_send(req: TelegramSendRequest):
    return await _tg(lambda: telegram_manager.send_message(req.label, req.target, req.text))

@app.get("/api/telegram/recent")
async def telegram_recent(label: str, limit: int = 10):
    return await _tg(lambda: telegram_manager.recent_messages(label, limit=limit))

@app.post("/api/telegram/logout")
async def telegram_logout(req: TelegramLogoutRequest):
    return await _tg(lambda: telegram_manager.logout(req.label, delete=req.delete))

class CronJobRequest(BaseModel):
    prompt: str
    name: str = "cron job"
    schedule: dict[str, Any] = Field(default_factory=lambda: {"interval_minutes": 60})
    mode: str = "fast"
    effort: str = "auto"
    session_id: str = ""
    enabled: bool = True
    job_id: str = ""

class EnqueueTaskRequest(BaseModel):
    task: str
    name: str = ""
    priority: int = 0
    schedule_at: float = 0.0
    max_attempts: int = 3

@app.get("/api/cron/jobs")
async def cron_list():
    return {"jobs": cron_scheduler.list_jobs()}

@app.post("/api/cron/jobs")
async def cron_add(req: CronJobRequest):
    try:
        job = cron_scheduler.add_job(
            prompt=req.prompt,
            name=req.name,
            schedule=req.schedule,
            mode=req.mode,
            effort=req.effort,
            session_id=req.session_id,
            enabled=req.enabled,
            job_id=req.job_id,
        )
        return {"status": "success", "job": job}
    except ValueError as e:
        return {"status": "error", "detail": str(e)}

@app.delete("/api/cron/jobs/{job_id}")
async def cron_delete(job_id: str):
    ok = cron_scheduler.remove_job(job_id)
    return {"status": "success" if ok else "error"}

@app.post("/api/cron/jobs/{job_id}/toggle")
async def cron_toggle(job_id: str):
    job = cron_scheduler.toggle_job(job_id)
    if job is None:
        return {"status": "error", "detail": "Unknown job"}
    return {"status": "success", "job": job}

@app.post("/api/cron/jobs/{job_id}/run")
async def cron_run(job_id: str):
    try:
        job = await cron_scheduler.run_now(job_id)
        return {"status": "success", "job": job}
    except KeyError as e:
        return {"status": "error", "detail": str(e)}

# ---- Phase 7: autonomous task queue ----

_task_queue_cache: Any = None  # lazy singleton (module-level avoids function-attr typing)


def _task_queue():
    global _task_queue_cache
    from .config import TASK_QUEUE_FILE as _qfile
    from .queue import TaskQueue

    if _task_queue_cache is None:
        _task_queue_cache = TaskQueue(_qfile)
    return _task_queue_cache

@app.get("/api/queue/tasks")
async def queue_list(status: str = "", limit: int = 20):
    try:
        tasks = _task_queue().list(status=status or None, limit=limit)
        return {"status": "success", "tasks": [t.to_dict() for t in tasks], "stats": _task_queue().stats()}
    except Exception as e:  # noqa: BLE001 - API surface
        return {"status": "error", "detail": str(e)}

@app.post("/api/queue/tasks")
async def queue_enqueue(req: EnqueueTaskRequest):
    try:
        tid = _task_queue().enqueue(
            req.task.strip(),
            name=req.name or None,
            priority=req.priority,
            schedule_at=req.schedule_at,
            max_attempts=req.max_attempts,
        )
        return {"status": "success", "task_id": tid}
    except Exception as e:  # noqa: BLE001 - API surface
        return {"status": "error", "detail": str(e)}

@app.delete("/api/queue/tasks/{task_id}")
async def queue_delete(task_id: int):
    ok = _task_queue().cancel(task_id)
    return {"status": "success" if ok else "error"}

@app.post("/api/queue/process-once")
async def queue_process_once():
    """Run the daemon loop once: claims and executes all currently-due tasks."""
    from .daemon import TaskDaemon

    try:
        daemon = TaskDaemon(_task_queue(), runner=_cron_runner_as_runner)
        processed = await daemon.run_once()
        return {"status": "success", "processed": processed}
    except Exception as e:  # noqa: BLE001 - API surface
        return {"status": "error", "detail": str(e)}


# ---- Phase 14: Human-in-the-loop API -----------------------------------------
# Backs the web UI approval queue. The live agent creates requests in the
# global `hitl_manager`; these endpoints list them and let a human decide.
# (All routes are guarded by the Phase 12 Bearer-token auth below.)

class HitlRequestCreate(BaseModel):
    action: str = Field(..., min_length=1, description="Tool/action name, e.g. delete_file")
    resource: str = "*"
    details: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class HitlDecideBody(BaseModel):
    request_id: str = Field(..., min_length=1)
    decision: str = Field(..., description="approve | deny | cancel")
    by: str = "human"


@app.get("/api/hitl")
async def hitl_index(limit: int = 200):
    """Audit trail: most recent approval requests, newest first."""
    reqs = hitl_manager.recent(limit=limit)
    return {
        "requests": [r.to_dict() for r in reqs],
        "counts": hitl_manager.status_counts(),
    }


@app.get("/api/hitl/pending")
async def hitl_pending():
    """Everything currently waiting for a human decision."""
    reqs = hitl_manager.pending()
    return {
        "pending": [r.to_dict() for r in reqs],
        "count": len(reqs),
        "counts": hitl_manager.status_counts(),
    }


@app.post("/api/hitl/request")
async def hitl_request(body: HitlRequestCreate):
    """Create a new approval request (manual or for tooling/tests)."""
    req = hitl_manager.request(body.action, body.resource, body.details, body.reason)
    return {"status": "success", "request": req.to_dict()}


@app.post("/api/hitl/decide")
async def hitl_decide(body: HitlDecideBody):
    """Resolve a pending request: approve, deny, or cancel."""
    if body.decision == "approve":
        req = hitl_manager.approve(body.request_id, by=body.by)
    elif body.decision == "deny":
        req = hitl_manager.deny(body.request_id, by=body.by)
    elif body.decision == "cancel":
        req = hitl_manager.cancel(body.request_id)
    else:
        raise HTTPException(status_code=422, detail="decision must be 'approve', 'deny' or 'cancel'")
    if req is None:
        raise HTTPException(status_code=404, detail=f"unknown request_id: {body.request_id}")
    return {"status": "success", "request": req.to_dict()}


@app.get("/api/hitl/{request_id}")
async def hitl_get(request_id: str):
    """Status of a single approval request."""
    req = hitl_manager.get(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"unknown request_id: {request_id}")
    return {"request": req.to_dict()}


class CheckpointReconcileBody(BaseModel):
    operator: str = Field(..., min_length=1, max_length=120)
    outcomes: dict[str, str] = Field(
        ..., min_length=1,
        description="Exact mapping of every pending tool_call_id to an operator-verified result.",
    )


@app.post("/api/checkpoints/{session_id}/reconcile")
async def checkpoint_reconcile(session_id: str, body: CheckpointReconcileBody):
    """Resume a classic interrupted batch only after explicit operator review.

    This endpoint records supplied outcomes; it never retries the underlying
    tool. Structured execution checkpoints cannot be reconciled this way.
    """
    store = agent.checkpoint_store
    if store is None:
        raise HTTPException(status_code=503, detail="checkpoint storage is unavailable")
    try:
        cp = store.reconcile_tool_batch(
            session_id,
            body.outcomes,
            operator=body.operator,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - never report a failed write as reconciled
        raise HTTPException(status_code=503, detail="checkpoint reconciliation failed") from exc
    return {
        "status": "reconciled",
        "session_id": cp.session_id,
        "checkpoint_status": cp.status,
        "reconciled_tool_call_ids": sorted(body.outcomes),
        "next_step": "resume this session with resume=true; no tool call was replayed",
    }


@app.get("/api/staff/roles")
async def staff_roles():
    """List of all available dedicated subagent specialist roles."""
    from .staff import staff_catalog
    return {"status": "success", "roles": staff_catalog()}


@app.get("/api/dag/status")
async def dag_status():
    """Returns current active Task Graph status and execution progress."""
    from .tools import ToolRegistry

    reg = ToolRegistry()
    last_res = getattr(reg, "_last_dag_result", None)
    if last_res:
        return {"status": "success", "active": True, "dag": last_res.to_dict()}
    return {
        "status": "success",
        "active": False,
        "message": "No active DAG execution.",
    }



# ---- Phase 15: structured logging API ---------------------------------------
# GET /api/logs/recent returns the newest JSON log records from the in-memory
# ring (no filesystem reads, no log-tailing). Auth-protected like every /api.

@app.get("/api/logs/recent")
async def logs_recent(limit: int = 100, level: str = ""):
    """Newest structured log records, optionally filtered by exact level name."""
    records = list(LOG_RING)
    records.reverse()  # newest first
    level_hint = level.strip().upper()
    if level_hint:
        records = [r for r in records if r.get("level") == level_hint]
    clipped = records[: max(1, min(limit, 1000))]
    return {
        "logs": clipped,
        "count": len(clipped),
        "total": len(records),
        "level_filter": level_hint or None,
    }


# ---- Phase 22: per-tool telemetry API --------------------------------------
# GET /api/tools/stats exposes the process-wide tool execution record (same
# collector every TitanAgent writes to by default). Auth-protected /api route.


@app.get("/api/tools/stats")
async def tools_stats():
    """Per-tool success/failure/latency record for this process."""
    summary = TOOL_STATS.summary()
    return {
        "totals": summary["totals"],
        "tools": summary["tools"],
    }


@app.get("/api/traces/recent")
async def recent_run_traces(limit: int = 100, run_id: str = ""):
    """Recent local run events; records exclude prompts, tool args, and outputs."""
    store = agent.trace_store
    if store is None:
        raise HTTPException(status_code=503, detail="Run trace store is unavailable.")
    events = store.recent_events(limit, run_id=run_id or None)
    return {"events": events, "count": len(events), "storage": "local_metadata_only"}


# ---- Phase 29: repeated-failure guard observability API ----------------------
# GET /api/guard/state exposes the LIVE agent's per-run repeated-failure guard
# counters (identical tool calls that keep failing and are about to be skipped)
# so an operator can see which calls the harness is blocking and why.


@app.get("/api/guard/state")
async def guard_state():
    """Per-run repeated-failure + malformed-arguments guard counters and the
    effective config."""
    from .config import (
        dead_end_window,
        malformed_guard_enabled,
        malformed_guard_limit,
        repeat_guard_enabled,
        repeat_guard_limit,
    )

    counters = dict(getattr(agent, "_guard_failures", {}) or {})
    malformed = dict(getattr(agent, "_malformed_calls", {}) or {})
    # Phase 38: guard decision log (most recent first) + totals and the
    # dead-end detector streak, so the endpoint shows what the harness BLOCKED
    # and why, not just the current counters.
    actions = list(reversed(getattr(agent, "_guard_actions", []) or []))[:50]
    totals = dict(getattr(agent, "_guard_totals", {}) or {})
    return {
        "enabled": repeat_guard_enabled(),
        "limit": repeat_guard_limit(),
        "blocked_patterns": [
            {
                "tool": key.split("\x00", 1)[0],
                "failures": count,
            }
            for key, count in sorted(
                counters.items(), key=lambda kv: -kv[1]
            )[:50]
        ],
        "malformed": {
            "enabled": malformed_guard_enabled(),
            "limit": malformed_guard_limit(),
            "patterns": [
                {"tool": name, "skipped": count}
                for name, count in sorted(
                    malformed.items(), key=lambda kv: -kv[1]
                )[:50]
            ],
        },
        "actions": actions,
        "totals": totals,
        "dead_end": {
            "window": dead_end_window(),
            "consecutive_failed_batches": int(
                getattr(agent, "_consecutive_failed_batches", 0) or 0
            ),
        },
    }


@app.post("/api/guard/reset")
async def guard_reset():
    """Phase 38: clear the live agent's per-run guard state (repeated-failure
    counters, malformed counters, decision log and totals) so an operator can
    give a stuck run a clean slate after fixing the underlying cause. The next
    run_task still resets everything fresh on its own."""
    agent._guard_failures = {}
    agent._malformed_calls = {}
    agent._guard_actions = []
    agent._guard_totals = {}
    agent._consecutive_failed_batches = 0
    return {"ok": True}


# ---- Phase 42: Prometheus & OpenTelemetry Metrics Exporter ------------------

@app.get("/api/metrics", response_class=PlainTextResponse)
@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible plain text metrics exporter."""
    from .skills import SkillRegistry

    stats = TOOL_STATS.summary()
    skills_count = len(SkillRegistry().list_skills())
    pending_hitl = len(hitl_manager.pending())

    lines = [
        "# HELP titan_tool_calls_total Total tool executions by name",
        "# TYPE titan_tool_calls_total counter",
    ]
    for tool_name, tstats in stats.get("tools", {}).items():
        total = tstats.get("total", 0)
        errors = tstats.get("errors", 0)
        lines.append(f'titan_tool_calls_total{{tool="{tool_name}",status="ok"}} {total - errors}')
        lines.append(f'titan_tool_calls_total{{tool="{tool_name}",status="error"}} {errors}')
        avg_ms = tstats.get("avg_latency_ms", 0.0)
        lines.append(f'titan_tool_latency_ms{{tool="{tool_name}"}} {avg_ms:.2f}')

    lines.extend([
        "# HELP titan_skills_total Total registered skill playbooks",
        "# TYPE titan_skills_total gauge",
        f"titan_skills_total {skills_count}",
        "# HELP titan_hitl_pending Current pending human-in-the-loop approvals",
        "# TYPE titan_hitl_pending gauge",
        f"titan_hitl_pending {pending_hitl}",
        "# HELP titan_system_info System status flag",
        "# TYPE titan_system_info gauge",
        'titan_system_info{status="ready"} 1',
    ])
    return "\n".join(lines) + "\n"


# ---- Phase 12: attach Bearer-token auth to every API route -------------------
# The liveness check (/health), web-UI shell (/), and metrics (/metrics, /api/metrics) stay public.
_PUBLIC_PATHS = frozenset({"/", "/health", "/metrics", "/api/metrics"})



def _apply_auth_dependency() -> None:
    """Append `Depends(require_api_key)` to every non-public route, in place.

    FastAPI builds each route's dependency tree (`route.dependant`) exactly once,
    inside ``APIRoute.__init__``, and the per-request handler captures that tree
    *by reference*. So we must mutate the LIVE tree node, not reassign
    ``route.dependant`` to a fresh object — the baked handler would keep solving
    against the old tree and the auth check would never run.

    Mirror exactly what ``APIRoute.__init__`` does for ``dependencies=...``:
    attach a parameterless sub-dependant (``get_parameterless_sub_dependant``)
    at the front of ``route.dependant.dependencies``. ``solve_dependencies``
    walks this list at request time, so the inserted node is enforced on every
    call even though the handler was created earlier.
    """
    from fastapi.dependencies.utils import get_parameterless_sub_dependant
    from fastapi.routing import APIRoute

    for route in list(app.routes):
        if not isinstance(route, APIRoute):
            continue
        if route.path in _PUBLIC_PATHS:
            continue
        if any(
            (getattr(d, "dependency", None) or getattr(d, "call", None))
            is require_api_key
            for d in route.dependant.dependencies
        ):
            continue
        route.dependencies.append(Depends(require_api_key))
        route.dependant.dependencies.insert(
            0,
            get_parameterless_sub_dependant(
                depends=Depends(require_api_key), path=route.path_format
            ),
        )


_apply_auth_dependency()
