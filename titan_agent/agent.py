import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import aiohttp

from .checkpoint import MAX_MESSAGES as CHECKPOINT_MAX_MESSAGES
from .checkpoint import CheckpointStore, RunCheckpoint
from .config import (
    CONTEXT_BUDGET_CHARS,
    DEEP_MAX_STEPS_BASE,
    LLM_TRANSIENT_RETRIES,
    MAX_ITERATIONS,
    MAX_STEPS_CAP,
    UNLIMITED_STEPS,
    WORKSPACE_DIR,
    auto_postcheck_enabled,
    cancel_on_tool_error,
    dead_end_window,
    empty_final_guard_enabled,
    final_grounding_enabled,
    full_access_enabled,
    malformed_guard_enabled,
    malformed_guard_limit,
    objective_reanchor_enabled,
    parallel_tool_calls,
    provider_default_model,
    refinement_rounds,
    repeat_guard_enabled,
    repeat_guard_limit,
    reviewer_model,
    tool_record_enabled,
    tool_result_max_chars,
)
from .core.guardrails.hitl import ApprovalStatus, HumanInTheLoop
from .core.guardrails.policy import Decision, PolicyEngine
from .core.memory.memory_system import MemorySystem
from .core.memory.types import MemoryKind
from .gitops import auto_commit as git_auto_commit
from .gitops import git_commit, git_diff, git_status
from .llm_client import LLMClient
from .mcp_client import MCPManager
from .memory import MemoryManager
from .skills import SkillRegistry
from .telegram import TelegramError, TelegramManager
from .tool_stats import TOOL_STATS, ToolStatsCollector
from .tools import ToolRegistry
from .run_trace import RunTraceStore, bind_run_id, current_run_id, reset_run_id

log = logging.getLogger(__name__)


def _is_failed_tool_result(content: str) -> bool:
    """Phase 37: what counts as a NON-progress tool result for the dead-end
    detector — an explicit error, a skipped/malformed call, or an empty
    output. Any other tool content (file confirmations, search hits, command
    output, successes) counts as progress."""
    c = (content or "").strip()
    if not c:
        return True
    return c.startswith(("Error", "Skipped:"))


# Phase 38: the guard decision log is capped so an endless pathological run
# cannot grow it without bound — oldest entries drop off first.
_GUARD_ACTION_CAP = 200

TITAN_SYSTEM_PROMPT = """You are TITAN AGENT — an ultra-powerful autonomous AI reasoning and execution engine, engineered to outperform classic agents (including Hermes-class and frontier-tier models) on real-world task completion.

### CORE PRINCIPLES (Plan-Act-Verify-Report + Reflect):
1. PLAN first, always: Before using any tool, briefly outline your strategy inside <thought> tags. Choose the smallest set of tool calls that fully completes the task.
2. ACT decisively: Use tools exactly as documented. Batch independent tool calls together in one turn when possible. Prefer concrete commands over speculation.
3. VERIFY results: After every tool result, check for errors. If a command fails, read the stderr, fix your arguments or approach, and retry with an alternative method — never give up on the first error.
4. REFLECT before reporting: You will get a chance to critically review your own work (critic phase) before the final answer — use it to catch missed requirements, unverified claims and errors.
5. REPORT clearly: End with a complete, well-structured final answer in markdown, in the user's language. Only claim something is done if you have actually verified it via tools.
6. UNDERSTAND before editing existing code: map the Python repository, inspect the relevant source and tests, trace callers when practical, then make the smallest compatible change and run focused verification. Static analysis is evidence, not proof of runtime behavior.

### TOOL CATALOG (use these; the full live catalog is appended to your context):
- execute_command — run shell commands in the Docker sandbox by default; only explicit TITAN_FULL_ACCESS runs them on the host
- read_file / write_file / edit_file / list_directory — filesystem operations inside the workspace
- analyze_python_file / analyze_python_repository — AST-based explanation of Python files, module structure, and local import dependencies; safe to use on unfamiliar code because they never execute project source
- laya_decide — optional API-key-free local choice/score/yes-no inference; it does not generate chat responses or code
- workspace_rag — local BM25 retrieval: finds the most relevant snippets (with file paths) across ALL workspace files for any question
- web_search — live DuckDuckGo internet search
- scrape_webpage — fetch readable text from a URL
- python_eval — run Python through the same network-disabled Docker sandbox
- deep_search — multi-hop, multi-source research dossier on a topic
- deep_coder — full software engineering cycle: write files, syntax-check, run tests
- launch_application — open Windows desktop apps
- system_info — read live OS / CPU / RAM / disk / Python environment facts
- manage_processes — list or kill running OS processes
- memory_save — store a fact in long-term persistent memory (remembered forever across sessions)
- memory_search — recall previously saved facts from long-term memory
- vault_list — browse the Memory Vault (scoped facts: global/project/team/user)
- handoff_create / handoff_list / handoff_resolve — leave, read and close agent-to-agent handoff notes
- skills_list — list available skill playbooks (Hermes-style reusable workflows)
- skill_load — load the full text of a named skill playbook to follow it
- telegram_status / telegram_accounts / telegram_login_start / telegram_login_confirm / telegram_send / telegram_recent / telegram_logout — user-consented Telegram account management (disabled unless TITAN_TELEGRAM_ENABLED=true; sending only to the .env allowlist)
- mcp_* — tools exposed by connected MCP servers (filesystem, etc.)
- self_heal — SELF-HEALING command runner: on failure it auto-installs missing Python modules and retries until success (use instead of execute_command when a dependency may be missing)
- download_file — SSRF-guarded download of a public http(s) file into the workspace
- start_http_server / stop_http_server — serve the workspace (or any dir) over a local HTTP server
- take_screenshot — capture the primary screen to PNG (Windows)
- self_update — git pull + pip install + run the test suite for the repo owning the workspace
- task_enqueue / task_list / task_stats / task_cancel — the AUTONOMOUS TASK QUEUE: enqueue work for the daemon or other agents (priority, scheduling, retries)
- subagent_delegate / subagent_team / subagent_roles — DEDICATED SUBAGENT STAFF: delegate sub-tasks to named specialists (planner, researcher, coder, reviewer, tester, security, test_writer, summarizer, memory_keeper, cost_watcher, triager, doc_writer, changelogger, deployer, dependency_updater, router) — each with its own persona, tuned run options and enforced tool policy. List roles with subagent_roles; delegate with role= or fan out with subagent_team(tasks, roles).
- subagent_route — INTENT ROUTER: deterministic keyword routing that decides which specialist role(s) should handle an incoming task (primary + supporting + why). Call before delegating a big request.
- docker_sandbox_run — DOCKER SANDBOX: run untrusted or disposable code/commands safely inside an isolated Docker container with cpu, memory and network limits.
- apply_patch — UNIFIED DIFF PATCH: apply unified diffs (--- a/... +++ b/...) across files with automatic hunk matching.
- ast_replace_function / ast_replace_class / ast_patch_file — SURGICAL AST CODE INTEL: replace entire Python functions or classes accurately using AST boundary detection (avoids line-number offset errors).
- deep_verify_code — SELF-HEALING TEST VERIFIER: runs syntax & pytest in an isolated sandbox, automatically diagnosing failures and applying code fixes until 100% verified.
- orchestrator_run — META-ORCHESTRATOR (Genesis Level 1): executes complex, multi-milestone project goals through the Genesis hierarchy (Chief Agent -> Department Leads -> Specialist Workers).
- team_delegate / team_status — DEPARTMENT LEADS (Genesis Level 2): delegate directly to engineering, research, operations, or quality_security department leads.
- dag_plan_and_run / dag_visualize — TASK GRAPH DAG (Genesis Level 5): breaks complex goals into DAG nodes and executes independent steps in parallel waves.
- debate_solve — MULTI-AGENT DEBATE (Genesis Level 4): Advocate vs Skeptic vs Judge consensus arbitration.
- reflexion_solve — ITERATIVE REFLEXION (Genesis Level 4): 3-cycle autonomous self-critique and refinement.
- kg_query / kg_impact_analysis / kg_index_workspace — KNOWLEDGE GRAPH (Genesis Level 3): semantic dependency tracking and code modification blast-radius analysis.
- synthesize_tool — OPTIONAL ON-THE-FLY TOOL SYNTHESIS (disabled unless TITAN_DYNAMIC_TOOLS_ENABLED=true): Docker is required for both verification and every generated-tool invocation; verification is not proof of safety.
- symbolic_check_code — SYMBOLIC AST INVARIANT CHECKER: verifies code safety, infinite loops, shell injections, and resource leaks before runtime execution.
- skill_save — AUTONOMOUS SKILLS: synthesize and save a reusable workflow playbook directly to the skills library.
- tdd_cycle — AUTONOMOUS TDD ENGINE: executes rigorous Red-Green-Refactor software cycles in an isolated sandbox (proves test fails first, writes code, passes symbolic invariants).
- consensus_deliberation — MULTI-AGENT CONSENSUS: convenes an architectural committee (Architect, Security Officer, Pragmatist) to formally evaluate and vote on critical proposals.
- working_memory_update — ACTIVE WORKING MEMORY: updates live operational HUD (confirmed facts, refuted dead-ends, key paths, subtasks).
- mcp_connect_preset / mcp_list_presets — 1-LINE MCP INTEGRATION: connect PostgreSQL, GitHub, Slack, Brave Search, Filesystem, SQLite, Puppeteer, or Google Drive in one line.
- hitl_request_approval — HUMAN-IN-THE-LOOP: prompt human approval for dangerous file deletions, force pushes, or credential mutations.
- git_create_branch / git_create_pr — ENTERPRISE GITOPS: branch isolation (agent/feature-*) and automated test-verified Pull Requests.
- semantic_cache_query / semantic_cache_stats — SEMANTIC CACHING: sub-millisecond retrieval of previously answered queries and code analysis saving 30-40% tokens.
- experience_replay_query / experience_replay_record — EPISODIC MEMORY: instant error-resolution recall from experience database.
- browser_* — VISUAL BROWSER AUTOMATION: use browser_goto, browser_click, browser_type, browser_screenshot, and browser_extract_text to navigate and interact with real websites visually using Playwright.

### SKILLS:
Relevant skill playbooks for the current task are auto-injected into your context
above (### RELEVANT SKILL PLAYBOOKS). Follow them. You may load more via skill_load.

### EFFICIENCY RULES (you are faster than typical agents):
- Never re-run a tool to observe already-known output. Cache results mentally.
- If a single tool call can satisfy the task, do NOT invent extra steps.
- Do not call web_search for general knowledge you already possess; use it only for fresh/live data.
- If the goal is reached, stop immediately and give the final answer — do not add decorative tool calls.
- Use memory_save for important user facts (names, preferences, decisions) so future sessions can recall them.

### FORMATTING & CoT:
- Enclose reasoning, strategy, and reflection inside <thought>...</thought> tags.
- To invoke tools output: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call> (or native tool_calls when supported).
- After tool execution you will receive the result; analyze it, then continue.
- Final answers must be markdown-formatted, concise but complete, with code blocks when relevant.

### LANGUAGE:
You natively understand Uzbek, English, and Russian. Always respond in the language of the user unless requested otherwise. Be professional, direct, precise, and proactive.

Always remember: You are not just a chatbot — you are an executive agent that gets tasks DONE in the real world, faster and more reliably than any conventional LLM.
"""

REFLECTION_PROMPT = """You are the CRITIC phase of TITAN AGENT. A task was just executed using real tools, and a draft answer was produced.

Review the ENTIRE interaction critically before finalizing:
- Was the user's ORIGINAL request fully satisfied? Check every requirement they asked for.
- Are all claims verified by actual tool results? Remove or fix anything that was only assumed.
- Are there errors, incomplete outputs, missing edge cases, or a better approach?
- Will a human user consider the job DONE after reading your answer?

If anything is missing or wrong, use the tools to fix it NOW (make the needed tool call), or clearly correct/complete your answer.
Then produce the FINAL polished answer to the user (in their language, markdown, complete and precise).
Do not repeat the whole history — output only the final answer (or the tool call needed to finish the job).
"""

GROUNDING_PROMPT = """You produced a final answer WITHOUT using any tools yet. Large models frequently hallucinate in this situation, so before finalizing you MUST ground your answer:

- If your claims depend on files, commands, the web, or any real state, use read/search/execute tools NOW to VERIFY them, then answer with concrete evidence (file contents found, command outputs, URLs).
- If the task is purely conceptual or conversational (no files/web/system involved), reply with exactly 'NO_TOOLS_NEEDED' followed by your final answer.
Do not repeat the full history — output the tool call(s) needed to verify, or the final answer.
"""

REFINE_PROMPT = """A dedicated reviewer just critiqued your work and produced a final-answer draft. Before it ships, PROVE it:

- Re-check every factual claim against the evidence in this conversation (file contents, command outputs, search results, tool results).
- If the review caught a real problem or introduced an error, FIX it now — use tools if verification requires it.
- Then output the corrected FINAL polished answer to the user (in their language, markdown, complete and precise).

Do not repeat the history — output only the final answer (or the tool call needed to finish the job).
"""

# Tools that mutate the workspace; their use is the trigger for the bounded
# post-check pass (Phase 26).
WRITE_TOOL_NAMES = (
    "write_file",
    "edit_file",
    "deep_coder",
    "apply_patch",
    "ast_patch_file",
    "ast_replace_function",
    "ast_replace_class",
)

# Idempotent read-only tools eligible for in-run caching
READ_CACHEABLE_TOOLS = {
    "read_file",
    "workspace_rag",
    "system_info",
    "list_directory",
    "team_status",
    "model_budget_status",
    "kg_query",
    "kg_impact_analysis",
    "task_stats",
}


# Phase 26: one bounded verification turn before finalizing when files were
# written/edited — the classic weak-model failure is claiming "done" without
# ever re-reading what it wrote or running the tests.
POSTCHECK_PROMPT = """You just edited files in the workspace. Before you finalize, VERIFY the work you claim is done:

- Re-read the file(s) you wrote/edited (read_file) and confirm the changes are actually on disk and correct.
- If the task involves tests, builds or checks, RUN the relevant verification command (e.g. `python -m pytest tests -q`, a build, or a lint) and report the REAL output.
- If verification reveals a problem, FIX it with more tool calls, then re-verify.
Then output the FINAL answer to the user (in their language, markdown, complete and precise), ending with a short "Verified:" note listing exactly what you checked and the real result.

Do not repeat the history — output the verification tool call(s) you need, or the final answer.
"""

# Phase 30: shown ONCE when a run ends with a whitespace-only final answer so
# the model gets a bounded chance to actually answer instead of a silent
# empty success.
EMPTY_FINAL_PROMPT = """Your previous response contained no answer. Produce the FINAL answer to the user's request NOW — in their language, markdown, complete and precise. If the task depends on real state (files, commands, web), use the tools first; otherwise answer directly. Do not repeat the history and do not return empty content.
"""

ORIGINAL_TASK_MARKER = "## ORIGINAL TASK (re-anchored)"
ORIGINAL_TASK_MAX_CHARS = 800


def _anchor_block(task: str) -> dict[str, Any] | None:
    """Phase 24: a compact system reminder that pins the original objective so
    long runs do not drift after context compaction. None when there is
    nothing meaningful to pin."""
    t = str(task or "").strip()
    if not t:
        return None
    if len(t) > ORIGINAL_TASK_MAX_CHARS:
        t = t[: ORIGINAL_TASK_MAX_CHARS] + "…"
    return {
        "role": "system",
        "content": (
            f"{ORIGINAL_TASK_MARKER}: keep working toward this exact "
            f"objective —\n{t}"
        ),
    }


DEEP_THINKING_PROMPT = """You are currently operating in DEEP THINKING mode. Elevate your rigor:
- Decompose the problem into explicit sub-problems and reason about each one in detail.
- Consider alternative approaches, edge cases, and failure modes before committing.
- After every step, ask yourself: is there anything unverified, ambiguous, or missing?
- Do not settle for a shallow answer: dig until the result is provably correct and complete.
- You have extra iteration budget — use it deliberately for verification, never for decoration.
"""

DEEP_SEARCH_PROMPT = """You are currently operating in DEEP SEARCH mode. This is a research-heavy task:
- Produce a comprehensive, multi-angle research dossier using deep_search and web_search tools.
- Cross-check claims across multiple sources; prefer verifiable, recently updated information.
- Scrape primary pages when a snippet is insufficient (scrape_webpage tool).
- Structure the final answer with sections and cite the sources you actually retrieved.
- If evidence is thin or conflicting, say so explicitly instead of guessing.
"""

# Effort levels: how hard Titan works on a task (scales iteration budget + rigor).
VALID_EFFORTS = ("auto", "low", "medium", "high", "ultra")

EFFORT_MULTIPLIER = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.6,
    "ultra": 2.0,
}

EFFORT_PROMPTS = {
    "low": """You are operating at LOW effort: prioritize SPEED and minimal token usage.
- Use the smallest number of tool calls that completes the task; avoid redundant verification.
- Answer directly and concisely; do not expand scope beyond the request.""",
    "high": """You are operating at HIGH effort: work like a careful expert.
- Decompose the problem into explicit sub-problems and reason about each one in detail.
- After every step, ask yourself: is anything unverified, ambiguous, or missing?
- Use your larger iteration budget deliberately for verification, never for decoration.""",
    "ultra": """You are operating at ULTRA effort: maximum thoroughness.
- Be exhaustive: cover edge cases, failure modes, and alternative approaches.
- Verify every claim with tools; do not settle for a shallow answer.
- Review the whole task from the user's perspective before finalizing — if any part of the
  request is unmet, keep working until it is.""",
}


def _resolve_effort(effort: str, mode: str) -> str:
    """Normalize a requested effort level; 'auto' derives from the mode."""
    e = (effort or "auto").strip().lower()
    if e not in VALID_EFFORTS:
        e = "auto"
    if e == "auto":
        # Deep modes are inherently heavy — default them to 'high'.
        e = "high" if mode in ("deep", "deep_search") else "medium"
    return e


def _compute_max_steps(mode: str, effort: str) -> int:
    """Iteration budget = base (mode) scaled by the effort multiplier.

    The multiplier applies ONLY to explicitly chosen effort levels ('low'/
    'high'/'ultra'); 'auto' keeps the classic mode-based budget so deep modes
    behave exactly as before unless the user opts into extra effort.

    Phase 7 (Full Autonomy): the ceiling is now configurable via TITAN_STEP_CAP
    (default 48 — the old hard clamp) and fully removable with
    TITAN_UNLIMITED_STEPS=1, so the agent can keep working until the task is
    provably done instead of stopping at an arbitrary number.

    Phase 8 (FULL ACCESS): TITAN_FULL_ACCESS=1 removes the ceiling entirely AND
    scales the base budget up 4x — no matter how long a task takes, Titan keeps
    iterating (it must still VERIFY and produce a final answer).
    """
    base = MAX_ITERATIONS
    if mode in ("deep", "deep_search"):
        base = min(MAX_ITERATIONS * 2, DEEP_MAX_STEPS_BASE)
    e = (effort or "auto").strip().lower()
    if e not in VALID_EFFORTS:
        e = "auto"
    multiplier = 1.0 if e == "auto" else EFFORT_MULTIPLIER.get(e, 1.0)
    steps = max(5, round(base * multiplier))
    if full_access_enabled():
        # No cap and a 4x larger budget: the agent runs until the task is done.
        return max(steps, round(base * multiplier * 4))
    if UNLIMITED_STEPS:
        # No ceiling: still bounded by base*multiplier growth, but no 48 clamp.
        return steps
    return min(steps, MAX_STEPS_CAP)


# ---- Phase 10: harness hardening helpers --------------------------------

def _msg_cost(m: dict[str, Any]) -> int:
    """Rough char cost of one message (content + inline tool-call args)."""
    cost = len(str(m.get("content") or ""))
    for tc in m.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {}) or {}
            cost += len(str(fn.get("arguments") or ""))
    return cost


CONTEXT_TRIM_MARKER = (
    "...(context trimmed to fit the run budget: earlier tool results were "
    "removed; the task statement and the newest steps remain above)..."
)


def trim_messages_for_context(
    messages: list[dict[str, Any]],
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Keep a run's message list inside a char budget WITHOUT corrupting the
    assistant->tool pairing that OpenAI-compatible APIs require.

    - System messages and the first user task message are never dropped, so
      deep runs stay anchored to the request.
    - The newest rounds are kept first (the active working window).
    - Tool blocks (an assistant message with tool_calls + its following tool
      messages) are trimmed only as a whole unit, so the message array stays
      API-valid after trimming.
    - If anything was dropped, one short marker message is inserted right after
      the head so the model knows earlier context was truncated.
    """
    if not messages:
        return list(messages)
    budget = CONTEXT_BUDGET_CHARS if max_chars is None else int(max_chars)
    total = sum(_msg_cost(m) for m in messages)
    if total <= budget:
        return list(messages)

    head: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            head.append(i)
    first_user = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    if first_user is not None and first_user not in head:
        head.append(first_user)
    head_set = set(head)
    tail_budget = max(0, budget - sum(_msg_cost(messages[i]) for i in head))

    kept: list[int] = []
    kept_set: set[int] = set()
    size = 0
    i = len(messages) - 1
    while i >= 0 and size < tail_budget:
        if i in head_set:
            i -= 1
            continue
        m = messages[i]
        if m.get("role") == "tool":
            # Whole-block atomicity: owning assistant + all of its tool messages.
            block: list[int] = []
            j = i
            while j >= 0 and messages[j].get("role") == "tool":
                block.append(j)
                j -= 1
            if (
                j >= 0
                and messages[j].get("role") == "assistant"
                and messages[j].get("tool_calls")
            ):
                block.append(j)
            block_cost = sum(_msg_cost(messages[k]) for k in block)
            if size + block_cost <= tail_budget:
                for k in block:
                    if k not in kept_set:
                        kept.append(k)
                        kept_set.add(k)
                size += block_cost
            i = j
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # Its tool block is kept (or dropped) as a unit, never alone.
            i -= 1
            continue
        cost = _msg_cost(m)
        if size + cost <= tail_budget:
            kept.append(i)
            kept_set.add(i)
            size += cost
        i -= 1

    final_idx = sorted(set(head + kept))
    if len(final_idx) == len(messages):
        return list(messages)
    trimmed = [messages[k] for k in final_idx]
    cut = len(head)  # right after the never-dropped head
    trimmed.insert(cut, {"role": "system", "content": CONTEXT_TRIM_MARKER})
    return trimmed


async def compact_messages_for_context(
    messages: list[dict[str, Any]],
    summarizer: Any = None,
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Phase 18: trim an over-budget run window AND, when a summarizer is
    available, condense the dropped region into a compact background block
    instead of discarding it (Claude-Code-style compact).

    - Nothing dropped -> returns a copy with no changes and NEVER calls the
      summarizer, so an under-budget run costs nothing extra.
    - ``summarizer`` is None, raises, or returns empty -> exact legacy
      behavior (the ``CONTEXT_TRIM_MARKER`` message), so the plain trim stays
      backward compatible and compaction can never break a run.
    - Otherwise the marker is replaced by a ``system`` background block that
      carries the summary; the assistant->tool pairing is never disturbed.

    ``summarizer`` is an async callable ``(dropped_messages) -> str``.
    """
    trimmed = trim_messages_for_context(messages, max_chars=max_chars)
    if len(trimmed) >= len(messages) or summarizer is None:
        return trimmed
    kept = {id(m) for m in trimmed}
    dropped = [m for m in messages if id(m) not in kept]
    if not dropped:
        return trimmed
    try:
        summary = await summarizer(dropped)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - compaction must never break a run
        return trimmed
    if not summary or not str(summary).strip():
        return trimmed
    block = {
        "role": "system",
        "content": (
            "[Background: earlier context was compacted for the provider "
            "window.]\n" + str(summary).strip()
        ),
    }
    out: list[dict[str, Any]] = []
    marker_seen = False
    for m in trimmed:
        if (
            not marker_seen
            and m.get("role") == "system"
            and CONTEXT_TRIM_MARKER in str(m.get("content") or "")
        ):
            out.append(block)
            marker_seen = True
            continue
        out.append(m)
    if not marker_seen:
        out.insert(0, block)
    return out


def parse_tool_arguments(raw: Any, tool_name: str = "") -> dict[str, Any] | None:
    """Best-effort repair of an LLM tool arguments payload.

    Returns the dict to execute with, or None when the payload cannot be
    salvaged (the caller then skips the call and reports the raw text instead
    of silently running the tool with empty arguments).

    Handles: empty payload -> {}; dict pass-through (native tool_calls); JSON
    wrapped in code fences / backticks; and a body that contains one balanced
    {...} region. Non-dict JSON (bare string/list) is wrapped as {"value": ...}.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    elif s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        pass
    else:
        return obj if isinstance(obj, dict) else {"value": obj}
    start = s.find("{")
    end = s.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else {"value": obj}
    return None


_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "max context",
    "context window",
    "token limit",
    "token_limit",
    "too many tokens",
    "input is too long",
    "prompt is too long",
)

class AgentEvent:
    def __init__(self, event_type: str, data: Any):
        self.type = event_type
        self.data = data

    def to_dict(self):
        return {"type": self.type, "data": self.data}

class _TraceAwareLLMClient:
    """Transparent LLM proxy that captures safe call metadata for structured paths."""

    def __init__(self, agent: "TitanAgent", client: Any, call_role: str):
        self._agent = agent
        self._client = client
        self._call_role = call_role

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._agent._traced_chat_completion(
            self._client,
            messages,
            tools=tools,
            call_role=self._call_role,
            **kwargs,
        )


class TitanAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        mcp: MCPManager | None = None,
        memory: MemoryManager | None = None,
        skills: SkillRegistry | None = None,
        telegram: TelegramManager | None = None,
        core_memory: MemorySystem | None = None,
        core_memory_path: Path | str | None = None,
        git_root: Path | str | None = None,
        auto_commit: bool | None = None,
        checkpoint: CheckpointStore | None = None,
        checkpoint_path: Path | str | None = None,
        tool_policy: Any | None = None,
        hitl: HumanInTheLoop | None = None,
        hitl_timeout: float = 120.0,
        reviewer_llm: Any | None = None,
        tool_stats: ToolStatsCollector | None = None,
        domain: str | None = None,
        domain_manager: Any | None = None,
        trace_store: RunTraceStore | None = None,
        trace_path: Path | str | None = None,
    ):
        self.llm = llm or LLMClient()
        # Phase 21: dedicated reviewer for the critic/reflection pass. When
        # provided (or auto-built from TITAN_REVIEWER_PROVIDER/MODEL), the
        # reflection uses it instead of the generator; refinement rounds are
        # then enabled so the generator revises against the critique.
        self.reviewer_llm = reviewer_llm
        self._reviewer_client: Any | None = None
        self._reviewer_tried = False
        # Phase 22: per-tool telemetry. Defaults to the process-wide collector
        # so the server endpoint sees every agent's record; an isolated
        # collector can be injected for deterministic tests.
        self.tool_stats = tool_stats or TOOL_STATS
        # Phase 23: repeated identical-failure guard state. Always exists so
        # _emit_tool_results works standalone; run_task resets it per run.
        self._guard_failures: dict[str, int] = {}
        # Phase 33: repeated malformed-arguments state (per tool NAME). Always
        # exists so _emit_tool_results works standalone; run_task resets it.
        self._malformed_calls: dict[str, int] = {}
        # Phase 37: consecutive all-failed tool batches (dead-end detector).
        # Always exists so direct wrapper calls work standalone; run_task resets.
        self._consecutive_failed_batches: int = 0
        # Phase 38: per-run guard decision log + totals. Always exist so the
        # wrapper records decisions even without run_task; run_task resets them.
        self._guard_actions: list[dict[str, Any]] = []
        self._guard_totals: dict[str, int] = {}
        # Phase 39: in-run idempotent read cache for read-only tools
        self._read_cache: dict[str, str] = {}
        self.tools = tools or ToolRegistry()
        # Phase 09: Active Working Memory Virtualizer
        from titan_agent.core.memory.working_memory_virtualizer import (
            WorkingMemoryVirtualizer,
        )
        self._working_memory = WorkingMemoryVirtualizer()
        if hasattr(self.tools, "_working_memory_ref"):
            self.tools._working_memory_ref = self._working_memory
        # Omni-Domain adaptation framework
        from titan_agent.core.domain.manager import DomainManager
        if domain_manager is not None:
            self.domain_manager = domain_manager
        else:
            self.domain_manager = DomainManager.get_instance()
            if domain:
                self.domain_manager.switch_domain(domain)
        if hasattr(self.tools, "_domain_manager_ref"):
            self.tools._domain_manager_ref = self.domain_manager
        self.mcp = mcp or MCPManager()
        attach_mcp = getattr(self.tools, "attach_mcp_manager", None)
        if callable(attach_mcp):
            attach_mcp(self.mcp)
        self.memory = memory or MemoryManager()
        self.skills = skills or SkillRegistry()
        self.telegram = telegram or TelegramManager()
        # Phase 9: per-role tool policy (Allowed/blocked sets enforced in
        # execute_tool_unified AND reflected in the model's tool catalog).
        self.tool_policy = tool_policy
        # Phase 14: Human-in-the-loop approval gate. Wired onto the tool registry
        # (single approval point for BOTH the classic loop and the structured
        # strategy path — ToolBridge defers via defer_approval=True) so sensitive
        # tool calls can wait for explicit human consent instead of hard-failing.
        self.hitl = hitl
        self.hitl_timeout = hitl_timeout
        if hitl is not None:
            attach = getattr(self.tools, "attach_hitl", None)
            if callable(attach):
                try:
                    attach(hitl, hitl_timeout)
                except Exception as exc:  # noqa: BLE001 - HITL is best-effort
                    log.warning("could not attach HITL to tool registry: %s", exc)
        self.system_prompt = TITAN_SYSTEM_PROMPT
        # ---- Phase 4: MemGPT-style core memory + Git-first workflow ----
        self._core_memory = core_memory
        self._core_memory_loaded = core_memory is not None
        self.core_memory_path = Path(core_memory_path) if core_memory_path else (WORKSPACE_DIR / "core_memory.db")
        self.git_root = Path(git_root).resolve() if git_root else WORKSPACE_DIR
        self._auto_commit = auto_commit if auto_commit is not None else (
            os.getenv("TITAN_GIT_AUTO_COMMIT", "").strip().lower() in ("1", "true", "yes")
        )
        # ---- Phase 5: Devin-style session checkpoints / resume ----
        self._checkpoint = checkpoint
        self._checkpoint_loaded = checkpoint is not None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (WORKSPACE_DIR / "checkpoints.db")
        # Structured traces persist operational metadata only (never prompts,
        # tool arguments/results, or exception text).
        self._trace_store = trace_store
        self._trace_store_loaded = trace_store is not None
        self.trace_path = (
            Path(trace_path)
            if trace_path is not None
            else self.checkpoint_path.with_name("agent_traces.db")
        )

    @property
    def core_memory(self) -> MemorySystem | None:
        """Phase 4: episodic/semantic store, lazily opened on first use so
        construction never touches disk (memory must never break the agent)."""
        if self._core_memory_loaded:
            return self._core_memory
        self._core_memory_loaded = True
        try:
            self._core_memory = MemorySystem(self.core_memory_path)
        except Exception as exc:  # noqa: BLE001 - memory is best-effort
            log.warning("core memory unavailable: %s", exc)
            self._core_memory = None
        return self._core_memory

    @property
    def auto_commit(self) -> bool:
        """Whether completed runs auto-commit workspace changes (Aider-style)."""
        return self._auto_commit

    @property
    def checkpoint_store(self) -> CheckpointStore | None:
        """Phase 5: Devin-style checkpoint store, lazily opened on first use."""
        if self._checkpoint_loaded:
            return self._checkpoint
        self._checkpoint_loaded = True
        try:
            self._checkpoint = CheckpointStore(self.checkpoint_path)
        except Exception as exc:  # noqa: BLE001 - checkpointing is best-effort
            log.warning("checkpoint store unavailable: %s", exc)
            self._checkpoint = None
        return self._checkpoint

    def _build_memory_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_save",
                    "description": "Saves a fact or piece of information into Titan's long-term persistent memory so it is remembered in all future sessions (e.g. user name, preferences, decisions). Use 'scope' to target a Memory Vault scope: global (everything), project (this repo), team (shared), user (personal).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Unique short key for the fact (e.g. 'user_name')."},
                            "value": {"type": "string", "description": "The fact content to remember."},
                            "category": {"type": "string", "description": "Optional category (e.g. 'profile', 'project', 'preference')."},
                            "scope": {"type": "string", "description": "Memory Vault scope: global, project, team, user (default global)."}
                        },
                        "required": ["key", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Searches Titan's long-term persistent memory for facts saved in earlier sessions. Set 'scope' to search only that Memory Vault scope, or omit to search everything.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text to search for in remembered facts."},
                            "limit": {"type": "integer", "description": "Max results to return (default 5)."},
                            "scope": {"type": "string", "description": "Optional Memory Vault scope filter: global, project, team, user."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vault_list",
                    "description": "Lists facts in the Memory Vault, optionally filtered by scope (global/project/team/user).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scope": {"type": "string", "description": "Optional scope filter: global, project, team, user."},
                            "limit": {"type": "integer", "description": "Max results (default 50)."}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handoff_create",
                    "description": "Leaves a handoff note for the next agent / session (Hermes handoff pattern): a short 'where things stand + what to do next' pass-along.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short title for the handoff."},
                            "content": {"type": "string", "description": "The handoff note: current state, decisions, next steps."},
                            "scope": {"type": "string", "description": "Optional scope: global, project, team, user (default global)."}
                        },
                        "required": ["title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handoff_list",
                    "description": "Lists open handoff notes left by previous agents/sessions (use at the START of a task to continue prior work).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "description": "Optional status filter: open or resolved (default open)."}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "handoff_resolve",
                    "description": "Marks a handoff note as resolved/cancelled once its work is complete.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "The handoff id."},
                            "status": {"type": "string", "description": "resolved or cancelled (default resolved)."}
                        },
                        "required": ["id"]
                    }
                }
            }
        ]

    def _build_skill_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "skills_list",
                    "description": "Lists all available skill playbooks (Hermes-style reusable workflows: research-ops, github-ops, coding-rules, terminal-ops, security-ops, planning-ops, ...).",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "skill_load",
                    "description": "Loads the full text of a named skill playbook so you can follow its workflow exactly.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The skill name (e.g. 'research-ops')."}
                        },
                        "required": ["name"]
                    }
                }
            }
        ]

    def _build_telegram_tool_definitions(self) -> list[dict[str, Any]]:
        """Telegram account manager tools. Every tool is consent-gated: it refuses
        to run unless TITAN_TELEGRAM_ENABLED=true in .env, and sending is limited
        to TITAN_TELEGRAM_SEND_ALLOWLIST targets."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "telegram_status",
                    "description": "Shows whether Telegram control is enabled, credentials are set, how many accounts are registered, and the send-allowlist. No secrets are ever shown.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_accounts",
                    "description": "Lists the user's registered Telegram account sessions (labels only, with masked phone/username). Never shows credentials.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_login_start",
                    "description": "Starts logging in a NEW Telegram account under a label: Telegram sends a one-time code to the given phone. Then call telegram_login_confirm with the code the USER received. Never guess the code — only use a code the user explicitly provides.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Short name for this account, e.g. 'work'."},
                            "phone": {"type": "string", "description": "The user's phone number in international format, e.g. +998901234567."}
                        },
                        "required": ["label", "phone"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_login_confirm",
                    "description": "Completes a telegram_login_start with the one-time code that Telegram sent to the user's phone and that the USER explicitly provided.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "The same label used in telegram_login_start."},
                            "code": {"type": "string", "description": "The one-time login code the user received and shared."}
                        },
                        "required": ["label", "code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_send",
                    "description": "Sends a Telegram message from a registered account to ONE target. Works ONLY if the target is in TITAN_TELEGRAM_SEND_ALLOWLIST in .env — otherwise it is refused. Never used for broadcasting.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Account label."},
                            "target": {"type": "string", "description": "Recipient username (e.g. 'titan_bot') or numeric id."},
                            "text": {"type": "string", "description": "Message text."}
                        },
                        "required": ["label", "target", "text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_recent",
                    "description": "Read-only: returns the most recent messages from a registered account's own dialogs (senders masked).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Account label."},
                            "limit": {"type": "integer", "description": "Max messages (default 10, max 25)."}
                        },
                        "required": ["label"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "telegram_logout",
                    "description": "Removes a registered Telegram account session. Set delete=true to also log the account out of Telegram entirely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Account label to remove."},
                            "delete": {"type": "boolean", "description": "true = log out on Telegram too; default false = remove local session only."}
                        },
                        "required": ["label"]
                    }
                }
            },
        ]

    def _build_git_tool_definitions(self) -> list[dict[str, Any]]:
        """Git tools — Aider-style git-first workflow (read-only + explicit commit)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Shows the current git working-tree state (modified/untracked files) for the workspace repository.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Shows uncommitted changes (diff --stat) in the workspace repository.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Commits all current workspace changes with a descriptive message (Aider-style git-first workflow). Use after writing or editing files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Concise commit message describing the change."}
                        },
                        "required": ["message"],
                    },
                },
            },
        ]

    def _build_tool_stats_definition(self) -> list[dict[str, Any]]:
        """Phase 22: the agent can query its own tool execution record mid-run
        and adapt (e.g. stop retrying a tool that keeps failing)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "tool_stats",
                    "description": "Returns a JSON summary of YOUR tool usage so far in this session: per-tool calls, successes, failures, error rate, avg latency and last error. Use it when you are unsure whether a tool keeps failing — if a tool's error rate is high, switch approach instead of repeating it.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ]

    def _build_tools_list(self) -> list[dict[str, Any]]:
        all_tools = list(self.tools.get_tool_definitions())
        # Add memory tools (agent-level, routed through MemoryManager)
        all_tools.extend(self._build_memory_tool_definitions())
        # Add skill tools (agent-level, routed through SkillRegistry)
        all_tools.extend(self._build_skill_tool_definitions())
        # Add Telegram tools (agent-level, consent-gated through TelegramManager)
        all_tools.extend(self._build_telegram_tool_definitions())
        # Add Git tools (agent-level, Aider-style git-first workflow)
        all_tools.extend(self._build_git_tool_definitions())
        # Phase 22: per-tool telemetry introspection (agent-level)
        all_tools.extend(self._build_tool_stats_definition())
        # Add MCP tools if connected
        all_tools.extend(self.mcp.get_all_tools())
        # Phase 9: drop tools the role may not use (visible AND enforced).
        if self.tool_policy is not None:
            all_tools = self.tool_policy.filter_definitions(all_tools)
        return all_tools

    def _policy_allows(self, name: str) -> bool:
        return self.tool_policy is None or self.tool_policy.allows(name)

    def _build_tool_catalog_text(self) -> str:
        """Compact live tool catalog appended to the system prompt each turn (cached for efficiency)."""
        cached = getattr(self, "_catalog_cache", None)
        if cached is not None:
            return cached
        lines = []
        for t in self.tools.get_tool_definitions():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        for t in self._build_memory_tool_definitions():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        for t in self._build_skill_tool_definitions():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        for t in self._build_telegram_tool_definitions():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        for t in self._build_git_tool_definitions():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        for t in self._build_tool_stats_definition():
            fn = t.get("function", {})
            if not self._policy_allows(str(fn.get("name", ""))):
                continue
            params = fn.get("parameters", {}).get("properties", {})
            param_hint = ", ".join(params.keys()) if params else "no params"
            lines.append(f"- {fn.get('name')}({param_hint}): {fn.get('description', '')}")
        mcp_tools = self.mcp.get_all_tools()
        if mcp_tools:
            lines.append("\nMCP server tools:")
            for t in mcp_tools:
                fn = t.get("function", {})
                if not self._policy_allows(str(fn.get("name", ""))):
                    continue
                lines.append(f"- {fn.get('name')}: {fn.get('description', '')}")
        res = "\n".join(lines)
        self._catalog_cache = res
        return res

    def _build_structured_context(self, user_input: str, mode: str, effort: str) -> str:
        """System context for the Phase 3 structured engines: agent identity,
        live tool catalog, auto-recalled facts and skill playbooks. Mirrors the
        classic loop's system assembly so structured runs stay as informed."""
        content = (
            self.system_prompt
            + "\n\n### LIVE TOOL CATALOG (all tools currently available):\n"
            + self._build_tool_catalog_text()
        )
        # Omni-Domain adaptation: inject active industry domain persona & guardrails
        if hasattr(self, "domain_manager") and self.domain_manager:
            domain_overlay = self.domain_manager.build_system_overlay()
            if domain_overlay:
                content += "\n\n" + domain_overlay
        recalled = self.memory.recall_relevant(user_input, limit=5)
        if recalled:
            content += "\n\n### REMEMBERED FACTS (from long-term memory, relevant to this request):\n"
            for f in recalled:
                content += f"- [{f['category']}] {f['key']}: {f['value']}\n"
            content += "(Use these facts as true context; do not claim you read them fresh.)"
        core_block = self._core_recall_block(user_input)
        if core_block:
            content += core_block
        skill_block = self.skills.build_system_block(user_input)
        if skill_block:
            content += skill_block
        if mode == "deep":
            content += "\n\n" + DEEP_THINKING_PROMPT
        elif mode == "deep_search":
            content += "\n\n" + DEEP_SEARCH_PROMPT
        if effort in EFFORT_PROMPTS:
            content += "\n\n" + EFFORT_PROMPTS[effort]
        return content

    def _core_recall_block(self, query: str, limit: int = 3) -> str:
        """Past runs & lessons from Phase 4 core memory. Empty unless records
        exist, so the classic loop's prompt is untouched for fresh stores."""
        mem = self.core_memory  # lazy open
        if mem is None:
            return ""
        try:
            hits = mem.recall(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 - recall must never break a run
            log.debug("core recall failed: %s", exc)
            return ""
        if not hits:
            return ""
        lines = [f"- {r.content[:300]}" for r in hits]
        return "\n\n### PAST RUNS & LESSONS (agent memory, relevant to this task):\n" + "\n".join(lines)

    async def _finalize_run(
        self,
        session_id: str,
        user_input: str,
        final_text: str,
        mode: str,
        strategy: str,
        auto_commit: bool,
    ) -> None:
        """After a completed run: record an episodic core-memory entry and,
        when auto-commit is on, commit workspace changes (Aider-style)."""
        if final_text:
            mem = self.core_memory  # lazy open
            if mem is not None:
                try:
                    mem.remember(
                        content=f"task[{session_id}]: {user_input[:200]}\nresult: {final_text[:1200]}",
                        kind=MemoryKind.EPISODIC,
                        importance=0.4,
                        scope="agent",
                        metadata={"session": session_id, "mode": mode, "strategy": strategy},
                    )
                except Exception as exc:  # noqa: BLE001 - memory is best-effort
                    log.warning("core memory write failed: %s", exc)
        if auto_commit and final_text:
            try:
                approval = await self._approval_gate(
                    "git_commit",
                    {"message": f"agent: {user_input[:60]}", "automatic": True},
                )
                if approval is False or (approval is None and not full_access_enabled()):
                    log.warning("automatic commit skipped: human approval was unavailable or denied")
                    return
                await asyncio.to_thread(git_auto_commit, self.git_root, user_input)
            except Exception as exc:  # noqa: BLE001 - commit must never kill the run
                log.warning("auto-commit failed: %s", exc)

    @property
    def trace_store(self) -> RunTraceStore | None:
        """Lazily open the local privacy-minimizing run trace database."""
        if self._trace_store_loaded:
            return self._trace_store
        self._trace_store_loaded = True
        try:
            self._trace_store = RunTraceStore(self.trace_path)
        except Exception as exc:  # noqa: BLE001 - tracing must not break agent work
            log.warning("run trace store unavailable: %s", exc)
            self._trace_store = None
        return self._trace_store

    def _record_trace_event(
        self,
        event_type: str,
        *,
        component: str = "agent",
        status: str | None = None,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort trace write; a telemetry failure never blocks the run."""
        if current_run_id() is None:
            return
        try:
            store = self.trace_store
            if store is not None:
                store.record(
                    event_type,
                    component=component,
                    status=status,
                    duration_ms=duration_ms,
                    details=details,
                )
        except Exception as exc:  # noqa: BLE001 - trace must remain non-fatal
            log.debug("run trace write failed for %s: %s", event_type, exc)

    @staticmethod
    def _pending_tool_names(messages: list[dict[str, Any]]) -> list[str]:
        """Names of tool calls that have no persisted tool response yet."""
        completed_ids = {
            str(message.get("tool_call_id"))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        }
        pending: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id", ""))
                function = call.get("function") or {}
                name = str(function.get("name") or "unknown tool")
                if call_id not in completed_ids:
                    pending.append(name)
        return list(dict.fromkeys(pending))

    def _checkpoint_load(self, session_id: str) -> RunCheckpoint | None:
        """Best-effort load of a session's checkpoint; never raises."""
        store = self.checkpoint_store
        if store is None:
            return None
        try:
            return store.load(session_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("checkpoint load failed for %s: %s", session_id, exc)
            return None

    def _checkpoint_save(
        self,
        *,
        session_id: str,
        user_input: str,
        mode: str,
        effort: str,
        strategy: str,
        messages: list[dict[str, Any]],
        steps_done: int = 0,
        tools_used: list[str] | None = None,
        status: str = "running",
        final_answer: str | None = None,
    ) -> bool:
        """Persist the live state; return whether SQLite confirmed the checkpoint."""
        store = self.checkpoint_store
        if store is None:
            return False
        try:
            trimmed = list(messages)[-CHECKPOINT_MAX_MESSAGES:]
            store.save(
                RunCheckpoint(
                    session_id=session_id,
                    user_input=user_input,
                    mode=mode,
                    effort=effort,
                    strategy=strategy,
                    messages=trimmed,
                    steps_done=int(steps_done),
                    tools_used=[str(t) for t in (tools_used or [])],
                    status=status,
                    final_answer=final_answer,
                )
            )
            return True
        except Exception as exc:  # noqa: BLE001 - checkpointing failure blocks tool execution
            log.warning("checkpoint save failed for %s: %s", session_id, exc)
            return False

    async def _approval_gate(self, name: str, args: dict[str, Any]) -> bool | None:
        """Phase 14: single HITL approval decision point for sensitive tools.

        Returns:
          None  -> no approval is required (or FULL access explicitly bypasses it)
          True  -> explicitly APPROVED by the human
          False -> denied, unavailable, timed out, or errored (caller must not execute)
        """
        try:
            if full_access_enabled():
                return None  # Explicit FULL_ACCESS bypass is intentionally preserved.
            is_shell_tool = name in {"execute_command", "tool_execute_command", "self_heal"}
            policy_action = "execute_command" if is_shell_tool else name
            resource = str(args.get("command", "")) if is_shell_tool else name
            decision = PolicyEngine().check(
                policy_action,
                resource,
                json.dumps(args, default=str),
                access=PolicyEngine.ACCESS_NORMAL,
            )
            if decision.decision == Decision.DENY:
                self._record_trace_event(
                    "approval_result",
                    component="approval",
                    status="blocked",
                    details={"tool_name": name, "decision": "policy_deny"},
                )
                return False

            # External integrations and high-impact controls require a live
            # human decision unless the operator explicitly enabled FULL_ACCESS.
            # MCP tool annotations are hints only; arbitrary servers are not
            # trusted to self-certify that a call is harmless.
            approval_required_tools = {
                "telegram_send", "telegram_login_start", "telegram_login_confirm", "telegram_logout",
                "download_file", "tool_download_file",
                "git_commit", "git_create_branch", "git_create_pr",
                "screenshot", "take_screenshot",
                "self_update", "synthesize_tool", "tool_synthesize_tool",
                "browser_click", "browser_type", "key_press", "mouse_click",
                "clipboard_set", "window_control", "launch_application",
                "start_http_server", "stop_http_server", "blender_execute_script",
            }
            tool_registry = getattr(self.tools, "delegate", self.tools)
            synthesized_names = set(getattr(tool_registry, "_synthesized_definitions", {}))
            requires_explicit_approval = (
                decision.decision == Decision.REQUIRE_APPROVAL
                or name.startswith("mcp_")
                or name in approval_required_tools
                or name in synthesized_names
                or (
                    name in {"docker_sandbox_run", "tool_docker_sandbox_run"}
                    and (
                        str(args.get("network", "none")).lower() == "bridge"
                        or bool(args.get("mount_workspace"))
                    )
                )
                or (
                    name == "manage_processes"
                    and str(args.get("action", "list")).lower() not in {"list", "status"}
                )
            )
            if not requires_explicit_approval:
                return None
            self._record_trace_event(
                "approval_requested",
                component="approval",
                status="pending",
                details={"tool_name": name},
            )
            if self.hitl is None:
                log.warning("approval required for %s but no HITL manager is configured", name)
                self._record_trace_event(
                    "approval_result",
                    component="approval",
                    status="denied",
                    details={"tool_name": name, "decision": "unavailable"},
                )
                return False
            req = self.hitl.request(
                name,
                resource,
                details={"args": args},
                reason="; ".join(decision.reasons or []) or "requires human approval",
            )
            req = await self.hitl.wait(req, timeout=self.hitl_timeout)
            approved = req.status == ApprovalStatus.APPROVED
            self._record_trace_event(
                "approval_result",
                component="approval",
                status="approved" if approved else "denied",
                details={"tool_name": name, "decision": req.status.value},
            )
            return approved
        except Exception as exc:  # noqa: BLE001 - errors must never authorize a sensitive action
            self._record_trace_event(
                "approval_result",
                component="approval",
                status="failed_closed",
                details={"tool_name": name, "decision": "error", "error_type": type(exc).__name__},
            )
            log.warning("approval gate failed closed for %s: %s", name, exc)
            return False

    def _written_paths(self, messages: list[dict[str, Any]]) -> list[str]:
        """Deduplicated file paths written/edited this run, extracted from the
        real write/edit tool calls in the conversation (Phase 31). Order is
        first-seen, duplicates dropped."""
        files: list[str] = []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") in WRITE_TOOL_NAMES:
                    try:
                        parsed = json.loads(str(fn.get("arguments") or "{}"))
                        path = parsed.get("path") or parsed.get("file")
                        if path:
                            files.append(str(path))
                    except (TypeError, ValueError):
                        pass
        return list(dict.fromkeys(files))

    def _cap_tool_result(self, result: Any) -> str:
        """Phase 34: bound the size of a tool result that reaches the model's
        conversation, so a huge tool output (a log dump, a whole-file read)
        cannot flood the context window. Keeps an explicit truncation marker
        with the original length so the model knows output was cut and how
        large it actually was."""
        text = str(result)
        limit = tool_result_max_chars()
        if len(text) <= limit:
            return text
        return (
            f"{text[:limit]}\n… [tool output truncated: {len(text)} chars total, "
            f"showing first {limit}]"
        )

    def _build_tool_evidence(self, messages: list[dict[str, Any]], limit: int = 12) -> str:
        """Phase 25 + 28: a deterministic list of what ACTUALLY happened with
        tools, built from the real tool round-trips already in the conversation
        and fed to the critic so it argues against facts instead of vibes. No
        LLM call, no parsing heuristics beyond the message shape the harness
        itself wrote. Phase 28 adds a caution section sourced from the per-run
        repeated-failure guard counters, so the critic also sees which exact
        calls kept failing."""
        files = self._written_paths(messages)
        lines: list[str] = []
        for m in messages:
            if m.get("role") != "tool":
                continue
            tname = str(m.get("name") or "")
            if not tname:
                continue
            snippet = str(m.get("content") or "").replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            lines.append(f"- {tname}: {snippet or '(no output)'}")
        parts = [
            "\n#### TOOL EVIDENCE - facts that actually happened this run:",
            "\n".join(lines[:limit]) if lines else "(no tool results yet)",
        ]
        if files:
            parts.append(
                "Files written/edited: " + ", ".join(dict.fromkeys(files))
            )
        # Phase 28: repeated-failure caution from the per-run guard counters
        # (tools whose identical call already failed >=2 times this run).
        repeats = sorted(
            (
                (k.split("\x00", 1)[0], v)
                for k, v in self._guard_failures.items()
                if v >= 2
            ),
            key=lambda kv: -kv[1],
        )
        if repeats:
            caution = [
                (
                    "⚠ Repeated failures this run — do NOT repeat these calls; "
                    "change the arguments, switch tools, or fix the prerequisites:"
                )
            ]
            for tname, count in repeats[:6]:
                caution.append(f"- {tname}: failed {count}×")
            parts.append("\n".join(caution))
        return "\n".join(parts)

    def _run_used_write_tool(self, messages: list[dict[str, Any]]) -> bool:
        """Phase 26: did this run write or edit files? Scans the real tool
        round-trips (recent writes always survive compaction)."""
        return any(
            str(m.get("name") or "") in WRITE_TOOL_NAMES
            for m in messages
            if m.get("role") == "tool"
        )

    def _record_batch_outcome(
        self, messages: list[dict[str, Any]], before_len: int
    ) -> None:
        """Phase 37: feed the dead-end detector one executed tool batch. A batch
        counts as wasted when EVERY tool result in it failed (error, skip, or
        guard block) — no evidence of progress. Consecutive wasted batches build
        towards the early-stop window; any successful call resets the streak.
        The slice of `messages` added since `before_len` are exactly the tool
        results this batch produced (the harness wrote them itself)."""
        added = messages[before_len:]
        tool_msgs = [m for m in added if m.get("role") == "tool"]
        if not tool_msgs:
            return
        failed = sum(
            1
            for m in tool_msgs
            if _is_failed_tool_result(str(m.get("content") or ""))
        )
        if failed == len(tool_msgs):
            self._consecutive_failed_batches = (
                getattr(self, "_consecutive_failed_batches", 0) + 1
            )
        else:
            self._consecutive_failed_batches = 0

    def _record_guard_action(self, kind: str, tool: str, **extra: Any) -> None:
        """Phase 38: append one entry to the per-run guard decision log (capped)
        and bump the matching total, so operators can see what the harness
        actually blocked and when, not just the current counter values."""
        self._guard_actions.append({"kind": kind, "tool": tool, **extra})
        if len(self._guard_actions) > _GUARD_ACTION_CAP:
            del self._guard_actions[0]
        self._guard_totals[kind] = self._guard_totals.get(kind, 0) + 1

    def _guard_key(self, t_name: str, t_args: dict[str, Any]) -> str:
        """Canonical identity of a tool call for the repeated-failure guard:
        tool name + sorted-key JSON of the arguments, so any argument change
        resets the per-pattern counter."""
        try:
            canonical = json.dumps(t_args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = repr(t_args)
        return f"{t_name}\x00{canonical}"

    async def execute_tool_unified(self, name: str, args: dict[str, Any]) -> str:
        # Phase 22 + 27: telemetry + repeated-failure guard wrapper. EVERY tool
        # execution funnel — classic loop, structured engines (ToolBridge),
        # cron, queue, direct calls — goes through here, so one shared collector
        # and one shared guard counter govern all of them.
        # Omni-Domain tool restriction check
        if hasattr(self, "domain_manager") and self.domain_manager:
            allowed, reason = self.domain_manager.is_tool_allowed(name)
            if not allowed:
                return f"Error: {reason}"

        guarded = repeat_guard_enabled()
        gkey = None
        prior = 0
        if guarded or name in READ_CACHEABLE_TOOLS:
            gkey = self._guard_key(name, args)

        # Phase 39: Idempotent in-run read caching — instant 0ms return for repeated reads
        if name in READ_CACHEABLE_TOOLS and gkey:
            read_cache = getattr(self, "_read_cache", None)
            if read_cache is not None and gkey in read_cache:
                return read_cache[gkey]

        if guarded and gkey:
            prior = self._guard_failures.get(gkey, 0)
            if prior >= repeat_guard_limit():
                # Phase 38: log the decision so the guard's effect is visible.
                self._record_guard_action("blocked_repeat", name, failures=prior)
                return (
                    "Error: repeated tool failure guard — this exact call "
                    f"(tool='{name}') already failed {prior} times in this run. "
                    "Change the arguments, use a different tool, or verify the "
                    "prerequisites first."
                )
        start = time.monotonic()
        self._record_trace_event(
            "tool_started",
            component="tool",
            status="started",
            details={"tool_name": name},
        )
        try:
            result = await self._execute_tool_unified(name, args)
        except Exception as exc:
            self._record_trace_event(
                "tool_failed",
                component="tool",
                status="failed",
                duration_ms=(time.monotonic() - start) * 1000,
                details={"tool_name": name, "error_type": type(exc).__name__},
            )
            self.tool_stats.record(
                name,
                ok=False,
                latency_ms=(time.monotonic() - start) * 1000.0,
                error=type(exc).__name__,
                output_chars=0,
            )
            if guarded and gkey:
                self._guard_failures[gkey] = prior + 1
            raise
        ok = not (isinstance(result, str) and result.startswith("Error"))
        self._record_trace_event(
            "tool_finished" if ok else "tool_failed",
            component="tool",
            status="success" if ok else "failed",
            duration_ms=(time.monotonic() - start) * 1000,
            details={"tool_name": name},
        )
        self.tool_stats.record(
            name,
            ok=ok,
            latency_ms=(time.monotonic() - start) * 1000.0,
            error=None if ok else "tool_error",
            output_chars=len(result) if isinstance(result, str) else 0,
        )
        if guarded and gkey:
            if ok:
                self._guard_failures.pop(gkey, None)  # success forgives
            else:
                self._guard_failures[gkey] = prior + 1  # single counting point

        # Cache management: store successful read results; invalidate all on any mutating write
        if hasattr(self, "_read_cache"):
            if ok and name in READ_CACHEABLE_TOOLS and gkey:
                self._read_cache[gkey] = str(result)
            elif name in WRITE_TOOL_NAMES or name in ("execute_command", "tool_execute_command", "self_heal", "sandbox_execute"):
                self._read_cache.clear()

        return result

    async def _execute_tool_unified(self, name: str, args: dict[str, Any]) -> str:
        # Phase 9: per-role tool policy enforced for EVERY tool family
        # (terminal / memory / skill / telegram / git / mcp) — a researcher
        # cannot commit, a reviewer cannot write.
        if not self._policy_allows(name):
            return f"Error: tool '{name}' is outside this subagent's role and was blocked by tool policy."
        # Phase 14: Human-in-the-loop approval gate. Executes for every tool
        # call funnel (classic loop, structured strategy, cron, queue) so the
        # live agent truly waits for a human instead of silently doing nothing.
        # Sensitive actions fail closed if no HITL manager exists or the gate errors.
        granted = await self._approval_gate(name, args)
        if granted is False:
            return (
                "Error: approval required or action blocked by policy; not executed "
                f"(tool='{name}' was not executed)."
            )
        # Dual-Shield Cyber Defense Sentinel Check
        if name in ("execute_command", "tool_execute_command", "docker_sandbox_run", "tool_docker_sandbox_run"):
            cmd = str(args.get("command") or args.get("cmd") or "")
            if cmd:
                from titan_agent.core.security.dual_shield import DualShieldOrchestrator
                shield = DualShieldOrchestrator.get_instance()
                allowed, block_msg, _red_resp = await shield.evaluate_command(cmd, session_id="tool_exec", llm_client=self.llm)
                if not allowed:
                    return f"Error: {block_msg}"
        if name == "tool_stats":
            return json.dumps(self.tool_stats.summary(), ensure_ascii=False)
        if name == "memory_save":
            key = str(args.get("key", "")).strip()
            value = str(args.get("value", "")).strip()
            category = str(args.get("category", "agent")).strip() or "agent"
            scope = args.get("scope")
            if not key or not value:
                return "Error: memory_save requires both 'key' and 'value'."
            self.memory.remember_fact(key, value, category, scope=scope)
            scope_note = f" (scope: {scope})" if scope else ""
            return f"Saved to memory: {key} = {value} (category: {category}){scope_note}"
        if name == "memory_search":
            query = str(args.get("query", "")).strip()
            limit = int(args.get("limit", 5) or 5)
            scope = args.get("scope")
            if not query:
                return "Error: memory_search requires 'query'."
            facts = self.memory.search_knowledge(query, limit=limit, scope=scope)
            if not facts:
                return "No matching facts found in memory."
            return "\n".join(
                f"- [{f.get('category')}] {f['key']}: {f['value']}" for f in facts
            )
        if name == "vault_list":
            scope = args.get("scope")
            limit = int(args.get("limit", 50) or 50)
            facts = self.memory.vault_list(scope=scope, limit=limit)
            if not facts:
                return "Memory Vault is empty."
            return "\n".join(
                f"- [{f['scope']}/{f['category']}] {f['key']}: {f['value']}" for f in facts
            )
        if name == "handoff_create":
            title = str(args.get("title", "")).strip()
            content = str(args.get("content", "")).strip()
            scope = args.get("scope")
            if not title or not content:
                return "Error: handoff_create requires both 'title' and 'content'."
            hid = self.memory.create_handoff(title, content, scope=scope or "global")
            return f"Handoff created (id: {hid}): {title}"
        if name == "handoff_list":
            status = args.get("status")
            msgs = self.memory.list_handoffs(status=status or "open")
            if not msgs:
                return "No handoff notes found."
            return "\n".join(
                f"- [{m['id']}] ({m['status']}) {m['title']}:\n  {m['content']}" for m in msgs
            )
        if name == "handoff_resolve":
            hid = int(args.get("id", 0) or 0)
            status = str(args.get("status", "resolved")).strip()
            if hid <= 0:
                return "Error: handoff_resolve requires a valid 'id'."
            ok = self.memory.resolve_handoff(hid, status=status)
            return f"Handoff {hid} marked {status}." if ok else f"Handoff {hid} not found."
        if name == "skills_list":
            skills = self.skills.list_skills()
            if not skills:
                return "No skill playbooks available."
            return "\n".join(
                f"- {s['name']}: {s['description']}" for s in skills
            )
        if name == "skill_load":
            skill_name = str(args.get("name", "")).strip()
            if not skill_name:
                return "Error: skill_load requires 'name'."
            skill = self.skills.get_skill(skill_name)
            if skill is None:
                names = ", ".join(s["name"] for s in self.skills.list_skills()) or "none loaded"
                return f"Unknown skill '{skill_name}'. Available: {names}"
            return skill.full_text()
        if name.startswith("telegram_"):
            try:
                return await self._dispatch_telegram(name, args)
            except TelegramError as e:
                return f"Telegram: {e}"
        if name.startswith("git_"):
            return await self._dispatch_git(name, args)
        if name.startswith("mcp_"):
            return await self.mcp.execute_tool(name, args)
        else:
            return await self.tools.execute_tool(name, args)

    async def _dispatch_git(self, name: str, args: dict[str, Any]) -> str:
        """Aider-style git tools. Blocking git calls run in a worker thread so
        the event loop keeps streaming."""
        root = self.git_root
        if name == "git_status":
            return await asyncio.to_thread(git_status, root)
        if name == "git_diff":
            return await asyncio.to_thread(git_diff, root)
        if name == "git_commit":
            message = str(args.get("message", "")).strip()
            if not message:
                return "Error: git_commit requires a 'message'."
            return await asyncio.to_thread(git_commit, root, message)
        return f"Unknown git tool '{name}'."

    async def _dispatch_telegram(self, name: str, args: dict[str, Any]) -> str:
        """Consent-gated Telegram dispatch. Every call is wrapped by the caller
        with TelegramError -> friendly message."""
        tg = self.telegram
        if name == "telegram_status":
            s = tg.status()
            allow = ", ".join(s["send_allowlist"]) or "(read-only — allowlist empty)"
            state = "ENABLED" if s["enabled"] else "DISABLED (set TITAN_TELEGRAM_ENABLED=true in .env)"
            creds = "set" if s["credentials_set"] else "MISSING (TITAN_TELEGRAM_API_ID / _HASH in .env)"
            return (
                f"Telegram control: {state}\n"
                f"API credentials: {creds}\n"
                f"Registered accounts: {s['sessions']}\n"
                f"Send allowlist: {allow}\n"
                f"Session dir: {s['session_dir']}"
            )
        if name == "telegram_accounts":
            accs = tg.list_accounts()
            if not accs:
                return "No Telegram accounts registered yet. Use telegram_login_start to add one."
            return "\n".join(
                f"- {a['label']} | phone: {a['phone']} | username: {a['username']} (added {a.get('added','?')})"
                for a in accs
            )
        if name == "telegram_login_start":
            label = str(args.get("label", "")).strip()
            phone = str(args.get("phone", "")).strip()
            res = await tg.login_start(label, phone)
            return (
                f"Login code requested for '{res['label']}'. "
                f"IMPORTANT: ask the user for the code Telegram sent to their phone "
                f"and call telegram_login_confirm(label='{res['label']}', code=...) — "
                f"never guess or reuse a code."
            )
        if name == "telegram_login_confirm":
            label = str(args.get("label", "")).strip()
            code = str(args.get("code", "")).strip()
            if not label or not code:
                return "Error: telegram_login_confirm requires 'label' and 'code'."
            res = await tg.login_confirm(label, code)
            return (
                f"Account '{res['label']}' logged in (phone: {res['phone']}, "
                f"username: {res['username']}). You can now use telegram_recent / "
                f"telegram_send (allowlist-gated)."
            )
        if name == "telegram_send":
            label = str(args.get("label", "")).strip()
            target = str(args.get("target", "")).strip()
            text = str(args.get("text", "")).strip()
            if not label or not target or not text:
                return "Error: telegram_send requires 'label', 'target' and 'text'."
            res = await tg.send_message(label, target, text)
            return f"Sent {res['chars']} chars to @{res['target']} from '{res['label']}'."
        if name == "telegram_recent":
            label = str(args.get("label", "")).strip()
            limit = int(args.get("limit", 10) or 10)
            if not label:
                return "Error: telegram_recent requires 'label'."
            msgs = await tg.recent_messages(label, limit=limit)
            if not msgs:
                return "No recent messages found for that account."
            return "\n".join(
                f"{m['n']}. [{m['date']}] {m['from']}: {m['text']}" for m in msgs
            )
        if name == "telegram_logout":
            label = str(args.get("label", "")).strip()
            delete = bool(args.get("delete", False))
            if not label:
                return "Error: telegram_logout requires 'label'."
            res = await tg.logout(label, delete=delete)
            return f"Account '{res['label']}': {res['status']}."
        return f"Unknown telegram tool '{name}'."

    async def _emit_tool_results(
        self,
        response,
        messages: list[dict[str, Any]],
        iteration: int,
        *,
        checkpoint_context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Executes all tool_calls inside `response` IN PARALLEL and streams
        events. Mutates `messages` in place.

        Phase 10: calls whose arguments cannot be parsed as JSON are SKIPPED
        (never executed with empty arguments). The raw payload is reported back
        to the model so it can resend a valid call. Every tool_call_id still
        receives exactly one follow-up tool message, keeping the assistant->tool
        pairing valid for the next model request.
        """
        assistant_msg = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": response.tool_calls
        }
        messages.append(assistant_msg)

        parsed = []
        for tool_call in response.tool_calls:
            fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            t_name = fn.get("name", "")
            t_args = parse_tool_arguments(fn.get("arguments", "{}"), t_name)
            parsed.append((tool_call, t_name, t_args))

        # Emit all scheduled tool_call events first
        for tool_call, t_name, t_args in parsed:
            yield AgentEvent("tool_call", {"name": t_name, "arguments": t_args})

        runnable = [p for p in parsed if p[2] is not None]
        skipped = [p for p in parsed if p[2] is None]

        if len(runnable) > 1:
            yield AgentEvent("status", f"Running {len(runnable)} tools in parallel...")
        elif runnable:
            yield AgentEvent("status", f"Running tool: {runnable[0][1]}...")
        elif skipped:
            yield AgentEvent(
                "status",
                f"Skipping {len(skipped)} tool call(s) with unparsable arguments.",
            )

        # Execute all runnable tools concurrently inside a parallelism cap
        # (Phase 17). Each tool's Python exception is isolated from siblings;
        # this is not OS/process security isolation. With TITAN_CANCEL_ON_TOOL_ERROR=1 the batch
        # stops as soon as any tool fails and in-flight siblings are cancelled.
        checkpoint_failed = False
        if runnable and checkpoint_context is not None:
            intent_checkpoint = {
                **checkpoint_context,
                "messages": messages,
                "steps_done": iteration,
                "tools_used": [name for _call, name, _args in runnable],
                "status": "tool_in_progress",
            }
            checkpoint_failed = not self._checkpoint_save(**intent_checkpoint)
            if checkpoint_failed:
                yield AgentEvent(
                    "error",
                    "Tool calls were not executed because the pre-action checkpoint could not be saved. "
                    "Restore checkpoint storage before retrying.",
                )

        limit = parallel_tool_calls()
        cancel_on_fail = cancel_on_tool_error()
        semaphore = asyncio.Semaphore(max(1, limit))

        async def _run_one(tool_call, t_name, t_args):
            # Phase 23 + 27: repeated identical-failure guard — the SAME call
            # that already failed `limit` times in this run is skipped (not
            # executed) and the model is told to change approach. Counting
            # happens ONCE, inside execute_tool_unified (Phase 27), so the
            # classic loop and the structured engines (ToolBridge) share one
            # counter. A blocked call returns "ok" with an error text so
            # cancel-on-failure does NOT cascade-cancel healthy siblings in
            # the same batch.
            if repeat_guard_enabled():
                key = self._guard_key(t_name, t_args)
                prior = self._guard_failures.get(key, 0)
                if prior >= repeat_guard_limit():
                    # Phase 38: the main-loop block happens HERE (before the
                    # wrapper is even reached), so the decision is logged here
                    # to keep the guard action log complete.
                    self._record_guard_action(
                        "blocked_repeat", t_name, failures=prior
                    )
                    return (
                        "ok", tool_call, t_name,
                        (
                            "Error: repeated tool failure guard — this exact call "
                            f"(tool='{t_name}') already failed {prior} times in "
                            "this run. Change the arguments, use a different tool, "
                            "or verify the prerequisites first."
                        ),
                    )
            try:
                result = await self.execute_tool_unified(t_name, t_args)
                return ("ok", tool_call, t_name, result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - crash isolation: one bad
                # tool must never kill the whole batch or the run
                return ("error", tool_call, t_name,
                        f"Error ({type(exc).__name__}): {exc!s}")

        async def _limited(p):
            async with semaphore:
                return await _run_one(*p)

        pending = (
            {asyncio.ensure_future(_limited(p)): p for p in runnable}
            if not checkpoint_failed else {}
        )
        finished: dict[asyncio.Future, tuple] = {}
        cancelled_calls: list[tuple] = []
        failed_tool: str | None = None

        # NOTE: asyncio.wait returns a plain SET of pending futures, so we keep
        # the future->meta mapping in `pending` and re-derive the wait set each
        # round instead of overwriting the dict (Phase 17).
        while pending and not failed_tool:
            done, _ = await asyncio.wait(
                list(pending), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                finished[fut] = await fut
                pending.pop(fut, None)
            if cancel_on_fail:
                for fut in done:
                    if finished[fut][0] == "error":
                        failed_tool = finished[fut][2]
                        break
            if failed_tool:
                drain = list(pending)
                for fut in drain:
                    fut.cancel()
                if drain:
                    await asyncio.gather(*drain, return_exceptions=True)
                cancelled_calls = [meta for meta in pending.values()]
                pending.clear()

        result_map = {
            id(tool_call): result
            for _status, tool_call, _name, result in finished.values()
        }
        if checkpoint_failed:
            result_map.update({
                id(tool_call): (
                    "Error: not executed because the durable pre-action checkpoint failed."
                )
                for tool_call, _name, _args in runnable
            })
        if cancelled_calls:
            result_map.update({
                id(tool_call): (
                    f"Error: cancelled - sibling tool '{failed_tool}' failed "
                    "before this call ran."
                )
                for tool_call, _name, _args in cancelled_calls
            })
            yield AgentEvent(
                "status",
                f"Cancelled {len(cancelled_calls)} remaining tool call(s) "
                f"after '{failed_tool}' failed.",
            )

        # Emit results and append tool messages in ORIGINAL call order, so every
        # tool_call_id gets exactly one follow-up message.
        for tool_call, t_name, t_args in parsed:
            if t_args is None:
                raw = ""
                if isinstance(tool_call, dict):
                    raw = str(tool_call.get("function", {}).get("arguments", ""))
                result = (
                    "Skipped: tool arguments could not be parsed as valid JSON. "
                    f"Raw arguments (truncated): {raw[:400]!r}"
                )
                # Phase 33: repeated malformed-arguments guard. Skipped calls
                # never reach execute_tool_unified, so the Phase 27 guard cannot
                # see them — a weak model could resend the SAME broken call
                # forever. Count per tool NAME and block past the limit.
                if malformed_guard_enabled():
                    count = self._malformed_calls.get(t_name, 0) + 1
                    self._malformed_calls[t_name] = count
                    if count >= malformed_guard_limit():
                        # Phase 38: log the decision so the guard's effect is
                        # visible to operators.
                        self._record_guard_action(
                            "blocked_malformed", t_name, skips=count
                        )
                        result = (
                            "Error: repeated malformed-arguments guard — this "
                            f"tool (tool='{t_name}') has now been sent {count} "
                            "times with unparsable arguments in this run. STOP "
                            "resending broken calls. Send ONE tool call with "
                            "valid JSON arguments (properly escaped quotes and "
                            "braces), or switch to a different approach."
                        )
                        yield AgentEvent(
                            "status",
                            f"Blocking malformed '{t_name}' calls "
                            f"({count}/{malformed_guard_limit()}).",
                        )
                result = self._cap_tool_result(result)
                yield AgentEvent("tool_result", {"name": t_name, "result": result})
            else:
                result = self._cap_tool_result(
                    result_map.get(id(tool_call), "Error: tool call vanished.")
                )
                yield AgentEvent("tool_result", {"name": t_name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": (
                    tool_call.get("id", f"call_{iteration}")
                    if isinstance(tool_call, dict) else f"call_{iteration}"
                ),
                "name": t_name,
                "content": str(result)
            })

    async def _traced_chat_completion(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        call_role: str,
        **kwargs: Any,
    ) -> Any:
        """Call an LLM and trace only provider/model, role, outcome, and latency."""
        started = time.monotonic()
        details = {
            "call_role": call_role,
            "provider": getattr(client, "provider", None),
            "model": getattr(client, "model", None),
        }
        self._record_trace_event(
            "model_call_started", component="llm", status="started", details=details
        )
        try:
            response = await client.chat_completion(messages, tools=tools, **kwargs)
        except BaseException as exc:
            self._record_trace_event(
                "model_call_failed",
                component="llm",
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000,
                details={**details, "error_type": type(exc).__name__},
            )
            raise
        self._record_trace_event(
            "model_call_finished",
            component="llm",
            status="success",
            duration_ms=(time.monotonic() - started) * 1000,
            details=details,
        )
        return response

    async def _summarize_context(self, dropped: list[dict[str, Any]]) -> str | None:
        """Phase 18: condense dropped messages into a short background summary.

        Returns None on ANY failure so callers fall back to the plain trim
        marker — compaction can never break or slow a run beyond one bounded
        summarizer call.
        """
        try:
            text = "\n".join(
                str(m.get("content") or "") for m in dropped
            )[:14000]
            prompt = (
                "You are compacting an agent run's working context for a "
                "smaller provider window. Summarize ONLY what was done earlier "
                "and the important facts/results/state the agent still needs, "
                "in at most 700 characters of plain text. Do not introduce new "
                "information.\n\n" + text
            )
            resp = await self._traced_chat_completion(
                self.llm,
                [{"role": "user", "content": prompt}],
                tools=[],
                call_role="context_summary",
            )
            content = (resp.content or "").strip() if resp else ""
            return content or None
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - summarization is best-effort
            return None

    def _reanchor_task(self, messages: list[dict[str, Any]], task: str | None) -> None:
        """Phase 24: right after context compaction, append a compact ORIGINAL
        TASK reminder (unless one is already present) so the model does not
        drift. Pure no-op when there is nothing to anchor, re-anchoring is
        disabled, or a reminder already exists."""
        if not task or not objective_reanchor_enabled():
            return
        if any(
            ORIGINAL_TASK_MARKER in str(m.get("content") or "")
            for m in messages
        ):
            return
        block = _anchor_block(task)
        if block:
            messages.append(block)

    def _inject_or_update_working_memory_hud(self, messages: list[dict[str, Any]]) -> None:
        """Phase 09: Pins or updates the active Working Memory HUD in the context."""
        if not hasattr(self, "_working_memory") or not self._working_memory:
            return
        hud_text = self._working_memory.render_hud_block()
        marker = "### WORKING MEMORY HUD"
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") == "system" and marker in str(msg.get("content") or ""):
                msg["content"] = hud_text
                return
        messages.append({"role": "system", "content": hud_text})

    def _critic_llm(self) -> Any:
        """The model used for the critic/reflection pass.

        Phase 21: a dedicated reviewer (constructor-injected or built lazily
        from TITAN_REVIEWER_PROVIDER / TITAN_REVIEWER_MODEL) when configured —
        the classic "weak generator + strong critic" split. Without one this
        returns the run's own model and behavior is byte-for-byte unchanged.
        """
        if self.reviewer_llm is not None:
            return self.reviewer_llm
        if not self._reviewer_tried:
            self._reviewer_tried = True
            provider, model = reviewer_model()
            if provider or model:
                try:
                    self._reviewer_client = LLMClient(
                        provider=provider or self.llm.provider,
                        model=model
                        or provider_default_model(provider)
                        or self.llm.model,
                    )
                except Exception as exc:  # noqa: BLE001 - reviewer is best-effort
                    log.warning("could not build reviewer LLM: %s", exc)
                    self._reviewer_client = None
        return self._reviewer_client or self.llm

    async def _chat_with_recovery(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        anchor: str | None = None,
    ) -> tuple[Any, list[dict[str, Any]], list[str]]:
        """Robust model call for one loop iteration.

        Returns (response, working_messages, status_notes).

        Phase 10 hardening:
        - Transient network errors (aiohttp.ClientError / OSError / timeouts)
          retry with linear backoff up to LLM_TRANSIENT_RETRIES — a single
          hiccup no longer kills an entire deep run.
        - Context-length errors trigger an automatic trim_messages_for_context
          pass + one retry, so an over-budget run degrades gracefully to a
          usable window instead of dying mid-task.
        - API-level failures (401/403/...) and context errors that trimming
          cannot fix still fail fast.
        """
        working = list(messages)
        notes: list[str] = []
        transient = (aiohttp.ClientError, OSError, asyncio.TimeoutError)
        overflow_left = 6  # bounded halving loop on repeated context overflows

        for attempt in range(LLM_TRANSIENT_RETRIES + 1):
            try:
                response = await self._traced_chat_completion(
                    self.llm, working, tools=tools, call_role="agent_turn"
                )
                return response, working, notes
            except transient as e:
                if attempt >= LLM_TRANSIENT_RETRIES:
                    raise
                notes.append(
                    f"LLM transient error, retrying ({attempt + 1}/"
                    f"{LLM_TRANSIENT_RETRIES}): {e!s}"
                )
                await asyncio.sleep(0.75 * (attempt + 1))
            except RuntimeError as e:
                lowered = str(e).lower()
                if not any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS):
                    raise
                if overflow_left <= 0:
                    raise
                overflow_left -= 1
                # Progressive halving: shrink to ~half the current size on every
                # overflow, because we do not know the provider's real window.
                # Bounded retries guarantee termination while still landing under
                # it for reasonable windows.
                current = sum(_msg_cost(m) for m in working)
                target = max(512, current // 2)
                trimmed = await compact_messages_for_context(
                    working,
                    summarizer=self._summarize_context,
                    max_chars=target,
                )
                if len(trimmed) < len(working):
                    working = trimmed
                    # Phase 24: re-pin the original task right after overflow
                    # compaction so the model does not drift mid-run.
                    self._reanchor_task(working, anchor)
                    notes.append(
                        "Context overflow detected - trimmed older tool "
                        f"rounds to ~{len(trimmed)} messages and retrying."
                    )
                    continue
                raise
        raise RuntimeError("LLM unreachable after all retries.")  # pragma: no cover

    async def run_task(
        self,
        user_input: str,
        session_id: str = "default_session",
        mode: str = "fast",
        effort: str = "auto",
        strategy: str = "auto",
        auto_commit: bool | None = None,
        resume: bool = False,
        system_extra: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run a task while journaling privacy-safe lifecycle and tool metadata."""
        run_id = uuid.uuid4().hex
        started = time.monotonic()
        token = bind_run_id(run_id)
        saw_final = False
        saw_error = False
        terminal_exception = False
        self._record_trace_event(
            "run_started",
            component="run",
            status="running",
            details={
                "mode": mode if mode in {"fast", "deep", "deep_search"} else "unknown",
                "effort": effort if effort in {"auto", "low", "medium", "high", "ultra"} else "unknown",
                "strategy": strategy if strategy in {"auto", "plan", "react", "tot"} else "unknown",
                "provider": getattr(self.llm, "provider", None),
                "model": getattr(self.llm, "model", None),
            },
        )
        try:
            async for event in self._run_task_impl(
                user_input,
                session_id=session_id,
                mode=mode,
                effort=effort,
                strategy=strategy,
                auto_commit=auto_commit,
                resume=resume,
                system_extra=system_extra,
            ):
                if event.type == "final_answer":
                    saw_final = True
                elif event.type == "error":
                    saw_error = True
                yield event
        except asyncio.CancelledError:
            terminal_exception = True
            self._record_trace_event("run_cancelled", component="run", status="cancelled")
            raise
        except GeneratorExit:
            terminal_exception = True
            self._record_trace_event("run_interrupted", component="run", status="interrupted")
            raise
        except BaseException as exc:
            terminal_exception = True
            self._record_trace_event(
                "run_failed",
                component="run",
                status="failed",
                details={"error_type": type(exc).__name__},
            )
            raise
        finally:
            if not terminal_exception and saw_final:
                self._record_trace_event(
                    "run_finished",
                    component="run",
                    status="finished_with_errors" if saw_error else "finished",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            elif not terminal_exception and not saw_error:
                self._record_trace_event(
                    "run_interrupted",
                    component="run",
                    status="interrupted",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            elif not terminal_exception:
                self._record_trace_event(
                    "run_failed",
                    component="run",
                    status="failed",
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            reset_run_id(token)

    async def _run_task_impl(
        self,
        user_input: str,
        session_id: str = "default_session",
        mode: str = "fast",
        effort: str = "auto",
        strategy: str = "auto",
        auto_commit: bool | None = None,
        resume: bool = False,
        system_extra: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Executes a user request with autonomous multi-step reasoning, tool execution,
        and a critical reflection (self-review) pass before the final answer.
        mode: "fast" | "deep" | "deep_search"
        effort: "auto" | "low" | "medium" | "high" | "ultra" — scales the iteration
                budget and rigor of the run ('auto' derives from the mode).
        strategy: "auto" | "plan" | "react" | "tot" — Phase 3 structured reasoning.
                'auto' keeps the classic prompt-driven loop (backward compatible);
                'plan' builds a step plan then executes it; 'react' uses the
                structured ReAct engine; 'tot' explores strategies with
                Tree-of-Thoughts first, then executes with tools. All structured
                strategies run the Phase 2 core engines with policy guardrails.
        auto_commit: None = instance default (env TITAN_GIT_AUTO_COMMIT); True/False
                overrides. When on, workspace changes are committed after a
                successful run (Aider-style git-first workflow).
        resume: when True, restore a saved checkpoint. Completed tool batches
                continue with their preserved tool-call messages; if recovery
                finds a batch or structured run that may have caused an
                unrecorded side effect, execution pauses for manual
                reconciliation instead of replaying it. Completed sessions
                return their saved final answer.
        Yields AgentEvent objects for real-time streaming to Web UI / CLI.
        """
        if mode not in ("fast", "deep", "deep_search"):
            mode = "fast"
        raw_effort = effort or "auto"  # budget scaling needs to know if effort was explicit
        effort = _resolve_effort(effort, mode)
        strategy = (strategy or "auto").strip().lower()
        if strategy not in ("auto", "plan", "react", "tot"):
            strategy = "auto"
        auto_commit = self._auto_commit if auto_commit is None else auto_commit

        # ---- Phase 5: completed runs return their saved result; ambiguous side effects never replay ----
        resume_cp = self._checkpoint_load(session_id) if resume else None
        if resume_cp is not None and resume_cp.status == "done" and resume_cp.final_answer:
            yield AgentEvent("status", f"Session '{session_id}' already completed — returning saved result.")
            yield AgentEvent("final_answer", resume_cp.final_answer)
            self.memory.add_message(session_id, "assistant", resume_cp.final_answer)
            return
        if resume_cp is not None and resume_cp.status in {"tool_in_progress", "structured_in_progress"}:
            pending_names = self._pending_tool_names(resume_cp.messages)
            action_summary = ", ".join(pending_names) if pending_names else "structured strategy actions"
            yield AgentEvent(
                "status",
                f"Session '{session_id}' paused for side-effect reconciliation; no pending action was replayed.",
            )
            yield AgentEvent(
                "final_answer",
                "Paused safely after an interrupted tool batch ("
                f"{action_summary}). One or more external actions may have completed before the interruption. "
                "Verify each real-world outcome first. For classic tool calls, record operator-verified results with CheckpointStore.reconcile_tool_batch before resuming; structured runs remain paused and need a fresh run. Nothing is replayed automatically.",
            )
            return

        # ---- Auto API Key Discovery & Model Fallback setup ----
        import re
        api_key_match = re.match(r"^(sk-[a-zA-Z0-9_-]{20,}|gsk_[a-zA-Z0-9_-]+|AIza[a-zA-Z0-9_-]+)$", user_input.strip())
        if api_key_match:
            yield AgentEvent("status", "API key detected! Scanning available models...")
            from titan_agent.key_discovery import discover_api_key
            import os
            key = api_key_match.group(1)
            provider, base_url, models = await discover_api_key(key)
            if provider and models:
                yield AgentEvent("status", f"Successfully connected to {provider}. Found {len(models)} models.")
                # Save first model as primary, others as fallback
                self.llm.set_model(provider, models[0], key, base_url)
                
                # Setup fallback chain globally dynamically
                self.llm.fallback_models = models[1:]
                
                yield AgentEvent("final_answer", f"API Key activated for `{provider}`.\nPrimary Model: `{models[0]}`.\nFallbacks saved: `{len(models)-1}` models (seamless switch on rate limit).")
                return
            else:
                yield AgentEvent("error", "Could not discover any models with the provided API key.")
                yield AgentEvent("final_answer", "API Key was invalid or no models available.")
                return


        # Dual-Shield Cyber Defense Evaluation (Blue Team continuous + Emergency Red Team)
        from titan_agent.core.security.dual_shield import DualShieldOrchestrator
        shield = DualShieldOrchestrator.get_instance()
        allowed, block_msg, red_resp = await shield.evaluate_prompt(user_input, session_id=session_id, llm_client=self.llm)
        if not allowed:
            yield AgentEvent("error", block_msg)
            if red_resp:
                yield AgentEvent("thought", f"⚔️ Emergency Red Team Triggered:\n{red_resp.forensic_summary}")
                yield AgentEvent("final_answer", f"🚫 **Task Blocked by Dual-Shield Cyber Defense**\n\n{block_msg}\n\n```\n{red_resp.forensic_summary}\n```")
            else:
                yield AgentEvent("final_answer", f"🛡️ **Blocked by Blue Team Sentinel**\n\n{block_msg}")
            return

        # Avoid duplicating the original turn in long-term chat history on resume.
        if resume_cp is None:
            self.memory.add_message(session_id, "user", user_input)

        # Retrieve conversation history
        history = self.memory.get_recent_messages(session_id, limit=10)

        # Format messages for LLM
        catalog_text = self._build_tool_catalog_text()
        system_content = (
            self.system_prompt
            + "\n\n### LIVE TOOL CATALOG (all tools currently available):\n"
            + catalog_text
        )
        # Phase 9: role persona overlay (subagent specialist identity) — placed
        # right after the base identity so it steers behaviour from the start.
        if system_extra:
            system_content += "\n\n" + system_extra
        # Omni-Domain adaptation: inject active industry domain persona & guardrails
        if hasattr(self, "domain_manager") and self.domain_manager:
            domain_overlay = self.domain_manager.build_system_overlay()
            if domain_overlay:
                system_content += "\n\n" + domain_overlay
        # Auto-recall: seed remembered facts relevant to this request (Memory
        # Agent pattern) so the model starts the turn already knowing the user.
        recalled = self.memory.recall_relevant(user_input, limit=5)
        if recalled:
            recall_block = "\n\n### REMEMBERED FACTS (from long-term memory, relevant to this request):\n"
            for f in recalled:
                recall_block += f"- [{f['category']}] {f['key']}: {f['value']}\n"
            recall_block += "(Use these facts as true context; do not claim you read them fresh.)"
            system_content += recall_block
        core_block = self._core_recall_block(user_input)
        if core_block:
            system_content += core_block
        # Auto-skill load: inject relevant Hermes-style playbooks for this task
        skill_block = self.skills.build_system_block(user_input)
        if skill_block:
            system_content += skill_block
        # Phase 22: adaptive tool record (opt-in via TITAN_TOOL_RECORD). When
        # enabled, tools that failed repeatedly in recent runs are surfaced so
        # the model adapts instead of retrying a broken approach.
        if tool_record_enabled():
            tool_block = self.tool_stats.prompt_block()
            if tool_block:
                system_content += "\n\n" + tool_block
        if mode == "deep":
            system_content += "\n\n" + DEEP_THINKING_PROMPT
        elif mode == "deep_search":
            system_content += "\n\n" + DEEP_SEARCH_PROMPT
        # Effort-level guidance (LOW/HIGH/ULTRA)
        if effort in EFFORT_PROMPTS:
            system_content += "\n\n" + EFFORT_PROMPTS[effort]
        # Phase 8: Full Access mode — every capability boundary is removed.
        if full_access_enabled():
            system_content += (
                "\n\n### FULL ACCESS MODE ACTIVE (TITAN_FULL_ACCESS=1)\n"
                "Every capability boundary is removed for this run:\n"
                "- No step budget limit — keep working until the task is VERIFIABLY complete.\n"
                "- No 45s command timeout (up to 10 minutes per command).\n"
                "- Approvals are auto-granted (delete_file / screenshot / ports / paths).\n"
                "- Downloads have no 100 MB cap; token rate limiting is disabled.\n"
                "- Subagent teams may run up to 8 workers in parallel.\n"
                "Use the extra scope deliberately, and still VERIFY every claim with tools before reporting."
            )

        messages = [{"role": "system", "content": system_content}]
        for msg in history:
            m_dict = {"role": msg["role"], "content": msg["content"]}
            messages.append(m_dict)

        # ---- Phase 5: Devin-style resume — restore the session's live state ----
        if resume_cp is not None and resume_cp.messages:
            yield AgentEvent(
                "status",
                f"Resuming session '{session_id}' from checkpoint ({resume_cp.steps_done} "
                f"steps done, last status: {resume_cp.status}).",
            )
            messages = [
                dict(message)
                for message in resume_cp.messages
                if isinstance(message, dict) and "role" in message
            ][-CHECKPOINT_MAX_MESSAGES:]
        # Always-on checkpointing (Devin-style): persist run state so an interrupted
        # session can be resumed without losing work.
        self._checkpoint_save(
            session_id=session_id,
            user_input=user_input,
            mode=mode,
            effort=effort,
            strategy=strategy,
            messages=messages,
            steps_done=resume_cp.steps_done if resume_cp is not None else 0,
            tools_used=resume_cp.tools_used if resume_cp is not None else [],
            status="running",
        )

        max_steps = _compute_max_steps(mode, raw_effort)

        # ---- Phase 3: structured reasoning (core engines) -------------------
        # Explicit strategy (plan/react/tot) runs the Phase 2 core engines with
        # policy guardrails. 'auto' keeps the classic loop below untouched.
        if strategy != "auto":
            yield AgentEvent("status", f"Structured reasoning engaged (strategy={strategy}, effort={effort}, max steps: {max_steps})")
            structured_final = None
            try:
                from .structured import StructuredEngine
                engine = StructuredEngine(
                    _TraceAwareLLMClient(self, self.llm, "structured"),
                    self.execute_tool_unified,
                    self._build_tools_list,
                    session_id=session_id,
                    # Phase 14: defer approval to the registry's HITL gate so the
                    # structured and classic paths share ONE approval decision.
                    hitl=None,
                    hitl_timeout=self.hitl_timeout,
                    defer_approval=True,
                )
                structured_context = self._build_structured_context(user_input, mode, effort)
                structured_checkpointed = self._checkpoint_save(
                    session_id=session_id,
                    user_input=user_input,
                    mode=mode,
                    effort=effort,
                    strategy=strategy,
                    messages=messages,
                    steps_done=0,
                    status="structured_in_progress",
                )
                if not structured_checkpointed:
                    yield AgentEvent(
                        "error",
                        "Structured execution was not started because its durable checkpoint could not be saved.",
                    )
                    yield AgentEvent(
                        "final_answer",
                        "Paused safely: structured tool execution requires writable checkpoint storage. No structured tools were run.",
                    )
                    return
                async for ev in engine.run(
                    user_input,
                    context=structured_context,
                    strategy=strategy,
                    max_steps=max_steps,
                ):
                    if ev.type == "final_answer":
                        structured_final = ev.data
                    if ev.type == "error":
                        structured_final = None  # fall back below
                        break
                    yield ev
            except (RuntimeError, OSError, ImportError, ValueError) as exc:
                yield AgentEvent("error", f"Structured reasoning unavailable ({exc!s}); using standard loop.")
            else:
                if structured_final:
                    # The final answer event was already streamed above; also save
                    # it to the conversation so future turns have full context.
                    self.memory.add_message(session_id, "assistant", structured_final)
                    await self._finalize_run(
                        session_id, user_input, structured_final, mode, strategy, auto_commit
                    )
                    self._checkpoint_save(
                        session_id=session_id, user_input=user_input, mode=mode,
                        effort=effort, strategy=strategy, messages=messages,
                        steps_done=max_steps, status="done", final_answer=structured_final,
                    )
                    return
                yield AgentEvent("error", "Structured reasoning finished without a final answer; using standard loop.")
            # fall through to the standard loop below if structured failed

        # Deep Search mode: seed the context with an auto-researched dossier first
        if mode == "deep_search":
            yield AgentEvent("status", "Building deep search dossier...")
            try:
                from .deep_search import DeepSearchEngine
                dossier = await DeepSearchEngine().run(user_input)
                context_block = (
                    f"### DEEP SEARCH DOSSIER (auto-researched):\n"
                    f"Topic: {dossier['topic']}\n"
                    f"Total sources found: {dossier['total_sources_found']}\n\n"
                    "Sources:\n"
                )
                for s in dossier.get("sources", [])[:6]:
                    context_block += f"- {s.get('title', '')}: {s.get('url', '')}\n"
                messages.append({"role": "system", "content": context_block})
                yield AgentEvent("status", f"Dossier ready: {dossier['total_sources_found']} sources found.")
            except (RuntimeError, OSError, ImportError) as e:
                yield AgentEvent("status", f"Auto deep search unavailable: {e!s}")

        yield AgentEvent("status", f"Planning and analyzing the task... (effort: {effort}, max steps: {max_steps})")

        iteration = resume_cp.steps_done if resume_cp is not None else 0
        checkpoint_context = {
            "session_id": session_id,
            "user_input": user_input,
            "mode": mode,
            "effort": effort,
            "strategy": strategy,
        }
        used_tools = bool(resume_cp.tools_used) if resume_cp is not None else False
        run_tools: list[str] = list(resume_cp.tools_used) if resume_cp is not None else []
        reflect_done = False
        postcheck_done = False  # Phase 26: bounded auto post-check for edit runs
        empty_retried = False   # Phase 30: bounded retry on whitespace-only finals
        # Phase 23: per-run consecutive identical-failure counters, keyed by
        # "tool\x00canonical-args". Reset on every run_task.
        self._guard_failures: dict[str, int] = {}
        # Phase 33: per-run repeated malformed-arguments counters, keyed by
        # tool NAME. Reset on every run_task.
        self._malformed_calls: dict[str, int] = {}
        # Phase 37: per-run consecutive all-failed tool batches (dead-end
        # detector streak). Reset on every run_task.
        self._consecutive_failed_batches = 0
        # Phase 38: per-run guard decision log + totals. Reset on every run_task.
        self._guard_actions: list[dict[str, Any]] = []
        self._guard_totals: dict[str, int] = {}
        self._read_cache.clear()
        grounded = False  # Phase 20: zero-tool answers get exactly ONE verification pass

        # Phase 41: System 3 Metacognitive Overseer & Bayesian Hypothesis Engine
        from titan_agent.core.reasoning.metacognitive_overseer import (
            InterventionType,
            MetacognitiveOverseer,
        )
        self._overseer = MetacognitiveOverseer(user_input)

        # Phase 09: Active Working Memory Virtualizer
        from titan_agent.core.memory.working_memory_virtualizer import (
            WorkingMemoryVirtualizer,
        )
        self._working_memory = WorkingMemoryVirtualizer(user_input)
        if hasattr(self.tools, "_working_memory_ref"):
            self.tools._working_memory_ref = self._working_memory

        while iteration < max_steps:
            iteration += 1
            yield AgentEvent("step_start", {"step": iteration, "max_steps": max_steps})

            # Phase 37: dead-end early stop. When the last N tool batches ALL
            # failed (errors, skips, guard blocks — no successful result), the
            # model is grinding on a broken path; stop now with an explicit
            # notice instead of burning the remaining steps and token budget.
            window = dead_end_window()
            if (
                window > 0
                and self._consecutive_failed_batches >= window
                and iteration > 1
            ):
                final_text = (
                    f"⚠ Stopped early: the last {window} tool batch(es) failed "
                    "completely (only errors, skipped calls and guard blocks — "
                    "no successful tool result). Continuing would waste the "
                    "remaining steps on the same broken path. The run state and "
                    "partial results above are preserved."
                )
                yield AgentEvent(
                    "status",
                    "Dead end detected: "
                    f"{self._consecutive_failed_batches} consecutive tool "
                    "batches with no successful result — stopping early.",
                )
                self._checkpoint_save(
                    session_id=session_id, user_input=user_input, mode=mode,
                    effort=effort, strategy=strategy, messages=messages,
                    steps_done=iteration, tools_used=["dead_end_stop"],
                    status="done", final_answer=final_text,
                )
                yield AgentEvent("final_answer", final_text)
                return

            # Phase 10: keep the in-run window inside the configured budget
            # proactively (a no-op until the context actually exceeds it), so
            # long runs never fight the provider window one step too late.
            _before_len = len(messages)
            messages = await compact_messages_for_context(
                messages, summarizer=self._summarize_context
            )
            # Phase 24: if compaction actually dropped messages, re-pin the
            # original task right before the next model call.
            if len(messages) < _before_len:
                self._reanchor_task(messages, user_input)
                if any(
                    ORIGINAL_TASK_MARKER in str(m.get("content") or "")
                    for m in messages
                ):
                    yield AgentEvent(
                        "status",
                        "Re-anchored to the original task after context compaction.",
                    )

            available_tools = self._build_tools_list()

            # Phase 09: Refresh Active Working Memory HUD in context
            self._inject_or_update_working_memory_hud(messages)

            try:
                response, messages, chat_notes = await self._chat_with_recovery(
                    messages, available_tools, anchor=user_input
                )
            except (RuntimeError, OSError, aiohttp.ClientError) as e:
                err_msg = f"Error connecting to LLM: {e!s}"
                yield AgentEvent("error", err_msg)
                self._checkpoint_save(
                    session_id=session_id, user_input=user_input, mode=mode,
                    effort=effort, strategy=strategy, messages=messages,
                    steps_done=iteration, status="error",
                )
                return
            for note in chat_notes:
                yield AgentEvent("status", note)

            # Yield thoughts if any
            if response.thoughts:
                yield AgentEvent("thought", response.thoughts)

            # If model produced tool calls, execute them (in parallel)
            if response.tool_calls:
                used_tools = True
                used_names = [
                    str(tc.get("function", {}).get("name", "tool"))
                    for tc in response.tool_calls
                    if isinstance(tc, dict)
                ]
                run_tools.extend(used_names)
                _before_batch = len(messages)
                async for ev in self._emit_tool_results(
                    response, messages, iteration, checkpoint_context=checkpoint_context
                ):
                    yield ev
                # Phase 37: feed the dead-end detector this batch's outcome
                # (all-failed batches build the streak, any success resets it).
                self._record_batch_outcome(messages, _before_batch)

                # Phase 41: System 3 Metacognitive Evaluation after tool execution
                latest_tool_content = messages[-1].get("content", "") if messages and messages[-1].get("role") == "tool" else ""
                first_tool_args = {}
                if response.tool_calls and isinstance(response.tool_calls[0], dict):
                    try:
                        first_tool_args = json.loads(str(response.tool_calls[0].get("function", {}).get("arguments") or "{}"))
                    except Exception:
                        pass

                # Phase 09: Working Memory autonomous extraction from tool outcome
                self._working_memory.auto_observe_tool_outcome(
                    tool_name=used_names[0] if used_names else "none",
                    tool_args=first_tool_args,
                    tool_result=latest_tool_content,
                )

                meta_report = self._overseer.evaluate_step(
                    step_idx=iteration,
                    thoughts=response.thoughts or "",
                    tool_name=used_names[0] if used_names else "none",
                    tool_args=first_tool_args,
                    tool_result=latest_tool_content,
                )

                if meta_report.intervention != InterventionType.CONTINUE:
                    yield AgentEvent(
                        "metacognition",
                        {
                            "step": iteration,
                            "intervention": meta_report.intervention.value,
                            "recommendation": meta_report.recommendation,
                            "entropy": meta_report.cognitive_entropy,
                            "repetition": meta_report.repetition_score,
                        },
                    )
                    # Inject prescriptive metacognitive directive to guide next reasoning turn
                    messages.append({
                        "role": "system",
                        "content": f"### METACOGNITIVE DIRECTIVE (System 3 Intervention):\n{meta_report.recommendation}",
                    })

                # Persist live state after each step (resume-safe)
                self._checkpoint_save(
                    session_id=session_id, user_input=user_input, mode=mode,
                    effort=effort, strategy=strategy, messages=messages,
                    steps_done=iteration, tools_used=run_tools, status="running",
                )

                # Check if iterations limit reached
                if iteration >= max_steps:
                    yield AgentEvent("final_answer", f"Reached the maximum number of steps ({max_steps}). The latest state and results are preserved above.")
                    return
                continue

            # ---- No tool calls: candidate final answer ----
            final_text = response.content or ""

            # ---- Phase 30: bounded retry on whitespace-only finals ----
            # An empty "answer" is never a valid final. Ask ONCE more (bounded);
            # if the model still returns nothing the emission point below turns
            # it into an explicit notice instead of a silent empty success.
            if (
                empty_final_guard_enabled()
                and not empty_retried
                and not final_text.strip()
            ):
                empty_retried = True
                messages.append({"role": "system", "content": EMPTY_FINAL_PROMPT})
                yield AgentEvent(
                    "status",
                    "Empty response — asking the model to produce the answer...",
                )
                continue

            # ---- Phase 20: ground zero-tool answers (anti-hallucination) ----
            # A model that answered without touching a tool once is the classic
            # hallucination path for weak local models. Give it ONE forced
            # verification turn: it may emit tool calls (executed below, loop
            # continues) or explicitly decline with NO_TOOLS_NEEDED for pure
            # conceptual tasks. Tool-using runs never pay for this call.
            if (
                not used_tools
                and not grounded
                and final_text.strip()
                and final_grounding_enabled()
            ):
                grounded = True
                yield AgentEvent(
                    "status",
                    "Verifying answer before finalizing (no tools used yet)...",
                )
                grounding_messages = list(messages) + [
                    {"role": "assistant", "content": final_text},
                    {"role": "user", "content": GROUNDING_PROMPT},
                ]
                try:
                    gresp, grounding_msgs, gnotes = await self._chat_with_recovery(
                        grounding_messages, available_tools, anchor=user_input
                    )
                except (RuntimeError, OSError, aiohttp.ClientError) as e:
                    yield AgentEvent("error", f"Grounding pass error: {e!s}")
                    gresp, grounding_msgs, gnotes = None, messages, []
                for note in gnotes:
                    yield AgentEvent("status", note)
                if gresp is not None:
                    if gresp.thoughts:
                        yield AgentEvent("thought", gresp.thoughts)
                    if gresp.tool_calls:
                        # Grounding decided real verification is needed — execute it
                        used_tools = True
                        async for ev in self._emit_tool_results(
                            gresp, grounding_msgs, iteration,
                            checkpoint_context=checkpoint_context,
                        ):
                            yield ev
                        messages = grounding_msgs
                        self._checkpoint_save(
                            session_id=session_id, user_input=user_input, mode=mode,
                            effort=effort, strategy=strategy, messages=messages,
                            steps_done=iteration, tools_used=run_tools,
                            status="running",
                        )
                        if iteration >= max_steps:
                            yield AgentEvent("final_answer", f"Reached the maximum number of steps ({max_steps}). The latest state and results are preserved above.")
                            return
                        continue
                    if gresp.content:
                        # Grounded (or explicitly declined) final answer
                        final_text = gresp.content or final_text

            # Reflection (critic) pass: after real tool use (or always in deep / high-effort runs)
            needs_reflection = used_tools or mode in ("deep", "deep_search") or effort in ("high", "ultra")
            if needs_reflection and not reflect_done:
                reflect_done = True
                yield AgentEvent("status", "Critically reviewing results (reflection)...")
                # Phase 25: feed the critic a deterministic list of what REALLY
                # happened with tools so it argues against facts, not vibes.
                critic_messages = list(messages) + [
                    {"role": "assistant", "content": final_text},
                    {
                        "role": "user",
                        "content": REFLECTION_PROMPT
                        + "\n"
                        + self._build_tool_evidence(messages),
                    },
                ]
                try:
                    crit = await self._traced_chat_completion(
                        self._critic_llm(),
                        critic_messages,
                        tools=available_tools,
                        call_role="reflection",
                    )
                except (RuntimeError, OSError, aiohttp.ClientError) as e:
                    yield AgentEvent("error", f"Reflection pass error: {e!s}")
                    crit = None

                if crit is not None:
                    if crit.thoughts:
                        yield AgentEvent("thought", crit.thoughts)
                    if crit.tool_calls:
                        # Reflection decided more work is needed — execute it
                        _before_batch = len(messages)
                        async for ev in self._emit_tool_results(
                            crit, messages, iteration,
                            checkpoint_context=checkpoint_context,
                        ):
                            yield ev
                        # Phase 37: reflection-grinding counts toward the same
                        # dead-end streak as main-loop batches.
                        self._record_batch_outcome(messages, _before_batch)
                        self._checkpoint_save(
                            session_id=session_id, user_input=user_input, mode=mode,
                            effort=effort, strategy=strategy, messages=messages,
                            steps_done=iteration, tools_used=run_tools, status="running",
                        )
                        if iteration >= max_steps:
                            yield AgentEvent("final_answer", f"Reached the maximum number of steps ({max_steps}). The latest state and results are preserved above.")
                            return
                        continue
                    elif crit.content:
                        # Reflection produced the polished final answer
                        final_text = crit.content or final_text

                        # Phase 21: bounded refinement when a dedicated reviewer
                        # is in play — the GENERATOR revises against the critic's
                        # output up to TITAN_REFINEMENT_ROUNDS times. Without a
                        # separate reviewer this loop never runs (0 rounds), so
                        # the classic single-reflection behavior is preserved.
                        refine_handoff = False
                        refine_round = 0
                        critic_llm = self._critic_llm()
                        while refine_round < (
                            refinement_rounds()
                            if critic_llm is not self.llm
                            else 0
                        ):
                            refine_round += 1
                            yield AgentEvent(
                                "status",
                                f"Refining final answer against critique "
                                f"(round {refine_round}/{refinement_rounds()})...",
                            )
                            refine_messages = critic_messages + [
                                {"role": "assistant", "content": final_text},
                                {"role": "user", "content": REFINE_PROMPT},
                            ]
                            try:
                                refined = await self._traced_chat_completion(
                                    self.llm,
                                    refine_messages,
                                    tools=available_tools,
                                    call_role="refinement",
                                )
                            except (RuntimeError, OSError, aiohttp.ClientError) as e:
                                yield AgentEvent("error", f"Refinement pass error: {e!s}")
                                break
                            if refined is None:
                                break
                            if refined.thoughts:
                                yield AgentEvent("thought", refined.thoughts)
                            if refined.tool_calls:
                                refine_handoff = True
                                _before_batch = len(messages)
                                async for ev in self._emit_tool_results(
                                    refined, messages, iteration,
                                    checkpoint_context=checkpoint_context,
                                ):
                                    yield ev
                                # Phase 37: refinement work counts toward the
                                # same dead-end streak as every other batch.
                                self._record_batch_outcome(messages, _before_batch)
                                break
                            if refined.content:
                                final_text = refined.content or final_text
                        if refine_handoff:
                            self._checkpoint_save(
                                session_id=session_id, user_input=user_input, mode=mode,
                                effort=effort, strategy=strategy, messages=messages,
                                steps_done=iteration, tools_used=run_tools,
                                status="running",
                            )
                            if iteration >= max_steps:
                                yield AgentEvent("final_answer", f"Reached the maximum number of steps ({max_steps}). The latest state and results are preserved above.")
                                return
                            continue

            # Phase 26: bounded auto post-check for edit runs. When files were
            # actually written/edited, inject exactly ONE extra verification
            # turn before finalizing (re-read changed files, run tests) — the
            # classic weak-model failure is claiming "done" without ever
            # verifying. Read-only runs never trigger this.
            if (
                auto_postcheck_enabled()
                and not postcheck_done
                and self._run_used_write_tool(messages)
            ):
                postcheck_done = True
                # Phase 31: tell the model EXACTLY which files it wrote so the
                # weak model does not have to remember — it re-reads the list.
                postcheck_prompt = POSTCHECK_PROMPT
                written = self._written_paths(messages)
                if written:
                    postcheck_prompt += (
                        "\n\nFiles written/edited this run: "
                        + ", ".join(written)
                    )
                messages.append(
                    {"role": "system", "content": postcheck_prompt}
                )
                yield AgentEvent(
                    "status", "Post-check: verifying edited files before finalizing..."
                )
                continue

            # Phase 30: an empty answer that survives the bounded retry is
            # turned into an explicit notice — never a silent empty success.
            if not final_text.strip():
                final_text = (
                    "⚠ The model produced an empty final answer. The run state "
                    "and partial results above are preserved."
                )

            yield AgentEvent("final_answer", final_text)
            await self._finalize_run(session_id, user_input, final_text, mode, strategy, auto_commit)
            self.memory.add_message(session_id, "assistant", final_text, thoughts=response.thoughts)
            self._checkpoint_save(
                session_id=session_id, user_input=user_input, mode=mode,
                effort=effort, strategy=strategy, messages=messages,
                steps_done=iteration, tools_used=run_tools,
                status="done", final_answer=final_text,
            )
            return