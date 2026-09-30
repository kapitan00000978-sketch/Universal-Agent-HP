"""Run the fixed benchmark catalog against the configured Titan Agent model.

This is an evaluation aid, not an OS-level sandbox. Each case gets a fresh
workspace, but command tools still execute with the privileges of this process.
Run only on a disposable machine/container and never enable safety probes on a
machine containing data you care about.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.catalog import CATALOG_PATH, load_catalog


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_SECONDS = 300


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def select_cases(
    cases: list[dict[str, Any]],
    selected_ids: set[str] | None = None,
    include_safety_probes: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (runnable, skipped); critical probes are opt-in and never implicit."""
    known_ids = {case["id"] for case in cases}
    unknown_ids = (selected_ids or set()) - known_ids
    if unknown_ids:
        raise ValueError("Unknown benchmark case id(s): " + ", ".join(sorted(unknown_ids)))

    considered = [case for case in cases if selected_ids is None or case["id"] in selected_ids]
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for case in considered:
        if case.get("risk") == "critical" and not include_safety_probes:
            skipped.append(case)
        else:
            runnable.append(case)
    return runnable, skipped


def _new_agent(workspace: Path, provider: str | None, model: str | None):
    """Construct an agent with per-case state isolated under the case workspace."""
    from titan_agent.agent import TitanAgent
    from titan_agent.core.domain.manager import DomainManager
    from titan_agent.core.guardrails.hitl import HumanInTheLoop
    from titan_agent.llm_client import LLMClient
    from titan_agent.mcp_client import MCPManager
    from titan_agent.memory import MemoryManager
    from titan_agent.tool_stats import ToolStatsCollector
    from titan_agent.tools import ToolRegistry

    workspace.mkdir(parents=True, exist_ok=True)
    return TitanAgent(
        llm=LLMClient(provider=provider, model=model),
        tools=ToolRegistry(workspace),
        mcp=MCPManager(config_file=None),
        memory=MemoryManager(workspace / "memory.db"),
        core_memory_path=workspace / "core_memory.db",
        git_root=workspace,
        auto_commit=False,
        checkpoint_path=workspace / "checkpoints.db",
        tool_stats=ToolStatsCollector(),
        # Non-interactive evaluation must never grant a pending approval.
        hitl=HumanInTheLoop(
            default_timeout=0.25,
            audit_path=workspace / "hitl_audit.jsonl",
        ),
        hitl_timeout=0.25,
        domain_manager=DomainManager(
            domains_dir=workspace / ".titan" / "domains",
            default_domain="universal",
        ),
    )


async def run_case(
    case: dict[str, Any],
    *,
    workspace: Path,
    provider: str | None,
    model: str | None,
    effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one case and retain final output plus non-sensitive execution metadata."""
    from titan_agent.config import full_access_enabled

    base: dict[str, Any] = {
        "case_id": case["id"],
        "category": case["category"],
        "risk": case["risk"],
        "workspace": str(workspace),
        "status": "failed",
        "final_answer": "",
        "tool_calls": [],
        "tool_stats": None,
        "errors": [],
        "provider": provider,
        "model": model,
        "elapsed_seconds": None,
        "output_chars": 0,
        "ratings": None,
        "score_percent": None,
    }
    if full_access_enabled():
        base["errors"].append(
            "Refusing to run benchmark while TITAN_FULL_ACCESS/TITAN_ABSOLUTE_ACCESS is enabled."
        )
        return base

    agent = _new_agent(workspace, provider, model)
    base["provider"] = getattr(agent.llm, "provider", provider)
    base["model"] = getattr(agent.llm, "model", model)
    started = time.monotonic()

    async def consume() -> None:
        async for event in agent.run_task(
            case["prompt"],
            session_id=f"bench-{uuid.uuid4().hex[:12]}",
            mode=case["mode"],
            effort=effort,
            strategy="auto",
            auto_commit=False,
            resume=False,
        ):
            if event.type == "final_answer":
                base["final_answer"] = str(event.data or "")
            elif event.type == "tool_call":
                data = event.data if isinstance(event.data, dict) else {}
                name = str(data.get("name", "unknown"))
                arguments = data.get("arguments")
                arguments = arguments if isinstance(arguments, dict) else {}
                from titan_agent.core.guardrails.dangerous_actions import DangerousActionClassifier

                resource = str(arguments.get("path", arguments.get("file", "")) or "")
                assessment = DangerousActionClassifier.assess_action(
                    name, resource, arguments
                )
                # Keep only the risk assessment; do not persist raw tool args,
                # which may contain secrets or sensitive file contents.
                base["tool_calls"].append({
                    "name": name,
                    "dangerous": assessment.is_dangerous,
                    "risk_level": assessment.risk_level,
                    "category": assessment.category,
                })
            elif event.type == "error":
                base["errors"].append(str(event.data)[:1000])

    try:
        await asyncio.wait_for(consume(), timeout=timeout_seconds)
        base["status"] = "completed" if base["final_answer"].strip() else "no_final_answer"
    except asyncio.TimeoutError:
        base["status"] = "timed_out"
        base["errors"].append(f"Case exceeded {timeout_seconds} second timeout.")
    except Exception as exc:  # benchmark should preserve a failed case and continue
        base["status"] = "failed"
        base["errors"].append(f"{type(exc).__name__}: {exc!s}"[:1000])
    finally:
        base["elapsed_seconds"] = round(time.monotonic() - started, 3)
        base["output_chars"] = len(base["final_answer"])
        stats_collector = getattr(agent, "tool_stats", None)
        if stats_collector is not None and callable(getattr(stats_collector, "summary", None)):
            summary = stats_collector.summary()
            base["tool_stats"] = {
                "totals": summary.get("totals", {}),
                "tools": {
                    name: {key: value for key, value in stats.items() if key != "last_error"}
                    for name, stats in summary.get("tools", {}).items()
                },
            }
    return base


def _parser() -> argparse.ArgumentParser:
    catalog = load_catalog()
    case_ids = [case["id"] for case in catalog["cases"]]
    parser = argparse.ArgumentParser(
        description="Run Universal Agent HP benchmark cases and save an auditable JSON result.",
        epilog=(
            "Warning: workspaces are isolated directories, NOT OS sandboxes. "
            "Use a disposable machine/container. Critical safety probes are skipped unless explicitly enabled."
        ),
    )
    parser.add_argument("--provider", help="LLM provider (defaults to project configuration)")
    parser.add_argument("--model", help="Model name (defaults to project configuration)")
    parser.add_argument("--effort", choices=["auto", "low", "medium", "high", "ultra"], default="auto")
    parser.add_argument("--case", action="append", choices=case_ids, help="Run only this case; may be repeated")
    parser.add_argument("--include-safety-probes", action="store_true", help="Opt in to critical-risk prompts; use only in a disposable, isolated environment")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help=f"Timeout per case in seconds (default: {DEFAULT_TIMEOUT_SECONDS})")
    parser.add_argument("--output", type=Path, help="Result JSON path (default: benchmark-results/results-<UTC timestamp>.json)")
    parser.add_argument("--workspace-root", type=Path, help="Root for per-case workspaces (default: benchmark-results/workspaces/<run-id>)")
    parser.add_argument("--list", action="store_true", help="List available case IDs and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.list:
        for case in catalog["cases"]:
            print(f"{case['id']}\t{case['category']}\t{case['risk']}")
        return 0
    if args.timeout < 1:
        parser.error("--timeout must be at least 1 second")

    try:
        runnable, skipped = select_cases(
            catalog["cases"],
            selected_ids=set(args.case) if args.case else None,
            include_safety_probes=args.include_safety_probes,
        )
    except ValueError as exc:
        parser.error(str(exc))

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    default_output = REPO_ROOT / "benchmark-results" / f"results-{run_id}.json"
    output_path = (args.output or default_output).expanduser().resolve()
    requested_workspace_root = args.workspace_root or output_path.parent / "workspaces"
    workspace_root = (requested_workspace_root / run_id).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = [
        {
            "case_id": case["id"],
            "category": case["category"],
            "risk": case["risk"],
            "status": "skipped_safety_probe",
            "reason": "Critical-risk case; use --include-safety-probes only in a disposable, isolated environment.",
            "ratings": None,
            "score_percent": None,
        }
        for case in skipped
    ]
    for index, case in enumerate(runnable, start=1):
        case_workspace = workspace_root / case["id"]
        print(f"[{index}/{len(runnable)}] {case['id']} ({case['category']}) ...", flush=True)
        result = asyncio.run(
            run_case(
                case,
                workspace=case_workspace,
                provider=args.provider,
                model=args.model,
                effort=args.effort,
                timeout_seconds=args.timeout,
            )
        )
        results.append(result)
        print(
            f"  {result['status']}; {result['elapsed_seconds']}s; "
            f"{len(result['tool_calls'])} tool call(s); "
            f"{result['output_chars']} answer chars",
            flush=True,
        )

    result_document = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": started_at.isoformat(),
        "catalog_version": catalog["version"],
        "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
        "agent_revision": _git_revision(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "provider_requested": args.provider,
        "model_requested": args.model,
        "effort": args.effort,
        "timeout_seconds": args.timeout,
        "attempted_count": len(runnable),
        "skipped_count": len(skipped),
        "completed_count": sum(result.get("status") == "completed" for result in results),
        "scored_count": 0,
        "overall_score_percent": None,
        "results": results,
        "notes": [
            "No score is claimed until rubric ratings are independently entered.",
            "Workspaces isolate files between cases but do not sandbox OS commands.",
        ],
    }
    output_path.write_text(json.dumps(result_document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark record: {output_path}")
    print(
        f"Completed {result_document['completed_count']}/{len(runnable)}; "
        f"skipped safety probes {len(skipped)}; scored 0/{len(results)}."
    )
    return 0 if all(result.get("status") == "completed" for result in results if result.get("status") != "skipped_safety_probe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
