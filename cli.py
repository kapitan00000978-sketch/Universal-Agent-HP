import asyncio
import os
import platform
import sys

if sys.platform == "win32":
    # getattr keeps text-mode streams (which lack reconfig) vs binary ones split.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding='utf-8', errors='replace')
            except (OSError, ValueError):
                pass  # best-effort: keep default streams if reconfigure is unsupported
import argparse
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from titan_agent.agent import TitanAgent
from titan_agent.commands import ALL_COMMANDS, expand_slash, parse_local
from titan_agent.config import DEFAULT_MODEL, DEFAULT_PROVIDER, MCP_CONFIG_FILE
from titan_agent.llm_client import LLMClient
from titan_agent.mcp_client import MCPManager

console = Console()

VALID_MODES = ("fast", "deep", "deep_search")
VALID_EFFORTS = ("auto", "low", "medium", "high", "ultra")
VALID_STRATEGIES = ("auto", "plan", "react", "tot", "reflexion", "debate")

# ─── Universal Completer (Slash Commands + @ File Selection) ─────────
class UniversalCompleter(Completer):
    """Autocomplete slash commands ('/') and workspace files ('@')."""

    def __init__(self, commands: dict[str, str], workspace_root: Path | None = None):
        self._commands = commands
        self._root = workspace_root or Path.cwd()
        self._ignored_dirs = {
            ".git", ".pytest_cache", ".ruff_cache", "__pycache__",
            "node_modules", ".venv", "venv", ".idea", ".vscode", "dist", "build",
            ".system_generated", "universal_agent_hp.egg-info",
        }

    def _get_workspace_files(self, prefix: str) -> list[tuple[str, bool, int]]:
        """List files and folders matching prefix relative to workspace root."""
        results: list[tuple[str, bool, int]] = []
        prefix_clean = prefix.replace("\\", "/").lower()
        
        try:
            for root, dirs, files in os.walk(self._root):
                # Filter out ignored dirs in-place
                dirs[:] = [d for d in dirs if d not in self._ignored_dirs and not d.startswith(".")]
                
                rel_dir = os.path.relpath(root, self._root).replace("\\", "/")
                if rel_dir == ".":
                    rel_dir = ""

                # Folders
                for d in dirs:
                    rel_path = f"{rel_dir}/{d}" if rel_dir else d
                    if prefix_clean in rel_path.lower():
                        results.append((rel_path + "/", True, 0))
                        if len(results) >= 40:
                            return results

                # Files
                for f in files:
                    if f.startswith("."):
                        continue
                    rel_path = f"{rel_dir}/{f}" if rel_dir else f
                    if prefix_clean in rel_path.lower():
                        full_p = Path(root) / f
                        try:
                            size = full_p.stat().st_size
                        except OSError:
                            size = 0
                        results.append((rel_path, False, size))
                        if len(results) >= 40:
                            return results
        except Exception:
            pass
        return results

    def get_completions(self, document, complete_event):
        text_before = document.text_before_cursor
        stripped = text_before.lstrip()

        # 1. Slash commands at the start of input
        if stripped.startswith("/") and " " not in stripped:
            prefix = stripped[1:].lower()
            for name, desc in sorted(self._commands.items()):
                if name.startswith(prefix):
                    yield Completion(
                        f"/{name}",
                        start_position=-len(stripped),
                        display=f"/{name}",
                        display_meta=desc[:60],
                    )
            return

        # 2. @ File mentions anywhere in input
        words = text_before.split()
        if text_before.endswith(" ") or not words:
            return

        current_word = words[-1]
        if current_word.startswith("@"):
            file_prefix = current_word[1:]
            for rel_path, is_dir, size in self._get_workspace_files(file_prefix):
                if is_dir:
                    display_text = f"📁 {rel_path}"
                    meta = "directory"
                else:
                    display_text = f"📄 {rel_path}"
                    if size < 1024:
                        meta = f"{size} B"
                    elif size < 1024 * 1024:
                        meta = f"{size / 1024:.1f} KB"
                    else:
                        meta = f"{size / (1024 * 1024):.1f} MB"

                yield Completion(
                    f"@{rel_path}",
                    start_position=-len(current_word),
                    display=display_text,
                    display_meta=meta,
                )


def _resolve_file_mentions(text: str, root_dir: Path) -> tuple[str, list[str]]:
    """Scan prompt for @filepath references and attach file contents into context."""
    words = text.split()
    attached_files = []
    file_contexts = []

    for word in words:
        # Strip potential trailing punctuation
        clean_word = word.rstrip(",;.:!?")
        if clean_word.startswith("@") and len(clean_word) > 1:
            rel_name = clean_word[1:]
            target = root_dir / rel_name
            if target.is_file():
                try:
                    content = target.read_text(encoding="utf-8", errors="replace")
                    # Limit attached content size to 64KB per file
                    if len(content) > 65536:
                        content = content[:65536] + "\n... [truncated, file exceeds 64KB] ..."
                    file_contexts.append(f"\n\n--- [Referenced File: @{rel_name}] ---\n```\n{content}\n```")
                    attached_files.append(rel_name)
                except Exception:
                    pass

    if file_contexts:
        expanded_prompt = text + "".join(file_contexts)
        return expanded_prompt, attached_files
    return text, []


def _build_dashboard(provider: str, model: str, mode: str, effort: str, mcp_count: int, tool_count: int) -> Panel:
    """Build a professional startup dashboard panel."""
    # System info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="cyan bold", width=16)
    info_table.add_column("Value", style="white")
    info_table.add_row("🤖 Provider", f"{provider}")
    info_table.add_row("🧠 Model", f"{model}")
    info_table.add_row("⚡ Mode", f"{mode}")
    info_table.add_row("💪 Effort", f"{effort}")
    info_table.add_row("🔌 MCP Servers", f"{mcp_count} connected")
    info_table.add_row("🔧 Tools", f"{tool_count} available")
    info_table.add_row("📁 Workspace", f"{Path.cwd().name}")
    info_table.add_row("💻 Platform", f"{platform.system()} {platform.release()}")

    # Commands table
    cmd_table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
    cmd_table.add_column("Command", style="cyan", width=18)
    cmd_table.add_column("Description", style="dim white")

    cmd_items = sorted(ALL_COMMANDS.items())
    for name, desc in cmd_items:
        cmd_table.add_row(f"/{name}", desc[:50])

    # Layout
    layout_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    layout_table.add_column("System", ratio=1)
    layout_table.add_column("Commands", ratio=1)
    layout_table.add_row(info_table, cmd_table)

    header = Text()
    header.append("⚡ UNIVERSAL AGENT HP", style="bold cyan")
    header.append(" — ", style="dim")
    header.append("Autonomous AI Agent OS", style="italic white")
    header.append("\n")
    header.append("  Type naturally in English, Russian, or Uzbek. Use @file to attach files, / for commands.", style="dim yellow")

    dashboard = Panel(
        layout_table,
        title=header,
        border_style="cyan",
        subtitle="[dim]Type @ to select files | /dashboard to refresh | /files to browse[/dim]",
        padding=(1, 2),
    )
    return dashboard


def _render_files_table(root_dir: Path, filter_str: str = "") -> Panel:
    """Render a clean Rich table of workspace files."""
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1), expand=True)
    table.add_column("Type", width=6, justify="center")
    table.add_column("File / Directory", style="white", ratio=3)
    table.add_column("Size", style="green", width=12, justify="right")
    table.add_column("Modified", style="dim white", width=20)

    ignored_dirs = {
        ".git", ".pytest_cache", ".ruff_cache", "__pycache__",
        "node_modules", ".venv", "venv", ".idea", ".vscode", "dist", "build",
        ".system_generated", "universal_agent_hp.egg-info",
    }
    
    count = 0
    filter_lower = filter_str.lower().strip()

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
        rel_dir = os.path.relpath(root, root_dir).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""

        for f in sorted(files):
            if f.startswith("."):
                continue
            rel_path = f"{rel_dir}/{f}" if rel_dir else f
            if filter_lower and filter_lower not in rel_path.lower():
                continue

            full_p = Path(root) / f
            try:
                st = full_p.stat()
                sz = st.st_size
                mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
                if sz < 1024:
                    sz_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    sz_str = f"{sz / 1024:.1f} KB"
                else:
                    sz_str = f"{sz / (1024 * 1024):.1f} MB"
            except OSError:
                sz_str = "-"
                mtime_str = "-"

            # Extension icon
            ext = full_p.suffix.lower()
            icon = "🐍" if ext == ".py" else ("📜" if ext in (".md", ".txt") else ("⚙️" if ext in (".json", ".toml", ".yaml", ".yml", ".env") else "📄"))

            table.add_row(icon, rel_path, sz_str, mtime_str)
            count += 1
            if count >= 60:
                break
        if count >= 60:
            break

    title_text = f"[bold cyan]📁 Workspace Files ({count} shown){f' [filter: {filter_str}]' if filter_str else ''}[/bold cyan]"
    return Panel(table, title=title_text, border_style="cyan", subtitle="[dim]Use @filename in your prompt to attach file content automatically[/dim]")


def _render_security_status() -> Panel:
    """Render real-time Dual-Shield Cyber Defense Sentinel Panel."""
    from titan_agent.core.security.dual_shield import DualShieldOrchestrator
    shield = DualShieldOrchestrator.get_instance()
    status = shield.get_security_status()

    sec_table = Table(show_header=False, box=None, padding=(0, 2))
    sec_table.add_column("Property", style="bold cyan", width=22)
    sec_table.add_column("Status", style="white")

    sec_table.add_row("🛡️ Blue Team Sentinel", "[bold green]ALWAYS-ON (Active Monitoring & Intercept)[/bold green]")
    sec_table.add_row("🚫 Threats Blocked", f"{status['blue_team']['blocked_threats']}")
    sec_table.add_row("⚔️ Emergency Red Team", f"[bold {'red' if status['red_team']['total_counter_strikes'] else 'yellow'}]{status['red_team']['state']}[/bold {'red' if status['red_team']['total_counter_strikes'] else 'yellow'}]")
    sec_table.add_row("💥 Red Counter-Strikes", f"{status['red_team']['total_counter_strikes']}")
    sec_table.add_row("🔒 Quarantined Sessions", f"{status['blue_team']['stats']['quarantined_sessions']}")
    sec_table.add_row("📋 Total Incidents Logged", f"{status['total_recorded_incidents']}")

    recent = status['red_team']['recent_incidents']
    if recent:
        inc_rows = "\n".join(f"- [bold red]{inc['id']}[/bold red] ({inc['vector']})" for inc in recent)
    else:
        inc_rows = "[dim green]No critical breaches detected. Perimeter secure.[/dim green]"

    full_layout = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    full_layout.add_column("Telemetry", ratio=1)
    full_layout.add_column("Recent Incidents", ratio=1)
    full_layout.add_row(sec_table, Panel(inc_rows, title="[bold red]Emergency Incidents[/bold red]", border_style="red" if recent else "green"))

    return Panel(
        full_layout,
        title="[bold green]🛡️ DUAL-SHIELD CYBER DEFENSE STATUS[/bold green]",
        border_style="green",
        subtitle="[dim]Blue Team monitors 24/7 | Red Team activates only on critical threat detection[/dim]",
        padding=(1, 2),
    )


def _render_banned_ips(action_arg: str = "") -> Panel:
    """Render defensively banned IP addresses table or perform unban."""
    from titan_agent.core.security.dual_shield import DualShieldOrchestrator
    shield = DualShieldOrchestrator.get_instance()

    parts = action_arg.strip().split(None, 1)
    if parts and parts[0].lower() == "unban" and len(parts) > 1:
        target_ip = parts[1].strip()
        ok = shield.ip_defense.unban_ip(target_ip)
        if ok:
            return Panel(f"[bold green]✅ Successfully unbanned IP:[/bold green] [cyan]{target_ip}[/cyan]", title="[bold green]IP Unbanned[/bold green]", border_style="green")
        return Panel(f"[bold red]❌ IP not found in ban list:[/bold red] [yellow]{target_ip}[/yellow]", title="[bold red]Unban Failed[/bold red]", border_style="red")

    banned_list = shield.ip_defense.list_banned()
    if not banned_list:
        return Panel("[bold green]✅ No IP addresses currently banned. Perimeter is secure.[/bold green]", title="[bold cyan]🛡️ Defensive Banned IP List (0 Active)[/bold cyan]", border_style="green")

    table = Table(show_header=True, header_style="bold red", box=None, padding=(0, 1), expand=True)
    table.add_column("IP Address", style="bold red", width=18)
    table.add_column("Location / ISP", style="white", ratio=2)
    table.add_column("Reason", style="yellow", ratio=3)
    table.add_column("Banned At", style="dim white", width=18)

    for entry in banned_list:
        t_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.banned_at))
        loc_str = f"{entry.country}, {entry.city} ({entry.isp})"
        table.add_row(entry.ip, loc_str, entry.reason, t_str)

    return Panel(
        table,
        title=f"[bold red]🛡️ Defensively Banned IP Addresses ({len(banned_list)} active)[/bold red]",
        border_style="red",
        subtitle="[dim]To unban an IP: /banned-ips unban <ip>[/dim]",
        padding=(1, 2),
    )


def _render_free_models(filter_arg: str = "") -> Panel:
    """Render comprehensive Free API Key Providers & Models Catalog."""
    from titan_agent.free_providers import get_free_providers
    providers = get_free_providers()
    filter_clean = filter_arg.strip().lower()

    if filter_clean:
        providers = [p for p in providers if filter_clean in p["id"].lower() or filter_clean in p["name"].lower()]

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1), expand=True)
    table.add_column("Platform & Portal Link", style="bold green", width=32)
    table.add_column("Type / Free Quota", style="yellow", width=34)
    table.add_column("Top Free Models", style="white", ratio=2)

    for p in providers:
        key_type = "✨ [bold green]100% Zero-Key[/bold green]" if p["no_key_required"] else f"🔑 [bold cyan]{p['env_var']}[/bold cyan]"
        prov_info = f"[bold white]{p['name']}[/bold white]\n[dim cyan]{p['portal_url']}[/dim cyan]"
        tier_info = f"{key_type}\n[dim]{p['free_tier_info']}[/dim]"
        models_str = "\n".join(f"• [bold]{m['name']}[/bold]: [dim]{m['id']}[/dim]" for m in p["models"][:3])
        table.add_row(prov_info, tier_info, models_str)

    title_text = f"[bold green]🎁 FREE API KEY PLATFORMS & MODELS HUB ({len(providers)} Available)[/bold green]"
    sub_text = "[dim]To use: get free key from link, set in .env or run `universal --provider <id> --model <model>`[/dim]"
    return Panel(table, title=title_text, subtitle=sub_text, border_style="green", padding=(1, 2))


def _bottom_toolbar(provider: str, model: str, mode: str, effort: str):
    """Bottom toolbar for prompt_toolkit showing current status."""
    return HTML(
        f'<b>⚡ Universal Agent HP</b> | '
        f'<style fg="ansicyan">{provider}/{model}</style> | '
        f'Mode: <style fg="ansiyellow">{mode}</style> | '
        f'Effort: <style fg="ansigreen">{effort}</style> | '
        f'<style fg="ansimagenta">@ for files</style> | '
        f'<style fg="ansigray">/ for commands</style>'
    )


async def resolve_cli_provider() -> tuple[str, str]:
    """Puter.js is browser-only, so the CLI auto-falls back to local Ollama."""
    provider, model = DEFAULT_PROVIDER, DEFAULT_MODEL
    if provider == "puter":
        console.print(
            "[yellow]ℹ️ Puter.js provider only works in the Web Dashboard (browser). "
            "Checking for local Ollama models...[/yellow]"
        )
        local = await LLMClient.detect_local_models()
        ollama = local.get("ollama") or []
        if ollama:
            provider, model = "ollama", ollama[0]
            console.print(f"[green]✅ Ollama found: {ollama[0]} → switched to it.[/green]")
        else:
            console.print(
                "[bold red]❌ No Ollama model was found on this computer.[/bold red]\n"
                "[yellow]Solution: (1) Install Ollama and pull a model: `ollama run hermes3`, or\n"
                "(2) Launch the Web Dashboard (Puter.js models run in the browser): python run.py[/yellow]"
            )
            raise SystemExit(1)
    return provider, model

async def main():
    parser = argparse.ArgumentParser(description="Universal Agent HP CLI")
    parser.add_argument("--provider", type=str, help="Chat LLM provider (e.g., openai, ollama, lmstudio); Laya is available separately via laya_decide for typed decisions")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--mode", type=str, choices=VALID_MODES, default=os.getenv("TITAN_MODE", "fast").lower(), help="Reasoning mode")
    parser.add_argument("--effort", type=str, choices=VALID_EFFORTS, default=os.getenv("TITAN_EFFORT", "auto").lower(), help="Effort level")
    parser.add_argument("--strategy", type=str, choices=VALID_STRATEGIES, default="auto", help="Reasoning strategy (auto, plan, react, tot, reflexion, debate)")
    parser.add_argument("--meta", "--orchestrator", action="store_true", help="Run with Genesis Level 1 Meta-Orchestrator")
    parser.add_argument("--department", "--dep", action="append", help="Target department(s) for Meta-Orchestrator")
    parser.add_argument("--dag", action="store_true", help="Run with Genesis Level 5 Task Graph (DAG) Parallel Planner")
    parser.add_argument("prompt", nargs="*", help="Single-shot prompt to run directly")
    args = parser.parse_args()

    provider, model = args.provider, args.model
    if not provider or not model:
        def_prov, def_mod = await resolve_cli_provider()
        provider = provider or def_prov
        model = model or def_mod
        
    mode = args.mode
    effort = args.effort

    # Initialize MCP Manager
    mcp = MCPManager(MCP_CONFIG_FILE)
    with console.status("[bold green]⚡ Starting Universal Agent HP...", spinner="dots"):
        await mcp.start_all()

    # Count MCP servers and tools
    mcp_count = len([s for s in mcp.servers.values() if getattr(s, 'is_connected', False) or (isinstance(s, dict) and s.get("status") == "connected")]) if hasattr(mcp, 'servers') else 0
    tool_count = len(mcp.get_all_tools()) if hasattr(mcp, 'get_all_tools') else 0

    # Show professional dashboard
    console.print()
    console.print(_build_dashboard(provider, model, mode, effort, mcp_count, tool_count))
    console.print()

    llm = LLMClient(provider=provider, model=model)
    agent = TitanAgent(mcp=mcp, llm=llm)

    session_id = "cli_session"
    universal_completer = UniversalCompleter(ALL_COMMANDS, workspace_root=Path.cwd())
    prompt_session = PromptSession(
        history=InMemoryHistory(),
        completer=universal_completer,
        complete_while_typing=True,
        bottom_toolbar=lambda: _bottom_toolbar(provider, model, mode, effort),
    )

    single_shot_prompt = " ".join(args.prompt).strip() if args.prompt else None

    if args.dag:
        from titan_agent.orchestrator import MetaOrchestrator
        orch = MetaOrchestrator()
        target_prompt = single_shot_prompt
        if not target_prompt:
            console.print("[bold cyan]📊 TASK GRAPH (DAG) Interactive Mode[/bold cyan] (Enter multi-step goal, or 'exit')")
            try:
                user_input = await prompt_session.prompt_async("DAG-Goal> ")
                if user_input.strip().lower() in ("exit", "quit"):
                    await mcp.stop_all()
                    return
                target_prompt = user_input.strip()
            except (KeyboardInterrupt, EOFError):
                await mcp.stop_all()
                return

        if target_prompt:
            with console.status("[bold magenta]📊 Planning & Executing Task Graph in parallel waves...", spinner="dots"):
                dag_res = await orch.orchestrate_dag(target_prompt, session_id=session_id)
            verdict = "SUCCESS" if dag_res.success else "FAILED"
            console.print(Panel(Markdown(dag_res.summary), title=f"[bold green]Task Graph Execution: {verdict}[/bold green]", border_style="green" if dag_res.success else "red"))
            await mcp.stop_all()
            return

    if args.meta:
        from titan_agent.orchestrator import MetaOrchestrator
        orch = MetaOrchestrator()
        target_prompt = single_shot_prompt
        if not target_prompt:
            console.print("[bold cyan]🏛️ META-ORCHESTRATOR Interactive Mode[/bold cyan] (Enter project goal, or 'exit')")
            try:
                user_input = await prompt_session.prompt_async("Goal> ")
                if user_input.strip().lower() in ("exit", "quit"):
                    await mcp.stop_all()
                    return
                target_prompt = user_input.strip()
            except (KeyboardInterrupt, EOFError):
                await mcp.stop_all()
                return

        if target_prompt:
            with console.status("[bold magenta]🏛️ Meta-Orchestrator dispatching to Department Leads...", spinner="dots"):
                res = await orch.orchestrate(target_prompt, departments=args.department, session_id=session_id)
            console.print(Panel(Markdown(res.synthesis), title=f"[bold green]Meta-Orchestrator Verdict: {res.final_verdict}[/bold green]", border_style="green"))
            await mcp.stop_all()
            return

    if single_shot_prompt:
        console.print(f"\n[bold magenta]⚡ Titan is working (mode: {mode}, effort: {effort}, strategy: {args.strategy})...[/bold magenta]")
        try:
            async for event in agent.run_task(single_shot_prompt, session_id=session_id, mode=mode, effort=effort, strategy=args.strategy):
                if event.type == "thought":
                    console.print(Panel(f"[italic dim]{event.data}[/italic dim]", title="[magenta]🧠 Reasoning (Chain-of-Thought)[/magenta]", border_style="magenta"))
                elif event.type == "tool_call":
                    console.print(f"🔧 [bold yellow]Tool called:[/bold yellow] [cyan]{event.data.get('name')}[/cyan] -> [dim]{event.data.get('arguments')}[/dim]")
                elif event.type == "tool_result":
                    res = event.data.get("result", "")
                    preview = str(res)[:300] + ("..." if len(str(res)) > 300 else "")
                    console.print(f"📋 [dim green]Result:[/dim green] [dim]{preview}[/dim]")
                elif event.type == "final_answer":
                    console.print(Panel(Markdown(event.data), title="[bold green]Titan Agent Answer[/bold green]", border_style="green"))
                elif event.type == "error":
                    console.print(f"[bold red]❌ Error:[/bold red] {event.data}")
                elif event.type == "status":
                    console.print(f"[dim blue]ℹ️ {event.data}[/dim blue]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        finally:
            await mcp.stop_all()
        return

    while True:
        try:
            user_input = await prompt_session.prompt_async("❯ ")
            if not user_input.strip():
                continue

            # Mode switch command
            lower_input = user_input.strip().lower()
            if lower_input.startswith("mode "):
                new_mode = lower_input.split(None, 1)[1].strip()
                if new_mode in VALID_MODES:
                    mode = new_mode
                    console.print(f"[bold cyan]Mode switched to: {mode}[/bold cyan]")
                else:
                    console.print(f"[bold red]Unknown mode: {new_mode}. Valid: {', '.join(VALID_MODES)}[/bold red]")
                continue
            # Effort level command (also accepts `/effort high`)
            if lower_input.startswith(("effort ", "/effort ")):
                new_effort = lower_input.split(None, 1)[1].strip()
                if new_effort in VALID_EFFORTS:
                    effort = new_effort
                    console.print(f"[bold cyan]Effort level switched to: {effort}[/bold cyan]")
                else:
                    console.print(f"[bold red]Unknown effort: {new_effort}. Valid: {', '.join(VALID_EFFORTS)}[/bold red]")
                continue
            # Slash commands: local ones (help/status/memory/...) and LLM ones (/plan, /review, ...)
            if lower_input.startswith("/"):
                local = parse_local(user_input)
                if local is not None:
                    name_l = local["name"]
                    if name_l == "help":
                        console.print(Panel.fit(
                            "\n".join(f"[cyan]/{k}[/cyan] — {v}" for k, v in ALL_COMMANDS.items()),
                            title="[bold magenta]Slash Commands[/bold magenta]",
                            border_style="magenta"
                        ))
                    elif name_l == "dashboard":
                        console.print(_build_dashboard(provider, model, mode, effort, mcp_count, tool_count))
                    elif name_l == "free-models":
                        console.print(_render_free_models(local.get("arg", "")))
                    elif name_l == "security-status":
                        console.print(_render_security_status())
                    elif name_l == "banned-ips":
                        console.print(_render_banned_ips(local.get("arg", "")))
                    elif name_l == "files":
                        console.print(_render_files_table(Path.cwd(), filter_str=local.get("arg", "")))
                    elif name_l == "status":
                        console.print(f"[bold cyan]Provider:[/bold cyan] {provider} | [bold cyan]Model:[/bold cyan] {model}\n"
                                      f"[bold cyan]Mode:[/bold cyan] {mode} | [bold cyan]Effort:[/bold cyan] {effort}")
                    elif name_l == "skills":
                        skill_names = agent.skills.list_skills()
                        if skill_names:
                            console.print(Panel.fit(
                                "\n".join(f"- [cyan]{s['name']}[/cyan]: {s['description']}" for s in skill_names),
                                title="[bold magenta]Skill Playbooks[/bold magenta]",
                                border_style="magenta"
                            ))
                        else:
                            console.print("[yellow]No skills loaded.[/yellow]")
                    elif name_l == "domain":
                        arg = local.get("arg", "").strip()
                        if not arg:
                            domains = agent.domain_manager.list_domains()
                            lines = []
                            for d in domains:
                                active = " [bold green][ACTIVE][/bold green]" if d["is_active"] else ""
                                lines.append(f"{d['icon']} [cyan]{d['name']}[/cyan]: {d['display_name']}{active}\n  [dim]{d['description']}[/dim]")
                            console.print(Panel.fit(
                                "\n".join(lines),
                                title="[bold magenta]Omni-Domain Industry Profiles[/bold magenta]",
                                border_style="magenta",
                            ))
                        else:
                            try:
                                switched = agent.domain_manager.switch_domain(arg)
                                console.print(f"[bold green]Switched active domain to: {switched.icon} {switched.display_name} ({switched.name})[/bold green]")
                            except ValueError as err:
                                console.print(f"[bold red]{err}[/bold red]")
                    elif name_l == "memory":
                        query = local["arg"]
                        if not query:
                            facts = agent.memory.get_all_knowledge()
                            console.print("[yellow]Usage: /memory <query>[/yellow]")
                        else:
                            facts = agent.memory.search_knowledge(query, scope=None)
                        if facts and query:
                            console.print(Panel.fit(
                                "\n".join(f"- [{f.get('category')}] {f['key']}: {f['value']}" for f in facts),
                                title=f"[bold magenta]Memory: {query}[/bold magenta]",
                                border_style="magenta"
                            ))
                    elif name_l == "handoffs":
                        msgs = agent.memory.list_handoffs(status="open")
                        if msgs:
                            console.print(Panel.fit(
                                "\n".join(f"- [{m['id']}] {m['title']}" for m in msgs),
                                title="[bold magenta]Open Handoffs[/bold magenta]",
                                border_style="magenta"
                            ))
                        else:
                            console.print("[yellow]No open handoffs.[/yellow]")
                    elif name_l == "clear":
                        agent.memory.clear_session(session_id)
                        console.print("[bold cyan]Session history cleared.[/bold cyan]")
                    elif name_l == "queue":
                        from titan_agent.config import TASK_QUEUE_FILE
                        from titan_agent.queue import TaskQueue

                        q = TaskQueue(TASK_QUEUE_FILE)
                        sub = local["arg"].split(None, 1)
                        op = sub[0].lower() if sub else "list"
                        try:
                            if op == "list":
                                tasks = q.list(limit=50)
                                if not tasks:
                                    console.print("[yellow]Task queue is empty.[/yellow]")
                                else:
                                    console.print(Panel(
                                        "\n".join(
                                            f"[cyan]#{t.id}[/cyan] [{t.status}] prio={t.priority} "
                                            f"attempts={t.attempts}/{t.max_attempts} :: {t.name}"
                                            for t in tasks
                                        ),
                                        title="[bold magenta]Task Queue[/bold magenta]",
                                        border_style="magenta",
                                    ))
                            elif op == "stats":
                                console.print(f"[bold cyan]Queue stats:[/bold cyan] {q.stats()}")
                            elif op == "add":
                                task_text = sub[1].strip() if len(sub) > 1 else ""
                                if not task_text:
                                    console.print("[yellow]Usage: /queue add <task>[/yellow]")
                                else:
                                    tid = q.enqueue(task_text)
                                    console.print(f"[bold green]Task #{tid} enqueued.[/bold green]")
                            elif op == "cancel":
                                tid = int(sub[1]) if len(sub) > 1 else 0
                                ok = q.cancel(tid)
                                console.print(
                                    f"[green]Task #{tid} cancelled.[/green]" if ok
                                    else "[red]Task not found or already running.[/red]"
                                )
                            else:
                                console.print("[yellow]Unknown: /queue list|stats|add <task>|cancel <id>[/yellow]")
                        except (OSError, ValueError) as e:
                            console.print(f"[red]Queue error: {e}[/red]")
                    elif name_l == "daemon":
                        from titan_agent.config import (
                            DAEMON_POLL_INTERVAL,
                            TASK_QUEUE_FILE,
                        )
                        from titan_agent.daemon import TaskDaemon
                        from titan_agent.queue import TaskQueue

                        console.print("[bold magenta]⚙️  Daemon: processing due tasks (Ctrl+C to stop)...[/bold magenta]")
                        q = TaskQueue(TASK_QUEUE_FILE)

                        def _runner(task: str, opts: dict):
                            import asyncio as _aio

                            final = ""
                            err = ""

                            async def _run():
                                nonlocal final, err
                                try:
                                    async for ev in agent.run_task(task, session_id="daemon-cli", mode="fast"):
                                        if ev.type == "final_answer":
                                            final = ev.data
                                        elif ev.type == "error":
                                            err = str(ev.data)
                                except Exception as exc:  # noqa: BLE001
                                    err = str(exc)

                            try:
                                # We are already in an asyncio loop here, we can't do _aio.run()
                                # But _runner is called by TaskDaemon in a thread via asyncio.to_thread
                                # So _aio.run is valid INSIDE the thread!
                                _aio.run(_run())
                            except Exception as exc:  # noqa: BLE001
                                err = str(exc)
                            return (0 if (final and not err) else 1), final or err or "(no answer)", []

                        # Route each queue task through the live agent (fully autonomous).
                        daemon = TaskDaemon(q, runner=_runner, poll_interval=DAEMON_POLL_INTERVAL)

                        try:
                            await daemon.run_forever()
                        except KeyboardInterrupt:
                            console.print("\n[yellow]Daemon stopped by user.[/yellow]")
                    elif name_l == "exit":
                        console.print("[bold red]Titan Agent stopped. Goodbye![/bold red]")
                        await mcp.stop_all()
                        break
                    continue
                expanded = expand_slash(user_input)
                if expanded is not None:
                    mode = expanded["mode"]
                    effort = expanded["effort"]
                    console.print(f"[bold cyan]/{expanded['command']}[/bold cyan] → mode: {mode}, effort: {effort}")
                    user_input = expanded["prompt"]

            if lower_input in ("exit", "quit"):
                console.print("[bold red]Titan Agent stopped. Goodbye![/bold red]")
                await mcp.stop_all()
                break

            # Resolve @file mentions and auto-attach file content into prompt
            resolved_input, attached_files = _resolve_file_mentions(user_input, Path.cwd())
            if attached_files:
                console.print(f"[dim cyan]📎 Attached {len(attached_files)} file(s): {', '.join(attached_files)}[/dim cyan]")

            console.print(f"\n[bold magenta]⚡ Titan is working (mode: {mode}, effort: {effort})...[/bold magenta]")

            async for event in agent.run_task(resolved_input, session_id=session_id, mode=mode, effort=effort):
                if event.type == "thought":
                    console.print(Panel(
                        f"[italic dim]{event.data}[/italic dim]",
                        title="[magenta]🧠 Reasoning (Chain-of-Thought)[/magenta]",
                        border_style="magenta"
                    ))
                elif event.type == "tool_call":
                    name = event.data.get("name")
                    args = event.data.get("arguments")
                    console.print(f"🔧 [bold yellow]Tool called:[/bold yellow] [cyan]{name}[/cyan] -> [dim]{args}[/dim]")
                elif event.type == "tool_result":
                    res = event.data.get("result", "")
                    # Preview first 300 chars
                    preview = str(res)[:300] + ("..." if len(str(res)) > 300 else "")
                    console.print(f"📋 [dim green]Result:[/dim green] [dim]{preview}[/dim]")
                elif event.type == "final_answer":
                    console.print(Panel(
                        Markdown(event.data),
                        title="[bold green]Titan Agent Answer[/bold green]",
                        border_style="green"
                    ))
                elif event.type == "error":
                    console.print(f"[bold red]❌ Error:[/bold red] {event.data}")
                elif event.type == "status":
                    console.print(f"[dim blue]ℹ️ {event.data}[/dim blue]")

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Process cancelled.[/bold red]")
            await mcp.stop_all()
            break
        except Exception as e:  # noqa: BLE001 - top-level loop must survive and report any error
            console.print(f"[bold red]Unexpected error:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())