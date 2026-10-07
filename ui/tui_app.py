"""
Universal Agent HP - Modern Terminal User Interface (TUI)
Built with Textual & Rich for ultra-modern terminal aesthetics (OpenCode / Claude Code style).
"""
from __future__ import annotations

import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Input, Static, Label, Button, RichLog
from textual.binding import Binding
from textual.screen import ModalScreen

try:
    import pyfiglet
    PYFIGLET_AVAILABLE = True
except ImportError:
    PYFIGLET_AVAILABLE = False


class AgentsModal(ModalScreen):
    """Modal displaying the 27 specialist agents in Universal Agent HP."""

    DEFAULT_CSS = """
    AgentsModal {
        align: center middle;
        background: rgba(13, 17, 23, 0.85);
    }
    #agents-dialog {
        width: 80%;
        max-width: 85;
        height: 70%;
        background: #161b22;
        border: solid #58a6ff;
        padding: 1 2;
    }
    #agents-title {
        text-align: center;
        text-style: bold;
        color: #58a6ff;
        margin-bottom: 1;
    }
    #agents-list {
        height: 1fr;
        border: solid #30363d;
        padding: 0 1;
    }
    #close-btn {
        margin-top: 1;
        width: 100%;
        background: #238636;
        color: #ffffff;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    AGENTS_DATA = [
        ("01. Meta-Orchestrator", "Task routing, DAG execution & high-level planning"),
        ("02. Architect", "System design, boundary invariants & API structures"),
        ("03. Security Officer", "Dual-Shield guardrails, AST injection & secret detection"),
        ("04. Pragmatist", "Feasibility, fallback execution & real-world sanity"),
        ("05. Dynamic Tool Synthesizer", "On-the-fly Python tool synthesis & sandboxed hot-reload"),
        ("06. Symbolic AST Checker", "Static scan for unbounded loops & AST invariants"),
        ("07. Autonomous TDD Engine", "Red-Green-Refactor test-driven code execution"),
        ("08. Working Memory Virtualizer", "Operational HUD pinned context against memory drift"),
        ("09. Consensus Engine", "Multi-agent committee voting with signed formal memos"),
        ("10. GitOps PR Engine", "Isolated branch creation, test-gating & automated PRs"),
        ("11. MCP Preset Manager", "1-line connector for Postgres, GitHub, Slack, Brave, etc."),
        ("12. Semantic Cache Sentinel", "Cosine similarity SQLite cache saving 30-40% tokens"),
        ("13. Episodic Experience Replay", "Fingerprint-matched error remediation repository"),
        ("14. Deep Verifier", "Pytest sandbox runner with self-healing feedback"),
        ("15. Multi-Hop ReAct Planner", "Iterative thought-action-observation chain"),
        ("16. Shannon Entropy Monitor", "Cognitive stagnation detection & strategy pivoting"),
        ("17. AST Surgical Patcher", "Exact line-free function/class replacement"),
        ("18. Workspace RAG Searcher", "Vector indexing & symbol reference graph"),
        ("19. Docker Sandbox Runner", "Hermetic container runtime for untrusted code"),
        ("20. Blue Team Sentinel", "Continuous IP defense & credential leak guard"),
        ("21. Red Team Auditor", "Forensic analysis of execution failures & threats"),
        ("22. Telegram Bot Bridge", "Remote notification & mobile human-in-the-loop"),
        ("23. Webhook Dispatcher", "Enterprise webhook events & telemetry broadcasting"),
        ("24. Token Budget Guard", "Rate limiting & financial spend optimization"),
        ("25. Multi-Model Router", "Zero-cost local/cloud fallbacks (Ollama, Groq, Puter)"),
        ("26. Playwright DOM Inspector", "Headless browser automated UI validation"),
        ("27. Singularity Evolutionist", "Continuous self-improvement & rule optimization"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="agents-dialog"):
            yield Label("⚡ 27 SPECIALIST AGENTS MATRIX", id="agents-title")
            with VerticalScroll(id="agents-list"):
                for name, desc in self.AGENTS_DATA:
                    yield Static(f"[bold cyan]{name}[/bold cyan]: [dim]{desc}[/dim]")
            yield Button("Press ESC or Click to Close", id="close-btn", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class CommandsModal(ModalScreen):
    """Modal command palette for Universal Agent HP."""

    DEFAULT_CSS = """
    CommandsModal {
        align: center middle;
        background: rgba(13, 17, 23, 0.85);
    }
    #commands-dialog {
        width: 80%;
        max-width: 75;
        height: 60%;
        background: #161b22;
        border: solid #f0883e;
        padding: 1 2;
    }
    #commands-title {
        text-align: center;
        text-style: bold;
        color: #f0883e;
        margin-bottom: 1;
    }
    #commands-list {
        height: 1fr;
        border: solid #30363d;
        padding: 0 1;
    }
    #close-cmd-btn {
        margin-top: 1;
        width: 100%;
        background: #f0883e;
        color: #0d1117;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
    ]

    COMMANDS_DATA = [
        ("/dag", "Run multi-phase dependency graph pipeline"),
        ("/debate", "Convene Architect, Security & Pragmatist consensus"),
        ("/mcp", "Connect or list 1-line MCP server presets"),
        ("/pr", "Open automated feature branch & GitHub Pull Request"),
        ("/cache", "Inspect Semantic Cache hits and saved tokens"),
        ("/memory", "Inspect Episodic Memory & Working Memory HUD"),
        ("/domain", "Switch industry domain profile (Finance, Healthcare, Legal, etc.)"),
        ("/rollback", "Undo last code modification or git checkpoint"),
        ("/status", "Show active LLM provider, latency and telemetry"),
        ("/clear", "Clear chat output and restart visual session"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="commands-dialog"):
            yield Label("⚙️ COMMAND PALETTE (CTRL+P)", id="commands-title")
            with VerticalScroll(id="commands-list"):
                for cmd, desc in self.COMMANDS_DATA:
                    yield Static(f"[bold yellow]{cmd:<12}[/bold yellow] [dim]{desc}[/dim]")
            yield Button("Press ESC to Close", id="close-cmd-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class UniversalAgentTUI(App):
    """Ultra-modern full-screen TUI for Universal Agent HP."""

    TITLE = "Universal Agent HP"
    SUB_TITLE = "Terminal AI Operating System"

    CSS = """
    Screen {
        background: #0d1117;
        align: center middle;
    }

    #session-bar {
        dock: top;
        height: 2;
        background: #161b22;
        border-bottom: solid #30363d;
        padding: 0 2;
    }

    .session-tab {
        color: #58a6ff;
        text-style: bold;
        margin-right: 2;
    }

    .session-new {
        color: #8b949e;
    }

    #main-container {
        width: 86%;
        max-width: 96;
        height: 1fr;
        align: center middle;
        padding: 1 0;
    }

    #logo {
        text-align: center;
        color: #e6edf3;
        margin-bottom: 1;
        text-style: bold;
    }

    #output-log {
        width: 100%;
        height: 10;
        background: #161b22;
        border: solid #30363d;
        color: #e6edf3;
        padding: 0 1;
        margin-bottom: 1;
    }

    #input-box {
        width: 100%;
        background: #161b22;
        border: solid #30363d;
        color: #e6edf3;
        padding: 0 1;
        height: 3;
    }

    #input-box:focus {
        border: solid #58a6ff;
    }

    #sub-bar {
        width: 100%;
        margin-top: 1;
        color: #8b949e;
        height: 1;
    }

    .badge-mode {
        color: #58a6ff;
        text-style: bold;
    }

    .separator {
        color: #30363d;
        margin: 0 1;
    }

    .badge-model {
        color: #f0883e;
        text-style: bold;
    }

    .sub-text {
        color: #8b949e;
        margin-left: 1;
    }

    #shortcuts-bar {
        width: 100%;
        margin-top: 1;
        color: #8b949e;
        text-align: right;
    }
    """

    BINDINGS = [
        Binding("shift+tab", "toggle_agents", "Agents"),
        Binding("ctrl+p", "open_commands", "Commands"),
        Binding("ctrl+l", "clear_output", "Clear"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    LOGO = """
  █  █ █▄ █ ▀█▀ █ █ █▀▀ █▀█ █▀▀ ▄▀█ █   
  █▄▄█ █ ▀█  █  ▀▄▀ ██▄ █▀▄ ▄██ █▀█ █▄▄ 
     ─── H P   A G E N T   O S ───
    """

    def __init__(self, agent=None):
        super().__init__()
        self._agent = agent
        self.output_log = RichLog(id="output-log", highlight=True, markup=True)

    def compose(self) -> ComposeResult:
        with Horizontal(id="session-bar"):
            yield Label("● Session #1 (Active)", classes="session-tab")
            yield Label("+ New session", classes="session-new")

        with Vertical(id="main-container"):
            # Markaziy ASCII logotip
            yield Static(self.LOGO, id="logo")

            # Jonli chiqish oynasi (Output log)
            yield self.output_log

            # Kiritish qatori (Input)
            yield Input(placeholder='Ask anything... "Fix broken tests or build a feature"', id="input-box")

            # Input ostidagi rejim va model ma'lumotlari
            with Horizontal(id="sub-bar"):
                yield Label("Build ", classes="badge-mode")
                yield Label("· ", classes="separator")
                yield Label("OmniRoute / Universal-R1 ", classes="badge-model", id="active-model")
                yield Label("Universal Zen", classes="sub-text")

            # Qisqa klaviatura buyruqlari
            with Horizontal(id="shortcuts-bar"):
                yield Label("shift+tab agents   ctrl+p commands   ctrl+l clear   v1.0.0")

        yield Footer()

    def on_mount(self) -> None:
        self.output_log.write("[bold green]Universal Agent HP Initialized.[/bold green] System ready.")
        self.output_log.write("[dim]Type your command above or press [bold cyan]ctrl+p[/bold cyan] for actions, [bold cyan]shift+tab[/bold cyan] for agents.[/dim]")
        input_box = self.query_one("#input-box", Input)
        input_box.focus()
        self.set_interval(1.0, self.update_model_badge)

    def update_model_badge(self) -> None:
        if self._agent and hasattr(self._agent, "llm"):
            try:
                badge = self.query_one("#active-model", Label)
                current_text = f"{self._agent.llm.provider} / {self._agent.llm.model} "
                if badge.renderable != current_text and "OmniRoute" not in str(badge.renderable):
                    self.output_log.write(f"[dim yellow]🔄 Model fallback initiated. Seamlessly switched to {current_text.strip()}[/dim yellow]")
                badge.update(current_text)
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        event.input.value = ""
        self.output_log.write(f"\n[bold yellow]User:[/bold yellow] {user_text}")

        # Command shortcuts handling
        if user_text.lower() == "/clear":
            self.action_clear_output()
            return
        elif user_text.lower() == "/agents":
            self.action_toggle_agents()
            return
        elif user_text.lower() == "/help" or user_text.lower() == "/commands":
            self.action_open_commands()
            return

        self.notify(f"Vazifa qabul qilindi: {user_text[:40]}...")
        asyncio.create_task(self._execute_agent_task(user_text))

    async def _execute_agent_task(self, prompt: str) -> None:
        self.output_log.write("[dim cyan]⚡ Routing intent -> Dual-Shield Guard -> Tri-Loop Engine...[/dim cyan]")
        try:
            if self._agent is None:
                from titan_agent.agent import TitanAgent
                self._agent = TitanAgent()

            final_answer = ""
            if hasattr(self._agent, "run_task"):
                async for event in self._agent.run_task(prompt):
                    ev_type = getattr(event, "type", "")
                    ev_data = getattr(event, "data", "")
                    if ev_type == "thought" and ev_data:
                        self.output_log.write(f"[dim magenta]🧠 {ev_data}[/dim magenta]")
                    elif ev_type == "tool_call":
                        tool_name = ev_data.get("name", "tool") if isinstance(ev_data, dict) else str(ev_data)
                        self.output_log.write(f"[yellow]🔧 Tool:[/yellow] [cyan]{tool_name}[/cyan]")
                    elif ev_type == "final_answer":
                        final_answer = str(ev_data)
                    elif ev_type == "error":
                        self.output_log.write(f"[bold red]❌ Error:[/bold red] {ev_data}")
                    elif ev_type == "status":
                        self.output_log.write(f"[dim blue]ℹ️ {ev_data}[/dim blue]")
            elif hasattr(self._agent, "run"):
                final_answer = await self._agent.run(prompt)
            else:
                final_answer = f"Command processed: {prompt}"

            if final_answer:
                self.output_log.write(f"[bold green]Extra LLM X:[/bold green]\n{final_answer}")
            else:
                self.output_log.write("[bold green]Extra LLM X:[/bold green] Task finished.")
        except Exception as e:
            self.output_log.write(f"[bold red]Execution Error:[/bold red] {e}")

    def action_toggle_agents(self) -> None:
        self.push_screen(AgentsModal())

    def action_open_commands(self) -> None:
        self.push_screen(CommandsModal())

    def action_clear_output(self) -> None:
        self.output_log.clear()
        self.output_log.write("[bold green]Terminal output cleared.[/bold green]")


# Backwards compatibility alias
TitanOS = UniversalAgentTUI

if __name__ == "__main__":
    app = UniversalAgentTUI()
    app.run()
