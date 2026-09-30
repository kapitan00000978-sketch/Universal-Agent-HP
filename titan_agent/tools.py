import asyncio
import ipaddress
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import shlex
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
from . import config as _cfg
from .config import TASK_QUEUE_FILE, WORKSPACE_DIR
from .core.guardrails.policy import PolicyEngine


async def _terminate_process(proc: Any, *, process_group: bool = False) -> None:
    """Terminate a subprocess (and optionally its POSIX process group)."""
    try:
        if process_group and os.name == "posix" and getattr(proc, "pid", None):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            killed = proc.kill()
            if asyncio.iscoroutine(killed):
                await killed
    except ProcessLookupError:
        pass
    waited = proc.wait()
    if asyncio.iscoroutine(waited):
        await waited


MAX_CAPTURE_BYTES_PER_STREAM = 64 * 1024
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
FULL_ACCESS_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


async def _read_bounded(stream: Any, limit: int, overflow: asyncio.Event) -> tuple[bytes, bool]:
    """Read a subprocess pipe without buffering unbounded command output."""
    captured = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(min(65536, limit - len(captured) + 1))
        if not chunk:
            break
        remaining = limit - len(captured)
        if len(chunk) > remaining:
            captured.extend(chunk[:remaining])
            truncated = True
            overflow.set()
            break
        captured.extend(chunk)
    return bytes(captured), truncated


async def _bounded_communicate(
    proc: Any,
    timeout: float,
    limit: int = MAX_CAPTURE_BYTES_PER_STREAM,
    *,
    process_group: bool = False,
) -> tuple[bytes, bytes, bool, bool]:
    """Collect stdout/stderr with a memory cap and stop timed-out/noisy processes.

    Returns ``stdout, stderr, timed_out, output_truncated``. Cancellation kills
    the direct subprocess and drains readers before propagating to the caller.
    """
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("subprocess output pipes are required")
    overflow = asyncio.Event()
    stdout_task = asyncio.create_task(_read_bounded(proc.stdout, limit, overflow))
    stderr_task = asyncio.create_task(_read_bounded(proc.stderr, limit, overflow))
    wait_task = asyncio.create_task(proc.wait())
    overflow_task = asyncio.create_task(overflow.wait())
    timed_out = False
    try:
        done, _pending = await asyncio.wait(
            {wait_task, overflow_task},
            timeout=max(0.1, float(timeout)),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            timed_out = True
            await _terminate_process(proc, process_group=process_group)
        elif overflow_task in done and overflow.is_set() and not wait_task.done():
            await _terminate_process(proc, process_group=process_group)
        if not wait_task.done():
            await wait_task
        stdout_result, stderr_result = await asyncio.gather(stdout_task, stderr_task)
        truncated = stdout_result[1] or stderr_result[1]
        return stdout_result[0], stderr_result[0], timed_out, truncated
    except asyncio.CancelledError:
        await _terminate_process(proc, process_group=process_group)
        await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)
        raise
    except Exception:
        await _terminate_process(proc, process_group=process_group)
        await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)
        raise
    finally:
        overflow_task.cancel()
        await asyncio.gather(overflow_task, return_exceptions=True)


async def _best_effort_docker_rm(docker_bin: str, container_name: str) -> None:
    """Force-remove a timed-out/cancelled container without retaining output."""
    try:
        cleanup = await asyncio.create_subprocess_exec(
            docker_bin, "rm", "-f", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(cleanup.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            await _terminate_process(cleanup)
    except (OSError, RuntimeError, ProcessLookupError):
        pass


def _command_timeout(base: float) -> float:
    """Command/process timeout in seconds.

    Phase 8 FULL access removes the 45s command ceiling (and the 60s raw-run
    default): long builds, big installs and slow network jobs are allowed to
    run for up to 10 minutes before the watchdog intervenes.
    """
    return 600.0 if _cfg.full_access_enabled() else base


def _subagent_worker_cap() -> int:
    """Parallel subagent worker bound. Phase 8 FULL access raises 2 -> 8."""
    return 8 if _cfg.full_access_enabled() else 2


def _subagent_result_text(res: Any) -> str:
    """Phase 36: render a delegated-subagent result for the parent's context.

    Success keeps the classic report, but a FAILED or ERRORED child is turned
    into an Error-prefixed result so the uniform tool funnel treats it as a
    failed TOOL execution: it records in telemetry, increments the
    repeated-failure guard keyed by (tool, task), and warns the critic via the
    tool-evidence snippet. A weak parent must never treat a crashing child's
    half-baked output as proven work — this makes the failure loud AND
    self-reinforcing (re-delegating the identical task eventually gets blocked).
    """
    label = str(getattr(res, "label", "worker"))
    if getattr(res, "error", None):
        head = f"### SUBAGENT [{label}] - ERROR"
        body = str(res.error)
    elif getattr(res, "exit_code", 0) != 0:
        head = f"### SUBAGENT [{label}] - FAILED (exit {res.exit_code})"
        body = str(getattr(res, "final", "") or "(no output)")
    else:
        return res.to_text()
    return (
        f"Error: {head}\n{body}"
        "\n"
        "\u26a0 This delegated subagent FAILED - its output is UNPROVEN. "
        "Do not present it as done work: verify the files/tests yourself, or "
        "re-delegate with a corrected task."
    )


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for lightweight lexical ranking."""
    return re.findall(r"[a-z0-9][a-z0-9_\-']*", text.lower())


class WorkspaceRAG:
    """Zero-dependency retrieval over the workspace.

    A lightweight, lexical (BM25-style) index over text files: documents are
    split into overlapping chunks, scored against the query, and the top
    chunks are returned with file paths so the LLM can answer WITH citations.
    No external embeddings, no API keys — everything runs locally.
    """

    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 140
    MAX_FILE_BYTES = 512 * 1024
    TEXT_SUFFIXES: frozenset[str] = frozenset({
        ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".md", ".txt",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".xml",
        ".sql", ".sh", ".ps1", ".bat", ".env", ".log",
    })

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)

    def _iter_documents(self):
        """Yield (relative_path, text) for every searchable file in the workspace."""
        if not self.workspace.exists():
            return
        for fpath in self.workspace.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in self.TEXT_SUFFIXES:
                continue
            try:
                if fpath.stat().st_size > self.MAX_FILE_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not text.strip():
                continue
            yield fpath.relative_to(self.workspace).as_posix(), text

    @staticmethod
    def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _bm25(chunk_tokens: list[str], query_tokens: list[str], avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
        if not chunk_tokens or not query_tokens or avg_len <= 0:
            return 0.0
        dl = len(chunk_tokens)
        freq: dict[str, int] = {}
        for t in chunk_tokens:
            freq[t] = freq.get(t, 0) + 1
        score = 0.0
        for qt in set(query_tokens):
            f = freq.get(qt, 0)
            if f == 0:
                continue
            tf_part = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avg_len)))
            score += tf_part
        return score

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        candidates: list[dict[str, Any]] = []
        for rel_path, text in self._iter_documents():
            chunks = self._chunk_text(text)
            chunk_lens = [len(_tokenize(c)) for c in chunks]
            avg_len = max(1.0, sum(chunk_lens) / len(chunk_lens)) if chunk_lens else 1.0
            for i, chunk in enumerate(chunks):
                ct = _tokenize(chunk)
                score = self._bm25(ct, query_tokens, avg_len)
                if score <= 0:
                    continue
                # Bonus for earlier chunks (files usually front-load meaning).
                score += max(0.0, 0.15 * (1 - i / max(len(chunks), 1)))
                candidates.append({
                    "path": rel_path,
                    "chunk_index": i,
                    "score": round(score, 4),
                    "snippet": chunk.strip()[:700],
                })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        # Keep at most one chunk per file unless the file is clearly central.
        picked: list[dict[str, Any]] = []
        per_file: dict[str, int] = {}
        for c in candidates:
            per_file[c["path"]] = per_file.get(c["path"], 0) + 1
            if per_file[c["path"]] <= 2 and len(picked) < max(top_k, 1):
                picked.append(c)
        return picked[:top_k]


class ToolRegistry:
    def __init__(self, workspace: Path = WORKSPACE_DIR):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Phase 14: optional Human-in-the-loop manager. The approval gate itself
        # lives in agent.execute_tool_unified (single point for every loop), so
        # here we only accept the wiring to keep construction uniform.
        self.hitl = None
        self.hitl_timeout = 120.0
        self.mcp_manager = None

    def attach_mcp_manager(self, mcp_manager) -> None:
        """Wire this registry to the agent's live MCP manager."""
        self.mcp_manager = mcp_manager

    def attach_hitl(self, hitl, hitl_timeout: float | None = None) -> None:
        """Accept the global HITL manager (approvals enforced at the agent layer)."""
        self.hitl = hitl
        if hitl_timeout is not None:
            self.hitl_timeout = hitl_timeout

    def _resolve_path(self, rel_or_abs: str | Path) -> Path:
        """Resolve a tool path and keep ordinary access inside the workspace.

        FULL_ACCESS is an explicit trust-mode bypass. Resolving symlinks before
        the containment check prevents a workspace symlink from escaping the
        boundary for file tools.
        """
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()
        if not _cfg.full_access_enabled():
            root = self.workspace.resolve()
            try:
                p.relative_to(root)
            except ValueError as exc:
                raise PermissionError(
                    f"Path is outside the configured workspace: {p}"
                ) from exc
        return p

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        base_defs = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Executes a shell command in a resource-limited Docker container by default, with network disabled and the configured workspace mounted. It fails closed if Docker is unavailable. Explicit TITAN_FULL_ACCESS bypasses container isolation and runs on the host.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The exact command line string to run (e.g. 'dir', 'git status', 'npm test')."
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Optional working directory. Defaults to workspace root."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads the content of a file from the filesystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read."
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_python_file",
                    "description": "Read and statically explain a Python file in the workspace: module purpose, imports, classes/functions, signatures, control flow, and internal calls. Uses AST parsing only; it never imports or executes the file. Use before changing unfamiliar Python code. Static summaries can miss dynamic runtime behavior.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Workspace-relative Python source file path."}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Creates a new file or overwrites an existing file with provided content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path where the file should be saved."
                            },
                            "content": {
                                "type": "string",
                                "description": "The full text content to write."
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Performs exact string replacement in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path of the file to edit."
                            },
                            "target_text": {
                                "type": "string",
                                "description": "Exact existing text block to be replaced."
                            },
                            "replacement_text": {
                                "type": "string",
                                "description": "New text block to insert in place of target_text."
                            }
                        },
                        "required": ["path", "target_text", "replacement_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "Lists contents of a directory with file names and sizes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list. Defaults to current workspace."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Searches the live internet using DuckDuckGo to get up-to-date web results, news, or facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results (default 5)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "scrape_webpage",
                    "description": "Fetches raw text content from a web URL for reading articles or documentation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Web URL to scrape."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_eval",
                    "description": "Executes Python code in a standalone process and returns stdout/stderr.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "workspace_rag",
                    "description": "Searches all text/code files inside the workspace using a fast local lexical (BM25) retrieval index and returns the most relevant snippets WITH their file paths. Use this instead of read_file when you need to answer a question from documents, notes, or code that may live anywhere in the workspace — it finds the exact relevant lines fast.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or keywords to find inside workspace files."
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "How many snippets to return (default 4, range 1-8)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_python_repository",
                    "description": "Build a bounded, non-executing map of Python modules, definitions, syntax errors, and local import dependencies in the current workspace. Use early on unfamiliar repositories to understand structure and change impact. Static analysis only; dynamic imports and runtime behavior are not covered.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_files": {"type": "integer", "description": "Maximum Python modules to inspect (default 50, range 1-100)."}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "laya_decide",
                    "description": "API-key-free LOCAL structured inference using Laya or Laya-MLX. Accepts one state and typed choice/score/noul questions; returns structured decisions, not free-form text or code. Requires the optional runtime package and downloads model weights on first use. Laya-MLX requires Apple Silicon macOS.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string", "description": "Text state/document to evaluate."},
                            "questions": {"type": "object", "description": "Laya typed questions: choice, score, or noul (yes/no probability)."},
                            "backend": {"type": "string", "enum": ["auto", "laya", "laya-mlx"], "description": "Auto-select MLX on supported Apple Silicon if installed; otherwise upstream Laya."},
                            "model": {"type": "string", "description": "Optional Laya checkpoint ID; primarily for laya-mlx."
                            }
                        },
                        "required": ["state", "questions"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "deep_search",
                    "description": "Performs an in-depth multi-hop web research on a topic by querying multiple angles, scraping top websites, and synthesizing findings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "The complex subject or question to research deeply."
                            }
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "deep_coder",
                    "description": "Autonomous deep software engineering cycle: writes multi-file code, verifies syntax, generates test harness, and executes tests.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_name": {
                                "type": "string",
                                "description": "Short identifier for the module or feature."
                            },
                            "files": {
                                "type": "object",
                                "description": "Dictionary of filename to file code content."
                            },
                            "test_code": {
                                "type": "string",
                                "description": "Python test script code that asserts correctness."
                            }
                        },
                        "required": ["task_name", "files"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "launch_application",
                    "description": "Launches a Windows desktop application or utility.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_or_command": {
                                "type": "string",
                                "description": "Application name or path (e.g. 'notepad', 'calc', 'explorer .', 'chrome')"
                            }
                        },
                        "required": ["app_or_command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "system_info",
                    "description": "Returns live information about the host system: OS version, CPU, RAM (total/free), disk space, Python version — useful for environment-aware decisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_processes",
                    "description": "Lists or kills running OS processes. action='list' to see running processes (optionally filtered by pattern), action='kill' to terminate a process by PID or image name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "kill"],
                                "description": "'list' to show processes, 'kill' to terminate."
                            },
                            "pattern": {
                                "type": "string",
                                "description": "For 'list': substring to filter process names. For 'kill': PID number or image name (e.g. 'notepad.exe')."
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clipboard_get",
                    "description": "Reads text from the system clipboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clipboard_set",
                    "description": "Writes text to the system clipboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to put on the clipboard."
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "screenshot",
                    "description": "Takes a screenshot of the entire screen or a specific monitor and returns it as base64 PNG.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "monitor": {
                                "type": "integer",
                                "description": "Monitor index (0 = primary, 1 = secondary, etc.). Default 0.",
                                "default": 0
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "key_press",
                    "description": "Simulates keyboard key presses (e.g. 'ctrl+c', 'enter', 'alt+tab', 'win+r').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "string",
                                "description": "Key combination to press (e.g. 'ctrl+c', 'enter', 'alt+tab', 'win+r', 'f5')."
                            }
                        },
                        "required": ["keys"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mouse_click",
                    "description": "Simulates a mouse click at the specified coordinates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X coordinate."
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate."
                            },
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Mouse button to click. Default 'left'.",
                                "default": "left"
                            },
                            "double": {
                                "type": "boolean",
                                "description": "Whether to double-click. Default false.",
                                "default": False
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mouse_move",
                    "description": "Moves the mouse cursor to the specified coordinates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X coordinate."
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate."
                            },
                            "duration": {
                                "type": "number",
                                "description": "Duration in seconds for smooth movement. Default 0 (instant).",
                                "default": 0
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_windows",
                    "description": "Lists all visible windows with their titles, handles, and process names.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "window_control",
                    "description": "Controls a window: minimize, maximize, restore, close, or bring to front.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["minimize", "maximize", "restore", "close", "foreground"],
                                "description": "Action to perform on the window."
                            },
                            "title": {
                                "type": "string",
                                "description": "Window title (partial match) or handle (HWND as string)."
                            }
                        },
                        "required": ["action", "title"]
                    }
                }
            },
            # ================= Phase 7: Full Autonomy tools =================
            {
                "type": "function",
                "function": {
                    "name": "self_heal",
                    "description": "SELF-HEALING: runs a command and, if it fails, automatically diagnoses the error and applies deterministic repairs (installs a missing Python module via pip, retries transient failures) then re-runs the command until success or attempts are exhausted. Use this instead of execute_command when a dependency or flaky failure is suspected.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The exact command line string to run."
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Optional working directory. Defaults to workspace root."
                            },
                            "max_attempts": {
                                "type": "integer",
                                "description": "Max run attempts including repairs (default 3)."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "download_file",
                    "description": "Downloads a file from a public http(s) URL into the workspace (SSRF-guarded: private/loopback targets are refused). Returns the saved path and size.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The public http(s) URL to download."
                            },
                            "dest": {
                                "type": "string",
                                "description": "Optional destination path inside the workspace (default: filename from the URL)."
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_http_server",
                    "description": "Serves a directory (default workspace) over HTTP on localhost so the agent or user can browse generated files. Returns the URL. The server runs until the agent stops it with stop_http_server.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "port": {
                                "type": "integer",
                                "description": "Port to bind (default 8000)."
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to serve (default workspace)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stop_http_server",
                    "description": "Stops a previously started local HTTP server.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "port": {
                                "type": "integer",
                                "description": "Port of the server to stop (default 8000)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_screenshot",
                    "description": "Captures the primary screen to a PNG in the workspace (Windows; PowerShell-based, no extra deps). Use to visually inspect the current UI state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dest": {
                                "type": "string",
                                "description": "Optional PNG filename in the workspace (default screenshot_<ts>.png)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_update",
                    "description": "Pulls the latest code from git, installs requirements and runs the test suite for the project containing the workspace. Returns the update log. Use to keep Titan's own runtime current.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "run_tests": {
                                "type": "boolean",
                                "description": "Whether to run the test suite after updating (default true)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "task_enqueue",
                    "description": "AUTONOMOUS TASK QUEUE: adds a task that the daemon or another agent processes independently (with priorities, scheduling and retries). Returns the task id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task instruction/prompt to execute."
                            },
                            "name": {
                                "type": "string",
                                "description": "Optional short label for the task."
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority: higher runs first (default 0)."
                            },
                            "schedule_at": {
                                "type": "number",
                                "description": "Optional epoch-seconds to run it at (default: now)."
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "task_list",
                    "description": "Lists tasks in the autonomous task queue (optionally filtered by status: pending/running/done/failed/cancelled).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "Optional status filter."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max tasks to return (default 20)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "task_stats",
                    "description": "Returns the autonomous task queue status counts (pending/running/done/failed/cancelled).",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "task_cancel",
                    "description": "Cancels a pending task in the autonomous queue.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "integer",
                                "description": "The task id to cancel."
                            }
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subagent_delegate",
                    "description": "DEDICATED SUBAGENT: runs one sub-task with a named specialist (fresh session/checkpoint) and returns its final answer. Roles: planner, researcher, coder, reviewer, tester, security, test_writer, summarizer, memory_keeper, cost_watcher, triager, doc_writer, changelogger, deployer, dependency_updater, router, generalist. Use to decompose a big task into isolated units of work with the right specialist per unit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The sub-task to delegate."
                            },
                            "role": {
                                "type": "string",
                                "description": "Specialist role: planner | researcher | coder | reviewer | tester | security | test_writer | summarizer | memory_keeper | cost_watcher | triager | doc_writer | changelogger | deployer | dependency_updater | router | generalist (default generalist)."
                            },
                            "label": {
                                "type": "string",
                                "description": "Short label for the subagent (defaults to the role)."
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subagent_team",
                    "description": "DEDICATED SUBAGENT TEAM: runs several sub-tasks in parallel, each with its own specialist role (roles list parallel to tasks; missing roles default to generalist). Returns all results together. Use to fan out independent work items.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of sub-tasks to run in parallel."
                            },
                            "roles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of specialist roles, one per task (planner/researcher/coder/reviewer/tester/security/test_writer/summarizer/memory_keeper/cost_watcher/triager/doc_writer/changelogger/deployer/dependency_updater/router/generalist)."
                            }
                        },
                        "required": ["tasks"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subagent_roles",
                    "description": "Lists the available dedicated subagent roles with a description of when to use each. Call before delegating to pick the right specialist.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "subagent_route",
                    "description": "INTENT ROUTER: decides which specialist role(s) should handle an incoming task (primary + supporting roles + why). Deterministic keyword routing — no model call. Use before delegating a big request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The request to route to a specialist role."
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "orchestrator_run",
                    "description": "META-ORCHESTRATOR (Genesis Level 1): Executes a high-level goal through the hierarchical organization (Chief Agent -> Department Leads -> Worker Specialists). Arbitrates conflicts and provides executive synthesis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "The high-level project goal or complex task to orchestrate."
                            },
                            "departments": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of departments to involve: engineering, research, operations, quality_security. Defaults to automatic routing."
                            }
                        },
                        "required": ["goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "team_delegate",
                    "description": "DEPARTMENT DELEGATE (Genesis Level 2): Directly delegates a task to one of the 4 Department Leads (engineering, research, operations, quality_security). The lead assigns workers and applies first-line quality verification.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "department": {
                                "type": "string",
                                "description": "The department to delegate to: engineering, research, operations, or quality_security."
                            },
                            "task": {
                                "type": "string",
                                "description": "The task for the department to execute."
                            },
                            "role": {
                                "type": "string",
                                "description": "Optional preferred specialist worker within the department."
                            }
                        },
                        "required": ["department", "task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "team_status",
                    "description": "Reports status, budget usage, and managed specialists across all 4 Department Leads and the Meta-Orchestrator.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "dag_plan_and_run",
                    "description": "TASK GRAPH (Genesis Level 5): Decomposes a complex goal into a Directed Acyclic Graph (DAG) and executes independent nodes in parallel waves. Supports selective replanning on failures.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "The complex multi-step goal to plan as a DAG and execute."
                            }
                        },
                        "required": ["goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "dag_visualize",
                    "description": "Generates a visual Mermaid diagram and node dependency summary for a planned task graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "The goal to generate a DAG diagram for."
                            }
                        },
                        "required": ["goal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "debate_solve",
                    "description": "MULTI-AGENT DEBATE (Genesis Level 4): Pits an Advocate against a Skeptic across multiple rounds on complex architectural or technical questions, with an authoritative Judge rendering the balanced consensus verdict.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The complex decision, architecture question, or trade-off to debate."
                            },
                            "rounds": {
                                "type": "integer",
                                "description": "Number of debate rounds (default 2)."
                            }
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reflexion_solve",
                    "description": "REFLEXION LOOP (Genesis Level 4): Solves a task with autonomous self-critique and iterative refinement up to 3 cycles, catching errors and improving before final response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task or problem to solve using self-critique and iterative refinement."
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_query",
                    "description": "KNOWLEDGE GRAPH (Genesis Level 3): Queries the causal and dependency knowledge graph around an entity up to N hops, returning related classes, functions, files, modules, and dependencies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "The entity identifier (e.g. file path, class name, or function name)."
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Graph traversal depth in hops (default 2)."
                            }
                        },
                        "required": ["entity_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_impact_analysis",
                    "description": "KNOWLEDGE GRAPH IMPACT (Genesis Level 3): Computes the blast radius and downstream dependencies that will be impacted if a given function, class, or file is modified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "The entity identifier to compute impact/blast radius for."
                            }
                        },
                        "required": ["entity_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_add_fact",
                    "description": "KNOWLEDGE GRAPH (Genesis Level 3): Adds a custom semantic fact or causal dependency between two entities in the knowledge graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "The source entity identifier."
                            },
                            "relation": {
                                "type": "string",
                                "description": "The relationship type (e.g., 'calls', 'depends_on', 'modifies', 'inherits')."
                            },
                            "target": {
                                "type": "string",
                                "description": "The target entity identifier."
                            }
                        },
                        "required": ["source", "relation", "target"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kg_index_workspace",
                    "description": "KNOWLEDGE GRAPH (Genesis Level 3): Scans Python ASTs in the workspace to construct an automated dependency and inheritance knowledge graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_files": {
                                "type": "integer",
                                "description": "Maximum number of Python files to scan (default 50)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_discover",
                    "description": "DYNAMIC TOOLS (Genesis Level 8): Searches and activates domain-specific tools on demand by keyword or category, keeping system prompt context lean.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search term, action, or tool name to look for (e.g. 'docker', 'browser', 'git', 'knowledge graph')."
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category filter: git, web_browser, genesis_orchestrator, reasoning, knowledge_graph, vector_rag, desktop_os, sandbox_verify."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of tools to return (default 8)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_reliability_report",
                    "description": "TOOL RELIABILITY (Genesis Level 8): Reports Bayesian/EWMA health scores, failure rates, and auto-mitigation recommendations for tools.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "model_route",
                    "description": "MODEL ROUTER (Genesis Level 7): Analyzes task complexity and determines the optimal LLM tier (FAST_CHEAP, STANDARD_CODING, DEEP_REASONING) with pricing estimates and failure escalation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task or prompt to analyze and route."
                            },
                            "prior_failures": {
                                "type": "integer",
                                "description": "Number of previous failures on this task (triggers escalation to higher reasoning tiers)."
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "model_budget_status",
                    "description": "COGNITIVE BUDGET (Genesis Level 7): Inspects cumulative token spend, model-by-model usage breakdown, and remaining USD budget.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox_execute",
                    "description": "EXECUTION SANDBOX (Genesis Level 6): Executes Python or shell code inside an isolated environment with filesystem snapshotting, static security AST scanning, timeout limits, and optional automatic rollback on failure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python or shell code to safely execute."
                            },
                            "language": {
                                "type": "string",
                                "description": "Language of the code: 'python' or 'shell' (default 'python')."
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Maximum execution time in seconds (default 30.0, max 120.0)."
                            },
                            "rollback_on_failure": {
                                "type": "boolean",
                                "description": "Whether to automatically rollback filesystem state to pre-execution snapshot if execution fails (default true)."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox_snapshot_create",
                    "description": "EXECUTION SANDBOX (Genesis Level 6): Takes an immediate point-in-time filesystem snapshot of the workspace with SHA-256 integrity hashes for safe rollback.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name / label for the snapshot (e.g. 'pre_refactor_migration')."
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sandbox_snapshot_rollback",
                    "description": "EXECUTION SANDBOX (Genesis Level 6): Restores workspace files to a previously saved snapshot, reverting modifications, additions, and deletions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the snapshot to restore."
                            }
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "drift_record_task",
                    "description": "DRIFT MONITORING (Genesis Level 9): Logs task execution telemetry (success, steps, latency, tokens, failed tools) for continuous quality regression tracking.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Unique identifier for the task or session."
                            },
                            "success": {
                                "type": "boolean",
                                "description": "Whether the task succeeded or failed."
                            },
                            "steps": {
                                "type": "integer",
                                "description": "Total steps / tool actions executed (default 1)."
                            },
                            "duration_sec": {
                                "type": "number",
                                "description": "Elapsed duration in seconds."
                            },
                            "tokens_used": {
                                "type": "integer",
                                "description": "Approximate token spend for this task."
                            },
                            "failed_tools": {
                                "type": "string",
                                "description": "Comma-separated list of tool names that failed during execution."
                            },
                            "category": {
                                "type": "string",
                                "description": "Task domain / category (e.g. coding, git, web, reasoning)."
                            }
                        },
                        "required": ["task_id", "success"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "drift_check",
                    "description": "DRIFT MONITORING (Genesis Level 9): Analyzes historical task telemetry for quality regression, success-rate drop, step inflation, and tool failure clusters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "window_size": {
                                "type": "integer",
                                "description": "Number of recent tasks to evaluate against baseline (default 10)."
                            },
                            "threshold_drop": {
                                "type": "number",
                                "description": "Success rate drop threshold for warning alert (default 0.20 for 20% drop)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "drift_status",
                    "description": "DRIFT MONITORING (Genesis Level 9): Displays longitudinal telemetry summary, total recorded tasks, and health status.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_improve_analyze_failure",
                    "description": "SELF-IMPROVEMENT (Genesis Level 10): Analyzes a failed task log, diagnoses root cause, and extracts a prescriptive operational rule/lesson.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Identifier of the failed task."
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The original task prompt."
                            },
                            "failure_log": {
                                "type": "string",
                                "description": "Execution traceback, error messages, or failure explanation."
                            },
                            "failed_tools": {
                                "type": "string",
                                "description": "Comma-separated names of tools that failed."
                            },
                            "category": {
                                "type": "string",
                                "description": "Task domain (e.g. coding, git, reasoning, general)."
                            }
                        },
                        "required": ["task_id", "prompt", "failure_log"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_improve_eval_run",
                    "description": "EVALUATION: Runs registered evaluation cases only when an explicit evaluation runner is configured. Without a runner, reports NOT RUN and never invents pass rates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Optional category filter: coding, reasoning, security, git (or leave empty for all)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "self_improve_crystallize_lesson",
                    "description": "SELF-IMPROVEMENT (Genesis Level 10): Permanently records an extracted lesson into the Skill playbook library and Knowledge Graph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lesson_title": {
                                "type": "string",
                                "description": "Summary title of the learned rule / playbook."
                            },
                            "guidance": {
                                "type": "string",
                                "description": "Prescriptive workflow instructions and best practices to prevent future failure."
                            },
                            "category": {
                                "type": "string",
                                "description": "Domain category (e.g. coding, git, docker, general)."
                            }
                        },
                        "required": ["lesson_title", "guidance"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "docker_sandbox_run",
                    "description": "Runs code or shell commands in a hardened disposable Docker container: no network by default, read-only container root, dropped capabilities, no-new-privileges, process/CPU/memory limits. Mounting the project workspace is opt-in.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to run inside the container (e.g. 'python -c \"print(1+1)\"' or 'sh -c \"ls -la\"')."
                            },
                            "image": {
                                "type": "string",
                                "description": "Locally available Docker image (no automatic pulls; default: 'titan-agent-sandbox:local'). Build or pull it on the host before use."
                            },
                            "memory_limit": {
                                "type": "string",
                                "description": "Memory limit from 64m to 4g (e.g. '256m', '512m', '1g'). Default: '512m'."
                            },
                            "cpu_quota": {
                                "type": "string",
                                "description": "CPU quota from 0.1 to 4.0 CPUs (e.g. '0.5', '1.0'). Default: '1.0'."
                            },
                            "mount_workspace": {
                                "type": "boolean",
                                "description": "Whether to mount the host workspace directory to /workspace inside the container. Default: false."
                            },
                            "network": {
                                "type": "string",
                                "description": "Container network mode: 'none' (default) or explicitly enabled 'bridge'."
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Execution timeout in seconds. Default: 60.0."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Applies a unified diff patch to one or more files in the workspace with automatic hunk matching and safe rollback on error.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {
                                "type": "string",
                                "description": "The unified diff patch string (containing '---', '+++', and '@@' hunk headers)."
                            }
                        },
                        "required": ["patch"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "skill_save",
                    "description": "Creates or updates a persistent skill playbook in the skills library. Auto-loaded in future sessions matching the keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name / slug (e.g. 'docker-deploy', 'api-refactor')."
                            },
                            "description": {
                                "type": "string",
                                "description": "One-line description of the skill."
                            },
                            "keywords": {
                                "type": "string",
                                "description": "Comma-separated keywords for automatic injection matching."
                            },
                            "guidance": {
                                "type": "string",
                                "description": "Full markdown guidance body containing workflow steps and best practices."
                            }
                        },
                        "required": ["name", "guidance"]
                    }
                }
            }
            ,
            {
                "type": "function",
                "function": {
                    "name": "ast_patch_file",
                    "description": "Applies a targeted text replacement to a file and validates the resulting Python AST to ensure no syntax errors were introduced. Safer than normal editing for Python files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to modify."},
                            "search_block": {"type": "string", "description": "The exact block of code to search for."},
                            "replace_block": {"type": "string", "description": "The new block of code to replace it with."}
                        },
                        "required": ["path", "search_block", "replace_block"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ast_replace_function",
                    "description": "Surgically replaces a complete Python function definition by name using AST analysis. Ensures no indentation or syntax errors are introduced.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the Python file."},
                            "function_name": {"type": "string", "description": "Name of the function to replace."},
                            "new_function_code": {"type": "string", "description": "Complete new function code (including def line and docstring)."}
                        },
                        "required": ["path", "function_name", "new_function_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ast_replace_class",
                    "description": "Surgically replaces a complete Python class definition by name using AST analysis. Ensures clean class boundary replacement without syntax errors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the Python file."},
                            "class_name": {"type": "string", "description": "Name of the class to replace."},
                            "new_class_code": {"type": "string", "description": "Complete new class code (including class line and methods)."}
                        },
                        "required": ["path", "class_name", "new_class_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "deep_verify_code",
                    "description": "Runs a Deep Verification Loop on target code: generates tests, runs them in the sandbox, and heals the code if they fail.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "The raw code to verify."},
                            "intent": {"type": "string", "description": "What the code is supposed to do."}
                        },
                        "required": ["code", "intent"]
                    }
                }
            }
        ]
        base_defs.extend([
            {
                "type": "function",
                "function": {
                    "name": "vector_rag_index",
                    "description": "Indexes the entire workspace into the Semantic Vector Database (ChromaDB) for advanced RAG.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_rag_search",
                    "description": "Searches the Semantic Vector Database for deeply relevant code snippets and context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query (natural language)."},
                            "top_k": {"type": "integer", "description": "Number of results to return (default 5)."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "sast_scan",
                    "description": "Performs Static Application Security Testing (SAST) & OWASP Top 10 code audit on source code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "The source code string to audit."},
                            "filename": {"type": "string", "description": "Optional filename (e.g. app.py, handler.js)."}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "dependency_audit",
                    "description": "Audits dependencies and package manifests (requirements.txt, package.json) for known vulnerabilities (CVEs).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "manifest_file": {"type": "string", "description": "Path to dependency manifest (default: requirements.txt)."}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "secret_scan",
                    "description": "Scans text, code, or logs for hardcoded API keys, tokens, and high-entropy credentials.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text or code to scan for secrets."}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "synthesize_tool",
                    "description": "AUTONOMOUS TOOL SYNTHESIS: Dynamically creates, verifies in an isolated sandbox, compiles, and registers a brand-new Python tool on-the-fly during the live session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Identifier name for the new tool (e.g. 'parse_pcap_header', 'calc_keccak256')."},
                            "description": {"type": "string", "description": "Clear description of what the tool does and how to use it."},
                            "python_code": {"type": "string", "description": "Self-contained Python code implementing the tool function. Can be async or sync."},
                            "test_code": {"type": "string", "description": "Python test script verifying that the tool behaves correctly in the sandbox."},
                            "parameters": {"type": "object", "description": "JSON Schema properties dict describing the tool's input arguments."}
                        },
                        "required": ["name", "description", "python_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "symbolic_check_code",
                    "description": "SYMBOLIC AST INVARIANT CHECKER: Deep static analysis verifying code safety, infinite loops, command injections, and resource leaks before runtime.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python source code string to statically verify."}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "tdd_cycle",
                    "description": "AUTONOMOUS TDD ENGINE: Executes a rigorous Red-Green-Refactor software cycle. Proves the test fails first (RED), writes the implementation to pass it (GREEN), and verifies symbolic invariants (REFACTOR).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_code": {"type": "string", "description": "Pytest test code asserting the expected feature or bugfix."},
                            "implementation_code": {"type": "string", "description": "Python source code implementing the requested feature."},
                            "test_filename": {"type": "string", "description": "Filename for the test (default: test_feature.py)."},
                            "code_filename": {"type": "string", "description": "Filename for the code module (default: feature.py)."}
                        },
                        "required": ["test_code", "implementation_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "consensus_deliberation",
                    "description": "MULTI-AGENT CONSENSUS: Convenes an architectural committee (Architect, Security Officer, Pragmatist) to formally evaluate and vote on critical code or design proposals.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "proposal": {"type": "string", "description": "The architectural, refactoring, or design proposal to deliberate upon."},
                            "context": {"type": "string", "description": "Current system context, constraints, and dependencies."}
                        },
                        "required": ["proposal"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "working_memory_update",
                    "description": "ACTIVE WORKING MEMORY: Pins confirmed operational facts, marks refuted dead-ends to avoid repeating, or updates current subtask.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confirmed_fact": {"type": "string", "description": "A verified fact to remember across the run."},
                            "dead_end": {"type": "string", "description": "A failed approach or path to avoid repeating."},
                            "subtask": {"type": "string", "description": "The active sub-problem currently being solved."},
                            "todo": {"type": "string", "description": "A pending todo item to track."}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_connect_preset",
                    "description": "MCP INTEGRATION: Connects a preset (postgres, github, slack, brave_search, filesystem, sqlite, puppeteer, gdrive). Never pass credential literals; store secrets in process environment and pass {ENV_VAR} references.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "preset_id": {"type": "string", "description": "Preset identifier: postgres, github, slack, brave_search, filesystem, sqlite, puppeteer, gdrive."},
                            "server_name": {"type": "string", "description": "Optional custom name for the connection (default: same as preset_id)."},
                            "env_overrides": {"type": "object", "description": "Optional field-to-environment-reference mapping; never include secret values. Example: {\"POSTGRES_URL\": \"{POSTGRES_URL}\"}. The referenced variables must already exist in the process environment."}
                        },
                        "required": ["preset_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_list_presets",
                    "description": "MCP CATALOG: Lists all available 1-line MCP server presets with setup instructions and environment requirements.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "hitl_request_approval",
                    "description": "HUMAN-IN-THE-LOOP: Requests explicit human approval before executing destructive actions (e.g., file deletes, force pushes, secret edits). Prompts user: 'I am about to execute this action. Do you authorize this? [Yes / No]'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "Action name (e.g. 'delete_file', 'git_push_force', 'edit_config')."},
                            "resource": {"type": "string", "description": "Target file or resource affected."},
                            "reason": {"type": "string", "description": "Justification explaining why this action is required."}
                        },
                        "required": ["action", "resource"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_create_branch",
                    "description": "AUTONOMOUS GITOPS: Creates and checks out an isolated feature branch (e.g. agent/feature-login-auth) before writing changes, protecting main branch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "Brief description of the task used to generate branch slug."},
                            "prefix": {"type": "string", "description": "Branch prefix (default: 'agent/feature-')."}
                        },
                        "required": ["task_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "git_create_pr",
                    "description": "AUTOMATED PULL REQUEST: Pushes the active feature branch to remote origin and opens a Pull Request on GitHub with automated verification evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "PR title describing the feature or bugfix."},
                            "body": {"type": "string", "description": "PR body in markdown with changelog and test results."},
                            "base_branch": {"type": "string", "description": "Target base branch (default: main)."}
                        },
                        "required": ["title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_cache_query",
                    "description": "SEMANTIC CACHE: Checks local SQLite semantic cache to retrieve answers for previously answered queries or code analyses at 0ms and zero token cost.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The prompt or question to look up in cache."},
                            "threshold": {"type": "number", "description": "Similarity threshold between 0.0 and 1.0 (default 0.90)."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_cache_stats",
                    "description": "SEMANTIC CACHE METRICS: Returns total cache entries, hits, hit ratio, and token/dollar cost savings.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "experience_replay_query",
                    "description": "EPISODIC MEMORY: Searches past error resolution experience replay database for proven fixes to errors, tracebacks, or version collisions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "error_text": {"type": "string", "description": "The error message or traceback snippet encountered."}
                        },
                        "required": ["error_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "experience_replay_record",
                    "description": "EPISODIC MEMORY RECORD: Stores a newly solved error and its verified resolution recipe so the agent remembers the solution forever.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "error_text": {"type": "string", "description": "The original error or failure message."},
                            "resolution": {"type": "string", "description": "The exact patch, fix command, or recipe that resolved the problem."},
                            "diagnosis": {"type": "string", "description": "Root cause explanation."}
                        },
                        "required": ["error_text", "resolution"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_probe",
                    "description": "MULTIMEDIA: Inspects video or audio file metadata (duration, resolution, fps, video/audio codecs, file size) via ffprobe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Absolute or relative path to the media file."}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "video_montage_command",
                    "description": "MULTIMEDIA: Generates optimized FFmpeg commands for video editing operations: trim, crop_vertical (9:16 for Reels/Shorts/TikTok), merge_audio, or speed adjustment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "Editing operation to perform.",
                                "enum": ["trim", "crop_vertical", "merge_audio", "speed"]
                            },
                            "input_video": {"type": "string", "description": "Input video file path."},
                            "output_video": {"type": "string", "description": "Target output video file path."},
                            "start_time": {"type": "string", "description": "Optional start timestamp (e.g. '00:01:30' or '10')."},
                            "duration": {"type": "string", "description": "Optional duration (e.g. '00:00:15' or '15')."},
                            "audio_track": {"type": "string", "description": "Audio file path for merge_audio operation."},
                            "speed": {"type": "number", "description": "Playback speed multiplier (e.g. 1.5, 2.0, 0.5)."}
                        },
                        "required": ["operation", "input_video", "output_video"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "blender_generate_scene",
                    "description": "3D GRAPHICS & RENDERING: Generates a complete standalone headless Python script (bpy) to build 3D geometry, configure PBR materials, set 3-point studio lighting, and render high-resolution images via Blender.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "primitive": {
                                "type": "string",
                                "description": "Base procedural mesh primitive.",
                                "enum": ["cube", "sphere", "cylinder", "torus"]
                            },
                            "output_image": {"type": "string", "description": "Output rendered image file path (e.g. 'render.png')."},
                            "engine": {
                                "type": "string",
                                "description": "Render engine: 'BLENDER_EEVEE' or 'CYCLES'.",
                                "enum": ["BLENDER_EEVEE", "CYCLES"]
                            },
                            "save_path": {"type": "string", "description": "Optional file path to save the generated Python bpy script."}
                        },
                        "required": ["primitive", "output_image"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "blender_execute_script",
                    "description": "3D GRAPHICS & RENDERING: Executes a Python bpy script in Blender in headless background mode (blender --background --python <script>). Automatically discovers Blender installation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_path": {"type": "string", "description": "Path to the Blender Python (.py) script to execute."}
                        },
                        "required": ["script_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "domain_list",
                    "description": "OMNI-DOMAIN: Lists all registered industry domain profiles (Finance, Healthcare, Legal, Software, Marketing, Science, etc.) and highlights the currently active one.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "domain_switch",
                    "description": "OMNI-DOMAIN: Switches the agent's active operational domain profile on-the-fly (e.g. 'finance', 'healthcare', 'legal', 'software_engineering', 'universal').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "Target domain identifier (e.g. 'finance', 'healthcare', 'legal', 'software_engineering', 'science', 'marketing', 'universal')."
                            }
                        },
                        "required": ["domain"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "domain_get_active",
                    "description": "OMNI-DOMAIN: Returns full operational guidelines, instructions, guardrails, and tool configurations for the active domain profile.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "domain_create",
                    "description": "OMNI-DOMAIN: Creates and permanently persists a custom domain profile tailored for any specific enterprise, workflow, or industry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique domain identifier (e.g. 'real_estate', 'aviation', 'biotech')."},
                            "display_name": {"type": "string", "description": "Human-readable title (e.g. 'Real Estate & Property Management')."},
                            "description": {"type": "string", "description": "Overview of domain responsibilities and scope."},
                            "system_prompt_overlay": {"type": "string", "description": "Specific domain methodology, cognitive rules, standards, and guidelines."},
                            "icon": {"type": "string", "description": "Optional single emoji or icon (default: 🌐)."},
                            "mandatory_guardrails": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of domain compliance rules and required disclaimers."
                            },
                            "preferred_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of prioritized tool names."
                            },
                            "forbidden_tools": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of restricted tool names."
                            }
                        },
                        "required": ["name", "display_name", "description", "system_prompt_overlay"]
                    }
                }
            }
        ])
        from .browser_automation import BrowserAutomation
        browser_defs = BrowserAutomation(self.workspace).get_tool_definitions()
        synthesized_defs = list(getattr(self, "_synthesized_definitions", {}).values())
        return base_defs + browser_defs + synthesized_defs

    @property
    def vector_rag(self):
        if getattr(self, "_vector_rag", None) is None:
            from .vector_rag import VectorRAG
            self._vector_rag = VectorRAG(self.workspace)
        return self._vector_rag

    def tool_vector_rag_index(self) -> str:
        return self.vector_rag.index_workspace()

    def tool_vector_rag_search(self, query: str, top_k: int = 5) -> str:
        return self.vector_rag.search(query, top_k)

    @property
    def browser_automation(self):
        if getattr(self, "_browser_automation", None) is None:
            from .browser_automation import BrowserAutomation
            self._browser_automation = BrowserAutomation(self.workspace)
        return self._browser_automation

    async def tool_browser_goto(self, url: str) -> str:
        return await self.browser_automation.tool_browser_goto(url)

    async def tool_browser_click(self, selector: str) -> str:
        return await self.browser_automation.tool_browser_click(selector)

    async def tool_browser_type(self, selector: str, text: str) -> str:
        return await self.browser_automation.tool_browser_type(selector, text)

    async def tool_browser_screenshot(self, filename: str = "screenshot.png") -> str:
        return await self.browser_automation.tool_browser_screenshot(filename)

    async def tool_browser_extract_text(self) -> str:
        return await self.browser_automation.tool_browser_extract_text()

    async def tool_browser_close(self) -> str:
        return await self.browser_automation.tool_browser_close()

    def tool_sast_scan(self, code: str, filename: str = "code.py") -> str:
        """Run SAST & OWASP Top 10 code audit."""
        from titan_agent.core.security.sast import SASTScanner
        report = SASTScanner().scan_code(code, filename=filename)
        if not report.findings:
            return "✅ SAST Scan Clean: No OWASP security issues detected."
        
        lines = [f"🛡️ **SAST & OWASP Security Report ({report.total_findings} findings)**:"]
        for f in report.findings:
            lines.append(f"- **[{f.severity}]** `{f.rule_id}` (line {f.line}): {f.title}")
            lines.append(f"  *Snippet:* `{f.snippet}`")
            lines.append(f"  *Fix:* {f.remediation}")
        return "\n".join(lines)

    def tool_dependency_audit(self, manifest_file: str = "requirements.txt") -> str:
        """Audit package manifests for vulnerabilities."""
        from titan_agent.core.security.dependency_auditor import DependencyAuditor
        target = self._resolve_path(manifest_file)
        if not target.is_file():
            # Check default workspace root
            target = self.workspace / manifest_file
        report = DependencyAuditor().audit_manifest_file(target)
        if not report.alerts:
            return f"✅ Dependency Audit Clean: {report.total_dependencies} dependencies analyzed, 0 known CVEs found."

        lines = [f"🛡️ **Supply-Chain Dependency Audit ({report.vulnerable_count} alerts)**:"]
        for a in report.alerts:
            lines.append(f"- **[{a.severity}]** `{a.package}` ({a.installed_spec})")
            lines.append(f"  *Advisory:* {a.advisory}")
            lines.append(f"  *Fix:* {a.recommendation}")
        return "\n".join(lines)

    def tool_secret_scan(self, text: str) -> str:
        """Scan text or code for high-entropy secrets and sensitive tokens."""
        from titan_agent.core.security.secret_scanner import SecretScanner
        scanner = SecretScanner()
        findings = scanner.scan(text)
        if not findings:
            return "✅ Secret Scan Clean: No exposed credentials or private keys detected."

        lines = [f"🚨 **Secret Scanner Alert ({len(findings)} secrets detected)**:"]
        for f in findings:
            masked = f.match[:4] + "..." if len(f.match) > 8 else "***"
            lines.append(f"- **[{f.severity}]** {f.secret_type}: `{masked}` (Entropy: {f.entropy})")
        lines.append(f"\n🔒 **Sanitized Redaction Preview:**\n```\n{scanner.redact(text)}\n```")
        return "\n".join(lines)

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            handler = getattr(self, f"tool_{name}", None)
            if not handler:
                return f"Error: Tool '{name}' does not exist."
            if asyncio.iscoroutinefunction(handler):
                return await handler(**args)
            else:
                return handler(**args)
        except (RuntimeError, OSError, ValueError) as e:
            return f"Tool execution failed for '{name}': {e!s}"

    async def tool_ast_patch_file(self, path: str, search_block: str, replace_block: str) -> str:
        from titan_agent.core.code_intel.ast_patcher import ASTPatcher, ASTPatchError
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File {target} not found."
        original = target.read_text(encoding="utf-8")
        try:
            new_code = ASTPatcher.apply_replacement(original, search_block, replace_block)
            target.write_text(new_code, encoding="utf-8")
            return f"AST Patch applied successfully to {target}."
        except ASTPatchError as e:
            return f"Error applying AST patch: {e}"

    async def tool_ast_replace_function(self, path: str, function_name: str, new_function_code: str) -> str:
        """Surgically replace a Python function definition using AST boundary detection."""
        from titan_agent.core.code_intel.ast_patcher import ASTPatcher, ASTPatchError
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File {target} not found."
        original = target.read_text(encoding="utf-8")
        try:
            new_code = ASTPatcher.replace_function(original, function_name, new_function_code)
            target.write_text(new_code, encoding="utf-8")
            return f"Function '{function_name}' in {target} successfully updated via AST."
        except ASTPatchError as e:
            return f"Error replacing function: {e}"

    async def tool_ast_replace_class(self, path: str, class_name: str, new_class_code: str) -> str:
        """Surgically replace a Python class definition using AST boundary detection."""
        from titan_agent.core.code_intel.ast_patcher import ASTPatcher, ASTPatchError
        target = self._resolve_path(path)
        if not target.exists():
            return f"Error: File {target} not found."
        original = target.read_text(encoding="utf-8")
        try:
            new_code = ASTPatcher.replace_class(original, class_name, new_class_code)
            target.write_text(new_code, encoding="utf-8")
            return f"Class '{class_name}' in {target} successfully updated via AST."
        except ASTPatchError as e:
            return f"Error replacing class: {e}"

    async def tool_deep_verify_code(self, code: str, intent: str) -> str:
        from titan_agent.core.verification.deep_verifier import DeepVerifier
        from titan_agent.llm_client import LLMClient
        llm = LLMClient()
        
        async def _docker_runner(script: str) -> str:
            raw = await self.tool_docker_sandbox_run(
                command=script,
                image=os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local"),
                timeout=60.0,
                network="none",
            )
            header = re.search(r"\(Exit (-?\d+)\)", raw)
            if not header:
                return json.dumps({"exit_code": -1, "stdout": "", "stderr": raw})
            exit_code = int(header.group(1))
            body = raw[header.end():].lstrip("\n")
            stdout = body.split("STDOUT:\n", 1)[1].split("\nSTDERR:\n", 1)[0] if "STDOUT:\n" in body else ""
            stderr = body.split("\nSTDERR:\n", 1)[1] if "\nSTDERR:\n" in body else ""
            return json.dumps({"exit_code": exit_code, "stdout": stdout, "stderr": stderr})

        verifier = DeepVerifier(llm, sandbox_runner=_docker_runner)
        result = await verifier.self_heal_loop(code, intent, max_iterations=3)
        if result["verified"]:
            return f"Deep Verify SUCCEEDED! Verified Code:\n{result['code']}\n\nFinal Output:\n{result['final_output']}"
        else:
            return f"Deep Verify FAILED after {result['iterations']} iterations.\nLast Output:\n{result['final_output']}\nLast Code:\n{result['code']}"

    async def _run_dynamic_tool_container(
        self,
        name: str,
        python_code: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> str:
        """Invoke a generated tool in a fresh, network-disabled Docker container."""
        import base64
        import uuid

        try:
            arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "Error: dynamic tool arguments must be JSON-serializable."
        if len(arguments_json.encode("utf-8")) > 64 * 1024:
            return "Error: dynamic tool arguments exceed the 64 KiB sandbox limit."
        if len(python_code.encode("utf-8")) > 128 * 1024:
            return "Error: generated tool source exceeds the 128 KiB sandbox limit."

        marker = f"__TITAN_DYNAMIC_RESULT_{uuid.uuid4().hex}__:"
        runner_script = "\n".join((
            "import asyncio, base64, inspect, json, os, sys",
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
            "import synthesized_module",
            f"tool_name = {name!r}",
            "with open('/workspace/arguments.json', encoding='utf-8') as f: arguments = json.load(f)",
            "try:",
            "    fn = getattr(synthesized_module, tool_name, None) or getattr(synthesized_module, 'tool_' + tool_name, None)",
            "    if fn is None: raise LookupError('declared entry point missing')",
            "    value = fn(**arguments)",
            "    if inspect.isawaitable(value): value = asyncio.run(value)",
            "    payload = {'ok': True, 'value': value}",
            "except BaseException as exc:",
            "    payload = {'ok': False, 'error': type(exc).__name__}",
            f"print({marker!r} + base64.b64encode(json.dumps(payload, ensure_ascii=False, default=str).encode()).decode())",
            "if not payload['ok']: raise SystemExit(1)",
            "",
        ))

        with tempfile.TemporaryDirectory(prefix="titan_dynamic_call_") as tmpdir:
            root = Path(tmpdir)
            (root / "synthesized_module.py").write_text(python_code, encoding="utf-8")
            (root / "arguments.json").write_text(arguments_json, encoding="utf-8")
            (root / "run_dynamic_tool.py").write_text(runner_script, encoding="utf-8")
            raw = await self.tool_docker_sandbox_run(
                command="python -I -s /workspace/run_dynamic_tool.py",
                image=os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local"),
                timeout=max(0.5, min(float(timeout), 30.0)),
                network="none",
                _mount_path=root,
            )
        match = re.search(r"\(Exit (-?\d+)\)", raw)
        if not match or int(match.group(1)) != 0:
            return "Error: isolated dynamic tool failed or was truncated; result not accepted."
        encoded = None
        for line in raw.splitlines():
            if marker in line:
                encoded = line.split(marker, 1)[1].strip()
        if not encoded:
            return "Error: isolated dynamic tool returned no valid result."
        try:
            payload = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, TypeError, json.JSONDecodeError):
            return "Error: isolated dynamic tool returned malformed output."
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            error_type = payload.get("error", "RuntimeError") if isinstance(payload, dict) else "RuntimeError"
            return f"Error: isolated dynamic tool raised {error_type}."
        value = payload.get("value")
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)

    async def tool_synthesize_tool(
        self,
        name: str,
        description: str,
        python_code: str,
        test_code: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Synthesize a tool whose verification and invocation run in Docker."""
        if os.getenv("TITAN_DYNAMIC_TOOLS_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            return (
                "Error: dynamic Python tool synthesis is disabled. Set "
                "TITAN_DYNAMIC_TOOLS_ENABLED=true only in a trusted environment; "
                "Docker is required for both verification and each invocation."
            )
        from titan_agent.core.synthesis.dynamic_tool_synthesizer import DynamicToolSynthesizer
        synthesizer = getattr(self, "_tool_synthesizer", None)
        if synthesizer is None:
            async def _dynamic_sandbox_runner(temp_workspace: Path, timeout: float):
                raw = await self.tool_docker_sandbox_run(
                    command="python -E -s /workspace/test_synthesized.py",
                    image=os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local"),
                    timeout=timeout,
                    network="none",
                    _mount_path=temp_workspace,
                )
                match = re.search(r"\(Exit (-?\d+)\)", raw)
                passed = bool(
                    match
                    and int(match.group(1)) == 0
                    and "___SYNTHESIS_TEST_PASSED___" in raw
                )
                return passed, raw

            async def _dynamic_runtime_runner(name, python_code, arguments, timeout):
                return await self._run_dynamic_tool_container(
                    name, python_code, arguments, timeout
                )

            synthesizer = DynamicToolSynthesizer(
                self.workspace,
                sandbox_runner=_dynamic_sandbox_runner,
                runtime_runner=_dynamic_runtime_runner,
            )
            self._tool_synthesizer = synthesizer

        params = parameters or {"type": "object", "properties": {}}
        success, message = await synthesizer.synthesize_and_register(
            name=name,
            description=description,
            parameters=params,
            python_code=python_code,
            test_code=test_code,
            registry=self,
        )
        return message

    def tool_symbolic_check_code(self, code: str) -> str:
        """Performs static AST symbolic invariant analysis on Python code before runtime."""
        from titan_agent.core.code_intel.symbolic_checker import SymbolicInvariantChecker
        report = SymbolicInvariantChecker.check_code(code)
        return report.summary()

    async def tool_tdd_cycle(
        self,
        test_code: str,
        implementation_code: str,
        test_filename: str = "test_feature.py",
        code_filename: str = "feature.py",
    ) -> str:
        """Executes a full autonomous Red-Green-Refactor software cycle in an isolated sandbox."""
        from titan_agent.core.code_intel.tdd_engine import AutonomousTDDEngine

        async def _sandbox_pytest(test_path: Path, temp_workspace: Path, timeout: float):
            started = time.monotonic()
            if _cfg.full_access_enabled():
                # Explicit trusted-mode bypass. This subprocess inherits host
                # privileges and is deliberately not described as a sandbox.
                spawn_options: dict[str, Any] = {}
                if os.name == "posix":
                    spawn_options["start_new_session"] = True
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pytest", str(test_path), "-v", "-s",
                    cwd=str(temp_workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_options,
                )
                stdout, stderr, timed_out, truncated = await _bounded_communicate(
                    proc, timeout, process_group=(os.name == "posix")
                )
                if timed_out:
                    return 124, "Trusted host test timed out.", time.monotonic() - started
                output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                if truncated:
                    output += f"\\n[Output truncated at {MAX_CAPTURE_BYTES_PER_STREAM} bytes per stream]"
                return (126 if truncated else proc.returncode or 0), output, time.monotonic() - started
            command = f"python -E -s -m pytest /workspace/{test_path.name} -v -s"
            raw = await self.tool_docker_sandbox_run(
                command=command,
                image=os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local"),
                timeout=timeout,
                network="none",
                _mount_path=temp_workspace,
            )
            match = re.search(r"\(Exit (-?\d+)\)", raw)
            return (int(match.group(1)) if match else 126, raw, time.monotonic() - started)

        engine = AutonomousTDDEngine(self.workspace, sandbox_runner=_sandbox_pytest)
        report = await engine.execute_tdd_cycle(
            test_code=test_code,
            implementation_code=implementation_code,
            test_filename=test_filename,
            code_filename=code_filename,
        )
        return report.format_text()

    def tool_consensus_deliberation(self, proposal: str, context: str = "") -> str:
        """Convenes an architectural committee to render formal consensus on critical changes."""
        from titan_agent.core.reasoning.consensus_engine import ConsensusEngine
        memo = ConsensusEngine.evaluate_heuristic(proposal, context)
        return memo.format_memo()

    def tool_working_memory_update(
        self,
        confirmed_fact: str = "",
        dead_end: str = "",
        subtask: str = "",
        todo: str = "",
    ) -> str:
        """Pins verified facts or marks dead-ends to avoid repeating in the live working memory."""
        agent_mem = getattr(self, "_working_memory_ref", None)
        if agent_mem is None:
            from titan_agent.core.memory.working_memory_virtualizer import WorkingMemoryVirtualizer
            agent_mem = WorkingMemoryVirtualizer()
            self._working_memory_ref = agent_mem

        if confirmed_fact:
            agent_mem.confirm_fact(confirmed_fact)
        if dead_end:
            agent_mem.record_dead_end(dead_end)
        if subtask:
            agent_mem.set_subtask(subtask)
        if todo:
            agent_mem.add_todo(todo)

        return agent_mem.render_hud_block()

    async def tool_mcp_connect_preset(
        self,
        preset_id: str,
        server_name: str = "",
        env_overrides: dict[str, str] | None = None,
    ) -> str:
        """Connects any ready MCP server preset in a single call."""
        from titan_agent.core.mcp.presets import MCPPresetManager
        name = server_name or preset_id
        if self.mcp_manager is not None:
            ok, message = await self.mcp_manager.connect_preset(
                preset_id=preset_id,
                server_name=name,
                env_overrides=env_overrides,
                workspace_dir=self.workspace,
            )
            return message if ok else f"Error: {message}"

        # A standalone ToolRegistry has no live manager to attach the process
        # to. Save only the preset template (never inline credential values) and
        # make the non-connected state explicit instead of claiming success.
        mgr = MCPPresetManager(self.workspace / "mcp_servers.json")
        cfg, err = mgr.generate_server_config(preset_id)
        if not cfg:
            return f"Error: {err}"
        if not mgr.save_server_to_config(name, cfg):
            return f"Error: failed to save MCP preset '{name}'."
        return (
            f"Successfully configured and saved MCP preset '{name}', but it is not connected "
            "because this ToolRegistry has no live MCP manager. Configure required environment "
            "variables and connect it through an agent session."
        )

    def tool_mcp_list_presets(self) -> str:
        """Lists available 1-line MCP presets."""
        from titan_agent.core.mcp.presets import MCPPresetManager
        mgr = MCPPresetManager()
        presets = mgr.list_presets()
        lines = ["### AVAILABLE 1-LINE MCP PRESETS:"]
        for p in presets:
            lines.append(f"- **{p['id']}** ({p['name']}): {p['description']}")
            if p["env_keys"]:
                lines.append(f"  *Required Env:* {', '.join(p['env_keys'])}")
        return "\n".join(lines)

    async def tool_hitl_request_approval(
        self,
        action: str,
        resource: str,
        reason: str = "",
    ) -> str:
        """Requests human confirmation for sensitive operations."""
        from titan_agent.core.guardrails.dangerous_actions import DangerousActionClassifier
        assessment = DangerousActionClassifier.assess_action(action, resource, {"reason": reason})
        prompt = assessment.suggested_prompt or f"I am about to execute ({action} on {resource}). Do you authorize this? [Yes / No]"
        if getattr(self, "hitl", None) is not None:
            gate = getattr(self, "_dangerous_gate", None)
            if not gate:
                from titan_agent.core.guardrails.dangerous_actions import DangerousActionGate
                gate = DangerousActionGate(self.hitl)
                self._dangerous_gate = gate
            ok, msg = await gate.check_and_gate(action, resource, {"reason": reason})
            return f"Approval Decision: {msg}"
        return f"Human Approval Prompt Generated:\n{prompt}\n(Waiting for user response [Ha / Yo'q])"

    def tool_git_create_branch(self, task_name: str, prefix: str = "agent/feature-") -> str:
        """Creates an isolated feature branch for the task."""
        from titan_agent.core.git.pr_engine import GitPREngine
        engine = GitPREngine(self.workspace)
        ok, msg = engine.create_feature_branch(task_name, prefix)
        return msg

    def tool_git_create_pr(
        self,
        title: str,
        body: str = "",
        base_branch: str = "main",
    ) -> str:
        """Runs pre-PR test check and opens Pull Request on GitHub."""
        from titan_agent.core.git.pr_engine import GitPREngine
        engine = GitPREngine(self.workspace)
        res = engine.create_pull_request(title, body, base_branch)
        return f"PR Status: {res.message}\nURL: {res.pr_url}"

    def tool_semantic_cache_query(self, query: str, threshold: float = 0.90) -> str:
        """Retrieves semantically identical past queries from local cache."""
        cache = getattr(self, "_semantic_cache", None)
        if not cache:
            from titan_agent.core.caching.semantic_cache import SemanticCache
            cache = SemanticCache(self.workspace / "semantic_cache.db")
            self._semantic_cache = cache
        cached_resp, sim = cache.get(query, threshold=threshold)
        if cached_resp:
            return f"### SEMANTIC CACHE HIT (Similarity: {sim * 100:.1f}%, 0ms, 0 tokens):\n{cached_resp}"
        return f"Semantic Cache Miss (Best match: {sim * 100:.1f}%, threshold: {threshold * 100:.0f}%)."

    def tool_semantic_cache_stats(self) -> str:
        """Returns statistics on token and dollar savings from semantic caching."""
        cache = getattr(self, "_semantic_cache", None)
        if not cache:
            from titan_agent.core.caching.semantic_cache import SemanticCache
            cache = SemanticCache(self.workspace / "semantic_cache.db")
            self._semantic_cache = cache
        return cache.get_stats().summary()

    def tool_experience_replay_query(self, error_text: str) -> str:
        """Searches experience replay memory for proven fixes to errors."""
        replay = getattr(self, "_experience_replay", None)
        if not replay:
            from titan_agent.core.memory.experience_replay import ExperienceReplayEngine
            replay = ExperienceReplayEngine(self.workspace / "experience_replay.db")
            self._experience_replay = replay
        match = replay.query_experience(error_text)
        if match:
            return match.format_hint()
        return "No prior experience found for this error pattern in episodic memory."

    def tool_experience_replay_record(
        self,
        error_text: str,
        resolution: str,
        diagnosis: str = "",
    ) -> str:
        """Stores a newly verified error fix into episodic memory."""
        replay = getattr(self, "_experience_replay", None)
        if not replay:
            from titan_agent.core.memory.experience_replay import ExperienceReplayEngine
            replay = ExperienceReplayEngine(self.workspace / "experience_replay.db")
            self._experience_replay = replay
        ep = replay.record_experience(error_text, resolution, diagnosis)
        return f"Successfully recorded experience episode #{ep.id} (`{ep.fingerprint}`) in episodic memory."

    async def tool_execute_command(
        self,
        command: str,
        cwd: str = "",
        _timeout_override: float | None = None,
    ) -> str:
        """Run commands in an isolated Docker container by default.

        The host-process route is available only through explicit FULL_ACCESS.
        Missing Docker never causes a silent fallback to host execution.
        """
        if not command or not str(command).strip():
            return "Error: command is required."
        timeout = _command_timeout(45.0)
        if _timeout_override is not None:
            timeout = min(timeout, max(0.5, float(_timeout_override)))
        if _cfg.full_access_enabled():
            return await self._tool_execute_command_host(command, cwd, timeout)

        docker_bin = shutil.which("docker")
        if not docker_bin:
            return (
                "Error: isolated command execution requires Docker, but Docker was not found. "
                "No host-shell fallback was attempted. Install Docker and build/pull the configured image locally, or explicitly "
                "enable TITAN_FULL_ACCESS only in a trusted environment."
            )

        try:
            workspace = self.workspace.resolve()
            working_dir = self._resolve_path(cwd) if cwd else workspace
            relative_cwd = working_dir.relative_to(workspace).as_posix()
        except (OSError, ValueError, PermissionError) as exc:
            return f"Error: command working directory must be inside the workspace: {exc}"

        image = os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:_@-]{0,254}", image):
            return "Error: TITAN_COMMAND_SANDBOX_IMAGE is not a valid Docker image reference."
        container_cwd = "/workspace" if relative_cwd in ("", ".") else f"/workspace/{relative_cwd}"
        container_name = f"titan-command-{uuid.uuid4().hex}"
        docker_cmd = [
            docker_bin, "run", "--name", container_name, "--rm", "--pull=never",
            "--network=none",
            "--memory=512m", "--cpus=1.0", "--pids-limit=128",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,src={workspace},dst=/workspace,rw",
            "--workdir", container_cwd,
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        docker_cmd.extend([image, "sh", "-lc", str(command)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr, timed_out, truncated = await _bounded_communicate(proc, timeout)
            if timed_out:
                await _best_effort_docker_rm(docker_bin, container_name)
                return f"Error: sandboxed command timed out after {timeout:.0f} seconds."
            if truncated:
                await _best_effort_docker_rm(docker_bin, container_name)
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = 125 if truncated and proc.returncode == 0 else proc.returncode
            result = [f"### DOCKER COMMAND SANDBOX (Exit {exit_code})"]
            if out_str:
                result.append(f"STDOUT:\n{out_str}")
            if err_str:
                result.append(f"STDERR:\n{err_str}")
            if truncated:
                result.append(f"[Output truncated at {MAX_CAPTURE_BYTES_PER_STREAM} bytes per stream; process stopped]")
            if not out_str and not err_str:
                result.append("(No output produced)")
            return "\n".join(result)
        except asyncio.CancelledError:
            await _best_effort_docker_rm(docker_bin, container_name)
            raise
        except (OSError, RuntimeError) as exc:
            return f"Error: sandboxed command could not start: {exc!s}"

    async def _tool_execute_command_host(
        self, command: str, cwd: str, timeout: float
    ) -> str:
        """Explicit FULL_ACCESS host-shell path; never selected as fallback."""
        working_dir = Path(cwd).resolve() if cwd else self.workspace.resolve()
        try:
            shell_cmd = (
                ["powershell", "-NoProfile", "-Command", command]
                if sys.platform == "win32"
                else ["bash", "-c", command]
            )
            spawn_options: dict[str, Any] = {}
            if os.name == "posix":
                spawn_options["start_new_session"] = True
            proc = await asyncio.create_subprocess_exec(
                *shell_cmd,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_options,
            )
            stdout, stderr, timed_out, truncated = await _bounded_communicate(
                proc, timeout, process_group=(os.name == "posix")
            )
            if timed_out:
                return f"Error: host command timed out after {timeout:.0f} seconds."
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = 125 if truncated and proc.returncode == 0 else proc.returncode
            result = [f"### HOST COMMAND (Exit {exit_code})"]
            if out_str:
                result.append(f"STDOUT:\n{out_str}")
            if err_str:
                result.append(f"STDERR:\n{err_str}")
            if truncated:
                result.append(f"[Output truncated at {MAX_CAPTURE_BYTES_PER_STREAM} bytes per stream; process stopped]")
            return "\n".join(result)
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError) as exc:
            return f"Command execution error: {exc!s}"

    def tool_read_file(self, path: str) -> str:
        fpath = self._resolve_path(path)
        if not fpath.exists():
            return f"Error: File '{fpath}' does not exist."
        if fpath.is_dir():
            return f"Error: '{fpath}' is a directory, not a file."
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content if content else "(File is empty)"
        except OSError as e:
            return f"Error reading file '{fpath}': {e!s}"

    def tool_analyze_python_file(self, path: str) -> str:
        """Read and explain Python structure using AST only; never execute source."""
        try:
            target = self._resolve_path(path)
        except (OSError, PermissionError, ValueError) as exc:
            return f"Error: cannot access source file: {exc!s}"
        if target.suffix.lower() != ".py":
            return "Error: analyze_python_file accepts Python (.py) files only."
        if not target.is_file():
            return f"Error: Python source file does not exist: {path}"
        try:
            from .core.code_intel.source_analyzer import analyze_python_source, format_analysis

            size = target.stat().st_size
            if size > 1_000_000:
                return "Error: source file exceeds the 1000000-byte static-analysis limit."
            source = target.read_text(encoding="utf-8")
            return format_analysis(analyze_python_source(source, filename=Path(path).as_posix()))
        except (OSError, UnicodeError, ValueError) as exc:
            return f"Error: could not analyze Python source: {type(exc).__name__}: {exc!s}"

    def tool_analyze_python_repository(self, max_files: int = 50) -> str:
        """Summarize local Python modules/imports without executing project code."""
        try:
            from .core.code_intel.repo_analyzer import analyze_python_repository, format_repository_analysis

            result = analyze_python_repository(self.workspace, max_files=max_files)
            formatted = format_repository_analysis(result)
            output_cap = 24_000
            if len(formatted) > output_cap:
                formatted = formatted[:output_cap] + "\n[Output truncated to keep repository analysis bounded.]"
            return formatted
        except (OSError, TypeError, ValueError) as exc:
            return f"Error: could not analyze Python repository: {type(exc).__name__}: {exc!s}"

    async def tool_laya_decide(
        self,
        state: str,
        questions: dict[str, Any],
        backend: str = "auto",
        model: str = "",
    ) -> str:
        """Run local, API-key-free typed decisions; this is not a chat-model call."""
        from .laya_decisions import predict_decisions

        return await predict_decisions(state, questions, backend=backend, model=model)

    def tool_write_file(self, path: str, content: str) -> str:
        fpath = self._resolve_path(path)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {fpath}."
        except OSError as e:
            return f"Error writing file '{fpath}': {e!s}"

    def tool_edit_file(self, path: str, target_text: str, replacement_text: str) -> str:
        fpath = self._resolve_path(path)
        if not fpath.exists():
            return f"Error: File '{fpath}' does not exist."
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if target_text not in content:
                return f"Error: target_text not found in '{fpath}'. Please verify exact character match."
            new_content = content.replace(target_text, replacement_text, 1)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"Successfully updated '{fpath}'."
        except OSError as e:
            return f"Error editing file '{fpath}': {e!s}"

    def tool_list_directory(self, path: str | Path = "") -> str:
        target = self._resolve_path(path) if path else self.workspace
        if not target.exists():
            return f"Error: Directory '{target}' does not exist."
        try:
            items = []
            for item in target.iterdir():
                kind = "DIR" if item.is_dir() else "FILE"
                size = item.stat().st_size if item.is_file() else "-"
                items.append(f"[{kind}] {item.name} ({size} bytes)")
            return "\n".join(items) if items else "(Directory is empty)"
        except OSError as e:
            return f"Error listing directory '{target}': {e!s}"

    async def tool_web_search(self, query: str, max_results: int = 5) -> str:
        try:
            loop = asyncio.get_running_loop()
            def _search():
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n---")
                return "\n".join(results) if results else "No results found."
            return await loop.run_in_executor(None, _search)
        except (RuntimeError, OSError) as e:
            return f"Search error: {e!s}"

    async def tool_scrape_webpage(self, url: str) -> str:
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            loop = asyncio.get_running_loop()
            def _fetch():
                with urllib.request.urlopen(req, timeout=15) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
                return "\n".join(clean_lines)[:6000]
            return await loop.run_in_executor(None, _fetch)
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            return f"Scraping error for {url}: {e!s}"

    async def tool_python_eval(self, code: str) -> str:
        """Execute Python using the same sandbox boundary as shell commands."""
        if not code or not str(code).strip():
            return "Error: Python code is required."
        command = f"python -I -c {shlex.quote(str(code))}"
        result = await self.tool_execute_command(command, cwd=".")
        return f"### PYTHON SANDBOX RESULT\n{result}"

    async def tool_deep_search(self, topic: str) -> str:
        from .deep_search import DeepSearchEngine
        engine = DeepSearchEngine()
        data = await engine.run(topic)
        summary = [f"### DEEP RESEARCH DOSSIER: {topic}"]
        summary.append(f"Sub-queries explored: {len(data['sub_queries'])}")
        summary.append(f"Unique sources analyzed: {data['total_sources_found']}\n")
        summary.append("#### Primary Sources:")
        for s in data["sources"][:5]:
            summary.append(f"- **{s['title']}**: {s['snippet']} (URL: {s['url']})")
        if data["deep_pages"]:
            summary.append("\n#### Deep Scraped Insights:")
            for p in data["deep_pages"]:
                summary.append(f"**[{p['title']}]**: {p['content'][:800]}...\n")
        return "\n".join(summary)

    async def tool_deep_coder(self, task_name: str, files: dict[str, str], test_code: str = "") -> str:
        from .deep_coder import DeepCoderEngine
        engine = DeepCoderEngine(self)
        res = await engine.execute_coding_cycle(task_name, files, test_code if test_code else None)
        out = [f"### DEEP CODING REPORT: {task_name}"]
        out.append(f"Status: {res['status']}")
        out.append("Created / Modified Files:")
        for f in res["created_files"]:
            out.append(f"- `{f['path']}`: {f['status']}")
        if res["syntax_checks"]:
            out.append("\nSyntax Validation:")
            for p, sc in res["syntax_checks"].items():
                out.append(f"- `{p}`: {'VALID' if sc['valid'] else 'ERROR: ' + sc['details']}")
        if res["test_passed"] is not None:
            out.append(f"\nUnit Test Execution: {'PASSED' if res['test_passed'] else 'FAILED'}")
            out.append(f"Output:\n```\n{res['test_output']}\n```")
        return "\n".join(out)

    def tool_workspace_rag(self, query: str, top_k: int = 4) -> str:
        try:
            rag = WorkspaceRAG(self.workspace)
            top_k = max(1, min(int(top_k or 4), 8))
            results = rag.search(query, top_k=top_k)
            if not results:
                return "No relevant snippets found in the workspace for this query."
            lines = [f"### WORKSPACE RAG RESULTS for: {query}"]
            for r in results:
                lines.append(f"\n**{r['path']}** (chunk {r['chunk_index']}, score {r['score']}):")
                lines.append(r["snippet"])
            return "\n".join(lines)
        except (RuntimeError, OSError, ValueError) as e:
            return f"Workspace RAG error: {e!s}"

    def tool_launch_application(self, app_or_command: str) -> str:
        try:
            if sys.platform == "win32":
                subprocess.Popen(f"start {app_or_command}", shell=True)
            else:
                subprocess.Popen(app_or_command, shell=True)
            return f"Launched application/command: '{app_or_command}'"
        except (OSError, subprocess.SubprocessError) as e:
            return f"Failed to launch application: {e!s}"

    def tool_system_info(self) -> str:
        """Reads live host environment facts (OS, CPU, RAM, disk, Python)."""
        import logging
        log = logging.getLogger(__name__)
        try:
            lines = []
            lines.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
            lines.append(f"Machine: {platform.machine()} | Node: {platform.node()}")
            lines.append(f"Python: {platform.python_version()} ({sys.executable})")
            lines.append(f"CPU cores: {os.cpu_count() or 'unknown'}")

            # RAM (total + free)
            try:
                if sys.platform == "win32":
                    import ctypes
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        total_gb = stat.ullTotalPhys / (1024 ** 3)
                        free_gb = stat.ullAvailPhys / (1024 ** 3)
                        lines.append(f"RAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({stat.dwMemoryLoad}% used)")
                    else:
                        lines.append("RAM: unable to read")
                else:
                    with open("/proc/meminfo") as f:
                        meminfo = {}
                        for line in f:
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                meminfo[parts[0].strip()] = parts[1].strip()
                    total_kb = int(meminfo.get("MemTotal", "0").split()[0])
                    avail_kb = int(meminfo.get("MemAvailable", "0").split()[0])
                    lines.append(f"RAM: {avail_kb/1048576:.1f} GB free / {total_kb/1048576:.1f} GB total")
            except (OSError, ValueError) as e:
                log.debug("RAM read failed: %s", e)
                lines.append(f"RAM: read failed ({e})")

            # Disk on workspace drive
            try:
                usage = shutil.disk_usage(str(self.workspace))
                lines.append(f"Disk: {usage.free/(1024**3):.1f} GB free / {usage.total/(1024**3):.1f} GB total")
            except OSError as e:
                log.debug("Disk usage failed: %s", e)

            # Software hints
            try:
                node_ver = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
                lines.append(f"Node.js: {node_ver or 'not found'}")
            except (OSError, subprocess.SubprocessError) as e:
                log.debug("Node version check failed: %s", e)
            try:
                git_ver = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5, check=False).stdout.strip()
                lines.append(f"Git: {git_ver or 'not found'}")
            except (OSError, subprocess.SubprocessError) as e:
                log.debug("Git version check failed: %s", e)

            lines.append(f"Workspace: {self.workspace}")
            return "\n".join(lines)
        except RuntimeError as e:
            log.error("System info error: %s", e)
            return f"System info error: {e!s}"

    async def tool_manage_processes(self, action: str = "list", pattern: str = "") -> str:
        """Lists or kills OS processes (tasklist/taskkill on Windows, ps/kill on Unix)."""
        action = (action or "list").lower()
        try:
            if sys.platform == "win32":
                if action == "list":
                    cmd = ["tasklist", "/FO", "CSV", "/NH"]
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                    if stderr and not stdout:
                        return f"Error listing processes: {stderr.decode('utf-8', errors='ignore')[:500]}"
                    rows = []
                    for line in stdout.decode("utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('","')
                        if len(parts) >= 2:
                            name = parts[0].strip('"')
                            pid = parts[1].strip('"')
                            if pattern and pattern.lower() not in name.lower():
                                continue
                            rows.append(f"PID {pid}: {name}")
                    if not rows:
                        return f"No processes found matching '{pattern}'." if pattern else "No processes found."
                    return "\n".join(rows[:100])
                elif action == "kill":
                    if not pattern:
                        return "Error: manage_processes kill requires 'pattern' (PID or image name)."
                    cmd = ["taskkill", "/F"]
                    if pattern.strip().isdigit():
                        cmd += ["/PID", pattern.strip()]
                    else:
                        cmd += ["/IM", pattern.strip()]
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                    out = (stdout or b"").decode("utf-8", errors="ignore").strip()
                    err = (stderr or b"").decode("utf-8", errors="ignore").strip()
                    return f"{out} {err}".strip() or f"Kill command finished (exit {proc.returncode})."
                else:
                    return f"Error: unknown action '{action}' (use 'list' or 'kill')."
            else:
                # Unix
                if action == "list":
                    proc = await asyncio.create_subprocess_exec(
                        "ps", "-eo", "pid,comm",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                    rows = []
                    for line in stdout.decode("utf-8", errors="ignore").splitlines():
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            pid, name = parts[0], parts[1]
                            if pattern and pattern.lower() not in name.lower():
                                continue
                            rows.append(f"PID {pid}: {name}")
                    return "\n".join(rows[:100]) if rows else f"No processes found matching '{pattern}'."
                elif action == "kill":
                    if not pattern:
                        return "Error: manage_processes kill requires 'pattern' (PID)."
                    proc = await asyncio.create_subprocess_exec(
                        "kill", "-9", pattern.strip(),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20.0)
                    out = (stdout or b"").decode("utf-8", errors="ignore").strip()
                    err = (stderr or b"").decode("utf-8", errors="ignore").strip()
                    return f"{out} {err}".strip() or f"Kill command finished (exit {proc.returncode})."
                else:
                    return f"Error: unknown action '{action}' (use 'list' or 'kill')."
        except asyncio.TimeoutError:
            return "Error: process query timed out after 20 seconds."
        except (OSError, RuntimeError) as e:
            return f"Manage processes error: {e!s}"

    def tool_clipboard_get(self) -> str:
        """Reads text from the system clipboard."""
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.OpenClipboard(0)
                try:
                    if ctypes.windll.user32.IsClipboardFormatAvailable(13):  # CF_UNICODETEXT
                        h_mem = ctypes.windll.user32.GetClipboardData(13)
                        if h_mem:
                            locked = ctypes.windll.kernel32.GlobalLock(h_mem)
                            if locked:
                                text = ctypes.wstring_at(locked)
                                ctypes.windll.kernel32.GlobalUnlock(h_mem)
                                return text
                    return "(Clipboard empty or not text)"
                finally:
                    ctypes.windll.user32.CloseClipboard()
            else:
                # Unix: use xclip or xsel
                for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "-b"]]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
                        if result.returncode == 0 and result.stdout:
                            return result.stdout
                    except (OSError, subprocess.SubprocessError):
                        continue
                return "(Clipboard tools not available: install xclip or xsel)"
        except RuntimeError as e:
            return f"Clipboard read error: {e!s}"

    def tool_clipboard_set(self, text: str) -> str:
        """Writes text to the system clipboard."""
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.OpenClipboard(0)
                try:
                    ctypes.windll.user32.EmptyClipboard()
                    h_mem = ctypes.windll.kernel32.GlobalAlloc(0x0042, (len(text) + 1) * 2)  # GMEM_MOVEABLE
                    if h_mem:
                        locked = ctypes.windll.kernel32.GlobalLock(h_mem)
                        if locked:
                            ctypes.memmove(locked, text.encode("utf-16-le"), len(text) * 2 + 2)
                            ctypes.windll.kernel32.GlobalUnlock(h_mem)
                            ctypes.windll.user32.SetClipboardData(13, h_mem)  # CF_UNICODETEXT
                            return f"Clipboard set ({len(text)} chars)"
                    return "Failed to allocate clipboard memory"
                finally:
                    ctypes.windll.user32.CloseClipboard()
            else:
                # Unix: use xclip or xsel
                for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]]:
                    try:
                        proc = subprocess.run(cmd, input=text, text=True, timeout=2, capture_output=True, check=False)
                        if proc.returncode == 0:
                            return f"Clipboard set ({len(text)} chars)"
                    except (OSError, subprocess.SubprocessError):
                        continue
                return "(Clipboard tools not available: install xclip or xsel)"
        except RuntimeError as e:
            return f"Clipboard write error: {e!s}"

    def tool_screenshot(self, monitor: int = 0) -> str:
        """Takes a screenshot of the specified monitor and returns base64 PNG."""
        try:
            if sys.platform == "win32":
                import base64
                import ctypes
                from ctypes import wintypes

                # wintypes in the stdlib stubs omits the monitor/bitmap structs,
                # so define them explicitly (fields match the Win32 SDK).
                class _MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD),
                    ]

                class _BITMAPINFOHEADER(ctypes.Structure):
                    _fields_ = [
                        ("biSize", wintypes.DWORD),
                        ("biWidth", ctypes.c_long),
                        ("biHeight", ctypes.c_long),
                        ("biPlanes", ctypes.c_ushort),
                        ("biBitCount", ctypes.c_ushort),
                        ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", ctypes.c_long),
                        ("biYPelsPerMeter", ctypes.c_long),
                        ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD),
                    ]

                class _BITMAPINFO(ctypes.Structure):
                    _fields_ = [
                        ("bmiHeader", _BITMAPINFOHEADER),
                        ("bmiColors", wintypes.DWORD * 1),
                    ]
                
                user32 = ctypes.windll.user32
                gdi32 = ctypes.windll.gdi32
                
                # Get monitor info
                monitors = []
                def enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    monitors.append(hMonitor)
                    return True
                MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
                user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(enum_proc), 0)
                
                if monitor >= len(monitors):
                    return f"Error: Monitor {monitor} not found (only {len(monitors)} monitors)"
                
                h_mon = monitors[monitor]
                mi = _MONITORINFO()
                mi.cbSize = ctypes.sizeof(mi)
                user32.GetMonitorInfoW(h_mon, ctypes.byref(mi))
                
                left, top, right, bottom = mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom
                width = right - left
                height = bottom - top
                
                hdc_screen = user32.GetDC(0)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
                hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
                old_bmp = gdi32.SelectObject(hdc_mem, hbmp)
                
                gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, left, top, 0x00CC0020)  # SRCCOPY
                
                # Get bitmap bits
                bmi = _BITMAPINFO()
                bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
                bmi.bmiHeader.biWidth = width
                bmi.bmiHeader.biHeight = -height  # negative for top-down
                bmi.bmiHeader.biPlanes = 1
                bmi.bmiHeader.biBitCount = 32
                bmi.bmiHeader.biCompression = 0  # BI_RGB
                
                bits = ctypes.create_string_buffer(width * height * 4)
                gdi32.GetDIBits(hdc_mem, hbmp, 0, height, bits, ctypes.byref(bmi), 0)
                
                # Convert to PNG using Python
                import io

                from PIL import Image
                img = Image.frombuffer('RGBA', (width, height), bits.raw, 'raw', 'BGRA', 0, 1)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                b64 = base64.b64encode(buf.getvalue()).decode('ascii')
                
                gdi32.SelectObject(hdc_mem, old_bmp)
                gdi32.DeleteObject(hbmp)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(0, hdc_screen)
                
                return f"data:image/png;base64,{b64}"
            else:
                # Unix: use scrot or maim
                import base64
                for cmd in [["scrot", "-u", "-"], ["maim", "-u"]]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, timeout=5, check=False)
                        if result.returncode == 0 and result.stdout:
                            b64 = base64.b64encode(result.stdout).decode('ascii')
                            return f"data:image/png;base64,{b64}"
                    except (OSError, subprocess.SubprocessError):
                        continue
                return "(Screenshot tools not available: install scrot or maim)"
        except RuntimeError as e:
            return f"Screenshot error: {e!s}"

    def tool_key_press(self, keys: str) -> str:
        """Simulates keyboard key presses."""
        try:
            if sys.platform == "win32":
                import ctypes
                import time
                
                user32 = ctypes.windll.user32
                
                # Parse key combination
                key_map = {
                    'ctrl': 0x11, 'control': 0x11,
                    'alt': 0x12,
                    'shift': 0x10,
                    'win': 0x5B, 'windows': 0x5B,
                    'enter': 0x0D, 'return': 0x0D,
                    'tab': 0x09,
                    'esc': 0x1B, 'escape': 0x1B,
                    'space': 0x20,
                    'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
                    'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
                    'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
                    'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
                    'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
                    'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
                    'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
                    'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
                    'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
                    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
                    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
                }
                
                parts = [p.strip().lower() for p in keys.split('+')]
                vk_codes = [key_map.get(p) for p in parts if p in key_map]
                
                if not vk_codes:
                    return f"Error: Unknown keys in '{keys}'"
                
                # Press modifiers first
                for vk in vk_codes[:-1]:
                    user32.keybd_event(vk, 0, 0, 0)
                    time.sleep(0.01)
                
                # Press main key
                user32.keybd_event(vk_codes[-1], 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(vk_codes[-1], 0, 2, 0)  # KEYEVENTF_KEYUP
                
                # Release modifiers
                for vk in reversed(vk_codes[:-1]):
                    user32.keybd_event(vk, 0, 2, 0)
                    time.sleep(0.01)
                
                return f"Pressed: {keys}"
            else:
                # Unix: use xdotool
                try:
                    subprocess.run(["xdotool", "key", keys], timeout=3, check=False)
                    return f"Pressed: {keys}"
                except (OSError, subprocess.SubprocessError):
                    return "(xdotool not installed)"
        except RuntimeError as e:
            return f"Key press error: {e!s}"

    def tool_mouse_click(self, x: int, y: int, button: str = "left", double: bool = False) -> str:
        """Simulates a mouse click at coordinates."""
        try:
            if sys.platform == "win32":
                import ctypes
                import time
                
                user32 = ctypes.windll.user32
                
                # Move to position
                user32.SetCursorPos(x, y)
                time.sleep(0.02)
                
                button_map = {"left": (0x02, 0x04), "right": (0x08, 0x10), "middle": (0x20, 0x40)}
                down, up = button_map.get(button, button_map["left"])
                
                def click_once():
                    user32.mouse_event(down, 0, 0, 0, 0)
                    time.sleep(0.02)
                    user32.mouse_event(up, 0, 0, 0, 0)
                
                click_once()
                if double:
                    time.sleep(0.1)
                    click_once()
                
                return f"Clicked {button} at ({x}, {y}){' (double)' if double else ''}"
            else:
                # Unix: use xdotool
                try:
                    btn_map = {"left": "1", "right": "3", "middle": "2"}
                    btn = btn_map.get(button, "1")
                    cmd = ["xdotool", "mousemove", str(x), str(y), "click"]
                    if double:
                        cmd.insert(-1, "--repeat")
                        cmd.insert(-1, "2")
                    else:
                        cmd.append(btn)
                    subprocess.run(cmd, timeout=3, check=False)
                    return f"Clicked {button} at ({x}, {y})"
                except (OSError, subprocess.SubprocessError):
                    return "(xdotool not installed)"
        except RuntimeError as e:
            return f"Mouse click error: {e!s}"

    def tool_mouse_move(self, x: int, y: int, duration: float = 0) -> str:
        """Moves mouse cursor to coordinates."""
        try:
            import time
            if sys.platform == "win32":
                import ctypes
                user32 = ctypes.windll.user32
                if duration > 0:
                    # Smooth move
                    cur_x, cur_y = user32.GetCursorPos()
                    steps = max(10, int(duration * 60))
                    for i in range(1, steps + 1):
                        t = i / steps
                        nx = int(cur_x + (x - cur_x) * t)
                        ny = int(cur_y + (y - cur_y) * t)
                        user32.SetCursorPos(nx, ny)
                        time.sleep(duration / steps)
                else:
                    user32.SetCursorPos(x, y)
                return f"Moved mouse to ({x}, {y})"
            else:
                subprocess.run(["xdotool", "mousemove", str(x), str(y)], timeout=3, check=False)
                return f"Moved mouse to ({x}, {y})"
        except RuntimeError as e:
            return f"Mouse move error: {e!s}"

    async def tool_list_windows(self) -> str:
        """Lists all visible windows."""
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                
                windows = []
                
                def enum_windows(hwnd, lparam):
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buff, length + 1)
                            title = buff.value
                            # Get process name
                            pid = wintypes.DWORD()
                            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                            try:
                                import psutil
                                proc = psutil.Process(pid.value)
                                proc_name = proc.name()
                            except RuntimeError:
                                proc_name = f"PID:{pid.value}"
                            windows.append(f"HWND:{hwnd} | {proc_name} | {title[:80]}")
                    return True
                
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                user32.EnumWindows(WNDENUMPROC(enum_windows), 0)
                
                return "\n".join(windows[:50]) if windows else "No visible windows"
            else:
                # Unix: use wmctrl
                try:
                    result = await asyncio.to_thread(
                        subprocess.run, ["wmctrl", "-l"],
                        capture_output=True, text=True, timeout=3, check=False,
                    )
                    if result.returncode == 0:
                        return result.stdout[:3000]
                    return "(wmctrl not available)"
                except (OSError, subprocess.SubprocessError):
                    return "(wmctrl not installed)"
        except RuntimeError as e:
            return f"List windows error: {e!s}"

    async def tool_window_control(self, action: str, title: str) -> str:
        """Controls a window: minimize, maximize, restore, close, or bring to front."""
        try:
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.windll.user32
                
                target_hwnd = None
                
                # Try as HWND first
                if title.isdigit():
                    target_hwnd = wintypes.HWND(int(title))
                    if not user32.IsWindow(target_hwnd):
                        target_hwnd = None
                
                # Find by title if not found
                if not target_hwnd:
                    def enum_windows(hwnd, lparam):
                        nonlocal target_hwnd
                        if user32.IsWindowVisible(hwnd):
                            length = user32.GetWindowTextLengthW(hwnd)
                            if length > 0:
                                buff = ctypes.create_unicode_buffer(length + 1)
                                user32.GetWindowTextW(hwnd, buff, length + 1)
                                if title.lower() in buff.value.lower():
                                    target_hwnd = hwnd
                                    return False  # Stop enumeration
                        return True
                    
                    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
                    user32.EnumWindows(WNDENUMPROC(enum_windows), 0)
                
                if not target_hwnd:
                    return f"Window not found: {title}"
                
                action_map = {
                    "minimize": 6,      # SW_MINIMIZE
                    "maximize": 3,      # SW_MAXIMIZE
                    "restore": 9,       # SW_RESTORE
                    "close": None,      # Special: send WM_CLOSE
                    "foreground": None, # Special: SetForegroundWindow
                }
                
                if action == "close":
                    user32.PostMessageW(target_hwnd, 0x0010, 0, 0)  # WM_CLOSE
                    return f"Sent close message to window: {title}"
                elif action == "foreground":
                    user32.SetForegroundWindow(target_hwnd)
                    return f"Brought window to front: {title}"
                elif action in action_map:
                    user32.ShowWindow(target_hwnd, action_map[action])
                    return f"Window {action}: {title}"
                else:
                    return f"Unknown action: {action}"
            else:
                # Unix: use wmctrl/xdotool
                try:
                    if action == "close":
                        await asyncio.to_thread(subprocess.run, ["wmctrl", "-c", title], timeout=3, check=False)
                    elif action in ("minimize", "maximize", "restore"):
                        state_map = {"minimize": "-b add,iconic", "maximize": "-b add,maximized_vert,maximized_horz", "restore": "-b remove,maximized_vert,maximized_horz"}
                        await asyncio.to_thread(subprocess.run, ["wmctrl", "-r", title, state_map[action]], timeout=3, check=False)
                    elif action == "foreground":
                        await asyncio.to_thread(subprocess.run, ["wmctrl", "-a", title], timeout=3, check=False)
                    return f"Window {action}: {title}"
                except (OSError, subprocess.SubprocessError):
                    return "(wmctrl not installed)"
        except RuntimeError as e:
            return f"Window control error: {e!s}"

    # ================= Phase 7: Full Autonomy tools =================

    async def _run_command_raw(
        self,
        command: str,
        cwd: str = "",
        timeout: float = 60.0,
    ) -> tuple[int, str, str]:
        """Execute a command and return (exit_code, stdout, stderr) — no decoration.

        Used by self_heal so the repair loop can inspect raw output.
        Phase 8: FULL access raises the ceiling to 10 minutes.
        """
        result = await self.tool_execute_command(
            command,
            cwd=cwd,
            _timeout_override=float(timeout or 60.0),
        )
        match = re.search(r"\(Exit (-?\d+)\)", result)
        if not match:
            return 126, "", result
        exit_code = int(match.group(1))
        body = result[match.end():].lstrip("\n")
        stdout = body.split("STDOUT:\n", 1)[1].split("\nSTDERR:\n", 1)[0] if "STDOUT:\n" in body else ""
        stderr = body.split("\nSTDERR:\n", 1)[1] if "\nSTDERR:\n" in body else ""
        return exit_code, stdout, stderr

    async def tool_self_heal(self, command: str, cwd: str = "", max_attempts: int = 3) -> str:
        """Self-healing command runner: run, diagnose failure, repair, re-run."""
        from .heal import SelfHealEngine

        engine = SelfHealEngine()
        result = await engine.heal(
            command,
            max_attempts=max_attempts,
            run=lambda cmd: self._run_command_raw(cmd, cwd=cwd),
        )
        return result.to_text()

    async def tool_download_file(self, url: str, dest: str = "") -> str:
        """Stream a bounded download into the workspace with DNS-level SSRF checks.

        Normal and FULL modes allow only public HTTP(S) addresses, including
        redirect targets. ABSOLUTE mode intentionally bypasses that restriction.
        Files are written to a same-directory temporary file and atomically
        replaced only after a complete download within the configured limit.
        """
        url = (url or "").strip()
        absolute = _cfg.absolute_access_enabled()
        try:
            parts = urlsplit(url)
        except ValueError:
            return "Error: URL is malformed."
        if not absolute:
            if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
                return "Error: only http(s) URLs with a valid hostname are allowed."
            if parts.username is not None or parts.password is not None:
                return "Error: URLs containing embedded credentials are not allowed."
            try:
                _ = parts.port  # Validate malformed and out-of-range ports.
            except ValueError:
                return "Error: URL contains an invalid port."
            policy = PolicyEngine()
            check = policy.check_network_target(url)
            if check.decision == "deny":
                return "Error: refused to download a private or loopback target."

        source_display = parts._replace(query="", fragment="").geturl()
        try:
            fname = Path(parts.path).name or "download.bin"
            if fname in {".", ".."}:
                fname = "download.bin"
            if dest:
                target = self._resolve_path(dest)
                if dest.endswith(("/", "\\")) or target.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    out_path = target / fname
                else:
                    out_path = target
            else:
                out_path = self._resolve_path(fname)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fetch_timeout = 120.0 if _cfg.full_access_enabled() else 30.0
            max_bytes = FULL_ACCESS_MAX_DOWNLOAD_BYTES if _cfg.full_access_enabled() else MAX_DOWNLOAD_BYTES
            fd, temp_name = tempfile.mkstemp(prefix=".titan-download-", dir=out_path.parent)
            os.close(fd)
            temp_path = Path(temp_name)

            class _DownloadTooLarge(Exception):
                pass

            async def _fetch_public() -> int:
                import aiohttp

                class _PublicOnlyResolver(aiohttp.abc.AbstractResolver):
                    async def resolve(self, host, port=0, family=socket.AF_INET):
                        loop = asyncio.get_running_loop()
                        try:
                            records = await loop.getaddrinfo(
                                host, port, family=family, type=socket.SOCK_STREAM
                            )
                        except OSError as exc:
                            raise OSError("DNS resolution failed") from exc
                        addresses = []
                        for af, _socktype, proto, _canonname, sockaddr in records:
                            address = ipaddress.ip_address(sockaddr[0])
                            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
                                address = address.ipv4_mapped
                            if not address.is_global:
                                raise OSError("refused non-public DNS address")
                            addresses.append({
                                "hostname": host,
                                "host": sockaddr[0],
                                "port": port,
                                "family": af,
                                "proto": proto,
                                "flags": socket.AI_NUMERICHOST,
                            })
                        if not addresses:
                            raise OSError("DNS returned no usable addresses")
                        return addresses

                    async def close(self):
                        return None

                connector = aiohttp.TCPConnector(
                    resolver=_PublicOnlyResolver(), use_dns_cache=False
                )
                timeout = aiohttp.ClientTimeout(total=fetch_timeout)
                total = 0
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout, trust_env=False
                ) as session:
                    async with session.get(url, allow_redirects=True, max_redirects=5) as response:
                        response.raise_for_status()
                        with temp_path.open("wb") as output:
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                total += len(chunk)
                                if total > max_bytes:
                                    raise _DownloadTooLarge
                                output.write(chunk)
                return total

            def _fetch_absolute() -> int:
                total = 0
                req = urllib.request.Request(url, headers={"User-Agent": "Titan-Agent/8.0"})
                with urllib.request.urlopen(req, timeout=fetch_timeout) as response, temp_path.open("wb") as output:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise _DownloadTooLarge
                        output.write(chunk)
                return total

            try:
                size = await asyncio.to_thread(_fetch_absolute) if absolute else await _fetch_public()
                os.replace(temp_path, out_path)
                return f"Downloaded {source_display}\nSaved: {out_path}\nSize: {size} bytes"
            except _DownloadTooLarge:
                return f"Error: download exceeds the configured {max_bytes}-byte safety limit."
            finally:
                temp_path.unlink(missing_ok=True)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return f"Download failed ({type(exc).__name__}); no partial file was installed."
        except Exception as exc:  # noqa: BLE001 - sanitize third-party/network exception details
            return f"Download failed ({type(exc).__name__}); no partial file was installed."

    def _start_server(self, port: int, directory: Path) -> str:
        """Start (or return existing) ThreadingHTTPServer on localhost:port."""
        from functools import partial

        if not hasattr(self, "_http_servers"):
            self._http_servers: dict[int, ThreadingHTTPServer] = {}
        existing = self._http_servers.get(port)
        if existing:
            return f"Server already running at http://127.0.0.1:{port}"
        directory.mkdir(parents=True, exist_ok=True)
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._http_servers[port] = server
        return f"Serving {directory} at http://127.0.0.1:{port}"

    def tool_start_http_server(self, port: int = 8000, directory: str | Path = "") -> str:
        try:
            directory = self._resolve_path(directory) if directory else self.workspace
            port = int(port or 8000)
            # Phase 8: FULL access widens the allowed range to any valid port.
            lo, hi = (1, 65535) if _cfg.full_access_enabled() else (1024, 49151)
            if not lo <= port <= hi:
                return f"Error: port must be in {lo}-{hi}."
            return self._start_server(port, directory)
        except (OSError, ValueError) as e:
            return f"Could not start HTTP server: {e!s}"

    def tool_stop_http_server(self, port: int = 8000) -> str:
        port = int(port or 8000)
        server = getattr(self, "_http_servers", {}).get(port)
        if not server:
            return f"No HTTP server running on port {port}."
        server.shutdown()
        server.server_close()
        self._http_servers.pop(port, None)
        return f"Stopped HTTP server on port {port}."

    async def tool_take_screenshot(self, dest: str = "") -> str:
        """Capture the primary screen to a PNG (Windows via PowerShell)."""
        if sys.platform != "win32":
            return "Error: take_screenshot currently supports Windows only."
        fname = dest or f"screenshot_{int(time.time())}.png"
        out = self._resolve_path(fname)
        out.parent.mkdir(parents=True, exist_ok=True)
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
            "$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            f"$bmp = New-Object System.Drawing.Bitmap($b.Width, $b.Height); "
            "$g = [System.Drawing.Graphics]::FromImage($bmp); "
            "$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size); "
            f"$bmp.Save('{out}'); $g.Dispose(); $bmp.Dispose()"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-Command", ps,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if not out.exists():
                return f"Screenshot failed: {stderr.decode('utf-8', errors='ignore')[:500]}"
            return f"Screenshot saved: {out} ({out.stat().st_size} bytes)"
        except (OSError, asyncio.TimeoutError) as e:
            return f"Screenshot error: {e!s}"

    async def tool_self_update(self, run_tests: bool = True) -> str:
        """Pull latest code, install requirements, run tests for the repo root."""
        repo = self.workspace
        # Walk up to find a .git dir
        while not (repo / ".git").exists() and repo.parent != repo:
            repo = repo.parent
        if not (repo / ".git").exists():
            return "No git repository found from workspace."
        lines = ["### SELF-UPDATE"]
        for cmd in ["git pull --ff-only", "git status --short"]:
            code, out, err = await self._run_command_raw(cmd, cwd=str(repo), timeout=120)
            lines.append(f"$ {cmd} (exit {code})")
            if out.strip():
                lines.append(out.strip())
            if err.strip():
                lines.append(err.strip())
        # requirements install (best-effort, optional)
        code, out, err = await self._run_command_raw(
            f'"{sys.executable}" -m pip install -r requirements.txt --quiet',
            cwd=str(repo),
            timeout=300,
        )
        lines.append(f"$ pip install -r requirements.txt (exit {code}){(' ' + err.strip()[:300]) if err.strip() else ''}")
        if run_tests and (repo / "tests").exists():
            code, out, err = await self._run_command_raw(
                f'"{sys.executable}" -m pytest -q',
                cwd=str(repo),
                timeout=600,
            )
            lines.append(f"\n$ pytest -q (exit {code})")
            if out.strip():
                lines.append(out.strip()[-1500:])
            if err.strip():
                lines.append(err.strip()[-500:])
        return "\n".join(lines)

    # ---- Autonomous task queue tools ----

    def _queue(self):
        from .queue import TaskQueue

        if not hasattr(self, "_task_queue") or self._task_queue is None:
            self._task_queue = TaskQueue(TASK_QUEUE_FILE)
        return self._task_queue

    def tool_task_enqueue(
        self,
        task: str,
        name: str = "",
        priority: int = 0,
        schedule_at: float = 0.0,
    ) -> str:
        if not task or not str(task).strip():
            return "Error: task is required."
        try:
            q = self._queue()
            tid = q.enqueue(
                str(task).strip(),
                name=name or None,
                priority=int(priority or 0),
                schedule_at=float(schedule_at or 0),
            )
            return f"Task #{tid} enqueued: {(name or task)[:80]}"
        except (OSError, ValueError) as e:
            return f"Enqueue failed: {e!s}"

    def tool_task_list(self, status: str = "", limit: int = 20) -> str:
        q = self._queue()
        tasks = q.list(status=status or None, limit=int(limit or 20))
        if not tasks:
            return "Task queue is empty."
        lines = ["### TASK QUEUE"]
        for t in tasks:
            lines.append(
                f"#{t.id} [{t.status}] prio={t.priority} attempts={t.attempts}/{t.max_attempts} :: {t.name}"
            )
        return "\n".join(lines)

    def tool_task_stats(self) -> str:
        q = self._queue()
        stats = q.stats()
        return "Queue stats: " + ", ".join(f"{k}={v}" for k, v in stats.items() if v) or "Queue is empty."

    def tool_task_cancel(self, task_id: int) -> str:
        ok = self._queue().cancel(int(task_id))
        return f"Task #{task_id} cancelled." if ok else f"Task #{task_id} not found or already running."

    # ---- Deep subagent tools (Phase 7) + dedicated staff (Phase 9) ----

    async def tool_subagent_delegate(
        self,
        task: str,
        role: str = "generalist",
        label: str = "",
    ) -> str:
        from .staff import StaffPool

        if not task or not str(task).strip():
            return "Error: task is required."
        pool = StaffPool()
        res = await pool.run(str(role or "generalist"), str(task).strip(), label=label or "")
        return _subagent_result_text(res)

    async def tool_subagent_team(
        self,
        tasks: list[str],
        roles: list[str] | None = None,
    ) -> str:
        from .staff import StaffPool

        if not tasks:
            return "Error: tasks list is required."
        # Phase 8: FULL access raises the parallel subagent bound 2 -> 8.
        pool = StaffPool(max_workers=min(_subagent_worker_cap(), len(tasks)))
        results = await pool.team(
            [str(t) for t in tasks],
            roles=[str(r) for r in roles] if roles else None,
        )
        return "\n\n".join(_subagent_result_text(r) for r in results)

    def tool_subagent_roles(self) -> str:
        from .staff import staff_catalog

        entries = staff_catalog()
        if not entries:
            return "No dedicated subagent roles configured."
        return "### DEDICATED SUBAGENT ROLES\n" + "\n".join(
            f"- {e['id']}: {e['title']} — {e['description']}" for e in entries
        )

    def tool_subagent_route(self, task: str) -> str:
        from .intent_router import route_intent

        if not task or not str(task).strip():
            return "Error: task is required."
        route = route_intent(str(task))
        return "### INTENT ROUTE\n" + route.plan_text()

    # ---- Phase 22 Genesis: Hierarchical Organization Tools ----

    async def tool_orchestrator_run(
        self,
        goal: str,
        departments: list[str] | None = None,
    ) -> str:
        from .orchestrator import MetaOrchestrator

        if not goal or not str(goal).strip():
            return "Error: goal is required."
        orch = getattr(self, "_meta_orchestrator", None)
        if orch is None:
            orch = MetaOrchestrator()
            self._meta_orchestrator = orch
        res = await orch.orchestrate(str(goal).strip(), departments=departments)
        return res.synthesis

    async def tool_team_delegate(
        self,
        department: str,
        task: str,
        role: str = "",
    ) -> str:
        from .team_leads import build_team_leads

        if not task or not str(task).strip():
            return "Error: task is required."
        dep = str(department or "engineering").strip().lower()
        leads = getattr(self, "_team_leads", None)
        if leads is None:
            leads = build_team_leads()
            self._team_leads = leads

        lead = leads.get(dep)
        if not lead:
            return f"Error: unknown department '{department}'. Choose from: engineering, research, operations, quality_security."

        res = await lead.execute_task(task=str(task).strip(), role=str(role).strip() if role else None)
        notes = f" (Notes: {', '.join(res.verification_notes)})" if res.verification_notes else ""
        return f"### TEAM DELIVERABLE: {lead.name} [{res.verification_verdict}]{notes}\nAssigned worker: {res.assigned_role}\n\n{res.output}"

    def tool_team_status(self) -> str:
        from .orchestrator import MetaOrchestrator

        orch = getattr(self, "_meta_orchestrator", None)
        if orch is None:
            orch = MetaOrchestrator()
            self._meta_orchestrator = orch

        b = orch.budget.summary()
        lines = [
            "### TITAN HIERARCHICAL ORGANIZATION STATUS",
            f"- **Meta-Orchestrator**: Active (Tokens spent: {b['total_tokens_spent']}, Steps spent: {b['total_steps_spent']})",
            f"- **Active Goal**: {orch.goal_memory.project_goal or '(None)'}",
            f"- **Milestones**: {len(orch.goal_memory.milestones)} recorded",
            "",
            "### DEPARTMENT LEADS (Level 2):",
        ]
        for dep, lead in orch.leads.items():
            lines.append(f"- **{lead.name}** (`{dep}`): Managing {len(lead.managed_roles)} roles ({', '.join(lead.managed_roles)})")

        return "\n".join(lines)

    async def tool_dag_plan_and_run(self, goal: str) -> str:
        from .orchestrator import MetaOrchestrator

        if not goal or not str(goal).strip():
            return "Error: goal is required."
        orch = getattr(self, "_meta_orchestrator", None)
        if orch is None:
            orch = MetaOrchestrator()
            self._meta_orchestrator = orch

        res = await orch.orchestrate_dag(str(goal).strip())
        self._last_dag_result = res
        return res.summary

    async def tool_dag_visualize(self, goal: str) -> str:
        from .core.dag import DAGPlanner

        if not goal or not str(goal).strip():
            return "Error: goal is required."
        planner = DAGPlanner()
        graph = await planner.create_dag_plan(str(goal).strip())
        mermaid = graph.to_mermaid()
        return f"### TASK GRAPH (DAG) VISUALIZATION\n```mermaid\n{mermaid}\n```\n\n**Topological Order**: {' -> '.join(graph.topological_sort())}"

    async def tool_debate_solve(self, question: str, rounds: int = 2) -> str:
        from .core.reasoning.debate import DebateEngine
        from .llm_client import LLMClient
        from .structured import LLMBridge

        if not question or not str(question).strip():
            return "Error: question is required."
        client = getattr(self, "_llm_client", None) or LLMClient()
        engine = DebateEngine(LLMBridge(client))
        res = await engine.run_debate(str(question).strip(), rounds=int(rounds or 2))
        return res.summary()

    async def tool_reflexion_solve(self, task: str) -> str:
        from .core.reasoning.reflexion import ReflexionEngine
        from .llm_client import LLMClient
        from .structured import LLMBridge

        if not task or not str(task).strip():
            return "Error: task is required."
        client = getattr(self, "_llm_client", None) or LLMClient()
        engine = ReflexionEngine(LLMBridge(client))
        res = await engine.run(str(task).strip(), max_cycles=3)
        status_line = f"Reflexion completed in {res.total_cycles} cycles (Verdict: {'PASS' if res.success else 'FAILED'}, Improved: {res.improved})"
        return f"### REFLEXION OUTCOME [{status_line}]\n\n{res.final_output}"

    @property
    def memory(self):
        if getattr(self, "_memory_mgr", None) is None:
            from .memory import MemoryManager
            self._memory_mgr = MemoryManager()
        return self._memory_mgr

    def tool_kg_query(self, entity_id: str, depth: int = 2) -> str:
        """KNOWLEDGE GRAPH: Queries the causal/dependency knowledge graph around an entity."""
        if not entity_id or not str(entity_id).strip():
            return "Error: entity_id is required."
        if not self.memory.knowledge_graph.entities:
            from .core.memory import WorkspaceASTGraphExtractor
            extractor = WorkspaceASTGraphExtractor(self.workspace)
            extractor.extract(kg=self.memory.knowledge_graph, max_files=50)

        res = self.memory.query_kg(str(entity_id).strip(), depth=int(depth or 2))
        lines = [f"### KNOWLEDGE GRAPH QUERY: {entity_id} ({res['total_connections']} connection(s))"]
        for c in res["connections"]:
            lines.append(f"- ({c['source']}) --[{c['relation']}]--> ({c['target']}) [neighbor: {c['neighbor_name']} ({c['neighbor_type']})]")
        if not res["connections"]:
            lines.append("No connections found for entity in knowledge graph.")
        return "\n".join(lines)

    def tool_kg_impact_analysis(self, entity_id: str) -> str:
        """KNOWLEDGE GRAPH: Computes downstream impact and blast radius if an entity is modified."""
        if not entity_id or not str(entity_id).strip():
            return "Error: entity_id is required."
        if not self.memory.knowledge_graph.entities:
            from .core.memory import WorkspaceASTGraphExtractor
            extractor = WorkspaceASTGraphExtractor(self.workspace)
            extractor.extract(kg=self.memory.knowledge_graph, max_files=50)

        res = self.memory.kg_impact(str(entity_id).strip())
        lines = [
            f"### KNOWLEDGE GRAPH IMPACT ANALYSIS: {entity_id}",
            f"- Direct Dependents: {len(res['direct_dependents'])} ({', '.join(res['direct_dependents']) if res['direct_dependents'] else 'none'})",
            f"- Total Blast Radius: {len(res['total_impacted_entities'])} entities",
            f"- Max Cascade Depth: {res['depth_reached']}",
        ]
        if res["total_impacted_entities"]:
            lines.append("\n**All Impacted Entities**:")
            for ent in res["total_impacted_entities"]:
                lines.append(f"- {ent}")
        return "\n".join(lines)

    def tool_kg_add_fact(self, source: str, relation: str, target: str) -> str:
        """KNOWLEDGE GRAPH: Adds a semantic fact or causal dependency between two entities."""
        if not source or not str(source).strip():
            return "Error: source is required."
        if not relation or not str(relation).strip():
            return "Error: relation is required."
        if not target or not str(target).strip():
            return "Error: target is required."
        s, r, t = str(source).strip(), str(relation).strip(), str(target).strip()
        self.memory.add_kg_fact(s, r, t)
        return f"Successfully added knowledge graph fact: ({s}) --[{r}]--> ({t})"

    def tool_kg_index_workspace(self, max_files: int = 50) -> str:
        """KNOWLEDGE GRAPH: Scans workspace ASTs to build knowledge graph."""
        from .core.memory import WorkspaceASTGraphExtractor
        extractor = WorkspaceASTGraphExtractor(self.workspace)
        kg = extractor.extract(kg=self.memory.knowledge_graph, max_files=int(max_files or 50))
        self.memory.knowledge_graph.save()
        return f"Indexed workspace AST into Knowledge Graph: {len(kg.entities)} entities, {len(kg.relations)} relations."

    @property
    def reliability_tracker(self):
        from .tool_stats import TOOL_RELIABILITY
        return TOOL_RELIABILITY

    def tool_discover(self, query: str, category: str = "", limit: int = 8) -> str:
        """DYNAMIC TOOLS: Searches and discovers available tools by keyword/category."""
        if not query or not str(query).strip():
            return "Error: query is required."
        from .core.tools.dynamic_registry import DynamicToolSelector
        defs = self.get_tool_definitions()
        matches = DynamicToolSelector.discover_tools(
            str(query).strip(),
            defs,
            category=str(category).strip(),
            limit=int(limit or 8),
        )
        if not matches:
            return f"No tools found matching query '{query}'."

        lines = [f"### DISCOVERED TOOLS ({len(matches)} matches for '{query}'):"]
        for td in matches:
            fn = td.get("function", td)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "").split("\n")[0][:120]
            grade = self.reliability_tracker.get_grade(name)
            score = self.reliability_tracker.get_score(name)
            lines.append(f"- **{name}** [Grade {grade} ({score:.2f})]: {desc}")
        return "\n".join(lines)

    def tool_reliability_report(self) -> str:
        """TOOL RELIABILITY: Reports EWMA scores, grades, and mitigation advice."""
        return self.reliability_tracker.format_report_text()

    @property
    def model_router(self):
        if getattr(self, "_model_router", None) is None:
            from .core.routing import ModelRouter
            self._model_router = ModelRouter()
        return self._model_router

    @property
    def budget_tracker(self):
        if getattr(self, "_budget_tracker", None) is None:
            from .core.routing import CognitiveBudgetTracker
            self._budget_tracker = CognitiveBudgetTracker()
        return self._budget_tracker

    def tool_model_route(self, task: str, prior_failures: int = 0) -> str:
        """MODEL ROUTER: Analyzes task and returns recommended model tier and pricing."""
        if not task or not str(task).strip():
            return "Error: task is required."
        decision = self.model_router.route(str(task).strip(), prior_failures=int(prior_failures or 0))
        lines = [
            f"### MODEL ROUTE DECISION: `{decision.model_name}` [{decision.tier.value.upper()}]",
            f"- **Rationale**: {decision.rationale}",
            f"- **Input Pricing**: ${decision.estimated_input_cost_per_1k:.5f} / 1k tokens",
            f"- **Output Pricing**: ${decision.estimated_output_cost_per_1k:.5f} / 1k tokens",
            f"- **Escalated**: {decision.is_escalated}",
        ]
        return "\n".join(lines)

    def tool_model_budget_status(self) -> str:
        """COGNITIVE BUDGET: Reports cumulative token usage and USD expenditure."""
        return self.budget_tracker.format_status_text()

    @property
    def sandbox_env(self):
        if getattr(self, "_sandbox_env", None) is None:
            from .core.sandbox import SandboxEnvironment
            self._sandbox_env = SandboxEnvironment(self.workspace)
        return self._sandbox_env

    @property
    def safe_runner(self):
        if getattr(self, "_safe_runner", None) is None:
            from .core.sandbox import SafeScriptRunner
            self._safe_runner = SafeScriptRunner(self.workspace, sandbox_env=self.sandbox_env)
        return self._safe_runner

    def tool_sandbox_snapshot_create(self, name: str) -> str:
        """Creates an immediate point-in-time filesystem snapshot of the workspace."""
        if not name or not str(name).strip():
            return "Error: snapshot name is required."
        snap = self.sandbox_env.create_snapshot(str(name).strip())
        return (
            f"### WORKSPACE SNAPSHOT CREATED: `{snap.name}`\n"
            f"- Total files indexed: {len(snap.file_hashes)}\n"
            f"- Timestamp: {snap.timestamp}\n"
            f"- Root path: `{snap.root_path}`"
        )

    def tool_sandbox_snapshot_rollback(self, name: str) -> str:
        """Reverts workspace files to a previously captured snapshot."""
        if not name or not str(name).strip():
            return "Error: snapshot name is required."
        report = self.sandbox_env.rollback(str(name).strip())
        return (
            f"### WORKSPACE ROLLBACK EXECUTED [{name}]:\n"
            f"- Restored files: {len(report['restored'])}\n"
            f"- Deleted newly-created files: {len(report['deleted_new'])}\n"
            f"- Status: {'SUCCESS' if report['success'] else 'FAILED'}"
        )

    async def tool_sandbox_execute(
        self,
        code: str,
        language: str = "python",
        timeout: float = 30.0,
        rollback_on_failure: bool = True,
    ) -> str:
        """Run Python inside the command Docker sandbox, optionally rolling back workspace edits.

        The old subprocess runner only separated interpreter state; it did not
        provide OS isolation. This tool now uses the default container path and
        fails closed when Docker is unavailable (except explicit FULL_ACCESS).
        """
        if not code or not str(code).strip():
            return "Error: code is required."
        if str(language or "python").lower().strip() != "python":
            return "Error: only Python is supported by the isolated execution tool."
        is_safe, reason = self.safe_runner.validate_code_safety(str(code))
        if not is_safe:
            return f"Error: execution blocked by the static safety check: {reason}"

        snapshot_name = f"pre_python_exec_{time.time_ns()}"
        snap_created = False
        script_path = self._resolve_path(f".titan_python_exec_{time.time_ns()}.py")
        try:
            if rollback_on_failure:
                snap_created = bool(self.sandbox_env.create_snapshot(snapshot_name))
            script_path.write_text(str(code), encoding="utf-8")
            if _cfg.full_access_enabled():
                executable_path = shlex.quote(str(script_path))
            else:
                executable_path = "/workspace/" + script_path.relative_to(self.workspace.resolve()).as_posix()
            command = f"python -I {executable_path}"
            # Preserve the caller's timeout by temporarily running this tool's
            # standard execution route under its normal bounded command timeout.
            result = await self.tool_execute_command(
                command,
                cwd=".",
                _timeout_override=float(timeout or 30.0),
            )
        except (OSError, RuntimeError, ValueError, PermissionError) as exc:
            result = f"Error: isolated Python execution failed: {exc!s}"
        finally:
            script_path.unlink(missing_ok=True)

        match = re.search(r"\(Exit (-?\d+)\)", result)
        success = bool(match and int(match.group(1)) == 0)
        rolled_back = False
        if not success and rollback_on_failure and snap_created:
            try:
                rolled_back = bool(self.sandbox_env.rollback(snapshot_name).get("success"))
            except Exception:  # noqa: BLE001 - execution error must remain visible
                rolled_back = False
        status = "PASSED" if success else "FAILED / NOT RUN"
        return (
            f"### SANDBOX EXECUTION RESULT (PYTHON | {status})\n"
            f"- **Auto-Rolled Back**: {rolled_back}\n{result}"
        )

    @property
    def drift_detector(self):
        if getattr(self, "_drift_detector", None) is None:
            from .core.monitoring import QualityDriftDetector
            storage = self.workspace / ".titan" / "drift_metrics.json"
            self._drift_detector = QualityDriftDetector(storage_path=storage)
        return self._drift_detector

    def tool_drift_record_task(
        self,
        task_id: str,
        success: bool,
        steps: int = 1,
        duration_sec: float = 1.0,
        tokens_used: int = 0,
        failed_tools: str = "",
        category: str = "general",
    ) -> str:
        """Logs task execution telemetry for continuous quality regression tracking."""
        if not task_id or not str(task_id).strip():
            return "Error: task_id is required."
        ft_list = [t.strip() for t in str(failed_tools or "").split(",") if t.strip()]
        metric = self.drift_detector.record_task(
            task_id=str(task_id).strip(),
            success=bool(success),
            steps=int(steps or 1),
            duration_sec=float(duration_sec or 1.0),
            tokens_used=int(tokens_used or 0),
            failed_tools=ft_list,
            category=str(category or "general"),
        )
        status_str = "SUCCESS" if metric.success else "FAILED"
        return (
            f"### DRIFT TELEMETRY RECORDED: `{metric.task_id}` [{status_str}]\n"
            f"- Category: {metric.category} | Steps: {metric.steps} | Duration: {metric.duration_sec:.2f}s\n"
            f"- Tokens: {metric.tokens_used} | Failed Tools: {', '.join(metric.failed_tools) or 'None'}\n"
            f"- Total historical records: {len(self.drift_detector.get_metrics())}"
        )

    def tool_drift_check(
        self,
        window_size: int = 10,
        threshold_drop: float = 0.20,
    ) -> str:
        """Analyzes historical task telemetry for quality regression and drift."""
        report = self.drift_detector.check_drift(
            window_size=int(window_size or 10),
            threshold_drop=float(threshold_drop or 0.20),
        )
        return report.format_report_text()

    def tool_drift_status(self) -> str:
        """Displays longitudinal telemetry summary and drift health status."""
        metrics = self.drift_detector.get_metrics()
        if not metrics:
            return "### DRIFT TELEMETRY: No task metrics recorded yet."
        successes = sum(1 for m in metrics if m.success)
        total = len(metrics)
        rate = (successes / total) * 100.0 if total > 0 else 0.0
        avg_steps = sum(m.steps for m in metrics) / total if total > 0 else 0.0
        avg_dur = sum(m.duration_sec for m in metrics) / total if total > 0 else 0.0
        last = metrics[-1]
        return (
            f"### DRIFT TELEMETRY STATUS (Total Tasks: {total})\n"
            f"- Cumulative Success Rate: {rate:.1f}% ({successes}/{total})\n"
            f"- Average Steps: {avg_steps:.1f} steps/task\n"
            f"- Average Latency: {avg_dur:.2f}s\n"
            f"- Most Recent Task: `{last.task_id}` ({'SUCCESS' if last.success else 'FAILED'}, {last.steps} steps)"
        )

    @property
    def self_improvement_loop(self):
        if getattr(self, "_self_improvement_loop", None) is None:
            from .core.self_improvement import SelfImprovementLoop
            self._self_improvement_loop = SelfImprovementLoop(workspace_root=self.workspace)
        return self._self_improvement_loop

    @property
    def eval_suite(self):
        if getattr(self, "_eval_suite", None) is None:
            from .core.self_improvement import EvalSuite
            self._eval_suite = EvalSuite()
        return self._eval_suite

    def tool_self_improve_analyze_failure(
        self,
        task_id: str,
        prompt: str,
        failure_log: str,
        failed_tools: str = "",
        category: str = "general",
    ) -> str:
        """Analyzes a failed task, diagnoses root cause, and synthesizes a prescriptive rule."""
        if not task_id or not str(task_id).strip():
            return "Error: task_id is required."
        if not prompt or not str(prompt).strip():
            return "Error: prompt is required."
        if not failure_log or not str(failure_log).strip():
            return "Error: failure_log is required."

        ft_list = [t.strip() for t in str(failed_tools or "").split(",") if t.strip()]
        lesson = self.self_improvement_loop.analyze_failure(
            task_id=str(task_id).strip(),
            prompt=str(prompt).strip(),
            failure_log=str(failure_log).strip(),
            failed_tools=ft_list,
            category=str(category or "general"),
        )
        return (
            f"### FAILURE ANALYSIS & LESSON LEARNED [{lesson.category.upper()}]\n"
            f"- **Task ID**: `{lesson.task_id}`\n"
            f"- **Root Cause**: {lesson.root_cause}\n"
            f"- **Guidance**: {lesson.guidance}\n"
            f"- **Extracted Rule**: > {lesson.rule_text}"
        )

    def tool_self_improve_eval_run(self, category: str = "") -> str:
        """Executes automated benchmark evaluation suite to verify capability and catch regressions."""
        cat = str(category or "").strip()
        report = self.eval_suite.run_suite(category=cat)
        lines = [f"### EVAL RESULTS (Category: '{cat or 'all'}')"]
        if report["attempted"] == 0:
            lines.append("- **Status**: NOT RUN — no live evaluation runner is configured; this tool did not evaluate the agent.")
            lines.append("- No pass rate or quality score is reported because no cases were executed.")
        else:
            lines.extend([
                f"- **Status**: Executed {report['attempted']}/{report['total_cases']} cases",
                f"- **Pass Rate**: {report['pass_rate']}% ({report['passed']}/{report['attempted']} attempted cases passed)",
                f"- **Average Quality Score**: {report['average_score']:.2f} / 1.0",
                f"- **Average Duration**: {report['average_duration_sec']:.3f}s",
            ])
        lines.append("\nCases Summary:")
        for result in report["results"]:
            if result["status"] == "not_run":
                status_symbol = "NOT RUN"
            else:
                status_symbol = "PASS" if result["passed"] else "FAIL"
            score = f" (Score: {result['score']})" if result["score"] is not None else ""
            lines.append(f"  • `{result['case_id']}`: {status_symbol}{score}")
            if result.get("error"):
                lines.append(f"    - {result['error']}")
        return "\n".join(lines)

    def tool_self_improve_crystallize_lesson(
        self,
        lesson_title: str,
        guidance: str,
        category: str = "general",
    ) -> str:
        """Permanently records an extracted lesson into the Skill playbook library and Knowledge Graph."""
        if not lesson_title or not str(lesson_title).strip():
            return "Error: lesson_title is required."
        if not guidance or not str(guidance).strip():
            return "Error: guidance is required."

        from .core.self_improvement import ImprovementLesson
        lesson = ImprovementLesson(
            task_id=f"manual_{int(time.time())}",
            category=str(category or "general").strip(),
            symptom=f"Learned rule: {lesson_title}",
            root_cause=str(lesson_title).strip(),
            guidance=str(guidance).strip(),
            rule_text=f"RULE [{category.upper()}]: {guidance}",
        )
        kg = getattr(self, "_kg", None)
        status = self.self_improvement_loop.crystallize_lesson(
            lesson,
            knowledge_graph=kg,
        )
        skill_name = status.get("skill_name", "unknown")
        return (
            f"### LESSON CRYSTALLIZED SUCCESSFULLY\n"
            f"- Saved as Skill Playbook: `{skill_name}` (Auto-injectable)\n"
            f"- Knowledge Graph Fact Added: {status.get('knowledge_fact_added', False)}\n"
            f"- Rule: {lesson.rule_text}"
        )

    async def tool_docker_sandbox_run(
        self,
        command: str,
        image: str = "titan-agent-sandbox:local",
        memory_limit: str = "512m",
        cpu_quota: str = "1.0",
        mount_workspace: bool = False,
        network: str = "none",
        timeout: float = 60.0,
        _mount_path: Path | str | None = None,
    ) -> str:
        """Execute in a hardened ephemeral container; host workspace is opt-in.

        ``_mount_path`` is an internal-only hook for trusted subsystems and is
        intentionally absent from the model-visible tool schema.
        """
        if not command or not str(command).strip():
            return "Error: command is required."
        img = str(image or os.getenv("TITAN_COMMAND_SANDBOX_IMAGE", "titan-agent-sandbox:local")).strip()
        mem = str(memory_limit or "512m").strip().lower()
        try:
            cpus_num = float(cpu_quota or 1.0)
        except (TypeError, ValueError):
            return "Error: cpu_quota must be a number between 0.1 and 4.0."
        net = str(network or "none").strip().lower()
        t = max(1.0, min(float(timeout or 60.0), 300.0))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:_@-]{0,254}", img):
            return "Error: invalid Docker image reference."
        mem_match = re.fullmatch(r"([0-9]+)([kmg])", mem)
        memory_bytes = 0
        if mem_match:
            memory_bytes = int(mem_match.group(1)) * {
                "k": 1024, "m": 1024**2, "g": 1024**3,
            }[mem_match.group(2)]
        if not (64 * 1024**2 <= memory_bytes <= 4 * 1024**3) or not 0.1 <= cpus_num <= 4.0:
            return "Error: memory_limit must be 64m..4g and cpu_quota must be 0.1..4.0."
        if net not in {"none", "bridge"}:
            return "Error: network must be 'none' or explicitly enabled 'bridge'."

        docker_bin = shutil.which("docker")
        if not docker_bin:
            return "Error: Docker is required for isolated execution; no host fallback was attempted."

        host_mount: Path | None = None
        if _mount_path is not None:
            host_mount = Path(_mount_path).resolve()
            if not host_mount.is_dir():
                return "Error: internal sandbox mount path must be an existing directory."
        elif mount_workspace:
            host_mount = self.workspace.resolve()
        container_name = f"titan-sandbox-{uuid.uuid4().hex}"
        docker_cmd = [
            docker_bin, "run", "--name", container_name, "--rm", "--pull=never",
            f"--memory={mem}", f"--cpus={cpus_num:.1f}", "--pids-limit=128",
            f"--network={net}", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        ]
        if host_mount is not None:
            docker_cmd.extend(["--mount", f"type=bind,src={host_mount},dst=/workspace,rw"])
        else:
            docker_cmd.extend(["--tmpfs", "/workspace:rw,nosuid,nodev,size=64m"])
        docker_cmd.extend(["--workdir", "/workspace"])
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        docker_cmd.extend([img, "sh", "-lc", str(command)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr, timed_out, truncated = await _bounded_communicate(proc, t)
            if timed_out:
                await _best_effort_docker_rm(docker_bin, container_name)
                return f"Error: Docker container execution timed out after {t:.0f}s."
            if truncated:
                await _best_effort_docker_rm(docker_bin, container_name)
            out_str = stdout.decode("utf-8", errors="replace").strip()
            err_str = stderr.decode("utf-8", errors="replace").strip()
            exit_code = 125 if truncated and proc.returncode == 0 else proc.returncode
            res = [f"### DOCKER SANDBOX [{img}] (Exit {exit_code})"]
            if out_str:
                res.append(f"STDOUT:\n{out_str}")
            if err_str:
                res.append(f"STDERR:\n{err_str}")
            if truncated:
                res.append(f"[Output truncated at {MAX_CAPTURE_BYTES_PER_STREAM} bytes per stream; process stopped]")
            if not out_str and not err_str:
                res.append("(No output produced)")
            return "\n".join(res)
        except asyncio.CancelledError:
            await _best_effort_docker_rm(docker_bin, container_name)
            raise
        except (OSError, RuntimeError) as exc:
            return f"Docker execution error: {exc!s}"

    def tool_apply_patch(self, patch: str) -> str:
        """Applies a unified diff patch to files in the workspace."""
        if not patch or not str(patch).strip():
            return "Error: patch is required."

        lines = patch.strip().splitlines()
        file_diffs: list[tuple[str, list[list[str]]]] = []
        cur_file = None
        cur_hunks: list[list[str]] = []
        cur_hunk: list[str] | None = None

        for line in lines:
            if line.startswith("--- "):
                pass
            elif line.startswith("+++ "):
                raw_path = line[4:].strip().removeprefix("b/")
                if cur_file and cur_hunks:
                    file_diffs.append((cur_file, cur_hunks))
                cur_file = raw_path
                cur_hunks = []
                cur_hunk = None
            elif line.startswith("@@"):
                if cur_hunk is not None:
                    cur_hunks.append(cur_hunk)
                cur_hunk = []
            elif cur_hunk is not None:
                cur_hunk.append(line)
        if cur_file and cur_hunk is not None:
            cur_hunks.append(cur_hunk)
            file_diffs.append((cur_file, cur_hunks))

        if not file_diffs:
            return "Error: No valid unified diff hunks found in patch."

        applied_files = []
        for rel_path, hunks in file_diffs:
            fpath = self._resolve_path(rel_path)
            orig_text = ""
            if fpath.exists():
                orig_text = fpath.read_text(encoding="utf-8", errors="ignore")

            orig_lines = orig_text.splitlines()
            new_lines = list(orig_lines)

            for hunk in hunks:
                old_hunk_lines = [l[1:] for l in hunk if l.startswith(("-", " "))]
                new_hunk_lines = [l[1:] for l in hunk if l.startswith(("+", " "))]

                match_idx = -1
                hunk_len = len(old_hunk_lines)
                if hunk_len == 0:
                    new_lines.extend(new_hunk_lines)
                    continue

                for i in range(len(new_lines) - hunk_len + 1):
                    if new_lines[i : i + hunk_len] == old_hunk_lines:
                        match_idx = i
                        break

                if match_idx == -1:
                    stripped_old = [l.strip() for l in old_hunk_lines]
                    for i in range(len(new_lines) - hunk_len + 1):
                        if [l.strip() for l in new_lines[i : i + hunk_len]] == stripped_old:
                            match_idx = i
                            break

                if match_idx != -1:
                    new_lines[match_idx : match_idx + hunk_len] = new_hunk_lines
                else:
                    return f"Error: Patch conflict in '{rel_path}' - hunk could not be matched."

            fpath.parent.mkdir(parents=True, exist_ok=True)
            ends_newline = orig_text.endswith("\n") or not orig_text
            fpath.write_text("\n".join(new_lines) + ("\n" if ends_newline else ""), encoding="utf-8")
            applied_files.append(rel_path)

        return f"### PATCH APPLIED SUCCESSFULLY\nModified files: {', '.join(applied_files)} ({len(file_diffs)} file(s))"

    def tool_skill_save(
        self,
        name: str,
        guidance: str,
        description: str = "",
        keywords: str = "",
    ) -> str:
        """Creates or updates a persistent skill playbook in the skills library."""
        from .skills import SkillRegistry

        if not name or not str(name).strip():
            return "Error: skill name is required."
        if not guidance or not str(guidance).strip():
            return "Error: skill guidance is required."
        registry = SkillRegistry()
        try:
            path = registry.save_skill(name, description, keywords, guidance)
            return f"Skill '{name}' saved successfully to {path.name} ({len(guidance)} chars guidance). It will be auto-injected for matching tasks."
        except Exception as e:  # noqa: BLE001
            return f"Error saving skill: {e!s}"

    def tool_video_probe(self, file_path: str) -> str:
        """Inspects media file metadata via VideoEngine."""
        from titan_agent.core.multimedia.video_engine import VideoEngine
        target = self._resolve_path(file_path)
        info = VideoEngine.probe_media(str(target))
        if "error" in info:
            return f"❌ Video Probe Failed: {info['error']}"
        if "notice" in info:
            return f"ℹ️ {info.get('file')}: Size={info.get('size_mb')}MB. {info['notice']}"

        v = info.get("video", {})
        a = info.get("audio", {})
        return (
            f"🎬 **Media Metadata for {Path(file_path).name}**:\n"
            f"- Duration: {info.get('duration_sec', 0):.2f}s | Size: {info.get('size_mb', 0)} MB | Format: {info.get('format_name', 'unknown')}\n"
            f"- Video Stream: Codec={v.get('codec')} | Resolution={v.get('width')}x{v.get('height')} | FPS={v.get('fps')}\n"
            f"- Audio Stream: Codec={a.get('codec')} | Channels={a.get('channels')} | Sample Rate={a.get('sample_rate')} Hz"
        )

    def tool_video_montage_command(
        self,
        operation: str,
        input_video: str,
        output_video: str,
        start_time: str | None = None,
        duration: str | None = None,
        audio_track: str | None = None,
        aspect_ratio: str | None = None,
        speed: float = 1.0,
    ) -> str:
        """Generates and provides the FFmpeg montage command via VideoEngine."""
        from titan_agent.core.multimedia.video_engine import VideoEngine
        in_p = self._resolve_path(input_video)
        out_p = self._resolve_path(output_video)
        audio_p = str(self._resolve_path(audio_track)) if audio_track else None

        res = VideoEngine.generate_montage_command(
            operation=operation,
            input_video=str(in_p),
            output_video=str(out_p),
            start_time=start_time,
            duration=duration,
            audio_track=audio_p,
            aspect_ratio=aspect_ratio,
            speed=speed,
        )
        if "error" in res:
            return f"❌ Montage Command Error: {res['error']}"

        status_icon = "✅" if res.get("ffmpeg_available") else "⚠️"
        availability = "FFmpeg found in PATH/system" if res.get("ffmpeg_available") else "FFmpeg not detected in PATH (install via 'winget install Gyan.FFmpeg')"

        return (
            f"{status_icon} **Video Montage Command ({operation})**:\n"
            f"```bash\n{res['command_str']}\n```\n"
            f"- Status: {availability}\n"
            f"- You can run this command directly with `execute_command`."
        )

    def tool_blender_generate_scene(
        self,
        primitive: str = "cube",
        output_image: str = "render.png",
        engine: str = "BLENDER_EEVEE",
        save_path: str | None = None,
    ) -> str:
        """Generates a standalone headless Python script (bpy) for Blender 3D scene creation and rendering."""
        from titan_agent.core.multimedia.blender_engine import BlenderEngine
        out_img = str(self._resolve_path(output_image))
        script_code = BlenderEngine.generate_procedural_scene_script(
            primitive=primitive,
            output_image=out_img,
            engine=engine,
        )
        if save_path:
            save_p = self._resolve_path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            save_p.write_text(script_code, encoding="utf-8")
            return (
                f"✅ Blender procedural scene script generated and saved to `{save_path}`!\n"
                f"To render headlessly, run:\n"
                f"```bash\nblender --background --python {save_p}\n```\n"
                f"Or execute using the `blender_execute_script` tool."
            )

        return f"```python\n{script_code}\n```"

    def tool_blender_execute_script(self, script_path: str) -> str:
        """Executes a Blender Python script in headless background mode."""
        from titan_agent.core.multimedia.blender_engine import BlenderEngine
        target = self._resolve_path(script_path)
        if not target.exists():
            return f"❌ Error: Script file not found: {target}"

        res = BlenderEngine.execute_blender_script(str(target))
        if not res.get("success"):
            err = res.get("error", "Unknown error")
            notice = res.get("notice", "")
            return f"❌ Blender execution failed: {err}\n{notice}"

        return (
            f"✅ Blender background execution completed successfully!\n"
            f"- Binary: `{res.get('blender_path')}`\n"
            f"- Exit Code: {res.get('returncode')}\n"
            f"```\n{res.get('stdout_tail')}\n```"
        )

    @property
    def domain_manager(self):
        """Lazy accessor for active DomainManager."""
        if getattr(self, "_domain_manager_ref", None) is None:
            from titan_agent.core.domain.manager import DomainManager
            self._domain_manager_ref = DomainManager.get_instance()
        return self._domain_manager_ref

    def tool_domain_list(self) -> str:
        """Lists all registered domain profiles and indicates the active one."""
        domains = self.domain_manager.list_domains()
        lines = [f"🌐 **Universal Agent Industry Domains ({len(domains)} available)**:"]
        for d in domains:
            active_marker = " 👈 **[ACTIVE]**" if d["is_active"] else ""
            builtin_label = "Built-in" if d["is_builtin"] else "Custom Enterprise"
            lines.append(
                f"- {d['icon']} **`{d['name']}`** — {d['display_name']} *({builtin_label})*{active_marker}\n"
                f"  {d['description']}"
            )
        lines.append("\n💡 *To switch active domain, call `domain_switch(domain='...')` or run CLI with `--domain <name>`.*")
        return "\n".join(lines)

    def tool_domain_switch(self, domain: str) -> str:
        """Switches the active industry domain profile."""
        try:
            profile = self.domain_manager.switch_domain(domain)
            return (
                f"✅ **Switched Active Domain**: {profile.icon} **{profile.display_name}** (`{profile.name}`)\n"
                f"- Description: {profile.description}\n"
                f"- Guardrails Active: {len(profile.mandatory_guardrails)}\n"
                f"- Recommended Tools: {', '.join(profile.preferred_tools) or 'All'}"
            )
        except ValueError as e:
            return f"❌ Error switching domain: {e}"

    def tool_domain_get_active(self) -> str:
        """Returns details of the currently active domain profile."""
        current = self.domain_manager.active_domain
        guardrails_str = "\n".join(f"- ⚠️ {g}" for g in current.mandatory_guardrails) if current.mandatory_guardrails else "(None)"
        tools_str = ", ".join(current.preferred_tools) if current.preferred_tools else "All standard tools"
        return (
            f"🌐 **Current Active Domain Profile**:\n"
            f"- **Identifier:** `{current.name}`\n"
            f"- **Name:** {current.icon} {current.display_name}\n"
            f"- **Type:** {'Built-in' if current.is_builtin else 'Custom Enterprise'}\n"
            f"- **Description:** {current.description}\n\n"
            f"**Operational Guidelines:**\n{current.system_prompt_overlay}\n\n"
            f"**Mandatory Guardrails:**\n{guardrails_str}\n\n"
            f"**Preferred Tools:** {tools_str}"
        )

    def tool_domain_create(
        self,
        name: str,
        display_name: str,
        description: str,
        system_prompt_overlay: str,
        icon: str = "🌐",
        mandatory_guardrails: list[str] | None = None,
        preferred_tools: list[str] | None = None,
        forbidden_tools: list[str] | None = None,
    ) -> str:
        """Defines and persists a new custom industry domain profile."""
        from titan_agent.core.domain.profile import DomainProfile
        norm_name = str(name).strip().lower().replace(" ", "_")
        if not norm_name:
            return "❌ Error: domain name cannot be empty."

        profile = DomainProfile(
            name=norm_name,
            display_name=display_name.strip() or norm_name,
            icon=icon.strip() or "🌐",
            description=description.strip(),
            system_prompt_overlay=system_prompt_overlay.strip(),
            mandatory_guardrails=mandatory_guardrails or [],
            preferred_tools=preferred_tools or [],
            forbidden_tools=forbidden_tools or [],
            is_builtin=False,
        )
        self.domain_manager.register_domain(profile, persist=True)
        return (
            f"✅ Custom Domain Profile `{norm_name}` created and persisted successfully!\n"
            f"- Name: {profile.icon} {profile.display_name}\n"
            f"- Stored to: `{self.domain_manager.domains_dir / f'{norm_name}.json'}`\n"
            f"- You can activate it now with `domain_switch(domain='{norm_name}')`."
        )




