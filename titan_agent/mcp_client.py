import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# A generous per-server concurrency cap: many servers can run in parallel,
# but a single healing server can't be flooded with unbounded concurrent calls.
MAX_CONCURRENT_CALLS = 32
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_START_TIMEOUT = 90.0
STDERR_TAIL_LIMIT = 40


class MCPServerConnection:
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process: asyncio.subprocess.Process | None = None
        self.tools: list[dict[str, Any]] = []
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail: list[str] = []
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        self.is_connected = False
        # Generation counter: every (re)start bumps it so listeners of an
        # old process can never touch the state of a new one.
        self._generation = 0

    @property
    def recent_stderr(self) -> str:
        """Last lines the server wrote to stderr (useful in error messages)."""
        return "\n".join(self._stderr_tail[-STDERR_TAIL_LIMIT:])

    async def start(self) -> bool:
        try:
            # Seamless restart: tear down any previous process first so stale
            # listeners can't leak into the new one.
            await self._teardown_process()
            self._generation += 1
            gen = self._generation

            full_env = os.environ.copy()
            full_env.update(self.env)
            # Ensure windows shell support if npx/npm/cmd
            cmd = self.command
            args = self.args

            # Windows executable resolution
            if sys.platform == "win32" and cmd in ("npx", "npm"):
                cmd = f"{cmd}.cmd"

            self.process = await asyncio.create_subprocess_exec(
                cmd,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env
            )
            self._read_task = asyncio.create_task(self._listen_stdout(self.process, gen))
            # Never let stderr fill up — otherwise a chatty server deadlocks us.
            # The tail is kept so failures stay debuggable.
            self._stderr_task = asyncio.create_task(self._drain_stderr(self.process, gen))
            self.is_connected = True

            # Send initialize handshake
            _ = await self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "TitanAgent",
                    "version": "1.0.0"
                }
            })

            # Send initialized notification
            await self.send_notification("notifications/initialized", {})

            # Fetch available tools
            tools_res = await self.send_request("tools/list", {})
            self.tools = tools_res.get("tools", [])
            return True
        except asyncio.CancelledError:
            raise
        except OSError as e:
            print(f"[MCP] Failed to start server '{self.name}': {e}")
            self.is_connected = False
            await self.stop()
            return False
        except RuntimeError as e:
            print(f"[MCP] Failed to start server '{self.name}': {e}")
            self.is_connected = False
            await self.stop()
            return False

    async def _teardown_process(self):
        """Kill/close the current process and its pipes (no-op if none)."""
        import logging
        log = logging.getLogger(__name__)
        
        proc = self.process
        self.process = None
        if self._read_task:
            self._read_task.cancel()
            try:
                await asyncio.wait_for(self._read_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except RuntimeError as e:
                log.debug("Error waiting for read task: %s", e)
            self._read_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await asyncio.wait_for(self._stderr_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except RuntimeError as e:
                log.debug("Error waiting for stderr task: %s", e)
            self._stderr_task = None
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError(f"MCP server '{self.name}' stopped."))
        self._pending_requests.clear()
        if proc:
            try:
                if proc.returncode is None:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        proc.kill()
            except OSError as e:
                log.debug("Error terminating process: %s", e)
            # Close pipes to avoid "unclosed transport" ResourceWarnings on Windows
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if pipe is not None and hasattr(pipe, "close"):
                        pipe.close()
                except OSError as e:
                    log.debug("Error closing pipe: %s", e)

    async def _listen_stdout(self, process: asyncio.subprocess.Process, gen: int):
        import logging
        log = logging.getLogger(__name__)
        try:
            while self.process is process and process.stdout:
                try:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='ignore').strip()
                    if not decoded:
                        continue
                    try:
                        data = json.loads(decoded)
                        if gen != self._generation:
                            break
                        req_id = data.get("id")
                        if req_id is not None and req_id in self._pending_requests:
                            future = self._pending_requests.pop(req_id)
                            if not future.done():
                                if "error" in data:
                                    future.set_exception(RuntimeError(data["error"]))
                                else:
                                    future.set_result(data.get("result", {}))
                    except json.JSONDecodeError:
                        pass
                except (asyncio.CancelledError, RuntimeError, OSError) as e:
                    log.debug("Listen stdout error: %s", e)
                    break
        finally:
            # Only the listener of the CURRENT process may flip the flag.
            if gen == self._generation and self.process is process:
                self.is_connected = False
                # The server went away: fail every request still waiting so
                # callers get a fast, actionable error instead of a 60s hang.
                pend = list(self._pending_requests.items())
                self._pending_requests.clear()
                for req_id, future in pend:
                    if not future.done():
                        future.set_exception(
                            RuntimeError(f"MCP server '{self.name}' closed the connection.")
                        )

    async def _drain_stderr(self, process: asyncio.subprocess.Process, gen: int):
        """Drain stderr continuously so the pipe never blocks the server."""
        import logging
        log = logging.getLogger(__name__)
        try:
            while self.process is process and process.stderr:
                try:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    text = line.decode('utf-8', errors='ignore').strip()
                    if text:
                        self._stderr_tail.append(text)
                        if len(self._stderr_tail) > STDERR_TAIL_LIMIT * 4:
                            del self._stderr_tail[: len(self._stderr_tail) - STDERR_TAIL_LIMIT * 4]
                except (asyncio.CancelledError, OSError) as e:
                    log.debug("Drain stderr error: %s", e)
                    break
        except RuntimeError as e:
            log.debug("Drain stderr outer error: %s", e)

    async def send_request(self, method: str, params: dict[str, Any], timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Any:
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"MCP server '{self.name}' is not running.")
        self._request_id += 1
        req_id = self._request_id
        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future

        msg = json.dumps(req) + "\n"
        self.process.stdin.write(msg.encode('utf-8'))
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise RuntimeError(
                f"MCP request '{method}' to server '{self.name}' timed out after {timeout}s"
            ) from None

    async def send_notification(self, method: str, params: dict[str, Any]):
        if not self.process or not self.process.stdin:
            return
        notif = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        msg = json.dumps(notif) + "\n"
        self.process.stdin.write(msg.encode('utf-8'))
        await self.process.stdin.drain()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Any:
        # Bounded concurrency per server, while still allowing many servers
        # (and many calls within one server) to run at the same time.
        async with self._semaphore:
            res = await self.send_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            }, timeout=timeout)
        # Extract content
        contents = res.get("content", [])
        text_outputs = []
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                text_outputs.append(c.get("text", ""))
        return "\n".join(text_outputs) if text_outputs else json.dumps(res)

    async def stop(self):
        await self._teardown_process()
        self.is_connected = False


class MCPManager:
    def __init__(self, config_file: Path | None = None):
        self.config_file = config_file
        self.servers: dict[str, MCPServerConnection] = {}
        # full tool name -> (server name, original tool name)
        # Built in get_all_tools; used by execute_tool for reliable resolution
        # even when server or tool names contain underscores.
        self._tool_map: dict[str, tuple[str, str]] = {}

    def load_config(self) -> dict[str, Any]:
        if not self.config_file or not self.config_file.exists():
            return {"mcpServers": {}}
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {"mcpServers": {}}

    def _resolve_server_command(self, details: dict[str, Any], workspace_dir: Path) -> tuple[str, list[str], dict[str, str]]:
        def _expand(s: str) -> str:
            s = s.replace("{WORKSPACE}", str(workspace_dir)).replace("{BASE_DIR}", str(workspace_dir.parent))
            # Expand any {ENV_VAR_NAME} token (args and command alike) so vault
            # paths / keys can live in .env — e.g. --vault notes={OBSIDIAN_VAULT}.
            # Missing vars become empty strings, mirroring the env-value policy:
            # startup stays non-fatal and the server fails its own validation.
            for m in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", s):
                s = s.replace("{" + m + "}", os.environ.get(m, ""))
            return s

        cmd = _expand(str(details.get("command", "")))
        args = [_expand(str(a)) for a in details.get("args", [])]
        # Expand {ENV_VAR_NAME} placeholders in env values from the process env
        # (e.g. {GITHUB_TOKEN}); missing vars become empty strings so the server
        # still launches and only auth-gated calls fail, never startup.
        raw_env = {str(k): str(v) for k, v in (details.get("env") or {}).items()}
        env = {}
        for k, v in raw_env.items():
            if v.startswith("{") and v.endswith("}"):
                var_name = v[1:-1]
                env[k] = os.environ.get(var_name, "")
            else:
                env[k] = v
        return cmd, args, env

    async def _start_one(self, conn: MCPServerConnection):
        """Start a single server with its own timeout. Failures never block others."""
        try:
            ok = await asyncio.wait_for(conn.start(), timeout=DEFAULT_START_TIMEOUT)
            if ok:
                self.servers[conn.name] = conn
                print(f"[MCP] Connected to '{conn.name}' ({len(conn.tools)} tools available)")
                return True
            print(f"[MCP] Server '{conn.name}' did not start (returned False).")
        except asyncio.TimeoutError:
            print(f"[MCP] Server '{conn.name}' timed out after {DEFAULT_START_TIMEOUT}s — skipped.")
        except (OSError, RuntimeError) as e:
            print(f"[MCP] Failed to start server '{conn.name}': {e}")
        await conn.stop()
        return False

    async def start_all(self):
        """Start every configured MCP server IN PARALLEL.

        Each server gets its own timeout and its failure is isolated — so 10+
        servers can come up at the same time and a single broken one never
        blocks the rest.
        """
        from .config import WORKSPACE_DIR
        cfg = self.load_config()
        servers = cfg.get("mcpServers", {})
        if not servers:
            return
        conns = []
        for name, details in servers.items():
            cmd, args, env = self._resolve_server_command(details, WORKSPACE_DIR)
            if cmd:
                conns.append(MCPServerConnection(name, cmd, args, env))
        if conns:
            print(f"[MCP] Starting {len(conns)} server(s) in parallel...")
            await asyncio.gather(*(self._start_one(c) for c in conns))
            print(f"[MCP] {len(self.servers)}/{len(conns)} server(s) connected.")

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Returns tools formatted for OpenAI LLM function calling"""
        formatted = []
        self._tool_map = {}
        for s_name, conn in self.servers.items():
            if not conn.is_connected:
                continue
            for tool in conn.tools:
                orig_name = tool.get("name", "")
                full_name = f"mcp_{s_name}_{orig_name}"
                self._tool_map[full_name] = (s_name, orig_name)
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": full_name,
                        "description": f"[MCP: {s_name}] {tool.get('description', '')}",
                        "parameters": tool.get("inputSchema", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                })
        return formatted

    def _resolve_tool_name(self, name: str) -> tuple[str, str] | None:
        """Reliably map a full tool name back to (server, original tool).

        Uses the exact lookup map first; falls back to parsing so names with
        underscores in server names still resolve.
        """
        hit = self._tool_map.get(name)
        if hit:
            return hit
        if not name.startswith("mcp_"):
            return None
        rest = name[4:]
        if "_" not in rest:
            return None
        s_name, orig = rest.split("_", 1)
        return s_name, orig

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        resolved = self._resolve_tool_name(name)
        if resolved is None:
            return f"Error: '{name}' is not an MCP tool."
        s_name, orig_name = resolved
        conn = self.servers.get(s_name)
        if conn is None:
            # The lookup map may be missing/stale (e.g. execute_tool called
            # before get_all_tools). Rebuild it from live servers and retry —
            # this also keeps names like "mcp_good_1_echo" resolvable when the
            # server name itself contains underscores.
            self._tool_map = {}
            _ = self.get_all_tools()
            resolved = self._resolve_tool_name(name)
            if resolved is None:
                return f"Error: '{name}' is not an MCP tool."
            s_name, orig_name = resolved
            conn = self.servers.get(s_name)
            if conn is None:
                return f"Error: MCP Server '{s_name}' is not connected."

        # Auto-reconnect once if the server dropped (seamless operation)
        if not conn.is_connected:
            try:
                print(f"[MCP] Server '{s_name}' disconnected — trying to restart...")
                ok = await asyncio.wait_for(conn.start(), timeout=DEFAULT_START_TIMEOUT)
                if not ok:
                    return f"Error: MCP Server '{s_name}' failed to restart."
                self._tool_map = {}
                _ = self.get_all_tools()
            except (OSError, RuntimeError, asyncio.TimeoutError) as e:
                return f"Error: MCP Server '{s_name}' could not restart: {e!s}"
        try:
            return await conn.call_tool(orig_name, arguments)
        except (OSError, RuntimeError) as e:
            hint = conn.recent_stderr[:400]
            if hint:
                return f"Error executing MCP tool '{orig_name}' on '{s_name}': {e!s}\nServer stderr (tail):\n{hint}"
            return f"Error executing MCP tool '{orig_name}' on '{s_name}': {e!s}"

    async def stop_all(self):
        conns = list(self.servers.values())
        if conns:
            await asyncio.gather(*(conn.stop() for conn in conns), return_exceptions=True)
        self.servers.clear()
        self._tool_map = {}

    async def connect_preset(
        self,
        preset_id: str,
        server_name: str | None = None,
        env_overrides: dict[str, str] | None = None,
        custom_args: list[str] | None = None,
        workspace_dir: Path | None = None,
    ) -> tuple[bool, str]:
        """Dynamically configures and hot-connects an MCP preset in one call."""
        from titan_agent.core.mcp.presets import MCPPresetManager
        mgr = MCPPresetManager(self.config_file)
        if env_overrides:
            for key, value in env_overrides.items():
                value = str(value).strip()
                if not re.fullmatch(r"\{[A-Za-z_][A-Za-z0-9_]*\}", value):
                    return False, (
                        f"Refusing inline value for MCP environment field '{key}'. "
                        "Store the value in the process environment and pass only a reference such as '{POSTGRES_URL}'."
                    )
                env_name = value[1:-1]
                if not os.environ.get(env_name):
                    return False, f"Required process environment variable '{env_name}' is not set."
        runtime_cfg, err = mgr.generate_server_config(preset_id, env_overrides, custom_args)
        if not runtime_cfg:
            return False, err
        # Persist only the preset template. env_overrides may contain tokens,
        # database URLs or other secrets, so they must never be written to the
        # shared MCP JSON config. Runtime values are used only for this process.
        persistent_cfg, persist_err = mgr.generate_server_config(preset_id)
        if not persistent_cfg:
            return False, persist_err

        name = server_name or preset_id
        persisted = mgr.save_server_to_config(name, persistent_cfg)

        ws = workspace_dir or (self.config_file.parent if self.config_file else Path.cwd())
        cmd, args, env = self._resolve_server_command(runtime_cfg, ws)
        conn = MCPServerConnection(name, cmd, args, env)
        ok = await self._start_one(conn)
        if ok:
            message = f"Successfully connected MCP server '{name}' with {len(conn.tools)} tools."
            if not persisted:
                message += " Runtime connection is active, but its template could not be saved for restart."
            else:
                message += " Credentials were not saved; configure them in the process environment for future starts."
            return True, message
        return False, f"Failed to start MCP server '{name}'. Check logs/dependencies."