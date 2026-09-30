import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from titan_agent.tools import MAX_CAPTURE_BYTES_PER_STREAM, ToolRegistry


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(tmp_path)


class _FakeProcess:
    def __init__(self, stdout=b"", stderr=b"", *, running=False):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self._done = asyncio.Event()
        self.wait_started = asyncio.Event()
        self.returncode = None if running else 0
        if stdout:
            self.stdout.feed_data(stdout)
        if stderr:
            self.stderr.feed_data(stderr)
        if not running:
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self._done.set()

    async def wait(self):
        self.wait_started.set()
        await self._done.wait()
        return self.returncode

    def kill(self):
        if not self._done.is_set():
            self.returncode = -9
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self._done.set()

    async def communicate(self):
        return b"", b""


def test_docker_sandbox_in_tool_definitions(registry):
    defs = registry.get_tool_definitions()
    names = [d["function"]["name"] for d in defs]
    assert "docker_sandbox_run" in names
    tool_def = next(d["function"] for d in defs if d["function"]["name"] == "docker_sandbox_run")
    assert "command" in tool_def["parameters"]["required"]
    props = tool_def["parameters"]["properties"]
    assert "image" in props
    assert "memory_limit" in props
    assert "mount_workspace" in props
    assert "network" in props


@pytest.mark.asyncio
async def test_docker_sandbox_empty_command(registry):
    res = await registry.tool_docker_sandbox_run("")
    assert "Error: command is required" in res


@pytest.mark.asyncio
async def test_docker_sandbox_no_docker(registry):
    with patch("shutil.which", return_value=None):
        res = await registry.tool_docker_sandbox_run("echo hello")
        assert "Docker is required" in res
        assert "no host fallback" in res


@pytest.mark.asyncio
async def test_docker_sandbox_success(registry):
    mock_proc = _FakeProcess(stdout=b"hello sandbox\n")

    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        res = await registry.tool_docker_sandbox_run(
            "echo hello sandbox",
            image="alpine:latest",
            memory_limit="256m",
            cpu_quota="0.5",
            mount_workspace=True,
            network="none",
        )
        assert "### DOCKER SANDBOX [alpine:latest] (Exit 0)" in res
        assert "STDOUT:\nhello sandbox" in res

        # Verify command flags passed to docker
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "/usr/bin/docker"
        assert "run" in args
        assert "--pull=never" in args
        assert "--name" in args
        assert args[args.index("--name") + 1].startswith("titan-sandbox-")
        assert "--memory=256m" in args
        assert "--cpus=0.5" in args
        assert "--network=none" in args
        assert "--workdir" in args
        assert "/workspace" in args[args.index("--mount") + 1]
        assert "--read-only" in args
        assert "--cap-drop=ALL" in args
        assert "--security-opt=no-new-privileges" in args
        assert "--pids-limit=128" in args
        assert "alpine:latest" in args


@pytest.mark.asyncio
async def test_docker_sandbox_timeout(registry):
    mock_proc = _FakeProcess(running=True)
    cleanup_proc = _FakeProcess()

    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch(
             "asyncio.create_subprocess_exec",
             side_effect=[mock_proc, cleanup_proc],
         ) as mock_exec:
        res = await registry.tool_docker_sandbox_run("sleep 100", timeout=1.0)
        assert "timed out after 1s" in res
        assert mock_exec.await_count == 2
        cleanup_args = mock_exec.await_args_list[1].args
        assert cleanup_args[:3] == ("/usr/bin/docker", "rm", "-f")
        assert cleanup_args[3].startswith("titan-sandbox-")


@pytest.mark.asyncio
async def test_docker_sandbox_caps_output_and_attempts_cleanup(registry):
    noisy_proc = _FakeProcess(stdout=b"x" * (MAX_CAPTURE_BYTES_PER_STREAM + 64))
    cleanup_proc = _FakeProcess()

    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch(
             "asyncio.create_subprocess_exec",
             side_effect=[noisy_proc, cleanup_proc],
         ) as mock_exec:
        result = await registry.tool_docker_sandbox_run("yes x")

    assert "(Exit 125)" in result  # truncated output cannot count as successful verification
    assert f"Output truncated at {MAX_CAPTURE_BYTES_PER_STREAM} bytes per stream" in result
    assert len(result) < MAX_CAPTURE_BYTES_PER_STREAM + 2_000
    assert mock_exec.await_count == 2
    assert mock_exec.await_args_list[1].args[1:3] == ("rm", "-f")


@pytest.mark.asyncio
async def test_docker_sandbox_cancellation_attempts_container_cleanup(registry):
    docker_proc = _FakeProcess(running=True)
    cleanup_proc = _FakeProcess()

    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch(
             "asyncio.create_subprocess_exec",
             side_effect=[docker_proc, cleanup_proc],
         ) as mock_exec:
        task = asyncio.create_task(
            registry.tool_docker_sandbox_run("sleep 100", timeout=30.0)
        )
        await docker_proc.wait_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert mock_exec.await_count == 2
    assert mock_exec.await_args_list[1].args[1:3] == ("rm", "-f")


@pytest.mark.asyncio
async def test_docker_sandbox_error(registry):
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("asyncio.create_subprocess_exec", side_effect=OSError("Docker daemon down")):
        res = await registry.tool_docker_sandbox_run("echo fail")
        assert "Docker execution error: Docker daemon down" in res
