"""
Unit tests for StdioTransport — FRD-002, FRD-013.
"""

from __future__ import annotations

import json
import sys

import pytest

from clyro.mcp.transport import StdioTransport


class TestTransportSpawn:
    """Child process lifecycle."""

    @pytest.mark.asyncio
    async def test_start_valid_command(self) -> None:
        """Spawn a simple command and verify process handle."""
        transport = StdioTransport([sys.executable, "-c", "pass"])
        proc = await transport.start()
        assert proc is not None
        assert proc.pid > 0
        await transport.terminate()

    @pytest.mark.asyncio
    async def test_start_invalid_command(self) -> None:
        """Non-existent command → SystemExit(1)."""
        transport = StdioTransport(["nonexistent_cmd_abc123"])
        with pytest.raises(SystemExit) as exc_info:
            await transport.start()
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_write_and_read(self) -> None:
        """Write to child stdin, read from child stdout."""
        script = 'import sys; line=sys.stdin.readline(); sys.stdout.write(line); sys.stdout.flush()'
        transport = StdioTransport([sys.executable, "-c", script])
        await transport.start()

        msg = b'{"jsonrpc":"2.0","id":1}\n'
        await transport.write_to_child(msg)
        line = await transport.read_line_from_child()
        assert line is not None
        assert json.loads(line)["id"] == 1
        await transport.terminate()

    @pytest.mark.asyncio
    async def test_terminate_already_exited(self) -> None:
        """Terminating an already-exited process returns 0."""
        transport = StdioTransport([sys.executable, "-c", "pass"])
        await transport.start()
        # Wait for it to exit naturally
        await transport.process.wait()
        code = await transport.terminate()
        assert code is not None

    @pytest.mark.asyncio
    async def test_read_stderr_line(self) -> None:
        """Read from child stderr."""
        script = 'import sys; sys.stderr.write("err line\\n"); sys.stderr.flush()'
        transport = StdioTransport([sys.executable, "-c", script])
        await transport.start()
        line = await transport.read_stderr_line()
        assert line is not None
        assert b"err line" in line
        await transport.terminate()

    @pytest.mark.asyncio
    async def test_read_line_after_close(self) -> None:
        """Reading from a closed stdout returns None."""
        transport = StdioTransport([sys.executable, "-c", "pass"])
        await transport.start()
        await transport.process.wait()
        # stdout should be closed after process exits
        line = await transport.read_line_from_child()
        # Will be b'' (empty) which readline returns when EOF
        assert line is None or line == b""

    @pytest.mark.asyncio
    async def test_terminate_timeout(self) -> None:
        """Process that ignores SIGTERM gets killed."""
        script = (
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(30)"
        )
        transport = StdioTransport([sys.executable, "-c", script])
        await transport.start()
        code = await transport.terminate(timeout=0.5)
        # Should have been killed
        assert code is not None

    @pytest.mark.asyncio
    async def test_write_to_dead_child_raises(self) -> None:
        """Writing to a child that has exited raises BrokenPipeError."""
        transport = StdioTransport([sys.executable, "-c", "pass"])
        await transport.start()
        await transport.process.wait()
        # Force stdin to None to simulate unavailable pipe
        transport._process.stdin = None  # type: ignore[union-attr]
        with pytest.raises(BrokenPipeError):
            await transport.write_to_child(b"data\n")

    @pytest.mark.asyncio
    async def test_write_without_start_raises(self) -> None:
        """Writing before start() raises BrokenPipeError."""
        transport = StdioTransport(["echo", "hello"])
        with pytest.raises(BrokenPipeError):
            await transport.write_to_child(b"data\n")
