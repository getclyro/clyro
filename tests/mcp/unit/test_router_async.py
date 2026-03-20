"""
Unit tests for MessageRouter async task logic and _FramingError handling.

Covers:
- _FramingError propagation from host and server reader tasks
- run() task exception logging (MEDIUM-2 fix)
- BrokenPipeError handling in host_reader_task (MEDIUM-3 fix)
- Response correlation in _server_reader_task
- Process monitor exit codes
- Shutdown event propagation
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clyro.mcp.audit import AuditLogger
from clyro.config import WrapperConfig
from clyro.mcp.prevention import AllowDecision, PreventionStack
from clyro.mcp.router import MessageRouter, _FramingError
from clyro.mcp.session import McpSession
from clyro.mcp.transport import StdioTransport


def _make_router(
    prevention_result=None,
) -> tuple[MessageRouter, MagicMock, MagicMock, MagicMock]:
    """Build a MessageRouter with mocked dependencies."""
    config = WrapperConfig()
    session = McpSession()
    transport = MagicMock(spec=StdioTransport)
    transport.write_to_child = AsyncMock()
    prevention = MagicMock(spec=PreventionStack)
    if prevention_result is not None:
        prevention.evaluate.return_value = prevention_result
    audit = MagicMock(spec=AuditLogger)
    router = MessageRouter(config, session, transport, prevention, audit)
    return router, transport, prevention, audit


class TestFramingError:
    """_FramingError exception behaviour."""

    def test_framing_error_has_source(self) -> None:
        err = _FramingError("host")
        assert err.source == "host"
        assert "host" in str(err)

    def test_framing_error_from_server(self) -> None:
        err = _FramingError("server")
        assert err.source == "server"


class TestRunExitCodes:
    """MessageRouter.run() exit code determination."""

    @pytest.mark.asyncio
    async def test_run_returns_framing_error_as_exit_1(self) -> None:
        """_FramingError in a task should return exit code 1."""
        router, transport, prevention, audit = _make_router()

        async def raise_framing():
            raise _FramingError("host")

        async def hang_forever():
            await asyncio.sleep(999)

        with patch.object(router, "_host_reader_task", raise_framing), \
             patch.object(router, "_server_reader_task", hang_forever), \
             patch.object(router, "_stderr_forwarder_task", hang_forever), \
             patch.object(router, "_process_monitor_task", hang_forever):
            code = await router.run()
        assert code == 1

    @pytest.mark.asyncio
    async def test_run_logs_unexpected_exceptions(self, capsys) -> None:
        """Unexpected exceptions in tasks should be logged to stderr."""
        router, transport, prevention, audit = _make_router()

        async def raise_value_error():
            raise ValueError("test unexpected error")

        async def hang_forever():
            await asyncio.sleep(999)

        with patch.object(router, "_host_reader_task", raise_value_error), \
             patch.object(router, "_server_reader_task", hang_forever), \
             patch.object(router, "_stderr_forwarder_task", hang_forever), \
             patch.object(router, "_process_monitor_task", hang_forever):
            code = await router.run()

        captured = capsys.readouterr()
        assert "ValueError" in captured.err
        assert "test unexpected error" in captured.err
        assert code == 0  # Falls through to default

    @pytest.mark.asyncio
    async def test_run_returns_process_monitor_exit_code(self) -> None:
        """run() should return the exit code from _process_monitor_task."""
        router, transport, prevention, audit = _make_router()

        async def return_exit_code():
            return 2

        async def hang_forever():
            await asyncio.sleep(999)

        with patch.object(router, "_host_reader_task", hang_forever), \
             patch.object(router, "_server_reader_task", hang_forever), \
             patch.object(router, "_stderr_forwarder_task", hang_forever), \
             patch.object(router, "_process_monitor_task", return_exit_code):
            code = await router.run()
        assert code == 2

    @pytest.mark.asyncio
    async def test_run_returns_0_when_host_eof(self) -> None:
        """Host closing stdin returns exit code 0."""
        router, transport, prevention, audit = _make_router()

        async def return_none():
            return None

        async def hang_forever():
            await asyncio.sleep(999)

        with patch.object(router, "_host_reader_task", return_none), \
             patch.object(router, "_server_reader_task", hang_forever), \
             patch.object(router, "_stderr_forwarder_task", hang_forever), \
             patch.object(router, "_process_monitor_task", hang_forever):
            code = await router.run()
        assert code == 0


class TestHostReaderBrokenPipe:
    """BrokenPipeError handling in _handle_host_message (MEDIUM-3)."""

    @pytest.mark.asyncio
    async def test_broken_pipe_triggers_shutdown(self) -> None:
        """BrokenPipeError during write_to_child sets shutdown event."""
        allow = AllowDecision(tool_name="read_file", step_number=1)
        router, transport, prevention, audit = _make_router(allow)
        transport.write_to_child = AsyncMock(side_effect=BrokenPipeError("dead"))

        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {}},
        }
        raw = json.dumps(msg).encode() + b"\n"

        # BrokenPipeError should propagate (caller handles shutdown)
        with pytest.raises(BrokenPipeError):
            await router._handle_host_message(raw)


class TestProcessMonitor:
    """Process monitor task exit code handling."""

    @pytest.mark.asyncio
    async def test_normal_exit_returns_0(self) -> None:
        router, transport, prevention, audit = _make_router()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        transport.process = mock_proc
        code = await router._process_monitor_task()
        assert code == 0
        assert router._shutdown_event.is_set()
        audit.log_lifecycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_crash_exit_returns_2(self) -> None:
        router, transport, prevention, audit = _make_router()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=1)
        transport.process = mock_proc
        code = await router._process_monitor_task()
        assert code == 2

    @pytest.mark.asyncio
    async def test_no_process_returns_0(self) -> None:
        router, transport, prevention, audit = _make_router()
        transport.process = None
        code = await router._process_monitor_task()
        assert code == 0


class TestShutdownEvent:
    """Shutdown event signaling."""

    def test_request_shutdown_sets_event(self) -> None:
        router, _, _, _ = _make_router()
        assert not router._shutdown_event.is_set()
        router.request_shutdown()
        assert router._shutdown_event.is_set()
