# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Message Router
# Implements FRD-001, FRD-014

"""
JSON-RPC message parsing, method-based routing, and response correlation.

Reads newline-delimited JSON from host stdin, classifies messages by
``method``, routes ``tools/call`` requests to the PreventionStack,
and passes all other messages through unchanged.

Response correlation:
    When a ``tools/call`` is forwarded, a ``PendingCall`` entry is stored
    so that the server response can be matched and its cost accumulated.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from clyro.config import WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.errors import format_error
from clyro.mcp.log import get_logger
from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
from clyro.mcp.session import McpSession, PendingCall
from clyro.mcp.transport import StdioTransport

logger = get_logger(__name__)

# Maximum line length we'll attempt to parse (10 MB guard — TDD §8.1)
_MAX_LINE_BYTES = 10 * 1024 * 1024


class _FramingError(Exception):
    """Raised when LSP Content-Length framing is detected (not supported in v1.0)."""

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(f"LSP framing detected from {source}")


class MessageRouter:
    """
    Coordinates the host -> wrapper -> server message flow.

    Owns the asyncio tasks for reading host stdin, reading server stdout,
    forwarding server stderr, and monitoring the child process.
    """

    def __init__(
        self,
        config: WrapperConfig,
        session: McpSession,
        transport: StdioTransport,
        prevention: PreventionStack,
        audit: AuditLogger,
    ) -> None:
        self._config = config
        self._session = session
        self._transport = transport
        self._prevention = prevention
        self._audit = audit
        self._pending_requests: dict[str | int, PendingCall] = {}
        self._shutdown_event = asyncio.Event()
        self._first_message_checked = False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> int:
        """
        Start all I/O tasks and wait until shutdown.

        Returns:
            Exit code (0 normal, 2 server crash, 3 zombie).
        """
        tasks = [
            asyncio.create_task(self._host_reader_task(), name="host_reader"),
            asyncio.create_task(self._server_reader_task(), name="server_reader"),
            asyncio.create_task(self._stderr_forwarder_task(), name="stderr_fwd"),
            asyncio.create_task(self._process_monitor_task(), name="proc_monitor"),
        ]

        # Wait for any task to finish (usually process_monitor or host EOF)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except asyncio.CancelledError:
                pass

        # Determine exit code from the completed task
        for t in done:
            exc = t.exception()
            if exc is not None:
                # HIGH-1 fix: propagate _FramingError as exit code 1
                if isinstance(exc, _FramingError):
                    return 1
                # MEDIUM-2 fix: log unexpected task exceptions instead of swallowing
                logger.error(
                    "task_failed",
                    task=t.get_name(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                continue
            result = t.result()
            if isinstance(result, int):
                return result
        return 0

    # ------------------------------------------------------------------
    # Host -> Server (with governance)
    # ------------------------------------------------------------------

    async def _host_reader_task(self) -> None:
        """Read host stdin, evaluate tools/call, forward or block."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        while not self._shutdown_event.is_set():
            line = await reader.readline()
            if not line:
                # Host closed stdin — initiate shutdown
                self._shutdown_event.set()
                return

            # Defensive framing check (TDD §2.2)
            if not self._first_message_checked:
                self._first_message_checked = True
                if line.strip().lower().startswith(b"content-length:"):
                    logger.error(
                        "lsp_framing_detected",
                        source="host",
                        hint="MCP server may require header-based framing",
                    )
                    raise _FramingError("host")

            # Oversized lines: log warning and forward raw bytes unchanged
            # (FRD-001: forward unparseable data as-is, do not truncate)
            if len(line) > _MAX_LINE_BYTES:
                logger.warning(
                    "oversized_message",
                    size_bytes=len(line),
                    action="forwarding_raw",
                )
                self._audit.log_parse_error(line[:200])
                try:
                    await self._transport.write_to_child(line)
                except BrokenPipeError:
                    self._shutdown_event.set()
                    return
                continue

            try:
                await self._handle_host_message(line)
            except BrokenPipeError:
                # MEDIUM-3 fix: child died — trigger clean shutdown
                self._shutdown_event.set()
                return

    async def _handle_host_message(self, raw: bytes) -> None:
        """Parse and route a single host message."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Malformed JSON — log and forward raw (FRD-001)
            self._audit.log_parse_error(raw)
            await self._transport.write_to_child(raw)
            return

        # Batch JSON-RPC (array) — passthrough as-is.
        # Checked before dict-specific checks to avoid type errors.
        if isinstance(msg, list):
            logger.warning("jsonrpc_batch_unsupported", action="forwarding_raw")
            await self._transport.write_to_child(raw)
            return

        # Notifications (no id) — always passthrough
        if "id" not in msg:
            await self._transport.write_to_child(raw)
            return

        method = msg.get("method", "")

        # Only govern tools/call
        if method != "tools/call":
            await self._transport.write_to_child(raw)
            return

        # Extract tool name and arguments
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments")
        request_id = msg.get("id")

        start = time.monotonic()
        decision = self._prevention.evaluate(tool_name, arguments, self._session)
        duration_ms = int((time.monotonic() - start) * 1000)

        if isinstance(decision, AllowDecision):
            # Forward to server
            params_json = json.dumps(arguments or {}, default=str)
            self._pending_requests[request_id] = PendingCall(
                request_id=request_id,
                tool_name=tool_name,
                params_json_len=len(params_json),
                forwarded_at=time.monotonic(),
            )
            await self._transport.write_to_child(raw)
            self._audit.log_tool_call(
                tool_name=tool_name,
                parameters=arguments,
                decision="allowed",
                step_number=decision.step_number,
                accumulated_cost_usd=self._session.accumulated_cost_usd,
                duration_ms=duration_ms,
                rule_results=decision.rule_results or None,
                request_id=request_id,
            )
        else:
            # Block — send error to host, never forward to server
            assert isinstance(decision, BlockDecision)
            error_line = format_error(request_id, decision.block_type, decision.details)
            sys.stdout.buffer.write(error_line.encode("utf-8"))
            sys.stdout.buffer.flush()
            # Extract rule_results from details (stored by PreventionStack)
            block_rule_results = decision.details.pop("_rule_results", None)
            self._audit.log_tool_call(
                tool_name=tool_name,
                parameters=arguments,
                decision="blocked",
                step_number=decision.step_number,
                accumulated_cost_usd=self._session.accumulated_cost_usd,
                block_reason=decision.block_type,
                block_details=decision.details,
                duration_ms=duration_ms,
                rule_results=block_rule_results,
            )

    # ------------------------------------------------------------------
    # Server -> Host (with cost correlation)
    # ------------------------------------------------------------------

    async def _server_reader_task(self) -> None:
        """Read server stdout, correlate responses, forward to host."""
        while not self._shutdown_event.is_set():
            line = await self._transport.read_line_from_child()
            if not line:
                # Server closed stdout
                self._shutdown_event.set()
                return

            # Defensive framing check on first server message
            if line.strip().lower().startswith(b"content-length:"):
                logger.error("lsp_framing_detected", source="server")
                raise _FramingError("server")

            # Try to correlate response with a pending tools/call
            try:
                msg = json.loads(line)
                resp_id = msg.get("id")
                # A response has 'id' but no 'method'
                is_response = resp_id is not None and "method" not in msg
                if is_response and resp_id in self._pending_requests:
                    pending = self._pending_requests.pop(resp_id)
                    # Compute response content length and accumulate cost
                    result_str = json.dumps(msg.get("result", ""), default=str)
                    cost = self._prevention.cost_tracker.accumulate(
                        pending.params_json_len, len(result_str)
                    )
                    self._session.add_cost(cost)
                    duration_ms = int((time.monotonic() - pending.forwarded_at) * 1000)
                    self._audit.log_tool_call_response(
                        tool_name=pending.tool_name,
                        request_id=pending.request_id,
                        call_cost_usd=cost,
                        accumulated_cost_usd=self._session.accumulated_cost_usd,
                        duration_ms=duration_ms,
                        response_content=result_str,
                    )
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Not JSON — forward as-is

            # Forward to host
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    # ------------------------------------------------------------------
    # Stderr forwarder
    # ------------------------------------------------------------------

    async def _stderr_forwarder_task(self) -> None:
        """Prefix child stderr with ``[server] `` and forward."""
        while not self._shutdown_event.is_set():
            line = await self._transport.read_stderr_line()
            if not line:
                return
            prefixed = b"[server] " + line
            sys.stderr.buffer.write(prefixed)
            sys.stderr.buffer.flush()

    # ------------------------------------------------------------------
    # Process monitor
    # ------------------------------------------------------------------

    async def _process_monitor_task(self) -> int:
        """Wait for child process exit (FRD-013)."""
        proc = self._transport.process
        if proc is None:
            return 0
        exit_code = await proc.wait()
        self._shutdown_event.set()

        self._audit.log_lifecycle(
            "server_exited",
            extra={"exit_code": exit_code},
        )

        return 2 if exit_code != 0 else 0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def request_shutdown(self) -> None:
        """Signal all tasks to stop."""
        self._shutdown_event.set()
