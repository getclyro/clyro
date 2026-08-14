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
from clyro.dry_run import WouldBlockLatch, log_would_block
from clyro.mcp.audit import AuditLogger
from clyro.mcp.errors import format_error, format_transport_error
from clyro.mcp.log import get_logger
from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
from clyro.mcp.server_transport import ServerTransport, SessionExpired, TransportError
from clyro.mcp.session import McpSession, PendingCall
from clyro.mcp.transport import StdioTransport

logger = get_logger(__name__)

# Maximum line length we'll attempt to parse (10 MB guard — TDD §8.1)
_MAX_LINE_BYTES = 10 * 1024 * 1024

# A10: map an MCP block_type → the four canonical check types (FRD-008/011).
_BLOCK_TYPE_TO_CHECK: dict[str, str] = {
    "loop_detected": "loop",
    "step_limit_exceeded": "step",
    "budget_exceeded": "cost",
    "policy_violation": "policy",
}


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
        transport: ServerTransport,
        prevention: PreventionStack,
        audit: AuditLogger,
        dry_run: bool | None = None,
    ) -> None:
        self._config = config
        self._session = session
        self._transport = transport
        self._prevention = prevention
        self._audit = audit
        self._pending_requests: dict[str | int, PendingCall] = {}
        self._shutdown_event = asyncio.Event()
        self._first_message_checked = False
        # Id of a non-tools/call request (e.g. initialize) currently being
        # forwarded. tools/call ids live in _pending_requests; this covers the
        # passthrough-request case so a mid-exchange transport failure can still
        # answer that id (FRD-020). Set around the forward, cleared on success.
        self._inflight_host_id: str | int | None = None

        # A10: the resolved dry-run mode. The CLI passes the CLI>env>config
        # precedence result explicitly; fall back to config (env>config) if not
        # given. Implements FRD-001/002/O-4.
        self._dry_run = dry_run if dry_run is not None else config.resolved_is_dry_run
        # A10 FRD-022: one would-block marker per distinct reason per session.
        self._wb_latch = WouldBlockLatch()

    def _would_block_key(
        self,
        check_type: str,
        rule_id: str | None,
        tool_name: str,
        decision: BlockDecision,
    ) -> str:
        """Build the de-dup key for an MCP would-block. Implements FRD-022.

        Mirrors the SDK's semantics so all three surfaces behave identically
        (NFR-002):
        - **step / cost** are *sticky* (once tripped, every later call trips) →
          one marker per session.
        - **loop** → one per distinct loop signature.
        - **policy** is per-action → one per (rule, tool).
        """
        session = self._session.session_id
        if check_type in ("step", "cost"):
            return f"{session}:{check_type}"
        if check_type == "loop":
            signature = (decision.details or {}).get("pattern_hash", "loop")
            return f"{session}:loop:{signature}"
        return WouldBlockLatch.policy_key(session, rule_id, tool_name, "block")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> int:
        """
        Start all I/O tasks and wait until shutdown.

        Returns:
            Exit code (0 normal, 2 server crash, 3 zombie).
        """
        # host/server readers are transport-blind (they use send/receive).
        tasks = [
            asyncio.create_task(self._host_reader_task(), name="host_reader"),
            asyncio.create_task(self._server_reader_task(), name="server_reader"),
        ]
        # stderr forwarding and child-exit monitoring are stdio-only (an HTTP
        # connection has neither a child stderr nor a process). The HTTP leg's
        # end-of-connection is detected by ``receive()`` returning None
        # (_server_reader_task), which settles in-flight calls (FRD-026).
        if isinstance(self._transport, StdioTransport):
            tasks.append(asyncio.create_task(self._stderr_forwarder_task(), name="stderr_fwd"))
            tasks.append(asyncio.create_task(self._process_monitor_task(), name="proc_monitor"))

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

        # FRD-026 / AC-4.2 / AC-4.3: settle anything STILL in flight as the
        # session ends. The connection-lost and transport-error paths already
        # settle, but a *normal* shutdown (host closed stdin) reaches neither —
        # a call sent but never answered would otherwise vanish: charged zero
        # (a silent under-count, the D9 violation) with its outcome absent from
        # the audit. No-ops when nothing is outstanding, so it is safe here for
        # every exit path.
        self._settle_pending()

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
                    await self._transport.send(line)
                except BrokenPipeError:
                    self._shutdown_event.set()
                    return
                except TransportError as exc:
                    # FRD-020/031: HTTP send lost the connection — tell the host.
                    self._fail_pending_transport(str(exc))
                    return
                continue

            try:
                await self._handle_host_message(line)
            except BrokenPipeError:
                # MEDIUM-3 fix: child died — trigger clean shutdown
                self._shutdown_event.set()
                return
            except SessionExpired as exc:
                # FRD-025: the server forgot our session and the transport could
                # not re-establish it. The CONNECTION is still healthy, so ending
                # the host's session here would turn a recoverable hiccup into a
                # disconnect (R-C2). Fail only the in-flight call and keep serving
                # — the host's own re-initialize can still recover the session.
                self._fail_pending_transport(str(exc))
                continue
            except TransportError as exc:
                # FRD-020/031: an HTTP send() failed mid-exchange. stdio signals
                # this via BrokenPipeError above; the HTTP transport raises
                # TransportError, which nothing else would catch — leaving the
                # host hung on the in-flight tools/call id. Surface it as a
                # transport error to the host and end the session.
                self._fail_pending_transport(str(exc))
                return

    async def _handle_host_message(self, raw: bytes) -> None:
        """Parse and route a single host message."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Malformed JSON — log and forward raw (FRD-001)
            self._audit.log_parse_error(raw)
            await self._transport.send(raw)
            return

        # Batch JSON-RPC (array) — passthrough as-is.
        # Checked before dict-specific checks to avoid type errors.
        if isinstance(msg, list):
            logger.warning("jsonrpc_batch_unsupported", action="forwarding_raw")
            await self._transport.send(raw)
            return

        # Notifications (no id) — always passthrough
        if "id" not in msg:
            await self._transport.send(raw)
            return

        method = msg.get("method", "")

        # Only govern tools/call
        if method != "tools/call":
            # Passthrough request (e.g. initialize): remember its id so a send
            # that fails mid-exchange can answer it with a transport error too.
            self._inflight_host_id = msg.get("id")
            await self._transport.send(raw)
            self._inflight_host_id = None
            return

        # Extract tool name and arguments
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments")
        request_id = msg.get("id")

        # FRD-048 critical section: assign-step + evaluate-against-counters +
        # update-counters run atomically inside PreventionStack.evaluate, which
        # is synchronous (no await). Because this router has exactly one host
        # reader task, that whole govern step is serialized — concurrent host
        # calls cannot read a stale counter. The network round-trip below (the
        # awaited send) is the only concurrent part and holds no counter state.
        start = time.monotonic()
        decision = self._prevention.evaluate(tool_name, arguments, self._session)
        duration_ms = int((time.monotonic() - start) * 1000)

        if isinstance(decision, AllowDecision):
            # Forward to server. Record the allow decision and register the
            # pending call *before* the send, so a send that fails mid-exchange
            # (FRD-020) still leaves a coherent audit trail: the allowed record
            # plus the "[unresolved: connection lost]" response from settle,
            # rather than a response with no matching call.
            params_json = json.dumps(arguments or {}, default=str)
            # V-24 / D19: an id colliding with an outstanding call must NOT
            # silently discard the earlier entry's accounting. That call was
            # already forwarded and executed; a blind overwrite destroys only its
            # cost record, so N calls run while 1 is billed and the FRD-006 budget
            # cap can never fire. Settle the displaced call at its pre-call
            # estimate first — it is never re-sent (FRD-045), only accounted for.
            if request_id in self._pending_requests:
                logger.warning(
                    "outstanding_id_collision",
                    request_id=request_id,
                    hint="settling the displaced call; host reused an in-flight id",
                )
                self._settle_one(request_id)
            self._pending_requests[request_id] = PendingCall(
                request_id=request_id,
                tool_name=tool_name,
                params_json_len=len(params_json),
                forwarded_at=time.monotonic(),
            )
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
            await self._transport.send(raw)
        else:
            assert isinstance(decision, BlockDecision)
            # Extract rule_results from details (stored by PreventionStack)
            block_rule_results = decision.details.pop("_rule_results", None)

            if self._dry_run and not decision.absolute:
                # A10 (FRD-004): forward to the server instead of returning a
                # JSON-RPC block error; audit a would-block (which emits the
                # distinct would_block event and NO enforced error sibling —
                # FRD-017). Implements FRD-004/017.
                #
                # FRD-021 exception: an absolute-ceiling block (decision.absolute)
                # is NOT forwarded even in dry_run — it falls through to the hard
                # block path below, returning a real JSON-RPC error to the host so
                # a genuine runaway is stopped on this surface too.
                params_json = json.dumps(arguments or {}, default=str)
                self._pending_requests[request_id] = PendingCall(
                    request_id=request_id,
                    tool_name=tool_name,
                    params_json_len=len(params_json),
                    forwarded_at=time.monotonic(),
                )

                # A10 FRD-022: the prevention stack re-evaluates EVERY tools/call,
                # so without a latch a tripped limit (sticky) or a policy rule
                # would emit one marker per call — unbounded, burning the org's
                # trace quota. Record one marker per distinct reason per session,
                # mirroring the SDK's latch (NFR-002 cross-surface consistency).
                check_type = _BLOCK_TYPE_TO_CHECK.get(decision.block_type, "policy")
                rule_id = decision.details.get("policy_id") if decision.details else None
                first = self._wb_latch.record(
                    self._would_block_key(check_type, rule_id, tool_name, decision)
                )

                self._audit.log_tool_call(
                    tool_name=tool_name,
                    parameters=arguments,
                    decision="would_block",
                    step_number=decision.step_number,
                    accumulated_cost_usd=self._session.accumulated_cost_usd,
                    block_reason=decision.block_type,
                    block_details=decision.details,
                    duration_ms=duration_ms,
                    rule_results=block_rule_results,
                    request_id=request_id,
                    # Repeats stay in the local JSONL audit (a per-call record) but
                    # emit no further backend marker or terminal line.
                    emit_marker=first,
                )

                # Forward AFTER recording the would-block + audit, mirroring the
                # allow branch's audit-before-send ordering (FRD-020): a send that
                # fails mid-exchange then still leaves a coherent trail — the
                # would_block marker plus the "[unresolved]" settle response — not
                # an orphan response for a call whose would-block was never recorded.
                # Use the ServerTransport protocol method, not the stdio-only
                # `write_to_child`: A11's HttpTransport implements `send` and has no
                # `write_to_child`, so the stdio-era call raised AttributeError and
                # dry-run could not forward over HTTP. On stdio `send` is a thin
                # alias for `write_to_child`, so behaviour is unchanged there.
                await self._transport.send(raw)
                if first:
                    log_would_block(check_type, tool_name, "block", rule_id)
                return

            # Block — send error to host, never forward to server
            error_line = format_error(request_id, decision.block_type, decision.details)
            sys.stdout.buffer.write(error_line.encode("utf-8"))
            sys.stdout.buffer.flush()
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

    def _fail_pending_transport(self, reason: str) -> None:
        """Report a transport failure to the host and end the session.

        The host path's ``send()`` raising a ``TransportError`` means the
        downstream connection was lost mid-exchange. Unlike stdio (which
        signals this with ``BrokenPipeError`` and settles via the server
        reader's ``receive()==None`` path), an HTTP ``send()`` failure surfaces
        as a ``TransportError`` that no reader would otherwise catch — leaving
        the host waiting forever on an in-flight ``tools/call`` id.

        For each in-flight request we write a JSON-RPC **transport** error to
        the host, distinguishable from a governance block (FRD-020, FRD-031),
        then settle those calls' cost (FRD-026/045 — never re-sent) and end the
        session so the wrapper does not exit silently.
        """
        ids: list[str | int] = list(self._pending_requests)
        # A passthrough request (initialize, etc.) in flight is not tracked in
        # _pending_requests but still awaits a response — include its id.
        if (
            self._inflight_host_id is not None
            and self._inflight_host_id not in self._pending_requests
        ):
            ids.append(self._inflight_host_id)
        for req_id in ids:
            error_line = format_transport_error(req_id, reason)
            sys.stdout.buffer.write(error_line.encode("utf-8"))
        sys.stdout.buffer.flush()
        # Audit + charge the tracked tool calls as unresolved, then shut down.
        self._settle_pending()
        self._inflight_host_id = None
        self._shutdown_event.set()

    def _settle_pending(self) -> None:
        """Settle in-flight calls when the connection is lost (FRD-026/045).

        Each unresolved call is charged a pre-call estimate from its already-
        captured request length (reuses the existing cost accumulator — no new
        methodology, scope §3) so a dropped call is never counted as zero
        (D9). Calls are never re-sent (FRD-045); they are settled and audited
        as unresolved.
        """
        for req_id in list(self._pending_requests):
            self._settle_one(req_id)

    def _settle_one(self, req_id: str | int) -> None:
        """Settle a single in-flight call at its pre-call estimate (FRD-026/D17).

        Charged at the request-side estimate, recorded as unresolved, and never
        re-sent (FRD-045). Used both when the session ends with calls in flight
        and when an outstanding id is displaced by a collision (V-24).
        """
        pending = self._pending_requests.pop(req_id, None)
        if pending is None:
            return
        # Pre-call estimate: request length on both sides (response unknown).
        est = self._prevention.cost_tracker.accumulate(
            pending.params_json_len, pending.params_json_len
        )
        self._session.add_cost(est)
        self._audit.log_tool_call_response(
            tool_name=pending.tool_name,
            request_id=pending.request_id,
            call_cost_usd=est,
            accumulated_cost_usd=self._session.accumulated_cost_usd,
            duration_ms=0,
            # State the fact, never a cause. Settlement runs on every exit —
            # including a clean shutdown with a call still in flight, where
            # nothing was "lost". Asserting a cause we did not observe put a
            # false diagnosis in the backend trace and pointed operators at the
            # network for a wrapper-side outcome.
            response_content="[unresolved: no response received]",
            unresolved=True,  # FRD-026: recorded as unresolved, not as a completed call
        )

    # ------------------------------------------------------------------
    # Server -> Host (with cost correlation)
    # ------------------------------------------------------------------

    async def _server_reader_task(self) -> None:
        """Read server stdout, correlate responses, forward to host."""
        while not self._shutdown_event.is_set():
            line = await self._transport.receive()
            if not line:
                # Connection ended (child stdout EOF, or HTTP connection lost).
                # Settle any in-flight calls before shutting down (FRD-026/045).
                self._settle_pending()
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
                # A response has 'id' but no 'method'. A server-initiated request
                # has BOTH id and method — it is not a response, so it falls
                # through to the passthrough below and is delivered to the host
                # (FRD-027); the host's reply returns via the host reader and is
                # forwarded to the server (FRD-028). An uncorrelatable response
                # (id not in pending) skips the cost block below (FRD-053).
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
