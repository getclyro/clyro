# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the HTTP mid-exchange transport-drop defect —
FRD-020 (surface a transport error to the host rather than exiting silently)
and FRD-031 (a transport failure must be distinguishable from a policy block).

Before the fix, an HTTP ``send()`` that raised ``TransportError`` was caught by
no reader — the host reader only guarded ``BrokenPipeError`` (which is how the
*stdio* transport signals a dead server). The HTTP transport signals the same
condition with ``TransportError``, so nothing settled the in-flight call and
the host hung forever on its ``tools/call`` id.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clyro.config import WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.errors import format_error, format_transport_error
from clyro.mcp.prevention import AllowDecision
from clyro.mcp.router import MessageRouter
from clyro.mcp.server_transport import ServerTransport, TransportError
from clyro.mcp.session import McpSession, PendingCall


def _make_router(prevention_result=None):
    config = WrapperConfig(default_action="allow")
    session = McpSession()
    transport = MagicMock(spec=ServerTransport)
    transport.send = AsyncMock()
    prevention = MagicMock()  # no spec: _settle_pending uses cost_tracker.accumulate
    if prevention_result is not None:
        prevention.evaluate.return_value = prevention_result
    prevention.cost_tracker.accumulate.return_value = 0.001
    audit = MagicMock(spec=AuditLogger)
    router = MessageRouter(config, session, transport, prevention, audit)
    return router, transport, prevention, audit


def _tools_call(request_id=1):
    msg = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": "get-sum", "arguments": {"a": 1, "b": 2}},
    }
    return json.dumps(msg).encode() + b"\n"


@pytest.mark.asyncio
async def test_transport_error_from_send_propagates_and_registers_pending() -> None:
    # An HTTP send() failure must raise TransportError (so the host reader's
    # task-level handler catches it) and the in-flight call must already be
    # registered, so the host can be told that specific id failed.
    allow = AllowDecision(tool_name="get-sum", step_number=1)
    router, transport, _prev, _audit = _make_router(allow)
    transport.send = AsyncMock(
        side_effect=TransportError("could not connect after 5 attempts")
    )

    with pytest.raises(TransportError):
        await router._handle_host_message(_tools_call(1))

    assert 1 in router._pending_requests  # the host can now be answered for id=1


def test_fail_pending_transport_reports_distinguishable_error(capsysbinary) -> None:
    # FRD-020/031: each in-flight call gets a *transport* error (not a policy
    # block), the session ends, and the call is settled — never re-sent.
    router, _transport, _prev, audit = _make_router()
    router._pending_requests[7] = PendingCall(
        request_id=7,
        tool_name="get-sum",
        params_json_len=10,
        forwarded_at=time.monotonic(),
    )

    router._fail_pending_transport("could not connect after 5 attempts")

    resp = json.loads(capsysbinary.readouterr().out.decode().strip())
    assert resp["id"] == 7
    err = resp["error"]
    assert err["code"] == -32001  # NOT -32600 (the policy-block code)
    assert err["message"].startswith("ClyroTransport:")  # NOT "ClyroPolicy:"
    assert err["data"]["type"] == "transport_error"
    assert router._shutdown_event.is_set()  # session ended (FRD-020)
    assert not router._pending_requests  # settled (FRD-026/045)
    audit.log_tool_call_response.assert_called_once()
    # FRD-026: the settled call must be recorded as *unresolved*, otherwise its
    # record is indistinguishable from a call that actually completed.
    assert audit.log_tool_call_response.call_args.kwargs["unresolved"] is True


@pytest.mark.asyncio
async def test_failed_initialize_passthrough_answers_its_id(capsysbinary) -> None:
    # A non-tools/call request (initialize) is not tracked in _pending_requests,
    # but if its forward fails the host must still get a transport error for that
    # id — otherwise the host hangs on the handshake (the dead-port case).
    router, transport, _prev, _audit = _make_router()
    transport.send = AsyncMock(side_effect=TransportError("connection refused"))
    init = json.dumps(
        {"jsonrpc": "2.0", "id": "init-1", "method": "initialize", "params": {}}
    ).encode() + b"\n"

    with pytest.raises(TransportError):
        await router._handle_host_message(init)
    # the task-level handler then reports it:
    router._fail_pending_transport("connection refused")

    resp = json.loads(capsysbinary.readouterr().out.decode().strip())
    assert resp["id"] == "init-1"
    assert resp["error"]["code"] == -32001
    assert resp["error"]["message"].startswith("ClyroTransport:")
    assert router._shutdown_event.is_set()


@pytest.mark.asyncio
async def test_pending_call_settled_when_session_ends_normally() -> None:
    # FRD-026 / AC-4.2 / AC-4.3: a call still awaiting its response when the
    # session ends *normally* (host closed stdin — nothing broke) must still be
    # settled. The connection-lost and transport-error paths settle; a clean
    # shutdown reached neither, so the call vanished: cost silently under-counted
    # (the D9 violation) and its outcome absent from the audit.
    router, _transport, _prev, audit = _make_router()
    router._pending_requests[7] = PendingCall(
        request_id=7,
        tool_name="echo",
        params_json_len=10,
        forwarded_at=time.monotonic(),
    )

    async def host_eof() -> None:
        return None  # host closed stdin: a clean, normal end — nothing failed

    async def hang() -> None:
        await asyncio.sleep(999)

    with patch.object(router, "_host_reader_task", host_eof), patch.object(
        router, "_server_reader_task", hang
    ):
        await router.run()

    assert not router._pending_requests  # settled on the way out
    audit.log_tool_call_response.assert_called_once()
    assert audit.log_tool_call_response.call_args.kwargs["unresolved"] is True


def test_transport_error_shape_differs_from_policy_block() -> None:
    # The exact FRD-031 distinction: a transport error must differ from a policy
    # block on both the JSON-RPC code and the message prefix.
    block = json.loads(format_error(1, "policy_violation", {"tool_name": "x"}))
    transport = json.loads(format_transport_error(1, "boom"))
    assert block["error"]["code"] != transport["error"]["code"]
    assert block["error"]["message"].startswith("ClyroPolicy:")
    assert transport["error"]["message"].startswith("ClyroTransport:")
    assert transport["error"]["data"]["type"] == "transport_error"
