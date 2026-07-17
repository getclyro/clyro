# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration: an HTTP-wrapped tool call driven through MessageRouter —
FRD-020 (governed over HTTP), FRD-026/045 (settlement), FRD-032 (transport in
records), FRD-031 (transport-blind router).

Uses httpx.MockTransport for the server; no network, no real rig (OQ-1).
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from clyro.config import AuditConfig, WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.auth import CredentialProvider
from clyro.mcp.http_transport import HttpTransport
from clyro.mcp.prevention import PreventionStack
from clyro.mcp.router import MessageRouter
from clyro.mcp.safety import SafetyFloor
from clyro.mcp.session import McpSession
from clyro.mcp.tls import TlsPolicy


def _http_transport(handler) -> HttpTransport:
    floor = SafetyFloor(resolver=lambda h: {"srv": ["10.0.0.9"]}.get(h) or [])

    def factory(verify):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

    return HttpTransport(
        "https://srv/mcp",
        floor=floor,
        tls=TlsPolicy(),
        auth=CredentialProvider("Bearer T"),
        client_factory=factory,
    )


def _router(transport, tmp_path):
    config = WrapperConfig(default_action="allow")
    session = McpSession()
    audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), session.session_id)
    audit.set_transport("http")  # FRD-032
    return (
        MessageRouter(config, session, transport, PreventionStack(config), audit),
        session,
        audit,
    )


@pytest.mark.asyncio
async def test_governed_call_over_http_end_to_end(tmp_path) -> None:
    # FRD-020/031: a tools/call is governed and forwarded over HTTP; the
    # response is correlated and cost accrues.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"rows": 3}})

    t = _http_transport(handler)
    await t.open()
    router, session, audit = _router(t, tmp_path)

    raw = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list", "arguments": {}}}
    ).encode()
    await router._handle_host_message(raw)  # host -> govern -> HTTP POST

    # server reply -> correlate + cost (drive one receive/correlate cycle)
    msg = await t.receive()
    reply = json.loads(msg)
    assert reply["result"]["rows"] == 3
    assert session.step_count == 1  # governed
    await t.close()

    # FRD-032: the audit record carries the transport.
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    tool_calls = [json.loads(x) for x in lines if json.loads(x).get("event") == "tool_call"]
    assert tool_calls and tool_calls[0]["transport"] == "http"


@pytest.mark.asyncio
async def test_settlement_on_connection_loss(tmp_path) -> None:
    # FRD-026/045: a call in flight when the connection drops is settled at a
    # non-zero estimate and never re-sent.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "ok"})

    t = _http_transport(handler)
    await t.open()
    router, session, audit = _router(t, tmp_path)

    # Manually register a pending call and settle it (simulates a drop).
    from clyro.mcp.session import PendingCall

    router._pending_requests[1] = PendingCall(
        request_id=1, tool_name="slow", params_json_len=40, forwarded_at=0.0
    )
    before = session.accumulated_cost_usd
    router._settle_pending()
    assert session.accumulated_cost_usd > before  # FRD-026: never zero (D9)
    assert 1 not in router._pending_requests  # settled, not left dangling
    await t.close()
