# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MCP session-id propagation in HttpTransport — TDD §5.1.

These use a STATEFUL mock server (unlike the earlier stateless stub) that
enforces the real MCP rule: a session must be initialized, and every later
request must echo the assigned Mcp-Session-Id. This is the exact behaviour a
real server (e.g. server-everything) requires, and the gap these tests close.
"""

from __future__ import annotations

import json

import httpx
import pytest

from clyro.mcp.auth import CredentialProvider
from clyro.mcp.http_transport import HttpTransport
from clyro.mcp.safety import SafetyFloor
from clyro.mcp.tls import TlsPolicy


class _StatefulServer:
    """Mimics a real Streamable-HTTP MCP server's session enforcement."""

    def __init__(self) -> None:
        self.session_id = "sess-abc123"
        self.initialized = False
        self.seen_session_headers: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        method = body.get("method")
        sid = request.headers.get("mcp-session-id")
        self.seen_session_headers.append(sid)

        if method == "initialize":
            self.initialized = True
            # Assign the session id via response header (case-insensitive).
            return httpx.Response(
                200,
                headers={"mcp-session-id": self.session_id},
                json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"caps": {}}},
            )

        # Any non-initialize request must carry the assigned session id.
        if sid != self.session_id:
            return httpx.Response(
                400,
                json={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32000, "message": "Server not initialized"},
                },
            )
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"ran": method}}
        )


def _transport(server: _StatefulServer) -> HttpTransport:
    floor = SafetyFloor(resolver=lambda h: {"srv": ["10.0.0.9"]}.get(h) or [])

    def factory(verify):
        return httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=False)

    return HttpTransport(
        "https://srv/mcp", floor=floor, tls=TlsPolicy(), auth=CredentialProvider(None),
        client_factory=factory,
    )


@pytest.mark.asyncio
async def test_session_id_captured_and_echoed() -> None:
    # TDD §5.1: initialize assigns a session id; the transport must store it and
    # echo it on the following request so the stateful server accepts it.
    server = _StatefulServer()
    t = _transport(server)
    await t.open()

    # 1. initialize
    await t.send(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
    init_reply = json.loads(await t.receive())
    assert "result" in init_reply
    assert t._mcp_session_id == "sess-abc123"  # captured

    # 2. a follow-up call now carries the session id and SUCCEEDS
    await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo"}}')
    call_reply = json.loads(await t.receive())
    assert call_reply["result"]["ran"] == "tools/call"  # not "Server not initialized"

    # The server saw: no id on initialize, then the assigned id on the call.
    assert server.seen_session_headers == [None, "sess-abc123"]
    await t.close()


@pytest.mark.asyncio
async def test_call_before_initialize_is_rejected_by_server() -> None:
    # Without initialize, no session id exists; the stateful server rejects it.
    # This reproduces the exact failure the review found, and proves the fix is
    # what closes it (the previous stateless stub could not show this).
    server = _StatefulServer()
    t = _transport(server)
    await t.open()
    await t.send(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo"}}')
    reply = json.loads(await t.receive())
    assert reply["error"]["message"] == "Server not initialized"
    assert t._mcp_session_id is None  # never assigned
    await t.close()
