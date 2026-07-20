# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for bounded reconnection, session reset, and the server-initiated
GET stream — FRD-025, FRD-027, FRD-043, FRD-045, FRD-056, TDD §4.2.
"""

from __future__ import annotations

import json

import httpx
import pytest

from clyro.mcp.auth import CredentialProvider
from clyro.mcp.http_transport import HttpTransport
from clyro.mcp.safety import SafetyFloor
from clyro.mcp.server_transport import TransportError
from clyro.mcp.tls import TlsPolicy


def _floor():
    return SafetyFloor(resolver=lambda h: {"srv": ["10.0.0.9"]}.get(h) or [])


def _transport(handler, *, max_reconnect=5, events=None):
    def factory(verify):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

    hook = (lambda e, f: events.append((e, f))) if events is not None else None
    return HttpTransport(
        "https://srv/mcp", floor=_floor(), tls=TlsPolicy(), auth=CredentialProvider(None),
        max_reconnect=max_reconnect, on_lifecycle=hook, client_factory=factory,
    )


class TestBoundedReconnect:
    @pytest.mark.asyncio
    async def test_retries_pre_delivery_connect_error_then_succeeds(self) -> None:
        # FRD-056: a connect error (message never sent → FRD-045-safe) is retried.
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] < 3:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json={"ok": True})

        events: list = []
        t = _transport(handler, events=events)
        await t.open()
        await t.send(b'{"id":1}')  # succeeds on the 3rd attempt
        assert state["n"] == 3
        assert [e for e, _ in events].count("reconnecting") == 2  # FRD-043 events
        await t.close()

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        t = _transport(handler, max_reconnect=2)
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')
        await t.close()

    @pytest.mark.asyncio
    async def test_read_timeout_is_NOT_retried(self) -> None:
        # FRD-045: a post-send failure must not be retried (tool may have run).
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            raise httpx.ReadTimeout("no reply", request=request)

        t = _transport(handler, max_reconnect=5)
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')
        assert state["n"] == 1  # tried exactly once — never re-sent
        await t.close()


class TestSessionReset:
    @pytest.mark.asyncio
    async def test_404_clears_session_and_errors(self) -> None:
        # FRD-025: server forgot the session → drop the id, signal re-init.
        def handler(request: httpx.Request) -> httpx.Response:
            if json.loads(request.content)["method"] == "initialize":
                return httpx.Response(200, headers={"mcp-session-id": "s1"}, json={"result": {}})
            return httpx.Response(404, json={"error": "unknown session"})

        events: list = []
        t = _transport(handler, events=events)
        await t.open()
        await t.send(b'{"id":1,"method":"initialize"}')
        await t.receive()
        assert t._mcp_session_id == "s1"
        with pytest.raises(TransportError):
            await t.send(b'{"id":2,"method":"tools/call"}')
        assert t._mcp_session_id is None  # cleared for re-initialize
        assert ("server_session_reset", {}) in events
        await t.close()


class TestServerInitiatedStream:
    @pytest.mark.asyncio
    async def test_get_stream_delivers_server_request(self) -> None:
        # TDD §4.2 / FRD-027: a server-initiated request on the GET stream is
        # enqueued for the host.
        sampling = '{"jsonrpc":"2.0","id":99,"method":"sampling/createMessage","params":{}}'

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"},
                    content=f"data: {sampling}\n\n",
                )
            # POST initialize -> assigns session id (triggers GET stream)
            return httpx.Response(200, headers={"mcp-session-id": "s1"}, json={"result": {}})

        t = _transport(handler)
        await t.open()
        await t.send(b'{"id":1,"method":"initialize"}')
        # first receive = init reply; then the server-initiated message arrives
        seen = []
        for _ in range(2):
            msg = await t.receive()
            if msg:
                seen.append(msg.decode())
        assert any("sampling/createMessage" in m for m in seen)  # FRD-027
        await t.close()
