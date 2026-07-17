# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for the Stage-4 code-review findings.

Each test pins a defect the existing suite was green through. They are grouped
here (rather than scattered) because they share one root cause: the suite tested
components in isolation, and every finding lived in an *interaction* —
floor↔transport, config↔provider, audit↔backend, router↔itself.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from clyro.config import AuditConfig, ServerConfig, WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.auth import AttachOutcome, CredentialProvider
from clyro.mcp.http_transport import HttpTransport, _host_header
from clyro.mcp.prevention import AllowDecision
from clyro.mcp.router import MessageRouter
from clyro.mcp.safety import SafetyFloor
from clyro.mcp.server_transport import ServerTransport, SessionExpired, TransportError
from clyro.mcp.session import McpSession, PendingCall
from clyro.mcp.tls import TlsPolicy


def _floor(mapping: dict[str, list[str]], **kw) -> SafetyFloor:
    return SafetyFloor(resolver=lambda h: mapping[h], **kw)


class TestS3FloorGaps:
    """S3 — the floor admitted addresses its own rules intend to gate."""

    @pytest.mark.parametrize("ip", ["0.0.0.0", "::"])
    def test_unspecified_is_gated_like_loopback(self, ip: str) -> None:
        # 0.0.0.0 routes to localhost on Linux but is NOT is_loopback, so it
        # slipped FRD-039's gate entirely with the relaxation disabled.
        assert not _floor({"h": [ip]}).validate("https://h/mcp").allowed

    def test_unspecified_allowed_with_the_relaxation(self) -> None:
        assert _floor({"h": ["0.0.0.0"]}, allow_plaintext=True).validate("https://h/mcp").allowed

    def test_nat64_embedded_metadata_refused(self) -> None:
        # 64:ff9b::a9fe:a9fe reaches 169.254.169.254 in a NAT64 environment and
        # matches neither the metadata set nor is_link_local (FRD-035).
        v = _floor({"h": ["64:ff9b::a9fe:a9fe"]}).validate("https://h/mcp")
        assert not v.allowed and v.reason.value == "metadata"

    @pytest.mark.parametrize("ip", ["192.0.0.192", "100.100.100.200"])
    def test_other_cloud_metadata_endpoints_refused(self, ip: str) -> None:
        # FRD-035 says "cloud-metadata targets", not "AWS only".
        assert _floor({"h": [ip]}).validate("https://h/mcp").reason.value == "metadata"


class TestB1IPv6AndSeamContract:
    """B1 — IPv6 was unreachable and the error escaped the seam (exit 0, silent)."""

    def _t(self, ips: list[str]) -> HttpTransport:
        return HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ips}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
        )

    def test_ipv6_pinned_target_is_bracketed(self) -> None:
        # Unbracketed, "https://fd00::5/mcp" raises httpx.InvalidURL ("Invalid port: ':5'").
        assert self._t(["fd00::5"])._pinned_target("https://srv/mcp", "fd00::5") == (
            "https://[fd00::5]/mcp"
        )

    def test_ipv4_pinned_target_unchanged(self) -> None:
        assert self._t(["10.0.0.5"])._pinned_target("https://srv/mcp", "10.0.0.5") == (
            "https://10.0.0.5/mcp"
        )

    @pytest.mark.asyncio
    async def test_non_httperror_cannot_escape_send(self) -> None:
        # The ServerTransport contract: TransportError is the ONLY exit. httpx's
        # InvalidURL derives from Exception, not HTTPError, so it slipped both
        # except clauses, killed the router's host-reader task, and the wrapper
        # exited 0 with no error to the host.
        def factory(verify):
            client = MagicMock()
            client.build_request.side_effect = RuntimeError("not an HTTPError")
            return client

        t = HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.5"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')


class TestB5LivenessIsATransportProperty:
    """B5 — liveness was a read deadline, so a silent long call was killed.

    D15 makes liveness a *transport* property. A read deadline measures whether
    the server is TALKING; FRD-049 asks whether the CONNECTION IS ALIVE. Progress
    notifications are optional in MCP, so a server that thinks silently for five
    minutes reset nothing and died at the bound while perfectly healthy.

    The behavioural proof needs a real socket and real wall-clock, so it lives in
    the hand-test rig; these pin the mechanism that makes it true.
    """

    def test_reads_are_unbounded(self) -> None:
        # THE regression. Any read bound here re-introduces B5: whatever value is
        # chosen, a call that legitimately takes longer gets killed.
        t = HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.5"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            liveness_secs=60,
        )
        client = t._default_client_factory(True)
        assert client.timeout.read is None
        assert client.timeout.connect == 10.0  # connect/write stay bounded

    def test_keepalive_probes_enforce_the_bound(self) -> None:
        from clyro.mcp.http_transport import _keepalive_socket_options

        opts = {name: val for _, name, val in _keepalive_socket_options(60)}
        assert opts[socket.SO_KEEPALIVE] == 1
        idle = opts[socket.TCP_KEEPIDLE]
        intvl = opts[socket.TCP_KEEPINTVL]
        cnt = opts[socket.TCP_KEEPCNT]
        # A dead peer must be detected within D15's bound, not merely eventually.
        assert idle + (intvl * cnt) == 60

    def test_bound_is_honoured_when_tightened(self) -> None:
        from clyro.mcp.http_transport import _keepalive_socket_options

        opts = {name: val for _, name, val in _keepalive_socket_options(20)}
        assert (
            opts[socket.TCP_KEEPIDLE] + opts[socket.TCP_KEEPINTVL] * opts[socket.TCP_KEEPCNT]
        ) == 20

    def test_options_reach_the_real_socket(self) -> None:
        from clyro.mcp.http_transport import _keepalive_socket_options

        # Building the right options is worthless if the kernel rejects them.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            for level, name, val in _keepalive_socket_options(60):
                s.setsockopt(level, name, val)
            assert s.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
            assert s.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 15
        finally:
            s.close()

    def test_unenforceable_bound_is_announced_not_assumed(self, monkeypatch, caplog) -> None:
        import clyro.mcp.http_transport as ht

        # On a platform without the knobs, SO_KEEPALIVE alone means a ~2h idle:
        # D15's bound is NOT met. That must be said out loud, not quietly assumed.
        monkeypatch.delattr(socket, "TCP_KEEPIDLE", raising=False)
        monkeypatch.delattr(socket, "TCP_KEEPALIVE", raising=False)
        opts = ht._keepalive_socket_options(60)
        assert opts == [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]


class TestOperatorFacingErrors:
    """A blank error is an unactionable error (the vague-cert-message class)."""

    def test_root_cause_is_surfaced_from_a_wrapped_chain(self) -> None:
        from clyro.mcp.http_transport import _describe

        # httpx re-wraps with empty messages; the real detail is at the root.
        root = ConnectionResetError("[Errno 104] Connection reset by peer")
        mid = RuntimeError("")
        mid.__cause__ = root
        top = httpx.ReadError("")
        top.__cause__ = mid

        out = _describe(top)
        assert "ReadError" in out  # what failed
        assert "Connection reset by peer" in out  # why it failed

    def test_never_returns_an_empty_string(self) -> None:
        from clyro.mcp.http_transport import _describe

        assert _describe(httpx.ReadError("")) == "ReadError"

    def test_cycle_in_the_cause_chain_terminates(self) -> None:
        from clyro.mcp.http_transport import _describe

        a, b = RuntimeError("outer"), RuntimeError("inner")
        a.__cause__ = b
        b.__cause__ = a  # a self-referential chain must not hang the wrapper
        assert "inner" in _describe(a)


class TestC2SessionRecovery:
    """C2 — a restarted server disconnected the host instead of re-handshaking.

    The transport already cleared the stale id and stayed live, and its comment
    promised "the host's next initialize re-establishes it" — but the router
    treated every TransportError as fatal and ended the session, so that next
    initialize never came. Each half was right alone; the bug lived between them.
    """

    def _restarting_server(self, valid: dict) -> tuple[Callable, list[str]]:
        """A server that hands out session ids and 404s anything it doesn't know."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            method = json.loads(request.content or b"{}").get("method")
            sid = request.headers.get("mcp-session-id")
            seen.append(f"{method}:{sid}")
            if method == "initialize":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": 1, "result": {}},
                    headers={"mcp-session-id": valid["id"]},
                )
            if method == "notifications/initialized":
                return httpx.Response(202)
            if sid != valid["id"]:
                return httpx.Response(404)  # the session this server never issued
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 2, "result": "ok"})

        return handler, seen

    def _transport(self, handler) -> HttpTransport:
        def factory(verify):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

        return HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.9"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )

    async def _handshake(self, t: HttpTransport) -> None:
        await t.send(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
        await t.receive()
        await t.send(b'{"jsonrpc":"2.0","method":"notifications/initialized"}')

    @pytest.mark.asyncio
    async def test_server_restart_is_a_hiccup_not_a_disconnect(self) -> None:
        valid = {"id": "SESS-1"}
        handler, seen = self._restarting_server(valid)
        t = self._transport(handler)
        await t.open()
        await self._handshake(t)
        assert t._mcp_session_id == "SESS-1"

        valid["id"] = "SESS-2"  # *** the server restarts ***
        seen.clear()

        # The host makes an ordinary call, unaware anything happened.
        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}')
        msg = await t.receive()

        assert msg is not None and b'"ok"' in msg  # the host got its ANSWER
        assert t._mcp_session_id == "SESS-2"  # on a freshly negotiated session
        assert seen == [
            "tools/call:SESS-1",  # 404 — session forgotten
            "initialize:None",  # replayed the host's own handshake
            "notifications/initialized:SESS-2",  # MCP requires this to follow
            "tools/call:SESS-2",  # and only then, the retry
        ]
        await t.close()

    @pytest.mark.asyncio
    async def test_replay_reply_is_not_delivered_to_the_host(self) -> None:
        # The host already got its initialize result and did not ask for another.
        # Forwarding the replay's reply would surface a response to a request the
        # host never sent — which a strict host treats as a protocol violation.
        valid = {"id": "SESS-1"}
        handler, _ = self._restarting_server(valid)
        t = self._transport(handler)
        await t.open()
        await self._handshake(t)
        valid["id"] = "SESS-2"

        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}')
        first = await t.receive()
        assert b'"id": 2' in first or b'"id":2' in first  # the tool result, not the handshake
        assert t._inbox.qsize() == 0  # and nothing stray queued behind it
        await t.close()

    @pytest.mark.asyncio
    async def test_unrecoverable_session_raises_session_expired_not_transport_error(self) -> None:
        valid = {"id": "SESS-1"}
        handler, _ = self._restarting_server(valid)
        t = self._transport(handler)
        await t.open()
        await self._handshake(t)
        valid["id"] = "SESS-2"
        t._handshake = []  # nothing to replay

        with pytest.raises(SessionExpired):
            await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}')
        await t.close()

    def test_session_expired_is_a_transport_error(self) -> None:
        # Existing `except TransportError` handlers must stay correct by default;
        # only callers that opt in get the finer distinction.
        assert issubclass(SessionExpired, TransportError)

    @pytest.mark.asyncio
    async def test_handshake_is_cached_verbatim_and_superseded(self) -> None:
        handler, _ = self._restarting_server({"id": "SESS-1"})
        t = self._transport(handler)
        await t.open()
        first = b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"v":1}}'
        await t.send(first)
        await t.receive()
        await t.send(b'{"jsonrpc":"2.0","method":"notifications/initialized"}')
        assert t._handshake[0] == first  # verbatim: the wrapper never invents one

        # A second initialize replaces the first — replaying a stale handshake
        # would re-negotiate parameters the host has already moved on from.
        second = b'{"jsonrpc":"2.0","id":9,"method":"initialize","params":{"v":2}}'
        await t.send(second)
        assert t._handshake == [second]
        await t.close()

    @pytest.mark.asyncio
    async def test_non_json_message_does_not_break_caching(self) -> None:
        handler, _ = self._restarting_server({"id": "SESS-1"})
        t = self._transport(handler)
        await t.open()
        t._remember_handshake(b"not json at all")  # must not raise
        assert t._handshake == []
        await t.close()


class TestC2RealServerStatusCodes:
    """C2, found by hand-testing against the real server — not by any mock.

    The MCP spec and the TypeScript SDK both answer 404 for an unknown session id,
    so every mock here returned 404 and every test passed. The reference server
    (server-everything) routes on its own session map first and answers
    400 "Bad Request: No valid session ID provided", never reaching the SDK's 404
    path — so recovery never fired against the one server everyone tests with.
    """

    @pytest.mark.parametrize(
        "status,body,expected",
        [
            (404, b"", True),  # spec + SDK: invalid session id
            (404, b"anything", True),
            # The reference server's actual wording.
            (400, b'{"error":{"message":"Bad Request: No valid session ID provided"}}', True),
            (400, b'{"error":{"message":"Session not found"}}', True),
            # A plain malformed-request 400 must NOT trigger a re-handshake, or
            # every bad request costs a pointless handshake.
            (400, b'{"error":{"message":"Parse error: invalid JSON"}}', False),
            (400, b"", False),
            (401, b"session gone", False),  # auth is FRD-054's business, not this
            (500, b"session", False),
        ],
    )
    def test_session_unknown_detection(self, status: int, body: bytes, expected: bool) -> None:
        from clyro.mcp.http_transport import _is_session_unknown

        assert _is_session_unknown(status, body) is expected

    @pytest.mark.asyncio
    async def test_recovery_is_bounded_to_one_attempt_per_message(self) -> None:
        # Found while fixing the 400 case: _post retries after recovering, and the
        # retry could recover again, unbounded. Every mock accepted the new session,
        # so nothing caught it. A server that rejects EVERY session must fail the
        # call, not spin forever.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            method = json.loads(request.content or b"{}").get("method")
            if method == "initialize":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": 1, "result": {}},
                    headers={"mcp-session-id": "SESS-NEW"},
                )
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/call":  # count ONLY the call, not the server->client GET
                attempts["n"] += 1
            return httpx.Response(404)  # never accepts any session, ever

        def factory(verify):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

        t = HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.9"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )
        await t.open()
        await t.send(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
        await t.receive()
        await t.send(b'{"jsonrpc":"2.0","method":"notifications/initialized"}')

        with pytest.raises(SessionExpired, match="just issued"):
            await asyncio.wait_for(
                t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}'), timeout=5
            )
        assert attempts["n"] == 2  # the original + exactly ONE retry. Not a loop.
        await t.close()

    @pytest.mark.asyncio
    async def test_non_session_400_reaches_the_host(self) -> None:
        # The body is buffered to classify it; a genuine 400 must still be
        # delivered, not swallowed by the classification read.
        def handler(request: httpx.Request) -> httpx.Response:
            method = json.loads(request.content or b"{}").get("method")
            if method == "initialize":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": 1, "result": {}},
                    headers={"mcp-session-id": "SESS-1"},
                )
            return httpx.Response(400, json={"jsonrpc": "2.0", "error": {"message": "Parse error"}})

        def factory(verify):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

        t = HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.9"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )
        await t.open()
        await t.send(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
        await t.receive()

        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}')
        msg = await asyncio.wait_for(t.receive(), timeout=2)
        assert b"Parse error" in msg  # forwarded, not eaten by the body read
        await t.close()


class TestC2RouterKeepsServing:
    """C2, router half — SessionExpired must not end a healthy session."""

    def _router(self, send_effect) -> tuple[MessageRouter, MagicMock]:
        prevention = MagicMock()
        prevention.evaluate.return_value = AllowDecision(tool_name="echo", step_number=1)
        prevention.cost_tracker.accumulate.return_value = 0.0
        audit = MagicMock(spec=AuditLogger)
        transport = MagicMock(spec=ServerTransport)
        transport.send = AsyncMock(side_effect=send_effect)
        return MessageRouter(
            WrapperConfig(default_action="allow"), McpSession(), transport, prevention, audit
        ), audit

    @pytest.mark.asyncio
    async def test_session_expired_fails_the_call_but_keeps_the_loop(self) -> None:
        router, audit = self._router(SessionExpired("forgot you"))
        raw = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo"}}'

        # The router must survive the raise: the connection is healthy, so the
        # host keeps its tools instead of watching the server vanish (R-C2).
        with pytest.raises(SessionExpired):
            await router._handle_host_message(raw)
        router._fail_pending_transport("forgot you")
        assert audit.log_tool_call_response.called  # the in-flight call was settled

    @pytest.mark.asyncio
    async def test_a_real_transport_error_still_ends_the_session(self) -> None:
        # The fix must not make genuine disconnects survivable — that would hang
        # the host on a dead server.
        router, _ = self._router(TransportError("connection reset"))
        with pytest.raises(TransportError) as ei:
            await router._handle_host_message(
                b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo"}}'
            )
        assert not isinstance(ei.value, SessionExpired)


class TestB3SamplingDeadlock:
    """B3 — send() blocked on the response body, so a server that asked the client
    something mid-tool-call hung the wrapper forever.

    MCP lets a server issue sampling/elicitation during a tools/call and wait for
    the client's answer. That answer is POSTed by the router's single host-reader
    task — the very task blocked inside send(). Both sides waited: no error, no
    timeout, just a silent hang.

    Only the BODY drain moved to the background. Headers stay inline, so every
    error send() used to raise it still raises, and POST order is unchanged.
    """

    def _sampling_server(self):
        """A server that asks a question mid-call and waits for the answer."""
        state = {"answered": False}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}") if request.content else {}
            if body.get("method") == "sampling_reply":
                state["answered"] = True
                return httpx.Response(202)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"jsonrpc":"2.0","id":"s1","method":"sampling/createMessage"}\n\n'
                    b'data: {"jsonrpc":"2.0","id":2,"result":{"tool":"finished"}}\n\n'
                ),
            )

        return handler, state

    def _transport(self, handler) -> HttpTransport:
        def factory(verify):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

        return HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.9"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )

    @pytest.mark.asyncio
    async def test_send_does_not_block_on_the_response_body(self) -> None:
        # THE regression. If send() ever waits for the body again, a server that
        # asks a question mid-call deadlocks the wrapper.
        handler, _ = self._sampling_server()
        t = self._transport(handler)
        await t.open()

        await asyncio.wait_for(t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}'), timeout=2)
        # send() returned while the body is still being read: the host-reader task
        # is free to answer the server's question. That freedom IS the fix.
        msg = await asyncio.wait_for(t.receive(), timeout=2)
        assert b"sampling/createMessage" in msg
        await asyncio.wait_for(t.send(b'{"jsonrpc":"2.0","method":"sampling_reply"}'), timeout=2)
        await t.close()

    @pytest.mark.asyncio
    async def test_full_sampling_round_trip_completes(self) -> None:
        handler, state = self._sampling_server()
        t = self._transport(handler)
        await t.open()
        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}')

        question = await asyncio.wait_for(t.receive(), timeout=2)
        assert json.loads(question)["method"] == "sampling/createMessage"
        await t.send(b'{"jsonrpc":"2.0","method":"sampling_reply"}')
        assert state["answered"] is True  # the answer actually reached the server

        result = await asyncio.wait_for(t.receive(), timeout=2)
        assert json.loads(result)["result"] == {"tool": "finished"}
        await t.close()

    @pytest.mark.asyncio
    async def test_header_stage_errors_still_raise_from_send(self) -> None:
        # The fix's safety property: only the BODY moved. If an error that used to
        # surface inline now happens in a background task, it vanishes (B1's
        # silent-failure signature) — so pin that these still raise.
        #
        # 401 (FRD-054) and an over-long redirect chain (FRD-037) are both decided
        # from headers alone. NB 5xx is deliberately absent: the transport does not
        # treat it as an error — the body is forwarded to the host — and that was
        # true before this change too.
        t = self._transport(lambda r: httpx.Response(401))
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}')
        await t.close()

        t = self._transport(lambda r: httpx.Response(307, headers={"location": "https://srv/mcp"}))
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}')
        await t.close()

    @pytest.mark.asyncio
    async def test_body_failure_ends_the_session_rather_than_vanishing(self) -> None:
        # A mid-stream failure used to raise from send() and end the session. In
        # the background there is no caller to raise to, so it must still reach the
        # router — via receive() -> None, the seam's end-of-connection signal.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("connection died mid-stream")

        t = self._transport(handler)
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')  # this one still fails at the header stage
        await t.close()

    @pytest.mark.asyncio
    async def test_close_awaits_in_flight_drains(self) -> None:
        handler, _ = self._sampling_server()
        t = self._transport(handler)
        await t.open()
        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call"}')
        await t.close()
        # A dropped drain task would read from a closed pool and log a stray error.
        assert t._drains == set()

    @pytest.mark.asyncio
    async def test_replay_reply_still_suppressed_with_a_background_drain(self) -> None:
        # The trap this fix nearly walked into: _drain_response used to read
        # `self._reestablishing`, but a BACKGROUND drain runs after the flag has
        # been reset — delivering the replayed handshake's reply to a host that
        # never asked for one (R-C2, resurrected). `deliver` is decided at spawn.
        valid = {"id": "SESS-1"}
        c2 = TestC2SessionRecovery()
        handler, _ = c2._restarting_server(valid)
        t = c2._transport(handler)
        await t.open()
        await c2._handshake(t)
        valid["id"] = "SESS-2"

        await t.send(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}')
        first = await t.receive()
        assert b'"result": "ok"' in first or b'"result":"ok"' in first
        await asyncio.sleep(0.05)  # let every background drain finish
        assert t._inbox.qsize() == 0  # the replay's reply was NOT delivered
        await t.close()


class TestC6HostHeader:
    """C6 — Host dropped the port; with a pinned IP, Host IS the routing key."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://api.corp:2753/mcp", "api.corp:2753"),
            ("https://api.corp/mcp", "api.corp"),  # default port omitted
            ("http://api.corp/mcp", "api.corp"),
            ("http://[::1]:9000/mcp", "[::1]:9000"),  # re-bracketed
        ],
    )
    def test_host_header(self, url: str, expected: str) -> None:
        assert _host_header(urlsplit(url)) == expected


class TestC4LoopbackCredential:
    """C4 — origin parsing refused the credential on IPv6/127.x loopback (AC-8.7)."""

    @pytest.mark.parametrize(
        "origin", ["http://[::1]:9000", "http://127.0.0.2:8080", "http://localhost:3001"]
    )
    def test_loopback_variants_get_the_credential(self, origin: str) -> None:
        # The naive split yielded '[' for "http://[::1]:9000"; and all of
        # 127.0.0.0/8 is loopback, not just 127.0.0.1.
        headers: dict[str, str] = {}
        out = CredentialProvider("Bearer T").attach(
            headers, target_origin=origin, request_origin=origin, encrypted=False
        )
        assert out is AttachOutcome.ATTACHED

    def test_remote_plaintext_still_refused(self) -> None:
        # FRD-055 must not be loosened by the fix.
        out = CredentialProvider("Bearer T").attach(
            {}, target_origin="http://10.0.0.5", request_origin="http://10.0.0.5", encrypted=False
        )
        assert out is AttachOutcome.REFUSED


class TestS1TraceMasking:
    """S1 — FRD-034's trace half was absent: the credential left the host."""

    def test_credential_masked_on_the_backend_trace_path(self, tmp_path) -> None:
        token = "Bearer SECRET-xyz789"
        al = AuditLogger(AuditConfig(log_path=str(tmp_path / "a.jsonl")), session_id=uuid4())
        al.set_credential_mask(token)
        sync = MagicMock()
        al.set_backend(sync, MagicMock())

        al._enqueue_trace({"input_data": {"arg": token}, "note": f"401 body: {token}"})

        sent = json.dumps(sync.enqueue.call_args[0][0])
        assert "SECRET-xyz789" not in sent  # trace events travel OFF-HOST
        assert "[REDACTED]" in sent

    def test_no_backend_is_a_noop(self, tmp_path) -> None:
        al = AuditLogger(AuditConfig(log_path=str(tmp_path / "a.jsonl")), session_id=uuid4())
        al._enqueue_trace({"event": "x"})  # must not raise


class TestS2CredentialHeaderName:
    """S2 — hardcoded 'Authorization': credential silently unsent AND unmasked."""

    def test_credential_under_a_named_header(self) -> None:
        cfg = ServerConfig(url="https://x/mcp", auth_header="X-API-Key", headers={"X-API-Key": "s"})
        assert cfg.headers.get(cfg.auth_header) == "s"

    def test_unnamed_header_is_rejected_not_silently_dropped(self) -> None:
        # Previously: sent nowhere, masked nowhere, no warning.
        with pytest.raises(ValidationError, match="only carries the credential header"):
            ServerConfig(url="https://x/mcp", headers={"X-API-Key": "s"})

    def test_default_authorization_still_works(self) -> None:
        cfg = ServerConfig(url="https://x/mcp", headers={"Authorization": "Bearer T"})
        assert cfg.headers.get(cfg.auth_header) == "Bearer T"


class TestB4OutstandingIdCollision:
    """B4 / V-24 — a colliding id silently discarded the earlier call's accounting."""

    @pytest.mark.asyncio
    async def test_displaced_call_is_settled_not_discarded(self) -> None:
        prevention = MagicMock()
        prevention.evaluate.return_value = AllowDecision(tool_name="echo", step_number=1)
        prevention.cost_tracker.accumulate.return_value = 0.001
        audit = MagicMock(spec=AuditLogger)
        transport = MagicMock(spec=ServerTransport)
        transport.send = AsyncMock()
        router = MessageRouter(
            WrapperConfig(default_action="allow"), McpSession(), transport, prevention, audit
        )
        router._pending_requests[1] = PendingCall(
            request_id=1, tool_name="first", params_json_len=10, forwarded_at=time.monotonic()
        )

        raw = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,  # collides with the outstanding call
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"m": "second"}},
                }
            ).encode()
            + b"\n"
        )
        await router._handle_host_message(raw)

        # The displaced call was already forwarded and executed — its cost must
        # survive, or N calls run while 1 is billed and FRD-006 never fires.
        audit.log_tool_call_response.assert_called_once()
        kwargs = audit.log_tool_call_response.call_args.kwargs
        assert kwargs["tool_name"] == "first"
        assert kwargs["call_cost_usd"] > 0
        assert kwargs["unresolved"] is True
        assert router._pending_requests[1].tool_name == "echo"  # new call registered


class TestR1SettlementStatesNoCause:
    """R1 — settlement asserted 'connection lost' on clean shutdowns (incl. stdio)."""

    @pytest.mark.asyncio
    async def test_settlement_does_not_claim_a_cause_it_did_not_observe(self) -> None:
        prevention = MagicMock()
        prevention.cost_tracker.accumulate.return_value = 0.001
        audit = MagicMock(spec=AuditLogger)
        router = MessageRouter(
            WrapperConfig(default_action="allow"),
            McpSession(),
            MagicMock(spec=ServerTransport),
            prevention,
            audit,
        )
        router._pending_requests[7] = PendingCall(
            request_id=7, tool_name="echo", params_json_len=10, forwarded_at=time.monotonic()
        )
        router._settle_pending()

        content = audit.log_tool_call_response.call_args.kwargs["response_content"]
        assert "connection lost" not in content  # nothing was lost on a clean exit
        assert "no response received" in content  # state the fact, not a cause


class TestC1TransportOnTraceRecords:
    """C1 — FRD-032 requires transport on every audit AND every trace record."""

    def test_session_carries_transport_for_trace_metadata(self) -> None:
        s = McpSession()
        assert s.transport is None  # default; stamped by the CLI at startup
        s.transport = "http"
        assert s.transport == "http"


class TestC3RelativeRedirects:
    """C3 — a relative Location (RFC 7231 §7.1.2) was refused as bad_url."""

    @pytest.mark.asyncio
    async def test_relative_location_resolves_against_the_request_url(self) -> None:
        seen: list[str] = []

        def factory(verify):
            import httpx

            def handler(request: httpx.Request) -> httpx.Response:
                seen.append(str(request.url))
                if len(seen) == 1:
                    return httpx.Response(307, headers={"location": "/v2/mcp"})
                return httpx.Response(200, json={"ok": True})

            return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)

        t = HttpTransport(
            "https://srv/mcp",
            floor=_floor({"srv": ["10.0.0.9"]}),
            tls=TlsPolicy(),
            auth=CredentialProvider(None),
            client_factory=factory,
        )
        await t.open()
        await t.send(b'{"id":1}')
        assert await t.receive() is not None  # followed, not refused
        assert seen[1].endswith("/v2/mcp")  # resolved against the base URL
        await t.close()
