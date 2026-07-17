# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for HttpTransport (Streamable HTTP) —
FRD-020, FRD-042, FRD-037, FRD-051, FRD-052, FRD-054, FRD-057.

No live server: an httpx.MockTransport backs the injected client, and a fixed
resolver backs the SafetyFloor, so everything is hermetic.
"""

from __future__ import annotations

import httpx
import pytest

from clyro.mcp.auth import CredentialProvider
from clyro.mcp.http_transport import HttpTransport
from clyro.mcp.safety import SafetyFloor
from clyro.mcp.server_transport import ServerTransport, TransportError
from clyro.mcp.tls import TlsPolicy

RESOLVER = {
    "srv": ["10.0.0.9"],
    "evil": ["169.254.169.254"],
    "other": ["10.0.0.10"],
}


def _floor() -> SafetyFloor:
    return SafetyFloor(resolver=lambda h: RESOLVER.get(h) or (_ for _ in ()).throw(OSError(h)))


def _make(url: str, handler, *, token: str | None = "Bearer T", floor: SafetyFloor | None = None):
    def factory(verify):
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=False
        )

    return HttpTransport(
        url,
        floor=floor or _floor(),
        tls=TlsPolicy(),
        auth=CredentialProvider(token),
        client_factory=factory,
    )


def test_conforms_to_protocol() -> None:
    assert isinstance(_make("https://srv/mcp", lambda r: httpx.Response(200)), ServerTransport)


@pytest.mark.asyncio
async def test_open_refuses_metadata_target() -> None:
    # FRD-035 via the floor: open() must refuse before any connection.
    t = _make("https://evil/mcp", lambda r: httpx.Response(200))
    with pytest.raises(TransportError):
        await t.open()


@pytest.mark.asyncio
async def test_send_receive_json() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.headers.get("Host")
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "ok"})

    t = _make("https://srv/mcp", handler)
    await t.open()
    await t.send(b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}')
    msg = await t.receive()
    assert msg is not None and b'"result"' in msg  # FRD-020
    assert seen["host"] == "srv"  # Host preserved
    assert seen["auth"] == "Bearer T"  # FRD-033 attached over https
    assert "10.0.0.9" in seen["url"]  # connected to the pinned IP (FRD-057)
    await t.close()


@pytest.mark.asyncio
async def test_auth_failed_401_raises() -> None:
    # FRD-054
    t = _make("https://srv/mcp", lambda r: httpx.Response(401))
    await t.open()
    with pytest.raises(TransportError):
        await t.send(b'{"id":1}')
    await t.close()


class TestRedirects:
    @pytest.mark.asyncio
    async def test_redirect_to_private_followed(self) -> None:
        # FRD-052: redirect re-validated and followed to a safe target.
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "10.0.0.9":
                return httpx.Response(307, headers={"location": "https://other/mcp"})
            return httpx.Response(200, json={"ok": True})

        t = _make("https://srv/mcp", handler)
        await t.open()
        await t.send(b'{"id":1}')
        assert await t.receive() is not None
        await t.close()

    @pytest.mark.asyncio
    async def test_redirect_to_metadata_refused(self) -> None:
        # FRD-052: a redirect destination is a target; metadata is refused.
        t = _make(
            "https://srv/mcp",
            lambda r: httpx.Response(307, headers={"location": "https://evil/mcp"}),
        )
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')
        await t.close()

    @pytest.mark.asyncio
    async def test_redirect_loop_bounded(self) -> None:
        # FRD-037: a chain that never terminates is bounded at 5 hops.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(307, headers={"location": "https://srv/mcp"})

        t = _make("https://srv/mcp", handler)
        await t.open()
        with pytest.raises(TransportError):
            await t.send(b'{"id":1}')
        await t.close()

    @pytest.mark.asyncio
    async def test_cross_origin_redirect_connects_to_validated_ip(self) -> None:
        # R-1: a cross-origin redirect must connect to the IP the floor validated
        # for that hop (pinned), not re-resolve the hostname (rebinding TOCTOU).
        seen_hosts = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            if request.url.host == "10.0.0.9":  # original, pinned
                return httpx.Response(307, headers={"location": "https://other/mcp"})
            return httpx.Response(200, json={"ok": True})

        t = _make("https://srv/mcp", handler)  # RESOLVER: other -> 10.0.0.10
        await t.open()
        await t.send(b'{"id":1}')
        await t.receive()
        # the redirected request went to other's validated IP (10.0.0.10), NOT "other"
        assert "10.0.0.10" in seen_hosts
        assert "other" not in seen_hosts  # hostname never used for the connection
        await t.close()

    @pytest.mark.asyncio
    async def test_cross_origin_redirect_strips_credential(self) -> None:
        # FRD-051: credential must not follow a redirect to a different origin.
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "10.0.0.9":
                return httpx.Response(307, headers={"location": "https://other/mcp"})
            seen["auth_on_other"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"ok": True})

        t = _make("https://srv/mcp", handler)
        await t.open()
        await t.send(b'{"id":1}')
        await t.receive()
        assert seen["auth_on_other"] is None  # credential withheld cross-origin
        await t.close()


@pytest.mark.asyncio
async def test_close_makes_receive_return_none_and_not_live() -> None:
    t = _make("https://srv/mcp", lambda r: httpx.Response(200))
    await t.open()
    assert t.is_live() is True
    await t.close()
    assert t.is_live() is False
    assert await t.receive() is None


@pytest.mark.asyncio
async def test_liveness_secs_sets_read_timeout() -> None:
    # FRD-049: liveness_secs is the transport read-timeout (no longer a dead param).
    t = HttpTransport(
        "https://srv/mcp", floor=_floor(), tls=TlsPolicy(),
        auth=CredentialProvider(None), liveness_secs=7,
    )
    await t.open()
    assert t._client.timeout.read == 7
    await t.close()


@pytest.mark.asyncio
async def test_read_timeout_becomes_transport_error() -> None:
    # FRD-049: an unresponsive transport surfaces as a TransportError, not a hang.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no bytes", request=request)

    t = _make("https://srv/mcp", handler)
    await t.open()
    with pytest.raises(TransportError):
        await t.send(b'{"id":1}')
    await t.close()


@pytest.mark.asyncio
async def test_tls_error_reported_clearly_and_not_retried() -> None:
    # Finding #5: a certificate failure is permanent — it must be reported as a
    # TLS error (not a generic "could not connect after N attempts") and must not
    # consume the reconnect budget (FRD-040/046).
    import ssl

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connect failed", request=request) from (
            ssl.SSLCertVerificationError("certificate verify failed: self-signed certificate")
        )

    t = _make("https://srv/mcp", handler)
    await t.open()
    with pytest.raises(TransportError) as exc_info:
        await t.send(b'{"id":1}')
    assert "TLS certificate verification failed" in str(exc_info.value)
    assert calls["n"] == 1  # raised on the first attempt — no wasted retries
    await t.close()


@pytest.mark.asyncio
async def test_sse_framed_response_enqueued() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = 'data: {"jsonrpc":"2.0","id":1,"result":"streamed"}\n\n'
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    t = _make("https://srv/mcp", handler)
    await t.open()
    await t.send(b'{"id":1}')
    msg = await t.receive()
    assert msg is not None and b'"streamed"' in msg
    await t.close()
