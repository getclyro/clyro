# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Native HTTP (Streamable HTTP) transport
# Implements FRD-020, FRD-042, FRD-057, FRD-037, FRD-051, FRD-052, FRD-049, FRD-050, FRD-054

"""
Native Streamable-HTTP transport for the MCP wrapper.

Conforms to :class:`~clyro.mcp.server_transport.ServerTransport` so the router
is transport-blind. On :meth:`open` the target passes the :class:`SafetyFloor`
(refuse metadata/link-local; pin the resolved IP — FRD-035/036/038) and the
connection is made to *that IP* with SNI/Host preserved for TLS (the mechanism
verified by the DD-1 spike), ignoring any proxy env (FRD-057).

Each host message is POSTed (FRD-020). Redirects are followed manually, bounded
at :data:`REDIRECT_MAX_HOPS` (FRD-037), re-validated per hop (FRD-052), with the
credential withheld across an origin change (FRD-051). Server replies — JSON or
``text/event-stream`` framed — are enqueued for :meth:`receive`.

The ``client_factory`` seam lets tests inject an ``httpx.AsyncClient`` backed by
a mock transport; production builds a real pinned-IP client.
"""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from clyro.mcp.auth import AttachOutcome, AuthOutcome, CredentialProvider
from clyro.mcp.log import get_logger
from clyro.mcp.safety import REDIRECT_MAX_HOPS, SafetyFloor
from clyro.mcp.server_transport import TransportError
from clyro.mcp.tls import TlsPolicy

logger = get_logger(__name__)


def _tls_cause(exc: BaseException) -> ssl.SSLError | None:
    """Return the TLS/certificate error in *exc*'s cause chain, if any.

    httpx surfaces a certificate-verification failure as an ``httpx.ConnectError``
    that wraps an ``ssl.SSLError`` (via ``__cause__``/``__context__``). Walking
    the chain lets the transport tell a *permanent* TLS failure apart from a
    *transient* connect failure (FRD-040/046).
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None

ClientFactory = Callable[[str | bool], httpx.AsyncClient]
LifecycleHook = Callable[[str, dict], None]


def _origin(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}"


class HttpTransport:
    """Streamable-HTTP downstream leg. Implements the ServerTransport seam (FRD-020)."""

    def __init__(
        self,
        url: str,
        *,
        floor: SafetyFloor,
        tls: TlsPolicy,
        auth: CredentialProvider,
        liveness_secs: int = 60,
        max_reconnect: int = 5,
        on_lifecycle: LifecycleHook | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._url = url
        self._floor = floor
        self._tls = tls
        self._auth = auth
        self._liveness_secs = liveness_secs
        self._max_reconnect = max_reconnect  # FRD-056: bounded reconnection
        self._on_lifecycle = on_lifecycle  # FRD-043: reconnect/session lifecycle events
        self._client_factory = client_factory or self._default_client_factory
        self._client: httpx.AsyncClient | None = None
        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False
        self._pinned_ip: str | None = None
        # MCP session id, assigned by the server on ``initialize`` and echoed on
        # every subsequent request (TDD §5.1). Without this a stateful server
        # rejects follow-up requests as "not initialized".
        self._mcp_session_id: str | None = None
        self._stream_task: asyncio.Task[None] | None = None  # server->client GET stream

    def _emit(self, event: str, **fields: object) -> None:
        """Record a connection lifecycle event (FRD-043)."""
        logger.info(event, **fields)
        if self._on_lifecycle is not None:
            self._on_lifecycle(event, fields)

    # ------------------------------------------------------------------
    # ServerTransport protocol
    # ------------------------------------------------------------------

    async def open(self) -> None:
        """Validate the target, pin its IP, and build the client (FRD-020/042/057/040)."""
        verdict = self._floor.validate(self._url)
        if not verdict.allowed:
            raise TransportError(
                f"target refused by safety floor: {verdict.reason.value}"
            )
        self._pinned_ip = verdict.resolved_ip
        # verify= from TLS policy (FRD-040/046/047); trust_env=False ignores proxy
        # env so the floor's local resolution is authoritative (FRD-057).
        self._client = self._client_factory(self._tls.verify_value())

    async def send(self, data: bytes) -> None:
        """POST one JSON-RPC message; enqueue the reply(ies). Implements FRD-020."""
        if self._client is None or self._closed:
            raise TransportError("transport not open")
        try:
            await self._post(self._url, data, hop=0)
        except httpx.TimeoutException as exc:
            # FRD-049: no transport-level activity within the liveness bound
            # (read timeout) means the connection is dead. An in-progress call
            # that keeps streaming bytes resets this, so long calls survive.
            raise TransportError(
                f"connection unresponsive for {self._liveness_secs}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"http transport error: {exc}") from exc

    async def receive(self) -> bytes | None:
        """Return the next server message, or None once closed."""
        return await self._inbox.get()

    async def close(self) -> None:
        self._closed = True
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stream_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await self._inbox.put(None)  # unblock a pending receive()

    async def _read_server_stream(self) -> None:
        """Open the server->client GET stream; enqueue server-initiated messages.

        Implements the server→client leg of Streamable HTTP (TDD §4.2): a
        long-lived ``GET`` carrying sampling/elicitation/notification requests
        that the router then delivers to the host (FRD-027/028). A server that
        offers no GET stream (4xx) is fine — this simply returns.
        """
        assert self._client is not None
        parts = urlsplit(self._url)
        headers = {"Accept": "text/event-stream", "Host": parts.hostname or ""}
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id
        target = self._pinned_target(self._url, self._pinned_ip)  # GET stream = original origin
        request = self._client.build_request("GET", target, headers=headers)
        if parts.hostname:
            request.extensions["sni_hostname"] = parts.hostname
        try:
            response = await self._client.send(request, stream=True)
            if response.status_code >= 400:
                await response.aclose()
                return
            async for line in response.aiter_lines():
                if self._closed:
                    break
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        await self._inbox.put(payload.encode("utf-8") + b"\n")  # FRD-027
            await response.aclose()
        except (httpx.HTTPError, asyncio.CancelledError):
            return

    def is_live(self) -> bool:
        return self._client is not None and not self._closed

    # ------------------------------------------------------------------
    # POST + manual redirect handling (FRD-037/051/052)
    # ------------------------------------------------------------------

    async def _post(
        self, url: str, data: bytes, *, hop: int, pinned_ip: str | None = None
    ) -> None:
        assert self._client is not None
        if hop > REDIRECT_MAX_HOPS:  # FRD-037
            raise TransportError(f"redirect chain exceeded {REDIRECT_MAX_HOPS} hops")
        # Pin the IP the floor just validated for THIS url — the original target
        # (self._pinned_ip) on hop 0, or the redirect hop's resolved IP on a
        # redirect. Connecting by IP closes the rebinding TOCTOU on every hop
        # (R-1 / T-2), not only the first.
        ip = pinned_ip if pinned_ip is not None else self._pinned_ip

        parts = urlsplit(url)
        encrypted = parts.scheme == "https"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Host": parts.hostname or "",
        }
        # TDD §5.1: echo the server-assigned session id on every request after
        # ``initialize`` so a stateful server recognises the session.
        if self._mcp_session_id is not None:
            headers["Mcp-Session-Id"] = self._mcp_session_id
        # FRD-051/055: attach the credential only to the *original* origin over
        # an encrypted (or loopback) connection.
        attach = self._auth.attach(
            headers,
            target_origin=_origin(url),
            request_origin=_origin(self._url),
            encrypted=encrypted,
        )
        if attach is AttachOutcome.REFUSED:
            raise TransportError("credential would be sent over cleartext; refused")

        target = self._pinned_target(url, ip)
        request = self._client.build_request("POST", target, content=data, headers=headers)
        if parts.hostname:
            request.extensions["sni_hostname"] = parts.hostname  # DD-1 mechanism

        response = await self._send_with_reconnect(request)

        if response.is_redirect:  # manual follow (client is follow_redirects=False)
            location = str(response.headers.get("location", ""))
            await response.aclose()
            await self._follow_redirect(location, data, hop)
            return

        # FRD-025: a 404 means the server no longer recognises the session. Drop
        # the stale id so the host's next `initialize` re-establishes it — the
        # wrapper cannot re-run the handshake itself (it does not own it).
        if response.status_code == 404 and self._mcp_session_id is not None:
            await response.aclose()
            self._mcp_session_id = None
            self._emit("server_session_reset")
            raise TransportError("server session expired; re-initialize required")

        if self._auth.classify_response(response.status_code) is AuthOutcome.AUTH_FAILED:
            await response.aclose()
            raise TransportError(f"authentication failed ({response.status_code})")  # FRD-054

        # TDD §5.1: capture the session id the server assigns on ``initialize``
        # (header name is case-insensitive) so later requests can echo it. Once
        # we have a session, open the server->client GET stream (TDD §4.2).
        assigned = response.headers.get("mcp-session-id")
        if assigned:
            self._mcp_session_id = assigned
            if self._stream_task is None and not self._closed:
                self._stream_task = asyncio.create_task(self._read_server_stream())

        await self._drain_response(response)

    async def _send_with_reconnect(self, request: httpx.Request) -> httpx.Response:
        """Send a request, retrying **only** pre-delivery connect failures.

        A ``ConnectError``/``ConnectTimeout`` means the request never reached
        the server, so retrying is safe and does not re-execute a tool
        (FRD-045). A read timeout or any post-send error is **not** retried —
        it may have been delivered. Bounded by ``max_reconnect`` (FRD-056),
        with backoff; each attempt records a lifecycle event (FRD-043).
        """
        assert self._client is not None
        attempt = 0
        while True:
            try:
                return await self._client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # A TLS/certificate failure is permanent — retrying cannot help
                # and would consume the reconnect budget while hiding the real
                # cause. Report it clearly and immediately (FRD-040/046/047).
                tls = _tls_cause(exc)
                if tls is not None:
                    raise TransportError(
                        f"TLS certificate verification failed: {tls}"
                    ) from exc
                attempt += 1
                if attempt > self._max_reconnect:
                    raise TransportError(
                        f"could not connect after {self._max_reconnect} attempts"
                    ) from exc
                self._emit("reconnecting", attempt=attempt)
                await asyncio.sleep(min(0.1 * 2 ** (attempt - 1), 5.0))

    async def _follow_redirect(self, location: str, data: bytes, hop: int) -> None:
        if not location:
            raise TransportError("redirect without a location")
        # FRD-052: a redirect destination is a target — re-validate through the floor.
        verdict = self._floor.validate(location, after_redirect=True)
        if not verdict.allowed:
            raise TransportError(
                f"redirect target refused by safety floor: {verdict.reason.value}"
            )
        # R-1: pin the hop's just-validated IP so the redirected request connects
        # to exactly what the floor cleared — no re-resolution, no rebinding gap.
        await self._post(location, data, hop=hop + 1, pinned_ip=verdict.resolved_ip)

    async def _drain_response(self, response: httpx.Response) -> None:
        """Enqueue JSON-RPC message(s) from a JSON or SSE-framed response."""
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        await self._inbox.put(payload.encode("utf-8") + b"\n")
            await response.aclose()
            return
        body = await response.aread()
        await response.aclose()
        if body.strip():
            await self._inbox.put(body.rstrip(b"\n") + b"\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pinned_target(self, url: str, pinned_ip: str | None) -> str:
        """Rewrite *url*'s host to *pinned_ip* (SNI/Host preserved separately)."""
        if pinned_ip is None:
            return url
        parts = urlsplit(url)
        netloc = pinned_ip if parts.port is None else f"{pinned_ip}:{parts.port}"
        return parts._replace(netloc=netloc).geturl()

    def _default_client_factory(self, verify: str | bool) -> httpx.AsyncClient:
        # follow_redirects=False: we validate every hop ourselves (FRD-052).
        # trust_env=False: ignore *_PROXY so the floor's resolution is authoritative (FRD-057).
        # read=liveness_secs: transport-level liveness — the max gap between bytes
        # before the connection is declared dead (FRD-049/D15). Streaming progress
        # resets it, so an in-progress call is not killed (FRD-049 failure clause).
        timeout = httpx.Timeout(connect=10.0, read=self._liveness_secs, write=10.0, pool=10.0)
        return httpx.AsyncClient(
            verify=verify, trust_env=False, follow_redirects=False, timeout=timeout
        )
