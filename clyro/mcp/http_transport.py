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
import ipaddress
import json
import socket
import ssl
import sys
from collections.abc import Callable
from urllib.parse import SplitResult, urljoin, urlsplit

import httpx

from clyro.mcp.auth import AttachOutcome, AuthOutcome, CredentialProvider
from clyro.mcp.log import get_logger
from clyro.mcp.safety import REDIRECT_MAX_HOPS, SafetyFloor
from clyro.mcp.server_transport import SessionExpired, TransportError
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


def _describe(exc: BaseException) -> str:
    """Render *exc* for an operator, never as an empty string.

    httpx re-wraps low-level errors with no message of their own, so ``{exc}``
    yields "http transport error: " — a blank line nobody can act on. The real
    detail sits at the *root* of the __cause__ chain, several frames down::

        httpx.ReadError('')
          httpcore.ReadError('')
            anyio.BrokenResourceError('')
              ConnectionResetError('[Errno 104] Connection reset by peer')  <- this

    So name the top-level type and append the deepest message we can find.
    """
    text = str(exc).strip() or type(exc).__name__
    detail = ""
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if msg := str(cur).strip():
            detail = msg  # keep descending; the deepest message is the useful one
        cur = cur.__cause__ or cur.__context__
    if detail and detail not in text:
        text = f"{text} ({detail})"
    return text


# FRD-025: statuses that *may* mean "I don't know your session". Only these are
# body-inspected; every other status flows straight through untouched.
_SESSION_ERROR_CODES = frozenset({400, 404})


def _is_session_unknown(status: int, body: bytes) -> bool:
    """True if *status*/*body* say the server does not recognise our session.

    The MCP spec and the TypeScript SDK both answer **404** for an unknown session
    id. The reference server does not: server-everything routes on its own session
    map first and answers **400 "Bad Request: No valid session ID provided"**,
    never reaching the SDK's 404 path. Handling only 404 is therefore correct by
    the spec and useless against the most common real server — so match both.

    The 400 arm is narrowed by the body, because 400 is also the ordinary
    malformed-request status: treating every 400 as a dead session would trigger a
    pointless re-handshake on each bad request. A caller-side ``recovered`` guard
    bounds the damage if this heuristic is ever wrong.
    """
    if status == 404:
        return True
    if status == 400:
        return "session" in body.decode("utf-8", "replace").lower()
    return False


def _keepalive_socket_options(liveness_secs: int) -> list[tuple[int, int, int]]:
    """Build TCP-keepalive socket options meeting FRD-049/D15's liveness bound.

    D15 makes liveness a *transport* property, so it is enforced at the transport:
    the kernel probes the socket itself. A server that is alive but computing ACKs
    those probes and survives (FRD-049's failure clause — a call in progress must
    never be declared dead, however long it runs); a server whose host vanished
    stops ACKing and the socket errors out, surfacing as a TransportError.

    This replaces a read deadline, which measured whether the server was *talking*.
    Progress notifications are optional in MCP, so a silent long call reset nothing
    and was killed at the bound — the connection was never unhealthy (R-B5).

    Detection lands within ``liveness_secs``: idle + (count x interval) == bound.
    """
    idle = max(1, liveness_secs // 4)
    opts: list[tuple[int, int, int]] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]

    # The tuning knobs are platform-spelled: Linux uses TCP_KEEPIDLE, macOS
    # TCP_KEEPALIVE, and Windows exposes none of them via setsockopt.
    idle_opt = getattr(socket, "TCP_KEEPIDLE", None) or getattr(socket, "TCP_KEEPALIVE", None)
    intvl_opt = getattr(socket, "TCP_KEEPINTVL", None)
    cnt_opt = getattr(socket, "TCP_KEEPCNT", None)

    if idle_opt is None or intvl_opt is None or cnt_opt is None:
        # SO_KEEPALIVE alone defaults to a ~2h idle, so D15's bound is NOT met
        # here. Say so rather than let the caller assume a guarantee we can't keep.
        logger.warning(
            "liveness_bound_not_enforceable",
            reason="platform does not expose TCP keepalive tuning via setsockopt",
            platform=sys.platform,
            bound_secs=liveness_secs,
            effect="a dead peer may go undetected until the OS default keepalive expires",
        )
        return opts

    opts.append((socket.IPPROTO_TCP, idle_opt, idle))
    opts.append((socket.IPPROTO_TCP, intvl_opt, idle))
    opts.append((socket.IPPROTO_TCP, cnt_opt, 3))  # idle + 3*idle == liveness_secs
    return opts


def _host_header(parts: SplitResult) -> str:
    """Build the ``Host`` header, keeping a non-default port (RFC 7230 §5.4).

    An IPv6 host is re-bracketed; ``parts.hostname`` strips the brackets.
    """
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal — hostname strips the brackets
        host = f"[{host}]"
    default = {"http": 80, "https": 443}.get(parts.scheme)
    if parts.port is not None and parts.port != default:
        return f"{host}:{parts.port}"
    return host


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
        # FRD-025: the host's handshake, kept verbatim so a forgotten session can
        # be re-established without the host's involvement. See _reestablish_session.
        self._handshake: list[bytes] = []
        self._reestablishing = False  # guard: a replay must never recurse into itself
        # In-flight background body drains (B3). Held so close() can await them and
        # so the tasks are not garbage-collected mid-flight.
        self._drains: set[asyncio.Task[None]] = set()

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
            raise TransportError(f"target refused by safety floor: {verdict.reason.value}")
        self._pinned_ip = verdict.resolved_ip
        # verify= from TLS policy (FRD-040/046/047); trust_env=False ignores proxy
        # env so the floor's local resolution is authoritative (FRD-057).
        self._client = self._client_factory(self._tls.verify_value())

    async def send(self, data: bytes) -> None:
        """POST one JSON-RPC message; enqueue the reply(ies). Implements FRD-020."""
        if self._client is None or self._closed:
            raise TransportError("transport not open")
        self._remember_handshake(data)
        try:
            await self._post(self._url, data, hop=0)
        except httpx.TimeoutException as exc:
            # Reads are unbounded (FRD-049): a slow call is not a dead connection,
            # so this is only ever connect/write/pool. Liveness itself is enforced
            # by TCP keepalive and arrives below as a transport-level HTTPError.
            raise TransportError(f"connection timed out: {_describe(exc)}") from exc
        except httpx.HTTPError as exc:
            raise TransportError(f"http transport error: {_describe(exc)}") from exc
        except TransportError:
            raise
        except Exception as exc:
            # FRD-031 / ServerTransport contract: TransportError must be the ONLY
            # way a wire problem leaves this seam. Some httpx errors are not
            # HTTPError subclasses (InvalidURL derives straight from Exception),
            # and an escape here kills the router's host-reader task, which then
            # exits 0 with no error to the host — the silent-failure signature
            # this feature exists to remove.
            raise TransportError(f"http transport error: {exc!r}") from exc

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
        # B3: in-flight body drains hold a response and the connection. Cancel and
        # await them before closing the client, or their reads race a closed pool.
        for task in list(self._drains):
            task.cancel()
        if self._drains:
            await asyncio.gather(*list(self._drains), return_exceptions=True)
            self._drains.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await self._inbox.put(None)  # unblock a pending receive()

    async def _read_server_stream(self) -> None:
        """Open the server->client GET stream; enqueue server-initiated messages.

        The server→client leg of Streamable HTTP: a long-lived ``GET`` carrying
        the *unsolicited* server-initiated traffic (list_changed notifications,
        resource updates, sampling not tied to an in-flight call) that the router
        then delivers to the host (FRD-027/028). Sampling raised *during* a
        ``tools/call`` arrives on that POST's own stream instead.

        The credential is attached here exactly as it is on a POST (FRD-033) —
        the GET goes to the same origin, and without it a token-authenticated
        server 401s and this whole leg silently disappears.
        """
        assert self._client is not None
        parts = urlsplit(self._url)
        headers = {"Accept": "text/event-stream", "Host": _host_header(parts)}
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id
        # FRD-033/051/055: same origin, same rules as _post.
        attach = self._auth.attach(
            headers,
            target_origin=_origin(self._url),
            request_origin=_origin(self._url),
            encrypted=parts.scheme == "https",
        )
        if attach is AttachOutcome.REFUSED:
            self._emit("server_stream_unavailable", reason="credential_over_cleartext")
            return
        target = self._pinned_target(self._url, self._pinned_ip)  # GET stream = original origin
        request = self._client.build_request("GET", target, headers=headers)
        if parts.hostname:
            request.extensions["sni_hostname"] = parts.hostname
        try:
            response = await self._client.send(request, stream=True)
            # FRD-054/027: an auth rejection is NOT the same as "this server
            # offers no GET stream" (405). Conflating them hides a real auth
            # failure and drops server-initiated traffic without trace.
            if self._auth.classify_response(response.status_code) is AuthOutcome.AUTH_FAILED:
                await response.aclose()
                self._emit("server_stream_auth_failed", status=response.status_code)
                return
            if response.status_code >= 400:
                await response.aclose()
                self._emit("server_stream_unavailable", status=response.status_code)
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
        self,
        url: str,
        data: bytes,
        *,
        hop: int,
        pinned_ip: str | None = None,
        recovered: bool = False,
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
            # RFC 7230 §5.4: Host carries the port when it is not the scheme's
            # default. Because we connect to a pinned IP, Host IS the routing key
            # — dropping the port can land the request on the wrong vhost.
            "Host": _host_header(parts),
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
            await self._follow_redirect(location, data, hop, url)
            return

        # FRD-025: the server no longer recognises the session (it restarted, or
        # the session aged out). Recoverable — re-run the handshake and retry this
        # one request rather than ending the host's session.
        if self._mcp_session_id is not None and response.status_code in _SESSION_ERROR_CODES:
            # Buffering is safe here: a session rejection is a small JSON error, and
            # only these two statuses are read. httpx caches it, so if this turns
            # out NOT to be a session problem the body still reaches the host below.
            body = await response.aread()
            if _is_session_unknown(response.status_code, body):
                await response.aclose()
                # The id is dead whatever happens next. Clear it before deciding,
                # or a failed recovery leaves us echoing an id the server has
                # already rejected — and the host's own re-initialize (the last
                # fallback) would carry the stale id and be rejected too.
                self._mcp_session_id = None
                if recovered:
                    # Already re-handshook once for this message. Recovering again
                    # would loop forever against a server that rejects every
                    # session; fail this one call and let the router keep serving.
                    raise SessionExpired("server rejected the session it had just issued")
                self._emit("server_session_reset")
                if await self._reestablish_session():
                    # A session rejection is proof this request was never processed
                    # — refused before dispatch — so re-sending cannot duplicate a
                    # side effect. FRD-045 guards the *unknown* case (a timeout),
                    # which this is not.
                    await self._post(url, data, hop=hop, pinned_ip=pinned_ip, recovered=True)
                    return
                raise SessionExpired("server session expired and could not be re-established")

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

        # Everything above is header-only and stays inline, so send()'s error
        # contract is unchanged. Only the body drain goes to the background — that
        # is the single thing that blocked, and the whole of B3 (see _spawn_drain).
        self._spawn_drain(response, deliver=not self._reestablishing)

    def _remember_handshake(self, data: bytes) -> None:
        """Keep the host's handshake messages verbatim (FRD-025 recovery).

        The wrapper does not *own* the handshake — it must never invent one. But
        it sees the host's own ``initialize`` go past, and replaying those exact
        bytes re-establishes the session with the parameters the host negotiated,
        which is not the same as making them up.
        """
        if self._reestablishing:
            return  # a replay must not re-cache itself
        try:
            msg = json.loads(data)
        except (ValueError, TypeError):
            return
        method = msg.get("method") if isinstance(msg, dict) else None
        if method == "initialize":
            self._handshake = [data]  # a new handshake supersedes any earlier one
        elif method == "notifications/initialized" and self._handshake:
            # MCP requires this to follow initialize; replaying one without the
            # other leaves a server that gates on it refusing every request.
            self._handshake.append(data)

    async def _reestablish_session(self) -> bool:
        """Re-run the host's cached handshake after a 404. True if a new session was granted.

        Returns False (rather than raising) when there is nothing to replay or the
        replay fails, so the caller can report a clean error instead of a hang.
        """
        if not self._handshake or self._reestablishing:
            return False
        self._mcp_session_id = None  # the old id is dead; do not echo it
        self._reestablishing = True
        try:
            for message in self._handshake:
                await self._post(self._url, message, hop=0)
        except (TransportError, httpx.HTTPError) as exc:
            logger.warning("session_reestablish_failed", error=_describe(exc))
            return False
        finally:
            self._reestablishing = False

        if self._mcp_session_id is None:
            # The replay was accepted but no id came back — a stateless server.
            # Nothing to re-establish, and nothing was broken; let the retry run.
            return True
        self._emit("session_reestablished", session_id=self._mcp_session_id)
        return True

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
                    raise TransportError(f"TLS certificate verification failed: {tls}") from exc
                attempt += 1
                if attempt > self._max_reconnect:
                    raise TransportError(
                        f"could not connect after {self._max_reconnect} attempts"
                    ) from exc
                self._emit("reconnecting", attempt=attempt)
                await asyncio.sleep(min(0.1 * 2 ** (attempt - 1), 5.0))

    async def _follow_redirect(self, location: str, data: bytes, hop: int, base_url: str) -> None:
        if not location:
            raise TransportError("redirect without a location")
        # RFC 7231 §7.1.2 permits a relative Location ("/v2/mcp"). Resolving it
        # against the request URL is required for correctness — an unresolved
        # relative target has no host, so the floor rejects it as bad_url and a
        # protocol-legal server becomes unreachable. urljoin leaves an absolute
        # location untouched, so cross-origin redirects behave exactly as before.
        target = urljoin(base_url, location)
        # FRD-052: a redirect destination is a target — re-validate through the floor.
        verdict = self._floor.validate(target, after_redirect=True)
        if not verdict.allowed:
            raise TransportError(f"redirect target refused by safety floor: {verdict.reason.value}")
        # R-1: pin the hop's just-validated IP so the redirected request connects
        # to exactly what the floor cleared — no re-resolution, no rebinding gap.
        await self._post(target, data, hop=hop + 1, pinned_ip=verdict.resolved_ip)

    def _spawn_drain(self, response: httpx.Response, *, deliver: bool) -> None:
        """Read the response body in the background so ``send()`` does not block on it.

        THE B3 FIX. Draining inline made ``send()`` wait for the server's whole
        reply — but MCP lets a server ask the *client* something mid-tool-call
        (sampling/elicitation) and wait for the answer. The router issues that
        answer from the same task that is stuck inside ``send()``, so both sides
        waited forever: a silent hang, no error, no timeout.

        Only the BODY moves. The response headers have already been handled
        inline by the caller — status, redirect, 404/session, auth, session-id —
        so every error that used to surface from ``send()`` still does, and POSTs
        are still issued in order (send returns once headers arrive).
        """
        task = asyncio.create_task(self._drain_task(response, deliver=deliver))
        self._drains.add(task)
        task.add_done_callback(self._drains.discard)

    async def _drain_task(self, response: httpx.Response, *, deliver: bool) -> None:
        """Background body drain. A failure here ends the session, as it did inline."""
        try:
            await self._drain_response(response, deliver=deliver)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Inline, this raised out of send() and the router ended the session.
            # In the background there is no caller to raise to, so use the seam's
            # existing end-of-connection signal: receive() -> None makes the router
            # settle in-flight calls (FRD-026/045) and shut down. Never swallow it
            # — an unretrieved task exception is the silent failure B1 was about.
            logger.warning("response_drain_failed", error=_describe(exc))
            if not self._closed:
                await self._inbox.put(None)

    async def _drain_response(self, response: httpx.Response, *, deliver: bool = True) -> None:
        """Enqueue JSON-RPC message(s) from a JSON or SSE-framed response.

        ``deliver`` is decided by the CALLER, not read from ``self`` here: this
        runs as a background task (see _spawn_drain), so by the time it executes
        ``_reestablishing`` has already been reset and reading it would deliver a
        replayed handshake's reply to a host that never asked for one (R-C2).
        """
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload and deliver:
                        await self._inbox.put(payload.encode("utf-8") + b"\n")
            await response.aclose()
            return
        body = await response.aread()
        await response.aclose()
        if body.strip() and deliver:
            await self._inbox.put(body.rstrip(b"\n") + b"\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pinned_target(self, url: str, pinned_ip: str | None) -> str:
        """Rewrite *url*'s host to *pinned_ip* (SNI/Host preserved separately).

        An IPv6 literal MUST be bracketed (RFC 3986 §3.2.2) or the colons in the
        address are parsed as a port separator — ``https://fd00::5/mcp`` raises
        ``httpx.InvalidURL: Invalid port: ':5'``, which is not an ``HTTPError``
        and so escapes the transport seam entirely (FRD-031).
        """
        if pinned_ip is None:
            return url
        parts = urlsplit(url)
        host = pinned_ip
        if ipaddress.ip_address(pinned_ip).version == 6:
            host = f"[{pinned_ip}]"
        netloc = host if parts.port is None else f"{host}:{parts.port}"
        return parts._replace(netloc=netloc).geturl()

    def _default_client_factory(self, verify: str | bool) -> httpx.AsyncClient:
        # follow_redirects=False: we validate every hop ourselves (FRD-052).
        # trust_env=False: ignore *_PROXY so the floor's resolution is authoritative (FRD-057).
        # read=None: liveness is a TCP property (see _keepalive_socket_options), NOT
        # a read deadline. A read deadline answers "is the server talking?"; FRD-049
        # asks "is the connection alive?". A silent in-progress call must never be
        # declared dead however long it runs, so nothing here may bound a read.
        timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
        transport = httpx.AsyncHTTPTransport(
            verify=verify,
            trust_env=False,
            retries=0,  # FRD-045: never re-send a possibly-delivered tools/call
            socket_options=_keepalive_socket_options(self._liveness_secs),
        )
        return httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False, timeout=timeout
        )
