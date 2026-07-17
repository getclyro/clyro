# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Outbound authentication (native HTTP transport)
# Implements FRD-033, FRD-051, FRD-054, FRD-055

"""
Outbound authentication for the native HTTP transport.

v1 ships a static bearer-token / static-header credential (D21). Authentication
sits behind the mechanism-agnostic :class:`AuthProvider` protocol so a future
mechanism (OAuth, mTLS, …) is an *additive* implementation — no change to
transport, governance, lifecycle, or policy (D21).

The static :class:`CredentialProvider`:

- attaches the credential header to an outbound request (FRD-033);
- withholds it on a cross-origin redirect (FRD-051) — a token must never follow
  a redirect to a different origin;
- refuses to send it over an unencrypted, non-loopback connection (FRD-055) —
  a credential never travels a plaintext wire;
- classifies a server response so the caller can detect a mid-session
  credential rejection (FRD-054).
"""

from __future__ import annotations

import ipaddress
from enum import Enum
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from clyro.mcp.log import get_logger

logger = get_logger(__name__)


class AttachOutcome(Enum):
    ATTACHED = "attached"  # credential added to the request
    OMITTED = "omitted"  # no credential configured, or withheld cross-origin
    REFUSED = "refused"  # would require sending a credential over cleartext


class AuthOutcome(Enum):
    OK = "ok"
    AUTH_FAILED = "auth_failed"  # server rejected the credential (401/403)


@runtime_checkable
class AuthProvider(Protocol):
    """Mechanism-agnostic outbound-auth seam (D21).

    A new mechanism implements this protocol; nothing else in the wrapper
    changes.
    """

    def attach(
        self, headers: dict[str, str], *, target_origin: str, request_origin: str, encrypted: bool
    ) -> AttachOutcome:
        """Add credentials to *headers* for this request, or decline to."""
        ...

    def classify_response(self, status_code: int) -> AuthOutcome:
        """Map a server status to OK / AUTH_FAILED (FRD-054)."""
        ...


def _is_loopback_origin(origin: str) -> bool:
    """True if *origin* points at this machine (FRD-055 / AC-8.7 exemption).

    Parsed with ``urlsplit`` rather than string-splitting on ``:`` — the naive
    form yields ``'['`` for ``http://[::1]:9000`` and so refuses the credential
    on IPv6 loopback. The whole of ``127.0.0.0/8`` is loopback, not just
    ``127.0.0.1``, so the check is by address, not by literal.
    """
    host = urlsplit(origin).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class CredentialProvider:
    """Static bearer-token / static-header credential (D21). Implements FRD-033.

    Args:
        header_name: header to carry the credential (default ``Authorization``).
        header_value: the exact value to send (e.g. ``"Bearer <token>"`` or
            ``"Basic <base64>"``). ``None`` = no credential (unauthenticated).
    """

    def __init__(self, header_value: str | None, *, header_name: str = "Authorization") -> None:
        self._header_name = header_name
        self._header_value = header_value

    @property
    def credential_value(self) -> str | None:
        """The exact transmitted credential string (for redaction, FRD-034)."""
        return self._header_value

    def attach(
        self,
        headers: dict[str, str],
        *,
        target_origin: str,
        request_origin: str,
        encrypted: bool,
    ) -> AttachOutcome:
        """Attach the credential, or decline. Implements FRD-033/051/055."""
        if self._header_value is None:
            return AttachOutcome.OMITTED

        # FRD-051: never send the credential to a different origin (redirect).
        if target_origin != request_origin:
            logger.warning("cred_withheld_cross_origin", target=target_origin)
            return AttachOutcome.OMITTED

        # FRD-055: never send the credential over an unencrypted, non-loopback
        # connection. Loopback is exempt — nothing is on a wire.
        if not encrypted and not _is_loopback_origin(target_origin):
            logger.warning("cred_refused_plaintext", target=target_origin)
            return AttachOutcome.REFUSED

        headers[self._header_name] = self._header_value
        return AttachOutcome.ATTACHED

    def classify_response(self, status_code: int) -> AuthOutcome:
        """A 401/403 is a credential rejection (FRD-054)."""
        if status_code in (401, 403):
            return AuthOutcome.AUTH_FAILED
        return AuthOutcome.OK
