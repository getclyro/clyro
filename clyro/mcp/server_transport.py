# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Server Transport seam
# Implements FRD-031 (transport-blind router; transport errors distinct from blocks)

"""
The transport seam.

``MessageRouter`` and the governance stack must not know whether the downstream
leg is a local child process or a remote HTTP connection (TDD §1, §4.1). Both
``StdioTransport`` (existing) and ``HttpTransport`` (new) conform to the
:class:`ServerTransport` protocol below.

Transports signal wire/connection problems by raising :class:`TransportError`.
The router maps that to a host-visible *transport* error — never a governance
block (FRD-031) — so the host can tell "the server refused this" apart from
"policy refused this".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class TransportError(Exception):
    """A wire/connection failure on the downstream leg.

    Distinct by type from any governance decision so the router never reports a
    transport failure as a policy block (FRD-031).
    """


@runtime_checkable
class ServerTransport(Protocol):
    """Uniform async interface over the wrapper's downstream leg (TDD §4.1)."""

    async def open(self) -> None:
        """Establish the downstream leg (spawn child | connect + validate).

        Raises:
            TransportError: if the leg cannot be established.
        """
        ...

    async def send(self, data: bytes) -> None:
        """Send one JSON-RPC message downstream (host -> server)."""
        ...

    async def receive(self) -> bytes | None:
        """Return the next message from the server, or ``None`` on clean EOF."""
        ...

    async def close(self) -> None:
        """Tear the leg down (terminate child | DELETE session)."""
        ...

    def is_live(self) -> bool:
        """True while the downstream leg is usable (transport-level, FRD-049)."""
        ...
