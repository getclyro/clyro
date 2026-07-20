# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Transport selection
# Implements FRD-021, FRD-022, FRD-042

"""
Resolve and validate which downstream transport to use.

- Unset transport defaults to STDIO (FRD-022).
- An unrecognised transport value refuses startup (FRD-021).
- HTTP requires a target URL (FRD-042); STDIO requires a server command.

Pure logic: no I/O, no connection. The caller (CLI) turns a
:class:`SelectionError` into a refuse-to-start with a named cause.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_TRANSPORTS = ("stdio", "http")


class SelectionError(Exception):
    """Invalid transport/target combination — the wrapper must refuse to start."""


@dataclass(frozen=True)
class Selection:
    transport: str
    server_command: list[str] | None
    url: str | None


def select_transport(
    *, transport: str | None, url: str | None, server_command: list[str]
) -> Selection:
    """Validate and resolve the transport selection (FRD-021/022/042)."""
    choice = (transport or "stdio").lower()  # FRD-022: default stdio

    if choice not in _VALID_TRANSPORTS:  # FRD-021
        raise SelectionError(
            f"unknown transport {transport!r}; expected one of {_VALID_TRANSPORTS}"
        )

    if choice == "http":
        if not url:  # FRD-042
            raise SelectionError("transport 'http' requires a target url")
        return Selection("http", None, url)

    # stdio
    if not server_command:
        raise SelectionError("transport 'stdio' requires a server command")
    return Selection("stdio", server_command, None)
