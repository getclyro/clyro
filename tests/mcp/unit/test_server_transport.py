# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the ServerTransport seam and StdioTransport's conformance —
FRD-031.
"""

from __future__ import annotations

import json
import sys

import pytest

from clyro.mcp.server_transport import ServerTransport, TransportError
from clyro.mcp.transport import StdioTransport


def test_transport_error_is_exception() -> None:
    assert issubclass(TransportError, Exception)


def test_stdio_transport_conforms_to_protocol() -> None:
    # runtime_checkable Protocol: StdioTransport must satisfy the seam.
    t = StdioTransport([sys.executable, "-c", "pass"])
    assert isinstance(t, ServerTransport)


class TestStdioConformanceMethods:
    """open/send/receive/close/is_live delegate correctly (FRD-031)."""

    @pytest.mark.asyncio
    async def test_open_send_receive_close(self) -> None:
        # Echo one line back on stdout.
        script = (
            "import sys; line=sys.stdin.readline(); "
            "sys.stdout.write(line); sys.stdout.flush()"
        )
        t = StdioTransport([sys.executable, "-c", script])
        await t.open()
        assert t.is_live() is True

        await t.send(b'{"jsonrpc":"2.0","id":7}\n')
        line = await t.receive()
        assert line is not None
        assert json.loads(line)["id"] == 7

        await t.close()

    @pytest.mark.asyncio
    async def test_is_live_false_before_open(self) -> None:
        t = StdioTransport([sys.executable, "-c", "pass"])
        assert t.is_live() is False

    @pytest.mark.asyncio
    async def test_is_live_false_after_exit(self) -> None:
        t = StdioTransport([sys.executable, "-c", "pass"])
        await t.open()
        await t.process.wait()
        assert t.is_live() is False

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_eof(self) -> None:
        # Child exits immediately -> stdout EOF -> receive() yields None.
        t = StdioTransport([sys.executable, "-c", "pass"])
        await t.open()
        assert await t.receive() is None
        await t.close()
