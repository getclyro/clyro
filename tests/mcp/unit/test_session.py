"""
Unit tests for McpSession and PendingCall.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from clyro.mcp.session import McpSession, PendingCall


class TestMcpSession:
    """Session state container."""

    def test_defaults(self) -> None:
        s = McpSession()
        assert isinstance(s.session_id, UUID)
        assert s.step_count == 0
        assert s.accumulated_cost_usd == 0.0

    def test_increment_step(self) -> None:
        s = McpSession()
        assert s.increment_step() == 1
        assert s.increment_step() == 2

    def test_add_cost(self) -> None:
        s = McpSession()
        s.add_cost(0.005)
        s.add_cost(0.003)
        assert abs(s.accumulated_cost_usd - 0.008) < 1e-10

    def test_custom_session_id(self) -> None:
        """Session can be created with a deterministic ID."""
        sid = UUID("00000000-0000-0000-0000-000000000001")
        s = McpSession(session_id=sid)
        assert s.session_id == sid


class TestPendingCall:
    """Frozen dataclass for pending request tracking."""

    def test_frozen(self) -> None:
        pc = PendingCall(request_id=1, tool_name="t", params_json_len=10, forwarded_at=0.0)
        assert pc.request_id == 1
        assert pc.tool_name == "t"

    def test_immutable(self) -> None:
        pc = PendingCall(request_id=1, tool_name="t", params_json_len=10, forwarded_at=0.0)
        with pytest.raises(AttributeError):
            pc.request_id = 2  # type: ignore[misc]
