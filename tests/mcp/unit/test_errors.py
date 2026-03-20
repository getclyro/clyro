"""
Unit tests for ErrorFormatter — TDD §11.1 tests #19–#23.
"""

from __future__ import annotations

import json

from clyro.mcp.errors import format_error


class TestErrorFormatter:
    """JSON-RPC 2.0 error response construction."""

    def _parse(self, raw: str) -> dict:
        return json.loads(raw.strip())

    def test_loop_detected(self) -> None:
        """TDD §11.1 #19 — correct structure for loop."""
        raw = format_error(
            request_id=42,
            block_type="loop_detected",
            details={
                "tool_name": "read_file",
                "repetition_count": 3,
                "threshold": 3,
                "pattern_hash": "abc123",
            },
        )
        msg = self._parse(raw)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 42
        assert msg["error"]["code"] == -32600
        assert "loop_detected" in msg["error"]["data"]["type"]
        assert msg["error"]["data"]["details"]["repetition_count"] == 3

    def test_step_limit(self) -> None:
        """TDD §11.1 #20 — correct structure for step limit."""
        raw = format_error(
            request_id=7,
            block_type="step_limit_exceeded",
            details={"step_count": 51, "max_steps": 50},
        )
        msg = self._parse(raw)
        assert msg["error"]["data"]["type"] == "step_limit_exceeded"
        assert msg["error"]["data"]["details"]["step_count"] == 51

    def test_budget_exceeded(self) -> None:
        """TDD §11.1 #21 — cost_estimated: true."""
        raw = format_error(
            request_id="abc",
            block_type="budget_exceeded",
            details={
                "accumulated_cost_usd": 10.5,
                "max_cost_usd": 10.0,
                "cost_estimated": True,
            },
        )
        msg = self._parse(raw)
        assert msg["error"]["data"]["type"] == "budget_exceeded"
        assert msg["error"]["data"]["details"]["cost_estimated"] is True

    def test_policy_violation(self) -> None:
        """TDD §11.1 #22 — rule details present."""
        raw = format_error(
            request_id=99,
            block_type="policy_violation",
            details={
                "rule_name": "no-drop",
                "tool_name": "query_database",
                "parameter": "sql",
                "operator": "contains",
                "expected": "DROP",
                "actual": "DROP TABLE users",
            },
        )
        msg = self._parse(raw)
        assert msg["error"]["data"]["type"] == "policy_violation"
        assert msg["error"]["data"]["details"]["rule_name"] == "no-drop"

    def test_preserves_request_id(self) -> None:
        """TDD §11.1 #23 — original request id in error response."""
        for rid in [1, "abc-123", None]:
            raw = format_error(rid, "loop_detected", {})
            msg = self._parse(raw)
            assert msg["id"] == rid

    def test_newline_terminated(self) -> None:
        """Each error response ends with \\n for newline-delimited framing."""
        raw = format_error(1, "loop_detected", {})
        assert raw.endswith("\n")

    def test_message_prefix(self) -> None:
        """Error message starts with 'ClyroPolicy:'."""
        raw = format_error(1, "budget_exceeded", {})
        msg = self._parse(raw)
        assert msg["error"]["message"].startswith("ClyroPolicy:")
