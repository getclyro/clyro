# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for clyro.mcp.terminal — MCP session summary + CLYRO_QUIET + error context.

Coverage targets:
- McpTerminalLogger.print_session_summary (all branches)
- is_quiet() with various CLYRO_QUIET values
- write_stderr silent failure
- McpTerminalLogger.format_error_with_context
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest

from clyro.mcp.terminal import (
    McpTerminalLogger,
    _ISSUE_TRACKER,
    is_quiet,
    write_stderr,
)


# ===========================================================================
# is_quiet()
# ===========================================================================


class TestIsQuiet:
    def test_default_not_quiet(self, monkeypatch):
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        assert is_quiet() is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES"])
    def test_quiet_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("CLYRO_QUIET", value)
        assert is_quiet() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "", "random"])
    def test_quiet_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("CLYRO_QUIET", value)
        assert is_quiet() is False


# ===========================================================================
# write_stderr
# ===========================================================================


class TestWriteStderr:
    def test_writes_to_stderr(self, capsys):
        write_stderr("hello test")
        captured = capsys.readouterr()
        assert "hello test" in captured.err
        assert captured.out == ""  # stdout must NEVER be touched

    def test_silent_on_closed_stderr(self):
        """write_stderr must not raise when stderr is closed."""
        with patch("clyro.mcp.terminal.print", side_effect=OSError("closed")):
            # Should not raise
            write_stderr("should not crash")


# ===========================================================================
# McpTerminalLogger.print_session_summary
# ===========================================================================


class TestSessionSummary:
    def test_basic_summary_local_mode(self, capsys, monkeypatch):
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=False)

        logger.print_session_summary(
            steps=5,
            cost_usd=0.001234,
            violations=[],
            controls_triggered=[],
        )

        captured = capsys.readouterr()
        assert "clyro-mcp governance summary" in captured.err
        assert "Steps:      5" in captured.err
        assert "$0.001234" in captured.err
        assert "Violations: 0" in captured.err
        assert "none triggered" in captured.err
        assert "Mode:       local" in captured.err
        assert "CLYRO_API_KEY" in captured.err  # Cloud CTA
        assert captured.out == ""

    def test_summary_cloud_mode_sync_ok(self, capsys, monkeypatch):
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=True)

        logger.print_session_summary(steps=10, cost_usd=0.05, sync_ok=True)

        captured = capsys.readouterr()
        assert "Mode:       cloud" in captured.err
        assert "Traces synced" in captured.err
        assert "CLYRO_API_KEY" not in captured.err  # No CTA when already cloud

    def test_summary_cloud_mode_sync_failed(self, capsys, monkeypatch):
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=True)

        logger.print_session_summary(steps=10, cost_usd=0.05, sync_ok=False)

        captured = capsys.readouterr()
        assert "Mode:       cloud" in captured.err
        assert "Trace sync failed" in captured.err
        assert "Traces synced" not in captured.err

    def test_summary_cloud_mode_sync_none(self, capsys, monkeypatch):
        """When sync_ok is None (no sync_manager), still shows synced."""
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=True)

        logger.print_session_summary(steps=10, cost_usd=0.05)

        captured = capsys.readouterr()
        assert "Mode:       cloud" in captured.err
        assert "Traces synced" in captured.err

    def test_summary_with_violations(self, capsys, monkeypatch):
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=False)

        logger.print_session_summary(
            steps=3,
            cost_usd=0.0,
            violations=[
                {"block_type": "policy_violation", "tool_name": "delete_db"},
                {"block_type": "budget_exceeded", "tool_name": "expensive_call"},
            ],
            controls_triggered=["budget_exceeded"],
        )

        captured = capsys.readouterr()
        assert "Violations: 2" in captured.err
        assert "delete_db" in captured.err
        assert "budget_exceeded" in captured.err

    def test_summary_suppressed_by_clyro_quiet(self, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        logger = McpTerminalLogger(is_backend_enabled=False)

        logger.print_session_summary(steps=5, cost_usd=0.01)

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_summary_suppressed_by_clyro_quiet_1(self, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "1")
        logger = McpTerminalLogger(is_backend_enabled=False)

        logger.print_session_summary(steps=5, cost_usd=0.01)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_summary_failsafe_on_exception(self, capsys, monkeypatch):
        """Summary must never crash — even if internal logic raises."""
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=False)

        # Pass bad data that would cause formatting errors
        logger.print_session_summary(
            steps="not_a_number",  # type: ignore
            cost_usd="bad",  # type: ignore
        )
        # Should not raise — fail-safe

    def test_summary_default_none_args(self, capsys, monkeypatch):
        """Violations and controls default to empty lists."""
        monkeypatch.delenv("CLYRO_QUIET", raising=False)
        logger = McpTerminalLogger(is_backend_enabled=False)

        logger.print_session_summary(steps=0, cost_usd=0.0)

        captured = capsys.readouterr()
        assert "Violations: 0" in captured.err
        assert "none triggered" in captured.err


# ===========================================================================
# McpTerminalLogger.format_error_with_context
# ===========================================================================


class TestErrorContext:
    def test_appends_issue_tracker(self):
        result = McpTerminalLogger.format_error_with_context(
            ValueError("something broke")
        )
        assert "something broke" in result
        assert _ISSUE_TRACKER in result

    def test_no_duplicate_if_already_enriched(self):
        """If error already contains the URL (ClyroError), don't duplicate."""
        from clyro.exceptions import ClyroError

        error = ClyroError("test error")
        result = McpTerminalLogger.format_error_with_context(error)
        assert result.count(_ISSUE_TRACKER) == 1

    def test_plain_exception(self):
        result = McpTerminalLogger.format_error_with_context(
            RuntimeError("timeout")
        )
        assert "timeout" in result
        assert "Report at" in result


# ===========================================================================
# Audit logger get_violations / get_controls_triggered
# ===========================================================================


class TestAuditSummaryAccessors:
    def test_violations_tracked_on_block(self, tmp_path):
        from clyro.config import AuditConfig
        from clyro.mcp.audit import AuditLogger
        from uuid import uuid4

        config = AuditConfig(
            log_path=str(tmp_path / "audit.jsonl"),
            redact_parameters=[],
        )
        audit = AuditLogger(config, uuid4())

        # Simulate an allowed call
        audit.log_tool_call(
            tool_name="safe_tool",
            parameters={},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.0,
        )
        assert audit.get_violations() == []
        assert audit.get_controls_triggered() == []

        # Simulate a blocked call
        audit.log_tool_call(
            tool_name="dangerous_tool",
            parameters={},
            decision="blocked",
            step_number=2,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={"rule_name": "no_delete"},
        )
        assert len(audit.get_violations()) == 1
        assert audit.get_violations()[0]["block_type"] == "policy_violation"
        assert audit.get_violations()[0]["tool_name"] == "dangerous_tool"

        # Simulate budget exceeded
        audit.log_tool_call(
            tool_name="expensive_tool",
            parameters={},
            decision="blocked",
            step_number=3,
            accumulated_cost_usd=10.0,
            block_reason="budget_exceeded",
        )
        assert len(audit.get_violations()) == 2
        assert "budget_exceeded" in audit.get_controls_triggered()

    def test_controls_not_duplicated(self, tmp_path):
        from clyro.config import AuditConfig
        from clyro.mcp.audit import AuditLogger
        from uuid import uuid4

        config = AuditConfig(
            log_path=str(tmp_path / "audit.jsonl"),
            redact_parameters=[],
        )
        audit = AuditLogger(config, uuid4())

        # Block twice with same reason
        for i in range(3):
            audit.log_tool_call(
                tool_name=f"tool_{i}",
                parameters={},
                decision="blocked",
                step_number=i,
                accumulated_cost_usd=0.0,
                block_reason="loop_detected",
            )

        # Control should appear only once
        assert audit.get_controls_triggered().count("loop_detected") == 1
        # But all violations are tracked
        assert len(audit.get_violations()) == 3

    def test_returns_copies(self, tmp_path):
        """Getters return copies, not internal references."""
        from clyro.config import AuditConfig
        from clyro.mcp.audit import AuditLogger
        from uuid import uuid4

        config = AuditConfig(
            log_path=str(tmp_path / "audit.jsonl"),
            redact_parameters=[],
        )
        audit = AuditLogger(config, uuid4())

        violations = audit.get_violations()
        violations.append({"fake": True})
        assert audit.get_violations() == []  # Internal state unaffected
