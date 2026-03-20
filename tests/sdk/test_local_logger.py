# Tests for local_logger.py — C5 (LocalTerminalLogger)
# Implements TDD §13.1 C5 test cases

"""
Test coverage targets: 90%+
"""

from __future__ import annotations

import os
from decimal import Decimal
from io import StringIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

from clyro.constants import APP_POLICIES_URL, APP_URL

import pytest

from clyro.config import ClyroConfig
from clyro.local_logger import LocalTerminalLogger, _is_quiet, reset_welcome_flag
from clyro.trace import EventType, TraceEvent


@pytest.fixture(autouse=True)
def _reset_welcome():
    reset_welcome_flag()
    yield
    reset_welcome_flag()


@pytest.fixture()
def local_config() -> ClyroConfig:
    return ClyroConfig(mode="local")


@pytest.fixture()
def logger(local_config: ClyroConfig) -> LocalTerminalLogger:
    return LocalTerminalLogger(local_config)


# ===========================================================================
# FRD-SOF-008: Welcome message
# ===========================================================================


class TestWelcome:
    def test_welcome_prints_once(self, logger: LocalTerminalLogger, capsys):
        logger.print_welcome()
        logger.print_welcome()  # second call should be no-op
        captured = capsys.readouterr()
        assert captured.err.count("[clyro]") == 2  # version + docs line
        assert "Runtime governance" in captured.err

    def test_welcome_suppressed_by_quiet(self, logger: LocalTerminalLogger, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        logger.print_welcome()
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_welcome_no_stdout(self, logger: LocalTerminalLogger, capsys):
        logger.print_welcome()
        captured = capsys.readouterr()
        assert captured.out == ""


# ===========================================================================
# FRD-SOF-006: Per-step policy event logging
# ===========================================================================


class TestLogEvent:
    def test_log_allow_event(self, logger: LocalTerminalLogger, capsys):
        event = TraceEvent(
            event_id=uuid4(),
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            session_id=uuid4(),
            step_number=1,
            metadata={
                "decision": "allow",
                "action_type": "llm_call",
                "evaluated_rules": 3,
                "rule_results": [
                    {"outcome": "passed"}, {"outcome": "passed"}, {"outcome": "skipped"},
                ],
            },
        )
        logger.log_event(event)
        captured = capsys.readouterr()
        assert "ALLOW" in captured.err
        assert "llm_call" in captured.err
        assert "0 violations" in captured.err

    def test_log_block_event(self, logger: LocalTerminalLogger, capsys):
        event = TraceEvent(
            event_id=uuid4(),
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            session_id=uuid4(),
            step_number=1,
            metadata={
                "decision": "block",
                "action_type": "tool_call",
                "rule_name": "no_internal",
                "rule_results": [{"outcome": "triggered"}],
            },
        )
        logger.log_event(event)
        captured = capsys.readouterr()
        assert "BLOCK" in captured.err
        assert "no_internal" in captured.err

    def test_log_suppressed_by_quiet(self, logger: LocalTerminalLogger, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        event = TraceEvent(
            event_id=uuid4(),
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            session_id=uuid4(),
            step_number=1,
            metadata={"decision": "allow", "action_type": "llm_call",
                       "evaluated_rules": 0, "rule_results": []},
        )
        logger.log_event(event)
        captured = capsys.readouterr()
        assert captured.err == ""


# ===========================================================================
# FRD-SOF-011: Violation context logging
# ===========================================================================


class TestLogViolation:
    def test_violation_context_format(self, logger: LocalTerminalLogger, capsys):
        logger.log_violation("tool_call", {
            "rule_name": "no_internal_endpoints",
            "parameter": "endpoint",
            "operator": "not_contains",
            "expected": "internal",
            "actual": "internal-api.corp.com",
        })
        captured = capsys.readouterr()
        assert "POLICY VIOLATION" in captured.err
        assert "no_internal_endpoints" in captured.err
        assert "endpoint" in captured.err
        assert "not_contains" in captured.err
        assert APP_POLICIES_URL in captured.err

    def test_violation_suppressed_by_quiet(self, logger: LocalTerminalLogger, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        logger.log_violation("tool_call", {"rule_name": "test"})
        captured = capsys.readouterr()
        assert captured.err == ""


# ===========================================================================
# FRD-SOF-007: Session-end governance summary
# ===========================================================================


class TestSessionSummary:
    def _make_session_mock(self):
        session = MagicMock()
        session.step_number = 23
        session.cumulative_cost = Decimal("0.047")
        session.session_id = uuid4()
        session.events = []
        return session

    def test_summary_format(self, logger: LocalTerminalLogger, capsys):
        session = self._make_session_mock()
        logger.print_session_summary(session)
        captured = capsys.readouterr()
        assert "governance summary" in captured.err
        assert "Steps:      23" in captured.err
        assert "$0.047" in captured.err
        assert "Mode:       local" in captured.err
        assert "CLYRO_API_KEY" in captured.err

    def test_summary_with_violations(self, logger: LocalTerminalLogger, capsys):
        session = self._make_session_mock()
        session.events = [
            TraceEvent(
                event_id=uuid4(),
                event_type=EventType.POLICY_CHECK,
                event_name="policy_check",
                session_id=session.session_id,
                step_number=5,
                metadata={
                    "decision": "block",
                    "action_type": "tool_call",
                    "rule_name": "no_internal",
                },
            ),
        ]
        logger.print_session_summary(session)
        captured = capsys.readouterr()
        assert "1" in captured.err
        assert "no_internal" in captured.err

    def test_summary_suppressed_by_quiet(self, logger: LocalTerminalLogger, capsys, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        session = self._make_session_mock()
        logger.print_session_summary(session)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_summary_cloud_mode_cta(self, capsys):
        """Cloud mode shows session URL instead of upgrade CTA."""
        config = ClyroConfig(
            mode="cloud", api_key="cly_test_abc123",
            agent_name="test",
        )
        cloud_logger = LocalTerminalLogger(config)
        session = self._make_session_mock()
        cloud_logger.print_session_summary(session)
        captured = capsys.readouterr()
        assert f"{APP_URL}/sessions/" in captured.err

    def test_summary_no_stdout(self, logger: LocalTerminalLogger, capsys):
        session = self._make_session_mock()
        logger.print_session_summary(session)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_summary_stderr_closed(self, logger: LocalTerminalLogger):
        """If stderr is closed, summary silently fails."""
        session = self._make_session_mock()
        with patch("clyro.local_logger._write_stderr", side_effect=OSError("closed")):
            # Should not raise
            logger.print_session_summary(session)


# ===========================================================================
# NFR-003: Quiet mode compliance
# ===========================================================================


class TestQuietMode:
    def test_is_quiet_true(self, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "true")
        assert _is_quiet() is True

    def test_is_quiet_True_uppercase(self, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "True")
        assert _is_quiet() is True

    def test_is_quiet_false_when_unset(self):
        # Ensure CLYRO_QUIET is not set
        os.environ.pop("CLYRO_QUIET", None)
        assert _is_quiet() is False

    def test_is_quiet_false_for_other_values(self, monkeypatch):
        monkeypatch.setenv("CLYRO_QUIET", "false")
        assert _is_quiet() is False
