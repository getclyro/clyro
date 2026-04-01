"""
Unit tests for AuditLogger backend integration (dual-mode emission).

Covers:
- set_backend attaches sync_manager and trace_factory
- log_tool_call emits trace event for allowed calls
- log_tool_call emits trace event for blocked calls
- log_lifecycle emits session_start and session_end trace events
- Backend sync failure does not block audit write (fail-open)
- File permission check on existing files (NFR-005)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

from clyro.config import AuditConfig
from clyro.mcp.audit import AuditLogger


def _make_logger(
    tmp_dir: str,
    redact: list[str] | None = None,
) -> AuditLogger:
    log_path = os.path.join(tmp_dir, "audit.jsonl")
    cfg = AuditConfig(log_path=log_path, redact_parameters=redact or [])
    return AuditLogger(cfg, session_id=uuid4())


class TestAuditBackendDualMode:
    """Dual-mode emission: local JSONL + backend trace events."""

    def test_set_backend_attaches_components(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        al.set_backend(sync, factory)
        assert al._sync_manager is sync
        assert al._trace_factory is factory

    def test_allowed_call_emits_policy_check_and_tool_call_act(self, tmp_path: Path) -> None:
        """Allowed tool call should emit session_start + policy_check + tool_call_act."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        act_event_id = str(uuid4())
        factory.session_start.return_value = {"event_type": "session_start"}
        factory.policy_check.return_value = {"event_type": "policy_check"}
        factory.tool_call_act.return_value = {"event_type": "tool_call", "event_id": act_event_id}
        al.set_backend(sync, factory)

        # session_start is deferred — not sent yet
        al.log_lifecycle("session_start")
        sync.enqueue.assert_not_called()

        al.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/tmp"},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.001,
            duration_ms=2,
        )

        factory.tool_call_act.assert_called_once_with(
            tool_name="read_file",
            params={"path": "/tmp"},
            step_number=1,
            duration_ms=2,
        )
        factory.policy_check.assert_called_once_with(
            tool_name="read_file",
            params={"path": "/tmp"},
            duration_ms=2,
            decision="allow",
            rule_results=None,
            parent_event_id=act_event_id,
        )
        # session_start + policy_check + tool_call_act = 3 enqueue calls
        assert sync.enqueue.call_count == 3

    def test_blocked_call_emits_policy_check_and_blocked_trace(self, tmp_path: Path) -> None:
        """Blocked tool call should emit policy_check + blocked_call trace events."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        act_event_id = str(uuid4())
        factory.tool_call_act.return_value = {"event_type": "tool_call", "event_id": act_event_id}
        factory.policy_check.return_value = {"event_type": "policy_check"}
        factory.blocked_call.return_value = {"event_type": "error"}
        al.set_backend(sync, factory)

        al.log_tool_call(
            tool_name="transfer",
            parameters={"amount": 9999},
            decision="blocked",
            step_number=3,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={"rule_name": "max_amount"},
        )

        factory.policy_check.assert_called_once_with(
            tool_name="transfer",
            params={"amount": 9999},
            duration_ms=0,
            decision="block",
            rule_results=None,
            parent_event_id=act_event_id,
        )
        factory.blocked_call.assert_called_once_with(
            tool_name="transfer",
            block_type="policy_violation",
            block_message="Blocked by policy_violation",
            block_details={"rule_name": "max_amount"},
        )
        # policy_check + blocked_call = 2 (ACT event not enqueued for blocked calls)
        assert sync.enqueue.call_count == 2

    def test_session_start_deferred_until_tool_call(self, tmp_path: Path) -> None:
        """session_start is buffered, not sent to backend until first tool call."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        factory.session_start.return_value = {"event_type": "session_start"}
        al.set_backend(sync, factory)

        al.log_lifecycle("session_start")
        factory.session_start.assert_called_once()
        # NOT enqueued yet — deferred
        sync.enqueue.assert_not_called()

    def test_discovery_session_skips_backend(self, tmp_path: Path) -> None:
        """Sessions with no tool calls (e.g. get_tools() discovery) skip backend entirely."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        factory.session_start.return_value = {"event_type": "session_start"}
        al.set_backend(sync, factory)

        al.log_lifecycle("session_start")
        al.log_lifecycle("session_end")
        # No tool calls → neither session_start nor session_end sent to backend
        sync.enqueue.assert_not_called()

    def test_session_end_emits_after_tool_call(self, tmp_path: Path) -> None:
        """session_end is emitted to backend only if session had tool activity."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        act_event_id = str(uuid4())
        factory.session_start.return_value = {"event_type": "session_start"}
        factory.policy_check.return_value = {"event_type": "policy_check"}
        factory.tool_call_act.return_value = {"event_type": "tool_call", "event_id": act_event_id}
        factory.session_end.return_value = {"event_type": "session_end"}
        al.set_backend(sync, factory)

        al.log_lifecycle("session_start")
        al.log_tool_call("read_file", {"path": "/"}, "allowed", 1, 0.0, duration_ms=0)
        al.log_lifecycle("session_end")

        factory.session_end.assert_called_once()
        # session_start + policy_check + tool_call_act + session_end = 4
        assert sync.enqueue.call_count == 4

    def test_server_exited_emits_after_tool_call(self, tmp_path: Path) -> None:
        """server_exited emits session_end trace only if session had tool activity."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        factory.session_start.return_value = {"event_type": "session_start"}
        factory.policy_check.return_value = {"event_type": "policy_check"}
        factory.tool_call_act.return_value = {"event_type": "tool_call", "event_id": str(uuid4())}
        factory.create_trace_event.return_value = {"event_type": "session_end"}
        al.set_backend(sync, factory)

        al.log_lifecycle("session_start")
        al.log_tool_call("read_file", {"path": "/"}, "allowed", 1, 0.0, duration_ms=0)
        al.log_lifecycle("server_exited", extra={"exit_code": 1})

        factory.create_trace_event.assert_called_once_with(
            "session_end",
            None,
            metadata={"reason": "server_exited", "exit_code": 1},
        )

    def test_server_exited_skips_backend_without_tool_call(self, tmp_path: Path) -> None:
        """server_exited skips backend if no tool calls occurred."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        factory = MagicMock()
        factory.session_start.return_value = {"event_type": "session_start"}
        al.set_backend(sync, factory)

        al.log_lifecycle("session_start")
        al.log_lifecycle("server_exited", extra={"exit_code": 0})
        sync.enqueue.assert_not_called()

    def test_backend_sync_failure_does_not_block(self, tmp_path: Path) -> None:
        """Backend sync failure should not prevent audit write."""
        al = _make_logger(str(tmp_path))
        sync = MagicMock()
        sync.enqueue.side_effect = RuntimeError("sync broken")
        factory = MagicMock()
        factory.tool_call_act.return_value = {"event_type": "tool_call", "event_id": str(uuid4())}
        al.set_backend(sync, factory)

        # Should NOT raise
        al.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/tmp"},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.0,
        )

        # Local audit file should still be written
        al.close()
        entries = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(entries) == 1
        assert json.loads(entries[0])["decision"] == "allowed"


class TestAuditFilePermissions:
    """File permission handling (NFR-005)."""

    def test_new_file_created_with_0o600(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        al.log_lifecycle("session_start")
        al.close()

        log_path = tmp_path / "audit.jsonl"
        mode = os.stat(str(log_path)).st_mode & 0o777
        assert mode == 0o600

    def test_existing_file_with_wrong_perms_tightened(
        self, tmp_path: Path, capsys
    ) -> None:
        """Existing file with 0o644 should be tightened to 0o600."""
        log_path = tmp_path / "audit.jsonl"
        log_path.touch()
        os.chmod(str(log_path), 0o644)

        al = _make_logger(str(tmp_path))
        al.log_lifecycle("session_start")
        al.close()

        mode = os.stat(str(log_path)).st_mode & 0o777
        assert mode == 0o600
        captured = capsys.readouterr()
        assert "tightening" in captured.err.lower()


class TestAuditViolationReporting:
    """Violation reporter integration (FRD-006)."""

    def test_policy_violation_block_enqueues_report(self, tmp_path: Path) -> None:
        """policy_violation block should call violation reporter."""
        al = _make_logger(str(tmp_path))
        reporter = MagicMock()
        al.set_violation_reporter(reporter, agent_id="agent-123")

        al.log_tool_call(
            tool_name="transfer",
            parameters={"amount": 9999},
            decision="blocked",
            step_number=1,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={
                "rule_name": "max_amount",
                "operator": "max_value",
                "expected": 1000,
                "actual": 9999,
                "policy_id": "some-uuid",
            },
        )

        reporter.assert_called_once()
        report = reporter.call_args[0][0]
        assert report["agent_id"] == "agent-123"
        assert report["action_type"] == "transfer"
        assert report["rule_name"] == "max_amount"
        assert report["operator"] == "max_value"
        assert report["decision"] == "block"
        assert report["policy_id"] == "some-uuid"
        assert len(report["parameters_hash"]) == 64

    def test_loop_detected_block_does_not_enqueue(self, tmp_path: Path) -> None:
        """loop_detected block should NOT call violation reporter."""
        al = _make_logger(str(tmp_path))
        reporter = MagicMock()
        al.set_violation_reporter(reporter, agent_id="agent-123")

        al.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/"},
            decision="blocked",
            step_number=1,
            accumulated_cost_usd=0.0,
            block_reason="loop_detected",
            block_details={"count": 3},
        )

        reporter.assert_not_called()

    def test_allowed_call_does_not_enqueue(self, tmp_path: Path) -> None:
        """Allowed call should NOT call violation reporter."""
        al = _make_logger(str(tmp_path))
        reporter = MagicMock()
        al.set_violation_reporter(reporter, agent_id="agent-123")

        al.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/"},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.0,
        )

        reporter.assert_not_called()

    def test_reporter_failure_does_not_block(self, tmp_path: Path) -> None:
        """Reporter failure should not prevent audit write."""
        al = _make_logger(str(tmp_path))
        reporter = MagicMock(side_effect=RuntimeError("reporter broken"))
        al.set_violation_reporter(reporter, agent_id="agent-123")

        # Should NOT raise
        al.log_tool_call(
            tool_name="transfer",
            parameters={"amount": 9999},
            decision="blocked",
            step_number=1,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={"rule_name": "max_amount", "operator": "max_value"},
        )

        # Local audit file should still be written
        al.close()
        entries = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(entries) == 1
        assert json.loads(entries[0])["decision"] == "blocked"

    def test_no_reporter_no_error(self, tmp_path: Path) -> None:
        """Without set_violation_reporter, no error on policy violation block."""
        al = _make_logger(str(tmp_path))
        # No reporter attached — should not raise
        al.log_tool_call(
            tool_name="transfer",
            parameters={"amount": 9999},
            decision="blocked",
            step_number=1,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={"rule_name": "max_amount"},
        )
        al.close()


class TestAuditNoBackend:
    """AuditLogger without backend attached."""

    def test_no_backend_no_trace_emission(self, tmp_path: Path) -> None:
        """Without set_backend, no trace events are emitted."""
        al = _make_logger(str(tmp_path))
        # Should work without any backend
        al.log_tool_call("read_file", {"path": "/"}, "allowed", 1, 0.0)
        al.log_lifecycle("session_start")
        al.log_lifecycle("session_end")
        al.close()

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
