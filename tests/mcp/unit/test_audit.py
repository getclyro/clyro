"""
Unit tests for AuditLogger — TDD §11.1 tests #28–#32.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from clyro.mcp.audit import AuditLogger
from clyro.config import AuditConfig


def _make_logger(
    tmp_dir: str,
    redact: list[str] | None = None,
) -> AuditLogger:
    log_path = os.path.join(tmp_dir, "audit.jsonl")
    cfg = AuditConfig(log_path=log_path, redact_parameters=redact or [])
    return AuditLogger(cfg, session_id=uuid4())


class TestAuditToolCallEntry:
    """TDD §11.1 #28 — correct JSONL fields for allowed call."""

    def test_allowed_entry(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        al.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/tmp/a.txt"},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.001,
        )
        al.close()

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "tool_call"
        assert entry["tool_name"] == "read_file"
        assert entry["decision"] == "allowed"
        assert entry["step_number"] == 1
        assert entry["cost_estimated"] is True
        assert "timestamp" in entry
        assert "session_id" in entry


class TestAuditBlockedEntry:
    """TDD §11.1 #29 — correct fields with block_reason."""

    def test_blocked_entry(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        al.log_tool_call(
            tool_name="transfer",
            parameters={"amount": 9999},
            decision="blocked",
            step_number=3,
            accumulated_cost_usd=0.0,
            block_reason="policy_violation",
            block_details={"rule_name": "max_amount"},
        )
        al.close()

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["decision"] == "blocked"
        assert entry["block_reason"] == "policy_violation"
        assert entry["block_details"]["rule_name"] == "max_amount"


class TestAuditRedaction:
    """TDD §11.1 #30 — *.password redacted to [REDACTED]."""

    def test_redaction(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path), redact=["*.password", "*.secret"])
        al.log_tool_call(
            tool_name="login",
            parameters={"username": "alice", "password": "s3cret", "secret": "key123"},
            decision="allowed",
            step_number=1,
            accumulated_cost_usd=0.0,
        )
        al.close()

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["parameters"]["username"] == "alice"
        assert entry["parameters"]["password"] == "[REDACTED]"
        assert entry["parameters"]["secret"] == "[REDACTED]"


class TestAuditDiskFullResilience:
    """TDD §11.1 #31 — IOError on write → logged to stderr, not raised."""

    def test_disk_full_no_raise(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        al = _make_logger(str(tmp_path))
        # Force open so we can sabotage the fd
        al.log_tool_call("t", {}, "allowed", 1, 0.0)
        al.close()

        # Now point at an unwritable path
        al._log_path = Path("/proc/0/impossible")
        al._fd = None

        # This should NOT raise
        al.log_tool_call("t", {}, "allowed", 2, 0.0)
        captured = capsys.readouterr()
        assert "audit_write_error" in captured.err


class TestAuditCreatesDirectory:
    """TDD §11.1 #32 — non-existent directory created."""

    def test_creates_dir(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "sub" / "dir" / "audit.jsonl"
        cfg = AuditConfig(log_path=str(deep_path))
        al = AuditLogger(cfg, session_id=uuid4())
        al.log_lifecycle("session_start")
        al.close()
        assert deep_path.exists()


class TestAuditLifecycle:
    """Lifecycle events (session_start, session_end, server_exited)."""

    def test_lifecycle_events(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        al.log_lifecycle("session_start")
        al.log_lifecycle("server_exited", extra={"exit_code": 1})
        al.close()

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "session_start"
        entry = json.loads(lines[1])
        assert entry["event"] == "server_exited"
        assert entry["exit_code"] == 1


class TestAuditParseError:
    """Parse error logging."""

    def test_parse_error(self, tmp_path: Path) -> None:
        al = _make_logger(str(tmp_path))
        al.log_parse_error(b"{{not json")
        al.close()

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["event"] == "parse_error"
        assert "not json" in entry["raw_preview"]
