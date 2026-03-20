"""Integration tests for trace end-to-end flow."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import load_hook_config
from clyro.hooks.models import HookInput, SessionState
from clyro.hooks.state import load_state, save_state
from clyro.hooks.tracer import handle_session_end, handle_tool_complete


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    pending = tmp_path / "pending"
    pending.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.backend.EVENT_QUEUE_DIR", pending):
        yield sessions


@pytest.fixture
def config_path(tmp_path):
    config_data = {
        "global": {"cost_per_token_usd": 0.00001},
        "backend": {"api_key": None},
        "audit": {
            "log_path": str(tmp_path / "audit.jsonl"),
            "redact_parameters": ["*password*"],
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config_data))
    return path


class TestToolCompleteFlow:
    def test_cost_adjustment(self, config_path, tmp_path, isolated_sessions):
        config = load_hook_config(str(config_path))

        # Create state with a pre-call estimate
        state = SessionState(
            session_id="trace-cost",
            step_count=3,
            accumulated_cost_usd=0.005,
            pre_call_cost_estimate=0.002,
        )
        save_state(state)

        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        hook_input = HookInput(
            session_id="trace-cost",
            tool_name="Bash",
            tool_input={"command": "ls -la"},
            tool_result={"stdout": "total 8\nfile1.txt\nfile2.txt", "exitCode": 0},
        )

        handle_tool_complete(hook_input, config, audit)
        audit.close()

        # Verify cost was adjusted
        updated = load_state("trace-cost")
        assert updated.pre_call_cost_estimate == 0.0
        # Should have replaced estimate with actual
        assert updated.accumulated_cost_usd != 0.005

    def test_writes_audit_entry(self, config_path, tmp_path, isolated_sessions):
        config = load_hook_config(str(config_path))

        state = SessionState(session_id="trace-audit", step_count=1)
        save_state(state)

        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        hook_input = HookInput(
            session_id="trace-audit",
            tool_name="Bash",
            tool_input={"command": "echo test"},
        )

        handle_tool_complete(hook_input, config, audit)
        audit.close()

        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "tool_call_observe"
        assert entry["tool_name"] == "Bash"


class TestSessionEndFlow:
    def test_session_summary(self, config_path, tmp_path, isolated_sessions):
        config = load_hook_config(str(config_path))

        state = SessionState(
            session_id="trace-end",
            step_count=25,
            accumulated_cost_usd=0.05,
        )
        save_state(state)

        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        hook_input = HookInput(session_id="trace-end")

        handle_session_end(hook_input, config, audit)
        audit.close()

        entry = json.loads(audit_path.read_text().strip())
        assert entry["event"] == "session_end"
        assert entry["total_steps"] == 25
        assert entry["total_cost_usd"] == 0.05
        assert entry["duration_seconds"] >= 0
