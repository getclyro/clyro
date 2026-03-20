"""Integration tests for evaluate end-to-end flow."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import load_hook_config
from clyro.hooks.evaluator import evaluate
from clyro.hooks.models import HookInput, SessionState
from clyro.hooks.state import StateLock, load_state, save_state


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path):
    """Redirect state and event queue to temp dir for isolation."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    pending = tmp_path / "pending"
    pending.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.backend.EVENT_QUEUE_DIR", pending):
        yield sessions


@pytest.fixture
def config_path(tmp_path):
    """Create a test config file."""
    config_data = {
        "global": {
            "max_steps": 10,
            "max_cost_usd": 1.0,
            "cost_per_token_usd": 0.00001,
            "loop_detection": {"threshold": 3, "window": 5},
            "policies": [
                {
                    "parameter": "command",
                    "operator": "contains",
                    "value": "rm -rf",
                    "name": "Block recursive force delete",
                },
            ],
        },
        "tools": {},
        "backend": {"api_key": None},
        "audit": {
            "log_path": str(tmp_path / "audit.jsonl"),
            "redact_parameters": ["*password*", "*secret*"],
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config_data))
    return path


class TestEndToEndAllow:
    def test_simple_allow(self, config_path, tmp_path):
        config = load_hook_config(str(config_path))
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        hook_input = HookInput(
            session_id="e2e-allow",
            tool_name="Bash",
            tool_input={"command": "ls -la"},
        )

        with StateLock("e2e-allow"):
            result = evaluate(hook_input, config, audit)

        assert result is None  # Allow

        # Verify state was updated
        state = load_state("e2e-allow")
        assert state.step_count == 1
        assert state.accumulated_cost_usd > 0

        # Verify audit entry — JSONL may have multiple lines (session_start, pre_tool_use, etc.)
        audit.close()
        audit_lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        # Find the pre_tool_use entry with a decision
        entries = [json.loads(line) for line in audit_lines]
        allow_entries = [e for e in entries if e.get("decision") == "allow"]
        assert len(allow_entries) >= 1


class TestEndToEndBlock:
    def test_policy_block(self, config_path, tmp_path):
        config = load_hook_config(str(config_path))
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        hook_input = HookInput(
            session_id="e2e-block",
            tool_name="Bash",
            tool_input={"command": "rm -rf /important"},
        )

        with StateLock("e2e-block"):
            result = evaluate(hook_input, config, audit)

        assert result is not None
        assert result.decision == "block"
        assert "Block recursive force delete" in result.reason

        # Verify audit entry — JSONL may have multiple lines
        audit.close()
        audit_lines = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
        entries = [json.loads(line) for line in audit_lines]
        block_entries = [e for e in entries if e.get("decision") == "block"]
        assert len(block_entries) >= 1

    def test_step_limit_block(self, config_path, tmp_path):
        config = load_hook_config(str(config_path))
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")

        # Pre-set state to max steps
        state = SessionState(session_id="e2e-steps", step_count=10)
        save_state(state)

        hook_input = HookInput(
            session_id="e2e-steps",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )

        with StateLock("e2e-steps"):
            result = evaluate(hook_input, config, audit)

        assert result is not None
        assert result.decision == "block"
        assert "Step limit" in result.reason
        audit.close()


class TestEndToEndStatePersistence:
    def test_step_count_increments_across_calls(self, config_path, tmp_path):
        config = load_hook_config(str(config_path))

        for i in range(5):
            audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
            hook_input = HookInput(
                session_id="e2e-persist",
                tool_name="Bash",
                tool_input={"command": f"echo {i}"},
            )
            with StateLock("e2e-persist"):
                evaluate(hook_input, config, audit)
            audit.close()

        state = load_state("e2e-persist")
        assert state.step_count == 5

    def test_loop_detection_triggers(self, config_path, tmp_path):
        config = load_hook_config(str(config_path))
        results = []

        # Same command 3 times should trigger loop (threshold=3)
        for i in range(4):
            audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
            hook_input = HookInput(
                session_id="e2e-loop",
                tool_name="Bash",
                tool_input={"command": "echo hello"},
            )
            with StateLock("e2e-loop"):
                result = evaluate(hook_input, config, audit)
            results.append(result)
            audit.close()

        # First two should allow, third or fourth should block
        allow_count = sum(1 for r in results if r is None)
        block_count = sum(1 for r in results if r is not None)
        assert block_count > 0
        blocked = [r for r in results if r is not None][0]
        assert "Loop detected" in blocked.reason
