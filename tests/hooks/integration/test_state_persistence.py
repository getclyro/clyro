"""Integration tests for state persistence under load."""

import json
from unittest.mock import patch

import pytest

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import HookConfig
from clyro.hooks.evaluator import evaluate
from clyro.hooks.models import HookInput, SessionState
from clyro.hooks.state import StateLock, load_state, save_state


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    pending = tmp_path / "pending"
    pending.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.backend.EVENT_QUEUE_DIR", pending):
        yield sessions


class TestSequentialCalls:
    def test_100_sequential_calls(self, tmp_path):
        """State file integrity under sequential load.

        FRD success metric: 0 corruption events in 1000 calls.
        """
        config = HookConfig.model_validate({
            "global": {
                "max_steps": 200,
                "max_cost_usd": 100.0,
                "loop_detection": {"threshold": 200, "window": 200},
            },
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {},
        })

        session_id = "stress-test"

        for i in range(100):
            audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
            hook_input = HookInput(
                session_id=session_id,
                tool_name="Bash",
                tool_input={"command": f"echo iteration-{i}"},
            )
            with StateLock(session_id):
                result = evaluate(hook_input, config, audit)
            assert result is None  # All should allow
            audit.close()

        # Verify final state
        state = load_state(session_id)
        assert state.step_count == 100
        assert state.accumulated_cost_usd > 0
        assert len(state.loop_history) > 0

    def test_corrupt_state_fails_closed(self, tmp_path, isolated_sessions):
        """Corrupt state should fail-closed — raise CorruptStateError, not silently reset."""
        from clyro.hooks.state import CorruptStateError, state_path

        # Write garbage state file
        path = state_path("corrupt-test")
        path.write_text("{{NOT VALID JSON!!!}")

        with pytest.raises(CorruptStateError):
            load_state("corrupt-test")
