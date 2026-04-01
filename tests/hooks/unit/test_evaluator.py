"""Unit tests for hook evaluator."""

from unittest.mock import patch

import pytest

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import HookConfig
from clyro.hooks.evaluator import _enrich_tool_input, evaluate
from clyro.hooks.models import HookInput, SessionState


@pytest.fixture(autouse=True)
def mock_sessions_dir(tmp_path):
    """Redirect state operations to temp dir."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    pending = tmp_path / "pending"
    pending.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.backend.EVENT_QUEUE_DIR", pending), \
         patch("clyro.hooks.evaluator.load_state") as mock_load, \
         patch("clyro.hooks.evaluator.save_state") as mock_save:
        # Default: return fresh state
        mock_load.return_value = SessionState(session_id="test-session")
        yield {"sessions": sessions, "load_state": mock_load, "save_state": mock_save}


class TestEnrichToolInput:
    def test_adds_clyro_prefix_params(self):
        result = _enrich_tool_input(
            {"command": "ls"},
            tool_name="Bash",
            session_id="s1",
            step_count=5,
            accumulated_cost_usd=0.01,
        )
        assert result["_clyro_tool_name"] == "Bash"
        assert result["_clyro_session_id"] == "s1"
        assert result["_clyro_step_number"] == 5
        assert result["_clyro_cost"] == 0.01
        # Original params preserved
        assert result["command"] == "ls"

    def test_does_not_overwrite_existing_keys(self):
        result = _enrich_tool_input(
            {"command": "ls", "tool_name": "original"},
            tool_name="Bash", session_id="s1", step_count=1, accumulated_cost_usd=0,
        )
        # Original key preserved, enriched uses prefix
        assert result["tool_name"] == "original"
        assert result["_clyro_tool_name"] == "Bash"

    def test_includes_agent_id_when_provided(self):
        result = _enrich_tool_input(
            {"command": "ls"},
            tool_name="Bash", session_id="s1", step_count=1,
            accumulated_cost_usd=0, agent_id="my-agent-123",
        )
        assert result["_clyro_agent_id"] == "my-agent-123"

    def test_omits_agent_id_when_none(self):
        result = _enrich_tool_input(
            {"command": "ls"},
            tool_name="Bash", session_id="s1", step_count=1,
            accumulated_cost_usd=0,
        )
        assert "_clyro_agent_id" not in result


class TestEvaluateAllow:
    def test_allow_on_clean_input(self, mock_sessions_dir, tmp_path):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls -la"},
        )

        result = evaluate(hook_input, config, audit)
        assert result is None  # None = allow (empty stdout)
        audit.close()


class TestEvaluateBlockStepLimit:
    def test_block_on_step_limit(self, mock_sessions_dir, tmp_path):
        # Set state to already at the limit
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=50,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        result = evaluate(hook_input, config, audit)
        assert result is not None
        assert result.decision == "block"
        assert "Step limit" in result.reason
        audit.close()


class TestEvaluateBlockCost:
    def test_block_on_cost_budget(self, mock_sessions_dir, tmp_path):
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", accumulated_cost_usd=9.99,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"max_steps": 50, "max_cost_usd": 10.0, "cost_per_token_usd": 1.0},
            "audit": {},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "a very long command " * 100},
        )

        result = evaluate(hook_input, config, audit)
        assert result is not None
        assert result.decision == "block"
        assert "Cost budget" in result.reason
        audit.close()


class TestEvaluateBlockPolicy:
    def test_block_on_policy_violation(self, mock_sessions_dir, tmp_path):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {
                "max_steps": 50,
                "max_cost_usd": 10.0,
                "policies": [
                    {
                        "parameter": "command",
                        "operator": "contains",
                        "value": "rm -rf",
                        "name": "Block recursive force delete",
                    },
                ],
            },
            "audit": {},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "rm -rf /important-data"},
        )

        result = evaluate(hook_input, config, audit)
        assert result is not None
        assert result.decision == "block"
        assert "Policy violation" in result.reason
        audit.close()


class TestEvaluateBlockPerToolPolicy:
    def test_block_on_per_tool_policy(self, mock_sessions_dir, tmp_path):
        """Per-tool policies in merged list should be evaluated (❌-2 fix)."""
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {
                "max_steps": 50,
                "max_cost_usd": 10.0,
                "policies": [],
            },
            "tools": {
                "Bash": {
                    "policies": [
                        {
                            "parameter": "command",
                            "operator": "contains",
                            "value": "sudo",
                            "name": "Block sudo commands",
                        },
                    ],
                },
            },
            "audit": {},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "sudo rm -rf /"},
        )

        result = evaluate(hook_input, config, audit)
        assert result is not None
        assert result.decision == "block"
        assert "sudo" in result.reason.lower() or "Block sudo" in result.reason
        audit.close()

    def test_both_global_and_per_tool_policies(self, mock_sessions_dir, tmp_path):
        """Both global and per-tool policies should be evaluated."""
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {
                "max_steps": 50,
                "max_cost_usd": 10.0,
                "policies": [
                    {
                        "parameter": "command",
                        "operator": "contains",
                        "value": "rm -rf",
                        "name": "Block rm -rf globally",
                    },
                ],
            },
            "tools": {
                "Bash": {
                    "policies": [
                        {
                            "parameter": "command",
                            "operator": "contains",
                            "value": "sudo",
                            "name": "Block sudo for Bash",
                        },
                    ],
                },
            },
            "audit": {},
            "backend": {},
        })
        # Test global policy triggers
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "rm -rf /"},
        )
        result = evaluate(hook_input, config, audit)
        assert result is not None
        assert result.decision == "block"
        audit.close()


class TestEvaluateFailOpen:
    def test_exception_in_prevention_stack_does_not_crash(self, mock_sessions_dir, tmp_path):
        """Unexpected exceptions should result in allow (caller handles via exit code)."""
        mock_sessions_dir["load_state"].side_effect = RuntimeError("unexpected")
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {},
            "audit": {},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        # Should raise, and CLI layer catches it for fail-open
        with pytest.raises(RuntimeError):
            evaluate(hook_input, config, audit)
        audit.close()


class TestEvaluateTraceEvents:
    """FRD-HK-008: Verify trace event emission for allow/block decisions."""

    def _config_with_api_key(self, tmp_path, **overrides):
        base = {
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "test-key"},
        }
        base.update(overrides)
        return HookConfig.model_validate(base)

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_allow_emits_policy_check_trace(self, mock_enqueue, mock_sessions_dir, tmp_path):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = self._config_with_api_key(tmp_path)
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        result = evaluate(hook_input, config, audit)
        assert result is None  # Allow

        # Should have emitted session_start + policy_check (first step)
        events = [call[0][1] for call in mock_enqueue.call_args_list]
        event_types = [e["event_type"] for e in events]
        assert "session_start" in event_types
        assert "policy_check" in event_types

        policy_event = next(e for e in events if e["event_type"] == "policy_check")
        assert policy_event["metadata"]["decision"] == "allow"
        assert "event_id" in policy_event
        audit.close()

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_allow_no_session_start_on_step_2(self, mock_enqueue, mock_sessions_dir, tmp_path):
        """session_start only emitted on first step."""
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=1,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = self._config_with_api_key(tmp_path)
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        evaluate(hook_input, config, audit)

        events = [call[0][1] for call in mock_enqueue.call_args_list]
        event_types = [e["event_type"] for e in events]
        assert "session_start" not in event_types
        assert "policy_check" in event_types
        audit.close()

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_step_limit_emits_block_trace(self, mock_enqueue, mock_sessions_dir, tmp_path):
        """FRD-HK-008: Step limit block emits policy_check + error trace events."""
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=50,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = self._config_with_api_key(tmp_path)
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        result = evaluate(hook_input, config, audit)
        assert result.decision == "block"

        events = [call[0][1] for call in mock_enqueue.call_args_list]
        event_types = [e["event_type"] for e in events]
        assert "policy_check" in event_types
        assert "error" in event_types

        error_event = next(e for e in events if e["event_type"] == "error")
        assert error_event["error_type"] == "step_limit_exceeded"
        assert error_event["parent_event_id"] is not None

        # Verify parent wiring
        policy_event = next(e for e in events if e["event_type"] == "policy_check")
        assert error_event["parent_event_id"] == policy_event["event_id"]
        audit.close()

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_cost_budget_emits_block_trace(self, mock_enqueue, mock_sessions_dir, tmp_path):
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", accumulated_cost_usd=9.99,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = self._config_with_api_key(
            tmp_path,
            **{"global": {"max_steps": 50, "max_cost_usd": 10.0, "cost_per_token_usd": 1.0}},
        )
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "a very long command " * 100},
        )

        result = evaluate(hook_input, config, audit)
        assert result.decision == "block"

        events = [call[0][1] for call in mock_enqueue.call_args_list]
        error_event = next(e for e in events if e["event_type"] == "error")
        assert error_event["error_type"] == "budget_exceeded"
        audit.close()

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_policy_violation_emits_block_trace(self, mock_enqueue, mock_sessions_dir, tmp_path):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = self._config_with_api_key(tmp_path, **{
            "global": {
                "max_steps": 50, "max_cost_usd": 10.0,
                "policies": [{
                    "parameter": "command", "operator": "contains",
                    "value": "rm -rf", "name": "Block rm -rf",
                }],
            },
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "rm -rf /"},
        )

        result = evaluate(hook_input, config, audit)
        assert result.decision == "block"

        events = [call[0][1] for call in mock_enqueue.call_args_list]
        error_event = next(e for e in events if e["event_type"] == "error")
        assert error_event["error_type"] == "policy_violation"
        audit.close()

    @patch("clyro.hooks.evaluator.enqueue_event")
    def test_no_trace_without_api_key(self, mock_enqueue, mock_sessions_dir, tmp_path):
        """No trace events emitted when no API key configured."""
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session", tool_name="Bash",
            tool_input={"command": "ls"},
        )

        # Ensure resolved_api_key returns None even if CLYRO_API_KEY env var
        # is set by the test runner or leaked by another test.
        with patch.object(type(config), "resolved_api_key", new_callable=lambda: property(lambda self: None)):
            evaluate(hook_input, config, audit)
        mock_enqueue.assert_not_called()
        audit.close()
