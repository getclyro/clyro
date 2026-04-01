"""Unit tests for trace handler."""

from unittest.mock import patch

import pytest

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import HookConfig
from clyro.hooks.models import HookInput, SessionState
from clyro.hooks.tracer import _tool_result_summary, handle_session_end, handle_tool_complete


class TestToolResultSummary:
    def test_empty_result(self):
        assert _tool_result_summary(None) == "no result"

    def test_bash_result(self):
        result = {"stdout": "hello world", "stderr": "", "exitCode": 0}
        summary = _tool_result_summary(result)
        assert "stdout: 11 chars" in summary
        assert "exitCode: 0" in summary

    def test_generic_result(self):
        result = {"output": "some output data"}
        summary = _tool_result_summary(result)
        assert "output: 16 chars" in summary


@pytest.fixture(autouse=True)
def mock_state(tmp_path):
    """Mock state operations."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.tracer.load_state") as mock_load, \
         patch("clyro.hooks.tracer.save_state") as mock_save, \
         patch("clyro.hooks.tracer.cleanup_stale_sessions"):
        mock_load.return_value = SessionState(
            session_id="test-session",
            step_count=5,
            accumulated_cost_usd=0.005,
            pre_call_cost_estimate=0.002,
        )
        yield {"load_state": mock_load, "save_state": mock_save}


class TestHandleToolComplete:
    def test_adjusts_cost(self, mock_state, tmp_path):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"cost_per_token_usd": 0.00001},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result={"stdout": "file.txt"},
        )

        handle_tool_complete(hook_input, config, audit)

        # Verify save_state was called with adjusted cost
        saved_state = mock_state["save_state"].call_args[0][0]
        assert saved_state.pre_call_cost_estimate == 0.0
        # Cost should be adjusted from pre-call estimate
        assert saved_state.accumulated_cost_usd != 0.005
        audit.close()

    def test_writes_audit_entry(self, mock_state, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        config = HookConfig.model_validate({
            "global": {},
            "audit": {"log_path": str(audit_path)},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )

        handle_tool_complete(hook_input, config, audit)
        audit.close()

        import json
        content = audit_path.read_text().strip()
        assert content  # Non-empty
        entry = json.loads(content)
        assert entry["event"] == "tool_call_observe"


class TestHandleToolCompleteTraceFields:
    def test_audit_includes_duration_ms(self, mock_state, tmp_path):
        """FRD-HK-008: duration_ms should be included in audit entry."""
        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        config = HookConfig.model_validate({
            "global": {"cost_per_token_usd": 0.00001},
            "audit": {"log_path": str(audit_path)},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result={"stdout": "file.txt"},
        )

        handle_tool_complete(hook_input, config, audit)
        audit.close()

        import json
        entry = json.loads(audit_path.read_text().strip())
        assert "duration_ms" in entry
        assert isinstance(entry["duration_ms"], int)


class TestHandleToolCompleteAgentId:
    def test_audit_includes_agent_id(self, mock_state, tmp_path):
        """FRD-HK-008: agent_id should be included in audit and trace events."""
        mock_state["load_state"].return_value = SessionState(
            session_id="test-session",
            agent_id="test-agent-id",
            step_count=5,
            accumulated_cost_usd=0.005,
            pre_call_cost_estimate=0.002,
        )
        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        config = HookConfig.model_validate({
            "global": {"cost_per_token_usd": 0.00001},
            "audit": {"log_path": str(audit_path)},
            "backend": {},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )

        handle_tool_complete(hook_input, config, audit)
        audit.close()

        import json
        entry = json.loads(audit_path.read_text().strip())
        assert entry.get("agent_id") == "test-agent-id"


class TestHandleToolCompleteEventQueue:
    def test_queues_event_when_api_key_set(self, mock_state, tmp_path):
        """FRD-HK-008: Trace events should be queued to event queue."""
        mock_state["load_state"].return_value = SessionState(
            session_id="test-session",
            agent_id="test-agent-id",
            step_count=5,
            accumulated_cost_usd=0.005,
            pre_call_cost_estimate=0.002,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"cost_per_token_usd": 0.00001},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "test-key"},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result={"stdout": "file.txt"},
        )

        with patch("clyro.hooks.tracer.enqueue_event") as mock_enqueue:
            handle_tool_complete(hook_input, config, audit)
            mock_enqueue.assert_called_once()
            event = mock_enqueue.call_args[0][1]
            assert event["event_type"] == "tool_call_observe"
            assert event["agent_id"] == "test-agent-id"
            assert event["metadata"]["tool_name"] == "Bash"
        audit.close()

    def test_event_uses_create_trace_event_format(self, mock_state, tmp_path):
        """FRD-HK-008: tool_call_observe uses create_trace_event with event_id, tokens."""
        mock_state["load_state"].return_value = SessionState(
            session_id="test-session",
            agent_id="test-agent-id",
            step_count=5,
            accumulated_cost_usd=0.005,
            pre_call_cost_estimate=0.002,
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {"cost_per_token_usd": 0.00001},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "test-key"},
        })
        hook_input = HookInput(
            session_id="test-session",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result={"stdout": "file.txt"},
        )

        with patch("clyro.hooks.tracer.enqueue_event") as mock_enqueue:
            handle_tool_complete(hook_input, config, audit)
            event = mock_enqueue.call_args[0][1]
            # Should have event_id (UUID format)
            from uuid import UUID
            UUID(event["event_id"])
            # Should have token counts
            assert "token_count_input" in event
            assert "token_count_output" in event
            assert event["token_count_input"] > 0
            # Should have framework and metadata from create_trace_event
            assert event["framework"] == "claude_code_hooks"
            assert event["metadata"]["_source"] == "claude_code_hooks"
            # Should have input_data and output_data
            assert event["input_data"]["name"] == "Bash"
            assert "summary" in event["output_data"]
        audit.close()


class TestHandleSessionEnd:
    def test_writes_session_summary(self, mock_state, tmp_path):
        audit_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path=audit_path)
        config = HookConfig.model_validate({
            "global": {},
            "audit": {"log_path": str(audit_path)},
            "backend": {},
        })
        hook_input = HookInput(session_id="test-session")

        handle_session_end(hook_input, config, audit)
        audit.close()

        import json
        entry = json.loads(audit_path.read_text().strip())
        assert entry["event"] == "session_end"
        assert entry["total_steps"] == 5
        assert "duration_seconds" in entry

    def test_session_end_flushes_event_queue(self, mock_state, tmp_path):
        """FRD-HK-009: Session-end should flush all queued events."""
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "test-key"},
        })
        hook_input = HookInput(session_id="test-session")

        with patch("clyro.hooks.tracer.enqueue_event") as mock_enqueue, \
             patch("clyro.hooks.tracer.flush_event_queue") as mock_flush:
            handle_session_end(hook_input, config, audit)
            mock_enqueue.assert_called_once()  # session_end event
            mock_flush.assert_called_once()
        audit.close()

    def test_session_end_uses_create_trace_event(self, mock_state, tmp_path):
        """FRD-HK-009: session_end uses create_trace_event with event_id, metadata."""
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        config = HookConfig.model_validate({
            "global": {},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "test-key"},
        })
        hook_input = HookInput(session_id="test-session")

        with patch("clyro.hooks.tracer.enqueue_event") as mock_enqueue, \
             patch("clyro.hooks.tracer.flush_event_queue"):
            handle_session_end(hook_input, config, audit)
            event = mock_enqueue.call_args[0][1]
            from uuid import UUID
            UUID(event["event_id"])
            assert event["event_type"] == "session_end"
            assert event["framework"] == "claude_code_hooks"
            assert event["metadata"]["cumulative_steps"] == 5
            assert "duration_seconds" in event["metadata"]
            assert event["step_number"] == 5
        audit.close()
