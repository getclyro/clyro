"""
Unit tests for TraceEventFactory — TDD §11.1 v1.1 tests.

FRD-015: Convert MCP wrapper events to SDK-compatible TraceEvent dicts.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from clyro.backend.trace_event_factory import TraceEventFactory
from clyro.mcp.session import McpSession

TEST_SESSION_ID = UUID("00000000-0000-0000-0000-000000000099")
TEST_AGENT_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def session() -> McpSession:
    return McpSession(session_id=TEST_SESSION_ID, agent_id=TEST_AGENT_ID)


@pytest.fixture
def factory(session: McpSession) -> TraceEventFactory:
    return TraceEventFactory(session)


class TestTraceEventCommonFields:
    """All trace events include required common fields (FRD-015)."""

    def test_event_has_event_id(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert "event_id" in event
        UUID(event["event_id"])  # Valid UUID

    def test_event_has_session_id(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["session_id"] == str(TEST_SESSION_ID)

    def test_event_has_agent_id(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["agent_id"] == str(TEST_AGENT_ID)

    def test_event_has_framework_mcp(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["framework"] == "mcp"

    def test_event_has_cost_estimated_in_metadata(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["metadata"]["cost_estimated"] is True

    def test_event_has_parent_event_id_null(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["parent_event_id"] is None

    def test_event_has_timestamp(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert "timestamp" in event

    def test_event_has_metadata_source_mcp(self, factory: TraceEventFactory) -> None:
        event = factory.create_trace_event("test_type", None)
        assert event["metadata"]["_source"] == "mcp"

    def test_agent_id_none_when_not_set(self) -> None:
        session = McpSession(session_id=TEST_SESSION_ID)
        factory = TraceEventFactory(session)
        event = factory.create_trace_event("test_type", None)
        assert event["agent_id"] is None


class TestSessionStartEvent:
    """session_start trace event (FRD-015)."""

    def test_session_start(self, factory: TraceEventFactory) -> None:
        event = factory.session_start()
        assert event["event_type"] == "session_start"
        assert "agent_stage" not in event  # Omitted so API default applies


class TestSessionEndEvent:
    """session_end trace event (FRD-015)."""

    def test_session_end(self, factory: TraceEventFactory, session: McpSession) -> None:
        session.step_count = 10
        session.accumulated_cost_usd = 0.05
        event = factory.session_end(total_duration_ms=5000)
        assert event["event_type"] == "session_end"
        assert event["duration_ms"] == 5000
        assert event["metadata"]["total_steps"] == 10

    def test_session_end_includes_cost(self, factory: TraceEventFactory) -> None:
        event = factory.session_end()
        assert "total_cost_usd" in event["metadata"]


class TestPolicyCheckEvent:
    """policy_check trace event — think stage (FRD-015)."""

    def test_policy_check(self, factory: TraceEventFactory) -> None:
        event = factory.policy_check("test_tool", {"arg": "val"}, duration_ms=2)
        assert event["event_type"] == "policy_check"
        assert event["agent_stage"] == "think"
        assert event["event_name"] == "policy_check"
        assert event["input_data"]["arguments"]["arg"] == "val"

    def test_policy_check_no_params(self, factory: TraceEventFactory) -> None:
        event = factory.policy_check("test_tool", None)
        assert event["input_data"] is None


class TestToolCallActEvent:
    """tool_call act stage — forwarded to server (FRD-015)."""

    def test_tool_call_act(self, factory: TraceEventFactory) -> None:
        event = factory.tool_call_act("query_db", {"sql": "SELECT 1"}, step_number=3)
        assert event["event_type"] == "tool_call"
        assert event["agent_stage"] == "act"
        assert event["token_count_input"] > 0


class TestToolCallObserveEvent:
    """tool_call observe stage — response received (FRD-015)."""

    def test_tool_call_observe(self, factory: TraceEventFactory) -> None:
        event = factory.tool_call_observe("query_db", "result data", cost_usd=0.001)
        assert event["event_type"] == "tool_call"
        assert event["agent_stage"] == "observe"
        assert event["token_count_output"] > 0
        assert event["cost_usd"] == 0.001

    def test_tool_call_observe_none_response(self, factory: TraceEventFactory) -> None:
        event = factory.tool_call_observe("tool", None, cost_usd=0.0)
        assert event["output_data"] is None
        assert event["token_count_output"] == 0


class TestBlockedCallEvent:
    """error trace event for blocked calls (FRD-015)."""

    def test_blocked_call(self, factory: TraceEventFactory) -> None:
        event = factory.blocked_call(
            "dangerous_tool",
            "policy_violation",
            "Blocked by policy_violation",
            {"rule_name": "no-drop"},
        )
        assert event["event_type"] == "error"
        assert event["agent_stage"] == "act"
        assert event["error_type"] == "policy_violation"
        assert event["error_message"] == "Blocked by policy_violation"


class TestOutputTruncation:
    """Output truncated to 10KB (TDD §2.13)."""

    def test_truncates_large_output(self, factory: TraceEventFactory) -> None:
        large_output = {"data": "x" * 20000}
        event = factory.create_trace_event(
            "tool_call", "observe", output_data=large_output
        )
        assert event["metadata"]["output_truncated"] is True
        # output_data should be truncated
        output_json = json.dumps(event["output_data"])
        assert len(output_json) <= 11000  # Some overhead

    def test_no_truncation_for_small_output(self, factory: TraceEventFactory) -> None:
        small_output = {"data": "small"}
        event = factory.create_trace_event(
            "tool_call", "observe", output_data=small_output
        )
        assert event["metadata"].get("output_truncated") is not True
