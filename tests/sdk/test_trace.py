# Tests for Clyro SDK Trace Models
# Implements PRD-005

"""Unit tests for trace event models."""

import json
from datetime import UTC
from decimal import Decimal
from uuid import UUID, uuid4

from clyro.trace import (
    AgentStage,
    EventType,
    Framework,
    TraceEvent,
    compute_state_hash,
    create_error_event,
    create_llm_call_event,
    create_retriever_call_event,
    create_session_end_event,
    create_session_start_event,
    create_state_transition_event,
    create_step_event,
    create_tool_call_event,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_exist(self):
        """Test all required event types exist."""
        assert EventType.SESSION_START.value == "session_start"
        assert EventType.SESSION_END.value == "session_end"
        assert EventType.LLM_CALL.value == "llm_call"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TASK_START.value == "task_start"
        assert EventType.TASK_END.value == "task_end"
        assert EventType.AGENT_COMMUNICATION.value == "agent_communication"
        assert EventType.TASK_DELEGATION.value == "task_delegation"
        assert EventType.STATE_TRANSITION.value == "state_transition"
        assert EventType.POLICY_CHECK.value == "policy_check"
        assert EventType.ERROR.value == "error"
        assert EventType.STEP.value == "step"


class TestFramework:
    """Tests for Framework enum."""

    def test_frameworks_exist(self):
        """Test all required frameworks exist."""
        assert Framework.LANGGRAPH.value == "langgraph"
        assert Framework.CREWAI.value == "crewai"
        assert Framework.GENERIC.value == "generic"


class TestAgentStage:
    """Tests for AgentStage enum (TDD v1.4)."""

    def test_agent_stages_exist(self):
        """Test all required agent stages exist."""
        assert AgentStage.THINK.value == "think"
        assert AgentStage.ACT.value == "act"
        assert AgentStage.OBSERVE.value == "observe"

    def test_agent_stage_count(self):
        """Test there are exactly 3 agent stages."""
        assert len(AgentStage) == 3


class TestTraceEvent:
    """Tests for TraceEvent model."""

    def test_basic_event_creation(self):
        """Test creating a basic trace event."""
        session_id = uuid4()
        event = TraceEvent(
            session_id=session_id,
            event_type=EventType.STEP,
            event_name="test_step",
        )

        assert event.session_id == session_id
        assert event.event_type == EventType.STEP
        assert event.event_name == "test_step"
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.framework == Framework.GENERIC
        assert event.agent_stage == AgentStage.THINK  # Default value

    def test_event_with_agent_stage(self):
        """Test creating event with explicit agent_stage (TDD v1.4)."""
        session_id = uuid4()

        for stage in [AgentStage.THINK, AgentStage.ACT, AgentStage.OBSERVE]:
            event = TraceEvent(
                session_id=session_id,
                event_type=EventType.STEP,
                event_name="test_step",
                agent_stage=stage,
            )
            assert event.agent_stage == stage

    def test_agent_stage_serialization(self):
        """Test agent_stage is properly serialized to string."""
        event = TraceEvent(
            session_id=uuid4(),
            event_type=EventType.TOOL_CALL,
            agent_stage=AgentStage.ACT,
        )

        data = event.to_dict()
        assert data["agent_stage"] == "act"

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["agent_stage"] == "act"

    def test_agent_stage_from_dict(self):
        """Test creating event from dict with agent_stage."""
        data = {
            "session_id": str(uuid4()),
            "event_type": "llm_call",
            "agent_stage": "observe",
        }
        event = TraceEvent.from_dict(data)
        assert event.agent_stage == AgentStage.OBSERVE

    def test_event_with_all_fields(self):
        """Test creating event with all fields."""
        session_id = uuid4()
        agent_id = uuid4()
        parent_id = uuid4()

        event = TraceEvent(
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_id,
            event_type=EventType.LLM_CALL,
            event_name="gpt-4",
            framework=Framework.LANGGRAPH,
            framework_version="0.2.0",
            input_data={"messages": []},
            output_data={"content": "response"},
            token_count_input=100,
            token_count_output=50,
            cost_usd=Decimal("0.0045"),
            step_number=5,
            cumulative_cost=Decimal("0.025"),
            duration_ms=500,
        )

        assert event.agent_id == agent_id
        assert event.parent_event_id == parent_id
        assert event.framework == Framework.LANGGRAPH
        assert event.framework_version == "0.2.0"
        assert event.token_count_input == 100
        assert event.token_count_output == 50
        assert event.cost_usd == Decimal("0.0045")
        assert event.step_number == 5
        assert event.duration_ms == 500

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        session_id = uuid4()
        event = TraceEvent(
            session_id=session_id,
            event_type=EventType.STEP,
            event_name="test",
        )

        data = event.to_dict()
        assert isinstance(data, dict)
        assert data["session_id"] == str(session_id)
        assert data["event_type"] == "step"
        assert data["event_name"] == "test"

    def test_event_to_json(self):
        """Test serializing event to JSON."""
        session_id = uuid4()
        event = TraceEvent(
            session_id=session_id,
            event_type=EventType.LLM_CALL,
            cost_usd=Decimal("0.001"),
        )

        json_str = event.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["session_id"] == str(session_id)
        assert data["cost_usd"] == "0.001"

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        session_id = uuid4()
        data = {
            "session_id": str(session_id),
            "event_type": "llm_call",
            "event_name": "gpt-4",
            "token_count_input": 100,
        }

        event = TraceEvent.from_dict(data)
        assert event.session_id == session_id
        assert event.event_type == EventType.LLM_CALL
        assert event.event_name == "gpt-4"
        assert event.token_count_input == 100

    def test_event_error_fields(self):
        """Test event with error fields."""
        event = TraceEvent(
            session_id=uuid4(),
            event_type=EventType.ERROR,
            error_type="ValueError",
            error_message="Invalid input",
            error_stack="Traceback...",
        )

        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid input"
        assert event.error_stack == "Traceback..."

    def test_event_timestamp_timezone(self):
        """Test event timestamp is timezone-aware."""
        event = TraceEvent(
            session_id=uuid4(),
            event_type=EventType.STEP,
        )

        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == UTC


class TestComputeStateHash:
    """Tests for compute_state_hash function."""

    def test_hash_none_state(self):
        """Test hashing None state."""
        result = compute_state_hash(None)
        assert result is None

    def test_hash_empty_dict(self):
        """Test hashing empty dictionary."""
        result = compute_state_hash({})
        assert result is not None
        assert len(result) == 64  # SHA-256 hex length

    def test_hash_deterministic(self):
        """Test that same state produces same hash."""
        state = {"key": "value", "number": 42}
        hash1 = compute_state_hash(state)
        hash2 = compute_state_hash(state)
        assert hash1 == hash2

    def test_hash_different_states(self):
        """Test that different states produce different hashes."""
        state1 = {"key": "value1"}
        state2 = {"key": "value2"}
        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)
        assert hash1 != hash2

    def test_hash_key_order_independent(self):
        """Test that key order doesn't affect hash."""
        state1 = {"a": 1, "b": 2}
        state2 = {"b": 2, "a": 1}
        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)
        assert hash1 == hash2

    def test_hash_complex_state(self):
        """Test hashing complex nested state."""
        state = {
            "messages": [{"role": "user", "content": "hello"}],
            "context": {"nested": {"value": 123}},
            "list": [1, 2, 3],
        }
        result = compute_state_hash(state)
        assert result is not None
        assert len(result) == 64

    def test_hash_excludes_timestamps(self):
        """Test that timestamp fields are excluded from hash (filtered behavior)."""
        state1 = {"data": "value", "timestamp": "2024-01-01T00:00:00Z"}
        state2 = {"data": "value", "timestamp": "2024-01-02T00:00:00Z"}

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        # Hashes should be equal because timestamp is excluded
        assert hash1 == hash2

    def test_hash_excludes_request_id(self):
        """Test that request_id is excluded from hash (filtered behavior)."""
        state1 = {"data": "value", "request_id": "req-123"}
        state2 = {"data": "value", "request_id": "req-456"}

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        # Hashes should be equal because request_id is excluded
        assert hash1 == hash2

    def test_hash_filters_nested_excluded_fields(self):
        """Test that excluded fields in nested dicts are filtered."""
        state1 = {"data": {"value": 1, "timestamp": "2024-01-01"}}
        state2 = {"data": {"value": 1, "timestamp": "2024-02-01"}}

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        # Hashes should be equal because nested timestamp is excluded
        assert hash1 == hash2


class TestEventFactoryFunctions:
    """Tests for event factory functions."""

    def test_create_session_start_event(self):
        """Test creating session start event."""
        session_id = uuid4()
        agent_id = uuid4()

        event = create_session_start_event(
            session_id=session_id,
            agent_id=agent_id,
            framework=Framework.LANGGRAPH,
            input_data={"query": "test"},
        )

        assert event.session_id == session_id
        assert event.agent_id == agent_id
        assert event.event_type == EventType.SESSION_START
        assert event.event_name == "session_start"
        assert event.framework == Framework.LANGGRAPH
        assert event.input_data == {"query": "test"}
        assert event.step_number == 0
        assert event.agent_stage == AgentStage.THINK  # Default for session start

    def test_create_session_start_event_with_agent_stage(self):
        """Test creating session start event with explicit agent_stage."""
        event = create_session_start_event(
            session_id=uuid4(),
            agent_stage=AgentStage.OBSERVE,
        )
        assert event.agent_stage == AgentStage.OBSERVE

    def test_create_session_end_event(self):
        """Test creating session end event."""
        session_id = uuid4()

        event = create_session_end_event(
            session_id=session_id,
            step_number=10,
            cumulative_cost=Decimal("0.5"),
            output_data={"result": "done"},
            duration_ms=5000,
        )

        assert event.session_id == session_id
        assert event.event_type == EventType.SESSION_END
        assert event.step_number == 10
        assert event.cumulative_cost == Decimal("0.5")
        assert event.output_data == {"result": "done"}
        assert event.duration_ms == 5000
        assert event.agent_stage == AgentStage.OBSERVE  # Default for session end

    def test_create_session_end_event_with_error(self):
        """Test creating session end event with error."""
        session_id = uuid4()

        event = create_session_end_event(
            session_id=session_id,
            error_type="RuntimeError",
            error_message="Something failed",
        )

        assert event.event_type == EventType.ERROR  # Error type when error present
        assert event.event_name == "session_error"
        assert event.error_type == "RuntimeError"
        assert event.error_message == "Something failed"

    def test_create_step_event(self):
        """Test creating step event."""
        session_id = uuid4()

        event = create_step_event(
            session_id=session_id,
            step_number=5,
            event_name="process_input",
            input_data={"data": "input"},
            output_data={"data": "output"},
            state_snapshot={"state": "value"},
            duration_ms=100,
        )

        assert event.session_id == session_id
        assert event.event_type == EventType.STEP
        assert event.event_name == "process_input"
        assert event.step_number == 5
        assert event.input_data == {"data": "input"}
        assert event.output_data == {"data": "output"}
        assert event.state_snapshot == {"state": "value"}
        assert event.state_hash is not None  # Auto-computed
        assert event.duration_ms == 100
        assert event.agent_stage == AgentStage.THINK  # Default for step

    def test_create_step_event_with_agent_stage(self):
        """Test creating step event with explicit agent_stage."""
        event = create_step_event(
            session_id=uuid4(),
            step_number=1,
            event_name="observe_result",
            agent_stage=AgentStage.OBSERVE,
        )
        assert event.agent_stage == AgentStage.OBSERVE

    def test_create_llm_call_event(self):
        """Test creating LLM call event."""
        session_id = uuid4()

        event = create_llm_call_event(
            session_id=session_id,
            step_number=3,
            model="gpt-4",
            input_data={"messages": []},
            output_data={"content": "response"},
            token_count_input=100,
            token_count_output=50,
            cost_usd=Decimal("0.0045"),
            duration_ms=1500,
        )

        assert event.event_type == EventType.LLM_CALL
        assert event.event_name == "gpt-4"
        assert event.token_count_input == 100
        assert event.token_count_output == 50
        assert event.cost_usd == Decimal("0.0045")
        assert event.duration_ms == 1500
        assert event.agent_stage == AgentStage.THINK  # Default for LLM call (reasoning)

    def test_create_llm_call_event_with_agent_stage(self):
        """Test creating LLM call event with explicit agent_stage."""
        event = create_llm_call_event(
            session_id=uuid4(),
            step_number=1,
            model="gpt-4",
            input_data={},
            agent_stage=AgentStage.OBSERVE,
        )
        assert event.agent_stage == AgentStage.OBSERVE

    def test_create_tool_call_event(self):
        """Test creating tool call event."""
        session_id = uuid4()

        event = create_tool_call_event(
            session_id=session_id,
            step_number=4,
            tool_name="search_web",
            input_data={"query": "test"},
            output_data={"results": []},
            duration_ms=2000,
        )

        assert event.event_type == EventType.TOOL_CALL
        assert event.event_name == "search_web"
        assert event.input_data == {"query": "test"}
        assert event.output_data == {"results": []}
        assert event.duration_ms == 2000
        assert event.agent_stage == AgentStage.ACT  # Default for tool call (executing action)

    def test_create_tool_call_event_with_agent_stage(self):
        """Test creating tool call event with explicit agent_stage."""
        event = create_tool_call_event(
            session_id=uuid4(),
            step_number=1,
            tool_name="test_tool",
            input_data={},
            agent_stage=AgentStage.THINK,
        )
        assert event.agent_stage == AgentStage.THINK

    def test_create_error_event(self):
        """Test creating error event."""
        session_id = uuid4()

        event = create_error_event(
            session_id=session_id,
            step_number=7,
            error_type="ValueError",
            error_message="Invalid value",
            error_stack="Traceback...",
        )

        assert event.event_type == EventType.ERROR
        assert event.event_name == "ValueError"
        assert event.error_type == "ValueError"
        assert event.error_message == "Invalid value"
        assert event.error_stack == "Traceback..."
        assert event.step_number == 7
        assert event.agent_stage == AgentStage.OBSERVE  # Default for error (processing results)

    def test_create_error_event_with_agent_stage(self):
        """Test creating error event with explicit agent_stage."""
        event = create_error_event(
            session_id=uuid4(),
            step_number=1,
            error_type="TestError",
            error_message="test",
            agent_stage=AgentStage.ACT,
        )
        assert event.agent_stage == AgentStage.ACT


# =============================================================================
# Trace Hierarchy — FRD-001: event_id + parent_event_id on factory functions
# =============================================================================


class TestFactoryFunctionEventIdAndParentEventId:
    """Tests for FRD-001: event_id and parent_event_id parameters on factory functions."""

    def test_step_event_uses_provided_event_id(self):
        """FRD-001: create_step_event uses provided event_id instead of auto-generating."""
        custom_eid = uuid4()
        event = create_step_event(
            session_id=uuid4(), step_number=1, event_name="s", event_id=custom_eid,
        )
        assert event.event_id == custom_eid

    def test_step_event_auto_generates_event_id_when_none(self):
        """FRD-001: create_step_event auto-generates event_id when not provided."""
        event = create_step_event(session_id=uuid4(), step_number=1, event_name="s")
        assert isinstance(event.event_id, UUID)

    def test_step_event_passes_parent_event_id(self):
        """FRD-001: create_step_event passes parent_event_id through."""
        parent = uuid4()
        event = create_step_event(
            session_id=uuid4(), step_number=1, event_name="s", parent_event_id=parent,
        )
        assert event.parent_event_id == parent

    def test_llm_call_event_with_event_id_and_parent(self):
        """FRD-001: create_llm_call_event accepts event_id and parent_event_id."""
        eid = uuid4()
        parent = uuid4()
        event = create_llm_call_event(
            session_id=uuid4(), step_number=1, model="gpt-4", input_data={},
            event_id=eid, parent_event_id=parent,
        )
        assert event.event_id == eid
        assert event.parent_event_id == parent

    def test_tool_call_event_with_event_id_and_parent(self):
        """FRD-001: create_tool_call_event accepts event_id and parent_event_id."""
        eid = uuid4()
        parent = uuid4()
        event = create_tool_call_event(
            session_id=uuid4(), step_number=1, tool_name="search", input_data={},
            event_id=eid, parent_event_id=parent,
        )
        assert event.event_id == eid
        assert event.parent_event_id == parent

    def test_retriever_call_event_with_event_id_and_parent(self):
        """FRD-001: create_retriever_call_event accepts event_id and parent_event_id."""
        eid = uuid4()
        parent = uuid4()
        event = create_retriever_call_event(
            session_id=uuid4(), step_number=1, retriever_name="r", query="q",
            event_id=eid, parent_event_id=parent,
        )
        assert event.event_id == eid
        assert event.parent_event_id == parent

    def test_error_event_with_event_id_and_parent(self):
        """FRD-001: create_error_event accepts event_id and parent_event_id."""
        eid = uuid4()
        parent = uuid4()
        event = create_error_event(
            session_id=uuid4(), step_number=1, error_type="E", error_message="m",
            event_id=eid, parent_event_id=parent,
        )
        assert event.event_id == eid
        assert event.parent_event_id == parent

    def test_state_transition_event_with_event_id_and_parent(self):
        """FRD-001: create_state_transition_event accepts event_id and parent_event_id."""
        eid = uuid4()
        parent = uuid4()
        event = create_state_transition_event(
            session_id=uuid4(), step_number=1, node_name="n",
            event_id=eid, parent_event_id=parent,
        )
        assert event.event_id == eid
        assert event.parent_event_id == parent

    def test_parent_event_id_defaults_to_none(self):
        """FRD-001: parent_event_id defaults to None when not provided."""
        event = create_llm_call_event(
            session_id=uuid4(), step_number=1, model="gpt-4", input_data={},
        )
        assert event.parent_event_id is None
