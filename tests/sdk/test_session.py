# Tests for Clyro SDK Session Management
# Implements PRD-001, PRD-005

"""Unit tests for session management."""

from decimal import Decimal
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    CostLimitExceededError,
    LoopDetectedError,
    StepLimitExceededError,
)
from clyro.session import (
    Session,
    get_current_session,
    set_current_session,
)
from clyro.trace import EventType, Framework


class TestSession:
    """Tests for Session class."""

    def test_session_creation(self):
        """Test creating a new session."""
        config = ClyroConfig()
        session = Session(config=config)

        assert session.session_id is not None
        assert session.config == config
        assert session.framework == Framework.GENERIC
        assert session.step_number == 0
        assert session.cumulative_cost == Decimal("0")
        assert session.is_active is False
        assert len(session.events) == 0

    def test_session_with_ids(self):
        """Test creating session with agent and org IDs."""
        config = ClyroConfig()
        agent_id = uuid4()
        org_id = uuid4()
        session_id = uuid4()

        session = Session(
            config=config,
            session_id=session_id,
            agent_id=agent_id,
            org_id=org_id,
            framework=Framework.LANGGRAPH,
            framework_version="0.2.0",
        )

        assert session.session_id == session_id
        assert session.agent_id == agent_id
        assert session.org_id == org_id
        assert session.framework == Framework.LANGGRAPH
        assert session.framework_version == "0.2.0"

    def test_session_start(self):
        """Test starting a session."""
        config = ClyroConfig()
        session = Session(config=config)

        event = session.start(input_data={"query": "test"})

        assert session.is_active is True
        assert event.event_type == EventType.SESSION_START
        assert event.input_data == {"query": "test"}
        assert len(session.events) == 1

    def test_session_end(self):
        """Test ending a session."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        event = session.end(output_data={"result": "done"})

        assert session.is_active is False
        assert event.event_type == EventType.SESSION_END
        assert event.output_data == {"result": "done"}
        assert len(session.events) == 2

    def test_session_end_with_error(self):
        """Test ending session with error."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        error = ValueError("Test error")
        event = session.end(error=error)

        assert session.is_active is False
        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"

    def test_session_duration(self):
        """Test session duration tracking."""
        config = ClyroConfig()
        session = Session(config=config)

        # Before start
        assert session.duration_ms == 0

        session.start()
        # During session
        assert session.duration_ms >= 0

        session.end()
        # After end
        assert session.duration_ms >= 0


class TestSessionStepRecording:
    """Tests for step recording."""

    def test_record_step(self):
        """Test recording a step."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        event = session.record_step(
            event_name="process",
            input_data={"data": "input"},
            output_data={"data": "output"},
            duration_ms=100,
        )

        assert session.step_number == 1
        assert event.event_type == EventType.STEP
        assert event.step_number == 1
        assert event.event_name == "process"
        assert event.duration_ms == 100
        assert len(session.events) == 2  # start + step

    def test_record_multiple_steps(self):
        """Test recording multiple steps."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        for i in range(5):
            session.record_step(event_name=f"step_{i}")

        assert session.step_number == 5
        assert len(session.events) == 6  # start + 5 steps

    def test_record_step_with_cost(self):
        """Test recording steps with cost."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        session.record_step(event_name="step1", cost_usd=Decimal("0.01"))
        session.record_step(event_name="step2", cost_usd=Decimal("0.02"))

        assert session.cumulative_cost == Decimal("0.03")


class TestSessionExecutionControls:
    """Tests for execution control enforcement."""

    def test_step_limit_enforcement(self):
        """Test step limit is enforced."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=3)
        )
        session = Session(config=config)
        session.start()

        # Should succeed
        session.record_step(event_name="step1")
        session.record_step(event_name="step2")
        session.record_step(event_name="step3")

        # Should fail
        with pytest.raises(StepLimitExceededError) as exc_info:
            session.record_step(event_name="step4")

        assert exc_info.value.limit == 3
        assert exc_info.value.current_step == 4

    def test_step_limit_disabled(self):
        """Test step limit can be disabled."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=2, enable_step_limit=False)
        )
        session = Session(config=config)
        session.start()

        # Should not fail even though over limit
        for i in range(10):
            session.record_step(event_name=f"step{i}")

        assert session.step_number == 10

    def test_cost_limit_enforcement(self):
        """Test cost limit is enforced."""
        config = ClyroConfig(
            controls=ExecutionControls(max_cost_usd=0.05)
        )
        session = Session(config=config)
        session.start()

        # Should succeed
        session.record_step(event_name="step1", cost_usd=Decimal("0.02"))
        session.record_step(event_name="step2", cost_usd=Decimal("0.02"))

        # Should fail
        with pytest.raises(CostLimitExceededError) as exc_info:
            session.record_step(event_name="step3", cost_usd=Decimal("0.02"))

        assert exc_info.value.limit_usd == 0.05
        assert exc_info.value.current_cost_usd > 0.05

    def test_cost_limit_disabled(self):
        """Test cost limit can be disabled."""
        config = ClyroConfig(
            controls=ExecutionControls(max_cost_usd=0.01, enable_cost_limit=False)
        )
        session = Session(config=config)
        session.start()

        # Should not fail even though over limit
        session.record_step(event_name="step1", cost_usd=Decimal("0.1"))
        assert session.cumulative_cost == Decimal("0.1")

    def test_loop_detection(self):
        """Test loop detection is enforced."""
        config = ClyroConfig(
            controls=ExecutionControls(loop_detection_threshold=3)
        )
        session = Session(config=config)
        session.start()

        same_state = {"counter": 42}

        # Should succeed first two times
        session.record_step(event_name="step1", state_snapshot=same_state)
        session.record_step(event_name="step2", state_snapshot=same_state)

        # Third time should fail
        with pytest.raises(LoopDetectedError) as exc_info:
            session.record_step(event_name="step3", state_snapshot=same_state)

        assert exc_info.value.iterations == 3

    def test_loop_detection_different_states(self):
        """Test loop detection allows different states."""
        config = ClyroConfig(
            controls=ExecutionControls(loop_detection_threshold=2)
        )
        session = Session(config=config)
        session.start()

        # Different states should not trigger loop detection
        for i in range(10):
            session.record_step(
                event_name=f"step{i}",
                state_snapshot={"counter": i},
            )

        assert session.step_number == 10

    def test_loop_detection_disabled(self):
        """Test loop detection can be disabled."""
        config = ClyroConfig(
            controls=ExecutionControls(
                loop_detection_threshold=2,
                enable_loop_detection=False,
            )
        )
        session = Session(config=config)
        session.start()

        same_state = {"stuck": True}

        # Should not fail even with repeated state
        for i in range(10):
            session.record_step(event_name=f"step{i}", state_snapshot=same_state)

        assert session.step_number == 10


class TestSessionErrorRecording:
    """Tests for error recording."""

    def test_record_error(self):
        """Test recording an error."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        error = ValueError("Test error")
        event = session.record_error(error, event_name="test_operation")

        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"
        assert event.error_stack is not None


class TestSessionEvents:
    """Tests for event management."""

    def test_record_event(self):
        """Test recording a pre-created event."""
        from clyro.trace import TraceEvent

        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        event = TraceEvent(
            session_id=uuid4(),  # Will be overwritten
            event_type=EventType.LLM_CALL,
            event_name="gpt-4",
            cost_usd=Decimal("0.01"),
        )

        session.record_event(event)

        assert event.session_id == session.session_id
        assert session.cumulative_cost == Decimal("0.01")

    def test_events_property_returns_copy(self):
        """Test that events property returns a copy."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        events = session.events
        events.clear()  # Should not affect session

        assert len(session.events) == 1


class TestSessionSummary:
    """Tests for session summary."""

    def test_get_summary(self):
        """Test getting session summary."""
        config = ClyroConfig()
        session = Session(
            config=config,
            framework=Framework.LANGGRAPH,
            framework_version="0.2.0",
        )
        session.start()
        session.record_step(event_name="step1", cost_usd=Decimal("0.01"))
        session.end()

        summary = session.get_summary()

        assert summary["session_id"] == str(session.session_id)
        assert summary["framework"] == "langgraph"
        assert summary["framework_version"] == "0.2.0"
        assert summary["is_active"] is False
        assert summary["step_count"] == 1
        assert summary["event_count"] == 3  # start + step + end
        assert summary["cumulative_cost_usd"] == 0.01
        assert summary["has_error"] is False


class TestCurrentSession:
    """Tests for current session context management."""

    def teardown_method(self):
        """Clear current session after each test."""
        set_current_session(None)

    def test_get_current_session_none(self):
        """Test getting current session when none set."""
        assert get_current_session() is None

    def test_set_current_session(self):
        """Test setting current session."""
        config = ClyroConfig()
        session = Session(config=config)

        set_current_session(session)
        assert get_current_session() == session

    def test_clear_current_session(self):
        """Test clearing current session."""
        config = ClyroConfig()
        session = Session(config=config)

        set_current_session(session)
        set_current_session(None)
        assert get_current_session() is None
