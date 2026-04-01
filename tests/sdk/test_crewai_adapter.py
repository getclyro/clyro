# Tests for Clyro SDK CrewAI Adapter
# Implements PRD-004

"""
Unit tests for the CrewAI adapter.

These tests use mock CrewAI objects to avoid requiring CrewAI
as a hard dependency while thoroughly testing the adapter functionality.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    CostLimitExceededError,
    FrameworkVersionError,
    LoopDetectedError,
    StepLimitExceededError,
)
from clyro.session import Session
from clyro.trace import AgentStage, EventType, Framework

# =============================================================================
# Mock CrewAI Module and Classes
# =============================================================================


class MockCrewAIModule(ModuleType):
    """Mock crewai module for testing without the actual dependency."""

    def __init__(self, version: str = "0.30.5"):
        super().__init__("crewai")
        self.__version__ = version


class MockAgent:
    """Mock CrewAI Agent class."""

    __module__ = "crewai.agent"

    def __init__(
        self,
        role: str = "Researcher",
        goal: str = "Research topics",
        backstory: str = "Expert researcher",
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools: list[Any] = []


class MockTask:
    """Mock CrewAI Task class."""

    __module__ = "crewai.task"

    def __init__(
        self,
        description: str = "Research the topic",
        expected_output: str = "A detailed report",
        agent: MockAgent | None = None,
    ):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.output: Any = None
        self.callback: Any = None


class MockCrewOutput:
    """Mock CrewAI CrewOutput class."""

    def __init__(self, raw: str = "Crew execution completed"):
        self.raw = raw
        self.tasks_output: list[Any] = []

    def __str__(self) -> str:
        return self.raw


class MockCrew:
    """Mock CrewAI Crew class."""

    __module__ = "crewai.crew"

    def __init__(
        self,
        agents: list[MockAgent] | None = None,
        tasks: list[MockTask] | None = None,
        name: str = "test_crew",
    ):
        self.agents = agents or []
        self.tasks = tasks or []
        self.name = name
        self._task_callbacks: dict[str, Any] = {}

    def kickoff(
        self,
        inputs: dict[str, Any] | None = None,
    ) -> MockCrewOutput:
        """Execute the crew."""
        # Simulate task execution
        for task in self.tasks:
            task.output = f"Output for: {task.description}"

        return MockCrewOutput(raw="Crew completed successfully")

    async def akickoff(
        self,
        inputs: dict[str, Any] | None = None,
    ) -> MockCrewOutput:
        """Execute the crew asynchronously."""
        return self.kickoff(inputs)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_crewai_module():
    """Create and install a mock crewai module."""
    mock_module = MockCrewAIModule(version="0.30.5")
    with patch.dict(sys.modules, {"crewai": mock_module}):
        yield mock_module


@pytest.fixture
def mock_crewai_unsupported():
    """Create a mock crewai module with unsupported version."""
    mock_module = MockCrewAIModule(version="0.25.0")
    with patch.dict(sys.modules, {"crewai": mock_module}):
        yield mock_module


@pytest.fixture
def mock_agent():
    """Create a mock Agent."""
    return MockAgent(
        role="Researcher",
        goal="Research AI topics",
        backstory="Expert AI researcher",
    )


@pytest.fixture
def mock_task(mock_agent):
    """Create a mock Task."""
    return MockTask(
        description="Research the latest AI trends",
        expected_output="A comprehensive report",
        agent=mock_agent,
    )


@pytest.fixture
def mock_crew(mock_agent, mock_task):
    """Create a mock Crew."""
    writer = MockAgent(role="Writer", goal="Write articles", backstory="Expert writer")
    writing_task = MockTask(
        description="Write an article",
        expected_output="A well-written article",
        agent=writer,
    )
    return MockCrew(
        agents=[mock_agent, writer],
        tasks=[mock_task, writing_task],
        name="research_crew",
    )


@pytest.fixture
def config():
    """Create a test configuration."""
    return ClyroConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_state=True,
    )


@pytest.fixture
def session(config):
    """Create a test session."""
    session = Session(
        config=config,
        agent_id=uuid4(),
        org_id=uuid4(),
        framework=Framework.CREWAI,
    )
    session.start()
    return session


# =============================================================================
# Version Detection and Validation Tests
# =============================================================================


class TestVersionDetection:
    """Tests for CrewAI version detection and validation."""

    def test_detect_version_returns_installed_version(self, mock_crewai_module):
        """Test that version detection returns the installed version."""
        from clyro.adapters.crewai import detect_crewai_version

        version = detect_crewai_version()
        assert version == "0.30.5"

    def test_detect_version_returns_none_when_not_installed(self):
        """Test that version detection returns None when CrewAI is not installed."""
        from clyro.adapters.crewai import detect_crewai_version

        with patch.dict(sys.modules, {"crewai": None}):
            sys.modules.pop("crewai", None)
            version = detect_crewai_version()
            assert version is None

    def test_validate_version_succeeds_for_supported(self, mock_crewai_module):
        """Test that validation succeeds for supported versions."""
        from clyro.adapters.crewai import validate_crewai_version

        version = validate_crewai_version()
        assert version == "0.30.5"

    def test_validate_version_raises_for_unsupported(self, mock_crewai_unsupported):
        """Test that validation raises FrameworkVersionError for unsupported versions."""
        from clyro.adapters.crewai import validate_crewai_version

        with pytest.raises(FrameworkVersionError) as exc_info:
            validate_crewai_version()

        assert exc_info.value.framework == "crewai"
        assert exc_info.value.version == "0.25.0"
        assert ">=0.30.0" in exc_info.value.supported

    def test_validate_version_raises_when_not_installed(self):
        """Test that validation raises when CrewAI is not installed."""
        from clyro.adapters.crewai import validate_crewai_version

        with patch.dict(sys.modules, {}):
            sys.modules.pop("crewai", None)
            with pytest.raises(FrameworkVersionError) as exc_info:
                validate_crewai_version()

            assert exc_info.value.version == "not installed"

    def test_parse_version_handles_various_formats(self):
        """Test version parsing handles various format strings."""
        from clyro.adapters.crewai import _parse_version

        assert _parse_version("0.30.0") == (0, 30, 0)
        assert _parse_version("0.30.5") == (0, 30, 5)
        assert _parse_version("1.0.0") == (1, 0, 0)
        assert _parse_version("0.30.0rc1") == (0, 30, 0)
        assert _parse_version("0.30.0dev1") == (0, 30, 0)
        assert _parse_version("0.30.0a1") == (0, 30, 0)


class TestIsCrewAIAgent:
    """Tests for CrewAI agent detection."""

    def test_detects_crew(self, mock_crew):
        """Test detection of Crew instances."""
        from clyro.adapters.crewai import is_crewai_agent

        assert is_crewai_agent(mock_crew) is True

    def test_detects_crew_by_module(self):
        """Test detection by module name."""
        from clyro.adapters.crewai import is_crewai_agent

        class CrewLikeClass:
            __module__ = "crewai.crew"

        assert is_crewai_agent(CrewLikeClass()) is True

    def test_detects_crew_by_attributes(self):
        """Test detection by CrewAI-specific attributes."""
        from clyro.adapters.crewai import is_crewai_agent

        class CrewWithAttributes:
            agents = []
            tasks = []

            def kickoff(self):
                pass

        assert is_crewai_agent(CrewWithAttributes()) is True

    def test_does_not_detect_regular_function(self):
        """Test that regular functions are not detected as CrewAI."""
        from clyro.adapters.crewai import is_crewai_agent

        def regular_func():
            return "hello"

        assert is_crewai_agent(regular_func) is False

    def test_does_not_detect_callable_class(self):
        """Test that callable classes are not detected as CrewAI."""
        from clyro.adapters.crewai import is_crewai_agent

        class MyCallable:
            def __call__(self):
                return "hello"

        assert is_crewai_agent(MyCallable()) is False


# =============================================================================
# CrewAIAdapter Tests
# =============================================================================


class TestCrewAIAdapterInit:
    """Tests for CrewAIAdapter initialization."""

    def test_init_with_crew(self, mock_crewai_module, mock_crew, config):
        """Test adapter initialization with a Crew."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)

        assert adapter.agent == mock_crew
        assert adapter.name == "research_crew"
        assert adapter.framework == Framework.CREWAI
        assert adapter.framework_version == "0.30.5"

    def test_init_raises_for_unsupported_version(
        self, mock_crewai_unsupported, mock_crew, config
    ):
        """Test that initialization raises for unsupported versions."""
        from clyro.adapters.crewai import CrewAIAdapter

        with pytest.raises(FrameworkVersionError):
            CrewAIAdapter(mock_crew, config)

    def test_init_skips_validation_when_disabled(
        self, mock_crewai_unsupported, mock_crew, config
    ):
        """Test that version validation can be skipped."""
        from clyro.adapters.crewai import CrewAIAdapter

        # Should not raise when validation is disabled
        adapter = CrewAIAdapter(mock_crew, config, validate_version=False)
        assert adapter.framework_version == "0.25.0"

    def test_init_extracts_name_from_crew(self, mock_crewai_module, config):
        """Test that adapter extracts name from crew."""
        from clyro.adapters.crewai import CrewAIAdapter

        crew = MockCrew(name="my_custom_crew")
        adapter = CrewAIAdapter(crew, config)
        assert adapter.name == "my_custom_crew"


class TestCrewAIAdapterHooks:
    """Tests for CrewAIAdapter lifecycle hooks."""

    def test_before_call_creates_handler(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that before_call creates a callback handler."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        kwargs: dict[str, Any] = {"inputs": {"topic": "AI"}}

        context = adapter.before_call(session, (), kwargs)

        assert "start_time" in context
        assert "step_number" in context
        assert "handler" in context
        assert "crew_context" in context

    def test_before_call_with_inputs(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that before_call handles inputs correctly."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        kwargs = {"inputs": {"topic": "AI trends", "year": 2024}}

        context = adapter.before_call(session, (), kwargs)

        handler = context["handler"]
        assert handler._crew_state["inputs"]["topic"] == "AI trends"

    def test_after_call_creates_step_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that after_call creates a step event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = MockCrewOutput(raw="Test output")
        event = adapter.after_call(session, result, context)

        assert event.event_type == EventType.STEP
        assert event.session_id == session.session_id
        assert "research_crew_complete" in event.event_name
        assert event.agent_stage == AgentStage.OBSERVE

    def test_after_call_captures_output_when_enabled(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that after_call captures output when enabled."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = MockCrewOutput(raw="Test result")
        event = adapter.after_call(session, result, context)

        assert event.output_data is not None

    def test_after_call_no_output_when_disabled(
        self, mock_crewai_module, mock_crew, session
    ):
        """Test that after_call doesn't capture output when disabled."""
        from clyro.adapters.crewai import CrewAIAdapter

        no_capture_config = ClyroConfig(capture_outputs=False)
        adapter = CrewAIAdapter(mock_crew, no_capture_config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = MockCrewOutput(raw="Secret data")
        event = adapter.after_call(session, result, context)

        assert event.output_data is None

    def test_on_error_creates_error_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_error creates an error event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        error = ValueError("Test error")
        event = adapter.on_error(session, error, context)

        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"
        assert event.error_stack is not None
        assert event.framework == Framework.CREWAI


# =============================================================================
# CrewAICallbackHandler Tests
# =============================================================================


class TestCrewAICallbackHandler:
    """Tests for CrewAICallbackHandler."""

    def test_on_crew_start_returns_context(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_crew_start returns proper context."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_crew_start(
            crew_name="test_crew",
            inputs={"topic": "AI"},
        )

        assert "start_time" in context
        assert "crew_name" in context
        assert context["crew_name"] == "test_crew"

    def test_on_task_start_creates_task_start_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_task_start creates a task start event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_task_start(
            task_description="Research AI trends",
            agent_role="Researcher",
            task_id="task_001",
        )

        assert "start_time" in context
        assert "task_key" in context
        assert context["agent_role"] == "Researcher"

        # Check that event was created
        events = handler.drain_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.TASK_START
        assert "task_start" in events[0].event_name

    def test_on_task_end_creates_task_end_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_task_end creates a task end event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_task_start(
            task_description="Research AI",
            agent_role="Researcher",
        )

        event = handler.on_task_end(
            task_output="Research completed",
            context=context,
        )

        assert event.event_type == EventType.TASK_END
        assert "task_end" in event.event_name
        assert event.framework == Framework.CREWAI

    def test_on_task_end_captures_output(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_task_end captures task output."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_task_start(
            task_description="Research AI",
            agent_role="Researcher",
        )

        # Clear the start event
        handler.drain_events()

        event = handler.on_task_end(
            task_output={"summary": "AI is advancing rapidly"},
            context=context,
        )

        assert event.output_data is not None
        assert "task_output" in event.output_data

    def test_on_task_error_creates_error_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_task_error creates an error event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_task_start(
            task_description="Failing task",
            agent_role="Researcher",
        )

        error = RuntimeError("Task execution failed")
        event = handler.on_task_error(error=error, context=context)

        assert event.event_type == EventType.ERROR
        assert event.error_type == "RuntimeError"
        assert "task_error" in event.metadata.get("event_subtype", "")

    def test_on_agent_action_creates_tool_call_event(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_agent_action creates appropriate events for tool calls."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        event = handler.on_agent_action(
            agent_role="Researcher",
            action_type="tool_call",
            action_input={"query": "AI trends"},
            action_output="Search results",
            tool_name="search_tool",
        )

        assert event.event_type == EventType.TOOL_CALL
        assert event.event_name == "search_tool"
        assert event.agent_stage == AgentStage.ACT

    def test_on_agent_action_creates_step_event_for_non_tools(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that on_agent_action creates step events for non-tool actions."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        event = handler.on_agent_action(
            agent_role="Researcher",
            action_type="think",
            action_input={"context": "Planning next steps"},
        )

        assert event.event_type == EventType.STEP
        assert event.agent_stage == AgentStage.THINK

    def test_drain_events_returns_and_clears(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that drain_events returns events and clears internal list."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Create multiple events
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="think",
        )
        handler.on_agent_action(
            agent_role="Writer",
            action_type="act",
        )

        events = handler.drain_events()
        assert len(events) == 2

        # Second drain should be empty
        events_again = handler.drain_events()
        assert len(events_again) == 0

    def test_determine_agent_stage_for_tool_actions(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that tool actions are classified as ACT stage."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("tool_call", "search")
        assert stage == AgentStage.ACT

        stage = handler._determine_agent_stage("execute", None)
        assert stage == AgentStage.ACT

    def test_determine_agent_stage_for_observe_actions(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that observation actions are classified as OBSERVE stage."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("observe", None)
        assert stage == AgentStage.OBSERVE

        stage = handler._determine_agent_stage("result", None)
        assert stage == AgentStage.OBSERVE

    def test_determine_agent_stage_defaults_to_think(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that default stage is THINK."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("plan", None)
        assert stage == AgentStage.THINK


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for data serialization in the handler."""

    def test_serialize_primitives(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test serialization of primitive types."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        assert handler._serialize_value(None) is None
        assert handler._serialize_value("string") == "string"
        assert handler._serialize_value(42) == 42
        assert handler._serialize_value(3.14) == 3.14
        assert handler._serialize_value(True) is True

    def test_serialize_collections(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test serialization of collections."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        assert handler._serialize_value([1, 2, 3]) == [1, 2, 3]
        assert handler._serialize_value((1, 2)) == [1, 2]
        assert handler._serialize_value({"a": 1}) == {"a": 1}

    def test_serialize_nested_structures(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test serialization of nested structures."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        nested = {"level1": {"level2": {"level3": "value"}}}
        result = handler._serialize_value(nested)

        assert result["level1"]["level2"]["level3"] == "value"

    def test_serialize_handles_max_depth(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that serialization handles max depth."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Create deeply nested structure
        deep: dict[str, Any] = {"a": None}
        current = deep
        for i in range(60):  # Exceeds MAX_SERIALIZE_DEPTH of 50
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should not raise, should truncate at max depth
        result = handler._serialize_data(deep)
        assert result is not None

    def test_serialize_excludes_private_attributes(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that private attributes are excluded from serialization."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        data = {"public": "value", "_private": "secret"}
        result = handler._serialize_value(data)

        assert "public" in result
        assert "_private" not in result

    def test_serialize_crew_output(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test serialization of CrewOutput objects."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        output = MockCrewOutput(raw="Test output")

        result = adapter._serialize_result(output)
        assert "result" in result
        assert result["result"] == "Test output"
        assert result["type"] == "MockCrewOutput"


# =============================================================================
# Integration with detect_adapter Tests
# =============================================================================


class TestDetectAdapterIntegration:
    """Tests for detect_adapter with CrewAI agents."""

    def test_detects_crew(self, mock_crew):
        """Test that detect_adapter identifies Crew."""
        from clyro.adapters.generic import detect_adapter

        result = detect_adapter(mock_crew)
        assert result == "crewai"

    def test_returns_generic_for_regular_function(self):
        """Test that detect_adapter returns generic for regular functions."""
        from clyro.adapters.generic import detect_adapter

        def my_func():
            return "hello"

        result = detect_adapter(my_func)
        assert result == "generic"


# =============================================================================
# Task Event Tests
# =============================================================================


class TestTaskEvents:
    """Tests for task-related event creation."""

    def test_task_lifecycle_events(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test complete task lifecycle event capture."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Start task
        context = handler.on_task_start(
            task_description="Research AI",
            agent_role="Researcher",
            task_id="task_1",
        )

        # End task
        handler.on_task_end(
            task_output="Research completed",
            context=context,
        )

        events = handler.drain_events()
        assert len(events) == 2

        # First event should be task_start
        assert "task_start" in events[0].event_name
        assert events[0].agent_stage == AgentStage.THINK

        # Second event should be task_end
        assert "task_end" in events[1].event_name
        assert events[1].agent_stage == AgentStage.OBSERVE

    def test_task_error_preserves_prior_events(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that task error preserves previously recorded events."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Complete first task
        context1 = handler.on_task_start(
            task_description="Task 1",
            agent_role="Agent1",
        )
        handler.on_task_end(task_output="Done", context=context1)

        # Start and fail second task
        context2 = handler.on_task_start(
            task_description="Task 2",
            agent_role="Agent2",
        )
        handler.on_task_error(
            error=RuntimeError("Failed"),
            context=context2,
        )

        events = handler.drain_events()

        # Should have 4 events: task1_start, task1_end, task2_start, task2_error
        assert len(events) == 4

        # First two events for successful task
        assert events[0].event_type == EventType.TASK_START
        assert events[1].event_type == EventType.TASK_END

        # Last event should be error
        assert events[3].event_type == EventType.ERROR
        assert events[3].error_type == "RuntimeError"

    def test_task_results_tracking(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that task results are tracked correctly."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Complete multiple tasks
        for i in range(3):
            context = handler.on_task_start(
                task_description=f"Task {i}",
                agent_role=f"Agent{i}",
                task_id=f"task_{i}",
            )
            handler.on_task_end(
                task_output=f"Result {i}",
                context=context,
            )

        results = handler.get_task_results()
        assert len(results) == 3
        assert results["task_0"] == "Result 0"
        assert results["task_1"] == "Result 1"
        assert results["task_2"] == "Result 2"


# =============================================================================
# Acceptance Criteria Tests (from PRD-004)
# =============================================================================


class TestAcceptanceCriteria:
    """Tests verifying PRD-004 acceptance criteria."""

    def test_wrap_crew_captures_task_events(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """
        PRD-004 AC1: Each agent task execution is captured as a distinct trace event.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Simulate multiple task executions
        for task in mock_crew.tasks:
            context = handler.on_task_start(
                task_description=task.description,
                agent_role=task.agent.role if task.agent else "unknown",
            )
            handler.on_task_end(task_output="Completed", context=context)

        events = handler.drain_events()

        # Each task should have start and end events
        assert len(events) == len(mock_crew.tasks) * 2

    def test_event_types_differentiated(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """
        PRD-004 AC1: event_type differentiates between task_start, task_end, and tool_call.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Task start
        context = handler.on_task_start(
            task_description="Test task",
            agent_role="Tester",
        )

        # Tool call
        handler.on_agent_action(
            agent_role="Tester",
            action_type="tool_call",
            tool_name="search",
        )

        # Task end
        handler.on_task_end(task_output="Done", context=context)

        events = handler.drain_events()

        assert events[0].event_type == EventType.TASK_START
        assert events[1].event_type == EventType.TOOL_CALL
        assert events[2].event_type == EventType.TASK_END

    def test_task_failure_captured_with_context(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """
        PRD-004 AC2: Task failure point and context are captured in the trace.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_task_start(
            task_description="Failing task",
            agent_role="FailingAgent",
            task_id="fail_task_001",
        )

        error = RuntimeError("Task execution failed at step 3")
        event = handler.on_task_error(error=error, context=context)

        assert event.error_type == "RuntimeError"
        assert event.error_message == "Task execution failed at step 3"
        assert event.error_stack is not None
        assert event.metadata["task_key"] == "fail_task_001"
        assert event.metadata["agent_role"] == "FailingAgent"

    def test_unsupported_version_raises_framework_error(self, mock_crewai_unsupported, mock_crew, config):
        """
        PRD-004 AC3: Unsupported version raises FrameworkVersionError.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        with pytest.raises(FrameworkVersionError) as exc_info:
            CrewAIAdapter(mock_crew, config)

        assert exc_info.value.framework == "crewai"
        assert "0.25.0" in exc_info.value.version
        assert ">=0.30.0" in exc_info.value.supported

    def test_no_partial_traces_on_version_error(self, mock_crewai_unsupported, mock_crew, config):
        """
        PRD-004 AC3: No partial traces are generated on version error.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        # Attempting to create adapter should fail before any tracing occurs
        try:
            adapter = CrewAIAdapter(mock_crew, config)
            # If we get here, the adapter was created (shouldn't happen)
            handler = adapter.create_callback_handler(MagicMock())
            handler.drain_events()
            # Should have no events since adapter creation should have failed
            raise AssertionError("Should have raised FrameworkVersionError")
        except FrameworkVersionError:
            # Expected behavior - no adapter created, no traces possible
            pass


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_crew(self, mock_crewai_module, config, session):
        """Test handling of empty crew (no agents or tasks)."""
        from clyro.adapters.crewai import CrewAIAdapter

        empty_crew = MockCrew(agents=[], tasks=[], name="empty_crew")
        adapter = CrewAIAdapter(empty_crew, config)

        context = adapter.before_call(session, (), {})
        event = adapter.after_call(session, MockCrewOutput(), context)

        assert event is not None
        assert event.event_type == EventType.STEP

    def test_long_task_description_truncation(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that long task descriptions are truncated appropriately."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        long_description = "A" * 1000  # Very long description
        handler.on_task_start(
            task_description=long_description,
            agent_role="Agent",
        )

        # Event name should be truncated
        events = handler.drain_events()
        assert len(events[0].event_name) < len(long_description)

    def test_none_inputs_handling(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test handling of None inputs."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        context = handler.on_crew_start(crew_name="test", inputs=None)
        assert context is not None
        assert handler._crew_state["inputs"] == {}

    def test_special_characters_in_names(
        self, mock_crewai_module, config, session
    ):
        """Test handling of special characters in crew/agent names."""
        from clyro.adapters.crewai import CrewAIAdapter

        special_crew = MockCrew(name="test-crew_v2.0 (beta)")
        adapter = CrewAIAdapter(special_crew, config)

        assert adapter.name == "test-crew_v2.0 (beta)"

        handler = adapter.create_callback_handler(session)
        handler.on_task_start(
            task_description="Task with 'quotes' and \"double quotes\"",
            agent_role="Agent/Role",
        )

        events = handler.drain_events()
        assert len(events) == 1


# =============================================================================
# Async Execution Tests
# =============================================================================


class TestAsyncExecution:
    """Tests for async CrewAI execution support."""

    @pytest.mark.asyncio
    async def test_mock_crew_async_kickoff(self, mock_crew):
        """Test that MockCrew supports async kickoff."""
        result = await mock_crew.akickoff(inputs={"topic": "AI"})
        assert result.raw == "Crew completed successfully"

    @pytest.mark.asyncio
    async def test_adapter_supports_async_crew(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that adapter can handle async crew execution context."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)

        # Verify adapter can be created for async-capable crew
        assert adapter.agent == mock_crew
        assert hasattr(mock_crew, "akickoff")

        # Test hooks work in async context
        kwargs: dict[str, Any] = {"inputs": {"topic": "Async Test"}}
        context = adapter.before_call(session, (), kwargs)

        # Simulate async execution
        result = await mock_crew.akickoff(inputs=kwargs.get("inputs"))

        event = adapter.after_call(session, result, context)
        assert event is not None
        assert "research_crew_complete" in event.event_name


# =============================================================================
# Step Number Tracking Tests
# =============================================================================


class TestStepNumberTracking:
    """Tests for step number tracking in events."""

    def test_step_numbers_increment_correctly(
        self, mock_crewai_module, mock_crew, config, session
    ):
        """Test that step numbers increment for each event."""
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Initial step number from session
        initial_step = session.step_number

        # Create multiple events
        context1 = handler.on_task_start(
            task_description="Task 1",
            agent_role="Agent1",
        )
        handler.on_task_end(task_output="Result 1", context=context1)

        handler.on_agent_action(
            agent_role="Agent1",
            action_type="tool_call",
            tool_name="search",
        )

        events = handler.drain_events()

        # Verify step numbers are sequential starting at initial_step + 1
        # (step 0 is reserved for SESSION_START to avoid record_event reassignment)
        assert len(events) == 3
        assert events[0].step_number == initial_step + 1  # task_start
        assert events[1].step_number == initial_step + 2  # task_end
        assert events[2].step_number == initial_step + 3  # agent_action

    def test_step_number_starts_from_session(
        self, mock_crewai_module, mock_crew, config
    ):
        """Test that step counter starts from session's step number."""
        from clyro.adapters.crewai import CrewAIAdapter

        # Create session with non-zero step number
        session_with_steps = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session_with_steps.start()
        # Simulate session already having some steps
        session_with_steps._step_number = 10

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session_with_steps)

        handler.on_task_start(
            task_description="Test task",
            agent_role="TestAgent",
        )

        events = handler.drain_events()
        # Handler starts at session.step_number + 1 to avoid step 0 collision
        assert events[0].step_number == 11


# =============================================================================
# Execution Control Enforcement Tests
# =============================================================================


class TestExecutionControlEnforcement:
    """Tests for step limit and cost limit enforcement during CrewAI execution."""

    def _make_session_with_step_limit(self, max_steps: int) -> Session:
        """Create a session with step limit enabled."""
        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            capture_state=True,
            controls=ExecutionControls(max_steps=max_steps),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()
        return session

    def _make_session_with_cost_limit(self, max_cost_usd: float) -> Session:
        """Create a session with cost limit enabled."""
        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            capture_state=True,
            controls=ExecutionControls(max_cost_usd=max_cost_usd, enable_loop_detection=False),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()
        return session

    def test_next_step_raises_step_limit_exceeded(self, mock_crewai_module, mock_crew):
        """Test that _next_step raises StepLimitExceededError when limit is exceeded."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_step_limit(max_steps=3)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # Steps 1, 2, 3 should be fine
        handler._next_step()  # step_counter becomes 1
        handler._next_step()  # step_counter becomes 2
        handler._next_step()  # step_counter becomes 3

        # Step 4 should raise (exceeds max_steps=3)
        with pytest.raises(StepLimitExceededError) as exc_info:
            handler._next_step()

        assert exc_info.value.limit == 3

    def test_next_step_syncs_session_step_number(self, mock_crewai_module, mock_crew):
        """Test that _next_step syncs the session's step number."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_step_limit(max_steps=100)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        assert session._step_number == 0

        handler._next_step()
        assert session._step_number == 1

        handler._next_step()
        assert session._step_number == 2

        handler._next_step()
        assert session._step_number == 3

    def test_step_limit_enforced_during_agent_action(self, mock_crewai_module, mock_crew):
        """Test that step limit is enforced during on_agent_action (LLM/tool calls)."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_step_limit(max_steps=2)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # First two actions should succeed
        handler.on_agent_action(agent_role="Agent", action_type="llm_call", model="gpt-4")
        handler.on_agent_action(agent_role="Agent", action_type="tool_call", tool_name="search")

        # Third action should raise
        with pytest.raises(StepLimitExceededError):
            handler.on_agent_action(agent_role="Agent", action_type="llm_call", model="gpt-4")

    def test_step_limit_enforced_during_task_start(self, mock_crewai_module, mock_crew):
        """Test that step limit is enforced during on_task_start."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_step_limit(max_steps=1)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # First task start uses step 1
        handler.on_task_start(
            task_description="Task 1",
            agent_role="Agent",
        )

        # Second task start should exceed limit
        with pytest.raises(StepLimitExceededError):
            handler.on_task_start(
                task_description="Task 2",
                agent_role="Agent",
            )

    def test_step_limit_not_enforced_when_disabled(self, mock_crewai_module, mock_crew):
        """Test that step limit is not enforced when max_steps is not set."""
        from clyro.adapters.crewai import CrewAIAdapter

        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Should be able to take many steps without raising
        for _ in range(50):
            handler._next_step()

        assert session._step_number == 50

    def test_cost_limit_enforced_during_llm_call(self, mock_crewai_module, mock_crew):
        """Test that cost limit is enforced in real-time during LLM calls.

        The handler tracks cumulative cost locally. When an LLM call pushes
        the total over the limit, CostLimitExceededError is raised immediately
        — not deferred to event draining.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        # Default pricing: $0.01/1K input + $0.03/1K output = $0.04 per call
        # With max_cost_usd=0.05, second call should exceed the limit
        session = self._make_session_with_cost_limit(max_cost_usd=0.05)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # First LLM call: cost = $0.04, cumulative = $0.04 (within limit)
        handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )
        assert handler._local_cumulative_cost == Decimal("0.04")

        # Second LLM call: cost = $0.04, cumulative = $0.08 (exceeds limit)
        with pytest.raises(CostLimitExceededError) as exc_info:
            handler.on_agent_action(
                agent_role="Agent",
                action_type="llm_call",
                model="unknown-model",
                token_count_input=1000,
                token_count_output=1000,
            )

        assert exc_info.value.limit_usd == 0.05
        assert exc_info.value.current_cost_usd == float(Decimal("0.08"))

    def test_cost_limit_not_triggered_by_tool_calls(self, mock_crewai_module, mock_crew):
        """Test that tool calls (which have no cost) don't trigger cost limit."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_cost_limit(max_cost_usd=0.01)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # Tool calls don't contribute cost, so they should never trigger cost limit
        for _ in range(10):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="tool_call",
                tool_name="search",
            )
        assert handler._local_cumulative_cost == Decimal("0")

    def test_cost_limit_not_enforced_when_disabled(self, mock_crewai_module, mock_crew):
        """Test that cost limit is not enforced when max_cost_usd is not set."""
        from clyro.adapters.crewai import CrewAIAdapter

        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            controls=ExecutionControls(enable_loop_detection=False),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Many expensive LLM calls should not raise
        for _ in range(20):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="llm_call",
                model="unknown-model",
                token_count_input=10000,
                token_count_output=10000,
            )

    def test_local_cumulative_cost_initializes_from_session(self, mock_crewai_module, mock_crew):
        """Test that handler's local cost tracker initializes from session's existing cost."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_cost_limit(max_cost_usd=1.0)
        # Simulate pre-existing cost from a prior session or pre-execution
        session._cumulative_cost = Decimal("0.50")

        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        assert handler._local_cumulative_cost == Decimal("0.50")

    def test_cost_limit_with_preexisting_session_cost(self, mock_crewai_module, mock_crew):
        """Test that cost limit accounts for pre-existing session cost.

        If the session already has $0.90 of cost and the limit is $1.00,
        a single LLM call costing $0.04 should push it over and raise.
        """
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_cost_limit(max_cost_usd=1.0)
        session._cumulative_cost = Decimal("0.98")

        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # $0.98 + $0.04 = $1.02 > $1.00
        with pytest.raises(CostLimitExceededError):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="llm_call",
                model="unknown-model",
                token_count_input=1000,
                token_count_output=1000,
            )

    def test_cumulative_cost_on_events_reflects_local_tracker(self, mock_crewai_module, mock_crew):
        """Test that events created during execution carry the correct cumulative_cost."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_cost_limit(max_cost_usd=10.0)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # First LLM call
        event1 = handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )
        assert event1.cumulative_cost == Decimal("0.04")

        # Tool call — no cost, cumulative stays the same
        event2 = handler.on_agent_action(
            agent_role="Agent",
            action_type="tool_call",
            tool_name="search",
        )
        assert event2.cumulative_cost == Decimal("0.04")

        # Second LLM call
        event3 = handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )
        assert event3.cumulative_cost == Decimal("0.08")

    def test_record_event_checks_step_limit(self, mock_crewai_module, mock_crew):
        """Test that session.record_event() checks step limits as a safety net."""
        from clyro.adapters.crewai import CrewAIAdapter
        from clyro.trace import create_step_event

        session = self._make_session_with_step_limit(max_steps=2)
        CrewAIAdapter(mock_crew, session.config)

        # Create an event with step_number that exceeds limit
        event = create_step_event(
            session_id=session.session_id,
            step_number=5,  # Exceeds max_steps=2
            event_name="test_step",
            agent_id=session.agent_id,
            framework=Framework.CREWAI,
        )

        with pytest.raises(StepLimitExceededError):
            session.record_event(event)

    def test_record_event_syncs_step_number(self, mock_crewai_module, mock_crew):
        """Test that session.record_event() syncs step number from adapter events."""
        from clyro.trace import create_step_event

        session = self._make_session_with_step_limit(max_steps=100)

        assert session._step_number == 0

        # Record event with pre-assigned step number (as CrewAI adapter does)
        event = create_step_event(
            session_id=session.session_id,
            step_number=7,
            event_name="test_step",
            agent_id=session.agent_id,
            framework=Framework.CREWAI,
        )
        session.record_event(event)

        # Session step number should be synced
        assert session._step_number == 7

    def test_full_crewai_step_limit_scenario(self, mock_crewai_module, mock_crew):
        """Test realistic CrewAI scenario: multiple events hit step limit mid-execution."""
        from clyro.adapters.crewai import CrewAIAdapter

        session = self._make_session_with_step_limit(max_steps=6)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # Simulate a typical CrewAI execution flow
        # Step 1: task_start
        ctx = handler.on_task_start(
            task_description="Research task",
            agent_role="Researcher",
        )
        # Step 2: llm_call
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="llm_call",
            model="gpt-4",
        )
        # Step 3: tool_call
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="tool_call",
            tool_name="search",
        )
        # Step 4: llm_call
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="llm_call",
            model="gpt-4",
        )
        # Step 5: task_end
        handler.on_task_end(task_output="Done", context=ctx)
        # Step 6: task_start for next task
        handler.on_task_start(
            task_description="Writing task",
            agent_role="Writer",
        )

        # Step 7: Should exceed max_steps=6
        with pytest.raises(StepLimitExceededError) as exc_info:
            handler.on_agent_action(
                agent_role="Writer",
                action_type="llm_call",
                model="gpt-4",
            )

        assert exc_info.value.limit == 6
        assert session._step_number == 7

    def test_full_crewai_cost_limit_scenario(self, mock_crewai_module, mock_crew):
        """Test realistic CrewAI scenario: LLM calls accumulate cost until limit is hit."""
        from clyro.adapters.crewai import CrewAIAdapter

        # Default pricing: $0.01/1K input + $0.03/1K output
        # Each call with 1000 input + 1000 output = $0.04
        # Limit at $0.10 → allows 2 calls ($0.08), fails on 3rd ($0.12)
        session = self._make_session_with_cost_limit(max_cost_usd=0.10)
        adapter = CrewAIAdapter(mock_crew, session.config)
        handler = adapter.create_callback_handler(session)

        # Task start (no cost)
        handler.on_task_start(
            task_description="Research task",
            agent_role="Researcher",
        )

        # LLM call 1: $0.04 cumulative
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )

        # Tool call (no cost)
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="tool_call",
            tool_name="search",
        )

        # LLM call 2: $0.08 cumulative (still within limit)
        handler.on_agent_action(
            agent_role="Researcher",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )
        assert handler._local_cumulative_cost == Decimal("0.08")

        # LLM call 3: $0.12 cumulative (exceeds $0.10 limit)
        with pytest.raises(CostLimitExceededError) as exc_info:
            handler.on_agent_action(
                agent_role="Researcher",
                action_type="llm_call",
                model="unknown-model",
                token_count_input=1000,
                token_count_output=1000,
            )

        assert exc_info.value.limit_usd == 0.10
        assert handler._local_cumulative_cost == Decimal("0.12")


# =============================================================================
# Cost-Based Policy Enforcement Tests
# =============================================================================


class TestCostPolicyEnforcement:
    """Tests for cost-based policy enforcement via backend policy evaluation.

    These tests verify that:
    1. Policy checks include the `cost` parameter from the handler's local tracker
    2. PolicyViolationError propagates correctly during event bus handler flow
    """

    def _make_session_with_mock_policy(self, mock_crewai_module, mock_crew):
        """Create a handler with a session whose check_policy can be observed."""
        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()
        # Mock check_policy so we can inspect calls without a real backend
        session.check_policy = MagicMock()

        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)
        return handler, session

    def test_policy_check_receives_cost_after_llm_call(self, mock_crewai_module, mock_crew):
        """Verify that the cost sent to check_policy matches the handler's local tracker.

        This simulates what the event bus on_llm_completed handler does:
        1. Calls handler.on_agent_action(action_type="llm_call") which accumulates cost
        2. Calls session.check_policy("llm_call", {"model": ..., "cost": ...})
        """
        handler, session = self._make_session_with_mock_policy(mock_crewai_module, mock_crew)

        # Simulate LLM call (cost = $0.04 with default pricing)
        handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )

        # Simulate what the event bus handler does after on_agent_action
        session.check_policy("llm_call", {
            "model": "unknown-model",
            "cost": float(handler._local_cumulative_cost),
        })

        session.check_policy.assert_called_once_with("llm_call", {
            "model": "unknown-model",
            "cost": float(Decimal("0.04")),
        })

    def test_policy_check_cost_accumulates_across_llm_calls(self, mock_crewai_module, mock_crew):
        """Verify cumulative cost is sent, not per-call cost."""
        handler, session = self._make_session_with_mock_policy(mock_crewai_module, mock_crew)

        # Two LLM calls — each costs $0.04
        for _ in range(2):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="llm_call",
                model="gpt-4",
                token_count_input=1000,
                token_count_output=1000,
            )

        # After 2 calls, cumulative cost = $0.08
        session.check_policy("llm_call", {
            "model": "gpt-4",
            "cost": float(handler._local_cumulative_cost),
        })

        call_args = session.check_policy.call_args
        assert call_args[0][1]["cost"] == float(Decimal("0.08"))

    def test_tool_call_policy_check_includes_cost(self, mock_crewai_module, mock_crew):
        """Verify that tool_call policy checks also include the current cost."""
        handler, session = self._make_session_with_mock_policy(mock_crewai_module, mock_crew)

        # LLM call to accumulate some cost
        handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )

        # Tool call (no cost itself)
        handler.on_agent_action(
            agent_role="Agent",
            action_type="tool_call",
            tool_name="search",
        )

        # Simulate what the event bus on_tool_finished handler does
        session.check_policy("tool_call", {
            "tool_name": "search",
            "cost": float(handler._local_cumulative_cost),
        })

        call_args = session.check_policy.call_args
        assert call_args[0][0] == "tool_call"
        assert call_args[0][1]["tool_name"] == "search"
        assert call_args[0][1]["cost"] == float(Decimal("0.04"))

    def test_policy_violation_propagates_from_llm_check(self, mock_crewai_module, mock_crew):
        """Verify that PolicyViolationError from cost-based policy check propagates."""
        from clyro.exceptions import PolicyViolationError

        handler, session = self._make_session_with_mock_policy(mock_crewai_module, mock_crew)

        # Configure mock to raise PolicyViolationError when cost exceeds threshold
        def cost_policy_check(action_type, params):
            if params.get("cost", 0) > 0.00005:
                raise PolicyViolationError(
                    rule_id="cost-cap",
                    rule_name="session_cost_limit",
                    message="Session cost exceeded $0.000050 limit.",
                    action_type=action_type,
                )

        session.check_policy = MagicMock(side_effect=cost_policy_check)

        # LLM call costs $0.04 — well above $0.00005 limit
        handler.on_agent_action(
            agent_role="Agent",
            action_type="llm_call",
            model="unknown-model",
            token_count_input=1000,
            token_count_output=1000,
        )

        # The policy check should raise
        with pytest.raises(PolicyViolationError) as exc_info:
            session.check_policy("llm_call", {
                "model": "unknown-model",
                "cost": float(handler._local_cumulative_cost),
            })

        assert exc_info.value.rule_id == "cost-cap"
        assert "cost exceeded" in exc_info.value.message.lower()


# =============================================================================
# Deferred Error Propagation Tests
# =============================================================================


class TestDeferredErrorPropagation:
    """Tests for deferred error propagation when CrewAI's event bus swallows exceptions.

    CrewAI's event bus wraps handler calls in try/except Exception and logs errors
    instead of propagating them. The adapter stores enforcement errors on the
    handler as _pending_error. These tests verify:
    1. Errors are stored as _pending_error
    2. _next_step() re-raises pending errors to prevent further tracking
    3. The wrapper's _raise_pending_adapter_error() detects and re-raises them
    """

    def test_pending_error_initialized_as_none(self, mock_crewai_module, mock_crew):
        """Test that _pending_error starts as None."""
        from clyro.adapters.crewai import CrewAIAdapter

        config = ClyroConfig(capture_inputs=True)
        session = Session(config=config, agent_id=uuid4(), org_id=uuid4(), framework=Framework.CREWAI)
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        assert handler._pending_error is None

    def test_next_step_raises_pending_policy_violation(self, mock_crewai_module, mock_crew):
        """Test that _next_step() raises a stored PolicyViolationError."""
        from clyro.adapters.crewai import CrewAIAdapter
        from clyro.exceptions import PolicyViolationError

        config = ClyroConfig(capture_inputs=True)
        session = Session(config=config, agent_id=uuid4(), org_id=uuid4(), framework=Framework.CREWAI)
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        # Simulate what happens when CrewAI's event bus swallows the error:
        # The handler stores it as _pending_error
        stored_error = PolicyViolationError(
            rule_id="cost-cap",
            rule_name="session_cost_limit",
            message="Session cost exceeded limit.",
            action_type="llm_call",
        )
        handler._pending_error = stored_error

        # Next step should re-raise the stored error
        with pytest.raises(PolicyViolationError) as exc_info:
            handler._next_step()

        assert exc_info.value is stored_error

    def test_next_step_raises_pending_step_limit_error(self, mock_crewai_module, mock_crew):
        """Test that _next_step() raises a stored StepLimitExceededError."""
        from clyro.adapters.crewai import CrewAIAdapter

        config = ClyroConfig(capture_inputs=True, controls=ExecutionControls(max_steps=100))
        session = Session(config=config, agent_id=uuid4(), org_id=uuid4(), framework=Framework.CREWAI)
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        stored_error = StepLimitExceededError(limit=5, current_step=6, session_id="test")
        handler._pending_error = stored_error

        with pytest.raises(StepLimitExceededError) as exc_info:
            handler._next_step()

        assert exc_info.value is stored_error

    def test_on_agent_action_raises_pending_error(self, mock_crewai_module, mock_crew):
        """Test that on_agent_action (which calls _next_step) raises pending error."""
        from clyro.adapters.crewai import CrewAIAdapter
        from clyro.exceptions import PolicyViolationError

        config = ClyroConfig(capture_inputs=True)
        session = Session(config=config, agent_id=uuid4(), org_id=uuid4(), framework=Framework.CREWAI)
        session.start()

        adapter = CrewAIAdapter(mock_crew, config)
        handler = adapter.create_callback_handler(session)

        handler._pending_error = PolicyViolationError(
            rule_id="test", rule_name="test", message="blocked", action_type="llm_call",
        )

        with pytest.raises(PolicyViolationError):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="llm_call",
                model="gpt-4",
            )

    def test_wrapper_raise_pending_adapter_error(self, mock_crewai_module, mock_crew):
        """Test that the wrapper's _raise_pending_adapter_error detects and raises."""
        from clyro.exceptions import PolicyViolationError
        from clyro.wrapper import WrappedAgent

        error = PolicyViolationError(
            rule_id="cost-cap",
            rule_name="session_cost_limit",
            message="Session cost exceeded limit.",
            action_type="llm_call",
        )

        # Simulate adapter_context with a handler that has a pending error
        mock_handler = MagicMock()
        mock_handler._pending_error = error
        adapter_context = {"handler": mock_handler}

        with pytest.raises(PolicyViolationError) as exc_info:
            WrappedAgent._raise_pending_adapter_error(adapter_context)

        assert exc_info.value is error

    def test_wrapper_raise_pending_no_error(self, mock_crewai_module, mock_crew):
        """Test that _raise_pending_adapter_error is a no-op when no error is pending."""
        from clyro.wrapper import WrappedAgent

        mock_handler = MagicMock()
        mock_handler._pending_error = None
        adapter_context = {"handler": mock_handler}

        # Should not raise
        WrappedAgent._raise_pending_adapter_error(adapter_context)

    def test_wrapper_raise_pending_no_context(self, mock_crewai_module, mock_crew):
        """Test that _raise_pending_adapter_error handles None adapter_context."""
        from clyro.wrapper import WrappedAgent

        # Should not raise
        WrappedAgent._raise_pending_adapter_error(None)
        WrappedAgent._raise_pending_adapter_error({})


# =============================================================================
# Trace Hierarchy — FRD-003/FRD-004: CrewAI task-level parent event tracking
# =============================================================================


class TestCrewAITraceHierarchy:
    """Tests for FRD-003/FRD-004: CrewAI task-level parent event tracking."""

    def _make_handler(self):
        """Create a handler with minimal dependencies for hierarchy testing."""
        from clyro.adapters.crewai import CrewAICallbackHandler

        config = ClyroConfig(agent_name="test")
        session = Session(config)
        handler = CrewAICallbackHandler(
            session=session, config=config, framework_version="0.55.0",
        )
        return handler

    def test_task_start_stores_event_id_in_map(self):
        """FRD-003: on_task_start stores task_key → event_id in _task_event_ids."""
        handler = self._make_handler()
        handler.on_task_start("Do research", "Researcher", task_id="task-1")
        assert "task-1" in handler._task_event_ids
        assert isinstance(handler._task_event_ids["task-1"], UUID)

    def test_agent_action_inherits_task_parent(self):
        """FRD-003: on_agent_action child events inherit task's event_id as parent_event_id."""
        handler = self._make_handler()
        handler.on_task_start("Do research", "Researcher", task_id="task-1")
        task_eid = handler._task_event_ids["task-1"]

        event = handler.on_agent_action(
            agent_role="Researcher", action_type="tool_call", tool_name="search",
        )
        assert event.parent_event_id == task_eid

    def test_llm_call_inherits_task_parent(self):
        """FRD-003: LLM_CALL events within a task inherit the task's event_id."""
        handler = self._make_handler()
        handler.on_task_start("Analyze data", "Analyst", task_id="task-2")
        task_eid = handler._task_event_ids["task-2"]

        event = handler.on_agent_action(
            agent_role="Analyst", action_type="llm_call", model="gpt-4",
            token_count_input=100, token_count_output=50,
        )
        assert event.parent_event_id == task_eid

    def test_task_start_event_has_no_parent(self):
        """FRD-003: TASK_START events have parent_event_id = None."""
        handler = self._make_handler()
        handler.on_task_start("Work", "Worker", task_id="t1")
        events = handler.drain_events()
        task_start_event = [e for e in events if e.event_type.value == "task_start"][0]
        assert task_start_event.parent_event_id is None

    def test_task_end_removes_entry_from_map(self):
        """FRD-004: on_task_end removes task_key from _task_event_ids."""
        handler = self._make_handler()
        ctx = handler.on_task_start("Work", "Worker", task_id="t1")
        assert "t1" in handler._task_event_ids
        handler.on_task_end("result", ctx)
        assert "t1" not in handler._task_event_ids

    def test_task_error_removes_entry_from_map(self):
        """FRD-004: on_task_error removes task_key from _task_event_ids."""
        handler = self._make_handler()
        ctx = handler.on_task_start("Work", "Worker", task_id="t1")
        assert "t1" in handler._task_event_ids
        handler.on_task_error(RuntimeError("fail"), ctx)
        assert "t1" not in handler._task_event_ids

    def test_concurrent_tasks_isolated_parents(self):
        """FRD-004: Concurrent tasks have independent parent tracking per task_key."""
        handler = self._make_handler()
        handler.on_task_start("Task A", "Agent-A", task_id="tA")
        handler.on_task_start("Task B", "Agent-B", task_id="tB")

        eid_a = handler._task_event_ids["tA"]
        eid_b = handler._task_event_ids["tB"]
        assert eid_a != eid_b

        # Simulate action under task B context (current_task = "tB")
        event_b = handler.on_agent_action(
            agent_role="Agent-B", action_type="tool_call", tool_name="calc",
        )
        assert event_b.parent_event_id == eid_b

    def test_action_outside_task_has_no_parent(self):
        """FRD-003: Actions outside any task have parent_event_id = None."""
        handler = self._make_handler()
        event = handler.on_agent_action(
            agent_role="Researcher", action_type="tool_call", tool_name="search",
        )
        assert event.parent_event_id is None

    def test_duplicate_task_key_overwrites(self):
        """FRD-004: Duplicate task_key overwrites existing entry in _task_event_ids."""
        handler = self._make_handler()
        handler.on_task_start("First", "A", task_id="dup")
        first_eid = handler._task_event_ids["dup"]
        handler.on_task_start("Second", "B", task_id="dup")
        second_eid = handler._task_event_ids["dup"]
        assert first_eid != second_eid


# =============================================================================
# Loop Detection Tests (PRD-010)
# =============================================================================


class TestCrewAILoopDetection:
    """Tests for loop detection during CrewAI execution."""

    def _make_session_with_loop_detection(self, threshold: int = 3) -> Session:
        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            controls=ExecutionControls(
                max_steps=100,
                loop_detection_threshold=threshold,
            ),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()
        return session

    def _make_handler(self, session, mock_crewai_module, mock_crew):
        from clyro.adapters.crewai import CrewAIAdapter

        adapter = CrewAIAdapter(mock_crew, session.config)
        return adapter.create_callback_handler(session)

    def test_loop_detected_on_repeated_actions(self, mock_crewai_module, mock_crew):
        """Loop detection triggers when the same action repeats >= threshold times."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_crewai_module, mock_crew)

        identical_input = {"query": "same"}

        with pytest.raises(LoopDetectedError):
            for _ in range(10):
                handler.on_agent_action(
                    agent_role="Agent",
                    action_type="tool_call",
                    tool_name="search",
                    action_input=identical_input,
                )

    def test_loop_detection_creates_error_event(self, mock_crewai_module, mock_crew):
        """Loop detection records an error event before raising."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_crewai_module, mock_crew)

        identical_input = {"query": "same"}

        with pytest.raises(LoopDetectedError):
            for _ in range(10):
                handler.on_agent_action(
                    agent_role="Agent",
                    action_type="tool_call",
                    tool_name="search",
                    action_input=identical_input,
                )

        # Check that an error event was recorded
        error_events = [e for e in handler._events if e.event_type.value == "error"]
        assert len(error_events) == 1
        assert error_events[0].error_type == "LoopDetectedError"
        assert error_events[0].metadata["event_name"] == "loop_detection"

    def test_no_loop_with_different_inputs(self, mock_crewai_module, mock_crew):
        """No loop detected when action inputs vary."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_crewai_module, mock_crew)

        for i in range(10):
            handler.on_agent_action(
                agent_role=f"Agent_{i}",
                action_type=f"tool_call_{i}",
                tool_name="search",
                action_input={"query": f"different_{i}"},
            )

    def test_loop_detection_disabled(self, mock_crewai_module, mock_crew):
        """No error when loop detection is disabled."""
        config = ClyroConfig(
            controls=ExecutionControls(
                max_steps=100,
                enable_loop_detection=False,
                loop_detection_threshold=3,
            ),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.CREWAI,
        )
        session.start()
        handler = self._make_handler(session, mock_crewai_module, mock_crew)

        identical_input = {"query": "same"}
        for _ in range(10):
            handler.on_agent_action(
                agent_role="Agent",
                action_type="tool_call",
                tool_name="search",
                action_input=identical_input,
            )
