# Tests for Clyro SDK LangGraph Adapter
# Implements PRD-003

"""
Unit tests for the LangGraph adapter.

These tests use mock LangGraph objects to avoid requiring LangGraph
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
from clyro.exceptions import FrameworkVersionError, LoopDetectedError, StepLimitExceededError
from clyro.session import Session
from clyro.trace import AgentStage, EventType, Framework

# =============================================================================
# Mock LangGraph Module and Classes
# =============================================================================


class MockLangGraphModule(ModuleType):
    """Mock langgraph module for testing without the actual dependency."""

    def __init__(self, version: str = "0.2.5"):
        super().__init__("langgraph")
        self.__version__ = version


class MockStateGraph:
    """Mock LangGraph StateGraph class."""

    __module__ = "langgraph.graph"

    def __init__(self, name: str = "test_graph"):
        self.name = name
        self._nodes: dict[str, Any] = {}
        self._edges: list[tuple[str, str]] = []

    def add_node(self, name: str, func: Any) -> None:
        """Add a node to the graph."""
        self._nodes[name] = func

    def add_edge(self, source: str, target: str) -> None:
        """Add an edge to the graph."""
        self._edges.append((source, target))

    def compile(self) -> MockCompiledGraph:
        """Compile the graph."""
        return MockCompiledGraph(self)


class MockCompiledGraph:
    """Mock LangGraph CompiledGraph class."""

    __module__ = "langgraph.graph.state"

    def __init__(self, graph: MockStateGraph | None = None):
        self.graph = graph
        self.name = graph.name if graph else "compiled_graph"

    def invoke(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the compiled graph synchronously."""
        # Simulate node execution
        result = dict(inputs)
        result["processed"] = True
        return result

    async def ainvoke(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the compiled graph asynchronously."""
        result = dict(inputs)
        result["processed"] = True
        return result

    def get_graph(self) -> MockStateGraph:
        """Get the underlying graph."""
        return self.graph or MockStateGraph()


class MockRunnableConfig:
    """Mock LangGraph RunnableConfig for callback injection."""

    def __init__(self, callbacks: list | None = None):
        self.callbacks = callbacks or []


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_langgraph_module():
    """Create and install a mock langgraph module."""
    mock_module = MockLangGraphModule(version="0.2.5")
    with patch.dict(sys.modules, {"langgraph": mock_module}):
        yield mock_module


@pytest.fixture
def mock_langgraph_unsupported():
    """Create a mock langgraph module with unsupported version."""
    mock_module = MockLangGraphModule(version="0.1.0")
    with patch.dict(sys.modules, {"langgraph": mock_module}):
        yield mock_module


@pytest.fixture
def mock_state_graph():
    """Create a mock StateGraph."""
    graph = MockStateGraph(name="test_agent")
    graph.add_node("agent", lambda x: x)
    graph.add_node("tools", lambda x: x)
    graph.add_edge("agent", "tools")
    return graph


@pytest.fixture
def mock_compiled_graph(mock_state_graph):
    """Create a mock CompiledGraph."""
    return mock_state_graph.compile()


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
        framework=Framework.LANGGRAPH,
    )
    session.start()
    return session


# =============================================================================
# Version Detection and Validation Tests
# =============================================================================


class TestVersionDetection:
    """Tests for LangGraph version detection and validation."""

    def test_detect_version_returns_installed_version(self, mock_langgraph_module):
        """Test that version detection returns the installed version."""
        from clyro.adapters.langgraph import detect_langgraph_version

        version = detect_langgraph_version()
        assert version == "0.2.5"

    def test_detect_version_returns_none_when_not_installed(self):
        """Test that version detection returns None when LangGraph is not installed."""
        from clyro.adapters.langgraph import detect_langgraph_version

        with patch.dict(sys.modules, {"langgraph": None}):
            # Remove from sys.modules entirely
            sys.modules.pop("langgraph", None)
            version = detect_langgraph_version()
            # Should return None or handle ImportError gracefully
            assert version is None

    def test_validate_version_succeeds_for_supported(self, mock_langgraph_module):
        """Test that validation succeeds for supported versions."""
        from clyro.adapters.langgraph import validate_langgraph_version

        version = validate_langgraph_version()
        assert version == "0.2.5"

    def test_validate_version_raises_for_unsupported(self, mock_langgraph_unsupported):
        """Test that validation raises FrameworkVersionError for unsupported versions."""
        from clyro.adapters.langgraph import validate_langgraph_version

        with pytest.raises(FrameworkVersionError) as exc_info:
            validate_langgraph_version()

        assert exc_info.value.framework == "langgraph"
        assert exc_info.value.version == "0.1.0"
        assert ">=0.2.0" in exc_info.value.supported

    def test_validate_version_raises_when_not_installed(self):
        """Test that validation raises when LangGraph is not installed."""
        from clyro.adapters.langgraph import validate_langgraph_version

        with patch.dict(sys.modules, {}):
            sys.modules.pop("langgraph", None)
            with pytest.raises(FrameworkVersionError) as exc_info:
                validate_langgraph_version()

            assert exc_info.value.version == "not installed"

    def test_parse_version_handles_various_formats(self):
        """Test version parsing handles various format strings."""
        from clyro.adapters.langgraph import _parse_version

        assert _parse_version("0.2.0") == (0, 2, 0)
        assert _parse_version("0.2.5") == (0, 2, 5)
        assert _parse_version("1.0.0") == (1, 0, 0)
        assert _parse_version("0.2.0rc1") == (0, 2, 0)
        assert _parse_version("0.2.0dev1") == (0, 2, 0)
        assert _parse_version("0.2.0a1") == (0, 2, 0)


class TestIsLangGraphAgent:
    """Tests for LangGraph agent detection."""

    def test_detects_state_graph(self, mock_state_graph):
        """Test detection of StateGraph instances."""
        from clyro.adapters.langgraph import is_langgraph_agent

        assert is_langgraph_agent(mock_state_graph) is True

    def test_detects_compiled_graph(self, mock_compiled_graph):
        """Test detection of CompiledGraph instances."""
        from clyro.adapters.langgraph import is_langgraph_agent

        assert is_langgraph_agent(mock_compiled_graph) is True

    def test_does_not_detect_regular_function(self):
        """Test that regular functions are not detected as LangGraph."""
        from clyro.adapters.langgraph import is_langgraph_agent

        def regular_func():
            return "hello"

        assert is_langgraph_agent(regular_func) is False

    def test_does_not_detect_callable_class(self):
        """Test that callable classes are not detected as LangGraph."""
        from clyro.adapters.langgraph import is_langgraph_agent

        class MyCallable:
            def __call__(self):
                return "hello"

        assert is_langgraph_agent(MyCallable()) is False


# =============================================================================
# LangGraphAdapter Tests
# =============================================================================


class TestLangGraphAdapterInit:
    """Tests for LangGraphAdapter initialization."""

    def test_init_with_compiled_graph(self, mock_langgraph_module, mock_compiled_graph, config):
        """Test adapter initialization with a CompiledGraph."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)

        assert adapter.agent == mock_compiled_graph
        assert adapter.name == "test_agent"
        assert adapter.framework == Framework.LANGGRAPH
        assert adapter.framework_version == "0.2.5"

    def test_init_with_state_graph(self, mock_langgraph_module, mock_state_graph, config):
        """Test adapter initialization with a StateGraph."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_state_graph, config)

        assert adapter.agent == mock_state_graph
        assert adapter.name == "test_agent"

    def test_init_raises_for_unsupported_version(
        self, mock_langgraph_unsupported, mock_compiled_graph, config
    ):
        """Test that initialization raises for unsupported versions."""
        from clyro.adapters.langgraph import LangGraphAdapter

        with pytest.raises(FrameworkVersionError):
            LangGraphAdapter(mock_compiled_graph, config)

    def test_init_skips_validation_when_disabled(
        self, mock_langgraph_unsupported, mock_compiled_graph, config
    ):
        """Test that version validation can be skipped."""
        from clyro.adapters.langgraph import LangGraphAdapter

        # Should not raise when validation is disabled
        adapter = LangGraphAdapter(mock_compiled_graph, config, validate_version=False)
        assert adapter.framework_version == "0.1.0"

    def test_init_extracts_name_from_graph(self, mock_langgraph_module, config):
        """Test that adapter extracts name from graph."""
        from clyro.adapters.langgraph import LangGraphAdapter

        graph = MockStateGraph(name="my_custom_agent")
        compiled = graph.compile()

        adapter = LangGraphAdapter(compiled, config)
        assert adapter.name == "my_custom_agent"


class TestLangGraphAdapterHooks:
    """Tests for LangGraphAdapter lifecycle hooks."""

    def test_before_call_creates_handler(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that before_call creates a callback handler."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        kwargs: dict[str, Any] = {}

        context = adapter.before_call(session, (), kwargs)

        assert "start_time" in context
        assert "step_number" in context
        assert "handler" in context
        # step_number is 0 at call start; increments during execution
        assert context["step_number"] == 0

    def test_before_call_injects_callbacks(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that before_call injects callbacks into kwargs."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        kwargs: dict[str, Any] = {}

        adapter.before_call(session, (), kwargs)

        assert "config" in kwargs
        assert "callbacks" in kwargs["config"]
        assert len(kwargs["config"]["callbacks"]) == 1

    def test_before_call_preserves_existing_config(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that before_call preserves existing config values."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        existing_callback = MagicMock()
        kwargs = {
            "config": {
                "callbacks": [existing_callback],
                "other_key": "preserved",
            }
        }

        adapter.before_call(session, (), kwargs)

        assert kwargs["config"]["other_key"] == "preserved"
        assert len(kwargs["config"]["callbacks"]) == 2
        assert existing_callback in kwargs["config"]["callbacks"]

    def test_after_call_creates_step_event(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that after_call creates a step event."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = {"output": "test_output"}
        event = adapter.after_call(session, result, context)

        assert event.event_type == EventType.STEP
        assert event.session_id == session.session_id
        assert "test_agent_complete" in event.event_name
        assert event.agent_stage == AgentStage.OBSERVE

    def test_after_call_captures_output_when_enabled(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that after_call captures output when enabled."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = {"key": "value"}
        event = adapter.after_call(session, result, context)

        assert event.output_data is not None

    def test_after_call_no_output_when_disabled(
        self, mock_langgraph_module, mock_compiled_graph, session
    ):
        """Test that after_call doesn't capture output when disabled."""
        from clyro.adapters.langgraph import LangGraphAdapter

        no_capture_config = ClyroConfig(capture_outputs=False)
        adapter = LangGraphAdapter(mock_compiled_graph, no_capture_config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        result = {"secret": "data"}
        event = adapter.after_call(session, result, context)

        assert event.output_data is None

    def test_on_error_creates_error_event(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that on_error creates an error event."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        kwargs: dict[str, Any] = {}
        context = adapter.before_call(session, (), kwargs)

        error = ValueError("Test error")
        event = adapter.on_error(session, error, context)

        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"
        assert event.error_stack is not None
        assert event.framework == Framework.LANGGRAPH


# =============================================================================
# LangGraphCallbackHandler Tests
# =============================================================================


class TestLangGraphCallbackHandler:
    """Tests for LangGraphCallbackHandler."""

    def test_on_chain_start_tracks_node_execution(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that on_chain_start tracks node executions when parent_run_id is present."""
        from uuid import uuid4

        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        graph_run_id = uuid4()
        node_run_id = uuid4()

        # Call on_chain_start with parent_run_id (indicates node execution)
        handler.on_chain_start(
            serialized={"name": "agent"},
            inputs={"message": "hello"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
            metadata={"langgraph_node": "agent"},
        )

        # Verify node tracking was set up
        rid = str(node_run_id)
        assert rid in handler._node_start_times
        assert rid in handler._node_names
        assert handler._node_names[rid] == "agent"

    def test_on_chain_end_creates_state_transition_event(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that on_chain_end creates a state transition event for nodes."""
        from uuid import uuid4

        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        graph_run_id = uuid4()
        node_run_id = uuid4()

        # Start node execution
        handler.on_chain_start(
            serialized={"name": "agent"},
            inputs={"message": "hello"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
            metadata={"langgraph_node": "agent"},
        )

        # End node execution - should create STATE_TRANSITION event
        handler.on_chain_end(
            outputs={"response": "world"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
        )

        # Check events
        events = handler.drain_events()
        assert len(events) == 1
        event = events[0]

        assert event.event_type == EventType.STATE_TRANSITION
        assert event.event_name == "agent"
        assert event.framework == Framework.LANGGRAPH

    def test_on_chain_end_captures_state_snapshot(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that on_chain_end captures state snapshot for nodes."""
        from uuid import uuid4

        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        # Simulate top-level graph start to set initial state
        handler.on_chain_start(
            serialized={"name": "graph"},
            inputs={"initial": "state"},
            run_id=uuid4(),
        )

        graph_run_id = uuid4()
        node_run_id = uuid4()

        # Start node execution
        handler.on_chain_start(
            serialized={"name": "agent"},
            inputs={"message": "hello"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
            metadata={"langgraph_node": "agent"},
        )

        # End node execution
        handler.on_chain_end(
            outputs={"response": "world"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
        )

        # Check events
        events = handler.drain_events()
        assert len(events) == 1
        event = events[0]

        assert event.state_snapshot is not None
        assert "initial" in event.state_snapshot
        assert "response" in event.state_snapshot

    def test_on_chain_error_creates_error_event_for_nodes(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that on_chain_error creates an error event for node failures."""
        from uuid import uuid4

        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        graph_run_id = uuid4()
        node_run_id = uuid4()

        # Start node execution
        handler.on_chain_start(
            serialized={"name": "failing_node"},
            inputs={"data": "input"},
            run_id=node_run_id,
            parent_run_id=graph_run_id,
            metadata={"langgraph_node": "failing_node"},
        )

        # Node execution fails
        error = RuntimeError("Node execution failed")
        event = handler.on_chain_error(
            error=error,
            run_id=node_run_id,
            parent_run_id=graph_run_id,
        )

        assert event.event_type == EventType.ERROR
        assert event.error_type == "RuntimeError"
        assert "failing_node" in event.metadata.get("node_name", "")

    def test_determine_agent_stage_for_tool_nodes(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that tool nodes are classified as ACT stage."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("tool_executor", {})
        assert stage == AgentStage.ACT

        stage = handler._determine_agent_stage("action_node", {})
        assert stage == AgentStage.ACT

    def test_determine_agent_stage_for_observe_nodes(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that observation nodes are classified as OBSERVE stage."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("observe_results", {})
        assert stage == AgentStage.OBSERVE

        stage = handler._determine_agent_stage("result_processor", {})
        assert stage == AgentStage.OBSERVE

    def test_determine_agent_stage_from_output_content(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that stage is determined from output content."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        # Tool calls in output -> ACT
        stage = handler._determine_agent_stage(
            "agent", {"tool_calls": [{"name": "search"}]}
        )
        assert stage == AgentStage.ACT

        # Observation in output -> OBSERVE
        stage = handler._determine_agent_stage(
            "agent", {"observation": "search results"}
        )
        assert stage == AgentStage.OBSERVE

    def test_determine_agent_stage_defaults_to_think(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that default stage is THINK."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        stage = handler._determine_agent_stage("agent", {"response": "thinking..."})
        assert stage == AgentStage.THINK


# =============================================================================
# Callback Injection Tests
# =============================================================================


class TestCallbackInjection:
    """Tests for callback injection into RunnableConfig."""

    def test_inject_callbacks_creates_new_config(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that inject_callbacks creates new config when none exists."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        result = adapter.inject_callbacks(None, handler)

        assert "callbacks" in result
        assert handler in result["callbacks"]

    def test_inject_callbacks_preserves_existing(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that inject_callbacks preserves existing callbacks."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        existing = MagicMock()
        original_config = {"callbacks": [existing], "other": "value"}

        result = adapter.inject_callbacks(original_config, handler)

        assert existing in result["callbacks"]
        assert handler in result["callbacks"]
        assert result["other"] == "value"

    def test_inject_callbacks_does_not_mutate_original(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that inject_callbacks doesn't mutate original config."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        original_config = {"callbacks": [], "key": "value"}
        original_callbacks_len = len(original_config["callbacks"])

        adapter.inject_callbacks(original_config, handler)

        # Original should be unchanged
        assert len(original_config["callbacks"]) == original_callbacks_len


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for data serialization in the handler."""

    def test_serialize_primitives(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test serialization of primitive types."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        assert handler._serialize_value(None) is None
        assert handler._serialize_value("string") == "string"
        assert handler._serialize_value(42) == 42
        assert handler._serialize_value(3.14) == 3.14
        assert handler._serialize_value(True) is True

    def test_serialize_collections(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test serialization of collections."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        assert handler._serialize_value([1, 2, 3]) == [1, 2, 3]
        assert handler._serialize_value((1, 2)) == [1, 2]
        assert handler._serialize_value({"a": 1}) == {"a": 1}

    def test_serialize_nested_structures(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test serialization of nested structures."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        nested = {"level1": {"level2": {"level3": "value"}}}
        result = handler._serialize_value(nested)

        assert result["level1"]["level2"]["level3"] == "value"

    def test_serialize_handles_max_depth(
        self, mock_langgraph_module, mock_compiled_graph, config, session
    ):
        """Test that serialization handles max depth."""
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        # Create deeply nested structure
        deep = {"a": None}
        current = deep
        for i in range(100):
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should not raise, should truncate at max depth
        result = handler._serialize_value(deep)
        assert result is not None


# =============================================================================
# Integration with detect_adapter Tests
# =============================================================================


class TestDetectAdapterIntegration:
    """Tests for detect_adapter with LangGraph agents."""

    def test_detects_compiled_graph(self, mock_compiled_graph):
        """Test that detect_adapter identifies CompiledGraph."""
        from clyro.adapters.generic import detect_adapter

        result = detect_adapter(mock_compiled_graph)
        assert result == "langgraph"

    def test_detects_state_graph(self, mock_state_graph):
        """Test that detect_adapter identifies StateGraph."""
        from clyro.adapters.generic import detect_adapter

        result = detect_adapter(mock_state_graph)
        assert result == "langgraph"

    def test_returns_generic_for_regular_function(self):
        """Test that detect_adapter returns generic for regular functions."""
        from clyro.adapters.generic import detect_adapter

        def my_func():
            return "hello"

        result = detect_adapter(my_func)
        assert result == "generic"


# =============================================================================
# State Transition Event Tests
# =============================================================================


class TestStateTransitionEvent:
    """Tests for state transition event creation."""

    def test_create_state_transition_event(self):
        """Test creating a state transition event."""
        from clyro.trace import create_state_transition_event

        session_id = uuid4()
        event = create_state_transition_event(
            session_id=session_id,
            step_number=1,
            node_name="agent",
            input_data={"message": "hello"},
            output_data={"response": "world"},
            state_snapshot={"conversation": ["hello", "world"]},
            duration_ms=100,
            cumulative_cost=Decimal("0.01"),
            framework=Framework.LANGGRAPH,
            framework_version="0.2.5",
            agent_stage=AgentStage.THINK,
        )

        assert event.event_type == EventType.STATE_TRANSITION
        assert event.event_name == "agent"
        assert event.session_id == session_id
        assert event.step_number == 1
        assert event.input_data == {"message": "hello"}
        assert event.output_data == {"response": "world"}
        assert event.state_snapshot == {"conversation": ["hello", "world"]}
        assert event.duration_ms == 100
        assert event.framework == Framework.LANGGRAPH
        assert event.framework_version == "0.2.5"
        assert event.agent_stage == AgentStage.THINK

    def test_state_transition_event_computes_state_hash(self):
        """Test that state transition event computes state hash."""
        from clyro.trace import create_state_transition_event

        event = create_state_transition_event(
            session_id=uuid4(),
            step_number=1,
            node_name="test",
            state_snapshot={"key": "value"},
        )

        assert event.state_hash is not None
        assert len(event.state_hash) > 0


# =============================================================================
# Execution Control Enforcement Tests
# =============================================================================


class TestLangGraphStepLimitEnforcement:
    """Tests for real-time step limit enforcement during LangGraph execution.

    Unlike CrewAI (which has an event bus that swallows exceptions), LangGraph
    callbacks have raise_error=True, so StepLimitExceededError raised inside
    a callback propagates directly through LangChain and stops the agent.
    """

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
            framework=Framework.LANGGRAPH,
        )
        session.start()
        return session

    def test_next_step_increments_counter(self, mock_langgraph_module, mock_compiled_graph):
        """Test that _next_step returns sequential step numbers."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=100)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        # Handler starts at session.step_number + 1 to avoid step 0 collision
        assert handler._next_step() == 1
        assert handler._next_step() == 2
        assert handler._next_step() == 3
        assert handler._step_counter == 4

    def test_next_step_syncs_session_step_number(self, mock_langgraph_module, mock_compiled_graph):
        """Test that _next_step syncs the session's step number."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=100)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        assert session._step_number == 0

        handler._next_step()  # step 1
        assert session._step_number == 1

        handler._next_step()  # step 2
        assert session._step_number == 2

        handler._next_step()  # step 3
        assert session._step_number == 3

    def test_next_step_raises_step_limit_exceeded(self, mock_langgraph_module, mock_compiled_graph):
        """Test that _next_step raises StepLimitExceededError when limit is exceeded."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=3)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        # Steps 0, 1, 2 should be fine (counter goes 1, 2, 3)
        handler._next_step()
        handler._next_step()
        handler._next_step()

        # Step counter is now 3, next _next_step makes it 4 → 4 > 3 → raises
        with pytest.raises(StepLimitExceededError) as exc_info:
            handler._next_step()

        assert exc_info.value.limit == 3

    def test_step_limit_not_enforced_when_disabled(self, mock_langgraph_module, mock_compiled_graph):
        """Test that step limit is not enforced when enable_step_limit is False."""
        from clyro.adapters.langgraph import LangGraphAdapter

        config = ClyroConfig(
            controls=ExecutionControls(max_steps=2, enable_step_limit=False),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.LANGGRAPH,
        )
        session.start()

        adapter = LangGraphAdapter(mock_compiled_graph, config)
        handler = adapter.create_callback_handler(session)

        # Should be able to take many steps without raising
        for _ in range(50):
            handler._next_step()

        assert session._step_number == 50

    def test_step_limit_enforced_on_llm_end(self, mock_langgraph_module, mock_compiled_graph):
        """Test that step limit is enforced when on_llm_end creates an event."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=2)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        # Simulate 2 LLM calls (each on_llm_end calls _next_step)
        mock_response = MagicMock()
        mock_response.llm_output = {}
        mock_response.generations = []

        handler.on_llm_start({"name": "gpt-4"}, ["prompt 1"], run_id=uuid4())
        handler.on_llm_end(mock_response, run_id=uuid4())

        handler.on_llm_start({"name": "gpt-4"}, ["prompt 2"], run_id=uuid4())
        handler.on_llm_end(mock_response, run_id=uuid4())

        # Third LLM call should raise (step counter 3 > max_steps 2)
        handler.on_llm_start({"name": "gpt-4"}, ["prompt 3"], run_id=uuid4())
        with pytest.raises(StepLimitExceededError):
            handler.on_llm_end(mock_response, run_id=uuid4())

    def test_step_limit_enforced_on_tool_end(self, mock_langgraph_module, mock_compiled_graph):
        """Test that step limit is enforced when on_tool_end creates an event."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=2)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        # Simulate 2 tool calls
        rid1, rid2, rid3 = uuid4(), uuid4(), uuid4()
        handler.on_tool_start({"name": "search"}, "query1", run_id=rid1)
        handler.on_tool_end("result1", run_id=rid1)

        handler.on_tool_start({"name": "search"}, "query2", run_id=rid2)
        handler.on_tool_end("result2", run_id=rid2)

        # Third tool call should raise
        handler.on_tool_start({"name": "search"}, "query3", run_id=rid3)
        with pytest.raises(StepLimitExceededError):
            handler.on_tool_end("result3", run_id=rid3)

    def test_step_limit_enforced_on_chain_start(self, mock_langgraph_module, mock_compiled_graph):
        """Test that step limit is enforced when on_chain_start pre-assigns a step number."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=2)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        # Simulate 2 node executions (chain_start with parent_run_id → chain_end)
        parent = uuid4()
        rid1, rid2, rid3 = uuid4(), uuid4(), uuid4()

        handler.on_chain_start({}, {"input": "a"}, run_id=rid1, parent_run_id=parent,
                               metadata={"langgraph_node": "agent"})
        handler.on_chain_end({"output": "x"}, run_id=rid1, parent_run_id=parent)

        handler.on_chain_start({}, {"input": "b"}, run_id=rid2, parent_run_id=parent,
                               metadata={"langgraph_node": "tools"})
        handler.on_chain_end({"output": "y"}, run_id=rid2, parent_run_id=parent)

        # Third node execution should raise in on_chain_start (step pre-assignment)
        with pytest.raises(StepLimitExceededError):
            handler.on_chain_start({}, {"input": "c"}, run_id=rid3, parent_run_id=parent,
                                   metadata={"langgraph_node": "agent"})

    def test_events_have_correct_step_numbers(self, mock_langgraph_module, mock_compiled_graph):
        """Test that events created during execution have correct step numbers."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=100)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        mock_response = MagicMock()
        mock_response.llm_output = {}
        mock_response.generations = []

        # LLM call → step 0
        rid1 = uuid4()
        handler.on_llm_start({"name": "gpt-4"}, ["prompt"], run_id=rid1)
        handler.on_llm_end(mock_response, run_id=rid1)

        # Tool call → step 1
        rid2 = uuid4()
        handler.on_tool_start({"name": "search"}, "query", run_id=rid2)
        handler.on_tool_end("result", run_id=rid2)

        events = handler.drain_events()
        assert len(events) == 2
        # Handler starts at session.step_number + 1 to avoid step 0 collision
        assert events[0].step_number == 1  # LLM call
        assert events[1].step_number == 2  # Tool call

    def test_full_langgraph_step_limit_scenario(self, mock_langgraph_module, mock_compiled_graph):
        """Test realistic LangGraph scenario: LLM → node → tool → node cycle hits step limit."""
        from clyro.adapters.langgraph import LangGraphAdapter

        session = self._make_session_with_step_limit(max_steps=4)
        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        handler = adapter.create_callback_handler(session)

        parent = uuid4()
        mock_response = MagicMock()
        mock_response.llm_output = {}
        mock_response.generations = []

        # Step 0: LLM call
        rid_llm1 = uuid4()
        handler.on_llm_start({"name": "gpt-4"}, ["prompt 1"], run_id=rid_llm1)
        handler.on_llm_end(mock_response, run_id=rid_llm1)

        # Step 1: Agent node end (STATE_TRANSITION)
        rid_node1 = uuid4()
        handler.on_chain_start({}, {"input": "a"}, run_id=rid_node1, parent_run_id=parent,
                               metadata={"langgraph_node": "agent"})
        handler.on_chain_end({"output": "tool_call"}, run_id=rid_node1, parent_run_id=parent)

        # Step 2: Tool call
        rid_tool = uuid4()
        handler.on_tool_start({"name": "search"}, "query", run_id=rid_tool)
        handler.on_tool_end("result", run_id=rid_tool)

        # Step 3: Tools node end (STATE_TRANSITION)
        rid_node2 = uuid4()
        handler.on_chain_start({}, {"result": "r"}, run_id=rid_node2, parent_run_id=parent,
                               metadata={"langgraph_node": "tools"})
        handler.on_chain_end({"observation": "data"}, run_id=rid_node2, parent_run_id=parent)

        # Step 4: Second LLM call → step counter becomes 5, 5 > 4 → raises
        rid_llm2 = uuid4()
        handler.on_llm_start({"name": "gpt-4"}, ["prompt 2"], run_id=rid_llm2)
        with pytest.raises(StepLimitExceededError) as exc_info:
            handler.on_llm_end(mock_response, run_id=rid_llm2)

        assert exc_info.value.limit == 4
        assert session._step_number == 5


# =============================================================================
# Trace Hierarchy — FRD-002: LangGraph parent-child event wiring
# =============================================================================


class TestLangGraphTraceHierarchy:
    """Tests for FRD-002: LangGraph parent-child event wiring via run_id → event_id map."""

    def _make_handler(self):
        """Create a handler with mocked dependencies for hierarchy testing."""
        from clyro.adapters.langgraph import LangGraphAdapter, LangGraphCallbackHandler

        config = ClyroConfig(agent_name="test")
        session = Session(config)
        adapter = LangGraphAdapter.__new__(LangGraphAdapter)
        adapter._framework_version = "0.2.0"
        adapter._name = "test"
        handler = LangGraphCallbackHandler(session=session, adapter=adapter, config=config)
        return handler

    def test_on_chain_start_registers_early_event_id(self):
        """FRD-002: on_chain_start pre-generates event_id and registers in map."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        handler.on_chain_start({}, {"input": "x"}, run_id=graph_rid)
        handler.on_chain_start({}, {"input": "y"}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "agent"})
        assert str(node_rid) in handler._run_id_to_event_id
        assert str(node_rid) in handler._node_event_ids

    def test_on_chain_end_uses_pre_generated_event_id(self):
        """FRD-002: on_chain_end uses the same event_id pre-generated in on_chain_start."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "agent"})
        pre_eid = handler._node_event_ids[str(node_rid)]
        handler.on_chain_end({"out": 1}, run_id=node_rid, parent_run_id=graph_rid)
        events = handler.drain_events()
        assert len(events) == 1
        assert events[0].event_id == pre_eid

    def test_on_chain_end_resolves_parent_event_id(self):
        """FRD-002: on_chain_end resolves parent_event_id from parent_run_id map."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "agent"})
        handler.on_chain_end({"out": 1}, run_id=node_rid, parent_run_id=graph_rid)
        events = handler.drain_events()
        # graph_rid is not tracked as a node, so parent should be None
        # (top-level graph has no parent registration)
        assert events[0].parent_event_id is None

    def test_llm_end_resolves_parent_from_node(self):
        """FRD-002: on_llm_end resolves parent_event_id from the node's run_id."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        llm_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "agent"})
        node_eid = handler._run_id_to_event_id[str(node_rid)]

        mock_resp = MagicMock()
        mock_resp.llm_output = {}
        mock_resp.generations = []
        handler.on_llm_start({"name": "gpt-4"}, ["prompt"], run_id=llm_rid, parent_run_id=node_rid)
        handler.on_llm_end(mock_resp, run_id=llm_rid, parent_run_id=node_rid)

        events = handler.drain_events()
        llm_event = [e for e in events if e.event_type.value == "llm_call"][0]
        assert llm_event.parent_event_id == node_eid

    def test_llm_end_registers_its_own_event_id(self):
        """FRD-002: on_llm_end registers its event in the run_id map."""
        handler = self._make_handler()
        llm_rid = uuid4()
        mock_resp = MagicMock()
        mock_resp.llm_output = {}
        mock_resp.generations = []
        handler.on_llm_start({"name": "gpt-4"}, ["p"], run_id=llm_rid)
        handler.on_llm_end(mock_resp, run_id=llm_rid)
        assert str(llm_rid) in handler._run_id_to_event_id

    def test_tool_end_resolves_parent_from_node(self):
        """FRD-002: on_tool_end resolves parent_event_id from the node's run_id."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        tool_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "tools"})
        node_eid = handler._run_id_to_event_id[str(node_rid)]
        handler.on_tool_start({"name": "search"}, "query", run_id=tool_rid, parent_run_id=node_rid)
        handler.on_tool_end("result", run_id=tool_rid, parent_run_id=node_rid)
        events = handler.drain_events()
        tool_event = [e for e in events if e.event_type.value == "tool_call"][0]
        assert tool_event.parent_event_id == node_eid

    def test_retriever_end_resolves_parent_from_node(self):
        """FRD-002: on_retriever_end resolves parent_event_id from the node's run_id."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        ret_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "retriever"})
        node_eid = handler._run_id_to_event_id[str(node_rid)]
        handler.on_retriever_start({"name": "faiss"}, "search query", run_id=ret_rid, parent_run_id=node_rid)
        handler.on_retriever_end([], run_id=ret_rid, parent_run_id=node_rid)
        events = handler.drain_events()
        ret_event = [e for e in events if e.event_type.value == "retriever_call"][0]
        assert ret_event.parent_event_id == node_eid

    def test_on_chain_error_resolves_parent(self):
        """FRD-002: on_chain_error resolves parent and uses pre-generated event_id."""
        handler = self._make_handler()
        graph_rid = uuid4()
        node_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "agent"})
        pre_eid = handler._node_event_ids[str(node_rid)]
        handler.on_chain_error(ValueError("fail"), run_id=node_rid, parent_run_id=graph_rid)
        events = handler.drain_events()
        assert events[0].event_id == pre_eid

    def test_on_llm_error_resolves_parent(self):
        """FRD-002: on_llm_error resolves parent_event_id from parent_run_id."""
        handler = self._make_handler()
        node_rid = uuid4()
        llm_rid = uuid4()
        # Register a fake node in the map
        fake_node_eid = uuid4()
        handler._run_id_to_event_id[str(node_rid)] = fake_node_eid
        handler.on_llm_start({"name": "gpt-4"}, ["p"], run_id=llm_rid, parent_run_id=node_rid)
        event = handler.on_llm_error(RuntimeError("boom"), run_id=llm_rid, parent_run_id=node_rid)
        assert event.parent_event_id == fake_node_eid

    def test_on_tool_error_resolves_parent(self):
        """FRD-002: on_tool_error resolves parent_event_id from parent_run_id."""
        handler = self._make_handler()
        node_rid = uuid4()
        tool_rid = uuid4()
        fake_node_eid = uuid4()
        handler._run_id_to_event_id[str(node_rid)] = fake_node_eid
        handler.on_tool_start({"name": "t"}, "input", run_id=tool_rid, parent_run_id=node_rid)
        event = handler.on_tool_error(RuntimeError("fail"), run_id=tool_rid, parent_run_id=node_rid)
        assert event.parent_event_id == fake_node_eid

    def test_on_retriever_error_resolves_parent(self):
        """FRD-002: on_retriever_error resolves parent_event_id from parent_run_id."""
        handler = self._make_handler()
        node_rid = uuid4()
        ret_rid = uuid4()
        fake_node_eid = uuid4()
        handler._run_id_to_event_id[str(node_rid)] = fake_node_eid
        handler.on_retriever_start({"name": "r"}, "q", run_id=ret_rid, parent_run_id=node_rid)
        event = handler.on_retriever_error(RuntimeError("fail"), run_id=ret_rid, parent_run_id=node_rid)
        assert event.parent_event_id == fake_node_eid

    def test_unresolved_parent_returns_none(self):
        """FRD-002: Unknown parent_run_id gracefully degrades to parent_event_id=None."""
        handler = self._make_handler()
        llm_rid = uuid4()
        unknown_parent = uuid4()
        mock_resp = MagicMock()
        mock_resp.llm_output = {}
        mock_resp.generations = []
        handler.on_llm_start({"name": "gpt-4"}, ["p"], run_id=llm_rid, parent_run_id=unknown_parent)
        handler.on_llm_end(mock_resp, run_id=llm_rid, parent_run_id=unknown_parent)
        events = handler.drain_events()
        assert events[0].parent_event_id is None

    def test_map_eviction_at_capacity(self):
        """FRD-002: Map evicts oldest entry when exceeding 10,000 capacity."""
        handler = self._make_handler()
        handler._RUN_ID_MAP_MAX_SIZE = 3  # Low limit for testing
        handler._register_event(uuid4(), uuid4())
        handler._register_event(uuid4(), uuid4())
        first_key = next(iter(handler._run_id_to_event_id))
        handler._register_event(uuid4(), uuid4())  # at capacity
        handler._register_event(uuid4(), uuid4())  # triggers eviction
        assert first_key not in handler._run_id_to_event_id
        assert len(handler._run_id_to_event_id) == 3

    def test_drain_events_does_not_clear_maps(self):
        """FRD-002: drain_events() clears _events but NOT _run_id_to_event_id."""
        handler = self._make_handler()
        node_rid = uuid4()
        graph_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)
        handler.on_chain_start({}, {}, run_id=node_rid,
                               parent_run_id=graph_rid, metadata={"langgraph_node": "n"})
        handler.on_chain_end({"o": 1}, run_id=node_rid, parent_run_id=graph_rid)
        handler.drain_events()
        # Map should still contain the node registration
        assert str(node_rid) in handler._run_id_to_event_id


# =============================================================================
# Loop Detection Tests (PRD-010)
# =============================================================================


class TestLangGraphLoopDetection:
    """Tests for loop detection during LangGraph execution."""

    def _make_session_with_loop_detection(self, threshold: int = 3) -> Session:
        """Create a session with loop detection enabled."""
        config = ClyroConfig(
            capture_inputs=True,
            capture_outputs=True,
            capture_state=True,
            controls=ExecutionControls(
                max_steps=100,
                loop_detection_threshold=threshold,
            ),
        )
        session = Session(
            config=config,
            agent_id=uuid4(),
            org_id=uuid4(),
            framework=Framework.LANGGRAPH,
        )
        session.start()
        return session

    def _make_handler(self, session, mock_langgraph_module, mock_compiled_graph):
        from clyro.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(mock_compiled_graph, session.config)
        return adapter.create_callback_handler(session)

    def _simulate_node_execution(self, handler, node_name: str, outputs: dict, graph_rid: UUID):
        """Simulate a full node start → end cycle."""
        node_rid = uuid4()
        handler.on_chain_start(
            {}, {}, run_id=node_rid, parent_run_id=graph_rid,
            metadata={"langgraph_node": node_name},
        )
        handler.on_chain_end(outputs, run_id=node_rid, parent_run_id=graph_rid)

    def test_loop_detected_on_repeated_state(self, mock_langgraph_module, mock_compiled_graph):
        """Loop detection triggers when the same state repeats >= threshold times."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_langgraph_module, mock_compiled_graph)

        graph_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)

        identical_output = {"result": "same_value"}

        # First 2 repetitions should be fine
        self._simulate_node_execution(handler, "node_a", identical_output, graph_rid)
        self._simulate_node_execution(handler, "node_a", identical_output, graph_rid)

        # Third repetition should trigger loop detection
        with pytest.raises(LoopDetectedError):
            self._simulate_node_execution(handler, "node_a", identical_output, graph_rid)

    def test_no_loop_with_different_states_and_actions(self, mock_langgraph_module, mock_compiled_graph):
        """No loop detected when both states and action names vary."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_langgraph_module, mock_compiled_graph)

        graph_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)

        for i in range(10):
            self._simulate_node_execution(handler, f"node_{i}", {"result": f"value_{i}"}, graph_rid)

    def test_loop_detection_disabled(self, mock_langgraph_module, mock_compiled_graph):
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
            framework=Framework.LANGGRAPH,
        )
        session.start()
        handler = self._make_handler(session, mock_langgraph_module, mock_compiled_graph)

        graph_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)

        identical_output = {"result": "same_value"}
        for _ in range(10):
            self._simulate_node_execution(handler, "node_a", identical_output, graph_rid)

    def test_loop_detection_with_action_sequence(self, mock_langgraph_module, mock_compiled_graph):
        """Loop detection triggers on repeating action (node name) sequences."""
        session = self._make_session_with_loop_detection(threshold=3)
        handler = self._make_handler(session, mock_langgraph_module, mock_compiled_graph)

        graph_rid = uuid4()
        handler.on_chain_start({}, {}, run_id=graph_rid)

        # Repeat A→B pattern with identical outputs enough times to trigger
        identical_output = {"status": "pending"}
        with pytest.raises(LoopDetectedError):
            for _ in range(10):
                self._simulate_node_execution(handler, "node_a", identical_output, graph_rid)
                self._simulate_node_execution(handler, "node_b", identical_output, graph_rid)
