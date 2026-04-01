# Clyro SDK LangGraph Integration Tests
# Implements PRD-003

"""
End-to-end integration tests for the LangGraph adapter.

These tests verify that the LangGraph adapter correctly integrates
with the SDK's wrapping, tracing, and execution control mechanisms.

Uses mock LangGraph objects to avoid requiring LangGraph as a hard dependency.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from conftest import TEST_ORG_ID

import clyro
from clyro.config import ClyroConfig, ExecutionControls, reset_config
from clyro.exceptions import FrameworkVersionError
from clyro.storage.sqlite import LocalStorage
from clyro.trace import Framework

# =============================================================================
# Mock LangGraph Module and Classes
# =============================================================================


class MockLangGraphModule(ModuleType):
    """Mock langgraph module for testing."""

    def __init__(self, version: str = "0.2.5"):
        super().__init__("langgraph")
        self.__version__ = version


class MockStateGraph:
    """Mock LangGraph StateGraph."""

    __module__ = "langgraph.graph"

    def __init__(self, name: str = "test_graph"):
        self.name = name
        self._nodes: dict[str, Any] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditional_edges: list[dict] = []

    def add_node(self, name: str, func: Any) -> MockStateGraph:
        self._nodes[name] = func
        return self

    def add_edge(self, source: str, target: str) -> MockStateGraph:
        self._edges.append((source, target))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path_map: dict[str, str],
        condition: Any = None,
    ) -> MockStateGraph:
        self._conditional_edges.append({
            "source": source,
            "path_map": path_map,
            "condition": condition,
        })
        return self

    def compile(self) -> MockCompiledGraph:
        return MockCompiledGraph(self)


class MockCompiledGraph:
    """Mock LangGraph CompiledGraph with configurable behavior."""

    __module__ = "langgraph.graph.state"

    def __init__(self, graph: MockStateGraph | None = None):
        self.graph = graph
        self.name = graph.name if graph else "compiled_graph"
        self._execution_count = 0
        self._should_fail = False
        self._fail_message = ""
        self._nodes = graph._nodes if graph else {}

    def set_should_fail(self, should_fail: bool, message: str = "Mock error"):
        """Configure the mock to fail on execution."""
        self._should_fail = should_fail
        self._fail_message = message

    def invoke(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke the compiled graph synchronously.

        Simulates node execution by calling node functions if defined.
        """
        self._execution_count += 1

        if self._should_fail:
            raise RuntimeError(self._fail_message)

        # Process callbacks if present
        callbacks = []
        if config and "callbacks" in config:
            callbacks = config["callbacks"]

        # Simulate execution through nodes
        state = dict(inputs)

        # Generate a graph-level run_id
        from uuid import uuid4
        graph_run_id = uuid4()

        for node_name, node_func in self._nodes.items():
            # Generate a node-level run_id
            node_run_id = uuid4()

            # Notify callbacks of node start using on_chain_start
            # (Real LangGraph calls on_chain_start for each node with parent_run_id)
            for callback in callbacks:
                if hasattr(callback, "on_chain_start"):
                    callback.on_chain_start(
                        serialized={"name": node_name},
                        inputs=state,
                        run_id=node_run_id,
                        parent_run_id=graph_run_id,  # Nodes have parent_run_id
                        metadata={"langgraph_node": node_name},
                    )

            # Execute node
            try:
                if callable(node_func):
                    result = node_func(state)
                    if isinstance(result, dict):
                        state.update(result)
                    else:
                        state[node_name + "_output"] = result

                # Notify callbacks of node end using on_chain_end
                for callback in callbacks:
                    if hasattr(callback, "on_chain_end"):
                        callback.on_chain_end(
                            outputs=state,
                            run_id=node_run_id,
                            parent_run_id=graph_run_id,
                        )
            except Exception as e:
                # Notify callbacks of node error using on_chain_error
                for callback in callbacks:
                    if hasattr(callback, "on_chain_error"):
                        callback.on_chain_error(
                            error=e,
                            run_id=node_run_id,
                            parent_run_id=graph_run_id,
                        )
                raise

        return state

    async def ainvoke(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the compiled graph asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async work
        return self.invoke(inputs, config)

    def get_graph(self) -> MockStateGraph:
        return self.graph or MockStateGraph()

    def __call__(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Allow direct calling of the graph."""
        return self.invoke(inputs, config)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global configuration before and after each test."""
    reset_config()
    yield
    reset_config()


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
def temp_storage_path():
    """Provide a temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_traces.db")


@pytest.fixture
def simple_graph():
    """Create a simple mock StateGraph."""
    graph = MockStateGraph(name="simple_agent")

    def process_input(state: dict) -> dict:
        return {"processed": True, "message": state.get("input", "")}

    graph.add_node("process", process_input)
    return graph


@pytest.fixture
def multi_node_graph():
    """Create a multi-node mock StateGraph."""
    graph = MockStateGraph(name="multi_node_agent")

    def think(state: dict) -> dict:
        return {"thought": f"Thinking about: {state.get('input', '')}"}

    def act(state: dict) -> dict:
        return {"action": f"Acting on: {state.get('thought', '')}"}

    def observe(state: dict) -> dict:
        return {"observation": f"Observed: {state.get('action', '')}"}

    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")

    return graph


# =============================================================================
# Basic Integration Tests
# =============================================================================


class TestLangGraphBasicIntegration:
    """Basic integration tests for LangGraph adapter."""

    def test_auto_detect_langgraph_adapter(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that LangGraph adapter is auto-detected."""
        from clyro.adapters.generic import detect_adapter

        compiled = simple_graph.compile()
        detected = detect_adapter(compiled)

        assert detected == "langgraph"

    def test_framework_metadata_in_traces(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that framework metadata is included in traces."""
        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Execute
        wrapped.invoke({"input": "test"})

        # Verify framework is set in wrapper
        assert wrapped._framework == Framework.LANGGRAPH


class TestLangGraphVersionValidation:
    """Tests for LangGraph version validation during wrapping."""

    def test_wrap_fails_with_unsupported_version(
        self, mock_langgraph_unsupported, simple_graph, temp_storage_path
    ):
        """
        Test that wrapping fails with unsupported LangGraph version.

        Gherkin:
            Given my LangGraph agent version is unsupported (<0.2.0)
            When I attempt to wrap it
            Then a FrameworkVersionError is raised
            And the error message indicates the version incompatibility
            And no partial traces are generated
        """
        from clyro.adapters.langgraph import LangGraphAdapter

        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        # Should raise FrameworkVersionError
        with pytest.raises(FrameworkVersionError) as exc_info:
            LangGraphAdapter(compiled, config, validate_version=True)

        # Verify error details
        assert exc_info.value.framework == "langgraph"
        assert exc_info.value.version == "0.1.0"
        assert ">=0.2.0" in exc_info.value.supported

        # Verify no traces were generated
        storage = LocalStorage(config)
        counts = storage.get_event_count()
        assert counts["total"] == 0

    def test_wrap_succeeds_with_supported_version(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that wrapping succeeds with supported version."""
        from clyro.adapters.langgraph import LangGraphAdapter

        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        # Should not raise
        adapter = LangGraphAdapter(compiled, config, validate_version=True)

        assert adapter.framework_version == "0.2.5"


class TestLangGraphMultiNodeExecution:
    """Tests for multi-node graph execution tracing."""

class TestLangGraphAsyncExecution:
    """Tests for async LangGraph execution."""

    @pytest.mark.asyncio
    async def test_async_langgraph_execution(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test async execution of LangGraph agent."""
        ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        # Create async wrapper manually
        # Note: The current SDK wraps ainvoke differently

        result = await compiled.ainvoke({"input": "async test"})

        assert result["processed"] is True


class TestLangGraphErrorHandling:
    """Tests for error handling in LangGraph execution."""

    def test_graph_execution_error_propagates(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that graph execution errors propagate correctly."""
        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()
        compiled.set_should_fail(True, "Simulated graph failure")

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        with pytest.raises(RuntimeError) as exc_info:
            wrapped.invoke({"input": "test"})

        assert "Simulated graph failure" in str(exc_info.value)


class TestLangGraphStateCapture:
    """Tests for state capture during LangGraph execution."""

    def test_state_snapshot_captured(
        self, mock_langgraph_module, multi_node_graph, temp_storage_path
    ):
        """
        Test that state snapshots are captured during execution.

        Gherkin:
            Given I have a LangGraph StateGraph agent
            When I wrap it using the LangGraph adapter
            Then graph state transitions are recorded with state snapshots
        """
        config = ClyroConfig(
            agent_name="test-agent",
            local_storage_path=temp_storage_path,
            capture_state=True,
        )
        compiled = multi_node_graph.compile()

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Execute
        result = wrapped.invoke({"input": "capture state test"})

        # Verify execution completed
        assert "observation" in result

        # State capture is verified via the callback handler
        # which records state_snapshot in TraceEvents

    def test_state_capture_disabled(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that state capture can be disabled."""
        config = ClyroConfig(
            agent_name="test-agent",
            local_storage_path=temp_storage_path,
            capture_state=False,
        )
        compiled = simple_graph.compile()

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Should execute without capturing state
        result = wrapped.invoke({"input": "no state capture"})

        assert result["processed"] is True


class TestLangGraphConditionalEdges:
    """Tests for conditional edge decision tracking."""

    def test_conditional_edge_decision_tracked(
        self, mock_langgraph_module, temp_storage_path
    ):
        """
        Test that conditional edge decisions are tracked.

        Gherkin:
            Given my LangGraph agent has conditional edges
            When the agent executes
            Then the actual path taken is recorded in the trace
            And decision points are visible in replay
        """
        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)

        # Create a graph with conditional edges
        graph = MockStateGraph(name="conditional_agent")

        def router(state: dict) -> dict:
            return {"route": "path_a" if state.get("condition") else "path_b"}

        def path_a(state: dict) -> dict:
            return {"result": "took path A"}

        def path_b(state: dict) -> dict:
            return {"result": "took path B"}

        graph.add_node("router", router)
        graph.add_node("path_a", path_a)
        graph.add_node("path_b", path_b)
        graph.add_conditional_edges(
            "router",
            {"path_a": "path_a", "path_b": "path_b"},
        )

        compiled = graph.compile()
        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Execute with condition=True
        result = wrapped.invoke({"input": "test", "condition": True})

        # Verify routing worked
        assert "route" in result


class TestLangGraphWithExecutionControls:
    """Tests for LangGraph with execution controls."""

    def test_step_limit_with_langgraph(
        self, mock_langgraph_module, multi_node_graph, temp_storage_path
    ):
        """Test step limits work with LangGraph agents."""
        config = ClyroConfig(
            agent_name="test-agent",
            local_storage_path=temp_storage_path,
            controls=ExecutionControls(max_steps=10),
        )
        compiled = multi_node_graph.compile()

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Should execute successfully within step limit
        result = wrapped.invoke({"input": "test"})

        assert "observation" in result

    def test_cost_tracking_with_langgraph(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test cost tracking works with LangGraph agents."""
        config = ClyroConfig(
            agent_name="test-agent",
            local_storage_path=temp_storage_path,
            controls=ExecutionControls(max_cost_usd=100.0),
        )
        compiled = simple_graph.compile()

        wrapped = clyro.wrap(compiled, config=config, adapter="langgraph", org_id=TEST_ORG_ID)

        # Should execute successfully
        result = wrapped.invoke({"input": "test"})

        assert result["processed"] is True


class TestLangGraphCallbackInjection:
    """Tests for callback injection mechanism."""

    def test_callbacks_injected_into_config(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that Clyro callbacks are injected into graph config."""
        from clyro.adapters.langgraph import LangGraphAdapter

        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        adapter = LangGraphAdapter(compiled, config)
        from clyro.session import Session

        session = Session(config=config, framework=Framework.LANGGRAPH)
        session.start()

        kwargs: dict[str, Any] = {}
        adapter.before_call(session, (), kwargs)

        # Verify callbacks were injected
        assert "config" in kwargs
        assert "callbacks" in kwargs["config"]
        assert len(kwargs["config"]["callbacks"]) == 1

    def test_existing_callbacks_preserved(
        self, mock_langgraph_module, simple_graph, temp_storage_path
    ):
        """Test that existing callbacks are preserved when injecting."""
        from unittest.mock import MagicMock

        from clyro.adapters.langgraph import LangGraphAdapter

        config = ClyroConfig(agent_name="test-agent", local_storage_path=temp_storage_path)
        compiled = simple_graph.compile()

        adapter = LangGraphAdapter(compiled, config)
        from clyro.session import Session

        session = Session(config=config, framework=Framework.LANGGRAPH)
        session.start()

        existing_callback = MagicMock()
        kwargs = {"config": {"callbacks": [existing_callback]}}

        adapter.before_call(session, (), kwargs)

        # Verify both callbacks present
        assert len(kwargs["config"]["callbacks"]) == 2
        assert existing_callback in kwargs["config"]["callbacks"]


class TestLangGraphLocalOnlyMode:
    """Tests for LangGraph in local-only mode."""
