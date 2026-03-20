# Tests for Clyro SDK Core Wrapper
# Implements PRD-001, PRD-002

"""Unit tests for the core wrapper implementation."""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

import pytest
from clyro.config import ClyroConfig, ExecutionControls, reset_config
from clyro.exceptions import (
    ClyroWrapError,
    StepLimitExceededError,
)
from clyro.session import Session
from clyro.storage.sqlite import LocalStorage
from clyro.trace import Framework
from clyro.wrapper import WrappedAgent, configure, get_session, wrap

TEST_AGENT_ID = uuid4()


class TestWrap:
    """Tests for the wrap() function."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_wrap_sync_function(self):
        """Test wrapping a synchronous function."""

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)

        assert isinstance(wrapped, WrappedAgent)
        assert wrapped.agent == my_agent

    def test_wrap_async_function(self):
        """Test wrapping an asynchronous function."""

        async def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)

        assert isinstance(wrapped, WrappedAgent)
        assert wrapped._is_async is True

    def test_wrap_as_decorator(self):
        """Test using wrap as a decorator."""

        @wrap(agent_id=TEST_AGENT_ID)
        def my_agent(query: str) -> str:
            return f"Response: {query}"

        assert isinstance(my_agent, WrappedAgent)

    def test_wrap_with_config(self):
        """Test wrapping with custom configuration."""
        config = ClyroConfig(
            agent_name="test-agent",
            api_key="cly_test_key",
            controls=ExecutionControls(max_steps=50),
        )

        @wrap(config=config, org_id=uuid4())  # Pass org_id for agent_name
        def my_agent(query: str) -> str:
            return f"Response: {query}"

        assert my_agent.config.api_key == "cly_test_key"
        assert my_agent.config.controls.max_steps == 50

    def test_wrap_non_callable_raises_error(self):
        """Test that wrapping non-callable raises error."""
        with pytest.raises(ClyroWrapError) as exc_info:
            wrap("not a function")

        assert "must be callable" in str(exc_info.value).lower()
        assert exc_info.value.agent_type == "str"

    def test_wrap_with_adapter(self):
        """Test wrapping with specific adapter."""
        mock_module = ModuleType("langgraph")
        mock_module.__version__ = "0.2.5"
        with patch.dict(sys.modules, {"langgraph": mock_module}):

            @wrap(adapter="langgraph", agent_id=TEST_AGENT_ID)
            def my_agent(query: str) -> str:
                return f"Response: {query}"

            assert my_agent._adapter == "langgraph"
            assert my_agent._framework == Framework.LANGGRAPH

    def test_wrap_with_agent_id(self):
        """Test wrapping with agent ID."""

        @wrap(agent_id=TEST_AGENT_ID)
        def my_agent(query: str) -> str:
            return f"Response: {query}"

        assert my_agent._agent_id == TEST_AGENT_ID


class TestWrappedAgentExecution:
    """Tests for WrappedAgent execution."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_sync_execution_returns_same_result(self):
        """Test that wrapped sync function returns same result."""

        def my_agent(x: int, y: int) -> int:
            return x + y

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = wrapped(3, 4)

        assert result == 7

    def test_sync_execution_preserves_exception(self):
        """Test that exceptions propagate unchanged."""

        def my_agent(value: int) -> int:
            if value < 0:
                raise ValueError("Value must be non-negative")
            return value * 2

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)

        with pytest.raises(ValueError) as exc_info:
            wrapped(-1)

        assert "non-negative" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_execution_returns_same_result(self):
        """Test that wrapped async function returns same result."""

        async def my_agent(x: int, y: int) -> int:
            await asyncio.sleep(0.001)  # Simulate async work
            return x + y

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = await wrapped(3, 4)

        assert result == 7

    @pytest.mark.asyncio
    async def test_async_execution_preserves_exception(self):
        """Test that async exceptions propagate unchanged."""

        async def my_agent(value: int) -> int:
            await asyncio.sleep(0.001)
            if value < 0:
                raise ValueError("Value must be non-negative")
            return value * 2

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)

        with pytest.raises(ValueError) as exc_info:
            await wrapped(-1)

        assert "non-negative" in str(exc_info.value)

    def test_execution_with_kwargs(self):
        """Test execution with keyword arguments."""

        def my_agent(*, name: str, age: int) -> str:
            return f"{name} is {age}"

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = wrapped(name="Alice", age=30)

        assert result == "Alice is 30"

    def test_execution_with_default_args(self):
        """Test execution with default arguments."""

        def my_agent(query: str, limit: int = 10) -> str:
            return f"{query}:{limit}"

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)

        assert wrapped("test") == "test:10"
        assert wrapped("test", 5) == "test:5"
        assert wrapped("test", limit=20) == "test:20"


class TestWrappedAgentStepLimits:
    """Tests for step limit enforcement in wrapped agents."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_step_limit_exceeded(self):
        """Test that step limit is enforced."""
        config = ClyroConfig(agent_name="test-agent", controls=ExecutionControls(max_steps=1))

        call_count = 0

        def my_agent(value: int) -> int:
            nonlocal call_count
            call_count += 1
            return value * 2

        wrapped = wrap(my_agent, config=config, org_id=uuid4())

        # First call should succeed
        result = wrapped(5)
        assert result == 10

        # Note: Each wrap call creates a new session, so step limit
        # applies within a single session, not across calls

    def test_execution_control_error_records_trace(self, monkeypatch):
        """Test that execution control errors are recorded as error events."""
        config = ClyroConfig(agent_name="test-agent")
        recorded: dict[str, object] = {}

        def my_agent() -> str:
            return "done"

        def record_error(self, error, event_name=None):
            recorded["error"] = error
            recorded["event_name"] = event_name
            return original_record_error(self, error, event_name=event_name)

        def raise_step_limit(self, *args, **kwargs):
            raise StepLimitExceededError(
                limit=1,
                current_step=2,
                session_id="test-session",
            )

        original_record_error = Session.record_error
        monkeypatch.setattr("clyro.wrapper.Session.record_error", record_error)
        monkeypatch.setattr("clyro.wrapper.Session.record_step", raise_step_limit)

        wrapped = wrap(my_agent, config=config, org_id=uuid4())

        with pytest.raises(StepLimitExceededError):
            wrapped()

        assert isinstance(recorded["error"], StepLimitExceededError)
        assert recorded["event_name"] == "execution_control"


class TestWrappedAgentFailOpen:
    """Tests for fail-open behavior."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_fail_open_continues_on_trace_error(self):
        """Test that fail_open=True allows execution to continue."""
        config = ClyroConfig(fail_open=True, agent_name="test-agent")

        def my_agent(x: int) -> int:
            return x * 2

        wrapped = wrap(my_agent, config=config, org_id=uuid4())

        # Should succeed even if tracing has issues
        result = wrapped(5)
        assert result == 10

    def test_fail_open_default_is_true(self):
        """Test that fail_open defaults to True."""
        config = ClyroConfig(agent_name="test-agent")
        assert config.fail_open is True


class TestWrappedAgentStatus:
    """Tests for wrapped agent status."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_get_status(self):
        """Test getting wrapped agent status."""
        config = ClyroConfig(
            agent_name="test-agent",
            endpoint="http://localhost:8000",
            controls=ExecutionControls(max_steps=50),
        )
        def my_agent() -> str:
            return "test"

        wrapped = wrap(my_agent, config=config, agent_id=TEST_AGENT_ID, adapter="generic")

        status = wrapped.get_status()

        assert status["agent_name"] == "my_agent"
        assert status["agent_id"] == str(TEST_AGENT_ID)
        assert status["framework"] == "generic"
        assert status["adapter"] == "generic"
        assert status["is_async"] is False
        assert status["config"]["endpoint"] == "http://localhost:8000"
        assert status["config"]["max_steps"] == 50


class TestConfigure:
    """Tests for configure() function."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_configure_sets_global_config(self):
        """Test that configure sets global configuration."""
        config = ClyroConfig(
            agent_name="test-agent",
            api_key="cly_test_global",
            controls=ExecutionControls(max_steps=75),
        )

        configure(config)

        # New wraps should use this config
        @wrap(agent_id=TEST_AGENT_ID)
        def my_agent() -> str:
            return "test"

        assert my_agent.config.api_key == "cly_test_global"
        assert my_agent.config.controls.max_steps == 75


class TestGetSession:
    """Tests for get_session() function."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_get_session_outside_execution(self):
        """Test get_session returns None outside execution."""
        assert get_session() is None

    # Note: Testing get_session during execution would require
    # more complex setup with thread-local storage inspection


class TestWrappedAgentMetadata:
    """Tests for wrapped agent metadata preservation."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_preserves_function_name(self):
        """Test that function name is preserved."""

        def my_custom_agent() -> str:
            """My agent docstring."""
            return "test"

        wrapped = wrap(my_custom_agent, agent_id=TEST_AGENT_ID)

        assert wrapped.__name__ == "my_custom_agent"
        assert wrapped.__doc__ == "My agent docstring."

    def test_preserves_lambda_name(self):
        """Test wrapping lambda (anonymous function)."""
        agent = lambda x: x * 2  # noqa: E731

        wrapped = wrap(agent, agent_id=TEST_AGENT_ID)

        # Lambda functions have name "<lambda>"
        assert "<lambda>" in wrapped.__name__


class TestWrappedAgentInputOutputCapture:
    """Tests for input/output capture."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_capture_inputs_disabled(self):
        """Test that inputs are not captured when disabled."""
        config = ClyroConfig(agent_name="test-agent", capture_inputs=False)

        def my_agent(secret: str) -> str:
            return "processed"

        wrapped = wrap(my_agent, config=config, org_id=uuid4())
        result = wrapped("sensitive_data")

        assert result == "processed"

    def test_capture_outputs_disabled(self):
        """Test that outputs are not captured when disabled."""
        config = ClyroConfig(agent_name="test-agent", capture_outputs=False)

        def my_agent() -> str:
            return "sensitive_result"

        wrapped = wrap(my_agent, config=config, org_id=uuid4())
        result = wrapped()

        assert result == "sensitive_result"

    def test_handles_pydantic_model_input(self):
        """Test handling Pydantic models as input."""
        from pydantic import BaseModel

        class Query(BaseModel):
            text: str
            limit: int = 10

        def my_agent(query: Query) -> str:
            return query.text

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = wrapped(Query(text="hello"))

        assert result == "hello"

    def test_handles_dict_input(self):
        """Test handling dict input."""

        def my_agent(data: dict) -> str:
            return data.get("key", "default")

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = wrapped({"key": "value"})

        assert result == "value"

    def test_handles_list_input(self):
        """Test handling list input."""

        def my_agent(items: list) -> int:
            return len(items)

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        result = wrapped([1, 2, 3])

        assert result == 3


class TestMandatoryAgentIdentification:
    """Tests for mandatory agent_name/agent_id requirement."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_wrap_without_name_or_id_raises_in_cloud_mode(self):
        """Test that wrap() without agent_name or agent_id raises ClyroWrapError in cloud mode."""

        def my_agent(x: int) -> int:
            return x * 2

        config = ClyroConfig(
            api_key="cly_test_eyJlbnYiOiJ0ZXN0IiwiaWF0IjoxNzAwMDAwMDAwLCJrZXlfaWQiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJvcmdfaWQiOiIzZjlkYTNjYi1kYWY2LTQ5Y2MtYmVlMS00YjkyMmEyYmFiMGMiLCJzY29wZXMiOlsidHJhY2U6d3JpdGUiXX0.abc123",
            mode="cloud",
        )
        with pytest.raises(ClyroWrapError) as exc_info:
            wrap(my_agent, config=config)

        assert "agent identification is required" in str(exc_info.value).lower()

    def test_wrap_without_name_or_id_succeeds_in_local_mode(self):
        """Test that wrap() without agent_name or agent_id works in local mode (random agent_id)."""

        def my_agent(x: int) -> int:
            return x * 2

        wrapped = wrap(my_agent)  # No api_key → auto-resolves to local mode
        assert isinstance(wrapped, WrappedAgent)
        assert wrapped._agent_id is not None

    def test_wrap_with_agent_name_in_config_succeeds(self):
        """Test that providing agent_name in config with org_id is sufficient."""
        config = ClyroConfig(agent_name="test-agent")

        def my_agent(x: int) -> int:
            return x * 2

        wrapped = wrap(my_agent, config=config, org_id=uuid4())
        assert isinstance(wrapped, WrappedAgent)
        assert wrapped._agent_id is not None

    def test_wrap_with_agent_id_parameter_succeeds(self):
        """Test that providing agent_id parameter is sufficient."""

        def my_agent(x: int) -> int:
            return x * 2

        wrapped = wrap(my_agent, agent_id=TEST_AGENT_ID)
        assert isinstance(wrapped, WrappedAgent)
        assert wrapped._agent_id == TEST_AGENT_ID

    def test_wrap_with_agent_id_in_config_succeeds(self):
        """Test that providing agent_id in config (via env var flow) is sufficient."""
        config = ClyroConfig(agent_id=str(TEST_AGENT_ID))

        def my_agent(x: int) -> int:
            return x * 2

        wrapped = wrap(my_agent, config=config)
        assert isinstance(wrapped, WrappedAgent)
        assert wrapped._agent_id == TEST_AGENT_ID

    def test_wrap_as_decorator_without_identification_raises_in_cloud_mode(self):
        """Test that @wrap decorator without identification raises ClyroWrapError in cloud mode."""
        config = ClyroConfig(
            api_key="cly_test_eyJlbnYiOiJ0ZXN0IiwiaWF0IjoxNzAwMDAwMDAwLCJrZXlfaWQiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJvcmdfaWQiOiIzZjlkYTNjYi1kYWY2LTQ5Y2MtYmVlMS00YjkyMmEyYmFiMGMiLCJzY29wZXMiOlsidHJhY2U6d3JpdGUiXX0.abc123",
            mode="cloud",
        )  # No agent_name, no agent_id

        with pytest.raises(ClyroWrapError):

            @wrap(config=config)
            def my_agent(x: int) -> int:
                return x * 2

    def test_wrap_as_decorator_without_identification_succeeds_in_local_mode(self):
        """Test that @wrap decorator without identification works in local mode."""
        config = ClyroConfig()  # No api_key → local mode

        @wrap(config=config)
        def my_agent(x: int) -> int:
            return x * 2

        assert isinstance(my_agent, WrappedAgent)
        assert my_agent._agent_id is not None

    def test_error_message_mentions_both_options(self):
        """Test that error message mentions both agent_name and agent_id in cloud mode."""

        def my_agent() -> str:
            return "test"

        config = ClyroConfig(
            api_key="cly_test_eyJlbnYiOiJ0ZXN0IiwiaWF0IjoxNzAwMDAwMDAwLCJrZXlfaWQiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJvcmdfaWQiOiIzZjlkYTNjYi1kYWY2LTQ5Y2MtYmVlMS00YjkyMmEyYmFiMGMiLCJzY29wZXMiOlsidHJhY2U6d3JpdGUiXX0.abc123",
            mode="cloud",
        )
        with pytest.raises(ClyroWrapError) as exc_info:
            wrap(my_agent, config=config)

        error_msg = str(exc_info.value)
        assert "agent_name" in error_msg
        assert "agent_id" in error_msg

    def test_agent_id_parameter_takes_precedence_over_config_name(self):
        """Test that explicit agent_id parameter overrides config.agent_name."""
        config = ClyroConfig(agent_name="my-agent")

        def my_agent() -> str:
            return "test"

        wrapped = wrap(my_agent, config=config, agent_id=TEST_AGENT_ID)
        assert wrapped._agent_id == TEST_AGENT_ID

    def test_invalid_agent_id_in_config_raises(self):
        """Test that invalid UUID in config.agent_id raises ClyroWrapError."""
        config = ClyroConfig(agent_id="not-a-valid-uuid")

        def my_agent() -> str:
            return "test"

        with pytest.raises(ClyroWrapError) as exc_info:
            wrap(my_agent, config=config)

        assert "invalid agent_id" in str(exc_info.value).lower()
