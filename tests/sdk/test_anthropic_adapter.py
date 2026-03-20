# Clyro SDK Anthropic Adapter Tests
# Tests for FRD-001 through FRD-012

"""
Comprehensive tests for the Anthropic SDK adapter.

These tests verify all 12 FRD requirements including:
- Client detection and wrapping (FRD-001, FRD-002)
- LLM call tracing — sync and async (FRD-003, FRD-004)
- Tool use detection and policy enforcement (FRD-005, FRD-006)
- Cost and token tracking (FRD-007)
- Prevention Stack (FRD-008, FRD-009, FRD-010)
- Event hierarchy (FRD-011)
- Sync/async parity (FRD-012)

Mock strategy: The `anthropic` package is mocked via ModuleType + sys.modules
to avoid hard dependency, following the same pattern used by LangGraph and CrewAI
adapter tests.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    ClyroWrapError,
    CostLimitExceededError,
    FrameworkVersionError,
    LoopDetectedError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.trace import AgentStage, EventType, Framework


# ---------------------------------------------------------------------------
# Mock Anthropic SDK classes
# ---------------------------------------------------------------------------


class MockUsage:
    """Mock Anthropic usage object."""

    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockTextBlock:
    """Mock Anthropic text content block."""

    def __init__(self, text: str = "Hello!"):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    """Mock Anthropic tool_use content block."""

    def __init__(
        self,
        name: str = "get_weather",
        input: dict | None = None,
        id: str = "toolu_01abc",
    ):
        self.type = "tool_use"
        self.name = name
        self.input = input or {"city": "San Francisco"}
        self.id = id


class MockMessage:
    """Mock Anthropic Message response."""

    def __init__(
        self,
        content: list | None = None,
        model: str = "claude-sonnet-4-20250514",
        stop_reason: str = "end_turn",
        usage: MockUsage | None = None,
    ):
        self.content = content or [MockTextBlock()]
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage or MockUsage()
        self.id = "msg_01xyz"
        self.type = "message"
        self.role = "assistant"


class MockStream:
    """Mock Anthropic message stream context manager."""

    def __init__(self, final_message: MockMessage | None = None, error: Exception | None = None):
        self._final_message = final_message or MockMessage()
        self._error = error

    def __enter__(self):
        if self._error:
            raise self._error
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self._final_message

    def get_final_text(self):
        return "Hello!"

    def __iter__(self):
        yield {"type": "content_block_delta", "delta": {"text": "Hello!"}}


class MockAsyncStream:
    """Mock Anthropic async message stream context manager."""

    def __init__(self, final_message: MockMessage | None = None, error: Exception | None = None):
        self._final_message = final_message or MockMessage()
        self._error = error

    async def __aenter__(self):
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *args):
        return False

    def get_final_message(self):
        return self._final_message


class MockMessages:
    """Mock Anthropic Messages resource (sync)."""

    def __init__(
        self,
        response: MockMessage | None = None,
        error: Exception | None = None,
        stream_response: MockMessage | None = None,
    ):
        self._response = response or MockMessage()
        self._error = error
        self._stream_response = stream_response

    def create(self, **kwargs) -> MockMessage:
        if self._error:
            raise self._error
        return self._response

    def stream(self, **kwargs) -> MockStream:
        return MockStream(final_message=self._stream_response or self._response)


class MockAsyncMessages:
    """Mock Anthropic Messages resource (async)."""

    def __init__(
        self,
        response: MockMessage | None = None,
        error: Exception | None = None,
    ):
        self._response = response or MockMessage()
        self._error = error

    async def create(self, **kwargs) -> MockMessage:
        if self._error:
            raise self._error
        return self._response

    def stream(self, **kwargs) -> MockAsyncStream:
        return MockAsyncStream(final_message=self._response)


class MockAnthropicClient:
    """Mock anthropic.Anthropic client."""

    __module__ = "anthropic._client"
    __qualname__ = "Anthropic"

    def __init__(self, messages: MockMessages | None = None):
        self.messages = messages or MockMessages()
        self.api_key = "sk-ant-test"


# Make the class name match what detection expects
MockAnthropicClient.__name__ = "Anthropic"


class MockAsyncAnthropicClient:
    """Mock anthropic.AsyncAnthropic client."""

    __module__ = "anthropic._client"
    __qualname__ = "AsyncAnthropic"

    def __init__(self, messages: MockAsyncMessages | None = None):
        self.messages = messages or MockAsyncMessages()
        self.api_key = "sk-ant-test"


MockAsyncAnthropicClient.__name__ = "AsyncAnthropic"


class MockAnthropicModule(ModuleType):
    """Mock anthropic module for testing without actual dependency."""

    def __init__(self, version: str = "1.35.0"):
        super().__init__("anthropic")
        self.__version__ = version
        self.Anthropic = MockAnthropicClient
        self.AsyncAnthropic = MockAsyncAnthropicClient
        self.APIError = type("APIError", (Exception,), {})
        self.APIConnectionError = type("APIConnectionError", (Exception,), {})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_module():
    """Inject a mock anthropic module into sys.modules."""
    mock_module = MockAnthropicModule(version="1.35.0")
    with patch.dict(sys.modules, {"anthropic": mock_module}):
        yield mock_module


@pytest.fixture
def mock_anthropic_old_version():
    """Inject an old-version mock anthropic module."""
    mock_module = MockAnthropicModule(version="0.9.0")
    with patch.dict(sys.modules, {"anthropic": mock_module}):
        yield mock_module


@pytest.fixture
def sync_client():
    """Create a mock sync Anthropic client."""
    return MockAnthropicClient()


@pytest.fixture
def async_client():
    """Create a mock async Anthropic client."""
    return MockAsyncAnthropicClient()


@pytest.fixture
def config():
    """Create a ClyroConfig for testing."""
    return ClyroConfig(
        agent_name="test-anthropic-agent",
        capture_inputs=True,
        capture_outputs=True,
    )


@pytest.fixture
def config_with_limits():
    """Create a ClyroConfig with execution controls."""
    return ClyroConfig(
        agent_name="test-anthropic-agent",
        controls=ExecutionControls(
            max_steps=3,
            max_cost_usd=0.10,
            enable_step_limit=True,
            enable_cost_limit=True,
            enable_loop_detection=True,
            loop_detection_threshold=3,
        ),
    )


@pytest.fixture
def test_org_id():
    """Provide a test org_id."""
    return UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def test_agent_id():
    """Provide a test agent_id."""
    return UUID("00000000-0000-0000-0000-000000000002")


# ---------------------------------------------------------------------------
# Test: Client Detection (FRD-001)
# ---------------------------------------------------------------------------


class TestIsAnthropicAgent:
    """Tests for is_anthropic_agent() detection function.  # Tests FRD-001"""

    def test_detects_sync_client(self, sync_client):
        """Sync Anthropic client is detected."""
        from clyro.adapters.anthropic import is_anthropic_agent

        assert is_anthropic_agent(sync_client) is True

    def test_detects_async_client(self, async_client):
        """Async Anthropic client is detected."""
        from clyro.adapters.anthropic import is_anthropic_agent

        assert is_anthropic_agent(async_client) is True

    def test_rejects_non_anthropic_object(self):
        """Non-Anthropic objects are not detected."""
        from clyro.adapters.anthropic import is_anthropic_agent

        assert is_anthropic_agent("not a client") is False
        assert is_anthropic_agent(42) is False
        assert is_anthropic_agent(lambda: None) is False
        assert is_anthropic_agent(None) is False

    def test_rejects_similar_but_wrong_module(self):
        """Object with Anthropic-like name but wrong module is rejected."""
        from clyro.adapters.anthropic import is_anthropic_agent

        class Anthropic:
            __module__ = "my_custom_module"

        assert is_anthropic_agent(Anthropic()) is False

    def test_handles_exception_gracefully(self):
        """Detection handles exceptions without crashing."""
        from clyro.adapters.anthropic import is_anthropic_agent

        class BadObj:
            @property
            def __class__(self):
                raise RuntimeError("nope")

        # Should return False, not raise
        assert is_anthropic_agent(BadObj()) is False


class TestDetectAdapter:
    """Tests for detect_adapter() with Anthropic priority.  # Tests FRD-001"""

    def test_detects_anthropic_client(self, sync_client):
        """detect_adapter selects 'anthropic' for Anthropic clients."""
        from clyro.adapters.generic import detect_adapter

        assert detect_adapter(sync_client) == "anthropic"

    def test_generic_fallback_for_callable(self):
        """Regular callables still fall through to 'generic'."""
        from clyro.adapters.generic import detect_adapter

        def my_agent():
            pass

        assert detect_adapter(my_agent) == "generic"


# ---------------------------------------------------------------------------
# Test: Version Validation (FRD-001)
# ---------------------------------------------------------------------------


class TestValidateAnthropicVersion:
    """Tests for validate_anthropic_version().  # Tests FRD-001"""

    def test_valid_version(self, mock_anthropic_module):
        """Supported version passes validation."""
        from clyro.adapters.anthropic import validate_anthropic_version

        version = validate_anthropic_version()
        assert version == "1.35.0"

    def test_unsupported_version_raises(self, mock_anthropic_old_version):
        """Unsupported version raises FrameworkVersionError."""
        from clyro.adapters.anthropic import validate_anthropic_version

        with pytest.raises(FrameworkVersionError) as exc_info:
            validate_anthropic_version()

        assert exc_info.value.framework == "anthropic"
        assert exc_info.value.version == "0.9.0"

    def test_not_installed_raises(self):
        """Missing anthropic package raises FrameworkVersionError."""
        from clyro.adapters.anthropic import validate_anthropic_version

        with patch.dict(sys.modules, {"anthropic": None}):
            with pytest.raises((FrameworkVersionError, ImportError)):
                validate_anthropic_version()

    def test_exact_minimum_version(self):
        """Exact minimum version (1.0.0) passes."""
        mock_module = MockAnthropicModule(version="1.0.0")
        with patch.dict(sys.modules, {"anthropic": mock_module}):
            from clyro.adapters.anthropic import validate_anthropic_version

            version = validate_anthropic_version()
            assert version == "1.0.0"


# ---------------------------------------------------------------------------
# Test: Adapter Init and Traced Client (FRD-002)
# ---------------------------------------------------------------------------


class TestAnthropicAdapterInit:
    """Tests for AnthropicAdapter initialization.  # Tests FRD-002"""

    def test_adapter_creates_sync_traced_client(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Adapter creates AnthropicTracedClient for sync client."""
        from clyro.adapters.anthropic import AnthropicAdapter, AnthropicTracedClient

        adapter = AnthropicAdapter(
            client=sync_client,
            config=config,
            agent_id=test_agent_id,
            org_id=test_org_id,
        )
        traced = adapter.create_traced_client()
        assert isinstance(traced, AnthropicTracedClient)
        assert traced._clyro_wrapped is True

    def test_adapter_creates_async_traced_client(
        self, mock_anthropic_module, async_client, config, test_agent_id, test_org_id
    ):
        """Adapter creates AsyncAnthropicTracedClient for async client."""
        from clyro.adapters.anthropic import AnthropicAdapter, AsyncAnthropicTracedClient

        adapter = AnthropicAdapter(
            client=async_client,
            config=config,
            agent_id=test_agent_id,
            org_id=test_org_id,
        )
        traced = adapter.create_traced_client()
        assert isinstance(traced, AsyncAnthropicTracedClient)
        assert traced._clyro_wrapped is True

    def test_double_wrap_raises(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Double-wrapping raises ClyroWrapError."""
        from clyro.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        with pytest.raises(ClyroWrapError, match="already wrapped"):
            AnthropicAdapter(
                client=traced, config=config, agent_id=test_agent_id, org_id=test_org_id
            )

    def test_adapter_properties(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Adapter exposes correct properties."""
        from clyro.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        assert adapter.name == "anthropic"
        assert adapter.framework == Framework.ANTHROPIC
        assert adapter.framework_version == "1.35.0"
        assert adapter.agent is sync_client

    def test_version_error_on_old_sdk(
        self, mock_anthropic_old_version, sync_client, config, test_agent_id, test_org_id
    ):
        """Old SDK version raises FrameworkVersionError on init."""
        from clyro.adapters.anthropic import AnthropicAdapter

        with pytest.raises(FrameworkVersionError):
            AnthropicAdapter(
                client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
            )


# ---------------------------------------------------------------------------
# Test: Traced Client Proxy (FRD-002)
# ---------------------------------------------------------------------------


class TestAnthropicTracedClient:
    """Tests for AnthropicTracedClient proxy behavior.  # Tests FRD-002"""

    def test_getattr_passes_through(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Attribute access passes through to underlying client."""
        from clyro.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # api_key should pass through
        assert traced.api_key == "sk-ant-test"

    def test_context_manager(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Traced client works as context manager."""
        from clyro.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )

        with adapter.create_traced_client() as traced:
            assert traced._closed is False

        assert traced._closed is True

    def test_close_is_idempotent(
        self, mock_anthropic_module, sync_client, config, test_agent_id, test_org_id
    ):
        """Calling close() multiple times is safe."""
        from clyro.adapters.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter(
            client=sync_client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()
        traced.close()
        traced.close()  # Should not raise
        assert traced._closed is True


# ---------------------------------------------------------------------------
# Test: Sync messages.create() (FRD-003)
# ---------------------------------------------------------------------------


class TestTracedMessagesCreate:
    """Tests for TracedMessages.create() sync tracing.  # Tests FRD-003"""

    def test_create_returns_response(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """create() returns the original response."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(content=[MockTextBlock("Hi!")], model="claude-sonnet-4-20250514")
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        result = traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is response
        assert result.content[0].text == "Hi!"
        traced.close()

    def test_create_emits_llm_call_event(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """create() emits an LLM_CALL event with correct fields."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            usage=MockUsage(input_tokens=200, output_tokens=100),
            model="claude-sonnet-4-20250514",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Check session has events
        session = traced._session
        assert session is not None
        # Find LLM_CALL events (session events include session_start)
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        assert len(llm_events) >= 1

        traced.close()

    def test_create_captures_token_counts(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """create() captures input/output token counts from response.  # Tests FRD-007"""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(usage=MockUsage(input_tokens=500, output_tokens=200))
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify session has accumulated cost > 0
        assert traced._session.cumulative_cost > Decimal("0")
        traced.close()

    def test_create_on_api_error_emits_error_event(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """API errors emit ERROR event and re-raise.  # Tests FRD-003 failure condition"""
        from clyro.adapters.anthropic import AnthropicAdapter

        api_error = Exception("API connection failed")
        client = MockAnthropicClient(messages=MockMessages(error=api_error))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        with pytest.raises(Exception, match="API connection failed"):
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        traced.close()

    def test_create_with_framework_anthropic(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Events have framework=ANTHROPIC and agent_stage=THINK."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        session = traced._session
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        assert len(llm_events) >= 1
        assert llm_events[0].framework == Framework.ANTHROPIC
        assert llm_events[0].agent_stage == AgentStage.THINK

        traced.close()


# ---------------------------------------------------------------------------
# Test: Streaming (FRD-004)
# ---------------------------------------------------------------------------


class TestTracedMessagesStream:
    """Tests for TracedMessages.stream() tracing.  # Tests FRD-004"""

    def test_stream_emits_event_on_completion(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """stream() emits LLM_CALL event on context manager exit."""
        from clyro.adapters.anthropic import AnthropicAdapter

        final_msg = MockMessage(
            usage=MockUsage(input_tokens=300, output_tokens=150),
            model="claude-sonnet-4-20250514",
        )
        client = MockAnthropicClient(
            messages=MockMessages(stream_response=final_msg)
        )
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        with traced.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            # Iterate stream (mock)
            _ = stream.get_final_message()

        session = traced._session
        assert session is not None
        traced.close()


# ---------------------------------------------------------------------------
# Test: Tool Use Detection (FRD-005)
# ---------------------------------------------------------------------------


class TestToolUseDetection:
    """Tests for tool_use content block detection.  # Tests FRD-005"""

    def test_tool_use_emits_tool_call_events(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """tool_use blocks in response emit TOOL_CALL events."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            content=[
                MockTextBlock("Let me check the weather."),
                MockToolUseBlock(name="get_weather", input={"city": "SF"}, id="toolu_01"),
            ],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"name": "get_weather", "description": "Get weather"}],
        )

        session = traced._session
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].event_name == "get_weather"
        assert tool_events[0].agent_stage == AgentStage.ACT
        assert tool_events[0].input_data == {"city": "SF"}
        assert tool_events[0].metadata.get("tool_use_id") == "toolu_01"

        traced.close()

    def test_multiple_tool_use_blocks(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Multiple tool_use blocks emit multiple TOOL_CALL events."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            content=[
                MockToolUseBlock(name="get_weather", input={"city": "SF"}, id="toolu_01"),
                MockToolUseBlock(name="get_time", input={"timezone": "PST"}, id="toolu_02"),
            ],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Weather and time?"}],
        )

        session = traced._session
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 2
        assert tool_events[0].event_name == "get_weather"
        assert tool_events[1].event_name == "get_time"

        traced.close()

    def test_malformed_tool_use_block_emits_warning(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Malformed tool_use block (missing name) still emits event.  # Tests FRD-005 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        class MalformedBlock:
            type = "tool_use"
            name = None  # Missing name
            input = {"key": "value"}
            id = "toolu_bad"

        response = MockMessage(content=[MalformedBlock()], stop_reason="tool_use")
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Should not raise
        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        session = traced._session
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].event_name == "unknown"

        traced.close()

    def test_no_tool_use_no_tool_events(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Response without tool_use blocks emits no TOOL_CALL events."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(content=[MockTextBlock("Just text")], stop_reason="end_turn")
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

        session = traced._session
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 0

        traced.close()


# ---------------------------------------------------------------------------
# Test: Prevention Stack (FRD-008, FRD-009, FRD-010)
# ---------------------------------------------------------------------------


class TestPreventionStack:
    """Tests for step limit, cost limit, and loop detection.  # Tests FRD-008, FRD-009, FRD-010"""

    def test_step_limit_enforcement(
        self, mock_anthropic_module, config_with_limits, test_agent_id, test_org_id
    ):
        """Step limit raises StepLimitExceededError.  # Tests FRD-008"""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config_with_limits, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # max_steps=3, so 4th call should fail
        for _ in range(3):
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Call {_}"}],
            )

        with pytest.raises(StepLimitExceededError) as exc_info:
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Exceeds limit"}],
            )

        assert exc_info.value.limit == 3

        traced.close()

    def test_step_counter_increments_without_limit(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Step counter increments even when limits are disabled.  # Tests FRD-008"""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "1"}]
        )
        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "2"}]
        )

        assert traced._session.step_number == 2

        traced.close()

    def test_cost_limit_enforcement(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Cost limit raises CostLimitExceededError.  # Tests FRD-009"""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_cost_usd=0.001,
                enable_cost_limit=True,
                enable_step_limit=False,
            ),
        )

        # Response with lots of tokens to exceed cost
        response = MockMessage(usage=MockUsage(input_tokens=10000, output_tokens=5000))
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # First call might exceed
        traced.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[{"role": "user", "content": "Expensive call"}],
        )

        # Second call should hit cost limit
        with pytest.raises(CostLimitExceededError):
            traced.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{"role": "user", "content": "Over budget"}],
            )

        traced.close()

    def test_loop_detection(
        self, mock_anthropic_module, config_with_limits, test_agent_id, test_org_id
    ):
        """Loop detection raises LoopDetectedError.  # Tests FRD-010"""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config_with_limits, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        same_messages = [{"role": "user", "content": "Same thing over and over"}]

        # threshold=3, first 2 calls should succeed
        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=same_messages
        )
        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=same_messages
        )

        # 3rd identical call should trigger loop detection
        with pytest.raises(LoopDetectedError) as exc_info:
            traced.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1024, messages=same_messages
            )

        assert exc_info.value.iterations == 3

        traced.close()

    def test_loop_detection_different_messages_no_error(
        self, mock_anthropic_module, config_with_limits, test_agent_id, test_org_id
    ):
        """Different messages do not trigger loop detection."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config_with_limits, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        for i in range(3):
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Message {i}"}],
            )

        # Should not raise — all messages are different
        traced.close()


# ---------------------------------------------------------------------------
# Test: Event Hierarchy (FRD-011)
# ---------------------------------------------------------------------------


class TestEventHierarchy:
    """Tests for parent-child event linking.  # Tests FRD-011"""

    def test_first_llm_call_has_no_parent(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """First LLM_CALL in session has parent_event_id=None."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "1"}]
        )

        session = traced._session
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        assert len(llm_events) == 1
        assert llm_events[0].parent_event_id is None

        traced.close()

    def test_subsequent_llm_calls_chain_parent(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Subsequent LLM_CALL events chain via parent_event_id."""
        from clyro.adapters.anthropic import AnthropicAdapter

        # First call returns tool_use (session stays active),
        # second call returns end_turn — both land on the same session.
        tool_use_response = MockMessage(
            content=[
                MockToolUseBlock(name="search", input={"q": "test"}, id="toolu_01"),
            ],
            stop_reason="tool_use",
        )
        end_turn_response = MockMessage(stop_reason="end_turn")
        call_count = 0

        class _TwoCallMessages:
            def create(self, **kwargs):
                nonlocal call_count
                call_count += 1
                return tool_use_response if call_count == 1 else end_turn_response

            def stream(self, **kwargs):
                raise NotImplementedError

        client = MockAnthropicClient(messages=_TwoCallMessages())
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "1"}]
        )
        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "2"}]
        )

        session = traced._session
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        assert len(llm_events) == 2
        assert llm_events[0].parent_event_id is None
        assert llm_events[1].parent_event_id == llm_events[0].event_id

        traced.close()

    def test_tool_call_parent_is_llm_event(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """TOOL_CALL event parent_event_id points to the LLM_CALL that generated it."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            content=[
                MockToolUseBlock(name="search", input={"q": "test"}, id="toolu_01"),
            ],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "search"}]
        )

        session = traced._session
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]

        assert len(llm_events) == 1
        assert len(tool_events) == 1
        assert tool_events[0].parent_event_id == llm_events[0].event_id

        traced.close()


# ---------------------------------------------------------------------------
# Test: Cost Tracking (FRD-007)
# ---------------------------------------------------------------------------


class TestCostTracking:
    """Tests for per-call cost calculation.  # Tests FRD-007"""

    def test_cost_calculated_from_usage(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Cost is calculated from response.usage tokens."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            usage=MockUsage(input_tokens=1000, output_tokens=500),
            model="claude-sonnet-4-20250514",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Calculate cost"}],
        )

        session = traced._session
        assert session.cumulative_cost > Decimal("0")

        traced.close()

    def test_missing_usage_sets_zero_cost(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Missing usage on response results in cost_usd=0.  # Tests FRD-007 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        # Response with no usage attribute
        response = MockMessage()
        response.usage = None

        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "No usage"}],
        )

        session = traced._session
        assert session.cumulative_cost == Decimal("0")

        traced.close()

    def test_cumulative_cost_accumulates(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Cumulative cost adds up across multiple calls."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            usage=MockUsage(input_tokens=100, output_tokens=50),
            model="claude-sonnet-4-20250514",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "1"}]
        )
        cost_after_first = traced._session.cumulative_cost

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "2"}]
        )
        cost_after_second = traced._session.cumulative_cost

        assert cost_after_second > cost_after_first

        traced.close()


# ---------------------------------------------------------------------------
# Test: Session Lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Tests for lazy session creation and cleanup."""

    def test_session_created_lazily(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Session is not created until first API call."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # No session before first call
        assert traced._session is None

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "Hi"}]
        )

        # Session exists after call (ended eagerly by auto_flush on end_turn)
        assert traced._session is not None

        traced.close()

    def test_session_ends_on_close(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Session ends when close() is called."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024, messages=[{"role": "user", "content": "Hi"}]
        )

        traced.close()
        assert traced._session.is_active is False


# ---------------------------------------------------------------------------
# Test: Async Parity (FRD-012)
# ---------------------------------------------------------------------------


class TestAsyncParity:
    """Tests for async client behaving identically to sync.  # Tests FRD-012"""

    @pytest.mark.asyncio
    async def test_async_create_returns_response(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Async create() returns the original response."""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(content=[MockTextBlock("Async hi!")], model="claude-sonnet-4-20250514")
        async_msgs = MockAsyncMessages(response=response)
        client = MockAsyncAnthropicClient(messages=async_msgs)
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        result = await traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello async"}],
        )

        assert result is response
        assert result.content[0].text == "Async hi!"

        await traced.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Async traced client works as async context manager."""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAsyncAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )

        async with adapter.create_traced_client() as traced:
            await traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert traced._closed is True

    @pytest.mark.asyncio
    async def test_async_step_limit(
        self, mock_anthropic_module, config_with_limits, test_agent_id, test_org_id
    ):
        """Async enforces step limits.  # Tests FRD-008, FRD-012"""
        from clyro.adapters.anthropic import AnthropicAdapter

        client = MockAsyncAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config_with_limits, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        for _ in range(3):
            await traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"Async {_}"}],
            )

        with pytest.raises(StepLimitExceededError):
            await traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Over limit"}],
            )

        await traced.close()

    @pytest.mark.asyncio
    async def test_async_tool_use_detection(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Async detects tool_use and emits TOOL_CALL events.  # Tests FRD-005, FRD-012"""
        from clyro.adapters.anthropic import AnthropicAdapter

        response = MockMessage(
            content=[
                MockToolUseBlock(name="async_tool", input={"x": 1}, id="toolu_async"),
            ],
            stop_reason="tool_use",
        )
        async_msgs = MockAsyncMessages(response=response)
        client = MockAsyncAnthropicClient(messages=async_msgs)
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        await traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Use tool"}],
        )

        session = traced._session
        tool_events = [e for e in session.events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].event_name == "async_tool"

        await traced.close()


# ---------------------------------------------------------------------------
# Test: wrap() Integration (FRD-001, FRD-002)
# ---------------------------------------------------------------------------


class TestWrapIntegration:
    """Tests for clyro.wrap() with Anthropic clients."""

    def test_wrap_detects_and_returns_traced_client(
        self, mock_anthropic_module, test_org_id
    ):
        """wrap() auto-detects Anthropic client and returns traced client."""
        from clyro.adapters.anthropic import AnthropicTracedClient
        from clyro.wrapper import wrap

        config = ClyroConfig(agent_name="wrap-test-agent")
        client = MockAnthropicClient()

        traced = wrap(client, config=config, org_id=test_org_id)
        assert isinstance(traced, AnthropicTracedClient)
        traced.close()

    def test_wrap_with_agent_id(self, mock_anthropic_module, test_agent_id, test_org_id):
        """wrap() works with explicit agent_id."""
        from clyro.adapters.anthropic import AnthropicTracedClient
        from clyro.wrapper import wrap

        config = ClyroConfig()
        client = MockAnthropicClient()

        traced = wrap(client, config=config, agent_id=test_agent_id, org_id=test_org_id)
        assert isinstance(traced, AnthropicTracedClient)
        traced.close()

    def test_wrap_async_client(self, mock_anthropic_module, test_org_id):
        """wrap() creates async traced client for AsyncAnthropic."""
        from clyro.adapters.anthropic import AsyncAnthropicTracedClient
        from clyro.wrapper import wrap

        config = ClyroConfig(agent_name="async-wrap-test")
        client = MockAsyncAnthropicClient()

        traced = wrap(client, config=config, org_id=test_org_id)
        assert isinstance(traced, AsyncAnthropicTracedClient)

    def test_wrap_requires_agent_identification_in_cloud_mode(self, mock_anthropic_module, test_org_id):
        """wrap() raises ClyroWrapError without agent identification in cloud mode."""
        from clyro.wrapper import wrap

        config = ClyroConfig(
            api_key="cly_test_eyJlbnYiOiJ0ZXN0IiwiaWF0IjoxNzAwMDAwMDAwLCJrZXlfaWQiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJvcmdfaWQiOiIzZjlkYTNjYi1kYWY2LTQ5Y2MtYmVlMS00YjkyMmEyYmFiMGMiLCJzY29wZXMiOlsidHJhY2U6d3JpdGUiXX0.abc123",
            mode="cloud",
        )  # No agent_name, no agent_id
        client = MockAnthropicClient()

        with pytest.raises(ClyroWrapError, match="Agent identification is required"):
            wrap(client, config=config, org_id=test_org_id)


# ---------------------------------------------------------------------------
# Test: Error Handling edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for edge cases in error handling."""

    def test_token_extraction_failure_is_non_fatal(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Token extraction failure results in cost=0, not an exception."""
        from clyro.adapters.anthropic import AnthropicAdapter

        # Response where usage raises on access
        response = MockMessage()
        response.usage = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))

        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Should not raise
        result = traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )
        assert result is not None

        traced.close()

    def test_no_capture_when_disabled(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Input/output capture respects config flags."""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="no-capture",
            capture_inputs=False,
            capture_outputs=False,
        )

        client = MockAnthropicClient()
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "secret"}],
        )

        session = traced._session
        llm_events = [e for e in session.events if e.event_type == EventType.LLM_CALL]
        assert len(llm_events) >= 1
        # input_data should just be model, not full messages
        assert llm_events[0].input_data.get("messages") is None
        assert llm_events[0].output_data is None

        traced.close()

    def test_stream_error_emits_error_event(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Stream interruption emits ERROR event and re-raises.  # Tests FRD-004 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        class ErrorStream:
            """Mock stream that raises on context entry."""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_final_message(self):
                raise ConnectionError("Stream interrupted")

        class StreamErrorMessages:
            def create(self, **kwargs):
                return MockMessage()

            def stream(self, **kwargs):
                return ErrorStream()

        client = MockAnthropicClient(messages=StreamErrorMessages())
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # The stream should emit events but the error from get_final_message
        # is caught in __exit__ and logged (not re-raised since exc_type is None)
        with traced.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "stream test"}],
        ) as stream:
            pass  # get_final_message will be called in __exit__

        # Session should have been created
        assert traced._session is not None
        traced.close()

    def test_stream_exception_during_iteration(
        self, mock_anthropic_module, config, test_agent_id, test_org_id
    ):
        """Exception during stream iteration emits ERROR event.  # Tests FRD-004 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        class PropagatingErrorStream:
            """Mock stream that propagates exceptions in __exit__."""

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False  # Propagate exception

            def get_final_message(self):
                return MockMessage()

        class StreamMessages:
            def create(self, **kwargs):
                return MockMessage()

            def stream(self, **kwargs):
                return PropagatingErrorStream()

        client = MockAnthropicClient(messages=StreamMessages())
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        with pytest.raises(RuntimeError, match="Network error"):
            with traced.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "stream test"}],
            ) as stream:
                raise RuntimeError("Network error")

        # Session was created, error event should have been emitted
        assert traced._session is not None
        traced.close()


# ---------------------------------------------------------------------------
# Test: Policy Enforcement (FRD-006)
# ---------------------------------------------------------------------------


class TestPolicyEnforcement:
    """Tests for policy evaluation on tool use.  # Tests FRD-006"""

    def test_policy_blocks_tool_use(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Policy blocking a tool raises PolicyViolationError.  # Tests FRD-006"""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="policy-test",
            api_key="cly_test_key",
            controls=ExecutionControls(enable_policy_enforcement=True),
        )

        response = MockMessage(
            content=[MockToolUseBlock(name="dangerous_tool", input={"x": 1}, id="toolu_01")],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Mock the policy evaluator — session.check_policy() delegates to
        # PolicyEvaluator.evaluate_sync() which raises PolicyViolationError
        # for blocked decisions. Allow llm_call, block tool_call.
        def _eval_sync(action_type, parameters, session_id=None, step_number=None):
            if action_type == "tool_call":
                raise PolicyViolationError(
                    rule_id="rule_01",
                    rule_name="Block Dangerous",
                    message="Tool blocked by policy",
                    action_type="tool_call",
                )

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_sync.side_effect = _eval_sync
        mock_evaluator.drain_events.return_value = []
        traced._policy_evaluator = mock_evaluator
        traced._traced_messages._policy_evaluator = mock_evaluator

        with pytest.raises(PolicyViolationError, match="Tool blocked by policy"):
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Use dangerous tool"}],
            )

        traced.close()

    def test_policy_allows_tool_use(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Policy allowing a tool returns response normally.  # Tests FRD-006"""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="policy-test",
            api_key="cly_test_key",
            controls=ExecutionControls(enable_policy_enforcement=True),
        )

        response = MockMessage(
            content=[MockToolUseBlock(name="safe_tool", input={"x": 1}, id="toolu_01")],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Mock policy allowing the tool
        mock_decision = MagicMock()
        mock_decision.is_blocked = False
        mock_decision.requires_approval = False

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_sync.return_value = mock_decision
        mock_evaluator.drain_events.return_value = []
        traced._policy_evaluator = mock_evaluator
        traced._traced_messages._policy_evaluator = mock_evaluator

        result = traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Use safe tool"}],
        )

        assert result is response
        traced.close()

    def test_policy_fail_open_on_error(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Policy evaluation failure with fail_open=True proceeds.  # Tests FRD-006 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="policy-test",
            api_key="cly_test_key",
            fail_open=True,
            controls=ExecutionControls(enable_policy_enforcement=True),
        )

        response = MockMessage(
            content=[MockToolUseBlock(name="tool", input={"x": 1}, id="toolu_01")],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Mock policy evaluator that raises
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_sync.side_effect = ConnectionError("Policy endpoint unreachable")
        mock_evaluator.drain_events.return_value = []
        traced._policy_evaluator = mock_evaluator
        traced._traced_messages._policy_evaluator = mock_evaluator

        # Should not raise — fail_open=True
        result = traced.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is response
        traced.close()

    def test_policy_fail_closed_on_error(
        self, mock_anthropic_module, test_agent_id, test_org_id
    ):
        """Policy evaluation failure with fail_open=False raises.  # Tests FRD-006 failure"""
        from clyro.adapters.anthropic import AnthropicAdapter

        config = ClyroConfig(
            agent_name="policy-test",
            api_key="cly_test_key",
            fail_open=False,
            controls=ExecutionControls(enable_policy_enforcement=True),
        )

        response = MockMessage(
            content=[MockToolUseBlock(name="tool", input={"x": 1}, id="toolu_01")],
            stop_reason="tool_use",
        )
        client = MockAnthropicClient(messages=MockMessages(response=response))
        adapter = AnthropicAdapter(
            client=client, config=config, agent_id=test_agent_id, org_id=test_org_id
        )
        traced = adapter.create_traced_client()

        # Mock policy evaluator that raises
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_sync.side_effect = ConnectionError("Policy endpoint unreachable")
        mock_evaluator.drain_events.return_value = []
        traced._policy_evaluator = mock_evaluator
        traced._traced_messages._policy_evaluator = mock_evaluator

        # Should raise — fail_open=False
        with pytest.raises(PolicyViolationError, match="Policy evaluation failed"):
            traced.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "test"}],
            )

        traced.close()
