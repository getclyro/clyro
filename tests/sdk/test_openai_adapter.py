# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for the Clyro SDK OpenAI-compatible adapter
# Implements FRD-SDK-001 through FRD-SDK-006; VR-S1/S2/S3

"""
Unit tests for the OpenAI-compatible adapter (clyro/adapters/openai.py).

Covers:
- FRD-SDK-001: client detection (incl. OpenRouter), wiring, version validation
- FRD-SDK-002: LLM_CALL capture; ERROR event + re-raise on API exception
- FRD-SDK-003: TOOL_CALL per tool_calls entry, linked to parent; malformed entries
- FRD-SDK-004: Prevention Stack on tool calls (block / fail-open / fail-closed)
- FRD-SDK-005: cached-token cost (priced below full rate)
- FRD-SDK-006: streamed usage capture (+ estimate fallback) and tool-delta assembly
"""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest

from clyro.adapters.generic import detect_adapter
from clyro.adapters.openai import (
    CACHED_INPUT_DISCOUNT,
    AsyncOpenAITracedClient,
    OpenAIAdapter,
    OpenAITracedClient,
    is_openai_agent,
    validate_openai_version,
)
from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    ClyroWrapError,
    CostLimitExceededError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.trace import EventType, Framework

TEST_AGENT_ID = UUID("00000000-0000-0000-0000-000000000002")
TEST_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")


# ---------------------------------------------------------------------------
# Mocks — an OpenAI-shaped client/response (no network, no real SDK calls)
# ---------------------------------------------------------------------------


class MockPromptDetails:
    def __init__(self, cached_tokens: int = 0):
        self.cached_tokens = cached_tokens


class MockUsage:
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50, cached_tokens: int | None = None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if cached_tokens is not None:
            self.prompt_tokens_details = MockPromptDetails(cached_tokens)


class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(name, arguments)


class MockMessage:
    def __init__(self, content: str | None = "Hi there", tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class MockChoice:
    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockCompletion:
    def __init__(self, model="gpt-4o", choices=None, usage: MockUsage | None = None):
        self.model = model
        self.choices = choices if choices is not None else [MockChoice(MockMessage())]
        self.usage = usage if usage is not None else MockUsage()


class MockCompletions:
    def __init__(self, response: MockCompletion | None = None, error: Exception | None = None):
        self._response = response or MockCompletion()
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class MockChat:
    def __init__(self, completions: MockCompletions):
        self.completions = completions


class MockOpenAIClient:
    """Looks like openai.OpenAI for detection (module + class name)."""

    def __init__(self, completions: MockCompletions | None = None, name: str = "OpenAI"):
        self.chat = MockChat(completions or MockCompletions())
        self.api_key = "sk-test"
        type(self).__module__ = "openai._client"
        type(self).__name__ = name


# Streaming chunk mocks ------------------------------------------------------


class MockDelta:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls


class MockChunkChoice:
    def __init__(self, delta: MockDelta):
        self.delta = delta


class MockChunk:
    def __init__(self, choices=None, model="gpt-4o", usage=None):
        self.choices = choices if choices is not None else []
        self.model = model
        self.usage = usage


class MockToolCallDelta:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = MockFunction(name, arguments) if (name is not None or arguments is not None) else None


# Async client mocks --------------------------------------------------------


class MockAsyncStream:
    """Async-iterable stream of chunks (what `await client...create(stream=True)` returns)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class MockAsyncCompletions:
    def __init__(self, response=None, error=None, stream_chunks=None):
        self._response = response
        self._error = error
        self._stream_chunks = stream_chunks
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if kwargs.get("stream"):
            return MockAsyncStream(self._stream_chunks or [])
        return self._response if self._response is not None else MockCompletion()


class MockAsyncChat:
    def __init__(self, completions: MockAsyncCompletions):
        self.completions = completions


class MockAsyncOpenAIClient:
    """Looks like openai.AsyncOpenAI for detection (module + class name)."""

    def __init__(self, completions: MockAsyncCompletions | None = None):
        self.chat = MockAsyncChat(completions or MockAsyncCompletions())
        self.api_key = "sk-test"
        type(self).__module__ = "openai._client"
        type(self).__name__ = "AsyncOpenAI"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> ClyroConfig:
    return ClyroConfig(agent_name="test-openai-agent", capture_inputs=True, capture_outputs=True)


def _traced(client: MockOpenAIClient, config: ClyroConfig) -> OpenAITracedClient:
    adapter = OpenAIAdapter(
        client=client, config=config, agent_id=TEST_AGENT_ID, org_id=TEST_ORG_ID, validate_version=False
    )
    return adapter.create_traced_client()


def _events(traced: OpenAITracedClient, event_type: EventType) -> list:
    session = traced._session
    return [e for e in session.events if e.event_type == event_type] if session else []


# ---------------------------------------------------------------------------
# FRD-SDK-001 — detection, wiring, version
# ---------------------------------------------------------------------------


class TestDetection:
    def test_is_openai_agent_true_for_openai_client(self):
        assert is_openai_agent(MockOpenAIClient()) is True

    def test_is_openai_agent_true_for_async_client(self):
        assert is_openai_agent(MockOpenAIClient(name="AsyncOpenAI")) is True

    def test_is_openai_agent_false_for_non_openai(self):
        assert is_openai_agent(object()) is False
        assert is_openai_agent(lambda: None) is False

    def test_openrouter_client_detected(self):
        # OpenRouter uses the same openai.OpenAI class (custom base_url) -> detected.
        client = MockOpenAIClient()
        client.base_url = "https://openrouter.ai/api/v1"
        assert is_openai_agent(client) is True

    def test_detect_adapter_returns_openai(self):
        assert detect_adapter(MockOpenAIClient()) == "openai"

    def test_create_traced_client_returns_proxy(self, config):
        traced = _traced(MockOpenAIClient(), config)
        assert isinstance(traced, OpenAITracedClient)
        traced.close()

    def test_async_client_returns_async_traced_client(self, config):
        # Async clients are now traced (not rejected) — they get the async proxy.
        adapter = OpenAIAdapter(client=MockAsyncOpenAIClient(), config=config, validate_version=False)
        traced = adapter.create_traced_client()
        assert isinstance(traced, AsyncOpenAITracedClient)

    def test_double_wrap_rejected(self, config):
        traced = _traced(MockOpenAIClient(), config)
        with pytest.raises(ClyroWrapError, match="already wrapped"):
            OpenAIAdapter(client=traced, config=config, validate_version=False)
        traced.close()

    def test_version_validation_passes_with_installed_sdk(self):
        # openai is installed in the test env -> returns a version string.
        assert isinstance(validate_openai_version(), str)

    def test_passthrough_to_underlying_client(self, config):
        client = MockOpenAIClient()
        traced = _traced(client, config)
        assert traced.api_key == "sk-test"  # __getattr__ pass-through
        traced.close()


# ---------------------------------------------------------------------------
# FRD-SDK-002 — LLM_CALL capture + ERROR/re-raise
# ---------------------------------------------------------------------------


class TestLLMCapture:
    def test_emits_llm_call_with_tokens_cost_duration(self, config):
        client = MockOpenAIClient(MockCompletions(MockCompletion(usage=MockUsage(200, 100))))
        traced = _traced(client, config)
        traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        llm = _events(traced, EventType.LLM_CALL)
        assert len(llm) == 1
        assert llm[0].event_name == "gpt-4o"
        assert llm[0].token_count_input == 200  # VR-S1
        assert llm[0].token_count_output == 100
        assert llm[0].cost_usd > Decimal("0")
        assert llm[0].framework == Framework.OPENAI
        traced.close()

    def test_returns_original_response_unchanged(self, config):
        response = MockCompletion()
        traced = _traced(MockOpenAIClient(MockCompletions(response)), config)
        result = traced.chat.completions.create(model="gpt-4o", messages=[])
        assert result is response
        traced.close()

    def test_api_exception_emits_error_and_reraises_unchanged(self, config):
        boom = RuntimeError("rate limited")
        traced = _traced(MockOpenAIClient(MockCompletions(error=boom)), config)
        with pytest.raises(RuntimeError) as exc:
            traced.chat.completions.create(model="gpt-4o", messages=[])
        assert exc.value is boom  # re-raised unchanged
        errors = _events(traced, EventType.ERROR)
        assert len(errors) == 1
        assert errors[0].error_type == "RuntimeError"
        traced.close()

    def test_tracing_failure_does_not_break_call(self, config, monkeypatch):
        # NFR-SDK-002: a Clyro-internal error must not break the agent's call.
        traced = _traced(MockOpenAIClient(), config)
        monkeypatch.setattr(traced.chat.completions, "_extract_tokens", lambda r: (_ for _ in ()).throw(ValueError("boom")))
        result = traced.chat.completions.create(model="gpt-4o", messages=[])
        assert result is not None  # call still returns
        traced.close()


# ---------------------------------------------------------------------------
# FRD-SDK-003 — TOOL_CALL capture
# ---------------------------------------------------------------------------


class TestToolCapture:
    def test_one_tool_call_event_per_entry_linked_to_parent(self, config):
        message = MockMessage(
            content=None,
            tool_calls=[
                MockToolCall("call_1", "search", '{"q": "x"}'),
                MockToolCall("call_2", "calc", '{"a": 1}'),
            ],
        )
        response = MockCompletion(choices=[MockChoice(message, finish_reason="tool_calls")])
        traced = _traced(MockOpenAIClient(MockCompletions(response)), config)
        traced.chat.completions.create(model="gpt-4o", messages=[])

        tools = _events(traced, EventType.TOOL_CALL)
        llm = _events(traced, EventType.LLM_CALL)
        assert len(tools) == 2
        assert {t.event_name for t in tools} == {"search", "calc"}
        assert tools[0].input_data == {"q": "x"}
        # VR-S2: every TOOL_CALL links to the parent LLM_CALL
        assert all(t.parent_event_id == llm[0].event_id for t in tools)
        traced.close()

    def test_malformed_tool_call_emitted_not_skipped(self, config):
        # A tool_call missing a name + non-JSON args -> emit with available data, never raise.
        bad = MockToolCall("call_x", None, "not-json")
        bad.function.name = None
        response = MockCompletion(choices=[MockChoice(MockMessage(content=None, tool_calls=[bad]))])
        traced = _traced(MockOpenAIClient(MockCompletions(response)), config)
        traced.chat.completions.create(model="gpt-4o", messages=[])

        tools = _events(traced, EventType.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].event_name == "unknown"
        assert tools[0].input_data == {"arguments": "not-json"}
        traced.close()

    def test_no_tool_calls_emits_only_llm(self, config):
        traced = _traced(MockOpenAIClient(), config)
        traced.chat.completions.create(model="gpt-4o", messages=[])
        assert len(_events(traced, EventType.TOOL_CALL)) == 0
        assert len(_events(traced, EventType.LLM_CALL)) == 1
        traced.close()


# ---------------------------------------------------------------------------
# FRD-SDK-004 — Prevention Stack on tool calls
# ---------------------------------------------------------------------------


def _client_with_tool() -> MockOpenAIClient:
    message = MockMessage(content=None, tool_calls=[MockToolCall("call_1", "danger", "{}")])
    return MockOpenAIClient(MockCompletions(MockCompletion(choices=[MockChoice(message, "tool_calls")])))


class TestPreventionStack:
    def test_policy_block_raises_violation(self, config, monkeypatch):
        from clyro.session import Session

        def _block(self, action_type, parameters=None, **kw):
            if action_type == "tool_call":
                raise PolicyViolationError(
                    rule_id="r1", rule_name="No danger", message="blocked", action_type="tool_call"
                )

        monkeypatch.setattr(Session, "check_policy", _block)
        traced = _traced(_client_with_tool(), config)
        with pytest.raises(PolicyViolationError):
            traced.chat.completions.create(model="gpt-4o", messages=[])
        traced.close()

    def test_policy_allow_proceeds(self, config, monkeypatch):
        from clyro.session import Session

        monkeypatch.setattr(Session, "check_policy", lambda self, *a, **k: None)
        traced = _traced(_client_with_tool(), config)
        traced.chat.completions.create(model="gpt-4o", messages=[])  # no raise
        assert len(_events(traced, EventType.TOOL_CALL)) == 1
        traced.close()

    def test_policy_unreachable_fail_open_proceeds(self, monkeypatch):
        from clyro.session import Session

        cfg = ClyroConfig(agent_name="t", fail_open=True)
        monkeypatch.setattr(Session, "check_policy", lambda self, *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        traced = _traced(_client_with_tool(), cfg)
        traced.chat.completions.create(model="gpt-4o", messages=[])  # proceeds despite policy error
        traced.close()

    def test_policy_unreachable_fail_closed_blocks(self, monkeypatch):
        from clyro.session import Session

        cfg = ClyroConfig(agent_name="t", fail_open=False)
        monkeypatch.setattr(Session, "check_policy", lambda self, *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        traced = _traced(_client_with_tool(), cfg)
        with pytest.raises(PolicyViolationError, match="policy_unavailable|Policy evaluation failed"):
            traced.chat.completions.create(model="gpt-4o", messages=[])
        traced.close()


# ---------------------------------------------------------------------------
# Execution controls — step / cost / loop limits (parity with other adapters)
# ---------------------------------------------------------------------------


class TestExecutionControls:
    def test_step_limit_enforced(self):
        cfg = ClyroConfig(agent_name="t", controls=ExecutionControls(max_steps=1, enable_step_limit=True))
        traced = _traced(MockOpenAIClient(), cfg)
        traced.chat.completions.create(model="gpt-4o", messages=[])  # step 1 — ok
        with pytest.raises(StepLimitExceededError):
            traced.chat.completions.create(model="gpt-4o", messages=[])  # step 2 > limit
        traced.close()

    def test_cost_limit_enforced(self):
        cfg = ClyroConfig(
            agent_name="t", controls=ExecutionControls(max_cost_usd=0.0000001, enable_cost_limit=True)
        )
        client = MockOpenAIClient(MockCompletions(MockCompletion(usage=MockUsage(1000, 1000))))
        traced = _traced(client, cfg)
        # Enforced at the FIRST crossing: this call's own recorded cost pushes the
        # cumulative over the ceiling, so it is blocked right after the LLM cost is logged.
        with pytest.raises(CostLimitExceededError):
            traced.chat.completions.create(model="gpt-4o", messages=[])
        traced.close()


# ---------------------------------------------------------------------------
# Pre-LLM (input) policy
# ---------------------------------------------------------------------------


class TestInputPolicy:
    def test_input_policy_blocks_before_api_call(self, config, monkeypatch):
        from clyro.session import Session

        def _block(self, action_type, parameters=None, **kw):
            if action_type == "llm_call":
                raise PolicyViolationError(
                    rule_id="r", rule_name="bad input", message="blocked", action_type="llm_call"
                )

        monkeypatch.setattr(Session, "check_policy", _block)
        completions = MockCompletions()
        traced = _traced(MockOpenAIClient(completions), config)
        with pytest.raises(PolicyViolationError):
            traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "bad"}])
        assert completions.calls == []  # blocked pre-call — provider never hit
        traced.close()


# ---------------------------------------------------------------------------
# Trace completeness + delivery (emit-before-block, auto-flush)
# ---------------------------------------------------------------------------


class TestTraceCompletenessAndFlush:
    def test_block_on_first_tool_not_emitted(self, config, monkeypatch):
        from clyro.session import Session

        def _block(self, action_type, parameters=None, **kw):
            if action_type == "tool_call":
                raise PolicyViolationError(rule_id="r", rule_name="n", message="m", action_type="tool_call")

        monkeypatch.setattr(Session, "check_policy", _block)
        message = MockMessage(
            content=None,
            tool_calls=[MockToolCall("c1", "a", "{}"), MockToolCall("c2", "b", "{}"), MockToolCall("c3", "c", "{}")],
        )
        response = MockCompletion(choices=[MockChoice(message, "tool_calls")])
        traced = _traced(MockOpenAIClient(MockCompletions(response)), config)
        with pytest.raises(PolicyViolationError):
            traced.chat.completions.create(model="gpt-4o", messages=[])
        # Policy is evaluated BEFORE emitting each TOOL_CALL, so a blocked tool is never
        # emitted as a tool_call — it surfaces as the raised PolicyViolationError instead.
        assert len(_events(traced, EventType.TOOL_CALL)) == 0
        traced.close()

    def test_completed_turn_auto_flushes(self, config):
        traced = _traced(MockOpenAIClient(), config)  # default finish_reason='stop'
        traced.chat.completions.create(model="gpt-4o", messages=[])
        # turn complete -> session ended eagerly (not deferred to close()/atexit)
        assert traced._session.is_active is False
        traced.close()

    def test_tool_loop_turn_stays_open(self, config):
        message = MockMessage(content=None, tool_calls=[MockToolCall("c1", "s", "{}")])
        response = MockCompletion(choices=[MockChoice(message, "tool_calls")])
        traced = _traced(MockOpenAIClient(MockCompletions(response)), config)
        traced.chat.completions.create(model="gpt-4o", messages=[])
        assert traced._session.is_active is True  # mid tool-loop -> session kept open
        traced.close()


# ---------------------------------------------------------------------------
# Streaming: usage injection + fail-open
# ---------------------------------------------------------------------------


class TestStreamingHardening:
    def test_stream_options_injected_for_usage(self, config):
        completions = MockCompletions([MockChunk(choices=[], usage=MockUsage(5, 5))])
        traced = _traced(MockOpenAIClient(completions), config)
        list(traced.chat.completions.create(model="gpt-4o", messages=[], stream=True))
        assert completions.calls[0].get("stream_options") == {"include_usage": True}
        traced.close()

    def test_stream_options_not_overridden(self, config):
        completions = MockCompletions([MockChunk(choices=[], usage=MockUsage(5, 5))])
        traced = _traced(MockOpenAIClient(completions), config)
        list(
            traced.chat.completions.create(
                model="gpt-4o", messages=[], stream=True, stream_options={"include_usage": False}
            )
        )
        assert completions.calls[0]["stream_options"] == {"include_usage": False}
        traced.close()

    def test_stream_tool_processing_failure_does_not_break_iteration(self, config, monkeypatch):
        # NFR-SDK-002: a Clyro-internal error during streamed tool processing must
        # NOT propagate into the caller's `for chunk in stream` loop.
        chunks = [
            MockChunk(choices=[MockChunkChoice(MockDelta(tool_calls=[MockToolCallDelta(0, id="c1", name="s", arguments="{}")]))]),
            MockChunk(choices=[], usage=MockUsage(5, 5)),
        ]
        traced = _traced(MockOpenAIClient(MockCompletions(chunks)), config)
        monkeypatch.setattr(
            traced.chat.completions,
            "_emit_tool_call",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        collected = list(traced.chat.completions.create(model="gpt-4o", messages=[], stream=True))
        # iteration completes without leaking the internal error; the injected
        # usage-only chunk is swallowed, leaving just the tool-delta chunk.
        assert len(collected) == 1
        traced.close()


# ---------------------------------------------------------------------------
# Async adapter (openai.AsyncOpenAI) — mirrors the sync coverage
# ---------------------------------------------------------------------------


def _async_traced(client: MockAsyncOpenAIClient, config: ClyroConfig) -> AsyncOpenAITracedClient:
    adapter = OpenAIAdapter(
        client=client, config=config, agent_id=TEST_AGENT_ID, org_id=TEST_ORG_ID, validate_version=False
    )
    return adapter.create_traced_client()


def _async_events(traced: AsyncOpenAITracedClient, event_type: EventType) -> list:
    session = traced._session
    return [e for e in session.events if e.event_type == event_type] if session else []


async def _adrain(stream) -> list:
    return [chunk async for chunk in stream]


class TestAsyncAdapter:
    async def test_returns_async_client(self, config):
        traced = _async_traced(MockAsyncOpenAIClient(), config)
        assert isinstance(traced, AsyncOpenAITracedClient)
        await traced.close()

    async def test_emits_llm_call_with_tokens_cost(self, config):
        client = MockAsyncOpenAIClient(MockAsyncCompletions(MockCompletion(usage=MockUsage(200, 100))))
        traced = _async_traced(client, config)
        await traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        llm = _async_events(traced, EventType.LLM_CALL)
        assert len(llm) == 1
        assert llm[0].event_name == "gpt-4o"
        assert llm[0].token_count_input == 200  # VR-S1
        assert llm[0].token_count_output == 100
        assert llm[0].cost_usd > Decimal("0")
        assert llm[0].framework == Framework.OPENAI
        await traced.close()

    async def test_returns_original_response_unchanged(self, config):
        response = MockCompletion()
        traced = _async_traced(MockAsyncOpenAIClient(MockAsyncCompletions(response)), config)
        result = await traced.chat.completions.create(model="gpt-4o", messages=[])
        assert result is response
        await traced.close()

    async def test_api_exception_emits_error_and_reraises_unchanged(self, config):
        boom = RuntimeError("rate limited")
        traced = _async_traced(MockAsyncOpenAIClient(MockAsyncCompletions(error=boom)), config)
        with pytest.raises(RuntimeError) as exc:
            await traced.chat.completions.create(model="gpt-4o", messages=[])
        assert exc.value is boom  # re-raised unchanged
        errors = _async_events(traced, EventType.ERROR)
        assert len(errors) == 1 and errors[0].error_type == "RuntimeError"
        await traced.close()

    async def test_tool_calls_emitted_and_linked(self, config):
        message = MockMessage(
            content=None,
            tool_calls=[MockToolCall("call_1", "search", '{"q": "x"}'), MockToolCall("call_2", "calc", '{"a": 1}')],
        )
        response = MockCompletion(choices=[MockChoice(message, finish_reason="tool_calls")])
        traced = _async_traced(MockAsyncOpenAIClient(MockAsyncCompletions(response)), config)
        await traced.chat.completions.create(model="gpt-4o", messages=[])

        tools = _async_events(traced, EventType.TOOL_CALL)
        llm = _async_events(traced, EventType.LLM_CALL)
        assert {t.event_name for t in tools} == {"search", "calc"}
        assert all(t.parent_event_id == llm[0].event_id for t in tools)  # VR-S2
        await traced.close()

    async def test_tracing_failure_does_not_break_call(self, config, monkeypatch):
        # NFR-SDK-002: a Clyro-internal error must not break the agent's call.
        traced = _async_traced(MockAsyncOpenAIClient(), config)
        monkeypatch.setattr(
            traced.chat.completions, "_extract_tokens", lambda r: (_ for _ in ()).throw(ValueError("boom"))
        )
        result = await traced.chat.completions.create(model="gpt-4o", messages=[])
        assert result is not None
        await traced.close()

    async def test_policy_block_raises_violation(self, config, monkeypatch):
        from clyro.session import Session

        async def _block(self, action_type, parameters=None, **kw):
            if action_type == "tool_call":
                raise PolicyViolationError(rule_id="r1", rule_name="n", message="blocked", action_type="tool_call")

        monkeypatch.setattr(Session, "check_policy_async", _block)
        message = MockMessage(content=None, tool_calls=[MockToolCall("c1", "danger", "{}")])
        response = MockCompletion(choices=[MockChoice(message, "tool_calls")])
        traced = _async_traced(MockAsyncOpenAIClient(MockAsyncCompletions(response)), config)
        with pytest.raises(PolicyViolationError):
            await traced.chat.completions.create(model="gpt-4o", messages=[])
        await traced.close()

    async def test_input_policy_blocks_before_api_call(self, config, monkeypatch):
        from clyro.session import Session

        async def _block(self, action_type, parameters=None, **kw):
            if action_type == "llm_call":
                raise PolicyViolationError(rule_id="r", rule_name="n", message="blocked", action_type="llm_call")

        monkeypatch.setattr(Session, "check_policy_async", _block)
        completions = MockAsyncCompletions()
        traced = _async_traced(MockAsyncOpenAIClient(completions), config)
        with pytest.raises(PolicyViolationError):
            await traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "bad"}])
        assert completions.calls == []  # blocked pre-call — provider never hit
        await traced.close()

    async def test_step_limit_enforced(self):
        cfg = ClyroConfig(agent_name="t", controls=ExecutionControls(max_steps=1, enable_step_limit=True))
        traced = _async_traced(MockAsyncOpenAIClient(), cfg)
        await traced.chat.completions.create(model="gpt-4o", messages=[])  # step 1 — ok
        with pytest.raises(StepLimitExceededError):
            await traced.chat.completions.create(model="gpt-4o", messages=[])  # step 2 > limit
        await traced.close()

    async def test_streaming_captures_real_usage_and_swallows_induced_chunk(self, config):
        # content chunk, then a usage-only chunk (what the provider returns once we
        # inject include_usage). The usage-only chunk must be swallowed from the caller.
        chunks = [
            MockChunk(choices=[MockChunkChoice(MockDelta(content="Hello"))]),
            MockChunk(choices=[], usage=MockUsage(200, 100)),
        ]
        completions = MockAsyncCompletions(stream_chunks=chunks)
        traced = _async_traced(MockAsyncOpenAIClient(completions), config)
        stream = await traced.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        collected = await _adrain(stream)

        assert completions.calls[0].get("stream_options") == {"include_usage": True}
        assert len(collected) == 1  # induced usage-only chunk swallowed
        llm = _async_events(traced, EventType.LLM_CALL)
        assert len(llm) == 1
        assert llm[0].token_count_input == 200  # real usage, not estimated
        assert llm[0].metadata.get("estimated") is None
        await traced.close()

    async def test_streaming_estimates_when_usage_absent(self, config):
        chunks = [MockChunk(choices=[MockChunkChoice(MockDelta(content="abcd" * 4))])]  # no usage chunk
        traced = _async_traced(MockAsyncOpenAIClient(MockAsyncCompletions(stream_chunks=chunks)), config)
        stream = await traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True)
        await _adrain(stream)
        llm = _async_events(traced, EventType.LLM_CALL)
        assert len(llm) == 1
        assert llm[0].metadata.get("estimated") is True  # VR-S3
        assert llm[0].token_count_output >= 1  # never zero
        await traced.close()

    async def test_async_context_manager_closes(self, config):
        async with _async_traced(MockAsyncOpenAIClient(), config) as traced:
            await traced.chat.completions.create(model="gpt-4o", messages=[])
        assert traced._closed is True


# ---------------------------------------------------------------------------
# FRD-SDK-005 — cached-token cost
# ---------------------------------------------------------------------------


class TestCachedTokenCost:
    def test_cached_tokens_priced_below_full_rate(self, config):
        traced = _traced(MockOpenAIClient(), config)
        completions = traced.chat.completions
        full = completions._compute_cost(1000, 0, cached_tokens=0, model="gpt-4o")
        with_cache = completions._compute_cost(1000, 0, cached_tokens=1000, model="gpt-4o")
        assert with_cache < full  # cached reads cost less than full-rate
        assert with_cache == full * CACHED_INPUT_DISCOUNT  # exactly the discount on a fully-cached call
        traced.close()

    def test_response_cached_tokens_reduce_event_cost(self, config):
        cached = MockOpenAIClient(MockCompletions(MockCompletion(usage=MockUsage(1000, 0, cached_tokens=1000))))
        uncached = MockOpenAIClient(MockCompletions(MockCompletion(usage=MockUsage(1000, 0, cached_tokens=0))))
        t1 = _traced(cached, config)
        t1.chat.completions.create(model="gpt-4o", messages=[])
        t2 = _traced(uncached, config)
        t2.chat.completions.create(model="gpt-4o", messages=[])
        assert _events(t1, EventType.LLM_CALL)[0].cost_usd < _events(t2, EventType.LLM_CALL)[0].cost_usd
        assert _events(t1, EventType.LLM_CALL)[0].metadata.get("cached_tokens") == 1000
        t1.close()
        t2.close()

    def test_absent_cached_fields_uses_standard_cost(self, config):
        traced = _traced(MockOpenAIClient(), config)
        c = traced.chat.completions
        assert c._compute_cost(500, 200, cached_tokens=0, model="gpt-4o") == c._compute_cost(500, 200, 0, "gpt-4o")
        traced.close()


# ---------------------------------------------------------------------------
# FRD-SDK-006 — streamed usage capture + tool-delta assembly
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_streamed_usage_captured_on_completion(self, config):
        chunks = [
            MockChunk(choices=[MockChunkChoice(MockDelta(content="Hel"))]),
            MockChunk(choices=[MockChunkChoice(MockDelta(content="lo"))]),
            MockChunk(choices=[], usage=MockUsage(40, 12)),  # final usage chunk
        ]
        traced = _traced(MockOpenAIClient(MockCompletions(chunks)), config)
        stream = traced.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        collected = list(stream)  # caller consumes the stream

        # We injected include_usage, so the trailing usage-only chunk is swallowed —
        # the caller sees only the content chunks it would have without Clyro.
        assert len(collected) == 2
        llm = _events(traced, EventType.LLM_CALL)
        assert len(llm) == 1
        assert llm[0].token_count_input == 40  # usage still captured from the swallowed chunk
        assert llm[0].token_count_output == 12
        assert llm[0].metadata.get("estimated") is None  # real usage, not estimated
        traced.close()

    def test_streamed_usage_absent_estimates_nonzero(self, config):
        chunks = [MockChunk(choices=[MockChunkChoice(MockDelta(content="some output text here"))])]
        traced = _traced(MockOpenAIClient(MockCompletions(chunks)), config)
        list(traced.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}], stream=True))

        llm = _events(traced, EventType.LLM_CALL)[0]
        assert llm.metadata.get("estimated") is True  # VR-S3
        assert llm.token_count_output > 0  # never zero
        traced.close()

    def test_streamed_tool_call_deltas_assembled_into_one(self, config):
        # Tool call arrives split across chunks -> assembled into a single TOOL_CALL.
        chunks = [
            MockChunk(choices=[MockChunkChoice(MockDelta(tool_calls=[MockToolCallDelta(0, id="call_1", name="search", arguments='{"q":')]))]),
            MockChunk(choices=[MockChunkChoice(MockDelta(tool_calls=[MockToolCallDelta(0, arguments='"weather"}')]))]),
            MockChunk(choices=[], usage=MockUsage(10, 5)),
        ]
        traced = _traced(MockOpenAIClient(MockCompletions(chunks)), config)
        list(traced.chat.completions.create(model="gpt-4o", messages=[], stream=True))

        tools = _events(traced, EventType.TOOL_CALL)
        assert len(tools) == 1  # not one per delta
        assert tools[0].event_name == "search"
        assert tools[0].input_data == {"q": "weather"}
        assert tools[0].parent_event_id == _events(traced, EventType.LLM_CALL)[0].event_id
        traced.close()


# ---------------------------------------------------------------------------
# OpenRouter provider naming (base_url detection)
# ---------------------------------------------------------------------------


def _openrouter_client() -> MockOpenAIClient:
    client = MockOpenAIClient()
    client.base_url = "https://openrouter.ai/api/v1"
    return client


class TestProviderNaming:
    def test_openrouter_base_url_tags_as_openrouter(self, config):
        client = _openrouter_client()
        assert is_openai_agent(client)  # still routed through the OpenAI adapter
        adapter = OpenAIAdapter(client=client, config=config, validate_version=False)
        assert adapter.framework == Framework.OPENROUTER
        assert adapter.name == "openrouter"

        traced = adapter.create_traced_client()
        traced.chat.completions.create(model="anthropic/claude-3.5-sonnet", messages=[])
        llm = _events(traced, EventType.LLM_CALL)[0]
        assert llm.framework == Framework.OPENROUTER  # events attributed to OpenRouter
        traced.close()

    def test_plain_openai_base_url_stays_openai(self, config):
        adapter = OpenAIAdapter(client=MockOpenAIClient(), config=config, validate_version=False)
        assert adapter.framework == Framework.OPENAI
        assert adapter.name == "openai"
        traced = adapter.create_traced_client()
        traced.chat.completions.create(model="gpt-4o", messages=[])
        assert _events(traced, EventType.LLM_CALL)[0].framework == Framework.OPENAI
        traced.close()


# ---------------------------------------------------------------------------
# Local-mode policy enforcement (no API key required)
# ---------------------------------------------------------------------------


class TestLocalModePolicy:
    def test_local_mode_wires_local_evaluator_without_api_key(self):
        from clyro.local_policy import SDKLocalPolicyEvaluator

        cfg = ClyroConfig(
            agent_name="t", mode="local", controls=ExecutionControls(enable_policy_enforcement=True)
        )
        assert cfg.api_key is None
        traced = _traced(MockOpenAIClient(), cfg)
        assert isinstance(traced._policy_evaluator, SDKLocalPolicyEvaluator)
        traced.close()

    def test_local_mode_enforces_policy_without_api_key(self, monkeypatch):
        from clyro.local_policy import SDKLocalPolicyEvaluator

        def _block(self, action_type, parameters, session_id, step_number):
            if action_type == "tool_call":
                raise PolicyViolationError(rule_id="r", rule_name="n", message="blocked", action_type="tool_call")

        monkeypatch.setattr(SDKLocalPolicyEvaluator, "evaluate_sync", _block)
        cfg = ClyroConfig(
            agent_name="t", mode="local", controls=ExecutionControls(enable_policy_enforcement=True)
        )
        traced = _traced(_client_with_tool(), cfg)
        with pytest.raises(PolicyViolationError):  # enforced with no api_key
            traced.chat.completions.create(model="gpt-4o", messages=[])
        traced.close()


# ---------------------------------------------------------------------------
# Streaming injection transparency (#1) + endpoint gating (#2)
# ---------------------------------------------------------------------------


class TestStreamUsageInjectionSafety:
    def test_user_supplied_stream_options_are_delivered(self, config):
        # Caller asked for usage themselves -> we don't inject, so we don't swallow.
        chunks = [
            MockChunk(choices=[MockChunkChoice(MockDelta(content="hi"))]),
            MockChunk(choices=[], usage=MockUsage(7, 3)),
        ]
        traced = _traced(MockOpenAIClient(MockCompletions(chunks)), config)
        collected = list(
            traced.chat.completions.create(
                model="gpt-4o", messages=[], stream=True, stream_options={"include_usage": True}
            )
        )
        assert len(collected) == 2  # the usage chunk the caller requested is delivered
        traced.close()

    def test_no_injection_on_non_openai_endpoint(self, config):
        # Azure / vLLM / Groq etc. may reject stream_options -> never inject there.
        completions = MockCompletions([MockChunk(choices=[MockChunkChoice(MockDelta(content="x"))])])
        client = MockOpenAIClient(completions)
        client.base_url = "https://myresource.openai.azure.com/openai"
        traced = _traced(client, config)
        list(traced.chat.completions.create(model="gpt-4o", messages=[], stream=True))
        assert "stream_options" not in completions.calls[0]  # not injected on Azure
        traced.close()

    def test_injection_on_openrouter_endpoint(self, config):
        completions = MockCompletions([MockChunk(choices=[], usage=MockUsage(5, 5))])
        client = _openrouter_client()
        client.chat = MockChat(completions)
        traced = _traced(client, config)
        list(traced.chat.completions.create(model="x", messages=[], stream=True))
        assert completions.calls[0].get("stream_options") == {"include_usage": True}
        traced.close()


# ---------------------------------------------------------------------------
# Tool-result backfill (FRD-SDK-003) — the next turn's role:"tool" message is
# attached to the matching TOOL_CALL event's output_data (parity w/ Anthropic).
# ---------------------------------------------------------------------------


class TestToolResultBackfill:
    def test_result_backfilled_onto_tool_call(self, config):
        msg = MockMessage(content=None, tool_calls=[MockToolCall("call_1", "get_weather", '{"city": "Paris"}')])
        client = MockOpenAIClient(MockCompletions(MockCompletion(choices=[MockChoice(msg, "tool_calls")])))
        traced = _traced(client, config)
        traced.chat.completions.create(model="gpt-4o", messages=[])
        tool = _events(traced, EventType.TOOL_CALL)[0]
        assert tool.output_data is None  # no result yet

        client.chat.completions._response = MockCompletion()  # next turn: normal stop
        traced.chat.completions.create(
            model="gpt-4o", messages=[{"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"}]
        )
        assert tool.output_data == {"result": "22C sunny"}  # backfilled by tool_call_id
        traced.close()

    def test_unmatched_id_not_backfilled(self, config):
        msg = MockMessage(content=None, tool_calls=[MockToolCall("call_1", "get_weather", "{}")])
        client = MockOpenAIClient(MockCompletions(MockCompletion(choices=[MockChoice(msg, "tool_calls")])))
        traced = _traced(client, config)
        traced.chat.completions.create(model="gpt-4o", messages=[])
        tool = _events(traced, EventType.TOOL_CALL)[0]
        client.chat.completions._response = MockCompletion()
        traced.chat.completions.create(
            model="gpt-4o", messages=[{"role": "tool", "tool_call_id": "other", "content": "r"}]
        )
        assert tool.output_data is None  # id didn't match -> left as-is
        traced.close()

    async def test_async_result_backfilled(self, config):
        msg = MockMessage(content=None, tool_calls=[MockToolCall("call_1", "get_weather", "{}")])
        client = MockAsyncOpenAIClient(MockAsyncCompletions(MockCompletion(choices=[MockChoice(msg, "tool_calls")])))
        traced = _traced(client, config)
        await traced.chat.completions.create(model="gpt-4o", messages=[])
        tool = _events(traced, EventType.TOOL_CALL)[0]
        assert tool.output_data is None

        client.chat.completions._response = MockCompletion()
        await traced.chat.completions.create(
            model="gpt-4o", messages=[{"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"}]
        )
        assert tool.output_data == {"result": "22C sunny"}
        await traced.close()
