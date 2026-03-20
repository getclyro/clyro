# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Anthropic Adapter
# Implements FRD-001 through FRD-012

"""
Anthropic SDK adapter for the Clyro SDK.

This adapter provides transparent tracing and governance for applications
built directly on the Anthropic Python SDK (`anthropic.Anthropic` and
`anthropic.AsyncAnthropic`). It intercepts `messages.create()` and
`messages.stream()` calls to emit trace events, enforce execution controls,
and evaluate policies on tool use.

The adapter uses a proxy pattern: `clyro.wrap()` returns a traced client
that behaves identically to the original Anthropic client for all methods
not explicitly instrumented.

Architecture:
    User Code → AnthropicTracedClient (proxy)
        → TracedMessages.create() / stream()
            → Prevention Stack (step/cost/loop checks)
            → Original client.messages.create() / stream()
            → Token extraction + cost calculation
            → LLM_CALL event emission
            → Tool use detection → TOOL_CALL events + policy enforcement
        → Return response to user

Note: The `anthropic` package is an optional/peer dependency. It is only
imported when an Anthropic client is detected — never at module level.
"""

from __future__ import annotations

import atexit
import time
import traceback
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog

from clyro.cost import AnthropicTokenExtractor, CostCalculator
from clyro.exceptions import (
    ClyroWrapError,
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.policy import PolicyEvaluator
from clyro.session import Session
from clyro.trace import (
    AgentStage,
    Framework,
    TraceEvent,
    create_error_event,
    create_llm_call_event,
    create_step_event,
    create_tool_call_event,
)
from clyro.transport import SyncTransport, Transport

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.policy import ApprovalHandler

logger = structlog.get_logger(__name__)

# Minimum supported Anthropic SDK version.
# The Anthropic Python SDK uses 0.x versioning (e.g. 0.39.0, 0.77.1).
# Tool-use support (messages API with tools) was added in 0.18.0.
MIN_ANTHROPIC_VERSION = "0.18.0"


# ---------------------------------------------------------------------------
# C1: Anthropic Adapter — detection, version validation, factory
# ---------------------------------------------------------------------------


def is_anthropic_agent(obj: Any) -> bool:
    """
    Detect whether an object is an Anthropic SDK client.  # Implements FRD-001

    Uses module inspection and class name checks — not isinstance — to avoid
    requiring the `anthropic` package as a hard dependency.

    Args:
        obj: Object to inspect.

    Returns:
        True if obj is an Anthropic or AsyncAnthropic client instance.
    """
    try:
        obj_type = type(obj)
        module = getattr(obj_type, "__module__", "") or ""
        name = getattr(obj_type, "__name__", "") or ""
        return module.startswith("anthropic") and name in ("Anthropic", "AsyncAnthropic")
    except Exception:
        return False


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '1.2.3' into a tuple of ints."""
    try:
        return tuple(int(p) for p in version_str.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def detect_anthropic_version() -> str | None:
    """Detect the installed Anthropic SDK version without validating."""
    try:
        import anthropic

        return getattr(anthropic, "__version__", None)
    except ImportError:
        return None


def validate_anthropic_version() -> str:
    """
    Validate that the installed Anthropic SDK version is supported.  # Implements FRD-001

    Returns:
        The detected version string.

    Raises:
        FrameworkVersionError: If the version is below the minimum.
    """
    version = detect_anthropic_version()

    if version is None:
        raise FrameworkVersionError(
            framework="anthropic",
            version="not installed",
            supported=f">={MIN_ANTHROPIC_VERSION}",
        )

    # If version can't be determined, assume compatible (fail-open)
    if version == "unknown":
        logger.warning(
            "anthropic_version_unknown",
            message="Could not determine Anthropic SDK version, assuming compatible",
        )
        return version

    if _parse_version(version) < _parse_version(MIN_ANTHROPIC_VERSION):
        raise FrameworkVersionError(
            framework="anthropic",
            version=version,
            supported=f">={MIN_ANTHROPIC_VERSION}",
        )

    return version


class AnthropicAdapter:
    """
    Adapter for Anthropic SDK clients.  # Implements FRD-001, FRD-002

    Detects Anthropic clients, validates versions, and creates traced client
    instances. Follows the same adapter pattern as LangGraphAdapter and
    CrewAIAdapter.
    """

    FRAMEWORK = Framework.ANTHROPIC
    FRAMEWORK_VERSION: str | None = None

    def __init__(
        self,
        client: Any,
        config: ClyroConfig,
        *,
        agent_id: UUID | None = None,
        org_id: UUID | None = None,
        approval_handler: ApprovalHandler | None | object = None,
        validate_version: bool = True,
    ):
        """
        Initialize the Anthropic adapter.

        Args:
            client: An anthropic.Anthropic or anthropic.AsyncAnthropic instance.
            config: Clyro SDK configuration.
            agent_id: Resolved agent UUID.
            org_id: Organization UUID.
            approval_handler: Optional approval handler for require_approval policies.
            validate_version: Whether to validate Anthropic SDK version.

        Raises:
            FrameworkVersionError: If SDK version is unsupported and validate_version=True.
            ClyroWrapError: If the client is already wrapped.
        """
        # Double-wrap detection  # Implements FRD-002
        if getattr(client, "_clyro_wrapped", False):
            raise ClyroWrapError(
                message="Client is already wrapped by Clyro",
                agent_type=type(client).__name__,
            )

        self._client = client
        self._config = config
        self._agent_id = agent_id
        self._org_id = org_id
        self._approval_handler = approval_handler

        # Validate version if requested (fail-open when validate_version=False)
        if validate_version:
            self.FRAMEWORK_VERSION = validate_anthropic_version()
        else:
            self.FRAMEWORK_VERSION = detect_anthropic_version() or "unknown"

        # Detect sync vs async
        client_name = type(client).__name__
        self._is_async = client_name == "AsyncAnthropic"

        logger.debug(
            "anthropic_adapter_init",
            client_type=client_name,
            is_async=self._is_async,
            version=self.FRAMEWORK_VERSION,
            agent_id=str(agent_id) if agent_id else None,
        )

    @property
    def agent(self) -> Any:
        """Get the underlying Anthropic client."""
        return self._client

    @property
    def name(self) -> str:
        """Get the adapter name."""
        return "anthropic"

    @property
    def framework(self) -> Framework:
        """Get the framework type."""
        return self.FRAMEWORK

    @property
    def framework_version(self) -> str | None:
        """Get the Anthropic SDK version."""
        return self.FRAMEWORK_VERSION

    def create_traced_client(self) -> AnthropicTracedClient | AsyncAnthropicTracedClient:
        """
        Create a traced client proxy.  # Implements FRD-002

        Returns:
            AnthropicTracedClient for sync clients,
            AsyncAnthropicTracedClient for async clients.
        """
        # Initialize transport
        if self._is_async:
            transport = Transport(self._config)
        else:
            transport = SyncTransport(self._config)

        # Initialize policy evaluator (if enabled)
        policy_evaluator: PolicyEvaluator | None = None
        if self._config.controls.enable_policy_enforcement and self._config.api_key:
            policy_evaluator = PolicyEvaluator(
                config=self._config,
                agent_id=self._agent_id,
                org_id=self._org_id,
                approval_handler=self._approval_handler,
            )

        if self._is_async:
            return AsyncAnthropicTracedClient(
                client=self._client,
                config=self._config,
                transport=transport,
                policy_evaluator=policy_evaluator,
                agent_id=self._agent_id,
                org_id=self._org_id,
                framework_version=self.FRAMEWORK_VERSION,
            )
        else:
            return AnthropicTracedClient(
                client=self._client,
                config=self._config,
                transport=transport,
                policy_evaluator=policy_evaluator,
                agent_id=self._agent_id,
                org_id=self._org_id,
                framework_version=self.FRAMEWORK_VERSION,
            )


# ---------------------------------------------------------------------------
# C2: AnthropicTracedClient — synchronous proxy
# ---------------------------------------------------------------------------


class AnthropicTracedClient:
    """
    Transparent proxy around anthropic.Anthropic with Clyro tracing.  # Implements FRD-002

    Intercepts `client.messages` to provide a TracedMessages namespace.
    All other attributes are passed through to the underlying client.
    """

    _clyro_wrapped: bool = True

    def __init__(
        self,
        client: Any,
        config: ClyroConfig,
        transport: SyncTransport,
        policy_evaluator: PolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework_version: str | None,
    ):
        self._client = client
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework_version = framework_version
        self._session: Session | None = None
        self._closed = False
        self._background_sync_started = False

        # Create the traced messages namespace
        self._traced_messages = TracedMessages(
            original_messages=client.messages,
            config=config,
            transport=transport,
            policy_evaluator=policy_evaluator,
            agent_id=agent_id,
            org_id=org_id,
            framework_version=framework_version,
            get_session=self._ensure_session,
            buffer_event=self._buffer_event,
        )

        # Start background sync
        if not config.is_local_only():
            self._start_background_sync()

        # Register atexit handler so session_end + flush happen on process
        # exit even if the user never calls close() explicitly.
        atexit.register(self.close)

    def _start_background_sync(self) -> None:
        """Start background sync with fail-open behavior."""
        if self._background_sync_started:
            return
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.start_background_sync()
            self._background_sync_started = True
        except Exception as e:
            logger.warning("background_sync_start_failed", error=str(e), fail_open=True)

    def _ensure_session(self) -> Session:
        """
        Lazily create and start a session on first use.  # Implements FRD-002

        Returns:
            The active session.
        """
        if self._session is not None and self._session.is_active:
            return self._session

        # Carry over step/cost counters and loop detector from a previous
        # (ended) session so that prevention stack limits accumulate across
        # session boundaries.
        prev_step = 0
        prev_cost = Decimal("0")
        prev_loop_detector = None
        if self._session is not None:
            prev_step = self._session._step_number
            prev_cost = self._session._cumulative_cost
            prev_loop_detector = self._session._loop_detector

        self._session = Session(
            config=self._config,
            agent_id=self._agent_id,
            org_id=self._org_id,
            framework=Framework.ANTHROPIC,
            framework_version=self._framework_version,
            agent_name=self._config.agent_name,
            policy_evaluator=self._policy_evaluator,
        )
        self._session._event_sink = self._buffer_event
        self._session._step_number = prev_step
        self._session._cumulative_cost = prev_cost
        if prev_loop_detector is not None:
            self._session._loop_detector = prev_loop_detector
        start_event = self._session.start()
        if start_event is not None:
            self._buffer_event(start_event)
        return self._session

    def _buffer_event(self, event: TraceEvent) -> None:
        """Buffer an event to transport with fail-open behavior."""
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.buffer_event(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    @property
    def messages(self) -> TracedMessages:
        """Instrumented messages namespace.  # Implements FRD-003, FRD-004"""
        return self._traced_messages

    def close(self) -> None:
        """
        Flush buffered events, end session, stop background sync.  # Implements FRD-002

        Idempotent — second call is a no-op.
        Each phase is isolated so a transport teardown failure cannot
        prevent session_end from being emitted and flushed.
        """
        if self._closed:
            return
        self._closed = True

        # Phase 1: Emit session_end event
        try:
            if self._session is not None and self._session.is_active:
                end_event = self._session.end()
                if end_event is not None:
                    self._buffer_event(end_event)
        except Exception as e:
            logger.warning("session_end_failed", error=str(e))

        # Phase 2: Flush buffered events to backend
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.flush()
        except Exception as e:
            logger.warning("close_flush_failed", error=str(e))

        # Phase 3: Stop background sync thread (may fail at atexit
        # due to event loop teardown — that's OK, events are already flushed)
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.close()
        except Exception as e:
            logger.debug("transport_close_failed", error=str(e))

    def __enter__(self) -> AnthropicTracedClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        """Pass-through to the underlying Anthropic client.  # Implements FRD-002"""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Shared mixin for prevention stack, loop detection, and data builders
# ---------------------------------------------------------------------------


class _TracedMessagesBase:
    """
    Mixin providing shared logic for sync and async TracedMessages.

    Eliminates duplication of _check_prevention_stack, _check_loop,
    _build_input_data, and _build_output_data between sync/async classes.
    """

    _config: ClyroConfig
    _pending_tool_events: dict[str, TraceEvent]

    def _backfill_tool_results(self, kwargs: dict[str, Any]) -> None:
        """Scan input messages for tool_result blocks and backfill output_data on pending tool_call events."""
        if not self._pending_tool_events:
            return
        messages = kwargs.get("messages", [])
        for msg in messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if tool_use_id and tool_use_id in self._pending_tool_events:
                    event = self._pending_tool_events.pop(tool_use_id)
                    result_content = block.get("content")
                    event.output_data = {"result": result_content}

    def _check_prevention_stack(self, session: Session, kwargs: dict[str, Any]) -> None:
        """
        Pre-call prevention stack checks.  # Implements FRD-008, FRD-009, FRD-010

        Raises appropriate exceptions before the API call if limits are exceeded.
        """
        # Step limit  # Implements FRD-008
        session._step_number += 1
        if self._config.controls.enable_step_limit:
            if session.step_number > self._config.controls.max_steps:
                raise StepLimitExceededError(
                    limit=self._config.controls.max_steps,
                    current_step=session.step_number,
                    session_id=str(session.session_id),
                )

        # Cost limit  # Implements FRD-009
        if self._config.controls.enable_cost_limit:
            if float(session.cumulative_cost) >= self._config.controls.max_cost_usd:
                raise CostLimitExceededError(
                    limit_usd=self._config.controls.max_cost_usd,
                    current_cost_usd=float(session.cumulative_cost),
                    session_id=str(session.session_id),
                    step_number=session.step_number,
                )

        # Loop detection via Session's LoopDetector  # Implements FRD-010
        if self._config.controls.enable_loop_detection:
            state = self._build_loop_state(kwargs)
            if state is not None:
                session._check_loop_detection(state, action="messages.create")

    def _build_loop_state(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Build a state dict from kwargs for loop detection."""
        try:
            return {
                "model": kwargs.get("model", ""),
                "messages": self._safe_serialize(kwargs.get("messages", [])),
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_serialize(obj: Any) -> Any:
        """Recursively convert objects to JSON-safe primitives.

        Handles Anthropic SDK objects (ContentBlock, ToolUseBlock, etc.)
        that have model_dump() or __dict__, as well as plain dicts and lists.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: _TracedMessagesBase._safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_TracedMessagesBase._safe_serialize(item) for item in obj]
        # Pydantic models (Anthropic SDK uses these for ContentBlock, etc.)
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        # Fallback: try dict conversion, then str
        if hasattr(obj, "__dict__"):
            try:
                return {
                    k: _TracedMessagesBase._safe_serialize(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception:
                pass
        return str(obj)

    def _extract_last_user_message(self, kwargs: dict[str, Any]) -> str | None:
        """Extract the last user message text from kwargs for policy evaluation."""
        try:
            messages = kwargs.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    # Handle list-of-blocks format
                    if isinstance(content, list):
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                texts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                texts.append(block)
                        return " ".join(texts) if texts else None
        except Exception:
            pass  # Best-effort — don't break on message parsing
        return None

    def _build_input_data(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build input_data dict from kwargs for trace events."""
        return {
            "model": kwargs.get("model"),
            "messages": self._safe_serialize(kwargs.get("messages")),
            "system": kwargs.get("system"),
            "tools": kwargs.get("tools"),
            "max_tokens": kwargs.get("max_tokens"),
        }

    def _build_output_data(self, response: Any) -> dict[str, Any] | None:
        """Build output_data dict from Anthropic response."""
        try:
            content = getattr(response, "content", None)
            stop_reason = getattr(response, "stop_reason", None)
            model = getattr(response, "model", None)

            # Serialize content blocks
            serialized_content = None
            if content is not None:
                serialized_content = []
                for block in content:
                    block_dict: dict[str, Any] = {"type": getattr(block, "type", "unknown")}
                    if hasattr(block, "text"):
                        block_dict["text"] = block.text
                    if hasattr(block, "name"):
                        block_dict["name"] = block.name
                    if hasattr(block, "input"):
                        block_dict["input"] = block.input
                    if hasattr(block, "id"):
                        block_dict["id"] = block.id
                    serialized_content.append(block_dict)

            return {
                "content": serialized_content,
                "stop_reason": stop_reason,
                "model": model,
            }
        except Exception as e:
            logger.warning("output_serialization_failed", error=str(e))
            return None


# ---------------------------------------------------------------------------
# C3: TracedMessages — synchronous message interception
# ---------------------------------------------------------------------------


class TracedMessages(_TracedMessagesBase):
    """
    Proxy around client.messages with tracing and enforcement.

    Intercepts create() and stream() to emit LLM_CALL and TOOL_CALL events,
    enforce the Prevention Stack, and evaluate policies on tool use.
    """

    def __init__(
        self,
        original_messages: Any,
        config: ClyroConfig,
        transport: SyncTransport | Transport,
        policy_evaluator: PolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework_version: str | None,
        get_session: Any,  # Callable[[], Session]
        buffer_event: Any,  # Callable[[TraceEvent], None]
    ):
        self._original_messages = original_messages
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework_version = framework_version
        self._get_session = get_session
        self._buffer_event = buffer_event

        # C6: Event hierarchy tracking  # Implements FRD-011
        self._last_llm_event_id: UUID | None = None

        # Track pending tool_call events by tool_use_id for result backfill
        self._pending_tool_events: dict[str, TraceEvent] = {}

        # Token extractor and cost calculator
        self._token_extractor = AnthropicTokenExtractor()
        self._cost_calculator: CostCalculator | None = None

    def _get_cost_calculator(self) -> CostCalculator:
        """Lazy init cost calculator."""
        if self._cost_calculator is None:
            self._cost_calculator = CostCalculator(self._config)
        return self._cost_calculator

    def create(self, **kwargs: Any) -> Any:
        """
        Traced messages.create() — emits LLM_CALL, TOOL_CALL events.  # Implements FRD-003

        Workflow (TDD §5.2):
        1. Ensure session initialized (lazy start)
        2. Pre-call enforcement (step/cost/loop)
        3. Call original messages.create()
        4. Extract tokens → calculate cost
        5. Emit LLM_CALL event
        6. Detect tool_use → emit TOOL_CALL events
        7. Evaluate policies on tool_use
        8. Auto-flush when conversation turn is complete (stop_reason != tool_use)
        9. Return response
        """
        session = self._get_session()

        # Backfill tool results from previous turn before processing this call
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        # Pre-LLM policy check — evaluate user input for text-based rules
        try:
            self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError as e:
            self._emit_error_event(session, e, kwargs, 0)
            self._auto_flush(session)
            raise

        # Pre-call enforcement  # Implements FRD-008, FRD-009, FRD-010
        try:
            self._check_prevention_stack(session, kwargs)
        except ExecutionControlError as e:
            # Emit error event, end session, and flush before re-raising
            # so that all previously buffered events are not lost.
            self._emit_error_event(session, e, kwargs, 0)
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)

        start_time = time.perf_counter()

        try:
            response = self._original_messages.create(**kwargs)
        except Exception as e:
            # Emit ERROR event, re-raise  # Implements FRD-003 failure condition
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._emit_error_event(session, e, kwargs, duration_ms)
            raise

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Post-call processing (fail-open — Clyro errors must not block agent)
        llm_event_id = None
        try:
            llm_event = self._process_response(session, response, kwargs, duration_ms)
            llm_event_id = llm_event.event_id
        except Exception:
            logger.warning("clyro_process_response_failed", fail_open=True)

        # Tool use detection and policy enforcement  # Implements FRD-005, FRD-006
        try:
            self._process_tool_use(session, response, llm_event_id)
        except PolicyViolationError as e:
            self._emit_error_event(session, e, kwargs, duration_ms)
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_process_tool_use_failed", fail_open=True)

        # Emit a STEP event for each create() call
        stop_reason = getattr(response, "stop_reason", None)
        self._emit_step_event(session, stop_reason, duration_ms, llm_event_id)

        # Auto-flush: when stop_reason is not tool_use the agent's turn is
        # complete — end the session and flush buffered events so traces
        # reach the backend without requiring the user to call close().
        if stop_reason != "tool_use":
            self._auto_flush(session)

        return response

    def _emit_step_event(
        self,
        session: Session,
        stop_reason: str | None,
        duration_ms: int,
        parent_event_id: UUID | None,
    ) -> None:
        """Emit a STEP event for the current create()/stream() call."""
        try:
            stage = AgentStage.ACT if stop_reason == "tool_use" else AgentStage.THINK
            event = create_step_event(
                session_id=session.session_id,
                step_number=session.step_number,
                event_name=f"anthropic_step_{session.step_number}",
                agent_id=self._agent_id,
                agent_stage=stage,
                duration_ms=duration_ms,
                cumulative_cost=session.cumulative_cost,
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=parent_event_id,
            )
            if event is not None:
                session._events.append(event)
                self._buffer_event(event)
        except Exception as e:
            logger.warning("step_event_failed", error=str(e), fail_open=True)

    def _auto_flush(self, session: Session) -> None:
        """End the session and flush all buffered events eagerly.

        Called when stop_reason != 'tool_use' — the agent's conversation turn
        is complete. We emit session_end and flush here (during normal execution)
        rather than deferring to close()/atexit, because SyncTransport.flush()
        fails at interpreter shutdown. Step/cost counters are carried over
        when _ensure_session creates the next session.
        """
        try:
            if session.is_active:
                end_event = session.end()
                if end_event is not None:
                    self._buffer_event(end_event)
        except Exception as e:
            logger.warning("auto_flush_session_end_failed", error=str(e), fail_open=True)
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.flush()
        except Exception as e:
            logger.warning("auto_flush_failed", error=str(e), fail_open=True)

    def stream(self, **kwargs: Any) -> TracedMessageStream:
        """
        Traced messages.stream() — wraps stream context manager.  # Implements FRD-004

        Returns a TracedMessageStream that emits events on completion.
        """
        session = self._get_session()

        # Backfill tool results from previous turn before processing this call
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        # Pre-LLM policy check — evaluate user input for text-based rules
        try:
            self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError as e:
            self._emit_error_event(session, e, kwargs, 0)
            self._auto_flush(session)
            raise

        # Pre-call enforcement  # Implements FRD-008, FRD-009, FRD-010
        try:
            self._check_prevention_stack(session, kwargs)
        except ExecutionControlError as e:
            self._emit_error_event(session, e, kwargs, 0)
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)

        start_time = time.perf_counter()

        # Get the original stream context manager
        original_stream = self._original_messages.stream(**kwargs)

        return TracedMessageStream(
            original_stream=original_stream,
            session=session,
            kwargs=kwargs,
            start_time=start_time,
            traced_messages=self,
        )

    # _check_prevention_stack, _check_loop, _build_input_data, _build_output_data
    # inherited from _TracedMessagesBase

    def _process_response(
        self,
        session: Session,
        response: Any,
        kwargs: dict[str, Any],
        duration_ms: int,
    ) -> TraceEvent:
        """
        Extract tokens, calculate cost, and emit LLM_CALL event.

        # Implements FRD-003, FRD-007
        """
        # Extract tokens  # Implements FRD-007
        input_tokens = 0
        output_tokens = 0
        cost_usd = Decimal("0")

        try:
            if self._token_extractor.can_extract(response):
                token_usage = self._token_extractor.extract(response)
                if token_usage is not None:
                    input_tokens = token_usage.input_tokens
                    output_tokens = token_usage.output_tokens
                    calculator = self._get_cost_calculator()
                    model = kwargs.get("model") or getattr(response, "model", None) or ""
                    cost_usd = calculator.calculate(input_tokens, output_tokens, model)
            else:
                logger.warning(
                    "token_extraction_no_usage",
                    session_id=str(session.session_id),
                )
        except Exception as e:
            # Token extraction failure — cost_usd remains 0  # Implements FRD-007 failure
            logger.warning(
                "token_extraction_failed",
                error=str(e),
                session_id=str(session.session_id),
            )

        # Update session cost (fail-open)
        try:
            session._cumulative_cost += cost_usd
        except Exception:
            logger.warning(
                "clyro_cost_update_failed", session_id=str(session.session_id), fail_open=True
            )

        # Build event data
        model = kwargs.get("model", "")
        input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
        output_data = self._build_output_data(response) if self._config.capture_outputs else None

        event_id = uuid4()
        llm_event = create_llm_call_event(
            session_id=session.session_id,
            step_number=session.step_number,
            model=model,
            input_data=input_data or {"model": model},
            output_data=output_data,
            agent_id=self._agent_id,
            token_count_input=input_tokens,
            token_count_output=output_tokens,
            cost_usd=cost_usd,
            cumulative_cost=session.cumulative_cost,
            duration_ms=duration_ms,
            agent_stage=AgentStage.THINK,
            framework=Framework.ANTHROPIC,
            framework_version=self._framework_version,
            event_id=event_id,
            parent_event_id=self._last_llm_event_id,  # Implements FRD-011
        )

        if llm_event is not None:
            session._events.append(llm_event)
            self._buffer_event(llm_event)

        # Update hierarchy tracker  # Implements FRD-011
        self._last_llm_event_id = event_id

        return llm_event

    def _process_tool_use(
        self,
        session: Session,
        response: Any,
        llm_event_id: UUID | None,
    ) -> None:
        """
        Detect tool_use content blocks and emit TOOL_CALL events.

        # Implements FRD-005, FRD-006
        """
        content = getattr(response, "content", None)
        if not content:
            return

        tool_use_blocks = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                tool_use_blocks.append(block)

        if not tool_use_blocks:
            return

        for block in tool_use_blocks:
            # Extract tool data with graceful handling  # Implements FRD-005 failure
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None) or {}
            tool_use_id = getattr(block, "id", None)

            if tool_name is None:
                logger.warning(
                    "tool_use_block_missing_name",
                    session_id=str(session.session_id),
                    tool_use_id=tool_use_id,
                )
                tool_name = "unknown"

            # Emit TOOL_CALL event  # Implements FRD-005
            tool_event = create_tool_call_event(
                session_id=session.session_id,
                step_number=session.step_number,
                tool_name=tool_name,
                input_data=tool_input
                if isinstance(tool_input, dict)
                else {"input": str(tool_input)},
                agent_id=self._agent_id,
                agent_stage=AgentStage.ACT,
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=llm_event_id,  # Implements FRD-011
                metadata={"tool_use_id": tool_use_id} if tool_use_id else {},
                cumulative_cost=session.cumulative_cost,
            )
            if tool_event is not None:
                session._events.append(tool_event)
                self._buffer_event(tool_event)

            # Track for tool result backfill
            if tool_use_id:
                self._pending_tool_events[tool_use_id] = tool_event

        # Policy enforcement on tool use  # Implements FRD-006
        for block in tool_use_blocks:
            self._evaluate_tool_policy(session, block, llm_event_id)

    def _evaluate_llm_policy(self, session: Session, kwargs: dict[str, Any]) -> None:
        """
        Pre-LLM policy check — evaluate user input for text-based rules.

        Uses session.check_policy() (same as LangGraph/CrewAI) which handles
        event draining via the finally block and _event_sink pattern.

        Raises:
            PolicyViolationError: If the LLM call is blocked by a policy
        """
        llm_policy_params: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "cost": float(session.cumulative_cost),
            "step_number": session.step_number,
        }
        user_input = self._extract_last_user_message(kwargs)
        if user_input is not None:
            llm_policy_params["input"] = user_input

        try:
            session.check_policy(
                "llm_call",
                llm_policy_params,
                cumulative_cost=session.cumulative_cost,
            )
        except PolicyViolationError:
            raise
        except Exception as e:
            # Non-policy error (network, timeout, etc.) — respect fail_open
            if self._config.fail_open:
                logger.warning("llm_policy_evaluation_failed", error=str(e), fail_open=True)
            else:
                raise PolicyViolationError(
                    rule_id="policy_unavailable",
                    rule_name="Policy Unavailable",
                    message=f"Policy evaluation failed: {e}",
                    action_type="llm_call",
                ) from e

    def _evaluate_tool_policy(
        self, session: Session, block: Any, llm_event_id: UUID | None = None
    ) -> None:
        """
        Evaluate policies on a tool_use block.  # Implements FRD-006

        Uses session.check_policy() (same as LangGraph/CrewAI) which handles
        event draining via the finally block and _event_sink pattern.

        Raises:
            PolicyViolationError: If the tool is blocked by a policy
        """
        tool_name = getattr(block, "name", "unknown")
        tool_input = getattr(block, "input", {}) or {}

        # Flatten tool input for policy matching
        parameters: dict[str, Any] = {"tool_name": tool_name}
        if isinstance(tool_input, dict):
            parameters.update(tool_input)

        try:
            session.check_policy(
                "tool_call",
                parameters,
                parent_event_id=llm_event_id,
                cumulative_cost=session.cumulative_cost,
            )
        except PolicyViolationError:
            raise
        except Exception as e:
            # Non-policy error (network, timeout, etc.) — respect fail_open
            if self._config.fail_open:
                logger.warning(
                    "policy_evaluation_failed", error=str(e), tool_name=tool_name, fail_open=True
                )
            else:
                raise PolicyViolationError(
                    rule_id="policy_unavailable",
                    rule_name="Policy Unavailable",
                    message=f"Policy evaluation failed: {e}",
                    action_type="tool_call",
                ) from e

    def _emit_error_event(
        self,
        session: Session,
        error: Exception,
        kwargs: dict[str, Any],
        duration_ms: int,
    ) -> None:
        """Emit an ERROR trace event for a failed API call or enforcement check."""
        try:
            model = kwargs.get("model", "")
            input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
            error_event = create_error_event(
                session_id=session.session_id,
                step_number=session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._agent_id,
                error_stack=traceback.format_exc(),
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=self._last_llm_event_id,
                cumulative_cost=session.cumulative_cost,
                input_data=input_data,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={"model": model, "duration_ms": duration_ms},
            )
            self._buffer_event(error_event)
        except Exception as emit_err:
            logger.warning("error_event_emission_failed", error=str(emit_err))

    # _build_input_data, _build_output_data inherited from _TracedMessagesBase

    def __getattr__(self, name: str) -> Any:
        """Pass-through to original messages namespace.  # Implements FRD-002"""
        return getattr(self._original_messages, name)


# ---------------------------------------------------------------------------
# C5: TracedMessageStream — sync stream wrapper
# ---------------------------------------------------------------------------


class TracedMessageStream:
    """
    Context manager wrapping Anthropic's stream for tracing.  # Implements FRD-004

    Emits LLM_CALL and TOOL_CALL events upon stream completion.
    """

    def __init__(
        self,
        original_stream: Any,
        session: Session,
        kwargs: dict[str, Any],
        start_time: float,
        traced_messages: TracedMessages,
    ):
        self._original_stream = original_stream
        self._session = session
        self._kwargs = kwargs
        self._start_time = start_time
        self._traced_messages = traced_messages
        self._entered_stream: Any = None

    def __enter__(self) -> Any:
        """Enter the original stream context manager."""
        self._entered_stream = self._original_stream.__enter__()
        return self._entered_stream

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        """
        Exit stream — emit events on completion or error.  # Implements FRD-004
        """
        duration_ms = int((time.perf_counter() - self._start_time) * 1000)

        if exc_type is not None:
            # Stream interrupted  # Implements FRD-004 failure condition
            self._traced_messages._emit_error_event(
                self._session,
                exc_val if exc_val is not None else Exception(str(exc_type)),
                self._kwargs,
                duration_ms,
            )
            # Propagate to original stream's __exit__
            return self._original_stream.__exit__(exc_type, exc_val, exc_tb)

        try:
            # Get final message from stream
            final_message = None
            if self._entered_stream is not None and hasattr(
                self._entered_stream, "get_final_message"
            ):
                final_message = self._entered_stream.get_final_message()

            if final_message is not None:
                # Process response same as create() (fail-open)
                llm_event_id = None
                try:
                    llm_event = self._traced_messages._process_response(
                        self._session, final_message, self._kwargs, duration_ms
                    )
                    llm_event_id = llm_event.event_id
                except Exception:
                    logger.warning("clyro_stream_process_response_failed", fail_open=True)

                # Tool use detection — PolicyViolationError must propagate
                try:
                    self._traced_messages._process_tool_use(
                        self._session, final_message, llm_event_id
                    )
                except PolicyViolationError:
                    raise
                except Exception:
                    logger.warning("clyro_stream_process_tool_use_failed", fail_open=True)

                # Step event
                stop_reason = getattr(final_message, "stop_reason", None)
                self._traced_messages._emit_step_event(
                    self._session, stop_reason, duration_ms, llm_event_id
                )
                # Auto-flush when conversation turn is complete
                if stop_reason != "tool_use":
                    self._traced_messages._auto_flush(self._session)
            else:
                logger.warning(
                    "stream_no_final_message",
                    session_id=str(self._session.session_id),
                )
        except PolicyViolationError:
            # Re-raise policy violations through __exit__
            self._original_stream.__exit__(None, None, None)
            raise
        except Exception as e:
            logger.warning("stream_post_processing_failed", error=str(e))

        return self._original_stream.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# C4: AsyncAnthropicTracedClient — async proxy
# ---------------------------------------------------------------------------


class AsyncAnthropicTracedClient:
    """
    Transparent proxy around anthropic.AsyncAnthropic with Clyro tracing.

    # Implements FRD-002, FRD-012
    """

    _clyro_wrapped: bool = True

    def __init__(
        self,
        client: Any,
        config: ClyroConfig,
        transport: Transport,
        policy_evaluator: PolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework_version: str | None,
    ):
        self._client = client
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework_version = framework_version
        self._session: Session | None = None
        self._closed = False
        self._background_sync_started = False

        # Create the async traced messages namespace
        self._traced_messages = AsyncTracedMessages(
            original_messages=client.messages,
            config=config,
            transport=transport,
            policy_evaluator=policy_evaluator,
            agent_id=agent_id,
            org_id=org_id,
            framework_version=framework_version,
            get_session=self._ensure_session,
            buffer_event=self._buffer_event,
        )

    async def _start_background_sync(self) -> None:
        """Start background sync for async transport."""
        if self._background_sync_started:
            return
        try:
            if isinstance(self._transport, Transport):
                await self._transport.start_background_sync()
            self._background_sync_started = True
        except Exception as e:
            logger.warning("background_sync_start_failed", error=str(e), fail_open=True)

    async def _ensure_session(self) -> Session:
        """
        Lazily create and start a session on first use.  # Implements FRD-002, FRD-012
        """
        if not self._background_sync_started and not self._config.is_local_only():
            await self._start_background_sync()

        if self._session is not None and self._session.is_active:
            return self._session

        # Carry over step/cost counters and loop detector from a previous
        # (ended) session so that prevention stack limits accumulate across
        # session boundaries.
        prev_step = 0
        prev_cost = Decimal("0")
        prev_loop_detector = None
        if self._session is not None:
            prev_step = self._session._step_number
            prev_cost = self._session._cumulative_cost
            prev_loop_detector = self._session._loop_detector

        self._session = Session(
            config=self._config,
            agent_id=self._agent_id,
            org_id=self._org_id,
            framework=Framework.ANTHROPIC,
            framework_version=self._framework_version,
            agent_name=self._config.agent_name,
            policy_evaluator=self._policy_evaluator,
        )
        self._session._event_sink = self._buffer_event_sync
        self._session._step_number = prev_step
        self._session._cumulative_cost = prev_cost
        if prev_loop_detector is not None:
            self._session._loop_detector = prev_loop_detector
        start_event = self._session.start()
        if start_event is not None:
            await self._buffer_event(start_event)
        return self._session

    def _buffer_event_sync(self, event: TraceEvent) -> None:
        """Synchronous event sink callback for session policy events.

        Transport is async-only, so we append to the session's event list.
        Events are picked up by the async buffer_event path on the next call.
        """
        try:
            if self._session is not None:
                self._session._events.append(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    async def _buffer_event(self, event: TraceEvent) -> None:
        """Buffer an event to async transport."""
        try:
            if isinstance(self._transport, Transport):
                await self._transport.buffer_event(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    @property
    def messages(self) -> AsyncTracedMessages:
        """Instrumented messages namespace.  # Implements FRD-003, FRD-004, FRD-012"""
        return self._traced_messages

    async def close(self) -> None:
        """
        Flush buffered events, end session, stop background sync.  # Implements FRD-002

        Idempotent — second call is a no-op.
        """
        if self._closed:
            return
        self._closed = True

        # Phase 1: Emit session_end event
        try:
            if self._session is not None and self._session.is_active:
                end_event = self._session.end()
                if end_event is not None:
                    await self._buffer_event(end_event)
        except Exception as e:
            logger.warning("session_end_failed", error=str(e))

        # Phase 2: Flush buffered events to backend
        try:
            if isinstance(self._transport, Transport):
                await self._transport.flush()
        except Exception as e:
            logger.warning("close_flush_failed", error=str(e))

        # Phase 3: Close transport
        try:
            if isinstance(self._transport, Transport):
                await self._transport.close()
        except Exception as e:
            logger.debug("transport_close_failed", error=str(e))

    async def __aenter__(self) -> AsyncAnthropicTracedClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit — calls close()."""
        await self.close()

    def __getattr__(self, name: str) -> Any:
        """Pass-through to the underlying Anthropic client.  # Implements FRD-002"""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# C4: AsyncTracedMessages — async message interception
# ---------------------------------------------------------------------------


class AsyncTracedMessages(_TracedMessagesBase):
    """
    Async proxy around client.messages with tracing and enforcement.

    # Implements FRD-003, FRD-004, FRD-012
    """

    def __init__(
        self,
        original_messages: Any,
        config: ClyroConfig,
        transport: Transport,
        policy_evaluator: PolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework_version: str | None,
        get_session: Any,  # Callable[[], Awaitable[Session]]
        buffer_event: Any,  # Callable[[TraceEvent], Awaitable[None]]
    ):
        self._original_messages = original_messages
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework_version = framework_version
        self._get_session = get_session
        self._buffer_event = buffer_event

        # C6: Event hierarchy tracking  # Implements FRD-011
        self._last_llm_event_id: UUID | None = None

        # Track pending tool_call events by tool_use_id for result backfill
        self._pending_tool_events: dict[str, TraceEvent] = {}

        # Token extractor and cost calculator
        self._token_extractor = AnthropicTokenExtractor()
        self._cost_calculator: CostCalculator | None = None

    def _get_cost_calculator(self) -> CostCalculator:
        """Lazy init cost calculator."""
        if self._cost_calculator is None:
            self._cost_calculator = CostCalculator(self._config)
        return self._cost_calculator

    async def create(self, **kwargs: Any) -> Any:
        """
        Traced async messages.create().  # Implements FRD-003, FRD-012
        """
        session = await self._get_session()

        # Backfill tool results from previous turn
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        # Pre-LLM policy check — evaluate user input for text-based rules
        try:
            await self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError as e:
            await self._emit_error_event(session, e, kwargs, 0)
            await self._async_auto_flush(session)
            raise

        # Pre-call enforcement
        try:
            self._check_prevention_stack(session, kwargs)
        except ExecutionControlError as e:
            await self._emit_error_event(session, e, kwargs, 0)
            await self._async_auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)

        start_time = time.perf_counter()

        try:
            response = await self._original_messages.create(**kwargs)
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await self._emit_error_event(session, e, kwargs, duration_ms)
            raise

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Post-call processing (fail-open — Clyro errors must not block agent)
        llm_event_id = None
        try:
            llm_event = await self._process_response(session, response, kwargs, duration_ms)
            llm_event_id = llm_event.event_id
        except Exception:
            logger.warning("clyro_process_response_failed", fail_open=True)

        # Tool use detection and policy enforcement
        try:
            await self._process_tool_use(session, response, llm_event_id)
        except PolicyViolationError as e:
            await self._emit_error_event(session, e, kwargs, duration_ms)
            await self._async_auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_process_tool_use_failed", fail_open=True)

        # Step event
        stop_reason = getattr(response, "stop_reason", None)
        await self._async_emit_step_event(session, stop_reason, duration_ms, llm_event_id)

        # Auto-flush when conversation turn is complete
        if stop_reason != "tool_use":
            await self._async_auto_flush(session)

        return response

    async def _async_emit_step_event(
        self,
        session: Session,
        stop_reason: str | None,
        duration_ms: int,
        parent_event_id: UUID | None,
    ) -> None:
        """Emit a STEP event (async)."""
        try:
            stage = AgentStage.ACT if stop_reason == "tool_use" else AgentStage.THINK
            event = create_step_event(
                session_id=session.session_id,
                step_number=session.step_number,
                event_name=f"anthropic_step_{session.step_number}",
                agent_id=self._agent_id,
                agent_stage=stage,
                duration_ms=duration_ms,
                cumulative_cost=session.cumulative_cost,
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=parent_event_id,
            )
            if event is not None:
                session._events.append(event)
                await self._buffer_event(event)
        except Exception as e:
            logger.warning("step_event_failed", error=str(e), fail_open=True)

    async def _async_auto_flush(self, session: Session) -> None:
        """End the session and flush all buffered events eagerly (async).

        Called when stop_reason != 'tool_use' — the agent's conversation turn
        is complete. We emit session_end and flush here during normal execution.
        Step/cost counters are carried over when _ensure_session creates the
        next session.
        """
        try:
            if session.is_active:
                end_event = session.end()
                if end_event is not None:
                    await self._buffer_event(end_event)
        except Exception as e:
            logger.warning("auto_flush_session_end_failed", error=str(e), fail_open=True)
        try:
            if isinstance(self._transport, Transport):
                await self._transport.flush()
        except Exception as e:
            logger.warning("auto_flush_failed", error=str(e), fail_open=True)

    async def stream(self, **kwargs: Any) -> AsyncTracedMessageStream:
        """
        Traced async messages.stream().  # Implements FRD-004, FRD-012
        """
        session = await self._get_session()

        # Backfill tool results from previous turn
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        # Pre-LLM policy check — evaluate user input for text-based rules
        try:
            await self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError as e:
            await self._emit_error_event(session, e, kwargs, 0)
            await self._async_auto_flush(session)
            raise

        try:
            self._check_prevention_stack(session, kwargs)
        except ExecutionControlError as e:
            await self._emit_error_event(session, e, kwargs, 0)
            await self._async_auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)

        start_time = time.perf_counter()
        original_stream = self._original_messages.stream(**kwargs)

        return AsyncTracedMessageStream(
            original_stream=original_stream,
            session=session,
            kwargs=kwargs,
            start_time=start_time,
            traced_messages=self,
        )

    # _check_prevention_stack, _check_loop, _build_input_data, _build_output_data
    # inherited from _TracedMessagesBase

    async def _process_response(
        self,
        session: Session,
        response: Any,
        kwargs: dict[str, Any],
        duration_ms: int,
    ) -> TraceEvent:
        """Extract tokens, calculate cost, and emit LLM_CALL event (async).  # Implements FRD-003, FRD-007"""
        input_tokens = 0
        output_tokens = 0
        cost_usd = Decimal("0")

        try:
            if self._token_extractor.can_extract(response):
                token_usage = self._token_extractor.extract(response)
                if token_usage is not None:
                    input_tokens = token_usage.input_tokens
                    output_tokens = token_usage.output_tokens
                    calculator = self._get_cost_calculator()
                    model = kwargs.get("model") or getattr(response, "model", None) or ""
                    cost_usd = calculator.calculate(input_tokens, output_tokens, model)
            else:
                logger.warning("token_extraction_no_usage", session_id=str(session.session_id))
        except Exception as e:
            logger.warning(
                "token_extraction_failed", error=str(e), session_id=str(session.session_id)
            )

        try:
            session._cumulative_cost += cost_usd
        except Exception:
            logger.warning(
                "clyro_cost_update_failed", session_id=str(session.session_id), fail_open=True
            )

        model = kwargs.get("model", "")
        input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
        output_data = self._build_output_data(response) if self._config.capture_outputs else None

        event_id = uuid4()
        llm_event = create_llm_call_event(
            session_id=session.session_id,
            step_number=session.step_number,
            model=model,
            input_data=input_data or {"model": model},
            output_data=output_data,
            agent_id=self._agent_id,
            token_count_input=input_tokens,
            token_count_output=output_tokens,
            cost_usd=cost_usd,
            cumulative_cost=session.cumulative_cost,
            duration_ms=duration_ms,
            agent_stage=AgentStage.THINK,
            framework=Framework.ANTHROPIC,
            framework_version=self._framework_version,
            event_id=event_id,
            parent_event_id=self._last_llm_event_id,
        )

        if llm_event is not None:
            session._events.append(llm_event)
            await self._buffer_event(llm_event)
        self._last_llm_event_id = event_id

        return llm_event

    async def _process_tool_use(
        self,
        session: Session,
        response: Any,
        llm_event_id: UUID | None,
    ) -> None:
        """Detect tool_use and emit TOOL_CALL events (async).  # Implements FRD-005, FRD-006"""
        content = getattr(response, "content", None)
        if not content:
            return

        tool_use_blocks = [b for b in content if getattr(b, "type", None) == "tool_use"]
        if not tool_use_blocks:
            return

        for block in tool_use_blocks:
            tool_name = getattr(block, "name", None)
            tool_input = getattr(block, "input", None) or {}
            tool_use_id = getattr(block, "id", None)

            if tool_name is None:
                logger.warning(
                    "tool_use_block_missing_name",
                    session_id=str(session.session_id),
                    tool_use_id=tool_use_id,
                )
                tool_name = "unknown"

            tool_event = create_tool_call_event(
                session_id=session.session_id,
                step_number=session.step_number,
                tool_name=tool_name,
                input_data=tool_input
                if isinstance(tool_input, dict)
                else {"input": str(tool_input)},
                agent_id=self._agent_id,
                agent_stage=AgentStage.ACT,
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=llm_event_id,
                metadata={"tool_use_id": tool_use_id} if tool_use_id else {},
                cumulative_cost=session.cumulative_cost,
            )
            if tool_event is not None:
                session._events.append(tool_event)
                await self._buffer_event(tool_event)

            # Track for tool result backfill
            if tool_use_id:
                self._pending_tool_events[tool_use_id] = tool_event

        for block in tool_use_blocks:
            await self._evaluate_tool_policy(session, block, llm_event_id)

    async def _evaluate_llm_policy(self, session: Session, kwargs: dict[str, Any]) -> None:
        """
        Pre-LLM policy check — evaluate user input for text-based rules (async).

        Uses session.check_policy_async() (same as LangGraph/CrewAI) which handles
        event draining via the finally block and _event_sink pattern.

        Raises:
            PolicyViolationError: If the LLM call is blocked by a policy
        """
        llm_policy_params: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "cost": float(session.cumulative_cost),
            "step_number": session.step_number,
        }
        user_input = self._extract_last_user_message(kwargs)
        if user_input is not None:
            llm_policy_params["input"] = user_input

        try:
            await session.check_policy_async(
                "llm_call",
                llm_policy_params,
                cumulative_cost=session.cumulative_cost,
            )
        except PolicyViolationError:
            raise
        except Exception as e:
            if self._config.fail_open:
                logger.warning("llm_policy_evaluation_failed", error=str(e), fail_open=True)
            else:
                raise PolicyViolationError(
                    rule_id="policy_unavailable",
                    rule_name="Policy Unavailable",
                    message=f"Policy evaluation failed: {e}",
                    action_type="llm_call",
                ) from e

    async def _evaluate_tool_policy(
        self, session: Session, block: Any, llm_event_id: UUID | None = None
    ) -> None:
        """
        Evaluate policies on a tool_use block (async).  # Implements FRD-006

        Uses session.check_policy_async() (same as LangGraph/CrewAI) which handles
        event draining via the finally block and _event_sink pattern.

        Raises:
            PolicyViolationError: If the tool is blocked by a policy
        """
        tool_name = getattr(block, "name", "unknown")
        tool_input = getattr(block, "input", {}) or {}

        # Flatten tool input for policy matching
        parameters: dict[str, Any] = {"tool_name": tool_name}
        if isinstance(tool_input, dict):
            parameters.update(tool_input)

        try:
            await session.check_policy_async(
                "tool_call",
                parameters,
                parent_event_id=llm_event_id,
                cumulative_cost=session.cumulative_cost,
            )
        except PolicyViolationError:
            raise
        except Exception as e:
            if self._config.fail_open:
                logger.warning(
                    "policy_evaluation_failed", error=str(e), tool_name=tool_name, fail_open=True
                )
            else:
                raise PolicyViolationError(
                    rule_id="policy_unavailable",
                    rule_name="Policy Unavailable",
                    message=f"Policy evaluation failed: {e}",
                    action_type="tool_call",
                ) from e

    async def _emit_error_event(
        self,
        session: Session,
        error: Exception,
        kwargs: dict[str, Any],
        duration_ms: int,
    ) -> None:
        """Emit an ERROR trace event for a failed API call or enforcement check (async)."""
        try:
            model = kwargs.get("model", "")
            input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
            error_event = create_error_event(
                session_id=session.session_id,
                step_number=session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._agent_id,
                error_stack=traceback.format_exc(),
                framework=Framework.ANTHROPIC,
                framework_version=self._framework_version,
                parent_event_id=self._last_llm_event_id,
                cumulative_cost=session.cumulative_cost,
                input_data=input_data,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={"model": model, "duration_ms": duration_ms},
            )
            await self._buffer_event(error_event)
        except Exception as emit_err:
            logger.warning("error_event_emission_failed", error=str(emit_err))

    # _build_input_data, _build_output_data inherited from _TracedMessagesBase

    def __getattr__(self, name: str) -> Any:
        """Pass-through to original messages namespace."""
        return getattr(self._original_messages, name)


# ---------------------------------------------------------------------------
# C5: AsyncTracedMessageStream — async stream wrapper
# ---------------------------------------------------------------------------


class AsyncTracedMessageStream:
    """
    Async context manager wrapping Anthropic's stream for tracing.

    # Implements FRD-004, FRD-012
    """

    def __init__(
        self,
        original_stream: Any,
        session: Session,
        kwargs: dict[str, Any],
        start_time: float,
        traced_messages: AsyncTracedMessages,
    ):
        self._original_stream = original_stream
        self._session = session
        self._kwargs = kwargs
        self._start_time = start_time
        self._traced_messages = traced_messages
        self._entered_stream: Any = None

    async def __aenter__(self) -> Any:
        """Enter the original async stream context manager."""
        self._entered_stream = await self._original_stream.__aenter__()
        return self._entered_stream

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        """Exit async stream — emit events on completion or error.  # Implements FRD-004"""
        duration_ms = int((time.perf_counter() - self._start_time) * 1000)

        if exc_type is not None:
            await self._traced_messages._emit_error_event(
                self._session,
                exc_val if exc_val is not None else Exception(str(exc_type)),
                self._kwargs,
                duration_ms,
            )
            return await self._original_stream.__aexit__(exc_type, exc_val, exc_tb)

        try:
            final_message = None
            if self._entered_stream is not None and hasattr(
                self._entered_stream, "get_final_message"
            ):
                final_message = self._entered_stream.get_final_message()

            if final_message is not None:
                # Process response (fail-open)
                llm_event_id = None
                try:
                    llm_event = await self._traced_messages._process_response(
                        self._session, final_message, self._kwargs, duration_ms
                    )
                    llm_event_id = llm_event.event_id
                except Exception:
                    logger.warning("clyro_stream_process_response_failed", fail_open=True)

                # Tool use detection — PolicyViolationError must propagate
                try:
                    await self._traced_messages._process_tool_use(
                        self._session, final_message, llm_event_id
                    )
                except PolicyViolationError:
                    raise
                except Exception:
                    logger.warning("clyro_stream_process_tool_use_failed", fail_open=True)

                # Step event
                stop_reason = getattr(final_message, "stop_reason", None)
                await self._traced_messages._async_emit_step_event(
                    self._session, stop_reason, duration_ms, llm_event_id
                )
                # Auto-flush when conversation turn is complete
                if stop_reason != "tool_use":
                    await self._traced_messages._async_auto_flush(self._session)
            else:
                logger.warning("stream_no_final_message", session_id=str(self._session.session_id))
        except PolicyViolationError:
            await self._original_stream.__aexit__(None, None, None)
            raise
        except Exception as e:
            logger.warning("stream_post_processing_failed", error=str(e))

        return await self._original_stream.__aexit__(None, None, None)
