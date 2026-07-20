# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK OpenAI-Compatible Adapter
# Implements FRD-SDK-001 through FRD-SDK-006

"""
OpenAI-compatible SDK adapter for the Clyro SDK.

Provides transparent, passive tracing and governance for applications built on
the OpenAI Python SDK (`openai.OpenAI`). Because OpenRouter is consumed through
the same OpenAI client (just a custom ``base_url``), this adapter governs
OpenRouter-on-the-OpenAI-SDK traffic for free (FRD-SDK-001).

It intercepts ``client.chat.completions.create()`` to emit LLM_CALL / TOOL_CALL
trace events, evaluate Clyro's Prevention Stack on tool calls, and compute cost
(including cached-token discounts), for both buffered and streamed responses.

Architecture (mirrors the Anthropic adapter):
    User Code -> OpenAITracedClient (proxy)
        -> client.chat.completions.create()  (TracedCompletions)
            -> original client.chat.completions.create()
            -> token extraction + cost calculation (cached-aware)
            -> LLM_CALL event emission
            -> tool_calls detection -> TOOL_CALL events + Prevention Stack
        -> Return response (or a wrapped stream) to user

Scope (S2 / FRD-SDK-001..006): both the synchronous ``openai.OpenAI`` client and
the asynchronous ``openai.AsyncOpenAI`` client are traced. Sync clients use
``OpenAITracedClient`` (over ``SyncTransport``); async clients use
``AsyncOpenAITracedClient`` (over the async ``Transport``) — the async traced
completions reuse all of the sync adapter's pure logic (cost, token extraction,
tool parsing, prevention stack) via subclassing, overriding only the awaitable
I/O paths (mirrors the Anthropic adapter's sync/async split).

Note: the ``openai`` package is an optional/peer dependency. It is never imported
at module level — only the installed version is inspected when validating.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import time
import traceback
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import structlog

from clyro.cost import OpenAITokenExtractor
from clyro.exceptions import (
    ClyroWrapError,
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.local_policy import SDKLocalPolicyEvaluator
from clyro.policy import PolicyEvaluator
from clyro.session import Session
from clyro.trace import (
    AgentStage,
    Framework,
    TraceEvent,
    create_error_event,
    create_llm_call_event,
    create_tool_call_event,
)
from clyro.transport import SyncTransport, Transport

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.policy import ApprovalHandler

logger = structlog.get_logger(__name__)

# Minimum supported OpenAI SDK version. The `chat.completions` client surface
# this adapter intercepts is the 1.x SDK shape.
MIN_OPENAI_VERSION = "1.0.0"

# OpenAI prices cached input tokens at a discount vs. fresh input tokens. The FRD
# only requires that cached reads NOT be billed at the full input rate
# (FRD-SDK-005); 0.5x is OpenAI's documented cached-input rate and a safe default.
CACHED_INPUT_DISCOUNT = Decimal("0.5")


# ---------------------------------------------------------------------------
# Detection + version validation
# ---------------------------------------------------------------------------


def is_openai_agent(obj: Any) -> bool:
    """
    Detect whether an object is an OpenAI-compatible SDK client.  # Implements FRD-SDK-001

    Uses module/class-name inspection (not isinstance) so the `openai` package is
    not a hard dependency. OpenRouter clients are `openai.OpenAI` instances with a
    custom base_url, so they match here without special-casing.

    Returns:
        True for `openai.OpenAI` / `openai.AsyncOpenAI` instances.
    """
    try:
        obj_type = type(obj)
        module = getattr(obj_type, "__module__", "") or ""
        name = getattr(obj_type, "__name__", "") or ""
        return module.startswith("openai") and name in ("OpenAI", "AsyncOpenAI")
    except Exception:
        return False


def _base_url_host(client: Any) -> str:
    """Lowercased hostname of the client's base_url ('' when unset/unparseable)."""
    try:
        raw = str(getattr(client, "base_url", "") or "")
        return (urlsplit(raw).hostname or "").lower()
    except Exception:
        return ""


def _host_is(host: str, domain: str) -> bool:
    """True when host is exactly domain or a subdomain of it."""
    return host == domain or host.endswith("." + domain)


def _resolve_framework(client: Any) -> Framework:
    """OpenAI vs OpenRouter, by the client's base_url.  # Implements FRD-SDK-001

    OpenRouter is consumed through the OpenAI client with a custom base_url; when
    that URL points at OpenRouter we tag events/agent/logs as `openrouter` rather
    than `openai` so traffic is attributed to the right provider.
    """
    host = _base_url_host(client)
    return Framework.OPENROUTER if _host_is(host, "openrouter.ai") else Framework.OPENAI


def _supports_stream_usage_option(client: Any) -> bool:
    """Whether the endpoint is known to accept ``stream_options.include_usage``.

    Only genuine OpenAI (default / api.openai.com base_url) and OpenRouter are
    known to accept it; other OpenAI-compatible endpoints (Azure, vLLM, Groq, …)
    may reject the param, so we must NOT inject it there.
    """
    host = _base_url_host(client)
    return host == "" or _host_is(host, "openai.com") or _host_is(host, "openrouter.ai")


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '1.2.3' into a tuple of ints."""
    try:
        return tuple(int(p) for p in version_str.split(".")[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def detect_openai_version() -> str | None:
    """Detect the installed OpenAI SDK version without validating."""
    try:
        import openai

        return getattr(openai, "__version__", None)
    except ImportError:
        return None


def validate_openai_version() -> str:
    """
    Validate that the installed OpenAI SDK version is supported.  # Implements FRD-SDK-001

    Raises:
        FrameworkVersionError: If the SDK is missing or below the minimum.
    """
    version = detect_openai_version()
    if version is None:
        raise FrameworkVersionError(
            framework="openai",
            version="not installed",
            supported=f">={MIN_OPENAI_VERSION}",
        )
    if version == "unknown":
        logger.warning(
            "openai_version_unknown",
            message="Could not determine OpenAI SDK version, assuming compatible",
        )
        return version
    if _parse_version(version) < _parse_version(MIN_OPENAI_VERSION):
        raise FrameworkVersionError(
            framework="openai",
            version=version,
            supported=f">={MIN_OPENAI_VERSION}",
        )
    return version


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert objects (incl. pydantic models) to JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return {k: _safe_serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    return str(obj)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an object attribute or a dict, returning default if absent."""
    if obj is None:
        return default
    val = getattr(obj, key, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(key)
    return default if val is None else val


def _extract_cached_tokens(usage: Any) -> int:
    """Extract OpenAI cached-input token count from a usage object/dict (FRD-SDK-005)."""
    details = _get(usage, "prompt_tokens_details")
    if details is None:
        return 0
    cached = _get(details, "cached_tokens", 0)
    try:
        return max(0, int(cached or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Factory adapter
# ---------------------------------------------------------------------------


class OpenAIAdapter:
    """OpenAI-compatible adapter factory.  # Implements FRD-SDK-001"""

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
        if getattr(client, "_clyro_wrapped", False):
            raise ClyroWrapError(
                message="Client is already wrapped by Clyro.",
                agent_type=type(client).__name__,
            )
        self._client = client
        self._config = config
        self._agent_id = agent_id
        self._org_id = org_id
        self._approval_handler = approval_handler
        self._is_async = type(client).__name__ == "AsyncOpenAI"
        # openai vs openrouter, by base_url (FRD-SDK-001)
        self._framework = _resolve_framework(client)
        # only inject stream_options on endpoints known to accept it (OpenAI/OpenRouter)
        self._inject_stream_usage = _supports_stream_usage_option(client)

        if validate_version:
            self.FRAMEWORK_VERSION = validate_openai_version()
        else:
            self.FRAMEWORK_VERSION = detect_openai_version() or "unknown"

    @property
    def agent(self) -> Any:
        return self._client

    @property
    def name(self) -> str:
        return self._framework.value  # "openai" or "openrouter"

    @property
    def framework(self) -> Framework:
        return self._framework

    @property
    def framework_version(self) -> str | None:
        return self.FRAMEWORK_VERSION

    def _build_policy_evaluator(self) -> PolicyEvaluator | SDKLocalPolicyEvaluator | None:
        """Build the policy evaluator, supporting cloud and local modes.

        Policy enforcement works in BOTH modes when enabled: cloud uses the
        backend PolicyEvaluator (needs an api_key); local mode uses the YAML-based
        SDKLocalPolicyEvaluator — so enforcement applies even with no api_key.
        Both expose evaluate_sync/evaluate_async, so the same evaluator serves the
        sync and async traced clients.
        """
        if not self._config.controls.enable_policy_enforcement:
            return None
        if self._config.mode == "local":
            return SDKLocalPolicyEvaluator(approval_handler=self._approval_handler)
        if self._config.api_key:
            return PolicyEvaluator(
                config=self._config,
                agent_id=self._agent_id,
                org_id=self._org_id,
                approval_handler=self._approval_handler,
            )
        return None

    def create_traced_client(self) -> OpenAITracedClient | AsyncOpenAITracedClient:
        """
        Create a traced client proxy.  # Implements FRD-SDK-001, FRD-SDK-002

        Async clients (openai.AsyncOpenAI) get an AsyncOpenAITracedClient over the
        async Transport; sync clients get an OpenAITracedClient over SyncTransport.
        """
        policy_evaluator = self._build_policy_evaluator()

        if self._is_async:
            return AsyncOpenAITracedClient(
                client=self._client,
                config=self._config,
                transport=Transport(self._config),
                policy_evaluator=policy_evaluator,
                agent_id=self._agent_id,
                org_id=self._org_id,
                framework=self._framework,
                framework_version=self.FRAMEWORK_VERSION,
                inject_stream_usage=self._inject_stream_usage,
            )

        return OpenAITracedClient(
            client=self._client,
            config=self._config,
            transport=SyncTransport(self._config),
            policy_evaluator=policy_evaluator,
            agent_id=self._agent_id,
            org_id=self._org_id,
            framework=self._framework,
            framework_version=self.FRAMEWORK_VERSION,
            inject_stream_usage=self._inject_stream_usage,
        )


# ---------------------------------------------------------------------------
# Traced client proxy (sync)
# ---------------------------------------------------------------------------


class _TracedChat:
    """Proxy for `client.chat` that swaps in the traced `.completions` namespace."""

    def __init__(self, original_chat: Any, traced_completions: TracedCompletions):
        self._original_chat = original_chat
        self._traced_completions = traced_completions

    @property
    def completions(self) -> TracedCompletions:
        return self._traced_completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_chat, name)


class OpenAITracedClient:
    """
    Transparent proxy around openai.OpenAI with Clyro tracing.  # Implements FRD-SDK-002

    Intercepts `client.chat.completions`; everything else passes through.
    """

    _clyro_wrapped: bool = True

    def __init__(
        self,
        client: Any,
        config: ClyroConfig,
        transport: SyncTransport,
        policy_evaluator: PolicyEvaluator | SDKLocalPolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework: Framework,
        framework_version: str | None,
        inject_stream_usage: bool = False,
    ):
        self._client = client
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework = framework
        self._framework_version = framework_version
        self._session: Session | None = None
        self._closed = False

        traced_completions = TracedCompletions(
            original_completions=client.chat.completions,
            config=config,
            transport=transport,
            policy_evaluator=policy_evaluator,
            agent_id=agent_id,
            org_id=org_id,
            framework=framework,
            framework_version=framework_version,
            get_session=self._ensure_session,
            buffer_event=self._buffer_event,
            inject_stream_usage=inject_stream_usage,
        )
        self._traced_chat = _TracedChat(client.chat, traced_completions)

        # Pin ONE event loop for this client's whole lifetime.
        #
        # SyncTransport._get_loop() creates a BRAND-NEW event loop on every call
        # (asyncio.get_running_loop() always raises at sync top-level, so it never
        # reuses a stopped loop despite its docstring). But the async Transport's
        # httpx client and asyncio locks are loop-affine: they bind to the first
        # loop, so the next buffer/flush op runs on a different loop and raises
        # "<...> is bound to a different event loop". In cloud mode that error is
        # swallowed fail-open and the events are dropped — so traces never reach the
        # backend and no agent appears. Pinning one loop keeps buffer/flush/send/close
        # on the same loop -> reliable, immediate delivery via _auto_flush().
        self._owned_loop: asyncio.AbstractEventLoop | None = None
        if isinstance(transport, SyncTransport):
            self._owned_loop = asyncio.new_event_loop()
            transport._loop = self._owned_loop
            transport._get_loop = self._pinned_loop

        # Background sync — same as every other adapter (Anthropic / LangGraph / CrewAI /
        # Generic): in cloud mode start the periodic SyncWorker + one-shot pricing-catalog
        # pull. It runs on its own thread/loop (independent of the pinned loop above), so
        # the worker's periodic sends are a best-effort BACKUP; the pinned-loop _auto_flush()
        # is what actually delivers each turn. This restores parity (the sync_worker_started
        # log + pricing fetch) without reintroducing the dual-loop delivery failure.
        self._background_sync_started = False
        if not config.is_local_only():
            self._start_background_sync()
        atexit.register(self.close)

    def _pinned_loop(self) -> asyncio.AbstractEventLoop:
        """Return this client's single persistent loop (replaces SyncTransport._get_loop).

        Preserves the original safety check: a SyncTransport must never run inside an
        already-running loop (that would deadlock run_until_complete).
        """
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running.is_running():
            raise RuntimeError(
                "SyncTransport cannot be used inside a running event loop. "
                "Use the async client (AsyncOpenAI) instead."
            )
        return self._owned_loop  # type: ignore[return-value]

    def _start_background_sync(self) -> None:
        """Start the periodic SyncWorker + pricing pull (parity with the other adapters)."""
        if self._background_sync_started:
            return
        try:
            self._transport.start_background_sync()
            self._background_sync_started = True
        except Exception as e:  # noqa: BLE001
            logger.warning("background_sync_start_failed", error=str(e), fail_open=True)

    def _ensure_session(self) -> Session:
        """Lazily create/start a session on first use.  # Implements FRD-SDK-002"""
        if self._session is not None and self._session.is_active:
            return self._session

        # Carry over step/cost counters and the loop detector from a previous
        # (auto-flushed) session so execution-control limits accumulate across the
        # tool-loop turns of one logical agent run.
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
            framework=self._framework,
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
        """Buffer an event to transport with fail-open behavior.  # Implements NFR-SDK-002

        The event's framework (openai / openrouter) is sent as-is — the backend accepts
        these values, so there's no wire relabel. Buffering the original object (not a copy)
        lets tool-result backfill mutate the very event the transport will send.
        """
        try:
            self._transport.buffer_event(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    @property
    def chat(self) -> _TracedChat:
        """Instrumented chat namespace.  # Implements FRD-SDK-002, FRD-SDK-003"""
        return self._traced_chat

    def close(self) -> None:
        """Flush events, end session, stop background sync. Idempotent.  # Implements FRD-SDK-002"""
        if self._closed:
            return
        self._closed = True

        try:
            if self._session is not None and self._session.is_active:
                end_event = self._session.end()
                if end_event is not None:
                    self._buffer_event(end_event)
        except Exception as e:
            logger.warning("session_end_failed", error=str(e))

        try:
            self._transport.flush()
        except Exception as e:
            logger.warning("close_flush_failed", error=str(e))

        try:
            self._transport.close()
        except Exception as e:
            logger.debug("transport_close_failed", error=str(e))

        # Close the pinned loop last (after the transport has finished using it).
        if self._owned_loop is not None and not self._owned_loop.is_closed():
            try:
                self._owned_loop.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("owned_loop_close_failed", error=str(e))

    def __enter__(self) -> OpenAITracedClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        """Pass-through to the underlying OpenAI client.  # Implements FRD-SDK-002"""
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Traced completions — the interception point
# ---------------------------------------------------------------------------


class TracedCompletions:
    """
    Proxy around client.chat.completions with tracing and enforcement.

    Intercepts create() (buffered and streamed) to emit LLM_CALL and TOOL_CALL
    events, evaluate the Prevention Stack on tool calls, and compute cost.
    """

    def __init__(
        self,
        original_completions: Any,
        config: ClyroConfig,
        transport: SyncTransport,
        policy_evaluator: PolicyEvaluator | SDKLocalPolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework: Framework,
        framework_version: str | None,
        get_session: Any,  # Callable[[], Session]
        buffer_event: Any,  # Callable[[TraceEvent], None]
        inject_stream_usage: bool = False,
    ):
        self._original_completions = original_completions
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework = framework
        self._framework_version = framework_version
        self._inject_stream_usage = inject_stream_usage
        self._get_session = get_session
        self._buffer_event = buffer_event
        self._last_llm_event_id: UUID | None = None
        self._token_extractor = OpenAITokenExtractor()
        # TOOL_CALL events awaiting their result, keyed by tool_call_id. OpenAI returns
        # tool results as role:"tool" messages on the NEXT create(), so we backfill
        # them onto the event then (parity with the Anthropic adapter). FRD-SDK-003.
        self._pending_tool_events: dict[str, TraceEvent] = {}

    def _backfill_tool_results(self, kwargs: dict[str, Any]) -> None:
        """Attach tool results to the matching earlier TOOL_CALL events.

        Scans this call's input for role:"tool" messages and sets the pending
        TOOL_CALL event's ``output_data`` by matching ``tool_call_id``. Best-effort:
        only updates events still in-flight (the open tool-loop turn, before flush).
        The buffered event is the same object, so this also updates what the transport sends.
        """
        if not self._pending_tool_events:
            return
        for msg in kwargs.get("messages", []) or []:
            if not (isinstance(msg, dict) and msg.get("role") == "tool"):
                continue
            event = self._pending_tool_events.pop(msg.get("tool_call_id"), None)
            if event is not None:
                event.output_data = {"result": msg.get("content")}

    def create(self, **kwargs: Any) -> Any:
        """
        Traced chat.completions.create().  # Implements FRD-SDK-002

        Pre-call: input (llm_call) policy + execution controls (step/cost/loop).
        Buffered: call original, emit LLM_CALL, then TOOL_CALL + policy, flush turn.
        Streamed (stream=True): return a wrapped iterator that emits on completion.
        On API exception: emit ERROR, flush, and re-raise unchanged (FRD-SDK-002).
        """
        session = self._get_session()

        # Backfill prior tool results onto their TOOL_CALL events (fail-open).
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        # Pre-LLM input policy — evaluate prompt/input before the call.
        try:
            self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError:
            self._auto_flush(session)  # persist the POLICY_CHECK record before raising
            raise

        # Execution controls — step / cost / loop limits (increments the step).
        # llm_step is this call's own step; it is threaded through rather than
        # re-read from the session, which concurrent calls advance underneath us.
        llm_step = 0
        try:
            llm_step = self._check_prevention_stack(session, kwargs)
        except ExecutionControlError as e:
            self._emit_error_event(session, e, kwargs, 0)  # audit record of the limit hit
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)
            # Fail-open: the step was not allocated, so fall back to the session's
            # current step rather than emitting events numbered 0.
            llm_step = session.step_number

        if kwargs.get("stream"):
            return self._create_streaming(session, kwargs, llm_step)  # Implements FRD-SDK-006

        start_time = time.perf_counter()
        try:
            response = self._original_completions.create(**kwargs)
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._emit_error_event(session, e, kwargs, duration_ms)
            self._auto_flush(session)  # ensure the ERROR reaches the backend
            raise  # FRD-SDK-002: re-raise the original exception unchanged

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self._handle_response(session, response, kwargs, duration_ms, llm_step)
        return response

    # -- prevention stack + policy (pre-call) ----------------------------

    def _next_step(self, session: Session) -> int:
        """Allocate the next step number and enforce the step limit.  # Implements FRD-008

        Called once per LLM call (pre-call, from the prevention stack) and once
        per tool call, so a step means "one unit of agent work" — matching the
        event-based counting the other adapters get from Session.record_event().
        Counting only LLM calls made OpenAI runs report far fewer steps than the
        same workload on Anthropic/LangGraph/CrewAI/Claude Agent SDK.

        Returns the allocated step number. Callers MUST use the returned value
        rather than re-reading session.step_number later: concurrent calls on one
        client share the session, so the counter can advance before the caller
        emits its events.
        """
        controls = self._config.controls
        session._step_number += 1
        step = session.step_number
        if controls.enable_step_limit and step > controls.max_steps:
            raise StepLimitExceededError(
                limit=controls.max_steps,
                current_step=step,
                session_id=str(session.session_id),
            )
        return step

    def _check_prevention_stack(self, session: Session, kwargs: dict[str, Any]) -> int:
        """Enforce execution controls before the call — step/cost/loop limits.

        Parity with the other adapters' Prevention Stack: raises StepLimit/
        CostLimitExceededError (or a loop error) when a configured limit is hit.

        Returns the step number allocated to this LLM call.
        """
        controls = self._config.controls
        llm_step = self._next_step(session)
        if controls.enable_cost_limit and float(session.cumulative_cost) >= controls.max_cost_usd:
            raise CostLimitExceededError(
                limit_usd=controls.max_cost_usd,
                current_cost_usd=float(session.cumulative_cost),
                session_id=str(session.session_id),
                step_number=llm_step,
            )
        if controls.enable_loop_detection:
            state = self._build_loop_state(kwargs)
            if state is not None:
                session._check_loop_detection(state, action="chat.completions.create")
        return llm_step

    def _enforce_cost_limit_post_call(
        self, session: Session, kwargs: dict[str, Any], duration_ms: int
    ) -> None:
        """Block immediately if the just-recorded LLM cost crossed the cost limit.

        Records the ERROR audit event, flushes the turn, and raises
        CostLimitExceededError before this turn's tool calls are emitted/run.
        """
        controls = self._config.controls
        if not controls.enable_cost_limit or float(session.cumulative_cost) < controls.max_cost_usd:
            return
        err = CostLimitExceededError(
            limit_usd=controls.max_cost_usd,
            current_cost_usd=float(session.cumulative_cost),
            session_id=str(session.session_id),
            step_number=session.step_number,
        )
        self._emit_error_event(session, err, kwargs, duration_ms)
        self._auto_flush(session)
        raise err

    @staticmethod
    def _build_loop_state(kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Build a state dict from kwargs for loop detection."""
        try:
            return {
                "model": kwargs.get("model", ""),
                "messages": _safe_serialize(kwargs.get("messages", [])),
            }
        except (TypeError, ValueError):
            return None

    def _evaluate_llm_policy(self, session: Session, kwargs: dict[str, Any]) -> None:
        """Evaluate input (llm_call) policy before the API call.

        Block -> PolicyViolationError. Policy unreachable -> fail-open/closed per
        config.fail_open (same contract as the tool-call check).
        """
        params: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "cost": float(session.cumulative_cost),
            "step_number": session.step_number,
        }
        user_input = self._extract_last_user_message(kwargs)
        if user_input is not None:
            params["input"] = user_input
        try:
            session.check_policy("llm_call", params, cumulative_cost=session.cumulative_cost)
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

    @staticmethod
    def _extract_last_user_message(kwargs: dict[str, Any]) -> str | None:
        """Extract the last user message text from kwargs for policy evaluation."""
        try:
            for msg in reversed(kwargs.get("messages", []) or []):
                if not (isinstance(msg, dict) and msg.get("role") == "user"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    return " ".join(texts) if texts else None
        except Exception:
            pass  # best-effort — never break on message parsing
        return None

    def _auto_flush(self, session: Session) -> None:
        """End the session and flush buffered events eagerly.

        Called when a turn completes (and on block/error) rather than deferring to
        close()/atexit, because SyncTransport.flush() is unreliable at interpreter
        shutdown — so a script that never calls close() would otherwise lose its
        events. Step/cost/loop counters carry over via _ensure_session.
        """
        try:
            if session.is_active:
                end_event = session.end()
                if end_event is not None:
                    self._buffer_event(end_event)
        except Exception as e:
            logger.warning("auto_flush_session_end_failed", error=str(e), fail_open=True)
        try:
            self._transport.flush()
        except Exception as e:
            logger.warning("auto_flush_failed", error=str(e), fail_open=True)

    # -- buffered response handling --------------------------------------

    def _handle_response(
        self,
        session: Session,
        response: Any,
        kwargs: dict[str, Any],
        duration_ms: int,
        llm_step: int,
    ) -> None:
        llm_event_id = None
        try:
            input_tokens, output_tokens, cached = self._extract_tokens(response)
            model = kwargs.get("model") or _get(response, "model") or ""
            input_data = (
                self._build_input_data(kwargs) if self._config.capture_inputs else {"model": model}
            )
            output_data = (
                self._build_output_data(response) if self._config.capture_outputs else None
            )
            llm_event_id = self._emit_llm_call(
                session,
                model,
                input_tokens,
                output_tokens,
                cached,
                duration_ms,
                input_data,
                output_data,
                estimated=False,
                llm_step=llm_step,
            )
        except Exception:
            logger.warning("clyro_process_response_failed", fail_open=True)

        # Post-call cost enforcement: this LLM call's cost was just recorded. Cost is the
        # one control that can only be known AFTER the call returns, so if it pushed the
        # cumulative over the limit, stop NOW — before emitting/running this turn's tool
        # calls — instead of waiting for the next create(). (The pre-call check still
        # guards the next turn for the other controls.)
        self._enforce_cost_limit_post_call(session, kwargs, duration_ms)

        # Tool calls: evaluate policy BEFORE emitting each TOOL_CALL, so a blocked tool is
        # recorded as a POLICY_CHECK(block) and NOT as a tool_call that looks like it ran.
        # Tools requested before the blocked one are still emitted; the block stops the rest.
        # Implements FRD-SDK-003, FRD-SDK-004
        try:
            tools = [self._parse_tool_call(tc) for tc in self._extract_tool_calls(response)]
            for name, arguments, tool_id in tools:
                self._evaluate_tool_policy(
                    session, name, arguments, llm_event_id
                )  # may block -> raise
                self._emit_tool_call(session, name, arguments, tool_id, llm_event_id)
        except PolicyViolationError:
            self._auto_flush(session)  # persist the POLICY_CHECK record before raising
            raise
        except ExecutionControlError as e:
            # A per-tool step allocation hit the step limit — propagate (like a
            # policy block) instead of being swallowed by the fail-open below.
            # Emit the ERROR audit record first, matching the pre-call limit path.
            self._emit_error_event(session, e, kwargs, duration_ms)
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_process_tool_calls_failed", fail_open=True)

        # Auto-flush when the agent's turn is complete (not mid tool-loop), so
        # events reach the backend without relying on close()/atexit.
        if self._first_finish_reason(response) != "tool_calls":
            self._auto_flush(session)

    # -- streamed response handling (FRD-SDK-006) ------------------------

    def _create_streaming(self, session: Session, kwargs: dict[str, Any], llm_step: int) -> Any:
        # Ask the provider to include usage in the final chunk so streamed cost is
        # real, not estimated (NFR-SDK-006) — but only on endpoints known to accept
        # it (OpenAI/OpenRouter) and only when the caller didn't set stream_options.
        injected_usage = False
        if "stream_options" not in kwargs and self._inject_stream_usage:
            kwargs = {**kwargs, "stream_options": {"include_usage": True}}
            injected_usage = True

        start_time = time.perf_counter()
        try:
            stream = self._original_completions.create(**kwargs)
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._emit_error_event(session, e, kwargs, duration_ms)
            raise
        return self._wrap_stream(session, stream, kwargs, start_time, injected_usage, llm_step)

    def _wrap_stream(
        self,
        session: Session,
        stream: Any,
        kwargs: dict[str, Any],
        start_time: float,
        injected_usage: bool,
        llm_step: int,
    ) -> Any:
        """Yield chunks to the caller, accumulate, then emit on completion (FRD-SDK-006)."""
        model = kwargs.get("model") or ""
        usage: Any = None
        finish_reason: str | None = None
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        try:
            for chunk in stream:
                skip_yield = False
                try:
                    chunk_usage = _get(chunk, "usage")
                    if chunk_usage is not None:
                        usage = chunk_usage
                    chunk_model = _get(chunk, "model")
                    if chunk_model:
                        model = chunk_model
                    chunk_finish = self._first_finish_reason(chunk)
                    if chunk_finish is not None:
                        finish_reason = chunk_finish
                    self._accumulate_chunk(chunk, content_parts, tool_acc)
                    # Transparency: if WE injected include_usage, the provider emits a
                    # trailing usage-only chunk (empty choices). Swallow it so the
                    # caller's iteration is exactly what it would be without Clyro.
                    if injected_usage and chunk_usage is not None and not _get(chunk, "choices"):
                        skip_yield = True
                except Exception:
                    logger.warning("clyro_stream_accumulate_failed", fail_open=True)
                if not skip_yield:
                    yield chunk
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            self._emit_error_event(session, e, kwargs, duration_ms)
            self._auto_flush(session)  # ensure the ERROR reaches the backend
            raise

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        self._finalize_stream(
            session, model, usage, finish_reason, content_parts, tool_acc, kwargs, duration_ms, llm_step
        )

    @staticmethod
    def _accumulate_chunk(
        chunk: Any, content_parts: list[str], tool_acc: dict[int, dict[str, Any]]
    ) -> None:
        """Accumulate streamed content + assemble tool-call deltas by index (Edge Case)."""
        choices = _get(chunk, "choices")
        if not choices:
            return
        delta = _get(choices[0], "delta")
        if delta is None:
            return
        content = _get(delta, "content")
        if content:
            content_parts.append(content)
        for tc in _get(delta, "tool_calls", []) or []:
            idx = _get(tc, "index", 0) or 0
            entry = tool_acc.setdefault(idx, {"id": None, "name": None, "args": ""})
            tc_id = _get(tc, "id")
            if tc_id:
                entry["id"] = tc_id
            func = _get(tc, "function")
            if func is not None:
                fname = _get(func, "name")
                if fname:
                    entry["name"] = fname
                fargs = _get(func, "arguments")
                if fargs:
                    entry["args"] += fargs

    def _finalize_stream(
        self,
        session: Session,
        model: str,
        usage: Any,
        finish_reason: str | None,
        content_parts: list[str],
        tool_acc: dict[int, dict[str, Any]],
        kwargs: dict[str, Any],
        duration_ms: int,
        llm_step: int,
    ) -> None:
        llm_event_id = None
        try:
            text = "".join(content_parts)
            cached = 0
            estimated = False
            if usage is not None:
                token_usage = self._token_extractor.extract({"usage": usage, "model": model})
                input_tokens = token_usage.input_tokens if token_usage else 0
                output_tokens = token_usage.output_tokens if token_usage else 0
                cached = _extract_cached_tokens(usage)
            else:
                # FRD-SDK-006: usage absent -> estimate, never zero, flag estimated (VR-S3)
                estimated = True
                input_tokens = self._estimate_input_tokens(kwargs)
                output_tokens = max(1, len(text) // 4)

            input_data = (
                self._build_input_data(kwargs) if self._config.capture_inputs else {"model": model}
            )
            output_data = {"content": text} if self._config.capture_outputs else None
            llm_event_id = self._emit_llm_call(
                session,
                model,
                input_tokens,
                output_tokens,
                cached,
                duration_ms,
                input_data,
                output_data,
                estimated=estimated,
                llm_step=llm_step,
            )
        except Exception:
            logger.warning("clyro_stream_finalize_failed", fail_open=True)

        # Assembled tool calls: emit ALL first, then evaluate policy. The whole
        # block is fail-open — a Clyro-internal error must NOT propagate into the
        # caller's stream iteration (NFR-SDK-002).  # Implements FRD-SDK-003, FRD-SDK-004
        try:
            tools = [
                (tool_acc[i]["name"], self._parse_arguments(tool_acc[i]["args"]), tool_acc[i]["id"])
                for i in sorted(tool_acc)
            ]
            for name, arguments, tool_id in tools:
                self._emit_tool_call(session, name, arguments, tool_id, llm_event_id)
            for name, arguments, _ in tools:
                self._evaluate_tool_policy(session, name, arguments, llm_event_id)
        except PolicyViolationError:
            self._auto_flush(session)
            raise
        except ExecutionControlError as e:
            self._emit_error_event(session, e, kwargs, duration_ms)
            self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_stream_tool_processing_failed", fail_open=True)

        # Auto-flush when the turn is complete (not mid tool-loop).
        if finish_reason != "tool_calls":
            self._auto_flush(session)

    @staticmethod
    def _estimate_input_tokens(kwargs: dict[str, Any]) -> int:
        """Rough input-token estimate from message text (~4 chars/token)."""
        try:
            total = 0
            for msg in kwargs.get("messages", []) or []:
                content = msg.get("content") if isinstance(msg, dict) else None
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            total += len(str(block.get("text", "")))
            return max(1, total // 4)
        except Exception:
            return 1

    # -- token / cost ----------------------------------------------------

    def _extract_tokens(self, response: Any) -> tuple[int, int, int]:
        """Return (input_tokens, output_tokens, cached_tokens) from a response."""
        token_usage = None
        if self._token_extractor.can_extract(response):
            token_usage = self._token_extractor.extract(response)
        input_tokens = token_usage.input_tokens if token_usage else 0
        output_tokens = token_usage.output_tokens if token_usage else 0
        cached = _extract_cached_tokens(_get(response, "usage"))
        return input_tokens, output_tokens, cached

    def _compute_cost(
        self, input_tokens: int, output_tokens: int, cached_tokens: int, model: str
    ) -> Decimal:
        """
        Cost in USD, pricing cached input tokens at a discount.  # Implements FRD-SDK-005

        Cached reads are billed at CACHED_INPUT_DISCOUNT x the input rate, never the
        full rate. With cached_tokens == 0 this equals the standard calculation.
        """
        if input_tokens == 0 and output_tokens == 0:
            return Decimal("0")
        input_price, output_price = self._config.get_model_pricing(model or "unknown")
        cached = max(0, min(cached_tokens, input_tokens))
        uncached = input_tokens - cached
        input_cost = (Decimal(uncached) * input_price) / Decimal("1000")
        cached_cost = (Decimal(cached) * input_price * CACHED_INPUT_DISCOUNT) / Decimal("1000")
        output_cost = (Decimal(output_tokens) * output_price) / Decimal("1000")
        return input_cost + cached_cost + output_cost

    # -- event emission --------------------------------------------------

    def _emit_llm_call(
        self,
        session: Session,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        duration_ms: int,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        *,
        estimated: bool,
        llm_step: int,
    ) -> UUID:
        """Compute cost, emit an LLM_CALL event, return its id.  # Implements FRD-SDK-002, FRD-SDK-005

        llm_step is this call's own step, threaded from the prevention stack —
        session.step_number may have advanced (tool steps, concurrent calls).
        """
        cost_usd = self._compute_cost(input_tokens, output_tokens, cached_tokens, model)
        session._cumulative_cost += cost_usd
        event_id = uuid4()
        metadata: dict[str, Any] = {"model": model}
        if cached_tokens:
            metadata["cached_tokens"] = cached_tokens
        if estimated:
            metadata["estimated"] = True  # VR-S3
        event = create_llm_call_event(
            session_id=session.session_id,
            step_number=llm_step,
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
            framework=self._framework,
            framework_version=self._framework_version,
            event_id=event_id,
            parent_event_id=self._last_llm_event_id,
            metadata=metadata,
        )
        if event is not None:
            session._events.append(event)
            self._buffer_event(event)
        self._last_llm_event_id = event_id
        return event_id

    def _emit_tool_call(
        self,
        session: Session,
        tool_name: str | None,
        arguments: Any,
        tool_id: str | None,
        llm_event_id: UUID | None,
    ) -> None:
        """Emit one TOOL_CALL event (no policy — see _evaluate_tool_policy).  # Implements FRD-SDK-003

        Each tool call is its own step (FRD-008), so may raise StepLimitExceededError.
        """
        self._next_step(session)
        tool_step = session.step_number
        input_data = arguments if isinstance(arguments, dict) else {"arguments": arguments}
        event = create_tool_call_event(
            session_id=session.session_id,
            step_number=tool_step,
            tool_name=tool_name or "unknown",
            input_data=input_data,
            agent_id=self._agent_id,
            agent_stage=AgentStage.ACT,
            framework=self._framework,
            framework_version=self._framework_version,
            parent_event_id=llm_event_id,  # VR-S2: link to parent LLM_CALL
            cumulative_cost=session.cumulative_cost,
            metadata={"tool_call_id": tool_id} if tool_id else {},
        )
        if event is not None:
            session._events.append(event)
            self._buffer_event(event)
            if tool_id:  # remember it so its result can be backfilled next turn
                self._pending_tool_events[tool_id] = event

    def _evaluate_tool_policy(
        self, session: Session, tool_name: str | None, arguments: Any, llm_event_id: UUID | None
    ) -> None:
        """
        Evaluate the Prevention Stack on a tool call before returning.  # Implements FRD-SDK-004

        Block -> PolicyViolationError (a POLICY_CHECK event is emitted by the
        evaluator). Policy unreachable -> fail-open (proceed+warn) or fail-closed
        (block), per config.fail_open.
        """
        parameters: dict[str, Any] = {"tool_name": tool_name or "unknown"}
        if isinstance(arguments, dict):
            parameters.update(arguments)
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
        self, session: Session, error: Exception, kwargs: dict[str, Any], duration_ms: int
    ) -> None:
        """Emit an ERROR trace event for a failed API call.  # Implements FRD-SDK-002"""
        try:
            model = kwargs.get("model", "")
            input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
            event = create_error_event(
                session_id=session.session_id,
                step_number=session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._agent_id,
                error_stack=traceback.format_exc(),
                framework=self._framework,
                framework_version=self._framework_version,
                parent_event_id=self._last_llm_event_id,
                cumulative_cost=session.cumulative_cost,
                input_data=input_data,
                output_data={"error_type": type(error).__name__, "error_message": str(error)},
                metadata={"model": model, "duration_ms": duration_ms},
            )
            if event is not None:
                session._events.append(event)
                self._buffer_event(event)
        except Exception as emit_err:
            logger.warning("error_event_emission_failed", error=str(emit_err))

    # -- response parsing ------------------------------------------------

    @staticmethod
    def _first_finish_reason(response: Any) -> str | None:
        """Return choices[0].finish_reason (e.g. 'stop' | 'tool_calls'), or None."""
        choices = _get(response, "choices")
        if not choices:
            return None
        return _get(choices[0], "finish_reason")

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[Any]:
        """Return the tool_calls list from `response.choices[0].message` (or [])."""
        choices = _get(response, "choices")
        if not choices:
            return []
        message = _get(choices[0], "message")
        if message is None:
            return []
        return _get(message, "tool_calls", []) or []

    @staticmethod
    def _parse_arguments(raw_args: Any) -> Any:
        """Parse an OpenAI tool-call arguments JSON string into a dict (best effort)."""
        if isinstance(raw_args, str):
            if not raw_args:
                return {}
            try:
                return json.loads(raw_args)
            except (ValueError, TypeError):
                return {"arguments": raw_args}
        return raw_args if raw_args is not None else {}

    def _parse_tool_call(self, tool_call: Any) -> tuple[str | None, Any, str | None]:
        """
        Extract (name, arguments, id) from a tool_call entry.  # Implements FRD-SDK-003

        Malformed entries return available data (never skip/raise) — the caller
        still emits a TOOL_CALL with what was parsed.
        """
        func = _get(tool_call, "function")
        name = _get(func, "name")
        arguments = self._parse_arguments(_get(func, "arguments"))
        tool_id = _get(tool_call, "id")
        return name, arguments, tool_id

    # -- data builders ---------------------------------------------------

    @staticmethod
    def _build_input_data(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build input_data from create() kwargs."""
        return {
            "model": kwargs.get("model"),
            "messages": _safe_serialize(kwargs.get("messages")),
            "tools": kwargs.get("tools"),
            "max_tokens": kwargs.get("max_tokens"),
        }

    @staticmethod
    def _build_output_data(response: Any) -> dict[str, Any] | None:
        """Build output_data from an OpenAI chat-completion response."""
        try:
            choices = _get(response, "choices")
            serialized = None
            if choices:
                serialized = []
                for choice in choices:
                    message = _get(choice, "message")
                    serialized.append(
                        {
                            "content": _get(message, "content"),
                            "finish_reason": _get(choice, "finish_reason"),
                        }
                    )
            return {"choices": serialized, "model": _get(response, "model")}
        except Exception as e:
            logger.warning("output_serialization_failed", error=str(e))
            return None


# ---------------------------------------------------------------------------
# Async traced client proxy (openai.AsyncOpenAI)
# ---------------------------------------------------------------------------


class _AsyncTracedChat:
    """Proxy for `client.chat` that swaps in the async traced `.completions`."""

    def __init__(self, original_chat: Any, traced_completions: AsyncTracedCompletions):
        self._original_chat = original_chat
        self._traced_completions = traced_completions

    @property
    def completions(self) -> AsyncTracedCompletions:
        return self._traced_completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_chat, name)


class AsyncOpenAITracedClient:
    """
    Transparent proxy around openai.AsyncOpenAI with Clyro tracing.  # Implements FRD-SDK-002

    Async counterpart of OpenAITracedClient. Uses the async Transport, an async
    session/buffer path, and the AsyncTracedCompletions namespace; everything
    else passes through. Mirrors AsyncAnthropicTracedClient.
    """

    _clyro_wrapped: bool = True

    def __init__(
        self,
        client: Any,
        config: ClyroConfig,
        transport: Transport,
        policy_evaluator: PolicyEvaluator | SDKLocalPolicyEvaluator | None,
        agent_id: UUID | None,
        org_id: UUID | None,
        framework: Framework,
        framework_version: str | None,
        inject_stream_usage: bool = False,
    ):
        self._client = client
        self._config = config
        self._transport = transport
        self._policy_evaluator = policy_evaluator
        self._agent_id = agent_id
        self._org_id = org_id
        self._framework = framework
        self._framework_version = framework_version
        self._session: Session | None = None
        self._closed = False
        self._background_sync_started = False

        traced_completions = AsyncTracedCompletions(
            original_completions=client.chat.completions,
            config=config,
            transport=transport,
            policy_evaluator=policy_evaluator,
            agent_id=agent_id,
            org_id=org_id,
            framework=framework,
            framework_version=framework_version,
            get_session=self._ensure_session,
            buffer_event=self._buffer_event,
            inject_stream_usage=inject_stream_usage,
        )
        self._traced_chat = _AsyncTracedChat(client.chat, traced_completions)

    async def _start_background_sync(self) -> None:
        if self._background_sync_started:
            return
        try:
            await self._transport.start_background_sync()
            self._background_sync_started = True
        except Exception as e:
            logger.warning("background_sync_start_failed", error=str(e), fail_open=True)

    async def _ensure_session(self) -> Session:
        """Lazily create/start a session on first use (async).  # Implements FRD-SDK-002"""
        if not self._background_sync_started and not self._config.is_local_only():
            await self._start_background_sync()

        if self._session is not None and self._session.is_active:
            return self._session

        # Carry over step/cost counters and the loop detector from a previous
        # (auto-flushed) session so execution-control limits accumulate across the
        # tool-loop turns of one logical agent run.
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
            framework=self._framework,
            framework_version=self._framework_version,
            agent_name=self._config.agent_name,
            policy_evaluator=self._policy_evaluator,
        )
        # Transport is async-only; the session's policy-event sink must be sync,
        # so it appends to the session's event list (mirrors the async Anthropic
        # client). The async emit paths buffer LLM/TOOL/ERROR events directly.
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
        """Sync event sink for session policy events (appends to the session)."""
        try:
            if self._session is not None:
                self._session._events.append(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    async def _buffer_event(self, event: TraceEvent) -> None:
        """Buffer an event to the async transport with fail-open behavior.  # Implements NFR-SDK-002"""
        try:
            await self._transport.buffer_event(event)
        except Exception as e:
            if self._config.fail_open:
                logger.warning("event_buffer_failed", error=str(e), fail_open=True)
            else:
                raise

    @property
    def chat(self) -> _AsyncTracedChat:
        """Instrumented chat namespace.  # Implements FRD-SDK-002, FRD-SDK-003"""
        return self._traced_chat

    async def close(self) -> None:
        """Flush events, end session, close transport. Idempotent.  # Implements FRD-SDK-002"""
        if self._closed:
            return
        self._closed = True

        try:
            if self._session is not None and self._session.is_active:
                end_event = self._session.end()
                if end_event is not None:
                    await self._buffer_event(end_event)
        except Exception as e:
            logger.warning("session_end_failed", error=str(e))

        try:
            await self._transport.flush()
        except Exception as e:
            logger.warning("close_flush_failed", error=str(e))

        try:
            await self._transport.close()
        except Exception as e:
            logger.debug("transport_close_failed", error=str(e))

    async def __aenter__(self) -> AsyncOpenAITracedClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __getattr__(self, name: str) -> Any:
        """Pass-through to the underlying OpenAI client.  # Implements FRD-SDK-002"""
        return getattr(self._client, name)


class AsyncTracedCompletions(TracedCompletions):
    """
    Async proxy around client.chat.completions with tracing and enforcement.

    Subclasses the sync TracedCompletions to reuse all of its pure logic
    (cost computation, token extraction, tool parsing, prevention stack, loop
    state, data builders, chunk accumulation) and overrides only the awaitable
    I/O paths — create(), the streaming wrapper, event emission, policy
    evaluation, and auto-flush.
    """

    async def create(self, **kwargs: Any) -> Any:
        """
        Traced async chat.completions.create().  # Implements FRD-SDK-002

        Same contract as the sync create(): pre-call input policy + execution
        controls, then buffered or streamed handling, ERROR + re-raise on failure.
        """
        session = await self._get_session()

        # Backfill prior tool results onto their TOOL_CALL events (fail-open).
        try:
            self._backfill_tool_results(kwargs)
        except Exception:
            logger.warning("clyro_backfill_tool_results_failed", fail_open=True)

        try:
            await self._evaluate_llm_policy(session, kwargs)
        except PolicyViolationError:
            await self._auto_flush(session)
            raise

        llm_step = 0
        try:
            llm_step = self._check_prevention_stack(session, kwargs)  # pure — inherited
        except ExecutionControlError as e:
            await self._emit_error_event(session, e, kwargs, 0)
            await self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_prevention_stack_failed", fail_open=True)
            # Fail-open: the step was not allocated, so fall back to the session's
            # current step rather than emitting events numbered 0.
            llm_step = session.step_number

        if kwargs.get("stream"):
            return await self._create_streaming(session, kwargs, llm_step)  # Implements FRD-SDK-006

        start_time = time.perf_counter()
        try:
            response = await self._original_completions.create(**kwargs)
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await self._emit_error_event(session, e, kwargs, duration_ms)
            await self._auto_flush(session)
            raise  # FRD-SDK-002: re-raise the original exception unchanged

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        await self._handle_response(session, response, kwargs, duration_ms, llm_step)
        return response

    # -- policy + auto-flush (async overrides) ---------------------------

    async def _evaluate_llm_policy(self, session: Session, kwargs: dict[str, Any]) -> None:
        """Evaluate input (llm_call) policy before the API call (async)."""
        params: dict[str, Any] = {
            "model": kwargs.get("model", ""),
            "cost": float(session.cumulative_cost),
            "step_number": session.step_number,
        }
        user_input = self._extract_last_user_message(kwargs)  # pure — inherited
        if user_input is not None:
            params["input"] = user_input
        try:
            await session.check_policy_async(
                "llm_call", params, cumulative_cost=session.cumulative_cost
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

    async def _auto_flush(self, session: Session) -> None:
        """End the session and flush buffered events eagerly (async)."""
        try:
            if session.is_active:
                end_event = session.end()
                if end_event is not None:
                    await self._buffer_event(end_event)
        except Exception as e:
            logger.warning("auto_flush_session_end_failed", error=str(e), fail_open=True)
        try:
            await self._transport.flush()
        except Exception as e:
            logger.warning("auto_flush_failed", error=str(e), fail_open=True)

    # -- buffered response handling (async) ------------------------------

    async def _handle_response(
        self,
        session: Session,
        response: Any,
        kwargs: dict[str, Any],
        duration_ms: int,
        llm_step: int,
    ) -> None:
        llm_event_id = None
        try:
            input_tokens, output_tokens, cached = self._extract_tokens(response)  # pure
            model = kwargs.get("model") or _get(response, "model") or ""
            input_data = (
                self._build_input_data(kwargs) if self._config.capture_inputs else {"model": model}
            )
            output_data = (
                self._build_output_data(response) if self._config.capture_outputs else None
            )
            llm_event_id = await self._emit_llm_call(
                session,
                model,
                input_tokens,
                output_tokens,
                cached,
                duration_ms,
                input_data,
                output_data,
                estimated=False,
                llm_step=llm_step,
            )
        except Exception:
            logger.warning("clyro_process_response_failed", fail_open=True)

        # Tool calls: emit ALL events first, then evaluate policy.  # Implements FRD-SDK-003, FRD-SDK-004
        try:
            tools = [self._parse_tool_call(tc) for tc in self._extract_tool_calls(response)]
            for name, arguments, tool_id in tools:
                await self._emit_tool_call(session, name, arguments, tool_id, llm_event_id)
            for name, arguments, _ in tools:
                await self._evaluate_tool_policy(session, name, arguments, llm_event_id)
        except PolicyViolationError:
            await self._auto_flush(session)
            raise
        except ExecutionControlError as e:
            await self._emit_error_event(session, e, kwargs, duration_ms)
            await self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_process_tool_calls_failed", fail_open=True)

        if self._first_finish_reason(response) != "tool_calls":
            await self._auto_flush(session)

    # -- streamed response handling (async, FRD-SDK-006) ----------------

    async def _create_streaming(self, session: Session, kwargs: dict[str, Any], llm_step: int) -> Any:
        injected_usage = False
        if "stream_options" not in kwargs and self._inject_stream_usage:
            kwargs = {**kwargs, "stream_options": {"include_usage": True}}
            injected_usage = True

        start_time = time.perf_counter()
        try:
            stream = await self._original_completions.create(**kwargs)
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await self._emit_error_event(session, e, kwargs, duration_ms)
            raise
        return self._wrap_stream(session, stream, kwargs, start_time, injected_usage, llm_step)

    async def _wrap_stream(
        self,
        session: Session,
        stream: Any,
        kwargs: dict[str, Any],
        start_time: float,
        injected_usage: bool,
        llm_step: int,
    ) -> Any:
        """Async-generator: yield chunks, accumulate, emit on completion (FRD-SDK-006)."""
        model = kwargs.get("model") or ""
        usage: Any = None
        finish_reason: str | None = None
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        try:
            async for chunk in stream:
                skip_yield = False
                try:
                    chunk_usage = _get(chunk, "usage")
                    if chunk_usage is not None:
                        usage = chunk_usage
                    chunk_model = _get(chunk, "model")
                    if chunk_model:
                        model = chunk_model
                    chunk_finish = self._first_finish_reason(chunk)
                    if chunk_finish is not None:
                        finish_reason = chunk_finish
                    self._accumulate_chunk(chunk, content_parts, tool_acc)  # pure
                    # Transparency: swallow the trailing usage-only chunk we induced.
                    if injected_usage and chunk_usage is not None and not _get(chunk, "choices"):
                        skip_yield = True
                except Exception:
                    logger.warning("clyro_stream_accumulate_failed", fail_open=True)
                if not skip_yield:
                    yield chunk
        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            await self._emit_error_event(session, e, kwargs, duration_ms)
            await self._auto_flush(session)
            raise

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        await self._finalize_stream(
            session, model, usage, finish_reason, content_parts, tool_acc, kwargs, duration_ms, llm_step
        )

    async def _finalize_stream(
        self,
        session: Session,
        model: str,
        usage: Any,
        finish_reason: str | None,
        content_parts: list[str],
        tool_acc: dict[int, dict[str, Any]],
        kwargs: dict[str, Any],
        duration_ms: int,
        llm_step: int,
    ) -> None:
        llm_event_id = None
        try:
            text = "".join(content_parts)
            cached = 0
            estimated = False
            if usage is not None:
                token_usage = self._token_extractor.extract({"usage": usage, "model": model})
                input_tokens = token_usage.input_tokens if token_usage else 0
                output_tokens = token_usage.output_tokens if token_usage else 0
                cached = _extract_cached_tokens(usage)
            else:
                # FRD-SDK-006: usage absent -> estimate, never zero, flag estimated (VR-S3)
                estimated = True
                input_tokens = self._estimate_input_tokens(kwargs)
                output_tokens = max(1, len(text) // 4)

            input_data = (
                self._build_input_data(kwargs) if self._config.capture_inputs else {"model": model}
            )
            output_data = {"content": text} if self._config.capture_outputs else None
            llm_event_id = await self._emit_llm_call(
                session,
                model,
                input_tokens,
                output_tokens,
                cached,
                duration_ms,
                input_data,
                output_data,
                estimated=estimated,
                llm_step=llm_step,
            )
        except Exception:
            logger.warning("clyro_stream_finalize_failed", fail_open=True)

        try:
            tools = [
                (tool_acc[i]["name"], self._parse_arguments(tool_acc[i]["args"]), tool_acc[i]["id"])
                for i in sorted(tool_acc)
            ]
            for name, arguments, tool_id in tools:
                await self._emit_tool_call(session, name, arguments, tool_id, llm_event_id)
            for name, arguments, _ in tools:
                await self._evaluate_tool_policy(session, name, arguments, llm_event_id)
        except PolicyViolationError:
            await self._auto_flush(session)
            raise
        except ExecutionControlError as e:
            await self._emit_error_event(session, e, kwargs, duration_ms)
            await self._auto_flush(session)
            raise
        except Exception:
            logger.warning("clyro_stream_tool_processing_failed", fail_open=True)

        if finish_reason != "tool_calls":
            await self._auto_flush(session)

    # -- event emission (async) ------------------------------------------

    async def _emit_llm_call(
        self,
        session: Session,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        duration_ms: int,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        *,
        estimated: bool,
        llm_step: int,
    ) -> UUID:
        """Compute cost, emit an LLM_CALL event, return its id (async).  # Implements FRD-SDK-002, FRD-SDK-005"""
        cost_usd = self._compute_cost(input_tokens, output_tokens, cached_tokens, model)  # pure
        session._cumulative_cost += cost_usd
        event_id = uuid4()
        metadata: dict[str, Any] = {"model": model}
        if cached_tokens:
            metadata["cached_tokens"] = cached_tokens
        if estimated:
            metadata["estimated"] = True  # VR-S3
        event = create_llm_call_event(
            session_id=session.session_id,
            step_number=llm_step,
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
            framework=self._framework,
            framework_version=self._framework_version,
            event_id=event_id,
            parent_event_id=self._last_llm_event_id,
            metadata=metadata,
        )
        if event is not None:
            session._events.append(event)
            await self._buffer_event(event)
        self._last_llm_event_id = event_id
        return event_id

    async def _emit_tool_call(
        self,
        session: Session,
        tool_name: str | None,
        arguments: Any,
        tool_id: str | None,
        llm_event_id: UUID | None,
    ) -> None:
        """Emit one TOOL_CALL event (async; no policy — see _evaluate_tool_policy).  # Implements FRD-SDK-003

        Each tool call is its own step (FRD-008), so may raise StepLimitExceededError.
        """
        self._next_step(session)
        tool_step = session.step_number
        input_data = arguments if isinstance(arguments, dict) else {"arguments": arguments}
        event = create_tool_call_event(
            session_id=session.session_id,
            step_number=tool_step,
            tool_name=tool_name or "unknown",
            input_data=input_data,
            agent_id=self._agent_id,
            agent_stage=AgentStage.ACT,
            framework=self._framework,
            framework_version=self._framework_version,
            parent_event_id=llm_event_id,  # VR-S2: link to parent LLM_CALL
            cumulative_cost=session.cumulative_cost,
            metadata={"tool_call_id": tool_id} if tool_id else {},
        )
        if event is not None:
            session._events.append(event)
            await self._buffer_event(event)
            if tool_id:  # remember it so its result can be backfilled next turn
                self._pending_tool_events[tool_id] = event

    async def _evaluate_tool_policy(
        self, session: Session, tool_name: str | None, arguments: Any, llm_event_id: UUID | None
    ) -> None:
        """Evaluate the Prevention Stack on a tool call before returning (async).  # Implements FRD-SDK-004"""
        parameters: dict[str, Any] = {"tool_name": tool_name or "unknown"}
        if isinstance(arguments, dict):
            parameters.update(arguments)
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
        self, session: Session, error: Exception, kwargs: dict[str, Any], duration_ms: int
    ) -> None:
        """Emit an ERROR trace event for a failed API call (async).  # Implements FRD-SDK-002"""
        try:
            model = kwargs.get("model", "")
            input_data = self._build_input_data(kwargs) if self._config.capture_inputs else None
            event = create_error_event(
                session_id=session.session_id,
                step_number=session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._agent_id,
                error_stack=traceback.format_exc(),
                framework=self._framework,
                framework_version=self._framework_version,
                parent_event_id=self._last_llm_event_id,
                cumulative_cost=session.cumulative_cost,
                input_data=input_data,
                output_data={"error_type": type(error).__name__, "error_message": str(error)},
                metadata={"model": model, "duration_ms": duration_ms},
            )
            if event is not None:
                session._events.append(event)
                await self._buffer_event(event)
        except Exception as emit_err:
            logger.warning("error_event_emission_failed", error=str(emit_err))
