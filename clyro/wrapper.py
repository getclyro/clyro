# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Core Wrapper
# Implements PRD-001, PRD-002

"""
Core wrapper implementation for the Clyro SDK.

This module provides the main `wrap()` function that wraps any callable
agent with tracing, execution controls, and policy enforcement.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import inspect
import json
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload
from uuid import UUID, uuid4, uuid5

import structlog

from clyro.adapters.anthropic import AnthropicAdapter
from clyro.adapters.claude_agent_sdk import ClaudeAgentAdapter
from clyro.adapters.crewai import CrewAIAdapter
from clyro.adapters.generic import GenericAdapter, detect_adapter
from clyro.adapters.langgraph import LangGraphAdapter
from clyro.config import ClyroConfig, get_config, set_config
from clyro.exceptions import (
    ClyroWrapError,
    ExecutionControlError,
    PolicyViolationError,
)
from clyro.local_logger import LocalTerminalLogger
from clyro.local_policy import SDKLocalPolicyEvaluator
from clyro.policy import ApprovalHandler, PolicyEvaluator
from clyro.quota_prompt import QuotaPromptManager
from clyro.session import Session, get_current_session, set_current_session
from clyro.telemetry_client import submit_telemetry
from clyro.trace import Framework, TraceEvent
from clyro.transport import SyncTransport, Transport

logger = structlog.get_logger(__name__)

# Fixed namespace UUID for local-mode agent_id generation.
# In local mode there is no org_id (no API key / no JWT), so we use this
# constant namespace instead.  Cross-org collisions are irrelevant locally.
_LOCAL_MODE_NAMESPACE = UUID("00000000-0000-0000-0000-00000000cafe")

# Sentinel to distinguish "user didn't pass approval_handler" from "user passed None".
# When not passed, PolicyEvaluator auto-detects ConsoleApprovalHandler for interactive terminals.
_APPROVAL_HANDLER_NOT_SET = PolicyEvaluator._NO_HANDLER


def _extract_org_id_from_jwt_api_key(api_key: str) -> UUID | None:
    """
    Extract org_id from JWT-style API key WITHOUT signature verification.

    This is a client-side convenience function for extracting org_id
    from JWT API keys to use as namespace for agent_id generation.
    The backend will still verify the signature during authentication.

    Args:
        api_key: Complete API key (cly_{env}_{base64_payload}.{signature})

    Returns:
        Organization UUID if successfully extracted, None if parsing fails

    Warning:
        This function does NOT verify the signature. It's only used for
        extracting org_id for local agent_id generation. The backend
        performs full signature verification during authentication.

    Example:
        >>> key = "cly_live_eyJvcmdfaWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAifQ.abc123"
        >>> org_id = _extract_org_id_from_jwt_api_key(key)
        >>> org_id is not None
        True
    """
    try:
        # Parse key format: cly_{env}_{payload}.{signature}
        parts = api_key.split("_", 2)
        if len(parts) != 3:
            return None

        jwt_part = parts[2]
        jwt_components = jwt_part.rsplit(".", 1)
        if len(jwt_components) != 2:
            return None

        payload_b64, _ = jwt_components

        # Decode payload (no signature verification needed for org_id extraction)
        # Add padding if needed (base64url doesn't use padding)
        padding = "=" * (4 - len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload_json = payload_bytes.decode("utf-8")
        payload = json.loads(payload_json)

        # Extract org_id
        org_id_str = payload.get("org_id")
        if not org_id_str:
            return None

        return UUID(org_id_str)

    except (ValueError, json.JSONDecodeError, KeyError):
        # Parsing failed - return None
        return None


def _sanitize_agent_name(name: str) -> str:
    """
    Canonical agent name sanitization — MUST match the API's
    sanitize_agent_name() in agent_registration.py.

    Steps:
      1. Strip whitespace
      2. Replace invalid characters with hyphens
      3. Collapse consecutive hyphens
      4. Strip leading/trailing hyphens
      5. Truncate to 255 chars
      6. Lowercase
    """
    import re

    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    name = name[:255]
    return name.lower()


def _generate_agent_id_from_name(agent_name: str, org_id: UUID) -> UUID:
    """
    Generate a stable agent_id from agent_name with org_id namespace.

    Uses UUID5 (name-based) to generate deterministic UUIDs:
    - Same agent_name + org_id always generates the same agent_id
    - org_id namespace prevents cross-org collisions
    - Deterministic: same inputs always produce same output

    The agent_name is sanitized before UUID generation to ensure the
    same logical name always produces the same ID regardless of casing,
    whitespace, or special characters.

    Args:
        agent_name: Human-readable agent name
        org_id: Organization UUID for namespacing (REQUIRED)

    Returns:
        Stable UUID for this agent

    Example:
        >>> from uuid import uuid4
        >>> org_id = uuid4()
        >>> agent_id = _generate_agent_id_from_name("my-agent", org_id)
        >>> agent_id == _generate_agent_id_from_name("my-agent", org_id)
        True

    Note:
        org_id is now REQUIRED (no NAMESPACE_DNS fallback) to prevent
        agent_id collisions across organizations.
    """
    return uuid5(org_id, _sanitize_agent_name(agent_name))


# Type variables for generic wrapper
T = TypeVar("T")
R = TypeVar("R")


class WrappedAgent(Generic[T]):
    """
    A wrapped agent that captures traces and enforces execution controls.

    The wrapped agent behaves identically to the original but with
    observability, safety controls, and policy enforcement.

    Example:
        ```python
        def my_agent(query: str) -> str:
            return f"Response to: {query}"

        wrapped = clyro.wrap(my_agent)
        result = wrapped("Hello")  # Traces captured automatically
        ```
    """

    def __init__(
        self,
        agent: Callable[..., T],
        config: ClyroConfig | None = None,
        adapter: str | None = None,
        agent_id: UUID | None = None,
        org_id: UUID | None = None,
        approval_handler: ApprovalHandler | None | object = _APPROVAL_HANDLER_NOT_SET,
    ):
        """
        Initialize wrapped agent.

        Agent identification is required. Provide one of:
        - agent_name in ClyroConfig (auto-registration: generates agent_id)
        - agent_id parameter (manual registration: from POST /v1/agents/register)

        Args:
            agent: The callable agent to wrap
            config: Optional configuration (uses global config if not provided)
            adapter: Framework adapter to use ('langgraph', 'crewai', 'generic')
            agent_id: Agent UUID (from manual registration or config.agent_id)
            org_id: Optional organization UUID

        Raises:
            ClyroWrapError: If the agent is not callable or no agent identification
        """
        if not callable(agent) and not hasattr(agent, "invoke"):
            raise ClyroWrapError(
                message=f"Agent must be callable or have an invoke() method, got {type(agent).__name__}",
                agent_type=type(agent).__name__,
            )

        self._agent = agent
        self._config = config or get_config()
        self._adapter = adapter or detect_adapter(agent)

        # Extract org_id from JWT API key if not provided explicitly
        if org_id is None and self._config.api_key:
            extracted_org_id = _extract_org_id_from_jwt_api_key(self._config.api_key)
            if extracted_org_id:
                self._org_id = extracted_org_id
                logger.debug(
                    "org_id_extracted_from_api_key",
                    org_id=str(self._org_id),
                )
            else:
                self._org_id = None
        else:
            self._org_id = org_id

        # Resolve agent_id: config.agent_id can also provide it
        resolved_agent_id = agent_id
        if resolved_agent_id is None and self._config.agent_id:
            try:
                resolved_agent_id = UUID(self._config.agent_id)
            except (ValueError, TypeError):
                raise ClyroWrapError(
                    message=f"Invalid agent_id in config: '{self._config.agent_id}'. Must be a valid UUID.",
                    agent_type=type(agent).__name__,
                ) from None

        # Determine agent identification
        # In local mode, agent_id is optional — use a local namespace if needed
        # In cloud mode, agent_id is mandatory (org_id required for auto-registration)
        if resolved_agent_id is not None:
            # Flow 2: Manual registration — user provides agent_id directly
            self._agent_id = resolved_agent_id
        elif self._config.agent_name:
            # Flow 1: Auto-registration — generate agent_id from agent_name
            if self._org_id is not None:
                self._agent_id = _generate_agent_id_from_name(
                    self._config.agent_name, org_id=self._org_id
                )
            elif self._config.mode == "local":
                # Local mode: use a fixed local namespace (no cross-org collision risk)
                self._agent_id = _generate_agent_id_from_name(
                    self._config.agent_name, org_id=_LOCAL_MODE_NAMESPACE
                )
            else:
                # Cloud mode without org_id — cannot generate stable agent_id
                raise ClyroWrapError(
                    message=(
                        "org_id is required for agent_name auto-registration. "
                        "org_id can be extracted from JWT-style API keys automatically, "
                        "or passed explicitly via wrap(org_id=...). "
                        "Please ensure you have a valid JWT API key configured."
                    ),
                    agent_type=type(agent).__name__,
                )

            logger.debug(
                "agent_id_auto_generated",
                agent_name=self._config.agent_name,
                agent_id=str(self._agent_id),
                org_id=str(self._org_id) if self._org_id else "local",
            )
        elif self._config.mode == "local":
            # Local mode with no agent_name and no agent_id — generate a random one
            self._agent_id = uuid4()
            logger.debug("agent_id_random_local", agent_id=str(self._agent_id))
        else:
            # Cloud mode: neither provided — raise error with actionable message
            raise ClyroWrapError(
                message=(
                    "Agent identification is required. Provide one of:\n"
                    "  1. agent_name in ClyroConfig (for auto-registration)\n"
                    "  2. agent_id to wrap() or in ClyroConfig (from POST /v1/agents/register)\n"
                    "Example: clyro.configure(ClyroConfig(agent_name='my-agent'))"
                ),
                agent_type=type(agent).__name__,
            )

        self._is_async = asyncio.iscoroutinefunction(agent)
        self._is_runnable = not callable(agent) and hasattr(agent, "invoke")
        self._adapter_instance = self._create_adapter()

        # Determine framework
        self._framework = self._detect_framework()

        # Implements FRD-SOF-004, FRD-SOF-005: transport gating by mode
        self._local_logger: LocalTerminalLogger | None = None

        # Implements FRD-CT-004, FRD-CT-005, FRD-CT-006: quota upgrade prompts
        self._quota_prompt = QuotaPromptManager(self._config)

        if self._config.mode == "local":
            # LOCAL PATH: no transport, no cloud policy evaluator (FRD-SOF-005)
            self._transport: Transport | SyncTransport | None = None
            self._local_logger = LocalTerminalLogger(self._config)

            # Implements FRD-SOF-008: first-run welcome message
            self._local_logger.print_welcome()

            # Implements FRD-SOF-002: local policy evaluator
            if self._config.controls.enable_policy_enforcement:
                self._policy_evaluator: PolicyEvaluator | SDKLocalPolicyEvaluator | None = (
                    SDKLocalPolicyEvaluator(approval_handler=approval_handler)
                )
            else:
                self._policy_evaluator = None
        else:
            # CLOUD PATH: existing behavior unchanged
            if self._is_async:
                self._transport = Transport(self._config)
            else:
                self._transport = SyncTransport(self._config)

            # Policy enforcement (cloud)
            if self._config.controls.enable_policy_enforcement and self._config.api_key:
                self._policy_evaluator = PolicyEvaluator(
                    config=self._config,
                    agent_id=self._agent_id,
                    org_id=self._org_id,
                    approval_handler=approval_handler,
                )
            else:
                self._policy_evaluator = None

        # Track background sync state
        self._background_sync_started = False

        # Start background sync for sync agents (uses background thread)
        # For async agents, we start it on first execution (need running event loop)
        if not self._is_async and not self._config.is_local_only():
            self._start_background_sync_safe()

        # Preserve original function metadata (only for true callables, not runnables)
        if not self._is_runnable:
            functools.update_wrapper(self, agent)

        logger.debug(
            "agent_wrapped",
            agent_name=getattr(agent, "__name__", str(agent)),
            adapter=self._adapter,
            framework=self._framework.value,
            is_async=self._is_async,
        )

    def _detect_framework(self) -> Framework:
        """Detect the framework from adapter name or agent type."""
        adapter_map = {
            "anthropic": Framework.ANTHROPIC,
            "langgraph": Framework.LANGGRAPH,
            "crewai": Framework.CREWAI,
            "claude_agent_sdk": Framework.CLAUDE_AGENT_SDK,
            "generic": Framework.GENERIC,
        }
        return adapter_map.get(self._adapter, Framework.GENERIC)

    def _create_adapter(
        self,
    ) -> GenericAdapter | LangGraphAdapter | CrewAIAdapter | ClaudeAgentAdapter | None:
        """Create an adapter instance for the wrapped agent."""
        if self._adapter == "langgraph":
            return LangGraphAdapter(self._agent, self._config)
        elif self._adapter == "crewai":
            return CrewAIAdapter(self._agent, self._config)
        elif self._adapter == "claude_agent_sdk":
            return ClaudeAgentAdapter(self._agent, self._config)
        elif self._adapter == "generic":
            return None
        raise ClyroWrapError(
            message=f"Adapter '{self._adapter}' is not supported in this SDK build.",
            agent_type=type(self._agent).__name__,
        )

    def _start_background_sync_safe(self) -> None:
        """
        Safely start background sync with fail-open behavior.

        For sync agents: Uses background thread
        For async agents: Should be called from async context

        Errors are logged but not raised to prevent blocking agent execution.
        """
        if self._background_sync_started:
            return

        try:
            if isinstance(self._transport, SyncTransport):
                # Sync transport: uses background thread
                self._transport.start_background_sync()
            # For Transport (async), this will be called from async context
            self._background_sync_started = True
            logger.debug(
                "background_sync_enabled",
                agent_name=getattr(self._agent, "__name__", str(self._agent)),
            )
        except Exception as e:
            # Fail-open: log error but don't prevent agent from working
            logger.warning(
                "background_sync_start_failed",
                error=str(e),
                agent_name=getattr(self._agent, "__name__", str(self._agent)),
                fail_open=True,
            )

    async def _start_background_sync_async(self) -> None:
        """
        Start background sync for async agents.

        Must be called from within an async context (running event loop).
        Fail-open: errors are logged but not raised.
        """
        if self._background_sync_started:
            return

        try:
            if isinstance(self._transport, Transport):
                await self._transport.start_background_sync()
            self._background_sync_started = True
            logger.debug(
                "background_sync_enabled",
                agent_name=getattr(self._agent, "__name__", str(self._agent)),
            )
        except Exception as e:
            # Fail-open: log error but don't prevent agent from working
            logger.warning(
                "background_sync_start_failed",
                error=str(e),
                agent_name=getattr(self._agent, "__name__", str(self._agent)),
                fail_open=True,
            )

    @property
    def agent(self) -> Callable[..., T]:
        """Get the underlying agent callable."""
        return self._agent

    @property
    def config(self) -> ClyroConfig:
        """Get the configuration."""
        return self._config

    @property
    def session(self) -> Session | None:
        """Get the current session if active."""
        return get_current_session()

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute the wrapped agent.

        Args:
            *args: Positional arguments to pass to the agent
            **kwargs: Keyword arguments to pass to the agent

        Returns:
            The result from the agent

        Raises:
            Same exceptions as the original agent
            ExecutionControlError: If execution controls are violated
        """
        if self._is_async:
            # Return coroutine for async agents
            return self._execute_async(*args, **kwargs)  # type: ignore
        else:
            return self._execute_sync(*args, **kwargs)

    def _execute_sync(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute synchronous agent with tracing and session management.

        Orchestrates the full execution lifecycle:
        1. Creates and activates a session
        2. Starts the session and emits session_start event
        3. Executes the agent with error handling
        4. Records execution step with timing and output
        5. Cleans up session and flushes events (always runs)
        """
        # Phase 1: Setup session and capture input
        session = self._create_session()
        input_data = self._capture_input(args, kwargs)

        try:
            # Phase 2: Execute with tracing
            result = self._run_sync_with_tracing(session, input_data, args, kwargs)
            return result
        finally:
            # Phase 3: Cleanup (always runs, even on error)
            self._cleanup_session_sync(session)

    async def _execute_async(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute asynchronous agent with tracing and session management.

        Orchestrates the full execution lifecycle:
        1. Creates and activates a session
        2. Starts the session and emits session_start event
        3. Executes the agent with error handling
        4. Records execution step with timing and output
        5. Cleans up session and flushes events (always runs)
        """
        # Phase 0: Start background sync on first execution (if not local-only)
        if not self._background_sync_started and not self._config.is_local_only():
            await self._start_background_sync_async()

        # Phase 1: Setup session and capture input
        session = self._create_session()
        input_data = self._capture_input(args, kwargs)

        try:
            # Phase 2: Execute with tracing
            result = await self._run_async_with_tracing(session, input_data, args, kwargs)
            return result
        finally:
            # Phase 3: Cleanup (always runs, even on error)
            await self._cleanup_session_async(session)

    def _create_session(self) -> Session:
        """
        Create and activate a new session.

        Returns:
            Newly created and activated session
        """
        # Get framework_version from adapter instance if available
        framework_version = None
        if self._adapter_instance and hasattr(self._adapter_instance, "framework_version"):
            framework_version = self._adapter_instance.framework_version

        session = Session(
            config=self._config,
            agent_id=self._agent_id,
            org_id=self._org_id,
            framework=self._framework,
            framework_version=framework_version,
            agent_name=self._config.agent_name,
            policy_evaluator=self._policy_evaluator,
        )

        # Wire event sink: local mode → LocalTerminalLogger, cloud → transport buffer
        # Implements FRD-SOF-005: _event_sink switch
        if self._config.mode == "local" and self._local_logger is not None:
            session._event_sink = self._local_event_sink
        else:
            session._event_sink = self._buffer_event_sink

        set_current_session(session)
        return session

    def _run_sync_with_tracing(
        self,
        session: Session,
        input_data: dict[str, Any] | None,
        args: tuple,
        kwargs: dict,
    ) -> T:
        """
        Run synchronous agent and emit trace events.

        Handles:
        - Session start event emission
        - Agent execution timing
        - Step event recording with output capture
        - Error event emission for execution control and general errors
        - Session end event emission with results/errors

        Args:
            session: Active session for this execution
            input_data: Captured input data
            args: Positional arguments for agent
            kwargs: Keyword arguments for agent

        Returns:
            Result from agent execution

        Raises:
            Same exceptions as the original agent
        """
        # Start session and emit event (fail-open)
        start_event = session.start(input_data=input_data)
        if start_event is not None:
            self._buffer_event_sync(start_event)

        # Implements FRD-CT-004, FRD-CT-005: quota upgrade prompts at session start
        try:
            self._quota_prompt.check()
        except Exception:
            pass  # Fail-open: quota check must never crash user code

        result: T
        error: Exception | None = None
        start_time = time.perf_counter()

        adapter_context: dict[str, Any] | None = None
        call_kwargs = dict(kwargs)
        try:
            if self._adapter_instance:
                adapter_context = self._adapter_instance.before_call(session, args, call_kwargs)
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.warning("clyro_before_call_failed", error=str(e), fail_open=True)

        try:
            # Pre-execution policy check — ensure parameters is a dict
            policy_input = (
                input_data
                if isinstance(input_data, dict)
                else {"input": str(input_data) if input_data else ""}
            )
            session.check_policy("agent_execution", policy_input)

            # Execute the wrapped agent
            if self._is_runnable:
                result = self._agent.invoke(*args, **call_kwargs)
            else:
                result = self._agent(*args, **call_kwargs)

        except PolicyViolationError as e:
            # Policy violation — record and re-raise
            error = e
            # Implements FRD-SOF-011: log violation context in local mode
            if self._local_logger is not None and e.details:
                try:
                    self._local_logger.log_violation(
                        e.action_type or "unknown",
                        e.details,
                    )
                except Exception:
                    pass  # Fail-open
            try:
                self._record_adapter_events_sync(session, adapter_context)
                error_event = session.record_error(e, event_name="policy_violation")
                self._buffer_event_sync(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="policy_violation", fail_open=True
                )
            raise

        except ExecutionControlError as e:
            # Execution control violation (step limit, cost limit, loop detected)
            error = e
            try:
                self._record_adapter_events_sync(session, adapter_context)
                error_event = session.record_error(e, event_name="execution_control")
                self._buffer_event_sync(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="execution_control", fail_open=True
                )
            raise

        except Exception as e:
            # General agent error — record and re-raise
            error = e
            try:
                self._record_adapter_events_sync(session, adapter_context)
                if self._adapter_instance:
                    adapter_error_event = self._adapter_instance.on_error(
                        session,
                        e,
                        adapter_context or {},
                    )
                    session.record_event(adapter_error_event)
                    self._buffer_event_sync(adapter_error_event)
                error_event = session.record_error(e, event_name="agent_error")
                self._buffer_event_sync(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="agent_error", fail_open=True
                )
            raise

        else:
            # Agent succeeded — record step and post-process
            try:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                output_data = self._capture_output(result)

                try:
                    self._record_adapter_events_sync(session, adapter_context)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_adapter_events_failed", fail_open=True)

                # Check for deferred errors from event bus handlers.
                # CrewAI's event bus swallows all exceptions, so enforcement
                # errors are stored on the handler and re-raised here.
                self._raise_pending_adapter_error(adapter_context)

                try:
                    step_event = session.record_step(
                        event_name="agent_execution",
                        input_data=input_data,
                        output_data=output_data,
                        duration_ms=duration_ms,
                    )
                    if step_event is not None:
                        self._buffer_event_sync(step_event)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_step_recording_failed", fail_open=True)

                try:
                    if self._adapter_instance:
                        adapter_event = self._adapter_instance.after_call(
                            session,
                            result,
                            adapter_context or {},
                        )
                        if adapter_event is not None:
                            session.record_event(adapter_event)
                            self._buffer_event_sync(adapter_event)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_after_call_failed", fail_open=True)

                return result

            except (PolicyViolationError, ExecutionControlError) as e:
                # Enforcement error during post-processing — record and re-raise
                error = e
                try:
                    error_event = session.record_error(e, event_name="execution_control")
                    if error_event is not None:
                        self._buffer_event_sync(error_event)
                except Exception:
                    logger.warning(
                        "clyro_error_recording_failed",
                        error_type="execution_control",
                        fail_open=True,
                    )
                raise

        finally:
            # Always end the session, even on error
            try:
                output_data = self._capture_output(result) if error is None else None
                end_event = session.end(output_data=output_data, error=error)
                if end_event is not None:
                    self._buffer_event_sync(end_event)
            except Exception:
                logger.warning("clyro_session_end_failed", fail_open=True)

            # Implements FRD-SOF-007: session-end governance summary (local mode)
            if self._local_logger is not None:
                try:
                    self._local_logger.print_session_summary(session)
                except Exception:
                    pass  # Fail-open

            # Implements FRD-CT-008: telemetry submission at session end
            try:
                submit_telemetry(self._config, session)
            except Exception:
                pass  # Fail-open: telemetry must never crash user code

    async def _run_async_with_tracing(
        self,
        session: Session,
        input_data: dict[str, Any] | None,
        args: tuple,
        kwargs: dict,
    ) -> T:
        """
        Run asynchronous agent and emit trace events.

        Handles:
        - Session start event emission
        - Agent execution timing
        - Step event recording with output capture
        - Error event emission for execution control and general errors
        - Session end event emission with results/errors

        Args:
            session: Active session for this execution
            input_data: Captured input data
            args: Positional arguments for agent
            kwargs: Keyword arguments for agent

        Returns:
            Result from agent execution

        Raises:
            Same exceptions as the original agent
        """
        # Start session and emit event (fail-open)
        start_event = session.start(input_data=input_data)
        if start_event is not None:
            await self._buffer_event_async(start_event)

        # Implements FRD-CT-004, FRD-CT-005: quota upgrade prompts at session start
        try:
            self._quota_prompt.check()
        except Exception:
            pass  # Fail-open: quota check must never crash user code

        result: T
        error: Exception | None = None
        start_time = time.perf_counter()

        adapter_context: dict[str, Any] | None = None
        call_kwargs = dict(kwargs)
        try:
            if self._adapter_instance:
                adapter_context = self._adapter_instance.before_call(session, args, call_kwargs)
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.warning("clyro_before_call_failed", error=str(e), fail_open=True)

        try:
            # Pre-execution policy check — ensure parameters is a dict
            policy_input = (
                input_data
                if isinstance(input_data, dict)
                else {"input": str(input_data) if input_data else ""}
            )
            await session.check_policy_async("agent_execution", policy_input)

            # Execute the wrapped agent
            if self._is_runnable and hasattr(self._agent, "ainvoke"):
                result = await self._agent.ainvoke(*args, **call_kwargs)
            elif self._is_runnable:
                result = self._agent.invoke(*args, **call_kwargs)
            else:
                result = await self._agent(*args, **call_kwargs)

        except PolicyViolationError as e:
            # Policy violation — record and re-raise
            error = e
            # Implements FRD-SOF-011: log violation context in local mode
            if self._local_logger is not None and e.details:
                try:
                    self._local_logger.log_violation(
                        e.action_type or "unknown",
                        e.details,
                    )
                except Exception:
                    pass  # Fail-open
            try:
                await self._record_adapter_events_async(session, adapter_context)
                error_event = session.record_error(e, event_name="policy_violation")
                await self._buffer_event_async(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="policy_violation", fail_open=True
                )
            raise

        except ExecutionControlError as e:
            # Execution control violation (step limit, cost limit, loop detected)
            error = e
            try:
                await self._record_adapter_events_async(session, adapter_context)
                error_event = session.record_error(e, event_name="execution_control")
                await self._buffer_event_async(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="execution_control", fail_open=True
                )
            raise

        except Exception as e:
            # General agent error — record and re-raise
            error = e
            try:
                await self._record_adapter_events_async(session, adapter_context)
                if self._adapter_instance:
                    adapter_error_event = self._adapter_instance.on_error(
                        session,
                        e,
                        adapter_context or {},
                    )
                    session.record_event(adapter_error_event)
                    await self._buffer_event_async(adapter_error_event)
                error_event = session.record_error(e, event_name="agent_error")
                await self._buffer_event_async(error_event)
            except Exception:
                logger.warning(
                    "clyro_error_recording_failed", error_type="agent_error", fail_open=True
                )
            raise

        else:
            # Agent succeeded — record step and post-process
            try:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                output_data = self._capture_output(result)

                try:
                    await self._record_adapter_events_async(session, adapter_context)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_adapter_events_failed", fail_open=True)

                # Check for deferred errors from event bus handlers.
                # CrewAI's event bus swallows all exceptions, so enforcement
                # errors are stored on the handler and re-raised here.
                self._raise_pending_adapter_error(adapter_context)

                try:
                    step_event = session.record_step(
                        event_name="agent_execution",
                        input_data=input_data,
                        output_data=output_data,
                        duration_ms=duration_ms,
                    )
                    if step_event is not None:
                        await self._buffer_event_async(step_event)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_step_recording_failed", fail_open=True)

                try:
                    if self._adapter_instance:
                        adapter_event = self._adapter_instance.after_call(
                            session,
                            result,
                            adapter_context or {},
                        )
                        if adapter_event is not None:
                            session.record_event(adapter_event)
                            await self._buffer_event_async(adapter_event)
                except (PolicyViolationError, ExecutionControlError):
                    raise
                except Exception:
                    logger.warning("clyro_after_call_failed", fail_open=True)

                return result

            except (PolicyViolationError, ExecutionControlError) as e:
                # Enforcement error during post-processing — record and re-raise
                error = e
                try:
                    error_event = session.record_error(e, event_name="execution_control")
                    if error_event is not None:
                        await self._buffer_event_async(error_event)
                except Exception:
                    logger.warning(
                        "clyro_error_recording_failed",
                        error_type="execution_control",
                        fail_open=True,
                    )
                raise

        finally:
            # Always end the session, even on error
            try:
                output_data = self._capture_output(result) if error is None else None
                end_event = session.end(output_data=output_data, error=error)
                if end_event is not None:
                    await self._buffer_event_async(end_event)
            except Exception:
                logger.warning("clyro_session_end_failed", fail_open=True)

            # Implements FRD-SOF-007: session-end governance summary (local mode)
            if self._local_logger is not None:
                try:
                    self._local_logger.print_session_summary(session)
                except Exception:
                    pass  # Fail-open

            # Implements FRD-CT-008: telemetry submission at session end
            try:
                submit_telemetry(self._config, session)
            except Exception:
                pass  # Fail-open: telemetry must never crash user code

    def _cleanup_session_sync(self, session: Session) -> None:
        """
        Clean up session after execution completes.

        Flushes all buffered events and clears the session context.
        Always runs, even if agent execution failed.

        Args:
            session: Session to clean up (for future use; currently unused)
        """
        self._flush_sync()
        set_current_session(None)

    async def _cleanup_session_async(self, session: Session) -> None:
        """
        Clean up session after async execution completes.

        Flushes all buffered events and clears the session context.
        Always runs, even if agent execution failed.

        Args:
            session: Session to clean up (for future use; currently unused)
        """
        await self._flush_async()
        set_current_session(None)

    def _capture_input(self, args: tuple, kwargs: dict) -> dict[str, Any] | None:
        """Capture input data if enabled."""
        if not self._config.capture_inputs:
            return None

        try:
            # Get function signature to match args to parameter names
            target = self._agent.invoke if self._is_runnable else self._agent
            sig = inspect.signature(target)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Convert to serializable dict
            return {k: self._serialize_value(v) for k, v in bound.arguments.items()}
        except Exception as e:
            logger.warning("input_capture_failed", error=str(e))
            return {"args": [self._serialize_value(a) for a in args], "kwargs": kwargs}

    def _capture_output(self, result: Any) -> dict[str, Any] | None:
        """Capture output data if enabled."""
        if not self._config.capture_outputs:
            return None

        try:
            return {"result": self._serialize_value(result)}
        except Exception as e:
            logger.warning("output_capture_failed", error=str(e))
            return {"result": str(result)}

    def _serialize_value(self, value: Any, _depth: int = 0) -> Any:
        """
        Serialize a value for JSON storage using a dispatch pattern.

        Handles common Python types and objects with proper fallback logic.
        Includes recursion depth limit to prevent stack overflow attacks.

        Args:
            value: Value to serialize
            _depth: Current recursion depth (internal use only)

        Returns:
            JSON-serializable representation of the value
        """
        # Prevent stack overflow with deeply nested structures
        MAX_DEPTH = 100
        if _depth > MAX_DEPTH:
            logger.warning("serialization_depth_exceeded", depth=_depth, type=type(value).__name__)
            return f"<max_depth_exceeded: {type(value).__name__}>"

        # Primitives and None
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        # Collections - recursively serialize elements
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v, _depth + 1) for v in value]

        if isinstance(value, dict):
            return {k: self._serialize_value(v, _depth + 1) for k, v in value.items()}

        # Pydantic models - use model_dump()
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump()
            except Exception as e:
                logger.debug("pydantic_serialization_failed", error=str(e))
                return str(value)

        # Objects with __dict__ - serialize public attributes
        if hasattr(value, "__dict__"):
            try:
                return {
                    k: self._serialize_value(v, _depth + 1)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception as e:
                logger.debug("object_serialization_failed", error=str(e))
                return str(value)

        # Fallback to string representation
        return str(value)

    def _buffer_event_sink(self, event: TraceEvent) -> None:
        """Event sink callback for Session — buffers to whichever transport is active.

        Used as session._event_sink so that policy_check events from
        check_policy() / check_policy_async() reach the transport.
        Fail-open: never raises.
        """
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.buffer_event(event)
            elif isinstance(self._transport, Transport):
                # Async transport from sync context — schedule buffering.
                # This handles the edge case where check_policy (sync) is
                # called but the wrapper uses an async transport.
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._transport.buffer_event(event))
                except RuntimeError:
                    pass  # No running loop — drop event (fail-open)
        except Exception:
            pass  # Fail-open: event sink never blocks policy enforcement

    def _local_event_sink(self, event: TraceEvent) -> None:
        """Event sink for local mode — routes to LocalTerminalLogger.

        Implements FRD-SOF-006: terminal logging in local mode.
        Fail-open: never raises.
        """
        if self._local_logger is not None:
            try:
                self._local_logger.log_event(event)
            except Exception:
                pass  # Fail-open: logging never blocks

    def _buffer_event_sync(self, event: TraceEvent) -> None:
        """Buffer event synchronously with fail-open behavior."""
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.buffer_event(event)
        except Exception as e:
            if not self._config.fail_open:
                raise
            logger.warning("event_buffer_failed", error=str(e), fail_open=True)

    async def _buffer_event_async(self, event: TraceEvent) -> None:
        """Buffer event asynchronously with fail-open behavior."""
        try:
            if isinstance(self._transport, Transport):
                await self._transport.buffer_event(event)
        except Exception as e:
            if not self._config.fail_open:
                raise
            logger.warning("event_buffer_failed", error=str(e), fail_open=True)

    def _drain_adapter_events(self, adapter_context: dict[str, Any] | None) -> list[TraceEvent]:
        """Drain any adapter-recorded events from the execution context."""
        if not adapter_context:
            return []
        handler = adapter_context.get("handler")
        if handler and hasattr(handler, "drain_events"):
            return handler.drain_events()
        return []

    @staticmethod
    def _raise_pending_adapter_error(adapter_context: dict[str, Any] | None) -> None:
        """Raise any deferred error from adapter event bus handlers.

        CrewAI's event bus swallows all exceptions from handlers (catches Exception).
        Enforcement errors (PolicyViolationError, ExecutionControlError) are stored
        on the handler as _pending_error when they occur. After crew.kickoff()
        returns, we check here and re-raise so the wrapper's except blocks handle
        them properly.

        Also resolves deferred require_approval decisions. CrewAI's event bus
        fires events AFTER actions, so approval prompts are deferred to step
        boundaries. If the last action triggered require_approval and there was
        no subsequent _next_step(), we resolve it here.

        Raises:
            PolicyViolationError: If a policy violation was detected during execution
            ExecutionControlError: If a step/cost limit was exceeded during execution
        """
        if not adapter_context:
            return
        handler = adapter_context.get("handler")
        if not handler:
            return

        # Resolve deferred require_approval from the last action
        if hasattr(handler, "_resolve_pending_approval"):
            handler._resolve_pending_approval()

        if hasattr(handler, "_pending_error") and handler._pending_error is not None:
            raise handler._pending_error

    def _record_adapter_events_sync(
        self,
        session: Session,
        adapter_context: dict[str, Any] | None,
    ) -> None:
        """Record and buffer adapter events for sync execution (fail-open)."""
        try:
            for event in self._drain_adapter_events(adapter_context):
                session.record_event(event)
                self._buffer_event_sync(event)
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception:
            logger.warning("clyro_record_adapter_events_failed", fail_open=True)

    async def _record_adapter_events_async(
        self,
        session: Session,
        adapter_context: dict[str, Any] | None,
    ) -> None:
        """Record and buffer adapter events for async execution (fail-open)."""
        try:
            for event in self._drain_adapter_events(adapter_context):
                session.record_event(event)
                await self._buffer_event_async(event)
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception:
            logger.warning("clyro_record_adapter_events_failed", fail_open=True)

    def _flush_sync(self) -> None:
        """Flush events synchronously with fail-open behavior."""
        try:
            if isinstance(self._transport, SyncTransport):
                self._transport.flush()
        except Exception as e:
            if not self._config.fail_open:
                raise
            logger.warning("flush_failed", error=str(e), fail_open=True)

    async def _flush_async(self) -> None:
        """Flush events asynchronously with fail-open behavior."""
        try:
            if isinstance(self._transport, Transport):
                await self._transport.flush()
        except Exception as e:
            if not self._config.fail_open:
                raise
            logger.warning("flush_failed", error=str(e), fail_open=True)

    def close(self) -> None:
        """
        Close the wrapped agent and cleanup resources.

        Stops background sync and flushes any pending events.
        For sync agents only. Async agents should use close_async().
        """
        # Local mode: no transport, just clean up evaluator
        if self._transport is None:
            try:
                if self._policy_evaluator:
                    self._policy_evaluator.close_sync()
                logger.debug("wrapped_agent_closed", mode="local")
            except Exception as e:
                logger.warning("close_failed", error=str(e))
            return

        if not isinstance(self._transport, SyncTransport):
            logger.warning(
                "close_called_on_async_agent",
                message="Use close_async() for async agents or await the result",
            )
            return

        try:
            if self._policy_evaluator:
                self._policy_evaluator.close_sync()
            self._transport.close()
            logger.debug("wrapped_agent_closed")
        except Exception as e:
            logger.warning("close_failed", error=str(e))

    async def close_async(self) -> None:
        """
        Close the wrapped agent and cleanup resources (async version).

        Stops background sync and flushes any pending events.
        For async agents only.
        """
        # Local mode: no transport, just clean up evaluator
        if self._transport is None:
            try:
                if self._policy_evaluator:
                    await self._policy_evaluator.close_async()
                logger.debug("wrapped_agent_closed", mode="local")
            except Exception as e:
                logger.warning("close_async_failed", error=str(e))
            return

        if not isinstance(self._transport, Transport):
            logger.warning(
                "close_async_called_on_sync_agent",
                message="Use close() for sync agents",
            )
            return

        try:
            if self._policy_evaluator:
                await self._policy_evaluator.close_async()
            await self._transport.close()
            logger.debug("wrapped_agent_closed")
        except Exception as e:
            logger.warning("close_async_failed", error=str(e))

    def invoke(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute the wrapped agent (LangGraph-compatible API).

        This method provides API compatibility with LangGraph's CompiledGraph.invoke().
        It delegates to __call__ for the actual execution.

        Args:
            *args: Positional arguments to pass to the agent
            **kwargs: Keyword arguments to pass to the agent

        Returns:
            The result from the agent
        """
        return self(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> T:
        """
        Execute the wrapped agent asynchronously (LangGraph-compatible API).

        This method provides API compatibility with LangGraph's CompiledGraph.ainvoke().
        It delegates to __call__ for the actual execution.

        Args:
            *args: Positional arguments to pass to the agent
            **kwargs: Keyword arguments to pass to the agent

        Returns:
            The result from the agent
        """
        result = self(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def get_status(self) -> dict[str, Any]:
        """Get wrapper status information."""
        return {
            "agent_name": getattr(self._agent, "__name__", str(self._agent)),
            "agent_id": str(self._agent_id) if self._agent_id else None,
            "org_id": str(self._org_id) if self._org_id else None,
            "framework": self._framework.value,
            "adapter": self._adapter,
            "is_async": self._is_async,
            "background_sync_started": self._background_sync_started,
            "config": {
                "endpoint": self._config.endpoint,
                "local_only": self._config.is_local_only(),
                "fail_open": self._config.fail_open,
                "max_steps": self._config.controls.max_steps,
                "max_cost_usd": self._config.controls.max_cost_usd,
            },
            "transport": self._transport.get_sync_status()
            if isinstance(self._transport, SyncTransport)
            else None,
        }


@overload
def wrap(
    agent: Callable[..., T],
    *,
    config: ClyroConfig | None = None,
    adapter: str | None = None,
    agent_id: UUID | None = None,
    org_id: UUID | None = None,
    approval_handler: ApprovalHandler | None | object = _APPROVAL_HANDLER_NOT_SET,
) -> WrappedAgent[T]: ...


@overload
def wrap(
    agent: None = None,
    *,
    config: ClyroConfig | None = None,
    adapter: str | None = None,
    agent_id: UUID | None = None,
    org_id: UUID | None = None,
    approval_handler: ApprovalHandler | None | object = _APPROVAL_HANDLER_NOT_SET,
) -> Callable[[Callable[..., T]], WrappedAgent[T]]: ...


def wrap(
    agent: Callable[..., T] | None = None,
    *,
    config: ClyroConfig | None = None,
    adapter: str | None = None,
    agent_id: UUID | None = None,
    org_id: UUID | None = None,
    approval_handler: ApprovalHandler | None | object = _APPROVAL_HANDLER_NOT_SET,
) -> WrappedAgent[T] | Callable[[Callable[..., T]], WrappedAgent[T]]:
    """
    Wrap an agent callable with Clyro tracing and controls.

    Agent identification is required. Provide one of:
    - agent_name in ClyroConfig (Flow 1: auto-registration)
    - agent_id parameter or in ClyroConfig (Flow 2: manual registration)

    Can be used as a function or decorator:

    ```python
    # Flow 1: Auto-registration (agent_name in config)
    clyro.configure(ClyroConfig(agent_name="my-agent"))
    wrapped = clyro.wrap(my_agent)

    # Flow 2: Manual registration (agent_id from API)
    wrapped = clyro.wrap(my_agent, agent_id=UUID("..."))

    # As a decorator
    @clyro.wrap(config=ClyroConfig(agent_name="my-agent"))
    def my_agent(query: str) -> str:
        return f"Response: {query}"
    ```

    Args:
        agent: The agent callable to wrap. If None, returns a decorator.
        config: Optional configuration (uses global config if not provided)
        adapter: Framework adapter to use ('langgraph', 'crewai', 'generic').
                 Auto-detected if not specified.
        agent_id: Agent UUID (from manual registration or config.agent_id)
        org_id: Optional organization UUID
        approval_handler: Optional handler for require_approval decisions.
            If provided, the handler is called when a policy returns
            require_approval. Use ConsoleApprovalHandler() for interactive
            console prompts, or implement a custom handler.

    Returns:
        WrappedAgent if agent is provided, otherwise a decorator function.

    Raises:
        ClyroWrapError: If agent is not callable, or if neither agent_name
            nor agent_id is provided
    """

    def decorator(fn: Callable[..., T]) -> WrappedAgent[T]:
        return WrappedAgent(
            agent=fn,
            config=config,
            adapter=adapter,
            agent_id=agent_id,
            org_id=org_id,
            approval_handler=approval_handler,
        )

    if agent is not None:
        # Anthropic SDK adapter: non-callable client returns traced client  # Implements FRD-001, FRD-002
        resolved_adapter = adapter or detect_adapter(agent)
        if resolved_adapter == "anthropic":
            resolved_config = config or get_config()

            # Resolve org_id
            resolved_org_id = org_id
            if resolved_org_id is None and resolved_config.api_key:
                resolved_org_id = _extract_org_id_from_jwt_api_key(resolved_config.api_key)

            # Resolve agent_id
            resolved_agent_id = agent_id
            if resolved_agent_id is None and resolved_config.agent_id:
                try:
                    resolved_agent_id = UUID(resolved_config.agent_id)
                except (ValueError, TypeError):
                    raise ClyroWrapError(
                        message=f"Invalid agent_id in config: '{resolved_config.agent_id}'.",
                        agent_type=type(agent).__name__,
                    ) from None

            if resolved_agent_id is None and resolved_config.agent_name:
                if resolved_org_id is not None:
                    resolved_agent_id = _generate_agent_id_from_name(
                        resolved_config.agent_name, org_id=resolved_org_id
                    )
                elif resolved_config.mode == "local":
                    resolved_agent_id = _generate_agent_id_from_name(
                        resolved_config.agent_name, org_id=_LOCAL_MODE_NAMESPACE
                    )
                else:
                    raise ClyroWrapError(
                        message=(
                            "org_id is required for agent_name auto-registration. "
                            "org_id can be extracted from JWT-style API keys automatically, "
                            "or passed explicitly via wrap(org_id=...)."
                        ),
                        agent_type=type(agent).__name__,
                    )
            elif resolved_agent_id is None:
                if resolved_config.mode == "local":
                    resolved_agent_id = uuid4()
                else:
                    raise ClyroWrapError(
                        message=(
                            "Agent identification is required. Provide one of:\n"
                            "  1. agent_name in ClyroConfig (for auto-registration)\n"
                            "  2. agent_id to wrap() or in ClyroConfig"
                        ),
                        agent_type=type(agent).__name__,
                    )

            anthropic_adapter = AnthropicAdapter(
                client=agent,
                config=resolved_config,
                agent_id=resolved_agent_id,
                org_id=resolved_org_id,
                approval_handler=approval_handler
                if approval_handler is not _APPROVAL_HANDLER_NOT_SET
                else None,
            )
            return anthropic_adapter.create_traced_client()

        return decorator(agent)

    return decorator


def configure(config: ClyroConfig) -> None:
    """
    Set global SDK configuration.

    Args:
        config: Configuration object to use globally

    Example:
        ```python
        import clyro

        clyro.configure(ClyroConfig(
            api_key="cly_live_...",
            controls=ExecutionControls(max_steps=50),
        ))

        # All subsequent wraps will use this configuration
        wrapped = clyro.wrap(my_agent)
        ```
    """
    set_config(config)
    logger.info(
        "sdk_configured",
        endpoint=config.endpoint,
        local_only=config.is_local_only(),
        max_steps=config.controls.max_steps,
        max_cost_usd=config.controls.max_cost_usd,
    )


def get_session() -> Session | None:
    """
    Get the current active session.

    Returns:
        Current Session if one is active, None otherwise.
    """
    return get_current_session()
