# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Session Management
# Implements PRD-001, PRD-005, PRD-009, PRD-010

"""
Session management for the Clyro SDK.

This module handles trace sessions, which group related events
from a single agent execution.

Thread Safety:
    Session state is managed using contextvars.ContextVar to ensure
    proper isolation between concurrent agent executions. This prevents
    race conditions and session state corruption when multiple agents
    run in parallel threads or async contexts.

Execution Controls:
    Sessions enforce execution safety boundaries including:
    - Step limits: Maximum number of execution steps
    - Cost limits: Maximum cumulative LLM cost in USD
    - Loop detection: Detection of infinite loops via state hashing
"""

from __future__ import annotations

import contextvars
import json
import time
import traceback
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog

from clyro.cost import CostCalculator, TokenUsage
from clyro.exceptions import (
    CostLimitExceededError,
    StepLimitExceededError,
)
from clyro.loop_detector import LoopDetector
from clyro.trace import (
    EventType,
    Framework,
    TraceEvent,
    create_error_event,
    create_llm_call_event,
    create_session_end_event,
    create_session_start_event,
    create_step_event,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.policy import PolicyEvaluator

logger = structlog.get_logger(__name__)


class Session:
    """
    Represents a single agent execution session.

    A session tracks all events, state, and metrics for one invocation
    of a wrapped agent. Sessions enforce execution controls and provide
    the data needed for replay and debugging.
    """

    def __init__(
        self,
        config: ClyroConfig,
        session_id: UUID | None = None,
        agent_id: UUID | None = None,
        org_id: UUID | None = None,
        framework: Framework = Framework.GENERIC,
        framework_version: str | None = None,
        agent_name: str | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
    ):
        """
        Initialize a new session.

        Args:
            config: SDK configuration
            session_id: Optional session ID (generated if not provided)
            agent_id: Agent identifier
            org_id: Organization identifier
            framework: Agent framework being used
            framework_version: Version of the framework
            agent_name: Human-readable agent name (for auto-registration)
            policy_evaluator: Optional policy evaluator for enforcement
        """
        self.session_id = session_id or uuid4()
        self.agent_id = agent_id
        self.org_id = org_id
        self.agent_name = agent_name or config.agent_name
        self.config = config
        self.framework = framework
        self.framework_version = framework_version

        # Timing
        self._start_time: float | None = None
        self._end_time: float | None = None

        # Execution tracking
        self._step_number: int = 0
        self._cumulative_cost: Decimal = Decimal("0")
        self._events: list[TraceEvent] = []

        # Loop detection
        self._loop_detection_enabled: bool = self.config.controls.enable_loop_detection
        self._loop_detector: LoopDetector | None = None
        if self._loop_detection_enabled:
            self._loop_detector = LoopDetector(
                threshold=self.config.controls.loop_detection_threshold
            )

        # Policy enforcement
        self._policy_evaluator = policy_evaluator

        # Event sink: optional callback for transport buffering.
        # Set by the Wrapper so that events recorded via record_event()
        # (including policy_check events from check_policy) are also
        # sent to the transport for backend ingestion.
        self._event_sink: Any = None

        # Status
        self._is_active: bool = False
        self._error: Exception | None = None

    @property
    def step_number(self) -> int:
        """Current step number in the session."""
        return self._step_number

    @property
    def cumulative_cost(self) -> Decimal:
        """Total cost accumulated in this session."""
        return self._cumulative_cost

    @property
    def events(self) -> list[TraceEvent]:
        """All events captured in this session."""
        return self._events.copy()

    @property
    def is_active(self) -> bool:
        """Whether the session is currently active."""
        return self._is_active

    @property
    def duration_ms(self) -> int:
        """Duration of the session in milliseconds."""
        if self._start_time is None:
            return 0
        end = self._end_time or time.perf_counter()
        return int((end - self._start_time) * 1000)

    def start(self, input_data: dict[str, Any] | None = None) -> TraceEvent | None:
        """
        Start the session and emit a session_start event.

        Args:
            input_data: Initial input to the agent

        Returns:
            The session start event, or None on internal error (fail-open)
        """
        try:
            if self._is_active:
                logger.warning("session_already_active", session_id=str(self.session_id))

            self._start_time = time.perf_counter()
            self._is_active = True

            # Build metadata with agent_name for auto-registration
            metadata = {}
            if self.agent_name:
                metadata["agent_name"] = self.agent_name

            event = create_session_start_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                framework=self.framework,
                framework_version=self.framework_version,
                input_data=input_data if self.config.capture_inputs else None,
                metadata=metadata,
            )

            if event is not None:
                self._events.append(event)
            logger.debug(
                "session_started",
                session_id=str(self.session_id),
                agent_id=str(self.agent_id) if self.agent_id else None,
            )

            return event
        except Exception:
            # Fail-open: session start is infrastructure, must never block agent
            self._start_time = self._start_time or time.perf_counter()
            self._is_active = True
            logger.warning(
                "clyro_session_start_failed", session_id=str(self.session_id), fail_open=True
            )
            return None

    def end(
        self,
        output_data: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> TraceEvent | None:
        """
        End the session and emit a session_end event.

        Args:
            output_data: Final output from the agent
            error: Exception if the session ended with an error

        Returns:
            The session end event, or None on internal error (fail-open)
        """
        try:
            self._end_time = time.perf_counter()
            self._is_active = False
            self._error = error

            error_type = None
            error_message = None
            error_stack = None

            if error is not None:
                error_type = type(error).__name__
                error_message = str(error)
                error_stack = traceback.format_exc()

            event = create_session_end_event(
                session_id=self.session_id,
                agent_id=self.agent_id,
                step_number=self._step_number,
                cumulative_cost=self._cumulative_cost,
                output_data=output_data if self.config.capture_outputs else None,
                error_type=error_type,
                error_message=error_message,
                error_stack=error_stack,
                duration_ms=self.duration_ms,
                framework=self.framework,
                framework_version=self.framework_version,
            )

            if event is not None:
                self._events.append(event)
            logger.debug(
                "session_ended",
                session_id=str(self.session_id),
                steps=self._step_number,
                cost=str(self._cumulative_cost),
                duration_ms=self.duration_ms,
                error=error_type,
            )

            return event
        except Exception:
            # Fail-open: session end is infrastructure, must never block agent
            self._is_active = False
            self._error = error
            logger.warning(
                "clyro_session_end_failed", session_id=str(self.session_id), fail_open=True
            )
            return None

    def record_step(
        self,
        event_name: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        state_snapshot: dict[str, Any] | None = None,
        duration_ms: int = 0,
        cost_usd: Decimal = Decimal("0"),
    ) -> TraceEvent:
        """
        Record a step in the agent execution.

        This method:
        1. Increments the step counter
        2. Updates cumulative cost tracking
        3. Checks execution controls (step limit, cost limit, loop detection)
        4. Creates and stores a step event

        Execution Control Timing:
        - Controls are checked BEFORE creating the event
        - If a limit is exceeded, an exception is raised and no event is created
        - This ensures events accurately reflect what was actually executed

        Args:
            event_name: Name of the step (e.g., "agent_execution", "llm_call")
            input_data: Input to this step
            output_data: Output from this step
            state_snapshot: Agent state at this step (for loop detection)
            duration_ms: Duration of this step in milliseconds
            cost_usd: Cost incurred by this step in USD

        Returns:
            The created step event

        Raises:
            StepLimitExceededError: If step limit is exceeded
            CostLimitExceededError: If cost limit is exceeded
            LoopDetectedError: If an infinite loop is detected
        """
        # Increment step counter for this execution
        self._step_number += 1

        # Update cumulative cost with the cost from this step
        self._cumulative_cost += cost_usd

        # Check execution controls BEFORE creating event
        # If any limit is exceeded, these methods raise exceptions
        # and prevent the event from being created
        # ENFORCEMENT — these MUST raise
        self._check_step_limit()
        self._check_cost_limit()

        # Check for infinite loops if state snapshot is provided
        # Only runs if loop detection is enabled in config
        if state_snapshot is not None and self._loop_detection_enabled:
            self._check_loop_detection(state_snapshot, event_name)

        # Create step event with captured data (fail-open: event creation is infrastructure)
        try:
            event = create_step_event(
                session_id=self.session_id,
                step_number=self._step_number,
                event_name=event_name,
                agent_id=self.agent_id,
                input_data=input_data if self.config.capture_inputs else None,
                output_data=output_data if self.config.capture_outputs else None,
                state_snapshot=state_snapshot if self.config.capture_state else None,
                duration_ms=duration_ms,
                cumulative_cost=self._cumulative_cost,
                framework=self.framework,
                framework_version=self.framework_version,
            )

            # Store event in session history
            if event is not None:
                self._events.append(event)
            return event
        except Exception:
            logger.warning(
                "clyro_record_step_failed", session_id=str(self.session_id), fail_open=True
            )
            return None

    def record_event(self, event: TraceEvent) -> None:
        """
        Record a pre-created event.

        Use this for events created by adapters or external sources.
        Checks execution controls (step limit, cost limit) after recording.

        Args:
            event: The event to record

        Raises:
            StepLimitExceededError: If step limit is exceeded
            CostLimitExceededError: If cost limit is exceeded
        """
        # Infrastructure: event enrichment and appending (fail-open)
        cost_changed = False
        is_lifecycle_event = False
        try:
            # Update cumulative cost if event has cost
            cost_changed = event.cost_usd > 0
            if cost_changed:
                self._cumulative_cost += event.cost_usd

            # Update event with session context
            event.session_id = self.session_id

            # Sync cumulative cost: if the event carries a higher estimate
            # (e.g. from Claude Agent SDK handler's per-step estimator), adopt it.
            if event.cumulative_cost > self._cumulative_cost:
                self._cumulative_cost = event.cumulative_cost
            else:
                event.cumulative_cost = self._cumulative_cost

            # Inherit session framework for events that have the default (GENERIC),
            # e.g. policy_check events created by PolicyEvaluator.
            if event.framework == Framework.GENERIC and self.framework != Framework.GENERIC:
                event.framework = self.framework
                event.framework_version = self.framework_version

            is_lifecycle_event = event.event_type in (
                EventType.SESSION_START,
                EventType.SESSION_END,
            )

            # Policy checks are metadata about a step, not steps themselves.
            # They inherit the current step number without incrementing.
            is_policy_check = event.event_type == EventType.POLICY_CHECK

            if not is_lifecycle_event:
                if is_policy_check:
                    if event.step_number == 0:
                        event.step_number = self._step_number
                elif event.step_number == 0:
                    self._step_number += 1
                    event.step_number = self._step_number
                elif event.step_number > self._step_number:
                    # Sync session step counter with adapter's step counter
                    # (e.g., CrewAI handler tracks steps locally)
                    self._step_number = event.step_number

            self._events.append(event)
        except Exception:
            logger.warning(
                "clyro_record_event_failed", session_id=str(self.session_id), fail_open=True
            )
            return

        # ENFORCEMENT — these MUST raise
        if not is_lifecycle_event:
            self._check_step_limit()
            if cost_changed:
                self._check_cost_limit()

    def record_error(
        self,
        error: Exception,
        event_name: str | None = None,
    ) -> TraceEvent | None:
        """
        Record an error event.

        Args:
            error: The exception that occurred
            event_name: Optional name for the error event

        Returns:
            The created error event, or None on internal error (fail-open)
        """
        try:
            event = create_error_event(
                session_id=self.session_id,
                step_number=self._step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self.agent_id,
                error_stack=traceback.format_exc(),
                cumulative_cost=self._cumulative_cost,
                metadata={"event_name": event_name} if event_name else None,
                framework=self.framework,
                framework_version=self.framework_version,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )

            if event is not None:
                self._events.append(event)
            return event
        except Exception:
            logger.warning(
                "clyro_record_error_failed", session_id=str(self.session_id), fail_open=True
            )
            return None

    def add_cost(self, cost_usd: Decimal) -> None:
        """
        Add cost to the session without creating an event.

        Args:
            cost_usd: Cost to add in USD
        """
        self._cumulative_cost += cost_usd
        self._check_cost_limit()

    def estimate_call_cost(
        self,
        model: str,
        input_data: dict[str, Any] | str,
        max_tokens: int = 1000,
        safety_margin: float = 1.2,
    ) -> Decimal:
        """
        Estimate cost of an LLM call before execution.

        This method provides predictive cost estimation to enable proactive
        cost management. It uses tiktoken to estimate input tokens and assumes
        worst-case output token usage based on max_tokens parameter.

        Args:
            model: Model identifier (e.g., "gpt-4-turbo", "claude-3-sonnet")
            input_data: Input prompt/payload (dict or string)
            max_tokens: Maximum output tokens (worst case estimate). Default: 1000
            safety_margin: Multiplier for conservative estimate. Default: 1.2 (20% buffer)

        Returns:
            Estimated cost in USD as Decimal

        Example:
            ```python
            # Check if next call would exceed budget
            estimated = session.estimate_call_cost("gpt-4-turbo", prompt, max_tokens=1500)
            if session.cumulative_cost + estimated > Decimal("10.0"):
                logger.warning("Predicted cost exceeds limit, switching to cheaper model")
                model = "gpt-4o-mini"  # 97% cheaper
            ```
        """
        calculator = CostCalculator(self.config)

        # Serialize input_data to text for token estimation
        input_text = self._serialize_for_token_estimate(input_data)

        # Estimate input tokens using tiktoken
        from clyro.cost import TiktokenEstimator

        estimated_input = 0
        if input_text:
            token_count = TiktokenEstimator.count_tokens(input_text)
            if token_count is not None:
                estimated_input = token_count

        # Worst case: assume max_tokens are fully used for output
        estimated_output = max_tokens

        # Calculate base cost
        estimated_cost = calculator.calculate(estimated_input, estimated_output, model)

        # Apply safety margin for conservative estimate
        final_estimate = estimated_cost * Decimal(str(safety_margin))

        logger.debug(
            "cost_estimated",
            session_id=str(self.session_id),
            model=model,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            base_cost=str(estimated_cost),
            final_estimate=str(final_estimate),
            safety_margin=safety_margin,
        )

        return final_estimate

    def record_llm_call(
        self,
        model: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        llm_response: Any | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int = 0,
        generation_params: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """
        Record an LLM call with automatic cost calculation.

        This method extracts token usage and calculates costs from LLM responses.
        Cost extraction is attempted in this order:
        1. From llm_response if provided (auto-extraction)
        2. From explicit input_tokens/output_tokens if provided
        3. Zero cost if neither available

        Args:
            model: Model identifier (e.g., "gpt-4-turbo", "claude-3-sonnet")
            input_data: Input/prompt data sent to the LLM
            output_data: Output/completion data from the LLM
            llm_response: Raw LLM response object for automatic token extraction
            input_tokens: Explicit input token count (used if llm_response not available)
            output_tokens: Explicit output token count (used if llm_response not available)
            duration_ms: Duration of the LLM call in milliseconds
            generation_params: LLM generation parameters (optional) for cost analysis:
                - temperature: float (0.0-2.0) - Sampling temperature
                - top_p: float (0.0-1.0) - Nucleus sampling threshold
                - max_tokens: int - Maximum output tokens
                - frequency_penalty: float (-2.0 to 2.0) - Penalize frequent tokens
                - presence_penalty: float (-2.0 to 2.0) - Penalize existing tokens

        Returns:
            The created LLM call event

        Raises:
            StepLimitExceededError: If step limit is exceeded
            CostLimitExceededError: If cost limit is exceeded
        """
        # Increment step counter
        self._step_number += 1

        # Infrastructure: cost calculation (fail-open)
        cost_usd = Decimal("0")
        token_input = input_tokens or 0
        token_output = output_tokens or 0
        cost_tracking_available = False
        try:
            token_usage: TokenUsage | None = None
            calculator = CostCalculator(self.config)

            if llm_response is not None:
                cost_usd, token_usage = calculator.calculate_from_response(
                    llm_response, fallback_model=model
                )
                if token_usage:
                    token_input = token_usage.input_tokens
                    token_output = token_usage.output_tokens
                    cost_tracking_available = True
            elif input_tokens is not None or output_tokens is not None:
                cost_usd = calculator.calculate(token_input, token_output, model)
                cost_tracking_available = True
            elif self.config.controls.enable_cost_limit:
                input_text = self._serialize_for_token_estimate(input_data)
                output_text = self._serialize_for_token_estimate(output_data)
                estimated_cost, estimated_usage = calculator.calculate_from_text(
                    input_text=input_text,
                    output_text=output_text,
                    model=model,
                )
                if estimated_usage is not None:
                    cost_usd = estimated_cost
                    token_input = estimated_usage.input_tokens
                    token_output = estimated_usage.output_tokens
                    cost_tracking_available = True
                    logger.debug(
                        "cost_estimated_from_payload",
                        session_id=str(self.session_id),
                        model=model,
                        step_number=self._step_number,
                        input_tokens=token_input,
                        output_tokens=token_output,
                        cost_usd=str(cost_usd),
                    )
        except Exception:
            logger.warning(
                "clyro_cost_calculation_failed", session_id=str(self.session_id), fail_open=True
            )

        # Update cumulative cost
        self._cumulative_cost += cost_usd

        # ENFORCEMENT — these MUST raise
        self._check_step_limit()
        if cost_tracking_available:
            self._check_cost_limit()
        else:
            if self.config.controls.enable_cost_limit:
                logger.warning(
                    "cost_tracking_unavailable",
                    session_id=str(self.session_id),
                    model=model,
                    step_number=self._step_number,
                    hint="Install tiktoken for fallback estimation: pip install clyro[tiktoken]",
                )

        # ENFORCEMENT — policy check MUST raise PolicyViolationError
        output_text = ""
        if output_data is not None:
            if isinstance(output_data, dict):
                output_text = output_data.get("content", "") or str(output_data)
            elif isinstance(output_data, str):
                output_text = output_data
            else:
                output_text = str(output_data)
        self.check_policy(
            "llm_call",
            {
                "model": model,
                "cost": float(self._cumulative_cost),
                "step_number": self._step_number,
                "input": str(input_data) if input_data else "",
                "output": output_text,
            },
        )

        # Infrastructure: event creation (fail-open)
        try:
            metadata = {}
            if generation_params is not None:
                metadata["generation_params"] = generation_params

            if self.config.capture_outputs and output_data is not None:
                event_output_data = (
                    output_data if isinstance(output_data, dict) else {"content": output_text}
                )
            else:
                event_output_data = None if self.config.capture_outputs else None

            event = create_llm_call_event(
                session_id=self.session_id,
                step_number=self._step_number,
                model=model,
                input_data=input_data if self.config.capture_inputs else {},
                output_data=event_output_data,
                agent_id=self.agent_id,
                token_count_input=token_input,
                token_count_output=token_output,
                cost_usd=cost_usd,
                cumulative_cost=self._cumulative_cost,
                duration_ms=duration_ms,
                metadata=metadata if metadata else None,
                framework=self.framework,
                framework_version=self.framework_version,
            )

            if event is not None:
                self._events.append(event)
            logger.debug(
                "llm_call_recorded",
                model=model,
                input_tokens=token_input,
                output_tokens=token_output,
                cost_usd=str(cost_usd),
                cumulative_cost=str(self._cumulative_cost),
            )

            return event
        except Exception:
            logger.warning(
                "clyro_record_llm_call_failed", session_id=str(self.session_id), fail_open=True
            )
            return None

    def _check_step_limit(self) -> None:
        """Check if step limit is exceeded.  # Implements PRD-009"""
        if not self.config.controls.enable_step_limit:
            return

        max_steps = self.config.controls.max_steps
        if self._step_number > max_steps:
            raise StepLimitExceededError(
                limit=max_steps,
                current_step=self._step_number,
                session_id=str(self.session_id),
            )

    def _check_cost_limit(self) -> None:
        """Check if cost limit is exceeded.  # Implements PRD-010"""
        if not self.config.controls.enable_cost_limit:
            return

        max_cost = Decimal(str(self.config.controls.max_cost_usd))
        if self._cumulative_cost > max_cost:
            raise CostLimitExceededError(
                limit_usd=float(max_cost),
                current_cost_usd=float(self._cumulative_cost),
                session_id=str(self.session_id),
                step_number=self._step_number,
            )

    def _check_loop_detection(self, state: dict[str, Any], action: str | None) -> None:
        """Check for infinite loops via state hash comparison.  # Implements PRD-010"""
        if not self._loop_detector or not self._loop_detection_enabled:
            return

        state_hash = self._loop_detector.compute_state_hash(state)
        if state_hash is None:
            logger.warning(
                "loop_detection_skipped_unhashable_state",
                session_id=str(self.session_id),
                step_number=self._step_number,
            )
            return

        self._loop_detector.check(
            state=state,
            state_hash=state_hash,
            action=action,
            session_id=str(self.session_id),
        )

    def check_policy(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        parent_event_id: UUID | None = None,
        cumulative_cost: Decimal | None = None,
    ) -> None:
        """
        Check an action against policies (synchronous).

        This is the universal hook for policy enforcement. Called by:
        - Wrapper before agent execution ("agent_execution")
        - LangGraph adapter before tool calls ("tool_call")
        - LangGraph adapter before LLM calls ("llm_call")
        - CrewAI adapter after tool usage ("tool_call")

        No-op if policy enforcement is not configured.

        Args:
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            parent_event_id: Parent event ID for linking policy checks
                to the tool_call or llm_call they evaluated
            cumulative_cost: Real-time cumulative cost from adapter's local
                tracking. Stamped directly on policy events to avoid stale
                session cost (session._cumulative_cost is only updated when
                adapter events are drained post-execution).

        Raises:
            PolicyViolationError: If the action is blocked by a policy
        """
        if self._policy_evaluator is None:
            return

        try:
            self._policy_evaluator.evaluate_sync(
                action_type=action_type,
                parameters=parameters or {},
                session_id=self.session_id,
                step_number=self._step_number,
            )
        finally:
            # Drain audit events for both allow and block outcomes (fail-open)
            try:
                for event in self._policy_evaluator.drain_events():
                    if parent_event_id is not None:
                        event.parent_event_id = parent_event_id
                    self.record_event(event)
                    if cumulative_cost is not None:
                        event.cumulative_cost = cumulative_cost
                    if self._event_sink is not None:
                        self._event_sink(event)
            except Exception:
                logger.warning(
                    "clyro_policy_event_drain_failed",
                    session_id=str(self.session_id),
                    fail_open=True,
                )

    async def check_policy_async(
        self,
        action_type: str,
        parameters: dict[str, Any] | None = None,
        parent_event_id: UUID | None = None,
        cumulative_cost: Decimal | None = None,
    ) -> None:
        """
        Check an action against policies (asynchronous).

        Async equivalent of check_policy for use in async execution paths.

        Args:
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            parent_event_id: Parent event ID for linking policy checks
                to the tool_call or llm_call they evaluated
            cumulative_cost: Real-time cumulative cost from adapter's local
                tracking. Stamped directly on policy events.

        Raises:
            PolicyViolationError: If the action is blocked by a policy
        """
        if self._policy_evaluator is None:
            return

        try:
            await self._policy_evaluator.evaluate_async(
                action_type=action_type,
                parameters=parameters or {},
                session_id=self.session_id,
                step_number=self._step_number,
            )
        finally:
            # Drain audit events for both allow and block outcomes (fail-open)
            try:
                for event in self._policy_evaluator.drain_events():
                    if parent_event_id is not None:
                        event.parent_event_id = parent_event_id
                    self.record_event(event)
                    if cumulative_cost is not None:
                        event.cumulative_cost = cumulative_cost
                    if self._event_sink is not None:
                        self._event_sink(event)
            except Exception:
                logger.warning(
                    "clyro_policy_event_drain_failed",
                    session_id=str(self.session_id),
                    fail_open=True,
                )

    @staticmethod
    def _serialize_for_token_estimate(payload: Any) -> str | None:
        """Serialize payload data for token estimation."""
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(payload)

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the session.

        Returns:
            Dictionary with session summary information
        """
        return {
            "session_id": str(self.session_id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "org_id": str(self.org_id) if self.org_id else None,
            "framework": self.framework.value,
            "framework_version": self.framework_version,
            "is_active": self._is_active,
            "step_count": self._step_number,
            "event_count": len(self._events),
            "cumulative_cost_usd": float(self._cumulative_cost),
            "duration_ms": self.duration_ms,
            "has_error": self._error is not None,
            "error_type": type(self._error).__name__ if self._error else None,
        }


# Session context management using contextvars for thread-safety
_current_session: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "clyro_current_session", default=None
)


def get_current_session() -> Session | None:
    """
    Get the current active session, if any.

    Thread-safe: Uses contextvars to isolate session state per execution context.
    """
    return _current_session.get()


def set_current_session(session: Session | None) -> None:
    """
    Set the current session.

    Thread-safe: Uses contextvars to isolate session state per execution context.
    This ensures that concurrent agent executions don't interfere with each other.
    """
    _current_session.set(session)
