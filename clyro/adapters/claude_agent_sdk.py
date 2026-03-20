# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Claude Agent SDK Adapter
# Implements FRD-001 through FRD-013, NFR-001 through NFR-006

"""
Adapter for Claude Agent SDK (claude-agent-sdk v0.1.x).

This adapter integrates Clyro instrumentation into the Claude Agent SDK's
hook system, translating hook events into Clyro TraceEvents with full
support for policy enforcement and execution controls.

Integration point: ClaudeAgentOptions.hooks (declarative hook registration).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog

from clyro.exceptions import (
    CostLimitExceededError,
    FrameworkVersionError,
    LoopDetectedError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.trace import (
    AgentStage,
    EventType,
    Framework,
    TraceEvent,
    compute_state_hash,
    create_error_event,
    create_llm_call_event,
    create_session_end_event,
    create_session_start_event,
    create_state_transition_event,
    create_step_event,
    create_tool_call_event,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.policy import PolicyEvaluator
    from clyro.session import Session

logger = structlog.get_logger(__name__)

# Constants
MIN_SDK_VERSION = "0.1.40"
MAX_CORRELATOR_SIZE = 1_000
MAX_TRACKER_SIZE = 100
CORRELATOR_EVICT_COUNT = 500
TRACKER_EVICT_COUNT = 50
TRUNCATION_LIMIT = 10_000
ERROR_TRUNCATION_LIMIT = 2_000
POLICY_TIMEOUT_SECONDS = 2.0

# All 9 hook types supported by Claude Agent SDK v0.1.x
HOOK_TYPES = frozenset(
    {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "SubagentStart",
        "SubagentStop",
        "UserPromptSubmit",
        "Stop",
        "Notification",
        "PreCompact",
    }
)

# Marker to detect Clyro hooks during idempotency check
_CLYRO_HOOK_MARKER = "_clyro_instrumented"


# --- Data Structures (§3.3) ---


@dataclass
class PendingToolUse:
    """Stored in ToolUseCorrelator for Pre->Post matching."""

    event_id: UUID
    start_time: float  # time.monotonic()
    tool_name: str


@dataclass
class ActiveSubagent:
    """Stored in SubagentTracker for Start->Stop matching."""

    event_id: UUID
    start_time: float  # time.monotonic()
    agent_type: str


@dataclass
class CostEstimator:
    """Character-length heuristic cost estimation (FRD-011b).

    The Claude Agent SDK communicates with Claude Code CLI via subprocess
    and does not expose per-call token usage in hook callbacks. Real cost
    data is only available in ResultMessage at session end.

    This estimator uses the same character-length heuristic as the MCP wrapper:
        estimated_tokens = len(content) / 4  (approx 4 chars per token)
        estimated_cost   = estimated_tokens * cost_per_token_usd

    Hook handlers pass the content available at each hook (tool inputs,
    tool outputs, prompts, error messages) to accumulate a running cost
    estimate that enables real-time cost-based policy enforcement.
    """

    cost_per_token_usd: Decimal
    step_count: int = 0
    estimated_cumulative_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    _last_step_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Characters-per-token divisor (same heuristic as MCP wrapper)
    CHARS_PER_TOKEN: int = 4

    def estimate_content_cost(self, content: str | None) -> Decimal:
        """Estimate cost from content character length.

        Args:
            content: Text content (tool input/output, prompt, error, etc.)

        Returns:
            Estimated cost in USD for this content.
        """
        if not content:
            return Decimal("0")
        estimated_tokens = len(content) / self.CHARS_PER_TOKEN
        return Decimal(str(estimated_tokens)) * self.cost_per_token_usd

    def accumulate(self, *contents: str | None) -> Decimal:
        """Accumulate estimated cost from one or more content strings.

        Concatenates all non-None content lengths, estimates tokens,
        calculates cost, and adds to cumulative total.

        Args:
            *contents: Variable content strings (tool input, output, etc.)

        Returns:
            New estimated cumulative cost.
        """
        self.step_count += 1
        total_len = sum(len(c) for c in contents if c)
        if total_len > 0:
            estimated_tokens = Decimal(str(total_len / self.CHARS_PER_TOKEN))
            step_cost = estimated_tokens * self.cost_per_token_usd
        else:
            step_cost = Decimal("0")
        self._last_step_cost = step_cost
        self.estimated_cumulative_cost += step_cost
        return self.estimated_cumulative_cost

    @property
    def last_step_cost(self) -> Decimal:
        """Cost estimated for the most recent step."""
        return self._last_step_cost

    def reset(self) -> None:
        """Reset estimator state."""
        self.step_count = 0
        self.estimated_cumulative_cost = Decimal("0")
        self._last_step_cost = Decimal("0")
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# --- C3: ToolUseCorrelator (FRD-006) ---


class ToolUseCorrelator:
    """Correlate PreToolUse events with PostToolUse/PostToolUseFailure via tool_use_id.

    Implements FRD-006.
    """

    def __init__(self) -> None:
        self._pending: OrderedDict[str, PendingToolUse] = OrderedDict()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def start(
        self,
        tool_use_id: str | None,
        event_id: UUID,
        tool_name: str,
    ) -> str:
        """Store a PreToolUse entry. Returns the (possibly synthetic) tool_use_id."""
        if not tool_use_id:
            tool_use_id = str(uuid4())
            logger.warning(
                "clyro_synthetic_tool_use_id",
                synthetic_id=tool_use_id,
                tool_name=tool_name,
            )

        if tool_use_id in self._pending:
            logger.warning(
                "clyro_duplicate_tool_use_id",
                tool_use_id=tool_use_id,
                tool_name=tool_name,
            )

        self._pending[tool_use_id] = PendingToolUse(
            event_id=event_id,
            start_time=time.monotonic(),
            tool_name=tool_name,
        )

        self._check_overflow()
        return tool_use_id

    def complete(self, tool_use_id: str | None) -> tuple[UUID | None, int | None]:
        """Lookup and remove a pending entry. Returns (parent_event_id, duration_ms)."""
        if not tool_use_id or tool_use_id not in self._pending:
            if tool_use_id:
                logger.warning(
                    "clyro_correlator_miss",
                    tool_use_id=tool_use_id,
                )
            return None, None

        entry = self._pending.pop(tool_use_id)
        duration_ms = int((time.monotonic() - entry.start_time) * 1000)
        return entry.event_id, duration_ms

    def flush(self) -> None:
        """Log and discard all pending entries (W5a step 1)."""
        if self._pending:
            for tid, entry in self._pending.items():
                logger.warning(
                    "clyro_orphaned_tool_use",
                    tool_use_id=tid,
                    tool_name=entry.tool_name,
                )
            self._pending.clear()

    def _check_overflow(self) -> None:
        if len(self._pending) > MAX_CORRELATOR_SIZE:
            logger.error(
                "clyro_correlator_overflow",
                size=len(self._pending),
                evicting=CORRELATOR_EVICT_COUNT,
            )
            for _ in range(CORRELATOR_EVICT_COUNT):
                if self._pending:
                    self._pending.popitem(last=False)


# --- C4: SubagentTracker (FRD-007) ---


class SubagentTracker:
    """Track parent-child agent relationships for subagent hierarchy.

    Implements FRD-007.
    """

    def __init__(self) -> None:
        self._active: OrderedDict[str, ActiveSubagent] = OrderedDict()

    @property
    def active_count(self) -> int:
        return len(self._active)

    def start(self, agent_id: str, event_id: UUID, agent_type: str) -> None:
        """Store a SubagentStart entry."""
        self._active[agent_id] = ActiveSubagent(
            event_id=event_id,
            start_time=time.monotonic(),
            agent_type=agent_type,
        )
        self._check_overflow()

    def stop(self, agent_id: str) -> tuple[UUID | None, int | None]:
        """Lookup and remove subagent entry. Returns (parent_event_id, duration_ms)."""
        if agent_id not in self._active:
            logger.warning(
                "clyro_subagent_tracker_miss",
                agent_id=agent_id,
            )
            return None, None

        entry = self._active.pop(agent_id)
        duration_ms = int((time.monotonic() - entry.start_time) * 1000)
        return entry.event_id, duration_ms

    def flush(self) -> None:
        """Log and discard all active entries (W5a step 2)."""
        if self._active:
            for aid, entry in self._active.items():
                logger.warning(
                    "clyro_orphaned_subagent",
                    agent_id=aid,
                    agent_type=entry.agent_type,
                )
            self._active.clear()

    def _check_overflow(self) -> None:
        if len(self._active) > MAX_TRACKER_SIZE:
            logger.error(
                "clyro_subagent_tracker_overflow",
                size=len(self._active),
                evicting=TRACKER_EVICT_COUNT,
            )
            for _ in range(TRACKER_EVICT_COUNT):
                if self._active:
                    self._active.popitem(last=False)


# --- C2: ClaudeAgentHandler (FRD-002 through FRD-013) ---


class ClaudeAgentHandler:
    """Stateful handler that processes Claude Agent SDK hook events.

    Dispatches hook events to the appropriate handler method, manages
    session lifecycle, and delegates to ToolUseCorrelator / SubagentTracker.

    Implements FRD-002, FRD-003, FRD-004, FRD-005, FRD-008, FRD-009,
    FRD-010, FRD-011a, FRD-011b, FRD-011c, FRD-012, FRD-013.
    """

    def __init__(
        self,
        config: ClyroConfig,
        framework_version: str,
        session: Session | None = None,
        policy_evaluator: PolicyEvaluator | None = None,
        cost_per_token_usd: Decimal | str = "0.00001",
    ) -> None:
        self._config = config
        self._framework_version = framework_version
        self._session = session
        self._policy_evaluator = policy_evaluator

        # Session lifecycle state
        self._session_started = False
        self._session_id: str | None = None
        self._session_ended = False
        self._start_time: float = 0.0

        # Correlation and tracking (C3, C4)
        self._correlator = ToolUseCorrelator()
        self._subagent_tracker = SubagentTracker()

        # Execution controls (FRD-011a/b/c)
        self._step_count = 0
        self._cost_estimator = CostEstimator(cost_per_token_usd=Decimal(str(cost_per_token_usd)))
        self._loop_detector: Any = None  # Lazily imported; set on first use
        self._last_user_prompt: str = ""  # Last user prompt for policy evaluation

        # Accumulated events (drained by WrappedAgent or end_session)
        self._events: list[TraceEvent] = []

        # When True, session was injected by clyro.wrap() — skip duplicate
        # lifecycle events and let session.record_event() assign step_numbers.
        self._is_wrapped = False

        # Deferred error for WrappedAgent to re-raise after execution.
        # When a policy block or execution control error stops the agent
        # via continue_=False, we store the error here so the wrapper can
        # surface it to the user (same pattern as CrewAI's _pending_error).
        self._pending_error: (
            PolicyViolationError
            | StepLimitExceededError
            | CostLimitExceededError
            | LoopDetectedError
            | None
        ) = None

        # Dispatch table (built once)
        self._handlers = {
            "PreToolUse": self._handle_pre_tool_use,
            "PostToolUse": self._handle_post_tool_use,
            "PostToolUseFailure": self._handle_post_tool_use_failure,
            "UserPromptSubmit": self._handle_user_prompt_submit,
            "SubagentStart": self._handle_subagent_start,
            "SubagentStop": self._handle_subagent_stop,
            "Stop": self._handle_stop,
            "Notification": self._handle_notification,
            "PreCompact": self._handle_pre_compact,
        }

    def _reset_for_invocation(self, cost_per_token_usd: Decimal | None = None) -> None:
        """Reset all per-invocation state for handler reuse across calls.

        The handler is stored on ``options._clyro_handler`` and reused across
        ``invoke()`` calls.  Each call runs in a fresh ``asyncio.run()`` event
        loop, so every piece of mutable state accumulated during the previous
        run is stale and must be cleared before the next run begins.
        """
        self._session_started = False
        self._session_id = None
        self._session_ended = False
        self._start_time = 0.0
        self._step_count = 0
        self._cost_estimator = CostEstimator(
            cost_per_token_usd=cost_per_token_usd or self._cost_estimator.cost_per_token_usd
        )
        self._correlator = ToolUseCorrelator()
        self._subagent_tracker = SubagentTracker()
        self._loop_detector = None
        self._last_user_prompt = ""
        self._events.clear()
        self._pending_error = None

    # --- Public hook entry point ---

    async def handle_hook(
        self,
        hook_type: str,
        input_data: dict[str, Any],
        tool_use_id: str | None = None,
        context: Any = None,
    ) -> dict[str, Any] | None:
        """Entry point called by registered hook callbacks. Wraps _safe_hook."""
        return await self._safe_hook(hook_type, input_data, tool_use_id, context)

    async def _safe_hook(
        self,
        hook_type: str,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any] | None:
        """Error-isolating wrapper around _dispatch (§6.2)."""
        try:
            # B-02: Guard against missing session_id
            session_id = input_data.get("session_id")
            if not session_id:
                logger.error("clyro_hook_missing_session_id", hook_type=hook_type)
                return None
            return await self._dispatch(hook_type, input_data, tool_use_id)
        except (StepLimitExceededError, CostLimitExceededError, LoopDetectedError) as e:
            # Execution control errors must stop the agent entirely —
            # matches Anthropic/LangGraph/CrewAI where these propagate up.
            # Store for WrappedAgent to re-raise after SDK execution returns.
            self._pending_error = e
            return self._deny_response(hook_type, str(e), stop_agent=True)
        except PolicyViolationError as e:
            # Policy block must stop the agent entirely —
            # matches Anthropic/LangGraph/CrewAI where PolicyViolationError
            # stops execution, not just denies a single tool call.
            # Store for WrappedAgent to re-raise after SDK execution returns.
            self._pending_error = e
            return self._deny_response(hook_type, str(e), stop_agent=True)
        except Exception:
            logger.exception("clyro_hook_error", hook_type=hook_type)
            return None  # Fail-open

    async def _dispatch(
        self,
        hook_type: str,
        input_data: dict[str, Any],
        tool_use_id: str | None,
    ) -> dict[str, Any] | None:
        """Route hook event to the appropriate handler method."""
        session_id = input_data.get("session_id", "")

        # W1: Session initialization
        self._ensure_session(session_id, input_data)

        handler = self._handlers.get(hook_type)
        if handler is None:
            logger.warning("clyro_unknown_hook_type", hook_type=hook_type)
            return None

        return await handler(input_data, tool_use_id)

    # --- W1: Session Initialization ---

    def _ensure_session(self, session_id: str, input_data: dict[str, Any]) -> None:
        """Synthesize SESSION_START on first hook or handle session_id change (W1/W5a)."""
        if not self._session_started:
            self._start_new_session(session_id, input_data)
        elif session_id != self._session_id:
            # When wrapped, the Claude CLI's session_id naturally differs from
            # the Clyro session. Ignore session_id changes in wrapped mode —
            # the wrapper owns the session lifecycle.
            if not self._is_wrapped:
                self._handle_session_id_change(session_id, input_data)
            else:
                self._session_id = session_id  # Track for logging only

    def _start_new_session(self, session_id: str, input_data: dict[str, Any]) -> None:
        """Create Session and emit SESSION_START."""
        self._session_id = session_id
        self._session_started = True
        self._session_ended = False
        self._start_time = time.monotonic()

        # Create session if not provided externally (standalone usage)
        if self._session is None:
            from clyro.session import Session

            self._session = Session(
                config=self._config,
                framework=Framework.CLAUDE_AGENT_SDK,
                framework_version=self._framework_version,
            )

        # When wrapped, the wrapper already emits session_start — skip duplicate.
        if self._is_wrapped:
            return

        cwd = input_data.get("cwd", "")
        event = create_session_start_event(
            session_id=self._session.session_id,
            agent_id=self._session.agent_id,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            input_data={"cwd": cwd, "session_id": session_id}
            if self._config.capture_inputs
            else None,
            metadata={"hook_event": "session_start_synthesized"},
            agent_stage=AgentStage.THINK,
        )
        self._record_event(event)

    # --- W5a: Session ID Change Cleanup (FRD-009) ---

    def _handle_session_id_change(self, new_session_id: str, input_data: dict[str, Any]) -> None:
        """Flush correlator/tracker, emit SESSION_END, start new session."""
        logger.info(
            "clyro_session_id_change",
            old_session_id=self._session_id,
            new_session_id=new_session_id,
        )

        # Step 1: Flush correlator
        self._correlator.flush()

        # Step 2: Flush subagent tracker
        self._subagent_tracker.flush()

        # Step 3: Emit SESSION_END for active session
        if self._session and not self._session_ended:
            duration_ms = int((time.monotonic() - self._start_time) * 1000)
            end_event = create_session_end_event(
                session_id=self._session.session_id,
                agent_id=self._session.agent_id,
                step_number=self._event_step_number,
                duration_ms=duration_ms,
                metadata={"end_reason": "session_id_change"},
                framework=Framework.CLAUDE_AGENT_SDK,
                framework_version=self._framework_version,
            )
            self._record_event(end_event)
            self._session_ended = True

        # Step 4: Reset state and start new session
        self._step_count = 0
        self._cost_estimator.reset()
        self._session = None
        self._session_started = False
        self._start_new_session(new_session_id, input_data)

    # --- W2: PreToolUse (FRD-002) ---

    async def _handle_pre_tool_use(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle PreToolUse hook: execution controls, policy, event creation."""
        tool_name = input_data.get("tool_name") or "unknown"
        if tool_name == "unknown":
            logger.warning("clyro_empty_tool_name", hook="PreToolUse")

        tool_input = input_data.get("tool_input") or {}

        # 1a: Resolve tool_use_id (correlator generates synthetic ID if missing)
        tool_use_id = tool_use_id or input_data.get("tool_use_id") or ""

        # 3: Step limit (FRD-011a)
        self._step_count += 1
        if (
            self._config.controls.enable_step_limit
            and self._step_count > self._config.controls.max_steps
        ):
            msg = f"Clyro step limit exceeded ({self._config.controls.max_steps} steps)"
            self._record_error_event("StepLimitExceeded", msg)
            raise StepLimitExceededError(
                limit=self._config.controls.max_steps,
                current_step=self._step_count,
            )

        # 4: Cost estimation (character-length heuristic) + limit enforcement (FRD-011b)
        tool_input_str = json.dumps(tool_input, default=str) if tool_input else None
        estimated_cost = self._cost_estimator.accumulate(tool_input_str)
        if self._config.controls.enable_cost_limit:
            max_cost = Decimal(str(self._config.controls.max_cost_usd))
            if estimated_cost > max_cost:
                msg = f"Clyro cost limit exceeded (${max_cost})"
                self._record_error_event("CostLimitExceeded", msg)
                raise CostLimitExceededError(
                    limit_usd=float(max_cost),
                    current_cost_usd=float(estimated_cost),
                )

        # 5: Loop detection (FRD-011c)
        if self._config.controls.enable_loop_detection:
            try:
                state_for_hash = {"tool_name": tool_name, "tool_input": tool_input}
                state_hash = compute_state_hash(state_for_hash)
                if state_hash and self._session:
                    from clyro.loop_detector import LoopDetector

                    if self._loop_detector is None:
                        self._loop_detector = LoopDetector(
                            threshold=self._config.controls.loop_detection_threshold,
                        )
                    self._loop_detector.check(
                        state=state_for_hash,
                        state_hash=state_hash,
                        action=tool_name,
                        raise_on_loop=True,
                    )
            except LoopDetectedError:
                self._record_error_event(
                    "LoopDetected", "Clyro loop detected: same tool call repeated"
                )
                raise
            except Exception:
                logger.warning("clyro_loop_detection_error", tool_name=tool_name)

        # 7: Pre-generate event_id so policy check events can reference
        # the tool_call they guard as their parent.
        event_id = uuid4()

        # 6: Policy evaluation (FRD-010)
        deny_response = await self._evaluate_policy(tool_name, tool_input, parent_event_id=event_id)
        if deny_response is not None:
            return deny_response

        # 8: Store in correlator (may generate synthetic tool_use_id; use returned value)
        resolved_tool_use_id = self._correlator.start(tool_use_id, event_id, tool_name)

        # Estimated token count from input content
        estimated_input_tokens = (
            len(tool_input_str) // self._cost_estimator.CHARS_PER_TOKEN if tool_input_str else 0
        )
        self._cost_estimator.total_input_tokens += estimated_input_tokens

        event = create_tool_call_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            tool_name=tool_name,
            input_data=self._capture_input({"tool_name": tool_name, "tool_input": tool_input}),
            agent_id=self._session.agent_id,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            token_count_input=estimated_input_tokens,
            cost_usd=self._cost_estimator.last_step_cost,
            metadata={
                "tool_use_id": resolved_tool_use_id,
                "hook_event": "PreToolUse",
                "agent_id": input_data.get("agent_id"),
                "agent_type": input_data.get("agent_type"),
                "cost_estimated": True,
            },
            agent_stage=AgentStage.ACT,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            event_id=event_id,
        )
        self._record_event(event)

        return None  # Allow

    # --- W2 continued: PostToolUse (FRD-003) ---

    async def _handle_post_tool_use(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle PostToolUse hook: correlate with PreToolUse, create OBSERVE event."""
        tool_use_id = tool_use_id or input_data.get("tool_use_id")
        tool_name = input_data.get("tool_name", "unknown")
        # SDK field is "tool_response" (PostToolUseHookInput), not "tool_output"
        tool_output = input_data.get("tool_response") or input_data.get("tool_output")

        # Accumulate cost from tool response content
        tool_output_str = json.dumps(tool_output, default=str) if tool_output else None
        self._cost_estimator.accumulate(tool_output_str)

        parent_event_id, duration_ms = self._correlator.complete(tool_use_id)

        output_data = None
        if self._config.capture_outputs and tool_output is not None:
            output_data = self._safe_serialize_output(tool_output)

        # Estimated token count from output content
        estimated_output_tokens = (
            len(tool_output_str) // self._cost_estimator.CHARS_PER_TOKEN if tool_output_str else 0
        )
        self._cost_estimator.total_output_tokens += estimated_output_tokens

        event = create_tool_call_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            tool_name=tool_name,
            input_data=None,  # Input already captured in PreToolUse
            output_data=output_data,
            agent_id=self._session.agent_id,
            duration_ms=duration_ms or 0,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            token_count_output=estimated_output_tokens,
            cost_usd=self._cost_estimator.last_step_cost,
            metadata={
                "tool_use_id": tool_use_id or "",
                "hook_event": "PostToolUse",
                "cost_estimated": True,
            },
            agent_stage=AgentStage.OBSERVE,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            parent_event_id=parent_event_id,
        )
        self._record_event(event)
        return None

    # --- PostToolUseFailure (FRD-004) ---

    async def _handle_post_tool_use_failure(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle PostToolUseFailure: create ERROR (OBSERVE) event."""
        tool_use_id = tool_use_id or input_data.get("tool_use_id")
        tool_name = input_data.get("tool_name", "unknown")
        # SDK field is "error" (PostToolUseFailureHookInput), not "error_message"
        error_message = (
            input_data.get("error")
            or input_data.get("error_message")
            or "Unknown tool execution error"
        )
        if not (input_data.get("error") or input_data.get("error_message")):
            logger.warning("clyro_empty_error_message", tool_name=tool_name)

        # Accumulate cost from error content
        self._cost_estimator.accumulate(str(error_message) if error_message else None)

        parent_event_id, duration_ms = self._correlator.complete(tool_use_id)

        # Estimated token count from error content
        error_str = str(error_message) if error_message else ""
        estimated_output_tokens = (
            len(error_str) // self._cost_estimator.CHARS_PER_TOKEN if error_str else 0
        )
        self._cost_estimator.total_output_tokens += estimated_output_tokens

        event = create_error_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            error_type="ToolExecutionError",
            error_message=_truncate(error_message, ERROR_TRUNCATION_LIMIT),
            agent_id=self._session.agent_id,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            token_count_output=estimated_output_tokens,
            cost_usd=self._cost_estimator.last_step_cost,
            input_data={"tool_name": tool_name, "tool_use_id": tool_use_id or ""},
            output_data={
                "error_type": "ToolExecutionError",
                "error_message": _truncate(error_message, ERROR_TRUNCATION_LIMIT),
            },
            metadata={
                "tool_use_id": tool_use_id or "",
                "hook_event": "PostToolUseFailure",
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "cost_estimated": True,
            },
            agent_stage=AgentStage.OBSERVE,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            parent_event_id=parent_event_id,
        )
        self._record_event(event)
        return None

    # --- UserPromptSubmit (FRD-005) ---

    async def _handle_user_prompt_submit(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle UserPromptSubmit: create STEP (THINK) event."""
        # SDK field is "prompt" (UserPromptSubmitHookInput), not "prompt_text"
        prompt_text = input_data.get("prompt") or input_data.get("prompt_text", "")
        if not prompt_text:
            logger.warning("clyro_empty_prompt_text")

        # Store for policy evaluation during tool calls (e.g., keyword filters)
        self._last_user_prompt = prompt_text or ""

        # Policy evaluation on user prompt (e.g., keyword filters)
        deny_response = await self._evaluate_prompt_policy(prompt_text)
        if deny_response is not None:
            return deny_response

        # Accumulate cost from prompt content
        self._cost_estimator.accumulate(prompt_text)

        # Estimated token count from prompt content
        estimated_input_tokens = (
            len(prompt_text) // self._cost_estimator.CHARS_PER_TOKEN if prompt_text else 0
        )
        self._cost_estimator.total_input_tokens += estimated_input_tokens

        event = create_step_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            event_name="user_prompt_submit",
            agent_id=self._session.agent_id,
            input_data=self._capture_input({"prompt_text": prompt_text}),
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            token_count_input=estimated_input_tokens,
            cost_usd=self._cost_estimator.last_step_cost,
            metadata={"hook_event": "UserPromptSubmit", "cost_estimated": True},
            agent_stage=AgentStage.THINK,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        self._record_event(event)
        return None

    # --- W3: SubagentStart (FRD-007) ---

    async def _handle_subagent_start(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle SubagentStart: create TASK_START (ACT) event."""
        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")
        agent_tool_call_id = input_data.get("agent_tool_call_id", "")

        # W3 1a: Guard against subagent session_id divergence (FRD-007 assumption)
        hook_session_id = input_data.get("session_id", "")
        if hook_session_id and hook_session_id != self._session_id:
            logger.error(
                "clyro_subagent_session_id_mismatch",
                parent_session_id=self._session_id,
                subagent_session_id=hook_session_id,
                agent_id=agent_id,
            )
            # Treat as parent session per FRD-007 assumption

        event_id = uuid4()
        event = TraceEvent(
            event_id=event_id,
            session_id=self._session.session_id,
            agent_id=self._session.agent_id,
            event_type=EventType.TASK_START,
            event_name=f"subagent_start_{agent_type}",
            agent_stage=AgentStage.ACT,
            input_data=self._capture_input({"agent_type": agent_type, "agent_id": agent_id}),
            metadata={
                "hook_event": "SubagentStart",
                "agent_tool_call_id": agent_tool_call_id,
            },
            step_number=self._event_step_number,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        self._record_event(event)

        # Store in tracker
        self._subagent_tracker.start(agent_id, event_id, agent_type)
        return None

    # --- W3: SubagentStop (FRD-007) ---

    async def _handle_subagent_stop(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle SubagentStop: create TASK_END (OBSERVE) event."""
        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")
        transcript_path = input_data.get("agent_transcript_path", "")
        stop_hook_active = input_data.get("stop_hook_active", False)
        agent_tool_call_id = input_data.get("agent_tool_call_id", "")

        parent_event_id, duration_ms = self._subagent_tracker.stop(agent_id)

        event = TraceEvent(
            session_id=self._session.session_id,
            agent_id=self._session.agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.TASK_END,
            event_name=f"subagent_stop_{agent_type}",
            agent_stage=AgentStage.OBSERVE,
            output_data={"agent_transcript_path": transcript_path}
            if self._config.capture_outputs
            else None,
            duration_ms=duration_ms or 0,
            metadata={
                "hook_event": "SubagentStop",
                "agent_tool_call_id": agent_tool_call_id,
                "stop_hook_active": stop_hook_active,
            },
            step_number=self._event_step_number,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        self._record_event(event)
        return None

    # --- W5: Stop (FRD-009) ---

    async def _handle_stop(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle Stop hook: emit SESSION_END."""
        if self._session_ended:
            return None  # Ignore duplicate (FRD-009)

        # When wrapped, the wrapper emits session_end — skip duplicate.
        # But still flush correlator/tracker.
        if self._is_wrapped:
            self._session_ended = True
            self._correlator.flush()
            self._subagent_tracker.flush()
            return None

        self.end_session(input_data)
        return None

    # --- Notification (FRD-012) ---

    async def _handle_notification(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle Notification hook: create STEP (OBSERVE) event."""
        message = input_data.get("message", "")
        if not input_data.get("message"):
            logger.warning("clyro_empty_notification_message")

        output_data = (
            {
                "message": message,
                "title": input_data.get("title", ""),
                "notification_type": input_data.get("notification_type", ""),
            }
            if self._config.capture_outputs
            else None
        )
        event = create_step_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            event_name="notification",
            agent_id=self._session.agent_id,
            output_data=output_data,
            metadata={"hook_event": "Notification"},
            agent_stage=AgentStage.OBSERVE,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        self._record_event(event)
        return None

    # --- PreCompact (FRD-013) ---

    async def _handle_pre_compact(
        self, input_data: dict[str, Any], tool_use_id: str | None
    ) -> dict[str, Any] | None:
        """Handle PreCompact hook: create STATE_TRANSITION (THINK) event."""
        trigger = input_data.get("trigger", "")
        conversation_size = input_data.get("conversation_size", 0)
        if input_data.get("conversation_size") is None:
            logger.warning("clyro_missing_conversation_size")

        event = create_state_transition_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            node_name="pre_compact",
            agent_id=self._session.agent_id,
            state_snapshot={"trigger": trigger, "conversation_size": conversation_size},
            metadata={"hook_event": "PreCompact"},
            agent_stage=AgentStage.THINK,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        self._record_event(event)
        return None

    # --- W4: Policy Evaluation (FRD-010) ---

    async def _evaluate_policy(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        parent_event_id: UUID | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate policy and return deny response if blocked, else None."""
        if not self._config.controls.enable_policy_enforcement:
            return None
        if not self._policy_evaluator:
            return None

        # Flatten tool_input so policy rules can reference tool parameters
        # directly (e.g., field: "rmq_cluster") without dot notation
        # (e.g., field: "tool_input.rmq_cluster").
        policy_params: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "input": self._last_user_prompt,
            "cost": float(self._cost_estimator.estimated_cumulative_cost),
            "step_number": self._step_count,
        }
        if isinstance(tool_input, dict):
            policy_params.update(tool_input)

        try:
            decision = await asyncio.wait_for(
                self._policy_evaluator.evaluate_async(
                    action_type="tool_call",
                    parameters=policy_params,
                    session_id=self._session.session_id if self._session else None,
                    step_number=self._step_count,
                ),
                timeout=POLICY_TIMEOUT_SECONDS,
            )
        except PolicyViolationError as e:
            # Check if this is a require_approval decision that needs user prompt.
            # PolicyEvaluator._enforce_decision sets details={"decision": "require_approval"}
            # when the approval handler is None (e.g., sys.stdin.isatty() returns False
            # in VSCode or non-TTY environments). Handle approval directly here.
            is_approval_request = e.details and e.details.get("decision") == "require_approval"
            if is_approval_request:
                approved = await self._prompt_approval(
                    tool_name=tool_name,
                    rule_name=e.rule_name or "",
                    message=e.message or "Action requires approval",
                )
                if approved:
                    logger.info(
                        "clyro_policy_approval_granted",
                        tool_name=tool_name,
                        rule_id=e.rule_id,
                        rule_name=e.rule_name,
                    )
                    self._record_policy_event(
                        "allow",
                        parent_event_id=parent_event_id,
                        action_type="tool_call",
                        parameters=policy_params,
                        reason="user_approved",
                        rule_id=e.rule_id or "",
                        rule_name=e.rule_name or "",
                    )
                    return None

            # Policy blocked the action — record policy_check and error events
            self._record_policy_event(
                "block",
                parent_event_id=parent_event_id,
                action_type="tool_call",
                parameters=policy_params,
                rule_id=e.rule_id or "",
                rule_name=e.rule_name or "",
                message=e.message or "",
            )
            error_event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._event_step_number,
                error_type="PolicyViolation",
                error_message=e.message or "Policy violation",
                agent_id=self._session.agent_id,
                cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
                input_data={"tool_name": tool_name, "action_type": "tool_call"},
                output_data={
                    "error_type": "PolicyViolation",
                    "error_message": e.message or "Policy violation",
                    "rule_id": e.rule_id or "",
                    "rule_name": e.rule_name or "",
                },
                metadata={
                    "rule_id": e.rule_id or "",
                    "rule_name": e.rule_name or "",
                    "tool_name": tool_name,
                },
                framework=Framework.CLAUDE_AGENT_SDK,
                framework_version=self._framework_version,
            )
            self._record_event(error_event)
            raise  # Re-raise so handle_hook converts to deny response
        except TimeoutError:
            # Fail-open on timeout
            logger.warning("clyro_policy_timeout", tool_name=tool_name)
            self._record_policy_event(
                "allow",
                parent_event_id=parent_event_id,
                action_type="tool_call",
                parameters=policy_params,
                reason="policy_evaluation_timeout",
            )
            return None
        except Exception:
            logger.warning("clyro_policy_error", tool_name=tool_name, exc_info=True)
            self._record_policy_event(
                "allow",
                parent_event_id=parent_event_id,
                action_type="tool_call",
                parameters=policy_params,
                reason="policy_evaluation_error",
            )
            return None

        if decision.is_allowed:
            self._record_policy_event(
                "allow",
                parent_event_id=parent_event_id,
                action_type="tool_call",
                parameters=policy_params,
                evaluated_rules=decision.evaluated_rules,
            )
            return None

        # If require_approval decision reached here without raising, the
        # PolicyEvaluator's approval handler already approved the action.
        if decision.requires_approval:
            self._record_policy_event(
                "allow",
                parent_event_id=parent_event_id,
                action_type="tool_call",
                parameters=policy_params,
                reason="user_approved",
                rule_id=decision.rule_id or "",
                rule_name=decision.rule_name or "",
            )
            return None

        # Block decision (FRD-010)
        policy_kwargs: dict[str, Any] = {
            "rule_id": decision.rule_id or "",
            "rule_name": decision.rule_name or "",
            "message": decision.message or "",
        }

        self._record_policy_event(
            "block",
            parent_event_id=parent_event_id,
            action_type="tool_call",
            parameters=policy_params,
            **policy_kwargs,
        )

        # Record ERROR event
        error_event = create_error_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            error_type="PolicyViolation",
            error_message=decision.message or "Policy violation",
            agent_id=self._session.agent_id,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            input_data={
                "tool_name": policy_params.get("tool_name", ""),
                "action_type": "tool_call",
            },
            output_data={
                "error_type": "PolicyViolation",
                "error_message": decision.message or "Policy violation",
                "rule_id": decision.rule_id or "",
                "rule_name": decision.rule_name or "",
            },
        )
        self._record_event(error_event)

        raise PolicyViolationError(
            rule_id=decision.rule_id or "",
            rule_name=decision.rule_name or "",
            message=decision.message or "Policy violation",
            action_type="tool_call",
        )

    # --- W4b: User Prompt Policy Evaluation ---

    async def _evaluate_prompt_policy(self, prompt_text: str) -> dict[str, Any] | None:
        """Evaluate policy against the user prompt text.

        Enables keyword-filter and content-based policies (e.g., field: input,
        operator: contains) to block prompts before any tool calls happen.

        Returns a deny response dict if blocked, else None.
        """
        if not self._config.controls.enable_policy_enforcement:
            return None
        if not self._policy_evaluator:
            return None

        policy_params: dict[str, Any] = {
            "input": prompt_text,
            "cost": float(self._cost_estimator.estimated_cumulative_cost),
            "step_number": self._step_count,
        }

        try:
            decision = await asyncio.wait_for(
                self._policy_evaluator.evaluate_async(
                    action_type="user_prompt",
                    parameters=policy_params,
                    session_id=self._session.session_id if self._session else None,
                    step_number=self._step_count,
                ),
                timeout=POLICY_TIMEOUT_SECONDS,
            )
        except PolicyViolationError as e:
            self._record_policy_event(
                "block",
                action_type="user_prompt",
                parameters=policy_params,
                rule_id=e.rule_id or "",
                rule_name=e.rule_name or "",
                message=e.message or "",
            )
            error_event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._event_step_number,
                error_type="PolicyViolation",
                error_message=e.message or "Policy violation",
                agent_id=self._session.agent_id,
                cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
                input_data={"action_type": "user_prompt"},
                output_data={
                    "error_type": "PolicyViolation",
                    "error_message": e.message or "Policy violation",
                    "rule_id": e.rule_id or "",
                    "rule_name": e.rule_name or "",
                },
                metadata={
                    "rule_id": e.rule_id or "",
                    "rule_name": e.rule_name or "",
                    "action_type": "user_prompt",
                },
                framework=Framework.CLAUDE_AGENT_SDK,
                framework_version=self._framework_version,
            )
            self._record_event(error_event)
            raise
        except TimeoutError:
            logger.warning("clyro_prompt_policy_timeout")
            self._record_policy_event(
                "allow",
                action_type="user_prompt",
                parameters=policy_params,
                reason="policy_evaluation_timeout",
            )
            return None
        except Exception:
            logger.warning("clyro_prompt_policy_error", exc_info=True)
            self._record_policy_event(
                "allow",
                action_type="user_prompt",
                parameters=policy_params,
                reason="policy_evaluation_error",
            )
            return None

        if decision.is_allowed:
            self._record_policy_event(
                "allow",
                action_type="user_prompt",
                parameters=policy_params,
                evaluated_rules=decision.evaluated_rules,
            )
            return None

        if decision.requires_approval:
            self._record_policy_event(
                "allow",
                action_type="user_prompt",
                parameters=policy_params,
                reason="user_approved",
                rule_id=decision.rule_id or "",
                rule_name=decision.rule_name or "",
            )
            return None

        # Block
        self._record_policy_event(
            "block",
            action_type="user_prompt",
            parameters=policy_params,
            rule_id=decision.rule_id or "",
            rule_name=decision.rule_name or "",
            message=decision.message or "",
        )
        raise PolicyViolationError(
            rule_id=decision.rule_id or "",
            rule_name=decision.rule_name or "",
            message=decision.message or "Policy violation",
            action_type="user_prompt",
        )

    # --- W4a: Approval Prompting ---

    async def _prompt_approval(self, tool_name: str, rule_name: str, message: str) -> bool:
        """Prompt user for approval when policy returns require_approval.

        Uses asyncio.to_thread to avoid blocking the event loop while waiting
        for user input. Falls back to blocking input() if to_thread is
        unavailable. Returns False if stdin is not interactive.
        """
        # Only prompt if stdin is connected to a terminal or user input works
        # (sys.stdin.isatty() can return False in VSCode but input() still works)
        try:

            def _console_prompt() -> bool:
                print("\n" + "=" * 60)
                print("  POLICY APPROVAL REQUIRED")
                print("=" * 60)
                print(f"  Tool      : {tool_name}")
                print(f"  Rule      : {rule_name}")
                print(f"  Reason    : {message}")
                print("=" * 60)
                while True:
                    response = input("Approve this action? [y/n]: ").strip().lower()
                    if response in ("y", "yes"):
                        return True
                    if response in ("n", "no"):
                        return False
                    print("Please enter 'y' or 'n'.")

            return await asyncio.to_thread(_console_prompt)
        except (EOFError, OSError, RuntimeError):
            # stdin not available or event loop issue — deny by default
            logger.warning(
                "clyro_approval_prompt_unavailable",
                tool_name=tool_name,
                rule_name=rule_name,
            )
            return False

    # --- Session End (W5 / FRD-008) ---

    def end_session(
        self,
        input_data: dict[str, Any] | None = None,
        result_message: dict[str, Any] | None = None,
    ) -> None:
        """Emit SESSION_END with aggregated metrics."""
        if self._session_ended or not self._session_started:
            return

        self._session_ended = True
        duration_ms = int((time.monotonic() - self._start_time) * 1000)

        # FRD-008: Extract token/cost from ResultMessage if available
        metadata: dict[str, Any] = {"hook_event": "Stop"}
        token_input = 0
        token_output = 0
        cost_usd = Decimal("0")

        if result_message:
            usage = result_message.get("usage", {})
            token_input = usage.get("input_tokens", 0)
            token_output = usage.get("output_tokens", 0)
            total_cost = result_message.get("total_cost_usd")
            if total_cost is not None:
                cost_usd = Decimal(str(total_cost)).quantize(Decimal("0.000001"))
            metadata.update(
                {
                    "num_turns": result_message.get("num_turns"),
                    "subtype": result_message.get("subtype"),
                    "stop_reason": result_message.get("stop_reason"),
                    "cache_read_tokens": usage.get("cache_read_input_tokens"),
                    "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
                }
            )
        elif input_data:
            metadata["stop_hook_active"] = input_data.get("stop_hook_active", False)

        end_event = create_session_end_event(
            session_id=self._session.session_id,
            agent_id=self._session.agent_id,
            step_number=self._event_step_number,
            cumulative_cost=cost_usd or self._cost_estimator.estimated_cumulative_cost,
            duration_ms=duration_ms,
            metadata=metadata,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
        )
        # Add token/cost fields
        if token_input or token_output:
            end_event = end_event.model_copy(
                update={
                    "token_count_input": token_input,
                    "token_count_output": token_output,
                    "cost_usd": cost_usd,
                }
            )
        self._record_event(end_event)

        # Flush correlator and tracker
        self._correlator.flush()
        self._subagent_tracker.flush()

    def drain_events(self) -> list[TraceEvent]:
        """Return accumulated events and clear the buffer."""
        events = list(self._events)
        self._events.clear()
        return events

    # --- Internal helpers ---

    @property
    def _event_step_number(self) -> int:
        """Step number to use in events.

        When wrapped, return 0 so session.record_event() assigns sequential
        numbers (avoiding duplicates with the wrapper's own events).
        When standalone, use the handler's own step counter.
        """
        return 0 if self._is_wrapped else self._step_count

    def _record_event(self, event: TraceEvent) -> None:
        """Record event to buffer and optionally delegate to Session.

        When wrapped (via clyro.wrap), events are buffered here and later
        drained by the wrapper which calls session.record_event(). We skip
        the session call here to avoid double-recording and duplicate
        step_number assignment.

        When standalone (instrument_claude_agent), delegate directly to
        the session for immediate recording.
        """
        try:
            self._events.append(event)
            if not self._is_wrapped and self._session:
                self._session.record_event(event)
        except Exception:
            logger.exception("clyro_record_event_error")

    def _record_error_event(self, error_type: str, message: str) -> None:
        """Record an ERROR TraceEvent."""
        if not self._session:
            return
        event = create_error_event(
            session_id=self._session.session_id,
            step_number=self._event_step_number,
            error_type=error_type,
            error_message=message,
            agent_id=self._session.agent_id,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            output_data={
                "error_type": error_type,
                "error_message": message,
            },
        )
        self._record_event(event)

    def _record_policy_event(
        self,
        decision: str,
        *,
        parent_event_id: UUID | None = None,
        action_type: str | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a POLICY_CHECK TraceEvent."""
        if not self._session:
            return
        input_data = None
        if action_type or parameters:
            input_data = {"action_type": action_type or "unknown", "parameters": parameters or {}}
        event = TraceEvent(
            session_id=self._session.session_id,
            agent_id=self._session.agent_id,
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            agent_stage=AgentStage.ACT,
            step_number=self._event_step_number,
            cumulative_cost=self._cost_estimator.estimated_cumulative_cost,
            input_data=input_data,
            output_data={"decision": decision, **kwargs},
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self._framework_version,
            parent_event_id=parent_event_id,
        )
        self._record_event(event)

    def _capture_input(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """Return input_data if capture_inputs is enabled, else None (NFR-006)."""
        if not self._config.capture_inputs:
            return None
        return _truncate_dict(data, TRUNCATION_LIMIT)

    def _safe_serialize_output(self, output: Any) -> dict[str, Any]:
        """Safely serialize tool output with truncation."""
        try:
            if isinstance(output, dict):
                return _truncate_dict({"tool_output": output}, TRUNCATION_LIMIT)
            if isinstance(output, str):
                return {"tool_output": _truncate(output, TRUNCATION_LIMIT)}
            return {"tool_output": _truncate(repr(output), TRUNCATION_LIMIT)}
        except Exception:
            return {"tool_output": "<serialization_error>"}

    @staticmethod
    def _deny_response(hook_type: str, reason: str, *, stop_agent: bool = False) -> dict[str, Any]:
        """Build a deny hook response (§4.4).

        Args:
            hook_type: The hook event name.
            reason: Human-readable reason for denial.
            stop_agent: If True, sets continue_=False to halt the agent
                entirely (used for policy blocks and execution control errors
                to match other adapters where PolicyViolationError stops execution).
        """
        response: dict[str, Any] = {
            "hookSpecificOutput": {
                "hookEventName": hook_type,
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Clyro: {reason}",
            }
        }
        if stop_agent:
            response["continue_"] = False
            response["stopReason"] = f"Clyro: {reason}"
        return response


# --- C1: HookRegistrar (FRD-001) ---


class HookRegistrar:
    """Wire Clyro hook callbacks into ClaudeAgentOptions.hooks.

    Implements FRD-001.

    The Claude Agent SDK expects:
        hooks: dict[HookEvent, list[HookMatcher]]
    where HookMatcher is a dataclass with .matcher (str|None), .hooks (list[HookCallback]),
    and HookCallback signature is (HookInput, str|None, HookContext) -> Awaitable[HookJSONOutput].
    HookJSONOutput is SyncHookJSONOutput (TypedDict with continue_, hookSpecificOutput, etc.).
    """

    def __init__(self, config: ClyroConfig, handler: ClaudeAgentHandler) -> None:
        self._config = config
        self._handler = handler

    def register(self, hooks: dict[str, Any], options: Any = None) -> dict[str, Any]:
        """Register Clyro callbacks for all 9 hook types. Idempotent.

        Args:
            hooks: The ClaudeAgentOptions.hooks dict to modify in-place.
            options: The ClaudeAgentOptions object for idempotency marker.

        Returns:
            The modified hooks dict.
        """
        # Idempotency check — marker stored on the options object (not in
        # the hooks dict) because Claude Agent SDK iterates all hook keys
        # and a non-callable value like True causes TypeError.
        if options is None:
            raise ValueError("options is required for idempotency marker storage")
        if getattr(options, _CLYRO_HOOK_MARKER, False):
            logger.debug("clyro_hooks_already_registered")
            return hooks

        # SDK's _convert_hooks_to_internal_format uses attribute access
        # (matcher.matcher, matcher.hooks), so we must use HookMatcher instances.
        try:
            from claude_agent_sdk.types import HookMatcher
        except ImportError:
            from dataclasses import dataclass as _dc
            from dataclasses import field as _f

            @_dc
            class HookMatcher:  # type: ignore[no-redef]
                matcher: str | None = None
                hooks: list = _f(default_factory=list)
                timeout: float | None = None

        for hook_type in HOOK_TYPES:
            callback = self._make_callback(hook_type)
            matcher = HookMatcher(matcher=None, hooks=[callback])
            if hook_type not in hooks:
                hooks[hook_type] = []
            if isinstance(hooks[hook_type], list):
                hooks[hook_type].append(matcher)
            else:
                hooks[hook_type] = [hooks[hook_type], matcher]

        setattr(options, _CLYRO_HOOK_MARKER, True)
        return hooks

    def _make_callback(self, hook_type: str) -> Any:
        """Create an async callback matching HookCallback signature.

        HookCallback = Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]
        Returns SyncHookJSONOutput: {} for allow, or dict with hookSpecificOutput for deny.
        """
        handler = self._handler

        async def clyro_hook(
            input_data: dict[str, Any],
            tool_use_id: str | None = None,
            context: Any = None,
        ) -> dict[str, Any]:
            result = await handler.handle_hook(hook_type, input_data, tool_use_id, context)
            if result is None:
                # Allow — return empty SyncHookJSONOutput
                return {}
            # Deny or modified response — already a valid SyncHookJSONOutput from _deny_response
            return result

        clyro_hook.__name__ = f"clyro_{hook_type}_hook"
        clyro_hook._clyro_hook = True  # type: ignore[attr-defined]
        return clyro_hook


# --- C5: ClaudeAgentAdapter (WrappedAgent integration) ---


class ClaudeAgentAdapter:
    """Adapter class for WrappedAgent pattern integration.

    Implements FRD-001 (alternative entry point via clyro.wrap()).
    """

    FRAMEWORK = Framework.CLAUDE_AGENT_SDK
    FRAMEWORK_VERSION: str | None = None

    def __init__(self, agent: Any, config: ClyroConfig) -> None:
        self._agent = agent
        self._config = config
        self._name = getattr(agent, "__name__", type(agent).__name__)
        self.FRAMEWORK_VERSION = detect_claude_agent_sdk_version()
        self._handler: ClaudeAgentHandler | None = None

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def name(self) -> str:
        return self._name

    @property
    def framework(self) -> Framework:
        return self.FRAMEWORK

    @property
    def framework_version(self) -> str:
        return self.FRAMEWORK_VERSION or "unknown"

    def before_call(self, session: Session, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Register Clyro hooks into the agent's ClaudeAgentOptions.

        Mirrors LangGraph/CrewAI adapters which set up instrumentation in
        before_call so that clyro.wrap() provides full observability.
        Idempotent — safe to call on every invocation.

        Supports wrapping the agent instance, its options, or a bound method
        (resolves via __self__ for bound methods).
        """
        # Resolve the agent object — handle bound methods (e.g. clyro.wrap(agent._run))
        agent_obj = self._agent
        if hasattr(agent_obj, "__self__"):
            agent_obj = agent_obj.__self__

        options = getattr(agent_obj, "options", None) or agent_obj
        if hasattr(options, "hooks"):
            instrument_claude_agent(options, self._config)
        # Pass handler in context so wrapper can drain hook events (tool_call, etc.)
        handler = getattr(options, "_clyro_handler", None)
        if handler is not None:
            # Reset ALL per-invocation state.  The handler is stored on
            # options._clyro_handler and reused across invoke() calls.
            # Each call runs in a fresh asyncio.run() event loop, so every
            # piece of mutable state from the previous run is stale:
            #   - _session_started / _session_id → wrong _ensure_session branch
            #   - _policy_evaluator → httpx client bound to dead loop
            #   - _step_count / _cost_estimator → wrong enforcement checks
            #   - _pending_error → immediate false re-raise
            handler._reset_for_invocation()
            # Inject the wrapper's session so hook events share the same
            # session_id (not the Claude CLI's internal session_id).
            handler._session = session
            handler._is_wrapped = True
            # Re-inject the wrapper's policy evaluator (freshly created per
            # call, bound to the current event loop).
            if session._policy_evaluator is not None:
                handler._policy_evaluator = session._policy_evaluator
        return {"start_time": time.perf_counter(), "handler": handler}

    def after_call(self, session: Session, result: Any, context: dict[str, Any]) -> TraceEvent:
        """Synthesize an llm_call event for the Claude Agent SDK execution.

        The Claude Agent SDK hooks don't expose per-LLM-call data, so we
        create a summary llm_call event covering the entire agent run.
        Uses step_number=0 so session.record_event() auto-increments it
        to the next unique step number.
        """
        duration_ms = int((time.perf_counter() - context["start_time"]) * 1000)

        # Extract cumulative cost and token counts from handler's estimator
        handler = context.get("handler")
        cumulative_cost = Decimal("0")
        total_input_tokens = 0
        total_output_tokens = 0
        if handler and hasattr(handler, "_cost_estimator"):
            cumulative_cost = handler._cost_estimator.estimated_cumulative_cost
            total_input_tokens = handler._cost_estimator.total_input_tokens
            total_output_tokens = handler._cost_estimator.total_output_tokens

        # Serialize agent result for output_data
        output_data = None
        if result is not None:
            try:
                if isinstance(result, dict):
                    output_data = {"result": result}
                elif isinstance(result, str):
                    output_data = {"result": result[:TRUNCATION_LIMIT]}
                else:
                    output_data = {"result": str(result)[:TRUNCATION_LIMIT]}
            except Exception:
                output_data = {"result": "<serialization_error>"}

        return create_llm_call_event(
            session_id=session.session_id,
            step_number=0,  # Auto-increment to next unique step number
            model="claude",
            input_data={"model": "claude", "framework": "claude_agent_sdk"},
            output_data=output_data,
            agent_id=session.agent_id,
            duration_ms=duration_ms,
            cumulative_cost=cumulative_cost,
            token_count_input=total_input_tokens,
            token_count_output=total_output_tokens,
            cost_usd=cumulative_cost,
            metadata={"source": "claude_agent_sdk", "synthesized": True, "cost_estimated": True},
            agent_stage=AgentStage.THINK,
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self.FRAMEWORK_VERSION,
        )

    def on_error(self, session: Session, error: Exception, context: dict[str, Any]) -> TraceEvent:
        """Create error event on agent failure."""
        import traceback as tb

        return create_error_event(
            session_id=session.session_id,
            step_number=session.step_number,
            error_type=type(error).__name__,
            error_message=str(error),
            agent_id=session.agent_id,
            error_stack=tb.format_exc(),
            framework=Framework.CLAUDE_AGENT_SDK,
            framework_version=self.FRAMEWORK_VERSION,
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )


# --- Public API ---


def instrument_claude_agent(
    options: Any,
    config: ClyroConfig,
    policy_evaluator: PolicyEvaluator | None = None,
) -> Any:
    """Register Clyro instrumentation hooks into Claude Agent SDK options.

    Implements FRD-001.

    Args:
        options: ClaudeAgentOptions instance to modify in-place.
        config: ClyroConfig with agent identification, transport, and controls.
        policy_evaluator: Optional PolicyEvaluator for policy enforcement.

    Returns:
        The modified ClaudeAgentOptions (same reference, modified in-place).

    Raises:
        ImportError: If claude-agent-sdk is not installed.
        FrameworkVersionError: If SDK version is unsupported.
        ValueError: If options.hooks has unexpected structure.
    """
    # Validate SDK is installed and version is supported
    version = validate_claude_agent_sdk_version()

    # Validate hooks structure
    if not hasattr(options, "hooks"):
        options.hooks = {}
    if options.hooks is None:
        options.hooks = {}
    if not isinstance(options.hooks, dict):
        raise ValueError("ClaudeAgentOptions.hooks must be a dictionary")

    # If hooks are already registered, reuse the existing handler —
    # creating a new handler would leave the registered hook callbacks
    # pointing at the OLD handler while options._clyro_handler points
    # at the NEW one, causing state divergence across invoke() calls.
    if getattr(options, _CLYRO_HOOK_MARKER, False):
        logger.debug("clyro_hooks_already_registered")
        return options

    # Create handler and registrar
    handler = ClaudeAgentHandler(
        config=config,
        framework_version=version,
        policy_evaluator=policy_evaluator,
    )
    registrar = HookRegistrar(config=config, handler=handler)
    registrar.register(options.hooks, options=options)

    # Store handler on options so ClaudeAgentAdapter.before_call() can
    # retrieve it for event draining (same pattern as LangGraph/CrewAI).
    options._clyro_handler = handler

    return options


# --- Detection and Version Validation ---


def detect_claude_agent_sdk_version() -> str | None:
    """Detect installed claude-agent-sdk version."""
    try:
        return importlib.metadata.version("claude-agent-sdk")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import claude_agent_sdk

        return getattr(claude_agent_sdk, "__version__", None)
    except ImportError:
        return None


def validate_claude_agent_sdk_version(version: str | None = None) -> str:
    """Validate claude-agent-sdk is installed and version is supported.

    Raises:
        ImportError: If not installed.
        FrameworkVersionError: If version is below minimum.
    """
    if version is None:
        version = detect_claude_agent_sdk_version()

    if version is None:
        raise ImportError(
            "claude-agent-sdk is required. Install with: pip install clyro[claude-agent-sdk]"
        )

    # If version can't be determined, assume compatible (fail-open)
    if version == "unknown":
        logger.warning(
            "claude_agent_sdk_version_unknown",
            message="Could not determine Claude Agent SDK version, assuming compatible",
        )
        return version

    if not _is_version_supported(version):
        raise FrameworkVersionError(
            framework="Claude Agent SDK",
            version=version,
            supported=f">={MIN_SDK_VERSION}",
        )
    return version


def is_claude_agent_sdk_agent(agent: Any) -> bool:
    """Check if agent is a Claude Agent SDK client or options object."""
    module = getattr(type(agent), "__module__", "") or ""
    if "claude_agent_sdk" in module:
        return True
    # Check for Claude Agent SDK-specific attributes
    if hasattr(agent, "hooks") and hasattr(agent, "model"):
        cls_name = type(agent).__name__
        if "claude" in cls_name.lower() or "agent" in cls_name.lower():
            return True
    return False


# --- Utility Functions ---


def _is_version_supported(version: str) -> bool:
    """Check if version meets minimum requirement."""
    try:
        parsed = _parse_version(version)
        minimum = _parse_version(MIN_SDK_VERSION)
        return parsed >= minimum
    except (ValueError, TypeError):
        return False


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse version string to comparable tuple."""
    return tuple(int(x) for x in version.split(".")[:3])


def _truncate(text: str, limit: int) -> str:
    """Truncate text to limit characters."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _truncate_dict(data: dict[str, Any], limit: int) -> dict[str, Any]:
    """Truncate string values in a dict to limit characters.

    Handles str, dict, and list values. Falls back to JSON serialization
    for complex types and truncates if the serialized form exceeds limit.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _truncate(value, limit)
        elif isinstance(value, (dict, list)):
            try:
                serialized = json.dumps(value, default=str)
                if len(serialized) > limit:
                    result[key] = _truncate(serialized, limit)
                else:
                    result[key] = value
            except (TypeError, ValueError):
                result[key] = _truncate(repr(value), limit)
        else:
            result[key] = value
    return result
