# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK CrewAI Adapter
# Implements PRD-004

"""
CrewAI adapter for wrapping Crew instances with task execution tracing.

This adapter provides framework-specific instrumentation for CrewAI,
capturing task executions and agent actions during Crew execution.

Supported CrewAI version: 0.30.0+

Event Capture Strategy:
    Uses CrewAI's native event bus (crewai_event_bus) to capture:
    - LLM calls (model, tokens, messages, response)
    - Tool usage (tool name, args, output, duration)
    - Task lifecycle (start, end, error)
    - Agent lifecycle (start, end, errors)

    The event bus is global, so it works with both direct Crew
    instances and wrapper classes without monkey-patching.

Example usage:
    from crewai import Crew, Agent, Task
    import clyro

    crew = Crew(agents=[...], tasks=[...])
    wrapped_crew = clyro.wrap(crew)
    result = wrapped_crew.kickoff(inputs={"topic": "AI trends"})
"""

from __future__ import annotations

import time
import traceback
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

import structlog

from clyro.cost import CostCalculator
from clyro.exceptions import (
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    PolicyViolationError,
)
from clyro.session import Session
from clyro.trace import (
    AgentStage,
    EventType,
    Framework,
    TraceEvent,
    create_error_event,
    create_llm_call_event,
    create_state_transition_event,
    create_step_event,
    create_tool_call_event,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Version constraints
MIN_CREWAI_VERSION = "0.30.0"
MIN_VERSION_TUPLE = (0, 30, 0)

# Truncation limits for serialized data
MAX_MESSAGE_LENGTH = 1000
MAX_CONTEXT_LENGTH = 500
MAX_TASK_DESCRIPTION_LENGTH = 200
MAX_EVENT_NAME_LENGTH = 50


def _parse_version(version_str: str) -> tuple[int, int, int]:
    """
    Parse a version string into a tuple of integers.

    Handles various version formats:
    - "0.30.0" -> (0, 30, 0)
    - "0.30.0rc1" -> (0, 30, 0)
    - "0.30.0.dev1" -> (0, 30, 0)

    Args:
        version_str: Version string to parse

    Returns:
        Tuple of (major, minor, patch) integers
    """
    import re

    # Extract numeric parts only
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    # Fallback for malformed versions
    return (0, 0, 0)


def detect_crewai_version() -> str | None:
    """
    Detect the installed CrewAI version.

    Returns:
        Version string if CrewAI is installed, None otherwise.
    """
    try:
        import crewai

        return getattr(crewai, "__version__", None)
    except ImportError:
        return None


def validate_crewai_version() -> str:
    """
    Validate that the installed CrewAI version is supported.

    Returns:
        The validated version string.

    Raises:
        FrameworkVersionError: If CrewAI is not installed or version is unsupported.
    """
    version = detect_crewai_version()

    if version is None:
        raise FrameworkVersionError(
            framework="crewai",
            version="not installed",
            supported=f">={MIN_CREWAI_VERSION}",
        )

    # If version can't be determined, assume compatible (fail-open)
    if version == "unknown":
        logger.warning(
            "crewai_version_unknown",
            message="Could not determine CrewAI version, assuming compatible",
        )
        return version

    version_tuple = _parse_version(version)
    if version_tuple < MIN_VERSION_TUPLE:
        raise FrameworkVersionError(
            framework="crewai",
            version=version,
            supported=f">={MIN_CREWAI_VERSION}",
        )

    return version


def is_crewai_agent(agent: Any) -> bool:
    """
    Check if an object is a CrewAI Crew instance or wrapper.

    Detection priority:
    1. Module starts with 'crewai' (actual CrewAI package)
    2. Class name is 'Crew' with kickoff/agents/tasks attributes
    3. Has all three CrewAI-specific public attributes
    4. Has CrewAI-specific private attributes
    5. Function/class module imports CrewAI symbols

    Args:
        agent: Object to check

    Returns:
        True if the object is a CrewAI agent, False otherwise.
    """
    agent_type = type(agent).__name__
    module = getattr(type(agent), "__module__", "") or ""

    # Check module name - must start with 'crewai' to be the actual package
    module_lower = module.lower()
    if module_lower.startswith("crewai") or module_lower.startswith("crewai."):
        return True

    # Check for CrewAI-specific attributes combination
    has_kickoff = hasattr(agent, "kickoff") and callable(getattr(agent, "kickoff", None))
    has_agents = hasattr(agent, "agents")
    has_tasks = hasattr(agent, "tasks")

    if agent_type == "Crew" and has_kickoff and has_agents and has_tasks:
        return True

    if has_kickoff and has_agents and has_tasks:
        return True

    # Check for CrewAI-specific private attributes
    crewai_private_attrs = (
        "_original_tasks",
        "_original_agents",
        "_task_output_handler",
        "_rpm_controller",
        "_execution_span",
    )
    for attr in crewai_private_attrs:
        if hasattr(agent, attr):
            return True

    # Check if a function's defining module imports crewai symbols
    _CREWAI_SYMBOLS = frozenset({"Crew", "Agent", "Task", "Process"})
    if callable(agent) and hasattr(agent, "__globals__"):
        func_globals = agent.__globals__
        for name in _CREWAI_SYMBOLS:
            obj = func_globals.get(name)
            if obj is not None:
                obj_module = getattr(obj, "__module__", "") or ""
                if obj_module.startswith("crewai"):
                    return True

    # Check if the class's defining module imports CrewAI symbols
    import sys

    agent_module_name = getattr(type(agent), "__module__", "")
    if agent_module_name and agent_module_name in sys.modules:
        module_dict = vars(sys.modules[agent_module_name])
        for name in _CREWAI_SYMBOLS:
            obj = module_dict.get(name)
            if obj is not None:
                obj_module = getattr(obj, "__module__", "") or ""
                if obj_module.startswith("crewai"):
                    return True

    return False


class CrewAICallbackHandler:
    """
    Callback handler for capturing CrewAI execution events.

    Captures crew lifecycle, task lifecycle, and agent actions
    (LLM calls, tool calls). Events are accumulated and drained
    after execution completes.
    """

    MAX_SERIALIZE_DEPTH = 50

    def __init__(
        self,
        session: Session,
        config: ClyroConfig,
        framework_version: str,
    ):
        self._session = session
        self._config = config
        self._framework_version = framework_version
        self._crew_agents: list[Any] = []  # Populated from event bus for force-stopping
        self._events: list[TraceEvent] = []
        # Start at step_number + 1 so first handler event gets step 1 (not 0).
        # step 0 is reserved for SESSION_START. If a handler event gets step 0,
        # session.record_event() reassigns it to session._step_number + 1,
        # which breaks chronological ordering when events are drained after
        # execution (by that time session._step_number has advanced).
        self._step_counter: int = session.step_number + 1
        self._local_cumulative_cost: Decimal = session.cumulative_cost
        self._task_start_times: dict[str, float] = {}
        self._current_task: str | None = None
        self._current_agent: str | None = None
        self._crew_state: dict[str, Any] = {
            "tasks_completed": [],
            "current_task": None,
            "inputs": {},
        }
        self._task_results: dict[str, Any] = {}
        self._drained: bool = False  # Set after drain_events(); gates jump-ahead logic
        # Implements FRD-003/FRD-004: Per-task parent event tracking
        self._task_event_ids: dict[str, UUID] = {}  # task_key → TASK_START event_id
        # Deferred error from event bus handlers. CrewAI's event bus swallows
        # all exceptions raised by handlers (catches Exception), so we store
        # enforcement errors here and re-raise them at the next control point
        # (_next_step) and after crew.kickoff() returns (in the wrapper).
        self._pending_error: PolicyViolationError | ExecutionControlError | None = None
        # Deferred approval from post-execution policy checks. CrewAI fires
        # events AFTER actions complete, so interactive approval prompts
        # inside event handlers are confusing (the action already ran).
        # We store require_approval decisions here and resolve them at the
        # next step boundary (_next_step), where the prompt appears BEFORE
        # the next action.
        # Tuple of (PolicyViolationError, action_type, approval_handler)
        self._pending_approval: tuple[PolicyViolationError, str, Any] | None = None

        logger.debug(
            "crewai_callback_handler_init",
            session_id=str(session.session_id),
            framework_version=framework_version,
        )

    # -- Implements FRD-003: task-level parent resolution --

    def _resolve_task_parent_event_id(self) -> UUID | None:
        """Resolve the parent event_id from the current task context.

        Returns the TASK_START event_id for the current task, or None if
        no task is active. Per FRD-003, TASK_START/TASK_END/TASK_ERROR
        have parent_event_id = None; only child events (LLM_CALL, TOOL_CALL,
        STEP) inherit the task's event_id.
        """
        if self._current_task is None:
            return None
        return self._task_event_ids.get(self._current_task)

    def on_crew_start(
        self,
        crew_name: str,
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Called when Crew execution begins."""
        context = {
            "start_time": time.perf_counter(),
            "crew_name": crew_name,
            "step_number": self._session.step_number,
        }
        self._crew_state = {
            "crew_name": crew_name,
            "inputs": self._serialize_data(inputs) if inputs else {},
            "tasks_completed": [],
            "current_task": None,
        }
        logger.debug(
            "crewai_crew_start",
            crew_name=crew_name,
            session_id=str(self._session.session_id),
        )
        return context

    def on_crew_end(
        self,
        crew_name: str,
        result: Any,
        context: dict[str, Any],
    ) -> TraceEvent:
        """Called when Crew execution completes successfully."""
        duration_ms = int((time.perf_counter() - context["start_time"]) * 1000)
        self._crew_state["result"] = self._serialize_data(result)

        output_data = None
        if self._config.capture_outputs:
            output_data = {"result": self._serialize_data(result)}

        event = create_step_event(
            session_id=self._session.session_id,
            step_number=self._next_step(),
            event_name=f"{crew_name}_complete",
            agent_id=self._session.agent_id,
            output_data=output_data,
            state_snapshot=self._crew_state if self._config.capture_state else None,
            duration_ms=duration_ms,
            cumulative_cost=self._session.cumulative_cost,
            agent_stage=AgentStage.OBSERVE,
            framework=Framework.CREWAI,
            framework_version=self._framework_version,
            metadata={
                "tasks_completed": len(self._crew_state.get("tasks_completed", [])),
            },
        )
        logger.debug(
            "crewai_crew_end",
            crew_name=crew_name,
            duration_ms=duration_ms,
            session_id=str(self._session.session_id),
        )
        return event

    def on_task_start(
        self,
        task_description: str,
        agent_role: str,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Called when a task execution begins."""
        task_key = task_id or task_description[:50]
        start_time = time.perf_counter()

        self._task_start_times[task_key] = start_time
        self._current_task = task_key
        self._current_agent = agent_role

        step_number = self._next_step()
        context = {
            "start_time": start_time,
            "step_number": step_number,
            "task_key": task_key,
            "task_description": task_description,
            "agent_role": agent_role,
        }

        self._crew_state["current_task"] = task_key

        input_data = None
        if self._config.capture_inputs:
            input_data = {
                "task_description": task_description,
                "agent_role": agent_role,
            }

        event = create_state_transition_event(
            session_id=self._session.session_id,
            step_number=step_number,
            node_name=f"task_start:{task_key[:30]}",
            agent_id=self._session.agent_id,
            input_data=input_data,
            state_snapshot=self._crew_state.copy() if self._config.capture_state else None,
            framework=Framework.CREWAI,
            framework_version=self._framework_version,
            agent_stage=AgentStage.THINK,
            metadata={
                "event_subtype": "task_start",
                "task_id": task_id,
                "agent_role": agent_role,
            },
        )
        event.event_type = EventType.TASK_START
        self._events.append(event)

        # Implements FRD-003: Store task_key → event_id for child event parent resolution.
        # FRD-004: Per-task-key dict (not single global), duplicate keys overwrite.
        self._task_event_ids[task_key] = event.event_id

        logger.debug(
            "crewai_task_start",
            task_key=task_key,
            agent_role=agent_role,
            session_id=str(self._session.session_id),
        )
        return context

    def on_task_end(
        self,
        task_output: Any,
        context: dict[str, Any],
    ) -> TraceEvent:
        """Called when a task completes successfully."""
        task_key = context["task_key"]
        start_time = context["start_time"]
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        self._task_results[task_key] = self._serialize_data(task_output)
        self._crew_state["tasks_completed"].append(task_key)
        self._crew_state["current_task"] = None

        output_data = None
        if self._config.capture_outputs:
            output_data = {"task_output": self._serialize_data(task_output)}

        event = create_state_transition_event(
            session_id=self._session.session_id,
            step_number=self._next_step(),
            node_name=f"task_end:{task_key[:30]}",
            agent_id=self._session.agent_id,
            output_data=output_data,
            state_snapshot=self._crew_state.copy() if self._config.capture_state else None,
            duration_ms=duration_ms,
            cumulative_cost=self._session.cumulative_cost,
            framework=Framework.CREWAI,
            framework_version=self._framework_version,
            agent_stage=AgentStage.OBSERVE,
            metadata={
                "event_subtype": "task_end",
                "task_key": task_key,
                "agent_role": context.get("agent_role"),
            },
        )
        event.event_type = EventType.TASK_END
        self._events.append(event)

        # Implements FRD-004: Remove task_key from parent tracking on completion
        self._task_event_ids.pop(task_key, None)

        self._current_task = None
        self._current_agent = None

        logger.debug(
            "crewai_task_end",
            task_key=task_key,
            duration_ms=duration_ms,
            session_id=str(self._session.session_id),
        )
        return event

    def on_task_error(
        self,
        error: Exception,
        context: dict[str, Any],
    ) -> TraceEvent:
        """Called when a task fails during execution."""
        task_key = context.get("task_key", "unknown")
        start_time = context.get("start_time", time.perf_counter())
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        event = create_error_event(
            session_id=self._session.session_id,
            step_number=context.get("step_number", self._session.step_number),
            error_type=type(error).__name__,
            error_message=str(error),
            agent_id=self._session.agent_id,
            error_stack=traceback.format_exc(),
            cumulative_cost=self._session.cumulative_cost,
            framework=Framework.CREWAI,
            framework_version=self._framework_version,
            input_data={"task_key": task_key, "agent_role": context.get("agent_role")},
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            metadata={
                "event_subtype": "task_error",
                "task_key": task_key,
                "agent_role": context.get("agent_role"),
                "duration_ms": duration_ms,
            },
        )
        self._events.append(event)

        # Implements FRD-004: Remove task_key from parent tracking on error
        self._task_event_ids.pop(task_key, None)

        self._crew_state["current_task"] = None
        self._crew_state["last_error"] = {
            "task_key": task_key,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        logger.debug(
            "crewai_task_error",
            task_key=task_key,
            error_type=type(error).__name__,
            session_id=str(self._session.session_id),
        )
        return event

    def on_agent_action(
        self,
        agent_role: str,
        action_type: str,
        action_input: dict[str, Any] | None = None,
        action_output: Any = None,
        tool_name: str | None = None,
        model: str | None = None,
        token_count_input: int = 0,
        token_count_output: int = 0,
        duration_ms: int = 0,
    ) -> TraceEvent:
        """Called when an agent performs an action (tool call, llm_call, etc.)."""
        input_data = None
        if self._config.capture_inputs and action_input:
            input_data = self._serialize_data(action_input)

        output_data = None
        if self._config.capture_outputs and action_output is not None:
            output_data = {"result": self._serialize_data(action_output)}

        agent_stage = self._determine_agent_stage(action_type, tool_name)

        step_number = self._next_step()

        # Implements FRD-003: Resolve task-level parent for child events
        task_parent_eid = self._resolve_task_parent_event_id()

        if action_type == "tool_call" and tool_name:
            event = create_tool_call_event(
                session_id=self._session.session_id,
                step_number=step_number,
                tool_name=tool_name,
                input_data=input_data or {},
                output_data=output_data,
                agent_id=self._session.agent_id,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                agent_stage=agent_stage,
                framework=Framework.CREWAI,
                framework_version=self._framework_version,
                metadata={
                    "agent_role": agent_role,
                    "current_task": self._current_task,
                },
                parent_event_id=task_parent_eid,
            )
        elif action_type == "llm_call":
            cost_usd = Decimal("0")
            if token_count_input > 0 or token_count_output > 0:
                calculator = CostCalculator(self._config)
                cost_usd = calculator.calculate(
                    token_count_input, token_count_output, model or "unknown"
                )

            # Track cost locally for real-time enforcement.
            # Session._cumulative_cost is only updated when events are drained
            # (after execution), so we must track and check here.
            if cost_usd > 0:
                self._local_cumulative_cost += cost_usd

            event = create_llm_call_event(
                session_id=self._session.session_id,
                step_number=step_number,
                model=model or "unknown",
                input_data=input_data or {},
                output_data=output_data,
                agent_id=self._session.agent_id,
                token_count_input=token_count_input,
                token_count_output=token_count_output,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                agent_stage=AgentStage.THINK,
                framework=Framework.CREWAI,
                framework_version=self._framework_version,
                metadata={
                    "agent_role": agent_role,
                    "current_task": self._current_task,
                },
                parent_event_id=task_parent_eid,
            )

            # Check cost limit after accumulating this LLM call's cost
            if cost_usd > 0:
                self._check_cost_limit()
        else:
            event = create_step_event(
                session_id=self._session.session_id,
                step_number=step_number,
                event_name=tool_name or action_type,
                agent_id=self._session.agent_id,
                input_data=input_data,
                output_data=output_data,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                agent_stage=agent_stage,
                framework=Framework.CREWAI,
                framework_version=self._framework_version,
                metadata={
                    "agent_role": agent_role,
                    "action_type": action_type,
                    "current_task": self._current_task,
                },
                parent_event_id=task_parent_eid,
            )

        self._events.append(event)

        # Implements PRD-010: Check for infinite loops after each agent action
        self._check_loop(action_type, action_input, agent_role)

        logger.debug(
            "crewai_agent_action",
            agent_role=agent_role,
            action_type=action_type,
            tool_name=tool_name,
            session_id=str(self._session.session_id),
        )
        return event

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear all accumulated events."""
        events = self._events.copy()
        self._events.clear()
        self._drained = True
        return events

    def get_task_results(self) -> dict[str, Any]:
        """Get all recorded task results."""
        return self._task_results.copy()

    def _next_step(self) -> int:
        """Get the next step number, increment the counter, and enforce step limit.

        Syncs the session's step counter with the handler's local counter
        so that step limit is enforced in real-time during CrewAI execution,
        not just after draining events.

        Also re-raises any pending error from a previous event bus handler
        that was swallowed by CrewAI's event bus. This prevents further event
        tracking after an enforcement error occurs.

        Cost limit is checked separately in on_agent_action() after LLM cost
        is calculated, since cost is not known at step-increment time.

        Raises:
            StepLimitExceededError: If step limit is exceeded
            PolicyViolationError: If a deferred policy violation is pending
            ExecutionControlError: If a deferred execution control error is pending
        """
        # Resolve any deferred require_approval decision BEFORE the next
        # action. The prompt appears here — between actions, not after.
        self._resolve_pending_approval()

        # Re-raise any deferred error from a previous event bus handler.
        # CrewAI's event bus swallows all exceptions, so we re-raise here
        # to prevent further event tracking.  Also force-stop the crew so
        # it doesn't keep running until max_iter.
        if self._pending_error is not None:
            self._force_stop_crew()
            raise self._pending_error

        step = self._step_counter
        self._step_counter += 1
        # After drain_events(), the wrapper calls session.record_step() which
        # advances session._step_number past our local counter.  Jump ahead to
        # avoid step number collisions (e.g., crew_complete vs agent_execution).
        # Before drain, session._step_number may be bumped by policy-check events
        # but that does NOT mean a collision — handler events are still buffered
        # locally, so we must NOT jump ahead pre-drain.
        if self._drained and step <= self._session._step_number:
            step = self._session._step_number + 1
            self._step_counter = step + 1
        # Sync session step counter for real-time execution control enforcement.
        if step > self._session._step_number:
            self._session._step_number = step
        self._session._check_step_limit()
        return step

    def _check_loop(self, action_type: str, action_input: dict[str, Any] | None, agent_role: str | None) -> None:
        """Check for infinite loops via session's LoopDetector.

        Builds a state snapshot from the current action context and delegates
        to the session's loop detector. Called after each agent action
        (LLM call, tool call) — same pattern as LangGraph's on_chain_end.

        If a loop is detected, an error event is recorded before the
        LoopDetectedError propagates, ensuring the event reaches ClickHouse
        even if CrewAI's event bus swallows the exception.

        Args:
            action_type: Type of action (llm_call, tool_call, etc.)
            action_input: Input data for the action
            agent_role: The CrewAI agent role performing the action

        Raises:
            LoopDetectedError: If a loop is detected
        """
        if not self._config.controls.enable_loop_detection:
            return
        # Skip if a loop was already detected — avoid duplicate error events
        # and repeated raises that CrewAI's event bus keeps swallowing.
        if self._pending_error is not None:
            return
        state = {
            "action_type": action_type,
            "agent_role": agent_role or "unknown",
            "input": action_input or {},
        }
        try:
            self._session._check_loop_detection(state, action=action_type)
        except Exception as e:
            # Record error event before propagating so it's captured
            # even if CrewAI's event bus swallows the exception.
            error_event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._session.step_number,
                error_type=type(e).__name__,
                error_message=str(e),
                agent_id=self._session.agent_id,
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.CREWAI,
                framework_version=self._framework_version,
                metadata={
                    "event_name": "loop_detection",
                    "action_type": action_type,
                    "agent_role": agent_role,
                },
            )
            if error_event is not None:
                self._events.append(error_event)
            # Force-stop the crew so it doesn't keep running until max_iter.
            # CrewAI's event bus swallows all exceptions, so raising alone
            # won't stop execution — we must set max_iter=0 on all agents.
            self._force_stop_crew()
            raise

    def _force_stop_crew(self) -> None:
        """Force-stop the CrewAI crew by setting max_iter=0 on all agents.

        CrewAI's executor copies max_iter from the agent at init time into
        its own self.max_iter. The iteration loop checks executor.max_iter,
        NOT agent.max_iter. So we must set it on both the agent AND the
        executor (agent.agent_executor.max_iter) for it to take effect.

        Agent references are captured from AgentExecutionStartedEvent in the
        event bus handlers.
        """
        if not self._crew_agents:
            return
        try:
            for agent in self._crew_agents:
                agent.max_iter = 0
                # The executor copies max_iter at init — must set on executor too
                executor = getattr(agent, "agent_executor", None)
                if executor is not None:
                    executor.max_iter = 0
            logger.info(
                "crewai_force_stopped",
                reason="execution_control_error",
                agents_stopped=len(self._crew_agents),
                session_id=str(self._session.session_id),
            )
        except Exception as e:
            logger.warning("crewai_force_stop_failed", error=str(e))

    def _check_cost_limit(self) -> None:
        """Check if the locally-tracked cumulative cost exceeds the limit.

        Cost is tracked locally in the handler because session._cumulative_cost
        is only updated when events are drained (after execution). This method
        provides real-time cost enforcement during CrewAI execution.

        Raises:
            CostLimitExceededError: If cost limit is exceeded
        """
        if not self._config.controls.enable_cost_limit:
            return

        max_cost = Decimal(str(self._config.controls.max_cost_usd))
        if self._local_cumulative_cost > max_cost:
            raise CostLimitExceededError(
                limit_usd=float(max_cost),
                current_cost_usd=float(self._local_cumulative_cost),
                session_id=str(self._session.session_id),
                step_number=self._step_counter,
            )

    def _check_policy_deferred_approval(self, action_type: str, params: dict[str, Any]) -> None:
        """Check policy with deferred approval handling for CrewAI.

        CrewAI's event bus fires AFTER actions complete, so interactive
        approval prompts inside event handlers are confusing (the action
        already ran). This method suppresses the approval handler during
        the policy check. If the backend returns require_approval, the
        decision is stored as _pending_approval and resolved at the next
        step boundary (_next_step), where the prompt appears BEFORE the
        next action.

        Block and allow decisions are handled normally:
        - allow: returns silently
        - block: raises PolicyViolationError (caught by event bus handler)
        - require_approval: stored as _pending_approval, returns silently
        """
        parent_eid = self._resolve_task_parent_event_id()
        evaluator = self._session._policy_evaluator
        if evaluator is None:
            # No policy enforcement configured — use normal check (no-op)
            self._session.check_policy(
                action_type,
                params,
                parent_event_id=parent_eid,
                cumulative_cost=self._local_cumulative_cost,
            )
            return

        # Save and suppress the approval handler during the check
        saved_handler = evaluator._approval_handler
        evaluator._approval_handler = None
        try:
            self._session.check_policy(
                action_type,
                params,
                parent_event_id=parent_eid,
                cumulative_cost=self._local_cumulative_cost,
            )
        except PolicyViolationError as e:
            if e.details and e.details.get("decision") == "require_approval":
                # Defer to _next_step() — don't propagate to event bus
                self._pending_approval = (e, action_type, saved_handler)
                return
            raise  # Regular block — propagate to _pending_error
        finally:
            evaluator._approval_handler = saved_handler

    def _resolve_pending_approval(self) -> None:
        """Resolve a deferred require_approval decision.

        Called from _next_step() (before the next action) and from the
        wrapper after crew.kickoff() returns.

        If an approval handler is available, prompts the user:
        - Approved: clears the pending state, execution continues
        - Denied: raises PolicyViolationError to stop execution

        If no handler is available, raises immediately (same as block).

        Raises:
            PolicyViolationError: If the action is denied or no handler exists
        """
        if self._pending_approval is None:
            return

        error, action_type, approval_handler = self._pending_approval
        self._pending_approval = None

        if approval_handler is not None:
            from clyro.policy import PolicyDecision

            decision = PolicyDecision(
                decision="require_approval",
                rule_id=error.rule_id,
                rule_name=error.rule_name,
                message=error.message,
            )
            try:
                approved = approval_handler(decision, action_type)
            except Exception as exc:
                logger.warning(
                    "deferred_approval_handler_error",
                    error=str(exc),
                    action_type=action_type,
                    rule_id=error.rule_id,
                )
                approved = False

            if approved:
                logger.info(
                    "policy_approval_granted_deferred",
                    action_type=action_type,
                    rule_id=error.rule_id,
                    rule_name=error.rule_name,
                )
                return  # User approved — continue execution

            logger.info(
                "policy_approval_denied_deferred",
                action_type=action_type,
                rule_id=error.rule_id,
                rule_name=error.rule_name,
            )

        # No handler or denied — stop execution
        raise error

    def _determine_agent_stage(
        self,
        action_type: str,
        tool_name: str | None = None,
    ) -> AgentStage:
        """Determine the agent execution stage based on action type."""
        action_lower = action_type.lower()

        if tool_name or action_lower in ("tool_call", "action", "execute", "run", "call"):
            return AgentStage.ACT

        if action_lower in ("observe", "result", "output", "response", "read", "get"):
            return AgentStage.OBSERVE

        return AgentStage.THINK

    def _serialize_data(self, data: Any, depth: int = 0) -> Any:
        """Serialize data for storage with depth limiting."""
        if depth >= self.MAX_SERIALIZE_DEPTH:
            return "<max_depth_exceeded>"
        return self._serialize_value(data, depth)

    def _serialize_value(self, value: Any, depth: int = 0) -> Any:
        """Recursively serialize a value for JSON storage."""
        if depth >= self.MAX_SERIALIZE_DEPTH:
            return "<max_depth_exceeded>"

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item, depth + 1) for item in value]

        if isinstance(value, dict):
            return {
                str(k): self._serialize_value(v, depth + 1)
                for k, v in value.items()
                if not str(k).startswith("_")
            }

        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception as e:
                logger.debug(
                    "serialization_model_dump_fallback",
                    value_type=type(value).__name__,
                    error=str(e),
                )

        if hasattr(value, "__dict__"):
            return {
                k: self._serialize_value(v, depth + 1)
                for k, v in value.__dict__.items()
                if not k.startswith("_")
            }

        try:
            return str(value)
        except Exception as e:
            logger.debug(
                "serialization_str_fallback",
                value_type=type(value).__name__,
                error=str(e),
            )
            return "<unserializable>"


class CrewAIAdapter:
    """
    Adapter for wrapping CrewAI Crew instances.

    Uses CrewAI's native event bus to capture:
    - LLM calls (model, tokens, messages)
    - Tool usage (tool name, args, output, duration)
    - Task lifecycle (start, end, error)
    - Error events
    """

    FRAMEWORK = Framework.CREWAI

    def __init__(
        self,
        agent: Any,
        config: ClyroConfig,
        validate_version: bool = True,
    ):
        if validate_version:
            self._framework_version = validate_crewai_version()
        else:
            self._framework_version = detect_crewai_version() or "unknown"

        self._agent = agent
        self._config = config
        self._name = self._extract_crew_name(agent)

        logger.debug(
            "crewai_adapter_init",
            crew_name=self._name,
            framework_version=self._framework_version,
        )

    @property
    def agent(self) -> Any:
        """Get the wrapped Crew instance."""
        return self._agent

    @property
    def name(self) -> str:
        """Get the crew name."""
        return self._name

    @property
    def framework(self) -> Framework:
        """Get the framework type."""
        return self.FRAMEWORK

    @property
    def framework_version(self) -> str:
        """Get the framework version."""
        return self._framework_version

    def create_callback_handler(self, session: Session) -> CrewAICallbackHandler:
        """Create a callback handler for this session."""
        return CrewAICallbackHandler(
            session=session,
            config=self._config,
            framework_version=self._framework_version,
        )

    def before_call(
        self,
        session: Session,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Hook called before the crew is invoked.

        Registers event bus handlers for automatic event capture.

        Args:
            session: Current session
            args: Positional arguments to kickoff
            kwargs: Keyword arguments to kickoff

        Returns:
            Context dictionary to pass to after_call
        """
        handler = self.create_callback_handler(session)
        inputs = kwargs.get("inputs", {})

        crew_context = handler.on_crew_start(
            crew_name=self._name,
            inputs=inputs,
        )

        registered_handlers = self._register_event_bus_handlers(handler)

        context: dict[str, Any] = {
            "start_time": time.perf_counter(),
            "step_number": session.step_number,
            "handler": handler,
            "crew_context": crew_context,
            "registered_handlers": registered_handlers,
        }

        logger.debug(
            "crewai_before_call",
            crew_name=self._name,
            handler_count=len(registered_handlers),
            session_id=str(session.session_id),
        )

        return context

    def after_call(
        self,
        session: Session,
        result: Any,
        context: dict[str, Any],
    ) -> TraceEvent:
        """Hook called after successful crew execution."""
        duration_ms = int((time.perf_counter() - context["start_time"]) * 1000)
        handler: CrewAICallbackHandler = context["handler"]

        self._unregister_event_bus_handlers(
            context.get("registered_handlers", []),
        )

        # Restore original BaseLLM._track_token_usage_internal
        self._restore_token_patch(handler)

        event = handler.on_crew_end(
            crew_name=self._name,
            result=result,
            context=context["crew_context"],
        )
        event.duration_ms = duration_ms
        event.metadata.update({"tasks_completed": len(handler.get_task_results())})

        logger.debug(
            "crewai_after_call",
            crew_name=self._name,
            duration_ms=duration_ms,
            session_id=str(session.session_id),
        )
        return event

    def on_error(
        self,
        session: Session,
        error: Exception,
        context: dict[str, Any],
    ) -> TraceEvent:
        """Hook called when crew execution fails."""
        handler = context.get("handler")
        self._unregister_event_bus_handlers(
            context.get("registered_handlers", []),
        )
        if handler:
            self._restore_token_patch(handler)

        duration_ms = 0
        if "start_time" in context:
            duration_ms = int((time.perf_counter() - context["start_time"]) * 1000)

        event = create_error_event(
            session_id=session.session_id,
            step_number=context.get("step_number", session.step_number),
            error_type=type(error).__name__,
            error_message=str(error),
            agent_id=session.agent_id,
            error_stack=traceback.format_exc(),
            cumulative_cost=session.cumulative_cost,
            framework=Framework.CREWAI,
            framework_version=self._framework_version,
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            metadata={
                "crew_name": self._name,
                "duration_ms": duration_ms,
            },
        )

        logger.debug(
            "crewai_on_error",
            crew_name=self._name,
            error_type=type(error).__name__,
            session_id=str(session.session_id),
        )
        return event

    def _register_event_bus_handlers(self, handler: CrewAICallbackHandler) -> list[tuple[Any, Any]]:
        """
        Register handlers on CrewAI's native event bus.

        Captures: LLM calls, tool usage, task lifecycle, agent lifecycle, agent errors.
        The event bus is global so this works for both direct Crew
        instances and wrapper classes.

        Returns:
            List of (event_type, handler_func) tuples for unregistration.
            Empty list if event bus is not available.
        """
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.events.types.agent_events import (
                AgentExecutionCompletedEvent,
                AgentExecutionErrorEvent,
                AgentExecutionStartedEvent,
            )
            from crewai.events.types.crew_events import (
                CrewKickoffCompletedEvent,
            )
            from crewai.events.types.llm_events import (
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
            )
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )
            from crewai.events.types.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
            )
        except ImportError:
            logger.debug("crewai_event_bus_not_available")
            return []

        registered: list[tuple[Any, Any]] = []

        # --- Class-level patch to capture per-call token usage ---
        # BaseLLM._track_token_usage_internal is called by ALL providers
        # (litellm, Anthropic, Gemini) with raw usage data BEFORE the event
        # bus emission. Patching at the class level works for LLM instances
        # created at any point (including inside wrapper invoke() methods).
        try:
            from crewai.llms.base_llm import BaseLLM

            original_track = BaseLLM._track_token_usage_internal
            handler._pending_tokens = []
            handler._original_track_token = original_track

            def _patched_track(self_llm: Any, usage_data: dict[str, Any]) -> None:
                original_track(self_llm, usage_data)
                prompt_tokens = (
                    usage_data.get("prompt_tokens")
                    or usage_data.get("prompt_token_count")
                    or usage_data.get("input_tokens")
                    or 0
                )
                completion_tokens = (
                    usage_data.get("completion_tokens")
                    or usage_data.get("candidates_token_count")
                    or usage_data.get("output_tokens")
                    or 0
                )
                handler._pending_tokens.append((prompt_tokens, completion_tokens))

            BaseLLM._track_token_usage_internal = _patched_track
            logger.debug("crewai_class_level_token_patch_applied")
        except ImportError:
            handler._pending_tokens = []
            handler._original_track_token = None
            logger.debug("crewai_base_llm_not_available_for_token_patch")

        # --- Crew Kickoff Completed (captures aggregate token usage) ---
        def on_crew_kickoff_completed(source: Any, event: Any) -> None:
            try:
                # source is the Crew instance; token_usage is set before this event fires
                token_usage = getattr(source, "token_usage", None)
                if token_usage:
                    handler._crew_prompt_tokens = getattr(token_usage, "prompt_tokens", 0) or 0
                    handler._crew_completion_tokens = (
                        getattr(token_usage, "completion_tokens", 0) or 0
                    )
                    handler._crew_total_tokens = getattr(token_usage, "total_tokens", 0) or 0
                else:
                    handler._crew_total_tokens = getattr(event, "total_tokens", 0) or 0

                # Also try to get model from the crew's agents
                agents = getattr(source, "agents", None)
                if agents:
                    for agent_obj in agents:
                        llm = getattr(agent_obj, "llm", None)
                        if llm:
                            model_name = getattr(llm, "model", None) or getattr(
                                llm, "model_name", None
                            )
                            if model_name:
                                handler._crew_model = str(model_name)
                                break

                logger.debug(
                    "crewai_kickoff_completed_tokens",
                    prompt_tokens=getattr(handler, "_crew_prompt_tokens", 0),
                    completion_tokens=getattr(handler, "_crew_completion_tokens", 0),
                    total_tokens=getattr(handler, "_crew_total_tokens", 0),
                )
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_crew_kickoff_completed_error", error=str(e))

        crewai_event_bus.register_handler(CrewKickoffCompletedEvent, on_crew_kickoff_completed)
        registered.append((CrewKickoffCompletedEvent, on_crew_kickoff_completed))

        # --- LLM Call Completed ---
        def on_llm_completed(source: Any, event: Any) -> None:
            try:
                model = getattr(event, "model", None) or "unknown"
                response = getattr(event, "response", None)
                messages = getattr(event, "messages", None)
                agent_role = getattr(event, "agent_role", None) or "unknown"

                # Pop per-call tokens from the class-level patch buffer
                input_tokens, output_tokens = 0, 0
                if handler._pending_tokens:
                    input_tokens, output_tokens = handler._pending_tokens.pop(0)

                output_text = None
                if response:
                    if isinstance(response, str):
                        output_text = response
                    else:
                        choices = getattr(response, "choices", None)
                        if choices and len(choices) > 0:
                            message = getattr(choices[0], "message", None)
                            if message:
                                output_text = getattr(message, "content", None)

                input_data = None
                if messages:
                    if isinstance(messages, list):
                        last_msg = messages[-1] if messages else {}
                        content = (
                            last_msg.get("content", "")
                            if isinstance(last_msg, dict)
                            else str(last_msg)
                        )
                        input_data = {
                            "message_count": len(messages),
                            "last_message": str(content)[:MAX_MESSAGE_LENGTH],
                        }
                    else:
                        input_data = {"messages": str(messages)[:MAX_MESSAGE_LENGTH]}

                output_data = None
                if output_text:
                    output_data = {"content": str(output_text)[:MAX_MESSAGE_LENGTH]}

                handler.on_agent_action(
                    agent_role=agent_role,
                    action_type="llm_call",
                    action_input=input_data,
                    action_output=output_data,
                    model=model,
                    token_count_input=input_tokens,
                    token_count_output=output_tokens,
                )

                # Post-LLM policy check with cost for cost-based rules,
                # step_number for step-based rules, and input for text-based
                # rules (contains, not_contains). Cost must come from handler's
                # local tracker (session cost is stale during CrewAI execution).
                llm_policy_params: dict[str, Any] = {
                    "model": model,
                    "cost": float(handler._local_cumulative_cost),
                    "step_number": handler._step_counter,
                    # Implements FRD-013: Include `output` for response-content policies
                    "output": str(output_text)[:MAX_MESSAGE_LENGTH] if output_text else "",
                }
                # Extract last user message for text-based policy rules
                # (e.g. contains "DROP TABLE", not_contains "disclaimer").
                if messages and isinstance(messages, list):
                    for msg in reversed(messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            llm_policy_params["input"] = str(msg.get("content", ""))[
                                :MAX_MESSAGE_LENGTH
                            ]
                            break
                    else:
                        # No user message found — use last message content
                        last = messages[-1]
                        if isinstance(last, dict):
                            llm_policy_params["input"] = str(last.get("content", ""))[
                                :MAX_MESSAGE_LENGTH
                            ]
                        else:
                            llm_policy_params["input"] = str(last)[:MAX_MESSAGE_LENGTH]
                handler._check_policy_deferred_approval("llm_call", llm_policy_params)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_llm_completed_error", error=str(e))

        crewai_event_bus.register_handler(LLMCallCompletedEvent, on_llm_completed)
        registered.append((LLMCallCompletedEvent, on_llm_completed))

        # --- LLM Call Failed ---
        def on_llm_failed(source: Any, event: Any) -> None:
            try:
                error_msg = getattr(event, "error", "Unknown LLM error")
                model = getattr(event, "model", None) or "unknown"
                agent_role = getattr(event, "agent_role", None) or "unknown"

                handler.on_agent_action(
                    agent_role=agent_role,
                    action_type="llm_call",
                    action_input={"model": model},
                    action_output={"error": str(error_msg)[:MAX_MESSAGE_LENGTH]},
                    model=model,
                )
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_llm_failed_error", error=str(e))

        crewai_event_bus.register_handler(LLMCallFailedEvent, on_llm_failed)
        registered.append((LLMCallFailedEvent, on_llm_failed))

        # --- Tool Usage Finished ---
        def on_tool_finished(source: Any, event: Any) -> None:
            try:
                tool_name = getattr(event, "tool_name", "unknown")
                tool_args = getattr(event, "tool_args", {})
                output = getattr(event, "output", None)
                agent_role = getattr(event, "agent_role", None) or "unknown"

                duration_ms = 0
                started_at = getattr(event, "started_at", None)
                finished_at = getattr(event, "finished_at", None)
                if started_at and finished_at:
                    duration_ms = int((finished_at - started_at).total_seconds() * 1000)

                action_input = (
                    tool_args
                    if isinstance(tool_args, dict)
                    else {"args": str(tool_args)[:MAX_MESSAGE_LENGTH]}
                )

                handler.on_agent_action(
                    agent_role=agent_role,
                    action_type="tool_call",
                    action_input=action_input,
                    action_output=output,
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                )

                # Post-tool policy check — prevents further execution on violation.
                # Include tool_args for field-based rules (e.g. rmq_cluster, site_name),
                # cost for cost-based rules, and step_number for step-based rules.
                # Flatten one level of nested dicts so fields like "rmq_cluster"
                # are accessible directly (matching LangGraph's on_tool_start approach).
                tool_policy_params: dict[str, Any] = {
                    "tool_name": tool_name,
                    "cost": float(handler._local_cumulative_cost),
                    "step_number": handler._step_counter,
                }
                if isinstance(tool_args, dict):
                    tool_policy_params.update(tool_args)
                    for v in tool_args.values():
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                tool_policy_params.setdefault(k2, v2)
                handler._check_policy_deferred_approval("tool_call", tool_policy_params)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_tool_finished_error", error=str(e))

        crewai_event_bus.register_handler(ToolUsageFinishedEvent, on_tool_finished)
        registered.append((ToolUsageFinishedEvent, on_tool_finished))

        # --- Tool Usage Error ---
        def on_tool_error(source: Any, event: Any) -> None:
            try:
                tool_name = getattr(event, "tool_name", "unknown")
                error = getattr(event, "error", "Unknown tool error")
                agent_role = getattr(event, "agent_role", None) or "unknown"

                handler.on_agent_action(
                    agent_role=agent_role,
                    action_type="tool_call",
                    action_input={"tool_name": tool_name},
                    action_output={"error": str(error)[:MAX_MESSAGE_LENGTH]},
                    tool_name=tool_name,
                )
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_tool_error_error", error=str(e))

        crewai_event_bus.register_handler(ToolUsageErrorEvent, on_tool_error)
        registered.append((ToolUsageErrorEvent, on_tool_error))

        # --- Task Started ---
        def on_task_started(source: Any, event: Any) -> None:
            try:
                task = getattr(event, "task", None)

                task_description = "unknown_task"
                agent_role = "unknown"
                task_id_str = None

                if task:
                    task_description = getattr(task, "description", "unknown_task")
                    if hasattr(task, "agent") and task.agent:
                        agent_role = getattr(task.agent, "role", "unknown")
                    task_id_str = getattr(event, "task_id", None) or str(id(task))

                handler.on_task_start(
                    task_description=task_description,
                    agent_role=agent_role,
                    task_id=task_id_str,
                )
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_task_started_error", error=str(e))

        crewai_event_bus.register_handler(TaskStartedEvent, on_task_started)
        registered.append((TaskStartedEvent, on_task_started))

        # --- Task Completed ---
        def on_task_completed(source: Any, event: Any) -> None:
            try:
                task = getattr(event, "task", None)
                output = getattr(event, "output", None)

                task_description = "unknown_task"
                agent_role = "unknown"
                task_id_str = None

                if task:
                    task_description = getattr(task, "description", "unknown_task")
                    if hasattr(task, "agent") and task.agent:
                        agent_role = getattr(task.agent, "role", "unknown")
                    task_id_str = getattr(event, "task_id", None) or str(id(task))

                task_key = task_id_str or task_description[:50]
                start_time = handler._task_start_times.get(task_key, time.perf_counter())

                ctx = {
                    "task_key": task_key,
                    "start_time": start_time,
                    "task_description": task_description,
                    "agent_role": agent_role,
                }

                handler.on_task_end(task_output=output, context=ctx)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_task_completed_error", error=str(e))

        crewai_event_bus.register_handler(TaskCompletedEvent, on_task_completed)
        registered.append((TaskCompletedEvent, on_task_completed))

        # --- Task Failed ---
        def on_task_failed(source: Any, event: Any) -> None:
            try:
                error_msg = getattr(event, "error", "Unknown task error")
                task = getattr(event, "task", None)

                task_description = "unknown_task"
                agent_role = "unknown"
                task_id_str = None

                if task:
                    task_description = getattr(task, "description", "unknown_task")
                    if hasattr(task, "agent") and task.agent:
                        agent_role = getattr(task.agent, "role", "unknown")
                    task_id_str = getattr(event, "task_id", None) or str(id(task))

                task_key = task_id_str or task_description[:50]

                ctx = {
                    "task_key": task_key,
                    "start_time": handler._task_start_times.get(task_key, time.perf_counter()),
                    "task_description": task_description,
                    "agent_role": agent_role,
                    "step_number": handler._step_counter,
                }

                handler.on_task_error(error=Exception(str(error_msg)), context=ctx)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_task_failed_error", error=str(e))

        crewai_event_bus.register_handler(TaskFailedEvent, on_task_failed)
        registered.append((TaskFailedEvent, on_task_failed))

        # --- Agent Execution Error ---
        def on_agent_error(source: Any, event: Any) -> None:
            try:
                agent = getattr(event, "agent", None)
                error_msg = getattr(event, "error", "Unknown agent error")

                agent_role = "unknown"
                if agent:
                    agent_role = getattr(agent, "role", "unknown")

                # Implements FRD-003: agent error events are children of task_start
                task_parent_eid = handler._resolve_task_parent_event_id()

                error_event = create_error_event(
                    session_id=handler._session.session_id,
                    step_number=handler._next_step(),
                    error_type="AgentExecutionError",
                    error_message=str(error_msg),
                    agent_id=handler._session.agent_id,
                    cumulative_cost=handler._session.cumulative_cost,
                    framework=Framework.CREWAI,
                    framework_version=handler._framework_version,
                    output_data={
                        "error_type": "AgentExecutionError",
                        "error_message": str(error_msg),
                    },
                    metadata={
                        "event_subtype": "agent_execution_error",
                        "agent_role": agent_role,
                    },
                    parent_event_id=task_parent_eid,
                )
                handler._events.append(error_event)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_agent_error_error", error=str(e))

        crewai_event_bus.register_handler(AgentExecutionErrorEvent, on_agent_error)
        registered.append((AgentExecutionErrorEvent, on_agent_error))

        # --- Agent Execution Started ---
        def on_agent_started(source: Any, event: Any) -> None:
            try:
                agent = getattr(event, "agent", None)
                task = getattr(event, "task", None)

                # Capture agent reference for force-stopping on enforcement errors
                if agent and agent not in handler._crew_agents:
                    handler._crew_agents.append(agent)

                agent_role = "unknown"
                if agent:
                    agent_role = getattr(agent, "role", "unknown")

                task_description = ""
                if task:
                    task_description = getattr(task, "description", "")

                tools = getattr(event, "tools", None)
                tool_names = []
                if tools:
                    for t in tools:
                        name = getattr(t, "name", None) or str(t)
                        tool_names.append(name)

                # Implements FRD-003: agent lifecycle events are children of task_start
                task_parent_eid = handler._resolve_task_parent_event_id()

                event_obj = create_step_event(
                    session_id=handler._session.session_id,
                    step_number=handler._next_step(),
                    event_name=f"agent_start:{agent_role[:MAX_EVENT_NAME_LENGTH]}",
                    agent_id=handler._session.agent_id,
                    duration_ms=0,
                    cumulative_cost=handler._session.cumulative_cost,
                    framework=Framework.CREWAI,
                    framework_version=handler._framework_version,
                    agent_stage=AgentStage.THINK,
                    input_data={
                        "agent_role": agent_role,
                        "task_description": task_description[:MAX_TASK_DESCRIPTION_LENGTH],
                        "tools": tool_names,
                    }
                    if handler._config.capture_inputs
                    else None,
                    metadata={
                        "event_subtype": "agent_execution_started",
                        "agent_role": agent_role,
                    },
                    parent_event_id=task_parent_eid,
                )
                handler._events.append(event_obj)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_agent_started_error", error=str(e))

        crewai_event_bus.register_handler(AgentExecutionStartedEvent, on_agent_started)
        registered.append((AgentExecutionStartedEvent, on_agent_started))

        # --- Agent Execution Completed ---
        def on_agent_completed(source: Any, event: Any) -> None:
            try:
                agent = getattr(event, "agent", None)
                output = getattr(event, "output", None)

                agent_role = "unknown"
                if agent:
                    agent_role = getattr(agent, "role", "unknown")

                output_data = None
                if handler._config.capture_outputs and output:
                    output_data = {"output": str(output)[:MAX_MESSAGE_LENGTH]}

                # Implements FRD-003: agent lifecycle events are children of task_start
                task_parent_eid = handler._resolve_task_parent_event_id()

                event_obj = create_step_event(
                    session_id=handler._session.session_id,
                    step_number=handler._next_step(),
                    event_name=f"agent_end:{agent_role[:MAX_EVENT_NAME_LENGTH]}",
                    agent_id=handler._session.agent_id,
                    duration_ms=0,
                    cumulative_cost=handler._session.cumulative_cost,
                    framework=Framework.CREWAI,
                    framework_version=handler._framework_version,
                    agent_stage=AgentStage.OBSERVE,
                    output_data=output_data,
                    metadata={
                        "event_subtype": "agent_execution_completed",
                        "agent_role": agent_role,
                    },
                    parent_event_id=task_parent_eid,
                )
                handler._events.append(event_obj)
            except (PolicyViolationError, ExecutionControlError) as e:
                handler._pending_error = e
                raise
            except Exception as e:
                logger.error("event_bus_agent_completed_error", error=str(e))

        crewai_event_bus.register_handler(AgentExecutionCompletedEvent, on_agent_completed)
        registered.append((AgentExecutionCompletedEvent, on_agent_completed))

        logger.debug(
            "crewai_event_bus_handlers_registered",
            handler_count=len(registered),
        )
        return registered

    def _unregister_event_bus_handlers(
        self,
        registered_handlers: list[tuple[Any, Any]],
    ) -> None:
        """
        Unregister handlers from CrewAI's event bus.

        Manually removes handlers from the internal frozenset storage
        since CrewAI has no built-in unregister method.
        """
        if not registered_handlers:
            return
        try:
            from crewai.events.event_bus import crewai_event_bus

            for event_type, handler_func in registered_handlers:
                existing = crewai_event_bus._sync_handlers.get(event_type, frozenset())
                crewai_event_bus._sync_handlers[event_type] = existing - {handler_func}
                crewai_event_bus._execution_plan_cache.pop(event_type, None)

            logger.debug(
                "crewai_event_bus_handlers_unregistered",
                handler_count=len(registered_handlers),
            )
        except ImportError:
            pass
        except Exception as e:
            logger.error("crewai_event_bus_unregister_error", error=str(e))

    def _restore_token_patch(self, handler: CrewAICallbackHandler) -> None:
        """Restore original BaseLLM._track_token_usage_internal after execution."""
        original = getattr(handler, "_original_track_token", None)
        if original is not None:
            try:
                from crewai.llms.base_llm import BaseLLM

                BaseLLM._track_token_usage_internal = original
                logger.debug("crewai_class_level_token_patch_restored")
            except ImportError:
                pass

    def _extract_crew_name(self, agent: Any) -> str:
        """Extract the crew name from the Crew instance."""
        if hasattr(agent, "name") and agent.name:
            return str(agent.name)
        if hasattr(agent, "config") and hasattr(agent.config, "name"):
            return str(agent.config.name)
        return type(agent).__name__

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize the crew result for storage."""
        if result is None:
            return {"result": None}
        if isinstance(result, (str, int, float, bool)):
            return {"result": result}
        if isinstance(result, (list, tuple)):
            return {"result": list(result)}
        if isinstance(result, dict):
            return {"result": result}
        if hasattr(result, "raw"):
            return {"result": str(result.raw), "type": type(result).__name__}
        if hasattr(result, "model_dump"):
            return {"result": result.model_dump()}
        if hasattr(result, "__dict__"):
            return {"result": {k: v for k, v in result.__dict__.items() if not k.startswith("_")}}
        return {"result": str(result)}
