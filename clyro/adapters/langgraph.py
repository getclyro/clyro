# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK LangGraph Adapter
# Implements PRD-003

"""
LangGraph-specific adapter for StateGraph tracing.

This adapter integrates with LangGraph's callback system to capture:
- Node executions as distinct trace events
- State transitions with state snapshots
- Conditional edge decisions showing the actual path taken

Supported LangGraph versions: 0.2.0+
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

import structlog

from clyro.cost import CostCalculator
from clyro.exceptions import ExecutionControlError, FrameworkVersionError, PolicyViolationError
from clyro.session import Session
from clyro.trace import (
    AgentStage,
    Framework,
    TraceEvent,
    create_error_event,
    create_llm_call_event,
    create_retriever_call_event,
    create_state_transition_event,
    create_step_event,
    create_tool_call_event,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Version constraints
MIN_LANGGRAPH_VERSION = "0.2.0"
MAX_LANGGRAPH_VERSION = None  # None = latest supported


def _parse_version(version_str: str) -> tuple[int, ...]:
    """
    Parse a version string into a tuple of integers for comparison.

    Handles versions like "0.2.0", "0.2.0rc1", "0.2.0.dev1".

    Args:
        version_str: Version string to parse

    Returns:
        Tuple of version components (major, minor, patch)
    """
    # Remove common suffixes like rc1, dev1, a1, b1
    clean_version = version_str.split("rc")[0].split("dev")[0].split("a")[0].split("b")[0]
    # Remove any remaining non-numeric suffixes
    clean_version = clean_version.rstrip(".")

    try:
        parts = clean_version.split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, IndexError):
        # If parsing fails, return a safe minimum
        logger.warning("version_parse_failed", version=version_str)
        return (0, 0, 0)


def _is_version_supported(version: str) -> bool:
    """
    Check if a LangGraph version is supported.

    Args:
        version: Version string to check

    Returns:
        True if version is within supported range
    """
    parsed = _parse_version(version)
    min_parsed = _parse_version(MIN_LANGGRAPH_VERSION)

    if parsed < min_parsed:
        return False

    if MAX_LANGGRAPH_VERSION is not None:
        max_parsed = _parse_version(MAX_LANGGRAPH_VERSION)
        if parsed > max_parsed:
            return False

    return True


def detect_langgraph_version() -> str | None:
    """
    Detect the installed LangGraph version.

    Returns:
        Version string if LangGraph is installed, None otherwise
    """
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return None

    # Prefer importlib.metadata (works for all properly packaged distributions)
    try:
        from importlib.metadata import version

        return version("langgraph")
    except Exception:
        pass

    # Fallback to __version__ attribute
    ver = getattr(langgraph, "__version__", None)
    if ver:
        return ver

    return "unknown"


def validate_langgraph_version(version: str | None = None) -> str:
    """
    Validate that the LangGraph version is supported.

    Args:
        version: Optional version to validate. If None, detects installed version.

    Returns:
        The validated version string

    Raises:
        FrameworkVersionError: If LangGraph is not installed or version is unsupported
    """
    if version is None:
        version = detect_langgraph_version()

    if version is None:
        raise FrameworkVersionError(
            framework="langgraph",
            version="not installed",
            supported=f">={MIN_LANGGRAPH_VERSION}",
            details={"reason": "LangGraph package not found"},
        )

    # If version can't be determined, assume compatible (fail-open)
    if version == "unknown":
        logger.warning(
            "langgraph_version_unknown",
            message="Could not determine LangGraph version, assuming compatible",
        )
        return version

    if not _is_version_supported(version):
        supported_range = f">={MIN_LANGGRAPH_VERSION}"
        if MAX_LANGGRAPH_VERSION:
            supported_range += f", <={MAX_LANGGRAPH_VERSION}"

        raise FrameworkVersionError(
            framework="langgraph",
            version=version,
            supported=supported_range,
        )

    return version


class LangGraphCallbackHandler:
    """
    Callback handler for LangGraph execution events.

    This handler integrates with LangGraph's RunnableConfig callback system
    to capture detailed trace events during graph execution.

    The handler maintains internal state to track:
    - Current node being executed
    - State snapshots at each transition
    - Conditional edge decisions
    """

    # Required callback handler attributes expected by LangChain/LangGraph's
    # callback dispatcher. These mirror BaseCallbackHandler defaults without
    # requiring langchain_core as a dependency.
    raise_error: bool = True
    run_inline: bool = False
    ignore_llm: bool = False
    ignore_retry: bool = False
    ignore_chain: bool = False
    ignore_agent: bool = False
    ignore_chat_model: bool = False
    ignore_custom_event: bool = False

    def __init__(
        self,
        session: Session,
        adapter: LangGraphAdapter,
        config: ClyroConfig,
    ):
        """
        Initialize the callback handler.

        Args:
            session: Current trace session
            adapter: Parent LangGraphAdapter instance
            config: SDK configuration
        """
        self._session = session
        self._adapter = adapter
        self._config = config

        # Step tracking for real-time step limit enforcement.
        # Like CrewAI's handler, we track steps locally so that limits are
        # enforced DURING execution (in callbacks), not just post-execution.
        # Since LangGraph callbacks have raise_error=True, raising here
        # properly stops the agent mid-flight.
        # Start at step_number + 1 so first handler event gets step 1 (not 0).
        # step 0 is reserved for SESSION_START. If a handler event gets step 0,
        # session.record_event() reassigns it to session._step_number + 1,
        # which breaks chronological ordering when events are drained after
        # execution (by that time session._step_number has advanced).
        self._step_counter: int = session.step_number + 1

        # Local cost tracking for real-time policy enforcement.
        # session.cumulative_cost is only updated during post-execution
        # event draining, so we track cost locally (like CrewAI's handler)
        # to provide accurate cost values in policy checks.
        self._local_cumulative_cost: Decimal = session.cumulative_cost

        # Execution tracking
        self._node_start_times: dict[str, float] = {}
        self._node_inputs: dict[str, dict[str, Any]] = {}
        self._node_names: dict[str, str] = {}  # Map run_id to node name
        self._current_state: dict[str, Any] = {}
        self._events: list[TraceEvent] = []

        # LLM call tracking (keyed by str(run_id))
        self._llm_start_times: dict[str, float] = {}
        self._llm_inputs: dict[str, dict[str, Any]] = {}
        self._llm_models: dict[str, str] = {}

        # Tool call tracking (keyed by str(run_id))
        self._tool_start_times: dict[str, float] = {}
        self._tool_inputs: dict[str, dict[str, Any]] = {}
        self._tool_names: dict[str, str] = {}

        # Retriever call tracking (keyed by str(run_id))
        self._retriever_start_times: dict[str, float] = {}
        self._retriever_queries: dict[str, str] = {}
        self._retriever_names: dict[str, str] = {}

        # Implements FRD-002: Parent-child event hierarchy wiring
        self._run_id_to_event_id: dict[str, UUID] = {}  # run_id → event_id for parent resolution
        self._node_event_ids: dict[
            str, UUID
        ] = {}  # run_id → pre-generated event_id (early registration)
        self._node_step_numbers: dict[str, int] = {}  # run_id → pre-assigned step number
        self._RUN_ID_MAP_MAX_SIZE = 10_000

    @property
    def session(self) -> Session:
        """Get the current session."""
        return self._session

    @staticmethod
    def _parse_tool_input(input_str: Any) -> Any:
        """
        Parse tool input string into a dict.

        LangChain may pass tool input as JSON (double quotes) or as
        Python repr (single quotes, True/False/None). Handle both.
        """
        if not isinstance(input_str, str):
            return input_str
        # Try JSON first (most common for structured tool inputs)
        try:
            import json

            return json.loads(input_str)
        except (json.JSONDecodeError, ValueError):
            pass
        # Fall back to Python literal eval (handles single quotes, True/False/None)
        try:
            import ast

            result = ast.literal_eval(input_str)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass
        return input_str

    def _next_step(self) -> int:
        """Get the next step number, increment the counter, and enforce step limit.

        Provides real-time step limit enforcement during LangGraph execution.
        Since LangGraph callbacks have raise_error=True, raising here
        properly stops the agent mid-flight (unlike CrewAI which swallows).

        Also syncs the session's step counter so that post-execution
        event draining doesn't double-count steps.

        Raises:
            StepLimitExceededError: If step limit is exceeded
        """
        step = self._step_counter
        self._step_counter += 1
        # Sync session step counter for real-time enforcement.
        # Use `step` (pre-increment) not `self._step_counter` (post-increment)
        # to avoid pushing session._step_number one ahead of the actual step,
        # which would cause premature step-limit errors.
        if step > self._session._step_number:
            self._session._step_number = step
        self._session._check_step_limit()
        return step

    def _check_loop(self, state: dict[str, Any] | None, action: str | None) -> None:
        """Check for infinite loops via session's LoopDetector.  # Implements PRD-010

        Called after each node completes with the updated state snapshot and
        the node name as the action identifier.

        If a loop is detected, an error event is recorded before the
        LoopDetectedError propagates, ensuring the event reaches the backend.

        Args:
            state: Current state snapshot after node execution
            action: Node name that just executed

        Raises:
            LoopDetectedError: If a loop is detected
        """
        try:
            self._session._check_loop_detection(state or {}, action)
        except Exception as e:
            # Record error event before propagating so it's always captured.
            error_event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._session.step_number,
                error_type=type(e).__name__,
                error_message=str(e),
                agent_id=self._session.agent_id,
                cumulative_cost=Decimal("0"),
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                metadata={
                    "event_name": "loop_detection",
                    "node_name": action,
                },
            )
            if error_event is not None:
                self._events.append(error_event)
            raise

    # -- Implements FRD-002: parent-child hierarchy helpers --

    def _resolve_parent_event_id(self, parent_run_id: UUID | None) -> UUID | None:
        """Lookup parent_run_id in the map. Returns None if not found."""
        if parent_run_id is None:
            return None
        prid = str(parent_run_id)
        parent_eid = self._run_id_to_event_id.get(prid)
        if parent_eid is None:
            logger.debug("parent_run_id not found in event map", parent_run_id=prid)
        return parent_eid

    def _register_event(self, run_id: UUID | None, event_id: UUID) -> None:
        """Register run_id → event_id mapping. Handles eviction if map exceeds limit."""
        if run_id is None:
            return
        rid = str(run_id)
        if len(self._run_id_to_event_id) >= self._RUN_ID_MAP_MAX_SIZE:
            oldest_key = next(iter(self._run_id_to_event_id))
            del self._run_id_to_event_id[oldest_key]
            logger.warning("run_id_to_event_id map eviction", evicted_key=oldest_key)
        self._run_id_to_event_id[rid] = event_id

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a chain (graph/node) starts execution.

        In LangGraph, each node execution triggers this callback with a parent_run_id.
        We detect node executions and track them for creating STATE_TRANSITION events.
        """
        try:
            rid = str(run_id) if run_id else str(id(inputs))

            # Try to extract node name from various sources
            node_name = None

            # Check metadata for langgraph-specific node info
            if metadata:
                node_name = metadata.get("langgraph_node") or metadata.get("run_name")

            # Check serialized data for component name
            if not node_name and serialized:
                node_name = serialized.get("name") or serialized.get("id")

            # Check tags for node identifiers
            if not node_name and tags:
                for tag in tags:
                    if tag.startswith("node:"):
                        node_name = tag[5:]  # Remove "node:" prefix
                        break

            # If this looks like a node execution (has parent and node name), track it.
            # Skip nested chains within an already-tracked node — LangGraph 1.0+
            # fires multiple on_chain_start callbacks for inner RunnableSequence /
            # callable layers that share the same metadata.langgraph_node.
            # Tracking them would inflate step counts and trigger false loop
            # detection because the same state hash gets counted N times.
            parent_rid = str(parent_run_id) if parent_run_id else ""
            is_nested = parent_rid in self._node_start_times
            if parent_run_id and node_name and not is_nested:
                start_time = time.perf_counter()
                self._node_start_times[rid] = start_time
                if self._config.capture_inputs:
                    self._node_inputs[rid] = self._serialize_data(inputs)

                # Store node name for later use in on_chain_end
                self._node_names[rid] = node_name

                # Implements FRD-002: Early registration — pre-generate event_id and
                # register in map immediately so that child events (LLM/tool) firing
                # before on_chain_end can resolve their parent.
                pre_generated_event_id = uuid4()
                self._register_event(run_id, pre_generated_event_id)
                self._node_event_ids[rid] = pre_generated_event_id

                # Pre-assign step number so parent node always has a lower step
                # number than its children (LLM/tool events created in *_end
                # callbacks fire before on_chain_end).
                self._node_step_numbers[rid] = self._next_step()

                logger.debug(
                    "langgraph_node_start",
                    session_id=str(self._session.session_id),
                    node_name=node_name,
                    run_id=rid,
                )
            else:
                # This is the top-level graph execution, store initial state
                self._current_state = dict(inputs) if inputs else {}

                logger.debug(
                    "langgraph_chain_start",
                    session_id=str(self._session.session_id),
                    run_id=rid,
                    input_keys=list(inputs.keys()) if inputs else [],
                )
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_chain_start_failed", error=str(e), fail_open=True)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a chain (graph/node) completes execution.

        Creates STATE_TRANSITION events for node executions.
        """
        try:
            rid = str(run_id) if run_id else ""

            # Check if this was a tracked node execution
            if rid in self._node_start_times:
                # This is a node execution - create STATE_TRANSITION event
                node_name = self._node_names.pop(rid, "unknown_node")
                start_time = self._node_start_times.pop(rid, time.perf_counter())
                duration_ms = int((time.perf_counter() - start_time) * 1000)

                # Update current state with outputs (only when outputs is a dict)
                if outputs and isinstance(outputs, dict):
                    self._current_state.update(outputs)

                # Determine agent stage based on node type
                agent_stage = self._determine_agent_stage(
                    node_name, outputs if isinstance(outputs, dict) else None
                )

                # Capture input and output data
                input_data = self._node_inputs.pop(rid, None)
                output_data = (
                    self._serialize_data(outputs) if self._config.capture_outputs else None
                )
                state_snapshot = (
                    self._serialize_data(self._current_state)
                    if self._config.capture_state
                    else None
                )

                # Use pre-assigned step number from on_chain_start so parent
                # node has a lower step number than its children.
                if rid in self._node_step_numbers:
                    step_number = self._node_step_numbers.pop(rid)
                else:
                    step_number = self._next_step()

                # Implements FRD-002: Retrieve pre-generated event_id and resolve parent
                pre_generated_eid = self._node_event_ids.pop(rid, None)
                parent_eid = self._resolve_parent_event_id(parent_run_id)

                # Create state transition event
                event = create_state_transition_event(
                    session_id=self._session.session_id,
                    step_number=step_number,
                    node_name=node_name,
                    agent_id=self._session.agent_id,
                    input_data=input_data,
                    output_data=output_data,
                    state_snapshot=state_snapshot,
                    duration_ms=duration_ms,
                    cumulative_cost=self._local_cumulative_cost,
                    framework=Framework.LANGGRAPH,
                    framework_version=self._adapter.framework_version,
                    agent_stage=agent_stage,
                    metadata={},
                    event_id=pre_generated_eid,
                    parent_event_id=parent_eid,
                )

                self._events.append(event)

                # Implements PRD-010: Check for infinite loops after each node execution
                self._check_loop(self._current_state, node_name)

                logger.debug(
                    "langgraph_node_end",
                    session_id=str(self._session.session_id),
                    node_name=node_name,
                    duration_ms=duration_ms,
                    agent_stage=agent_stage.value,
                )
            else:
                # This is the top-level graph completion, just update state
                output_keys = list(outputs.keys()) if isinstance(outputs, dict) else []
                logger.debug(
                    "langgraph_chain_end",
                    session_id=str(self._session.session_id),
                    run_id=rid,
                    output_keys=output_keys,
                )

                # Update final state (only when outputs is a dict)
                if outputs and isinstance(outputs, dict):
                    self._current_state.update(outputs)
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_chain_end_failed", error=str(e), fail_open=True)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when a chain (graph/node) encounters an error.

        Returns:
            TraceEvent capturing the error
        """
        try:
            import traceback

            rid = str(run_id) if run_id else ""

            # Check if this was a tracked node execution that failed
            if rid in self._node_start_times:
                # This is a node error - create node-specific error event
                node_name = self._node_names.pop(rid, "unknown_node")
                self._node_start_times.pop(rid, None)
                node_input_data = self._node_inputs.pop(rid, None)

                # Use pre-assigned step number from on_chain_start
                if rid in self._node_step_numbers:
                    step_number = self._node_step_numbers.pop(rid)
                else:
                    step_number = self._session.step_number

                # Implements FRD-002: Retrieve pre-generated event_id and resolve parent
                pre_generated_eid = self._node_event_ids.pop(rid, None)
                parent_eid = self._resolve_parent_event_id(parent_run_id)

                event = create_error_event(
                    session_id=self._session.session_id,
                    step_number=step_number,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    agent_id=self._session.agent_id,
                    error_stack=traceback.format_exc(),
                    cumulative_cost=self._local_cumulative_cost,
                    framework=Framework.LANGGRAPH,
                    framework_version=self._adapter.framework_version,
                    input_data=node_input_data,
                    output_data={
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    metadata={
                        "node_name": node_name,
                    },
                    event_id=pre_generated_eid,
                    parent_event_id=parent_eid,
                )

                logger.debug(
                    "langgraph_node_error",
                    session_id=str(self._session.session_id),
                    node_name=node_name,
                    error_type=type(error).__name__,
                )
            else:
                # This is a top-level graph error
                logger.debug(
                    "langgraph_chain_error",
                    session_id=str(self._session.session_id),
                    error_type=type(error).__name__,
                )

                event = self._adapter.on_error(
                    self._session,
                    error if isinstance(error, Exception) else Exception(str(error)),
                    {"step_number": self._session.step_number},
                )

            self._events.append(event)
            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_chain_error_failed", error=str(e), fail_open=True)
            return None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when an LLM call starts.

        Records the start time, model info, and input prompts for
        pairing with on_llm_end to create a complete LLM_CALL event.

        Args:
            serialized: Serialized LLM info (contains model name, etc.)
            prompts: Input prompts sent to the LLM
            run_id: Unique run identifier for correlating start/end
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
        """
        try:
            rid = str(run_id) if run_id else str(id(prompts))
            self._llm_start_times[rid] = time.perf_counter()

            # Extract model name from serialized info
            model = (
                serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or serialized.get("name", "unknown")
            )
            self._llm_models[rid] = model

            if self._config.capture_inputs:
                self._llm_inputs[rid] = {"prompts": prompts}

            # Pre-LLM policy check — include prompt text for text-based rules,
            # cost for cost-based rules, and step_number for step-based rules.
            # Uses _local_cumulative_cost (real-time) instead of session.cumulative_cost (stale).
            llm_policy_params: dict[str, Any] = {
                "model": model,
                "cost": float(self._local_cumulative_cost),
                "step_number": self._step_counter,
            }
            if prompts:
                llm_policy_params["input"] = prompts[0]
            self._session.check_policy(
                "llm_call",
                llm_policy_params,
                parent_event_id=self._resolve_parent_event_id(parent_run_id),
                cumulative_cost=self._local_cumulative_cost,
            )

            logger.debug(
                "langgraph_llm_start",
                session_id=str(self._session.session_id),
                model=model,
                run_id=rid,
            )
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_llm_start_failed", error=str(e), fail_open=True)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a chat model (e.g. ChatOpenAI) call starts.

        LangChain dispatches chat models to on_chat_model_start instead of
        on_llm_start. This delegates to the same tracking logic.

        Args:
            serialized: Serialized model info (contains model name, etc.)
            messages: Input messages sent to the chat model
            run_id: Unique run identifier for correlating start/end
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
        """
        try:
            rid = str(run_id) if run_id else str(id(messages))
            self._llm_start_times[rid] = time.perf_counter()

            # Extract model name from serialized info
            model = (
                serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or serialized.get("name", "unknown")
            )
            self._llm_models[rid] = model

            if self._config.capture_inputs:
                # Convert messages to serializable format
                msg_texts = []
                for msg_list in messages:
                    for msg in msg_list if isinstance(msg_list, list) else [msg_list]:
                        content = getattr(msg, "content", str(msg))
                        msg_type = getattr(msg, "type", type(msg).__name__)
                        msg_texts.append({"type": msg_type, "content": content})
                self._llm_inputs[rid] = {"messages": msg_texts}

            # Pre-LLM policy check — include last user message for text-based rules,
            # cost for cost-based rules, and step_number for step-based rules.
            # Uses _local_cumulative_cost (real-time) instead of session.cumulative_cost (stale).
            llm_policy_params: dict[str, Any] = {
                "model": model,
                "cost": float(self._local_cumulative_cost),
                "step_number": self._step_counter,
            }
            try:
                for msg_list in messages:
                    for msg in reversed(msg_list if isinstance(msg_list, list) else [msg_list]):
                        if getattr(msg, "type", "") == "human":
                            llm_policy_params["input"] = getattr(msg, "content", "")
                            break
                    if "input" in llm_policy_params:
                        break
            except Exception:
                pass  # Best-effort — don't break on message parsing
            self._session.check_policy(
                "llm_call",
                llm_policy_params,
                parent_event_id=self._resolve_parent_event_id(parent_run_id),
                cumulative_cost=self._local_cumulative_cost,
            )

            logger.debug(
                "langgraph_chat_model_start",
                session_id=str(self._session.session_id),
                model=model,
                run_id=rid,
            )
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_chat_model_start_failed", error=str(e), fail_open=True)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when an LLM call completes.

        Creates an LLM_CALL trace event with timing, token usage,
        and cost information extracted from the response.

        Args:
            response: LLM response object (typically LLMResult)
            run_id: Unique run identifier for correlating with on_llm_start
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent for the LLM call
        """
        try:
            rid = str(run_id) if run_id else ""
            start_time = self._llm_start_times.pop(rid, time.perf_counter())
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            model = self._llm_models.pop(rid, "unknown")
            input_data = self._llm_inputs.pop(rid, {})

            # Extract token usage from response
            token_input = 0
            token_output = 0
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}
            if token_usage:
                token_input = token_usage.get("prompt_tokens", 0)
                token_output = token_usage.get("completion_tokens", 0)

            # Extract model name from response if available
            if isinstance(llm_output, dict) and llm_output.get("model_name"):
                model = llm_output["model_name"]

            # Extract output text
            output_data = None
            if self._config.capture_outputs:
                generations = getattr(response, "generations", None)
                if generations and len(generations) > 0:
                    gen_texts = []
                    for gen_list in generations:
                        for gen in gen_list if isinstance(gen_list, list) else [gen_list]:
                            text = getattr(gen, "text", None) or str(gen)
                            gen_texts.append(text)
                    output_data = {"generations": gen_texts}
                else:
                    output_data = {"response": str(response)}

            # Calculate cost from token counts
            cost_usd = Decimal("0")
            if token_input > 0 or token_output > 0:
                calculator = CostCalculator(self._config)
                cost_usd = calculator.calculate(token_input, token_output, model)

            # Track cost locally for real-time policy enforcement.
            # session.cumulative_cost is stale during execution (updated only
            # when events are drained). Policy checks use _local_cumulative_cost.
            if cost_usd > 0:
                self._local_cumulative_cost += cost_usd

            # Implements FRD-013: Post-completion policy check with `output` field.
            # Enables response-content policies (PII, grounding, attribution).
            output_text = ""
            if output_data:
                if isinstance(output_data, dict):
                    gens = output_data.get("generations", [])
                    output_text = " ".join(gens) if gens else str(output_data)
                else:
                    output_text = str(output_data)
            self._session.check_policy(
                "llm_call",
                {
                    "model": model,
                    "cost": float(self._local_cumulative_cost),
                    "step_number": self._step_counter,
                    "input": str(input_data) if input_data else "",
                    "output": output_text,
                },
                parent_event_id=self._resolve_parent_event_id(parent_run_id),
                cumulative_cost=self._local_cumulative_cost,
            )

            # Enforce step limit and get step number
            step_number = self._next_step()

            # Implements FRD-002: Resolve parent and register this event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_llm_call_event(
                session_id=self._session.session_id,
                step_number=step_number,
                model=model,
                input_data=input_data,
                output_data=output_data,
                agent_id=self._session.agent_id,
                token_count_input=token_input,
                token_count_output=token_output,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                metadata={
                    "run_id": rid,
                },
                parent_event_id=parent_eid,
            )

            self._register_event(run_id, event.event_id)
            self._events.append(event)

            logger.debug(
                "langgraph_llm_end",
                session_id=str(self._session.session_id),
                model=model,
                duration_ms=duration_ms,
                input_tokens=token_input,
                output_tokens=token_output,
                cost_usd=str(cost_usd),
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_llm_end_failed", error=str(e), fail_open=True)
            return None

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when an LLM call fails.

        Args:
            error: The exception that occurred
            run_id: Unique run identifier
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent capturing the LLM error
        """
        try:
            import traceback as tb

            rid = str(run_id) if run_id else ""
            model = self._llm_models.pop(rid, "unknown")
            self._llm_start_times.pop(rid, None)
            llm_input_data = self._llm_inputs.pop(rid, None)

            # Implements FRD-002: Resolve parent for error event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._session.agent_id,
                error_stack=tb.format_exc(),
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                input_data=llm_input_data,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={
                    "event_source": "llm_call",
                    "model": model,
                    "run_id": rid,
                },
                parent_event_id=parent_eid,
            )

            self._events.append(event)

            logger.debug(
                "langgraph_llm_error",
                session_id=str(self._session.session_id),
                model=model,
                error_type=type(error).__name__,
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_llm_error_failed", error=str(e), fail_open=True)
            return None

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a tool execution starts.

        Records the start time, tool name, and input for pairing
        with on_tool_end to create a complete TOOL_CALL event.

        Args:
            serialized: Serialized tool info (contains tool name, etc.)
            input_str: Input string passed to the tool
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
        """
        try:
            rid = str(run_id) if run_id else str(id(input_str))
            self._tool_start_times[rid] = time.perf_counter()

            tool_name = serialized.get("name", "unknown_tool")
            self._tool_names[rid] = tool_name

            if self._config.capture_inputs:
                self._tool_inputs[rid] = {"input": input_str}

            # Pre-tool policy check — include tool input for rule matching.
            # Flatten one level of nested dicts so fields like "rmq_cluster"
            # are accessible directly, even if the tool wraps args in a
            # container like {"state": {"rmq_cluster": "cluster2"}}.
            # Include cost for cost-based rules and step_number for step-based rules.
            # Uses _local_cumulative_cost (real-time) instead of session.cumulative_cost (stale).
            policy_params: dict[str, Any] = {
                "tool_name": tool_name,
                "cost": float(self._local_cumulative_cost),
                "step_number": self._step_counter,
            }
            tool_input = self._parse_tool_input(input_str)
            if isinstance(tool_input, dict):
                policy_params.update(tool_input)
                # Flatten: promote nested dict values to top level
                for v in tool_input.values():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            policy_params.setdefault(k2, v2)
            else:
                policy_params["input"] = str(input_str)

            logger.debug(
                "policy_tool_params",
                tool_name=tool_name,
                input_str_type=type(input_str).__name__,
                parsed_type=type(tool_input).__name__,
                policy_params_keys=list(policy_params.keys()),
                policy_params=policy_params,
            )
            self._session.check_policy(
                "tool_call",
                policy_params,
                parent_event_id=self._resolve_parent_event_id(parent_run_id),
                cumulative_cost=self._local_cumulative_cost,
            )

            logger.debug(
                "langgraph_tool_start",
                session_id=str(self._session.session_id),
                tool_name=tool_name,
                run_id=rid,
            )
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_tool_start_failed", error=str(e), fail_open=True)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when a tool execution completes.

        Creates a TOOL_CALL trace event with timing and output data.

        Args:
            output: Output from the tool execution
            run_id: Unique run identifier
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent for the tool call
        """
        try:
            rid = str(run_id) if run_id else ""
            start_time = self._tool_start_times.pop(rid, time.perf_counter())
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            tool_name = self._tool_names.pop(rid, "unknown_tool")
            input_data = self._tool_inputs.pop(rid, {})

            output_data = None
            if self._config.capture_outputs:
                output_data = {"result": self._serialize_value(output)}

            # Enforce step limit and get step number
            step_number = self._next_step()

            # Implements FRD-002: Resolve parent and register this event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_tool_call_event(
                session_id=self._session.session_id,
                step_number=step_number,
                tool_name=tool_name,
                input_data=input_data,
                output_data=output_data,
                agent_id=self._session.agent_id,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                metadata={
                    "run_id": rid,
                },
                parent_event_id=parent_eid,
            )

            self._register_event(run_id, event.event_id)
            self._events.append(event)

            logger.debug(
                "langgraph_tool_end",
                session_id=str(self._session.session_id),
                tool_name=tool_name,
                duration_ms=duration_ms,
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_tool_end_failed", error=str(e), fail_open=True)
            return None

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when a tool execution fails.

        Args:
            error: The exception that occurred
            run_id: Unique run identifier
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent capturing the tool error
        """
        try:
            import traceback as tb

            rid = str(run_id) if run_id else ""
            tool_name = self._tool_names.pop(rid, "unknown_tool")
            self._tool_start_times.pop(rid, None)
            tool_input_data = self._tool_inputs.pop(rid, None)

            # Implements FRD-002: Resolve parent for error event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._session.agent_id,
                error_stack=tb.format_exc(),
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                input_data=tool_input_data,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={
                    "event_source": "tool_call",
                    "tool_name": tool_name,
                    "run_id": rid,
                },
                parent_event_id=parent_eid,
            )

            self._events.append(event)

            logger.debug(
                "langgraph_tool_error",
                session_id=str(self._session.session_id),
                tool_name=tool_name,
                error_type=type(error).__name__,
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_tool_error_failed", error=str(e), fail_open=True)
            return None

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called when a retriever starts execution.

        Records the start time, retriever name, and query for pairing
        with on_retriever_end to create a complete RETRIEVER_CALL event.

        Args:
            serialized: Serialized retriever info (contains retriever name, etc.)
            query: Search query string
            run_id: Unique run identifier
            parent_run_id: Parent run identifier
            tags: Optional tags
            metadata: Optional metadata
        """
        try:
            rid = str(run_id) if run_id else str(id(query))
            self._retriever_start_times[rid] = time.perf_counter()

            retriever_name = serialized.get("name", "unknown_retriever")
            self._retriever_names[rid] = retriever_name
            self._retriever_queries[rid] = query

            logger.debug(
                "langgraph_retriever_start",
                session_id=str(self._session.session_id),
                retriever_name=retriever_name,
                query=query[:100],  # Truncate long queries in logs
                run_id=rid,
            )
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_retriever_start_failed", error=str(e), fail_open=True)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when a retriever completes execution.

        Creates a RETRIEVER_CALL trace event with timing and retrieved documents.

        Args:
            documents: Retrieved documents (typically a list of Document objects)
            run_id: Unique run identifier
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent for the retriever call
        """
        try:
            rid = str(run_id) if run_id else ""
            start_time = self._retriever_start_times.pop(rid, time.perf_counter())
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            retriever_name = self._retriever_names.pop(rid, "unknown_retriever")
            query = self._retriever_queries.pop(rid, "")

            # Serialize retrieved documents
            docs_data = None
            if documents:
                docs_list = []
                doc_list = documents if isinstance(documents, list) else [documents]
                for doc in doc_list:
                    # Handle LangChain Document objects
                    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                        docs_list.append(
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                        )
                    elif isinstance(doc, dict):
                        docs_list.append(doc)
                    else:
                        docs_list.append({"content": str(doc)})
                docs_data = docs_list

            # Enforce step limit and get step number
            step_number = self._next_step()

            # Implements FRD-002: Resolve parent and register this event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_retriever_call_event(
                session_id=self._session.session_id,
                step_number=step_number,
                retriever_name=retriever_name,
                query=query,
                documents=docs_data,
                agent_id=self._session.agent_id,
                duration_ms=duration_ms,
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                metadata={
                    "run_id": rid,
                    "document_count": len(docs_data) if docs_data else 0,
                },
                parent_event_id=parent_eid,
            )

            self._register_event(run_id, event.event_id)
            self._events.append(event)

            logger.debug(
                "langgraph_retriever_end",
                session_id=str(self._session.session_id),
                retriever_name=retriever_name,
                duration_ms=duration_ms,
                document_count=len(docs_data) if docs_data else 0,
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_retriever_end_failed", error=str(e), fail_open=True)
            return None

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> TraceEvent:
        """
        Called when a retriever execution fails.

        Args:
            error: The exception that occurred
            run_id: Unique run identifier
            parent_run_id: Parent run identifier

        Returns:
            TraceEvent capturing the retriever error
        """
        try:
            import traceback as tb

            rid = str(run_id) if run_id else ""
            retriever_name = self._retriever_names.pop(rid, "unknown_retriever")
            query = self._retriever_queries.pop(rid, "")
            self._retriever_start_times.pop(rid, None)

            # Implements FRD-002: Resolve parent for error event
            parent_eid = self._resolve_parent_event_id(parent_run_id)

            event = create_error_event(
                session_id=self._session.session_id,
                step_number=self._session.step_number,
                error_type=type(error).__name__,
                error_message=str(error),
                agent_id=self._session.agent_id,
                error_stack=tb.format_exc(),
                cumulative_cost=self._local_cumulative_cost,
                framework=Framework.LANGGRAPH,
                framework_version=self._adapter.framework_version,
                input_data={"query": query} if query else None,
                output_data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                metadata={
                    "event_source": "retriever_call",
                    "retriever_name": retriever_name,
                    "query": query,
                    "run_id": rid,
                },
                parent_event_id=parent_eid,
            )

            self._events.append(event)

            logger.debug(
                "langgraph_retriever_error",
                session_id=str(self._session.session_id),
                retriever_name=retriever_name,
                error_type=type(error).__name__,
            )

            return event
        except (PolicyViolationError, ExecutionControlError):
            raise
        except Exception as e:
            logger.error("on_retriever_error_failed", error=str(e), fail_open=True)
            return None

    def get_current_state(self) -> dict[str, Any]:
        """Get the current state snapshot."""
        return dict(self._current_state)

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear recorded trace events."""
        events = list(self._events)
        self._events.clear()
        return events

    def _determine_agent_stage(
        self,
        node_name: str,
        outputs: dict[str, Any] | None,
    ) -> AgentStage:
        """
        Determine the agent stage based on node name and outputs.

        Heuristics:
        - Nodes with "tool" in name or tool_calls in output -> ACT
        - Nodes with "agent" or LLM-related names -> THINK
        - Observation/result nodes -> OBSERVE

        Args:
            node_name: Name of the node
            outputs: Output data from the node

        Returns:
            AgentStage for this node execution
        """
        node_lower = node_name.lower()

        # Check for tool execution indicators
        if "tool" in node_lower or "action" in node_lower:
            return AgentStage.ACT

        if outputs and isinstance(outputs, dict):
            # Check for tool calls in output
            if "tool_calls" in outputs or "actions" in outputs:
                return AgentStage.ACT

            # Check for observations
            if "observation" in outputs or "result" in outputs:
                return AgentStage.OBSERVE

        # Check for observation/result nodes
        if "observe" in node_lower or "result" in node_lower or "output" in node_lower:
            return AgentStage.OBSERVE

        # Default to THINK for agent/LLM nodes
        return AgentStage.THINK

    def _serialize_data(self, data: Any) -> dict[str, Any] | None:
        """
        Serialize data for storage.

        Args:
            data: Data to serialize

        Returns:
            Serialized dictionary or None
        """
        if data is None:
            return None

        if isinstance(data, dict):
            return {k: self._serialize_value(v) for k, v in data.items()}

        return {"value": self._serialize_value(data)}

    def _serialize_value(self, value: Any, depth: int = 0) -> Any:
        """
        Recursively serialize a value for JSON storage.

        Args:
            value: Value to serialize
            depth: Current recursion depth

        Returns:
            JSON-serializable value
        """
        MAX_DEPTH = 50
        if depth > MAX_DEPTH:
            return f"<max_depth_exceeded: {type(value).__name__}>"

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v, depth + 1) for v in value]

        if isinstance(value, dict):
            return {str(k): self._serialize_value(v, depth + 1) for k, v in value.items()}

        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump()
            except Exception:
                return str(value)

        if hasattr(value, "__dict__"):
            try:
                return {
                    k: self._serialize_value(v, depth + 1)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception:
                return str(value)

        return str(value)


class LangGraphAdapter:
    """
    LangGraph-specific adapter for StateGraph tracing.

    Implements PRD-003: LangGraph Framework Adapter

    This adapter wraps LangGraph StateGraph and CompiledGraph instances
    to capture detailed trace events during execution.

    Features:
    - Node execution tracing with timing
    - State transition recording with snapshots
    - Conditional edge decision capture
    - Automatic callback injection into RunnableConfig

    Example:
        ```python
        from langgraph.graph import StateGraph
        import clyro

        # Create your LangGraph
        graph = StateGraph(...)
        compiled = graph.compile()

        # Wrap with Clyro
        wrapped = clyro.wrap(compiled, adapter="langgraph")
        result = wrapped.invoke({"input": "hello"})
        ```
    """

    FRAMEWORK = Framework.LANGGRAPH
    MIN_VERSION = MIN_LANGGRAPH_VERSION

    def __init__(
        self,
        agent: Any,
        config: ClyroConfig,
        validate_version: bool = True,
    ):
        """
        Initialize LangGraph adapter.

        Args:
            agent: LangGraph StateGraph or CompiledGraph instance
            config: SDK configuration
            validate_version: Whether to validate LangGraph version

        Raises:
            FrameworkVersionError: If LangGraph version is unsupported
        """
        self._agent = agent
        self._config = config
        self._name = self._extract_name(agent)

        # Validate version if requested
        if validate_version:
            self._framework_version = validate_langgraph_version()
        else:
            self._framework_version = detect_langgraph_version() or "unknown"

        # Current callback handler (created per-session)
        self._current_handler: LangGraphCallbackHandler | None = None

        logger.debug(
            "langgraph_adapter_init",
            agent_name=self._name,
            framework_version=self._framework_version,
        )

    def _extract_name(self, agent: Any) -> str:
        """
        Extract a human-readable name from the agent.

        Args:
            agent: LangGraph agent instance

        Returns:
            Name string
        """
        # Try common LangGraph name attributes
        if hasattr(agent, "name") and agent.name:
            return str(agent.name)

        if hasattr(agent, "graph") and hasattr(agent.graph, "name"):
            return str(agent.graph.name)

        # Try to get from class name
        class_name = type(agent).__name__
        if class_name not in ("StateGraph", "CompiledGraph", "Runnable"):
            return class_name

        return "langgraph_agent"

    @property
    def agent(self) -> Any:
        """Get the wrapped LangGraph agent."""
        return self._agent

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._name

    @property
    def framework(self) -> Framework:
        """Get the framework type."""
        return self.FRAMEWORK

    @property
    def framework_version(self) -> str:
        """Get the detected LangGraph version."""
        return self._framework_version

    def create_callback_handler(
        self,
        session: Session,
    ) -> LangGraphCallbackHandler:
        """
        Create a callback handler for a session.

        Args:
            session: Current trace session

        Returns:
            LangGraphCallbackHandler configured for the session
        """
        handler = LangGraphCallbackHandler(
            session=session,
            adapter=self,
            config=self._config,
        )
        self._current_handler = handler
        return handler

    def _accepts_config_param(self) -> bool:
        """Check if the wrapped agent accepts a 'config' keyword argument."""
        import inspect

        # LangGraph CompiledGraph.invoke/ainvoke always accept config
        if hasattr(self._agent, "invoke"):
            try:
                sig = inspect.signature(self._agent.invoke)
                return "config" in sig.parameters
            except (ValueError, TypeError):
                pass

        # For plain callables, check the callable itself
        try:
            sig = inspect.signature(self._agent)
            params = sig.parameters
            # Accept if there's an explicit 'config' param or **kwargs
            return "config" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (ValueError, TypeError):
            return False

    def inject_callbacks(
        self,
        config: dict[str, Any] | None,
        handler: LangGraphCallbackHandler,
    ) -> dict[str, Any]:
        """
        Inject Clyro callbacks into a RunnableConfig.

        This method adds our callback handler to an existing config
        or creates a new one if none exists.

        Args:
            config: Existing RunnableConfig dict or None
            handler: Callback handler to inject

        Returns:
            Updated config with callbacks injected
        """
        if config is None:
            config = {}
        else:
            config = dict(config)  # Don't mutate original

        # Get existing callbacks
        existing_callbacks = config.get("callbacks", [])
        if existing_callbacks is None:
            existing_callbacks = []

        # Add our handler
        config["callbacks"] = list(existing_callbacks) + [handler]

        return config

    def before_call(
        self,
        session: Session,
        args: tuple,
        kwargs: dict,
    ) -> dict[str, Any]:
        """
        Hook called before the agent is invoked.

        Sets up the callback handler and prepares for tracing.

        Args:
            session: Current session
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Context dictionary to pass to after_call
        """
        handler = self.create_callback_handler(session)

        # Inject callbacks into kwargs if RunnableConfig pattern is used
        if "config" in kwargs:
            kwargs["config"] = self.inject_callbacks(kwargs["config"], handler)
        elif self._accepts_config_param():
            kwargs["config"] = self.inject_callbacks(None, handler)

        context = {
            "start_time": time.perf_counter(),
            "step_number": 0,
            "handler": handler,
        }

        logger.debug(
            "langgraph_before_call",
            agent=self._name,
            session_id=str(session.session_id),
            step=context["step_number"],
        )

        return context

    def after_call(
        self,
        session: Session,
        result: Any,
        context: dict[str, Any],
    ) -> TraceEvent:
        """
        Hook called after successful agent execution.

        Args:
            session: Current session
            result: Return value from the agent
            context: Context from before_call

        Returns:
            TraceEvent for this execution
        """
        duration_ms = int((time.perf_counter() - context["start_time"]) * 1000)
        handler: LangGraphCallbackHandler | None = context.get("handler")

        # Gather data from handler
        output_data = None
        state_snapshot = None

        if handler:
            if self._config.capture_outputs:
                output_data = handler._serialize_data(result)
            if self._config.capture_state:
                state_snapshot = handler._serialize_data(handler.get_current_state())

        # Create completion event
        event = create_step_event(
            session_id=session.session_id,
            step_number=0,
            event_name=f"{self._name}_complete",
            agent_id=session.agent_id,
            output_data=output_data,
            state_snapshot=state_snapshot,
            duration_ms=duration_ms,
            cumulative_cost=session.cumulative_cost,
            framework=Framework.LANGGRAPH,
            framework_version=self._framework_version,
            metadata={},
            agent_stage=AgentStage.OBSERVE,
        )

        logger.debug(
            "langgraph_after_call",
            agent=self._name,
            session_id=str(session.session_id),
            duration_ms=duration_ms,
        )

        return event

    def on_error(
        self,
        session: Session,
        error: Exception,
        context: dict[str, Any],
    ) -> TraceEvent:
        """
        Hook called when agent execution fails.

        Args:
            session: Current session
            error: Exception that occurred
            context: Context from before_call

        Returns:
            TraceEvent for the error
        """
        import traceback

        event = create_error_event(
            session_id=session.session_id,
            step_number=0,
            error_type=type(error).__name__,
            error_message=str(error),
            agent_id=session.agent_id,
            error_stack=traceback.format_exc(),
            cumulative_cost=session.cumulative_cost,
            framework=Framework.LANGGRAPH,
            framework_version=self._framework_version,
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            metadata={
                "agent_name": self._name,
            },
        )

        logger.debug(
            "langgraph_on_error",
            agent=self._name,
            session_id=str(session.session_id),
            error_type=type(error).__name__,
        )

        return event

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize the result for storage."""
        if result is None:
            return {"result": None}
        if isinstance(result, (str, int, float, bool)):
            return {"result": result}
        if isinstance(result, (list, tuple)):
            return {"result": list(result)}
        if isinstance(result, dict):
            return {"result": result}
        if hasattr(result, "model_dump"):
            return {"result": result.model_dump()}
        if hasattr(result, "__dict__"):
            return {"result": {k: v for k, v in result.__dict__.items() if not k.startswith("_")}}
        return {"result": str(result)}


def is_langgraph_agent(agent: Any) -> bool:
    """
    Check if an agent is a LangGraph graph.

    This function inspects the agent to determine if it's a
    LangGraph StateGraph or CompiledGraph instance.

    Args:
        agent: The agent to check

    Returns:
        True if the agent is a LangGraph graph
    """
    agent_type = type(agent).__name__
    module = getattr(type(agent), "__module__", "") or ""

    # Check for actual langgraph module paths (not just substring match)
    # This prevents false positives from test files named "test_langgraph_*.py"
    module_parts = module.lower().split(".")
    if "langgraph" in module_parts:
        return True

    # Check common LangGraph class names
    if agent_type in ("StateGraph", "CompiledGraph", "CompiledStateGraph"):
        return True

    # Check for LangGraph-specific methods (both must be present)
    if hasattr(agent, "get_graph") and hasattr(agent, "invoke"):
        return True

    # Check for StateGraph pattern (both must be present)
    if hasattr(agent, "compile") and hasattr(agent, "add_node"):
        return True

    # Check if a function's defining module imports langgraph symbols.
    # This handles the common case: user wraps a plain function that
    # internally uses LangGraph (e.g., builds a graph and calls invoke).
    _LANGGRAPH_SYMBOLS = frozenset(
        {
            "StateGraph",
            "CompiledGraph",
            "CompiledStateGraph",
            "MessageGraph",
            "END",
            "START",
        }
    )
    if callable(agent) and hasattr(agent, "__globals__"):
        func_globals = agent.__globals__
        for name in _LANGGRAPH_SYMBOLS:
            obj = func_globals.get(name)
            if obj is not None:
                obj_module = getattr(obj, "__module__", "") or ""
                if "langgraph" in obj_module.split("."):
                    return True

    return False
