# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Trace Models
# Implements PRD-005

"""
Trace event models for the Clyro SDK.

This module defines the data structures for capturing trace events
during agent execution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, field_serializer

logger = structlog.get_logger(__name__)


class EventType(StrEnum):
    """Types of trace events that can be captured."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVER_CALL = "retriever_call"
    TASK_START = "task_start"
    TASK_END = "task_end"
    AGENT_COMMUNICATION = "agent_communication"
    TASK_DELEGATION = "task_delegation"
    STATE_TRANSITION = "state_transition"
    POLICY_CHECK = "policy_check"
    ERROR = "error"
    STEP = "step"


class Framework(StrEnum):
    """Supported agent frameworks."""

    LANGGRAPH = "langgraph"
    CREWAI = "crewai"
    GENERIC = "generic"
    MCP = "mcp"  # FRD-015: MCP wrapper as first-class framework
    CLAUDE_AGENT_SDK = "claude_agent_sdk"  # Claude Agent SDK adapter
    ANTHROPIC = "anthropic"  # Anthropic SDK adapter (FRD-001–FRD-012)


class AgentStage(StrEnum):
    """
    Agent execution stage for Think/Act/Observe visualization (TDD v1.4).

    These stages map to the cognitive cycle of autonomous agents:
    - THINK: Reasoning, planning, deciding what to do next
    - ACT: Executing an action (tool call, API request, etc.)
    - OBSERVE: Processing results, updating state, reflecting
    """

    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"


class TraceEvent(BaseModel):
    """
    A single trace event captured during agent execution.

    Trace events form the core observability data that enables
    replay, debugging, and reliability analysis.
    """

    # Identifiers
    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    agent_id: UUID | None = Field(default=None, description="Agent identifier")
    session_id: UUID = Field(description="Session this event belongs to")
    parent_event_id: UUID | None = Field(
        default=None, description="Parent event for nested operations"
    )

    # Event classification
    event_type: EventType = Field(description="Type of event")
    event_name: str | None = Field(default=None, description="Human-readable event name")

    # Timing
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred",
    )
    duration_ms: int = Field(default=0, ge=0, description="Event duration in milliseconds")

    # Framework context
    framework: Framework = Field(default=Framework.GENERIC, description="Agent framework")
    framework_version: str | None = Field(default=None, description="Framework version string")

    # Agent execution stage (TDD v1.4)
    agent_stage: AgentStage = Field(
        default=AgentStage.THINK,
        description="Agent execution stage: 'think' (reasoning/planning), "
        "'act' (executing action), 'observe' (processing results)",
    )

    # Event data
    input_data: dict[str, Any] | None = Field(default=None, description="Input to the operation")
    output_data: dict[str, Any] | None = Field(
        default=None, description="Output from the operation"
    )
    state_snapshot: dict[str, Any] | None = Field(
        default=None, description="State at time of event (for state_transition)"
    )

    # Metrics
    token_count_input: int = Field(default=0, ge=0, description="Input tokens used")
    token_count_output: int = Field(default=0, ge=0, description="Output tokens generated")
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0, description="Cost of this event in USD")

    # Execution control tracking
    step_number: int = Field(default=0, ge=0, description="Step number in session")
    cumulative_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cumulative cost up to this event"
    )
    state_hash: str | None = Field(default=None, description="Hash of state for loop detection")

    # Error context
    error_type: str | None = Field(default=None, description="Exception type if error")
    error_message: str | None = Field(default=None, description="Error message")
    error_stack: str | None = Field(default=None, description="Error stack trace")

    # Extensibility
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "forbid"}

    @field_serializer("event_id", "agent_id", "session_id", "parent_event_id")
    def serialize_uuid(self, v: UUID | None) -> str | None:
        """Serialize UUIDs to strings."""
        return str(v) if v is not None else None

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        """Serialize datetime to ISO format."""
        return v.isoformat()

    @field_serializer("cost_usd", "cumulative_cost")
    def serialize_decimal(self, v: Decimal) -> str:
        """Serialize Decimal to string."""
        return str(v)

    @field_serializer("event_type")
    def serialize_event_type(self, v: EventType) -> str:
        """Serialize enum to string."""
        return v.value

    @field_serializer("framework")
    def serialize_framework(self, v: Framework) -> str:
        """Serialize enum to string."""
        return v.value

    @field_serializer("agent_stage")
    def serialize_agent_stage(self, v: AgentStage) -> str:
        """Serialize enum to string."""
        return v.value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEvent:
        """Create from dictionary."""
        return cls.model_validate(data)


# Lazy import to avoid circular dependency
_loop_detector_instance = None


def compute_state_hash(state: dict[str, Any] | None) -> str | None:
    """
    Compute a deterministic hash of state for loop detection.

    This uses the LoopDetector's filtering logic to exclude non-deterministic
    fields (timestamps, request_ids, etc.) before hashing. This ensures
    consistency between runtime loop detection and stored state hashes.

    Args:
        state: State dictionary to hash

    Returns:
        SHA-256 hash of the filtered state, or None if state is None
    """
    global _loop_detector_instance
    if _loop_detector_instance is None:
        from clyro.loop_detector import LoopDetector

        _loop_detector_instance = LoopDetector()

    return _loop_detector_instance.compute_state_hash(state)


def create_session_start_event(
    session_id: UUID,
    agent_id: UUID | None = None,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.THINK,
) -> TraceEvent | None:
    """Create a session start event. Returns None on internal error (fail-open)."""
    try:
        return TraceEvent(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EventType.SESSION_START,
            event_name="session_start",
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            input_data=input_data,
            step_number=0,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="session_start", fail_open=True)
        return None


def create_session_end_event(
    session_id: UUID,
    agent_id: UUID | None = None,
    step_number: int = 0,
    cumulative_cost: Decimal = Decimal("0"),
    output_data: dict[str, Any] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    error_stack: str | None = None,
    duration_ms: int = 0,
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.OBSERVE,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
) -> TraceEvent | None:
    """Create a session end event. Returns None on internal error (fail-open)."""
    try:
        event_type = EventType.ERROR if error_type else EventType.SESSION_END
        return TraceEvent(
            session_id=session_id,
            agent_id=agent_id,
            event_type=event_type,
            event_name="session_end" if not error_type else "session_error",
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            cumulative_cost=cumulative_cost,
            output_data=output_data,
            error_type=error_type,
            error_message=error_message,
            error_stack=error_stack,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="session_end", fail_open=True)
        return None


def create_step_event(
    session_id: UUID,
    step_number: int,
    event_name: str,
    agent_id: UUID | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    state_snapshot: dict[str, Any] | None = None,
    duration_ms: int = 0,
    cumulative_cost: Decimal = Decimal("0"),
    token_count_input: int = 0,
    token_count_output: int = 0,
    cost_usd: Decimal = Decimal("0"),
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.THINK,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
) -> TraceEvent | None:
    """Create a step event for generic agent execution. Returns None on internal error (fail-open)."""
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.STEP,
            event_name=event_name,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data=input_data,
            output_data=output_data,
            state_snapshot=state_snapshot,
            state_hash=compute_state_hash(state_snapshot),
            duration_ms=duration_ms,
            cumulative_cost=cumulative_cost,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="step", fail_open=True)
        return None


def create_llm_call_event(
    session_id: UUID,
    step_number: int,
    model: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any] | None = None,
    agent_id: UUID | None = None,
    token_count_input: int = 0,
    token_count_output: int = 0,
    cost_usd: Decimal = Decimal("0"),
    cumulative_cost: Decimal = Decimal("0"),
    duration_ms: int = 0,
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.THINK,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
) -> TraceEvent | None:
    """Create an LLM call event. Returns None on internal error (fail-open)."""
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.LLM_CALL,
            event_name=model,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data=input_data,
            output_data=output_data,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            cost_usd=cost_usd,
            cumulative_cost=cumulative_cost,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="llm_call", fail_open=True)
        return None


def create_tool_call_event(
    session_id: UUID,
    step_number: int,
    tool_name: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any] | None = None,
    agent_id: UUID | None = None,
    duration_ms: int = 0,
    cumulative_cost: Decimal = Decimal("0"),
    token_count_input: int = 0,
    token_count_output: int = 0,
    cost_usd: Decimal = Decimal("0"),
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.ACT,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
) -> TraceEvent | None:
    """Create a tool call event. Returns None on internal error (fail-open)."""
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.TOOL_CALL,
            event_name=tool_name,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            cumulative_cost=cumulative_cost,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            cost_usd=cost_usd,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="tool_call", fail_open=True)
        return None


def create_retriever_call_event(
    session_id: UUID,
    step_number: int,
    retriever_name: str,
    query: str,
    documents: list[dict[str, Any]] | None = None,
    agent_id: UUID | None = None,
    duration_ms: int = 0,
    cumulative_cost: Decimal = Decimal("0"),
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.ACT,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
) -> TraceEvent:
    """
    Create a retriever call event for RAG operations.

    Args:
        session_id: Session identifier
        step_number: Step number in execution
        retriever_name: Name of the retriever
        query: Search query
        documents: Retrieved documents (list of dicts with content, metadata, etc.)
        agent_id: Agent identifier
        duration_ms: Duration in milliseconds
        cumulative_cost: Cumulative cost
        error_type: Error type if failed
        error_message: Error message if failed
        metadata: Additional metadata
        agent_stage: Agent execution stage
        framework: Framework being used
        framework_version: Framework version
        event_id: Optional override for auto-generated event ID
        parent_event_id: Optional parent event ID for hierarchy

    Returns:
        TraceEvent for the retriever call, or None on internal error (fail-open)
    """
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.RETRIEVER_CALL,
            event_name=retriever_name,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data={"query": query},
            output_data={"documents": documents} if documents else None,
            duration_ms=duration_ms,
            cumulative_cost=cumulative_cost,
            error_type=error_type,
            error_message=error_message,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="retriever_call", fail_open=True)
        return None


def create_error_event(
    session_id: UUID,
    step_number: int,
    error_type: str,
    error_message: str,
    agent_id: UUID | None = None,
    error_stack: str | None = None,
    cumulative_cost: Decimal = Decimal("0"),
    token_count_input: int = 0,
    token_count_output: int = 0,
    cost_usd: Decimal = Decimal("0"),
    metadata: dict[str, Any] | None = None,
    agent_stage: AgentStage = AgentStage.OBSERVE,
    framework: Framework = Framework.GENERIC,
    framework_version: str | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
) -> TraceEvent | None:
    """Create an error event. Returns None on internal error (fail-open)."""
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.ERROR,
            event_name=error_type,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data=input_data,
            output_data=output_data,
            error_type=error_type,
            error_message=error_message,
            error_stack=error_stack,
            cumulative_cost=cumulative_cost,
            token_count_input=token_count_input,
            token_count_output=token_count_output,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="error", fail_open=True)
        return None


def create_state_transition_event(
    session_id: UUID,
    step_number: int,
    node_name: str,
    agent_id: UUID | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    state_snapshot: dict[str, Any] | None = None,
    duration_ms: int = 0,
    cumulative_cost: Decimal = Decimal("0"),
    framework: Framework = Framework.LANGGRAPH,
    framework_version: str | None = None,
    agent_stage: AgentStage = AgentStage.THINK,
    metadata: dict[str, Any] | None = None,
    event_id: UUID | None = None,  # Implements FRD-001
    parent_event_id: UUID | None = None,  # Implements FRD-001
) -> TraceEvent | None:
    """
    Create a state transition event for LangGraph node execution.

    This event type captures graph node executions with state snapshots,
    enabling replay visualization of the Think/Act/Observe cycle.

    Returns:
        TraceEvent with event_type=STATE_TRANSITION, or None on internal error (fail-open)
    """
    try:
        return TraceEvent(
            event_id=event_id or uuid4(),
            session_id=session_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
            event_type=EventType.STATE_TRANSITION,
            event_name=node_name,
            framework=framework,
            framework_version=framework_version,
            agent_stage=agent_stage,
            step_number=step_number,
            input_data=input_data,
            output_data=output_data,
            state_snapshot=state_snapshot,
            state_hash=compute_state_hash(state_snapshot),
            duration_ms=duration_ms,
            cumulative_cost=cumulative_cost,
            metadata=metadata or {},
        )
    except Exception:
        logger.warning("clyro_create_event_failed", event_type="state_transition", fail_open=True)
        return None
