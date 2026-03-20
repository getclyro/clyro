# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Generic Adapter
# Implements PRD-001

"""
Generic adapter for wrapping any Python callable.

This adapter provides basic tracing for any callable agent
without framework-specific instrumentation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from clyro.session import Session
from clyro.trace import (
    Framework,
    TraceEvent,
    create_error_event,
    create_step_event,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class GenericAdapter:
    """
    Generic adapter for any Python callable.

    This adapter wraps a callable and captures:
    - Function invocation start/end
    - Input arguments
    - Return value or exception
    - Execution duration

    It provides basic tracing without framework-specific
    features like state transitions or node executions.
    """

    FRAMEWORK = Framework.GENERIC
    FRAMEWORK_VERSION = "1.0.0"

    def __init__(self, agent: Callable[..., T], config: ClyroConfig):
        """
        Initialize generic adapter.

        Args:
            agent: The callable to adapt
            config: SDK configuration
        """
        self._agent = agent
        self._config = config
        self._name = getattr(agent, "__name__", str(agent))

    @property
    def agent(self) -> Callable[..., T]:
        """Get the wrapped agent callable."""
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
        """Get the framework version."""
        return self.FRAMEWORK_VERSION

    def before_call(
        self,
        session: Session,
        args: tuple,
        kwargs: dict,
    ) -> dict[str, Any]:
        """
        Hook called before the agent is invoked.

        Args:
            session: Current session
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Context dictionary to pass to after_call
        """
        context = {
            "start_time": time.perf_counter(),
            "step_number": session.step_number + 1,
        }

        logger.debug(
            "generic_before_call",
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

        output_data = None
        if self._config.capture_outputs:
            output_data = self._serialize_result(result)

        event = create_step_event(
            session_id=session.session_id,
            step_number=context["step_number"],
            event_name=f"{self._name}_complete",
            agent_id=session.agent_id,
            output_data=output_data,
            duration_ms=duration_ms,
            cumulative_cost=session.cumulative_cost,
        )

        logger.debug(
            "generic_after_call",
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
            step_number=context.get("step_number", session.step_number),
            error_type=type(error).__name__,
            error_message=str(error),
            agent_id=session.agent_id,
            error_stack=traceback.format_exc(),
            cumulative_cost=session.cumulative_cost,
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            metadata={"agent_name": self._name},
        )

        logger.debug(
            "generic_on_error",
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


def detect_adapter(agent: Callable) -> str:
    """
    Detect the appropriate adapter for an agent.

    This function inspects the agent to determine which
    framework it belongs to and returns the adapter name.

    Detection priority:
    1. Anthropic (exact class match — highest specificity)
    2. LangGraph (StateGraph, CompiledGraph, get_graph method)
    3. CrewAI (Crew, Agent, kickoff method)
    4. Claude Agent SDK
    5. Generic (fallback for any callable)

    Args:
        agent: The agent callable or client to inspect

    Returns:
        Adapter name ('anthropic', 'langgraph', 'crewai', 'claude_agent_sdk', or 'generic')
    """
    # Import here to avoid circular imports
    from clyro.adapters.anthropic import is_anthropic_agent
    from clyro.adapters.claude_agent_sdk import is_claude_agent_sdk_agent
    from clyro.adapters.crewai import is_crewai_agent
    from clyro.adapters.langgraph import is_langgraph_agent

    # Check for Anthropic first (highest specificity)
    if is_anthropic_agent(agent):
        return "anthropic"

    # Check for LangGraph
    if is_langgraph_agent(agent):
        return "langgraph"

    # Check for CrewAI
    if is_crewai_agent(agent):
        return "crewai"

    # Check for Claude Agent SDK
    if is_claude_agent_sdk_agent(agent):
        return "claude_agent_sdk"

    return "generic"
