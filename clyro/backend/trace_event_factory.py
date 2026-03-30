# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Trace Event Factory
# Implements FRD-015

"""
Convert MCP wrapper audit entries and governance decisions into
SDK-compatible TraceEvent dicts for backend sync.

Event mapping (FRD-015, TDD §2.13):
    Process start      → event_type="session_start"
    Prevention eval    → event_type="policy_check",  agent_stage="think"
    Allowed tool call  → event_type="tool_call",     agent_stage="act"
    Tool response      → event_type="tool_call",     agent_stage="observe"
    Blocked tool call  → event_type="error",         agent_stage="act"
    Process exit       → event_type="session_end"
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from clyro import __version__
from clyro.mcp.session import McpSession

_OUTPUT_TRUNCATE_BYTES = 10 * 1024  # 10KB max output_data (TDD §2.13)
_DEFAULT_COST_PER_TOKEN_USD = 0.00001  # Same default as CostTracker


class TraceEventFactory:
    """
    Create SDK-compatible TraceEvent dicts from MCP wrapper events.

    All events include ``framework: "mcp"`` and ``cost_estimated: true``
    per FRD-006 / FRD-015 specification.

    Args:
        session: Current MCP session (for session_id, agent_id, step count, cost).
        cost_per_token_usd: Per-token rate for estimated cost on ACT events.
    """

    def __init__(
        self, session: McpSession, cost_per_token_usd: float = _DEFAULT_COST_PER_TOKEN_USD
    ) -> None:
        self._session = session
        self._cost_per_token_usd = cost_per_token_usd

    def create_trace_event(
        self,
        event_type: str,
        agent_stage: str | None,
        *,
        tool_name: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
        token_count_input: int = 0,
        token_count_output: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        step_number: int | None = None,
    ) -> dict[str, Any]:
        """Create a TraceEvent dict compatible with POST /v1/traces payload (FRD-015)."""
        # Truncate output_data to 10KB (TDD §2.13)
        truncated = False
        if output_data is not None:
            output_json = json.dumps(output_data, default=str)
            if len(output_json) > _OUTPUT_TRUNCATE_BYTES:
                output_data = {"_truncated": output_json[:_OUTPUT_TRUNCATE_BYTES]}
                truncated = True

        merged_metadata: dict[str, Any] = {
            **(metadata or {}),
            "_source": "mcp",
            "cost_estimated": True,
        }
        if truncated:
            merged_metadata["output_truncated"] = True

        event: dict[str, Any] = {
            # Identifiers
            "event_id": str(uuid4()),
            "session_id": str(self._session.session_id),
            "agent_id": str(self._session.agent_id) if self._session.agent_id else None,
            "parent_event_id": parent_event_id,
            # Classification
            "event_type": event_type,
            "event_name": tool_name or event_type,
            # Timing
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_ms": duration_ms,
            # Framework (FRD-015: always "mcp")
            "framework": "mcp",
            "framework_version": __version__,
            # Data capture
            "input_data": input_data,
            "output_data": output_data,
            "state_snapshot": None,
            # Metrics (FRD-006: character heuristic, always estimated)
            "token_count_input": token_count_input,
            "token_count_output": token_count_output,
            "cost_usd": cost_usd,
            # Execution tracking
            "step_number": step_number if step_number is not None else self._session.step_count,
            "cumulative_cost": str(self._session.accumulated_cost_usd),
            "state_hash": None,
            # Error context
            "error_type": error_type,
            "error_message": error_message,
            "error_stack": None,
            # Extensibility
            "metadata": merged_metadata,
        }

        # Only include agent_stage when set — omitting lets API default ("think") apply
        if agent_stage is not None:
            event["agent_stage"] = agent_stage

        return event

    def session_start(self) -> dict[str, Any]:
        """Create a session_start trace event."""
        metadata = {}
        if self._session.agent_name:
            metadata["agent_name"] = self._session.agent_name
        return self.create_trace_event("session_start", None, metadata=metadata or None)

    def session_end(self, total_duration_ms: int = 0) -> dict[str, Any]:
        """Create a session_end trace event."""
        return self.create_trace_event(
            "session_end",
            None,
            duration_ms=total_duration_ms,
            metadata={
                "total_steps": self._session.step_count,
                "total_cost_usd": str(self._session.accumulated_cost_usd),
            },
        )

    def policy_check(
        self,
        tool_name: str,
        params: dict[str, Any] | None,
        duration_ms: int = 0,
        *,
        decision: str | None = None,
        rule_results: list[dict[str, Any]] | None = None,
        parent_event_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a policy_check trace event (think stage).

        FRD-BE-011: When decision and rule_results are provided, they are
        merged into the event metadata for governance analytics consumption.
        """
        extra_metadata: dict[str, Any] = {}
        if decision is not None:
            extra_metadata["decision"] = decision
        if rule_results is not None:
            extra_metadata["rule_results"] = rule_results
        return self.create_trace_event(
            "policy_check",
            "think",
            input_data={"name": tool_name, "arguments": params} if params else None,
            duration_ms=duration_ms,
            metadata=extra_metadata if extra_metadata else None,
            parent_event_id=parent_event_id,
        )

    def tool_call_act(
        self,
        tool_name: str,
        params: dict[str, Any] | None,
        step_number: int,
        duration_ms: int = 0,
    ) -> dict[str, Any]:
        """Create a tool_call trace event (act stage — forwarded to server).

        Returns the event dict; callers can read event["event_id"] for parent wiring.
        """
        input_tokens = len(json.dumps(params or {}, default=str)) // 4
        # Estimate cost from input tokens using the session's cost-per-token rate
        estimated_cost = input_tokens * self._cost_per_token_usd
        return self.create_trace_event(
            "tool_call",
            "act",
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": params} if params else None,
            duration_ms=duration_ms,
            token_count_input=input_tokens,
            cost_usd=estimated_cost,
        )

    def tool_call_observe(
        self,
        tool_name: str,
        response_content: str | None,
        cost_usd: float,
        duration_ms: int = 0,
        parent_event_id: str | None = None,
        step_number: int | None = None,
    ) -> dict[str, Any]:
        """Create a tool_call trace event (observe stage — response received)."""
        output_tokens = len(response_content or "") // 4
        output_data = {"response": response_content} if response_content else None
        return self.create_trace_event(
            "tool_call",
            "observe",
            tool_name=tool_name,
            output_data=output_data,
            cost_usd=cost_usd,
            token_count_output=output_tokens,
            duration_ms=duration_ms,
            parent_event_id=parent_event_id,
            step_number=step_number,
        )

    def blocked_call(
        self,
        tool_name: str,
        block_type: str,
        block_message: str,
        block_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an error trace event for a blocked tool call."""
        return self.create_trace_event(
            "error",
            "act",
            tool_name=tool_name,
            input_data={"tool_name": tool_name},
            output_data={
                "error_type": block_type,
                "error_message": block_message,
            },
            error_type=block_type,
            error_message=block_message,
            metadata={"block_reason": block_type, **(block_details or {})},
        )
