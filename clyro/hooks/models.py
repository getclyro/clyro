# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Data Models
# Implements FRD-HK-001, FRD-HK-002

"""Pydantic models for hook I/O and session state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HookInput(BaseModel):
    """stdin JSON from Claude Code hook system."""

    model_config = {"extra": "ignore"}

    session_id: str
    tool_name: str = ""
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] | None = None


class HookOutput(BaseModel):
    """stdout JSON for PreToolUse block decisions."""

    decision: Literal["block", "allow"]
    reason: str | None = None


class PolicyCache(BaseModel):
    """Cached cloud policy state within a session."""

    fetched_at: datetime | None = None
    ttl_seconds: int = 300
    merged_policies: list[dict[str, Any]] = Field(default_factory=list)


class CircuitBreakerSnapshot(BaseModel):
    """Persisted circuit breaker state for ephemeral process reuse."""

    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    half_open_successes: int = 0
    opened_at: float | None = None
    total_trips: int = 0


class SessionState(BaseModel):
    """Persisted session state in JSON file. FRD-HK-002."""

    session_id: str
    agent_id: str | None = None
    step_count: int = 0
    accumulated_cost_usd: float = 0.0
    loop_history: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_tool_call_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy_cache: PolicyCache = Field(default_factory=PolicyCache)
    cloud_disabled: bool = False
    pre_call_cost_estimate: float = 0.0
    circuit_breaker: CircuitBreakerSnapshot = Field(default_factory=CircuitBreakerSnapshot)
    turn_ended: bool = False
    turn_start_step_count: int = 0
    turn_start_cost_usd: float = 0.0
    last_pre_tool_event_id: str | None = None
