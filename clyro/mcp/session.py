# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Session State
# Implements FRD-003

"""
Per-process session state container.

``McpSession`` holds all mutable state for a single wrapper process
lifetime: session ID, step counter, and accumulated cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class McpSession:
    """
    Per-process session state container.

    Implements FRD-003: process-scoped sessions.
    Created at process start, destroyed at process exit.
    """

    session_id: UUID = field(default_factory=uuid4)
    step_count: int = 0
    accumulated_cost_usd: float = 0.0
    agent_id: UUID | None = None  # FRD-016: set by AgentRegistrar on backend-enabled sessions

    def increment_step(self) -> int:
        """Increment and return the new step count."""
        self.step_count += 1
        return self.step_count

    def add_cost(self, cost: float) -> None:
        """Add estimated cost for a forwarded call."""
        self.accumulated_cost_usd += cost


@dataclass(frozen=True)
class PendingCall:
    """
    Tracks a forwarded tools/call awaiting a server response.

    Used by MessageRouter to correlate server responses with pending
    requests for CostTracker accumulation (TDD §2.2, §2.7).
    """

    request_id: str | int
    tool_name: str
    params_json_len: int
    forwarded_at: float
