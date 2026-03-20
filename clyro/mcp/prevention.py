# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Prevention Stack
# Implements FRD-004, FRD-005, FRD-006, FRD-007

"""
Orchestrate the four-stage evaluation pipeline for each ``tools/call``
request:

1. LoopDetector  → block if pattern repeats ≥ threshold
2. StepLimit     → block if step_count > max_steps
3. CostTracker   → block if budget would be exceeded
4. PolicyEvaluator → block if a business rule is violated

Short-circuits on the first violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clyro.config import WrapperConfig
from clyro.cost import CostTracker
from clyro.evaluation import enrich_tool_input
from clyro.loop_detector import LoopDetector
from clyro.mcp.session import McpSession
from clyro.policy import LocalPolicyEvaluator


@dataclass(frozen=True)
class AllowDecision:
    """Call is allowed — forward to server."""

    tool_name: str
    step_number: int
    rule_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BlockDecision:
    """Call is blocked — return error to host."""

    block_type: str
    tool_name: str
    step_number: int
    details: dict[str, Any] = field(default_factory=dict)


class PreventionStack:
    """
    Four-stage prevention pipeline evaluated for every ``tools/call``.

    Args:
        config: Validated wrapper configuration.
    """

    def __init__(self, config: WrapperConfig) -> None:
        self._config = config
        self._loop_detector = LoopDetector(
            threshold=config.global_.loop_detection.threshold,
            window=config.global_.loop_detection.window,
        )
        self._cost_tracker = CostTracker(
            max_cost_usd=config.global_.max_cost_usd,
            cost_per_token_usd=config.global_.cost_per_token_usd,
        )
        self._policy_evaluator = LocalPolicyEvaluator(config)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        session: McpSession,
    ) -> AllowDecision | BlockDecision:
        """
        Evaluate a ``tools/call`` against all prevention stages.

        Step counter is incremented **before** evaluation (FRD-005).

        Args:
            tool_name: ``params.name`` from the JSON-RPC request.
            arguments: ``params.arguments`` from the request.
            session: Current session state.

        Returns:
            ``AllowDecision`` or ``BlockDecision``.
        """
        # Step counter increments before evaluation (FRD-005)
        step = session.increment_step()

        # 1. Loop Detection (FRD-004)
        is_loop, loop_details = self._loop_detector.check(tool_name, arguments)
        if is_loop:
            return BlockDecision(
                block_type="loop_detected",
                tool_name=tool_name,
                step_number=step,
                details={**loop_details, "tool_name": tool_name},
            )

        # 2. Step Limit (FRD-005)
        if step > self._config.global_.max_steps:
            return BlockDecision(
                block_type="step_limit_exceeded",
                tool_name=tool_name,
                step_number=step,
                details={
                    "step_count": step,
                    "max_steps": self._config.global_.max_steps,
                },
            )

        # 3. Cost Bound (FRD-006)
        exceeds, cost_details = self._cost_tracker.check_budget(
            session.accumulated_cost_usd, arguments
        )
        if exceeds:
            return BlockDecision(
                block_type="budget_exceeded",
                tool_name=tool_name,
                step_number=step,
                details=cost_details,
            )

        # 4. Policy Evaluation (FRD-007)
        # Enrich arguments with session context so policy rules can target
        # session-level fields (cost, step_number, session_id, tool_name)
        enriched_args = enrich_tool_input(
            arguments,
            tool_name=tool_name,
            session_id=str(session.session_id),
            step_number=step,
            accumulated_cost_usd=session.accumulated_cost_usd,
            use_prefix=False,
        )

        violated, policy_details, rule_results = self._policy_evaluator.evaluate(
            tool_name, enriched_args
        )
        if violated:
            return BlockDecision(
                block_type="policy_violation",
                tool_name=tool_name,
                step_number=step,
                details={**policy_details, "_rule_results": rule_results},
            )

        return AllowDecision(
            tool_name=tool_name,
            step_number=step,
            rule_results=rule_results,
        )
