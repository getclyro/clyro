# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Hook Evaluator
# Implements FRD-HK-001, FRD-HK-003, FRD-HK-004, FRD-HK-005, FRD-HK-006

"""Orchestrates PreToolUse evaluation: prevention stack + session state."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from typing import Any

import structlog

from clyro.cost import CostTracker
from clyro.evaluation import enrich_tool_input
from clyro.loop_detector import LoopDetector
from clyro.policy import LocalPolicyEvaluator

from .audit import AuditLogger, redact_params
from .backend import create_trace_event, enqueue_event, estimate_tokens, report_violation
from .config import HookConfig
from .constants import (
    DEFAULT_AGENT_NAME,
    DEFAULT_REDACT_PARAMETERS,
)
from .models import HookInput, HookOutput, SessionState
from .policy_loader import get_merged_policies
from .state import load_state, save_state

logger = structlog.get_logger()


def _enrich_tool_input(
    tool_input: dict[str, Any],
    tool_name: str,
    session_id: str,
    step_count: int,
    accumulated_cost_usd: float,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Add _clyro_ prefixed synthetic parameters for policy evaluation.

    FRD-HK-006: Delegates to :func:`clyro.evaluation.enrich_tool_input`
    with ``use_prefix=True``.
    """
    return enrich_tool_input(
        tool_input,
        tool_name=tool_name,
        session_id=session_id,
        step_number=step_count,
        accumulated_cost_usd=accumulated_cost_usd,
        agent_id=agent_id,
        use_prefix=True,
    )


def _save_and_block(
    state: SessionState,
    loop_detector: LoopDetector,
    audit: AuditLogger,
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    reason: str,
    rule_results: list[dict[str, Any]] | None = None,
    *,
    config: HookConfig | None = None,
) -> HookOutput:
    """Save state, log audit block entry, and return block decision.

    Blocked calls still increment step_count and accumulate a cost estimate
    for the input payload so that observability data (step ordering and
    running cost) stays accurate across allowed *and* blocked calls.
    """
    # Increment step count so blocked calls are properly ordered
    state.step_count += 1

    # Estimate cost for the blocked call's input (no output cost)
    # Fail-closed: on serialization error, assume max budget so cost checks trigger
    if config is not None:
        try:
            params_json_len = len(json.dumps(tool_input, default=str))
        except Exception:
            logger.warning("cost_estimate_serialization_error", fallback="max_budget")
            params_json_len = int(
                config.global_.max_cost_usd / config.global_.cost_per_token_usd * 4
            )
        cost_estimate = params_json_len / 4 * config.global_.cost_per_token_usd
        state.accumulated_cost_usd += cost_estimate

    state.loop_history = list(loop_detector._call_history)
    state.last_tool_call_at = datetime.now(UTC)
    save_state(state)
    audit.log_pre_tool_use(
        session_id=session_id,
        tool_name=tool_name,
        decision="block",
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        tool_input=tool_input,
        reason=reason,
        rule_results=rule_results,
        agent_id=state.agent_id,
    )
    return HookOutput(decision="block", reason=reason)


def evaluate(
    hook_input: HookInput,
    config: HookConfig,
    audit: AuditLogger,
) -> HookOutput | None:
    """Run the four-stage prevention stack and return a decision.

    Returns HookOutput with decision="block" if any stage triggers,
    or None (empty stdout = allow) if all stages pass.

    FRD-HK-001: Pipeline order: loop → step → cost → policy.
    Short-circuits on first violation.
    """
    session_id = hook_input.session_id
    tool_name = hook_input.tool_name
    tool_input = hook_input.tool_input or {}

    # Load session state
    state = load_state(session_id)

    # Detect new turn: first tool call or after a Stop hook fired
    next_step_peek = state.step_count + 1
    is_new_turn = next_step_peek == 1 or state.turn_ended
    if is_new_turn:
        audit.log_session_start(
            session_id=session_id,
            agent_id=state.agent_id,
        )
        state.turn_ended = False
        # Snapshot current cumulative values so session_end can compute per-turn deltas
        state.turn_start_step_count = state.step_count
        state.turn_start_cost_usd = state.accumulated_cost_usd
        save_state(state)

    # Get merged policies (local + cloud with cache)
    merged_policies = get_merged_policies(config, state)

    # ── Stage 1: Loop Detection (FRD-HK-003) ──
    loop_detector = LoopDetector(
        threshold=config.global_.loop_detection.threshold,
        window=config.global_.loop_detection.window,
    )
    # Reconstruct history from persisted state
    loop_detector._call_history = deque(
        state.loop_history,
        maxlen=config.global_.loop_detection.window,
    )

    try:
        loop_triggered, loop_details = loop_detector.check(tool_name, tool_input)
    except Exception as e:
        # Fail-closed: loop detection failure blocks rather than skipping
        logger.error("loop_detection_error", error=str(e))
        reason = f"Loop detection error (fail-closed): {e}"
        result = _save_and_block(
            state,
            loop_detector,
            audit,
            session_id,
            tool_name,
            tool_input,
            reason,
            config=config,
        )
        _emit_block_trace_events(
            config,
            session_id,
            state,
            tool_name,
            tool_input,
            audit,
            block_type="loop_detection_error",
            reason=reason,
            is_first_step=is_new_turn,
        )
        return result

    if loop_triggered:
        reason = (
            f"Loop detected: same tool call repeated "
            f"{loop_details.get('repetition_count', '?')} times "
            f"(threshold: {loop_details.get('threshold', '?')})"
        )
        result = _save_and_block(
            state,
            loop_detector,
            audit,
            session_id,
            tool_name,
            tool_input,
            reason,
            config=config,
        )
        _emit_block_trace_events(
            config,
            session_id,
            state,
            tool_name,
            tool_input,
            audit,
            block_type="loop_detected",
            reason=reason,
            is_first_step=is_new_turn,
        )
        return result

    # ── Stage 2: Step Limit (FRD-HK-004) ──
    next_step = state.step_count + 1
    if next_step > config.global_.max_steps:
        reason = f"Step limit exceeded: {next_step} > {config.global_.max_steps}"
        result = _save_and_block(
            state,
            loop_detector,
            audit,
            session_id,
            tool_name,
            tool_input,
            reason,
            config=config,
        )
        _emit_block_trace_events(
            config,
            session_id,
            state,
            tool_name,
            tool_input,
            audit,
            block_type="step_limit_exceeded",
            reason=reason,
            is_first_step=is_new_turn,
        )
        return result

    # ── Stage 3: Cost Budget (FRD-HK-005) ──
    cost_tracker = CostTracker(
        max_cost_usd=config.global_.max_cost_usd,
        cost_per_token_usd=config.global_.cost_per_token_usd,
    )
    budget_exceeded, cost_details = cost_tracker.check_budget(
        accumulated_cost_usd=state.accumulated_cost_usd,
        params=tool_input,
    )
    if budget_exceeded:
        reason = (
            f"Cost budget exceeded: accumulated ${state.accumulated_cost_usd:.4f} "
            f"would exceed budget ${config.global_.max_cost_usd:.2f}"
        )
        result = _save_and_block(
            state,
            loop_detector,
            audit,
            session_id,
            tool_name,
            tool_input,
            reason,
            config=config,
        )
        _emit_block_trace_events(
            config,
            session_id,
            state,
            tool_name,
            tool_input,
            audit,
            block_type="budget_exceeded",
            reason=reason,
            is_first_step=is_new_turn,
        )
        return result

    # ── Stage 4: Policy Rules (FRD-HK-006) ──
    enriched_input = _enrich_tool_input(
        tool_input,
        tool_name,
        session_id,
        next_step,
        state.accumulated_cost_usd,
        agent_id=state.agent_id,
    )

    # Build eval config with all merged policies as global
    # FRD-HK-006: per-tool policies first, then global — PolicyEvaluator handles
    # tool-specific matching via its own tool routing. We put all merged policies
    # on global_ and clear per-tool to avoid double-evaluation.
    eval_config = config.model_copy(deep=True)
    eval_config.global_.policies = list(merged_policies)
    eval_config.tools = {}

    policy_evaluator = LocalPolicyEvaluator(config=eval_config)
    violated, violation_details, rule_results = policy_evaluator.evaluate(
        tool_name=tool_name,
        arguments=enriched_input,
    )

    if violated:
        rule_name = violation_details.get("rule_name", "Unknown rule")
        parameter = violation_details.get("parameter", "")
        operator = violation_details.get("operator", "")
        expected = violation_details.get("expected", "")
        actual = violation_details.get("actual", "")
        reason = (
            f"Policy violation: {rule_name} ({parameter} {operator} {expected}, actual: {actual})"
        )
        result = _save_and_block(
            state,
            loop_detector,
            audit,
            session_id,
            tool_name,
            tool_input,
            reason,
            rule_results=rule_results,
            config=config,
        )

        # Report violation to backend (fail-open)
        api_key = config.resolved_api_key
        if api_key and state.agent_id:
            try:
                report_violation(
                    api_key=api_key,
                    api_url=config.resolved_api_url,
                    agent_id=state.agent_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    reason=reason,
                    rule_results=rule_results,
                    circuit=state.circuit_breaker,
                    violation_details=violation_details,
                    tool_input=tool_input,
                    step_number=state.step_count,
                )
                save_state(state)  # Persist updated circuit breaker state
            except Exception as e:
                logger.warning("violation_report_error", error=str(e))

        # Emit pre_tool_use + policy_check + error trace events (FRD-HK-008)
        _emit_block_trace_events(
            config,
            session_id,
            state,
            tool_name,
            tool_input,
            audit,
            block_type="policy_violation",
            reason=reason,
            rule_results=rule_results,
            is_first_step=is_new_turn,
        )

        return result

    # ── All stages passed ──
    # Estimate pre-call cost
    # Fail-closed: on serialization error, assume max budget so cost checks trigger
    try:
        params_json_len = len(json.dumps(tool_input, default=str))
    except Exception:
        logger.warning("cost_estimate_serialization_error", fallback="max_budget")
        params_json_len = int(config.global_.max_cost_usd / config.global_.cost_per_token_usd * 4)
    cost_estimate = params_json_len / 4 * config.global_.cost_per_token_usd

    state.step_count = next_step
    state.loop_history = list(loop_detector._call_history)
    state.accumulated_cost_usd += cost_estimate
    state.pre_call_cost_estimate = cost_estimate
    state.last_tool_call_at = datetime.now(UTC)
    save_state(state)

    audit.log_pre_tool_use(
        session_id=session_id,
        tool_name=tool_name,
        decision="allow",
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        tool_input=tool_input,
        rule_results=rule_results,
        agent_id=state.agent_id,
    )

    # Emit session_start on new turn + pre_tool_use + policy_check trace (FRD-HK-008)
    _emit_allow_trace_events(
        config,
        session_id,
        state,
        tool_name,
        tool_input,
        audit,
        rule_results=rule_results,
        is_first_step=is_new_turn,
    )
    # Persist last_pre_tool_event_id so PostToolUse can wire parent
    save_state(state)

    return None  # Empty stdout = allow


def _emit_allow_trace_events(
    config: HookConfig,
    session_id: str,
    state: SessionState,
    tool_name: str,
    tool_input: dict[str, Any],
    audit: AuditLogger,
    *,
    rule_results: list[dict[str, Any]] | None = None,
    is_first_step: bool = False,
) -> None:
    """Emit session_start (on first step), pre_tool_use, and policy_check trace events.

    Events are queued to event queue file and flushed as a batch at session-end.
    """
    # Audit: policy_check (always logged regardless of backend)
    audit.log_policy_check(
        session_id=session_id,
        tool_name=tool_name,
        decision="allow",
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        tool_input=tool_input,
        rule_results=rule_results,
        agent_id=state.agent_id,
    )

    api_key = config.resolved_api_key
    if not api_key:
        return

    try:
        agent_name = getattr(config.backend, "agent_name", None) or DEFAULT_AGENT_NAME
        redact_patterns = config.audit.redact_parameters or DEFAULT_REDACT_PARAMETERS
        redacted_input = redact_params(tool_input, redact_patterns)
        input_tokens = estimate_tokens(len(json.dumps(redacted_input, default=str)))

        # Emit session_start on first tool call (FRD-HK-008)
        if is_first_step:
            start_event = create_trace_event(
                "session_start",
                session_id,
                agent_id=state.agent_id,
                agent_name=agent_name,
                step_number=state.turn_start_step_count,
                accumulated_cost_usd=state.turn_start_cost_usd,
            )
            enqueue_event(session_id, start_event)

        # Emit pre_tool_use — raw hook invocation trace
        # cost_usd = estimate for this call; accumulated_cost_usd = running total
        pre_tool_event = create_trace_event(
            "pre_tool_use",
            session_id,
            agent_id=state.agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": redacted_input},
            cost_usd=state.pre_call_cost_estimate,
            token_count_input=input_tokens,
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            metadata={"decision": "allow"},
        )
        pre_tool_event_id = pre_tool_event["event_id"]
        enqueue_event(session_id, pre_tool_event)

        # Store pre_tool_use event_id so tool_call_observe can wire parent
        state.last_pre_tool_event_id = pre_tool_event_id

        # Emit policy_check (think stage) — governance decision trace
        policy_event = create_trace_event(
            "policy_check",
            session_id,
            agent_id=state.agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": redacted_input},
            token_count_input=input_tokens,
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            parent_event_id=pre_tool_event_id,
            metadata={"decision": "allow", "rule_results": rule_results or []},
        )
        enqueue_event(session_id, policy_event)
    except Exception as e:
        logger.debug("trace_emission_failed", error=str(e))


def _emit_block_trace_events(
    config: HookConfig,
    session_id: str,
    state: SessionState,
    tool_name: str,
    tool_input: dict[str, Any],
    audit: AuditLogger,
    *,
    block_type: str,
    reason: str,
    rule_results: list[dict[str, Any]] | None = None,
    is_first_step: bool = False,
) -> None:
    """Emit pre_tool_use + policy_check + error trace events for blocked calls.

    Events are queued to event queue file and flushed as a batch at session-end.
    When ``is_first_step`` is True a session_start trace event is emitted first
    so that every turn is represented in ClickHouse regardless of whether its
    first tool call was allowed or blocked.
    """
    # Audit: policy_check + error (always logged regardless of backend)
    audit.log_policy_check(
        session_id=session_id,
        tool_name=tool_name,
        decision="block",
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        tool_input=tool_input,
        rule_results=rule_results,
        agent_id=state.agent_id,
    )
    audit.log_error(
        session_id=session_id,
        tool_name=tool_name,
        error_type=block_type,
        error_message=reason,
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        agent_id=state.agent_id,
    )

    api_key = config.resolved_api_key
    if not api_key:
        return

    try:
        agent_name = getattr(config.backend, "agent_name", None) or DEFAULT_AGENT_NAME
        redact_patterns = config.audit.redact_parameters or DEFAULT_REDACT_PARAMETERS
        redacted_input = redact_params(tool_input, redact_patterns)

        # Emit session_start on first tool call of a turn (FRD-HK-008)
        if is_first_step:
            start_event = create_trace_event(
                "session_start",
                session_id,
                agent_id=state.agent_id,
                agent_name=agent_name,
                step_number=state.turn_start_step_count,
                accumulated_cost_usd=state.turn_start_cost_usd,
            )
            enqueue_event(session_id, start_event)

        # Emit pre_tool_use — raw hook invocation trace
        pre_tool_event = create_trace_event(
            "pre_tool_use",
            session_id,
            agent_id=state.agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": redacted_input},
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            metadata={"decision": "block"},
        )
        pre_tool_event_id = pre_tool_event["event_id"]
        enqueue_event(session_id, pre_tool_event)

        # Emit policy_check (think stage) with block decision
        policy_event = create_trace_event(
            "policy_check",
            session_id,
            agent_id=state.agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": redacted_input},
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            parent_event_id=pre_tool_event_id,
            metadata={"decision": "block", "rule_results": rule_results or []},
        )
        policy_event_id = policy_event["event_id"]
        enqueue_event(session_id, policy_event)

        # Emit error event for the blocked call (parent wired to policy_check)
        error_event = create_trace_event(
            "error",
            session_id,
            agent_id=state.agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            parent_event_id=policy_event_id,
            error_type=block_type,
            error_message=reason,
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            metadata={"block_reason": block_type},
        )
        enqueue_event(session_id, error_event)
    except Exception as e:
        logger.debug("trace_emission_failed", error=str(e))
