# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Trace Handler
# Implements FRD-HK-008, FRD-HK-009

"""PostToolUse trace emission and Stop session lifecycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import structlog

from .audit import AuditLogger, redact_params
from .backend import create_trace_event, enqueue_event, estimate_tokens, flush_event_queue
from .config import HookConfig
from .constants import DEFAULT_AGENT_NAME, DEFAULT_REDACT_PARAMETERS
from .models import HookInput
from .state import StateLock, cleanup_stale_sessions, load_state, save_state

logger = structlog.get_logger()


def _tool_result_summary(tool_result: dict[str, Any] | None) -> str:
    """Summarize tool result without exposing raw content.

    FRD-HK-008: Only metadata (exit code, output length), not full output.
    """
    if not tool_result:
        return "no result"
    parts = []
    if "stdout" in tool_result:
        parts.append(f"stdout: {len(str(tool_result['stdout']))} chars")
    if "stderr" in tool_result:
        parts.append(f"stderr: {len(str(tool_result['stderr']))} chars")
    if "exitCode" in tool_result:
        parts.append(f"exitCode: {tool_result['exitCode']}")
    if "output" in tool_result:
        parts.append(f"output: {len(str(tool_result['output']))} chars")
    return ", ".join(parts) if parts else f"{len(str(tool_result))} chars total"


def handle_tool_complete(
    hook_input: HookInput,
    config: HookConfig,
    audit: AuditLogger,
) -> None:
    """Handle PostToolUse: adjust cost, emit trace, write audit.

    FRD-HK-008: Replace pre-call cost estimate with actual cost.
    Trace events are queued to event queue file and flushed at session-end.
    """
    session_id = hook_input.session_id
    tool_name = hook_input.tool_name
    tool_input = hook_input.tool_input or {}
    tool_result = hook_input.tool_result

    # Acquire state lock (TDD §5.2 step 3) to prevent race with concurrent PreToolUse
    with StateLock(session_id):
        state = load_state(session_id)

        # Compute actual cost from input + result lengths
        # Fail-closed: on serialization error, use pre-call estimate (don't reduce cost)
        try:
            params_len = len(json.dumps(tool_input, default=str))
        except Exception:
            logger.warning("cost_params_serialization_error", fallback="pre_call_estimate")
            params_len = (
                int(state.pre_call_cost_estimate / config.global_.cost_per_token_usd * 4)
                if config.global_.cost_per_token_usd > 0
                else 0
            )
        try:
            response_len = len(json.dumps(tool_result, default=str)) if tool_result else 0
        except Exception:
            logger.warning("cost_response_serialization_error", fallback="params_len")
            response_len = params_len  # Assume response is at least as large as params

        actual_cost = (params_len + response_len) / 4 * config.global_.cost_per_token_usd

        # Replace pre-call estimate with actual
        pre_call_estimate = state.pre_call_cost_estimate
        state.accumulated_cost_usd = state.accumulated_cost_usd - pre_call_estimate + actual_cost
        state.pre_call_cost_estimate = 0.0

        # Compute duration_ms from PreToolUse timestamp (FRD-HK-008)
        now = datetime.now(UTC)
        last_call = state.last_tool_call_at
        if last_call.tzinfo is None:
            last_call = last_call.replace(tzinfo=UTC)
        duration_ms = int((now - last_call).total_seconds() * 1000)

        agent_id = state.agent_id

        save_state(state)

    # Write audit log
    audit.log_post_tool_use(
        session_id=session_id,
        tool_name=tool_name,
        step_number=state.step_count,
        accumulated_cost_usd=state.accumulated_cost_usd,
        tool_input=tool_input,
        duration_ms=duration_ms,
        agent_id=agent_id,
    )

    # Queue trace event for batch flush at session-end (FRD-HK-008)
    api_key = config.resolved_api_key
    if api_key:
        redact_patterns = config.audit.redact_parameters or DEFAULT_REDACT_PARAMETERS
        agent_name = getattr(config.backend, "agent_name", None) or DEFAULT_AGENT_NAME
        redacted_input = redact_params(tool_input, redact_patterns)
        input_tokens = estimate_tokens(len(json.dumps(redacted_input, default=str)))
        result_summary = _tool_result_summary(tool_result)
        output_tokens = estimate_tokens(len(result_summary))

        trace_event = create_trace_event(
            "tool_call_observe",
            session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            input_data={"name": tool_name, "arguments": redacted_input},
            output_data={"summary": result_summary},
            duration_ms=duration_ms,
            cost_usd=actual_cost,
            token_count_input=input_tokens,
            token_count_output=output_tokens,
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            parent_event_id=state.last_pre_tool_event_id,
        )
        enqueue_event(session_id, trace_event)


def handle_session_end(
    hook_input: HookInput,
    config: HookConfig,
    audit: AuditLogger,
) -> None:
    """Handle Stop hook: flush event queue, emit session_end trace, clean up.

    FRD-HK-009: Session summary + event queue flush + cleanup.
    """
    session_id = hook_input.session_id

    # Load state (no lock needed — session is ending)
    state = load_state(session_id)
    if state.step_count == 0 and state.accumulated_cost_usd == 0.0:
        # No prior tool calls — warn and exit
        logger.warning("session_end_no_state", session_id=session_id)

    # Compute session summary
    now = datetime.now(UTC)
    started = state.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    duration_seconds = (now - started).total_seconds()

    agent_id = state.agent_id

    # Compute per-turn deltas from turn-start snapshot
    turn_steps = state.step_count - state.turn_start_step_count
    turn_cost_usd = state.accumulated_cost_usd - state.turn_start_cost_usd

    # Mark turn as ended so next PreToolUse emits session_start
    state.turn_ended = True
    save_state(state)

    # Write audit log (per-turn metrics)
    audit.log_session_end(
        session_id=session_id,
        total_steps=turn_steps,
        total_cost_usd=turn_cost_usd,
        duration_seconds=duration_seconds,
        agent_id=agent_id,
    )

    # Queue session_end trace event and flush all queued events as batch (FRD-HK-009)
    api_key = config.resolved_api_key
    if api_key:
        agent_name = getattr(config.backend, "agent_name", None) or DEFAULT_AGENT_NAME
        session_end_event = create_trace_event(
            "session_end",
            session_id,
            agent_id=agent_id,
            agent_name=agent_name,
            duration_ms=int(duration_seconds * 1000),
            cost_usd=turn_cost_usd,
            step_number=state.step_count,
            accumulated_cost_usd=state.accumulated_cost_usd,
            metadata={
                "turn_steps": turn_steps,
                "turn_cost_usd": turn_cost_usd,
                "cumulative_steps": state.step_count,
                "cumulative_cost_usd": state.accumulated_cost_usd,
                "duration_seconds": duration_seconds,
            },
        )
        enqueue_event(session_id, session_end_event)

        # Flush all queued events to backend in a single batch
        flush_event_queue(
            session_id=session_id,
            api_key=api_key,
            api_url=config.resolved_api_url,
            circuit=state.circuit_breaker,
        )
        # Persist updated circuit breaker state
        save_state(state)

    # Cleanup stale sessions (FRD-HK-002)
    cleanup_stale_sessions()
