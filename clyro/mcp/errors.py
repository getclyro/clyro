# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Error Formatter
# Implements FRD-011

"""
Build JSON-RPC 2.0 error responses for blocked tool calls.

All policy blocks use code ``-32600`` (Invalid Request) per FRD-011.
Each error type populates a structured ``data`` envelope so the MCP host
can programmatically distinguish the reason.
"""

from __future__ import annotations

import json
from typing import Any

from clyro.constants import ISSUE_TRACKER_URL

_ISSUE_TRACKER = ISSUE_TRACKER_URL


def _build_reason(block_type: str, details: dict[str, Any]) -> str:
    """Build a human-readable reason string including key details."""
    if block_type == "policy_violation":
        rule_name = details.get("rule_name", "")
        tool_name = details.get("tool_name", "")
        policy_id = details.get("policy_id", "")
        parts = ["Tool call blocked by policy rule"]
        if rule_name:
            parts.append(f"rule_name={rule_name}")
        if policy_id:
            parts.append(f"policy_id={policy_id}")
        if tool_name:
            parts.append(f"tool={tool_name}")
        parameter = details.get("parameter", "")
        operator = details.get("operator", "")
        if parameter and operator:
            expected = details.get("expected", "")
            actual = details.get("actual_value", details.get("actual", ""))
            parts.append(f"({parameter} {operator} {expected!r}, actual={actual!r})")
        return " ".join(parts)

    if block_type == "budget_exceeded":
        accumulated = details.get("accumulated_cost_usd", "?")
        max_cost = details.get("max_cost_usd", "?")
        return f"Estimated cost budget exceeded (accumulated=${accumulated}, limit=${max_cost})"

    if block_type == "step_limit_exceeded":
        step_count = details.get("step_count", "?")
        max_steps = details.get("max_steps", "?")
        return f"Maximum tool call steps reached (step={step_count}, limit={max_steps})"

    if block_type == "loop_detected":
        iterations = details.get("iterations", "?")
        tool_name = details.get("tool_name", "")
        reason = f"Repeated tool call pattern detected (iterations={iterations})"
        if tool_name:
            reason += f" tool={tool_name}"
        return reason

    return block_type


def format_error(
    request_id: str | int | None,
    block_type: str,
    details: dict[str, Any],
) -> str:
    """
    Build a newline-terminated JSON-RPC 2.0 error response.

    Args:
        request_id: Original JSON-RPC ``id`` from the request.
        block_type: One of ``loop_detected``, ``step_limit_exceeded``,
            ``budget_exceeded``, ``policy_violation``.
        details: Structured details for the ``data`` envelope.

    Returns:
        Newline-terminated JSON string ready for stdout.
    """
    # Build human-readable reason with key details so the LLM/user
    # can see exactly what triggered the block without parsing data.details.
    reason = _build_reason(block_type, details)

    error_response: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32600,
            "message": f"ClyroPolicy: {reason}",
            "data": {
                "type": block_type,
                "details": details,
                "issue_tracker": _ISSUE_TRACKER,
            },
        },
    }

    return json.dumps(error_response, default=str) + "\n"
