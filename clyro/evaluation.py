# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Shared Evaluation Utilities
# Used by hooks (clyro.hooks.evaluator) and MCP wrapper (clyro.mcp.prevention)

"""
Shared functions for the prevention/evaluation pipeline.

Both hooks and MCP inject session context into tool arguments before
policy evaluation so that rules can reference session-level fields
(cost, step number, tool name, etc.).  This module centralises the
enrichment logic to prevent drift between the two implementations.
"""

from __future__ import annotations

from typing import Any

# Prefix used by hooks for synthetic parameters (FRD-HK-006).
CLYRO_PARAM_PREFIX = "_clyro_"


def enrich_tool_input(
    tool_input: dict[str, Any] | None,
    tool_name: str,
    session_id: str,
    step_number: int,
    accumulated_cost_usd: float,
    agent_id: str | None = None,
    *,
    use_prefix: bool = True,
) -> dict[str, Any]:
    """Inject session context into tool arguments for policy evaluation.

    Args:
        tool_input: Original tool arguments (may be ``None``).
        tool_name: Name of the tool being called.
        session_id: Current session identifier.
        step_number: Current step count.
        accumulated_cost_usd: Accumulated cost so far.
        agent_id: Optional agent identifier.
        use_prefix: If ``True`` (default, hooks), keys are prefixed with
            ``_clyro_`` (e.g. ``_clyro_tool_name``).  If ``False`` (MCP),
            plain keys are used via ``setdefault`` to avoid overwriting
            user-supplied parameters.

    Returns:
        A **new** dict containing the original parameters plus the
        injected session context fields.
    """
    enriched = dict(tool_input or {})

    if use_prefix:
        # Hooks pattern: always set with prefix (no collision risk)
        enriched[f"{CLYRO_PARAM_PREFIX}tool_name"] = tool_name
        enriched[f"{CLYRO_PARAM_PREFIX}session_id"] = session_id
        enriched[f"{CLYRO_PARAM_PREFIX}step_number"] = step_number
        enriched[f"{CLYRO_PARAM_PREFIX}cost"] = accumulated_cost_usd
        if agent_id:
            enriched[f"{CLYRO_PARAM_PREFIX}agent_id"] = agent_id
    else:
        # MCP pattern: use setdefault to avoid overwriting user params
        enriched.setdefault("tool_name", tool_name)
        enriched.setdefault("session_id", session_id)
        enriched.setdefault("step_number", step_number)
        enriched.setdefault("cost", accumulated_cost_usd)

    return enriched
