# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Global dry-run (monitor) mode helpers (A10)
# Implements FRD-001, FRD-002, FRD-008, FRD-011, FRD-012, FRD-017, FRD-018

"""
Shared helpers for the global dry-run (monitor) mode.

Dry-run changes exactly one thing: whether Clyro *acts* on a decision it has
already computed. Every check still runs and every decision is still produced;
only the final raise / block / deny is gated, and a distinct ``would_block``
marker event is recorded instead.

This module is the single source of truth for the pieces that must be identical
across all three surfaces (SDK ``wrap()``, the MCP wrapper, Claude Code hooks),
per FRD-NFR-002:

* the mode resolver + fail-safe env override (FRD-001/002/NFR-005),
* the required literal ``CLYRO-DRYRUN`` log token (FRD-011),
* the one-time start banner (FRD-012),
* the ``would_block`` trace-event builder (FRD-008/017/018).

It deliberately imports only ``structlog`` and :mod:`clyro.trace` so every
low-level module (config, session, policy, adapters, mcp, hooks) can use it
without a circular import.
"""

from __future__ import annotations

import os
from typing import Any, Literal
from uuid import UUID

import structlog

from clyro.trace import EventType, Framework, TraceEvent

logger = structlog.get_logger(__name__)

# The single, greppable token every surface stamps on a would-block line so it
# is distinguishable from a real enforce-mode block by exact-string match
# (FRD-011). Do not change without updating the FRD and every conformance test.
DRY_RUN_TOKEN = "CLYRO-DRYRUN"

EnforcementMode = Literal["enforce", "dry_run"]

# The shared, ergonomic env lever (SC5: one env var). Only these recognised
# truthy values enable dry_run; every other *present* value (falsy or malformed)
# resolves to enforce — dry_run is never entered by accident (FRD-NFR-005).
_DRY_RUN_TRUTHY = frozenset({"true", "1", "yes", "on", "dry_run"})

CheckType = Literal["step", "cost", "loop", "policy"]
WouldBeOutcome = Literal["block", "require_approval"]


class WouldBlockLatch:
    """Bounded per-reason latch so a repeating would-block records **once**.

    Implements FRD-022 (extended to the policy check type). The step/cost/loop
    predicates are *sticky*, so ``Session`` latches them per reason. Policy is
    **not** sticky — every action is a genuinely new decision — so an agent that
    trips one rule on 10k actions would emit 10k near-identical markers, burning
    the org's trace quota and bloating the dry-run report for no added signal.

    This latches policy would-blocks per ``(session, rule, action_type)``: the
    first occurrence is recorded, later ones are counted but not re-emitted. The
    report answers "which rules would fire" from the events, and "how often" from
    :meth:`occurrences`.

    Memory is bounded: a long-lived evaluator serves many sessions, so the key
    map is cleared once it exceeds ``_MAX_KEYS`` (at worst a rule re-emits once
    more after a reset — never unbounded growth).
    """

    _MAX_KEYS = 10_000

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    @staticmethod
    def policy_key(
        session_id: object,
        rule_id: str | None,
        action_type: str,
        would_be_outcome: str = "block",
    ) -> str:
        """Build the latch key for a policy would-block.

        The key is the full **reason**: session + rule + action type + would-be
        outcome. The outcome is included because "would block" and "would require
        approval" are genuinely different findings — collapsing them would
        silently drop the second one from the report.
        """
        return f"{session_id}:policy:{rule_id or 'unknown'}:{action_type}:{would_be_outcome}"

    def record(self, key: str) -> bool:
        """Count an occurrence; return True only for the FIRST one (emit it)."""
        if len(self._counts) >= self._MAX_KEYS and key not in self._counts:
            # Bound memory across a long-lived evaluator's many sessions.
            self._counts.clear()
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return count == 1

    def occurrences(self, key: str) -> int:
        """How many times this reason has fired (including suppressed repeats)."""
        return self._counts.get(key, 0)

    @property
    def suppressed_total(self) -> int:
        """Total would-blocks suppressed as repeats (emitted = one per key)."""
        return sum(c - 1 for c in self._counts.values() if c > 1)


def normalize_enforcement_mode(value: object) -> EnforcementMode:
    """Normalize an arbitrary mode value to a resolved mode. Implements FRD-001.

    Any value other than the exact string ``"dry_run"`` (case/whitespace
    insensitive) resolves to ``"enforce"`` — an unrecognised or malformed value
    MUST NOT silently resolve to ``dry_run``.
    """
    if isinstance(value, str) and value.strip().lower() == "dry_run":
        return "dry_run"
    return "enforce"


def resolve_dry_run_env() -> bool | None:
    """Resolve the ``CLYRO_DRY_RUN`` override. Implements FRD-002 / FRD-NFR-005.

    Returns:
        ``True``  — an explicit recognised truthy value (enter dry_run).
        ``False`` — any other *present* value (falsy or malformed) → fail-safe
                    enforce; a well-formed override to enforce also lands here.
        ``None``  — the variable is unset (no override → fall through to config).
    """
    raw = os.getenv("CLYRO_DRY_RUN")
    if raw is None:
        return None
    return raw.strip().lower() in _DRY_RUN_TRUTHY


def resolve_is_dry_run(config_is_dry_run: bool) -> bool:
    """Combine the env override with the in-code config value. Implements FRD-002.

    Precedence: a well-formed ``CLYRO_DRY_RUN`` override wins over the in-code
    config value; absent the override, the config value is used. The result is
    never ``dry_run`` by accident — see :func:`resolve_dry_run_env`.
    """
    override = resolve_dry_run_env()
    if override is not None:
        return override
    return config_is_dry_run


def log_mode_banner(surface: str, is_dry_run: bool) -> None:
    """Announce the resolved mode exactly once at start. Implements FRD-012.

    Only dry_run is announced (loudly, at WARNING) — the default enforce path is
    silent so it does not add noise (scope Open Q3). ``surface`` is one of
    ``"sdk"`` / ``"mcp"`` / ``"hooks"``.
    """
    if not is_dry_run:
        return
    logger.warning(
        f"{DRY_RUN_TOKEN} active — enforcement suppressed (mode=dry_run)",
        surface=surface,
    )


def log_would_block(
    check_type: CheckType,
    action: str,
    would_be_outcome: WouldBeOutcome = "block",
    rule_id: str | None = None,
) -> None:
    """Write the required would-block line carrying the literal token. Implements FRD-011.

    A fixed, greppable one-line summary (O-5). The exact-string ``CLYRO-DRYRUN``
    prefix is the distinguisher a test asserts on, and an enforce-mode block line
    never carries it.
    """
    logger.warning(
        f"{DRY_RUN_TOKEN} would-have-blocked",
        check=check_type,
        action=action,
        would_be=would_be_outcome,
        rule=rule_id,
    )


def build_would_block_event(
    *,
    session_id: UUID,
    check_type: CheckType,
    action: str,
    would_be_outcome: WouldBeOutcome = "block",
    agent_id: UUID | None = None,
    parent_event_id: UUID | None = None,
    rule_id: str | None = None,
    rule_name: str | None = None,
    framework: Framework = Framework.GENERIC,
    input_data: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> TraceEvent:
    """Build the distinct non-enforced marker event. Implements FRD-008/017/018.

    The event carries, at minimum: the check type, the action/decision identity,
    the would-be outcome, and (for policy) the offending rule reference —
    all under ``metadata["would_block"]`` — plus the session-level
    ``metadata["dry_run"]`` marker the backend filters on (FRD-018).

    Invariant (FRD-017): a would-block carries **no** ``error_type`` and is
    ``event_type=would_block`` (never ``policy_check`` or ``error``), so no
    enforced aggregation counts it.
    """
    metadata: dict[str, Any] = {
        # FRD-018 session marker — stamped on every dry-run event so drift / ARI /
        # cost can exclude the whole session, including all-allow sessions.
        "dry_run": True,
        # FRD-008 marker fields.
        "would_block": {
            "check_type": check_type,
            "would_be_outcome": would_be_outcome,
            "rule_id": rule_id,
            "rule_name": rule_name,
        },
    }
    if extra_metadata:
        # Never let a caller override the two load-bearing keys above.
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)

    return TraceEvent(
        session_id=session_id,
        agent_id=agent_id,
        parent_event_id=parent_event_id,
        event_type=EventType.WOULD_BLOCK,
        event_name=f"would_block:{check_type}",
        framework=framework,
        input_data=input_data or {"action": action},
        output_data={"would_be_outcome": would_be_outcome},
        metadata=metadata,
        # FRD-017 invariant: no error_type on a would-block.
    )
