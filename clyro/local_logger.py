# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Local Terminal Logger
# Implements FRD-SOF-006, FRD-SOF-007, FRD-SOF-008, FRD-SOF-011, NFR-003

"""
stderr-based terminal output for local mode.

All output is:
- Prefixed with ``[clyro]`` for grep filtering
- Written to stderr only (stdout is never touched)
- Suppressed by ``CLYRO_QUIET=true`` (NFR-003)
- Fail-safe: logging failures never crash the agent
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import clyro
from clyro.constants import APP_POLICIES_URL, APP_URL, DISCORD_URL, DOCS_URL

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.session import Session
    from clyro.trace import TraceEvent

# Module-level flag: welcome printed once per process (FRD-SOF-008)
_welcome_shown: bool = False


def _is_quiet() -> bool:
    """Check if CLYRO_QUIET suppresses output.  Implements NFR-003."""
    return os.environ.get("CLYRO_QUIET", "").lower() in ("true", "1", "yes")


def _write_stderr(text: str) -> None:
    """Write to stderr with silent failure on closed fd."""
    try:
        print(text, file=sys.stderr)
    except Exception:
        pass  # stderr closed — silently ignore (FRD-SOF-006)


class LocalTerminalLogger:
    """
    Terminal logger for local mode.

    Implements:
    - FRD-SOF-006: per-step policy evaluation logs
    - FRD-SOF-007: session-end governance summary
    - FRD-SOF-008: first-run welcome message
    - FRD-SOF-011: policy violation context logging
    - NFR-003: CLYRO_QUIET suppression
    """

    def __init__(self, config: ClyroConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # FRD-SOF-008: First-run welcome message
    # ------------------------------------------------------------------

    def print_welcome(self) -> None:
        """Print welcome message once per process."""
        global _welcome_shown
        if _welcome_shown or _is_quiet():
            return

        _welcome_shown = True
        version = getattr(clyro, "__version__", "0.0.0")
        _write_stderr(f"[clyro] v{version} \u2014 Runtime governance for AI agents")
        _write_stderr(f"[clyro] Docs: {DOCS_URL} | Community: {DISCORD_URL}")

    # ------------------------------------------------------------------
    # FRD-SOF-006: Per-step policy evaluation logs
    # ------------------------------------------------------------------

    def log_event(self, event: TraceEvent) -> None:
        """Log a trace event to stderr in human-readable form."""
        if _is_quiet():
            return

        from clyro.trace import EventType

        if event.event_type == EventType.POLICY_CHECK:
            self._log_policy_event(event)

    def _log_policy_event(self, event: TraceEvent) -> None:
        """Format and log a POLICY_CHECK event."""
        meta = event.metadata or {}
        decision = meta.get("decision", "unknown").upper()
        action_type = meta.get("action_type", "unknown")
        evaluated = meta.get("evaluated_rules", 0)
        rule_results = meta.get("rule_results", [])

        violations = sum(1 for r in rule_results if r.get("outcome") == "triggered")

        if decision == "BLOCK":
            rule_name = meta.get("rule_name", "unknown")
            _write_stderr(f'[clyro] Policy BLOCK \u2014 {action_type}: rule "{rule_name}" violated')
        else:
            _write_stderr(
                f"[clyro] Policy ALLOW \u2014 {action_type}: "
                f"{evaluated} rules evaluated, {violations} violations"
            )

    # ------------------------------------------------------------------
    # FRD-SOF-011: Policy violation context logging
    # ------------------------------------------------------------------

    def log_violation(
        self,
        action_type: str,
        violation_details: dict[str, Any],
    ) -> None:
        """Print detailed violation context before PolicyViolationError is raised."""
        if _is_quiet():
            return

        rule_name = violation_details.get("rule_name", "unknown")
        parameter = violation_details.get("parameter", "unknown")
        operator = violation_details.get("operator", "unknown")
        expected = violation_details.get("expected", "unknown")
        actual = violation_details.get("actual", "unknown")

        _write_stderr(f"[clyro] POLICY VIOLATION \u2014 {action_type} blocked")
        _write_stderr(f'  Rule:      "{rule_name}"')
        _write_stderr(f"  Parameter: {parameter}")
        _write_stderr(f"  Operator:  {operator}")
        _write_stderr(f'  Expected:  "{expected}"')
        _write_stderr(f'  Actual:    "{actual}"')

        # Cloud CTA only in local mode (FRD-SOF-011)
        if self._config.mode == "local":
            _write_stderr(f"  Configure team-wide policies \u2192 {APP_POLICIES_URL}")

    # ------------------------------------------------------------------
    # FRD-SOF-007: Session-end governance summary
    # ------------------------------------------------------------------

    def print_session_summary(self, session: Session) -> None:
        """Print the governance summary after session.end()."""
        if _is_quiet():
            return

        try:
            steps = session.step_number
            cost = session.cumulative_cost
            mode = self._config.mode or "local"

            # Count violations from recorded events
            from clyro.trace import EventType

            violations: list[str] = []
            for ev in session.events:
                if ev.event_type == EventType.POLICY_CHECK:
                    meta = ev.metadata or {}
                    if meta.get("decision") == "block":
                        rn = meta.get("rule_name", "unknown")
                        at = meta.get("action_type", "unknown")
                        violations.append(f'{at}: "{rn}"')

            # Count execution control triggers
            controls_triggered: list[str] = []
            for ev in session.events:
                if ev.event_name in (
                    "execution_control",
                    "step_limit_exceeded",
                    "cost_limit_exceeded",
                    "loop_detected",
                ):
                    controls_triggered.append(ev.event_name)

            violation_text = f"{len(violations)} ({', '.join(violations)})" if violations else "0"
            controls_text = (
                ", ".join(controls_triggered) if controls_triggered else "none triggered"
            )

            bar = "\u2501" * 41
            _write_stderr(bar)
            _write_stderr(" clyro governance summary")
            _write_stderr(bar)
            _write_stderr(f" Steps:      {steps}")
            _write_stderr(f" Cost:       ${cost:.3f}")
            _write_stderr(f" Violations: {violation_text}")
            _write_stderr(f" Controls:   {controls_text}")
            _write_stderr(f" Mode:       {mode}")
            _write_stderr(bar)

            # CTA based on mode (FRD-SOF-007)
            if mode == "local":
                _write_stderr(f" View full replay \u2192 {APP_URL}")
                _write_stderr(" Connect: set CLYRO_API_KEY to enable cloud")
            else:
                sid = str(session.session_id)
                _write_stderr(f" View full replay \u2192 {APP_URL}/sessions/{sid}")

            _write_stderr(bar)

        except Exception:
            pass  # Fail-open: summary generation must never crash the agent


def reset_welcome_flag() -> None:
    """Reset the welcome flag (for testing)."""
    global _welcome_shown
    _welcome_shown = False
