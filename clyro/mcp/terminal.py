# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Terminal Output
# These are internal requirement codes used during development (see CONTRIBUTING.md)

"""
stderr-based terminal output for MCP wrapper sessions.

All output is:
- Prefixed with ``[clyro-mcp]`` for grep filtering
- Written to stderr only (stdout is reserved for JSON-RPC)
- Suppressed by ``CLYRO_QUIET=true``
- Fail-safe: output failures never crash the wrapper or block tool calls

Architecture note:
    This module mirrors ``clyro.local_logger`` (SDK) but is tailored to the
    MCP wrapper's data model (``McpSession``, ``AuditLogger`` stats, no
    ``Session.events`` list).  The MCP wrapper runs as a stdio proxy —
    stdout is the JSON-RPC channel to the host and must NEVER be written
    to by governance code.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import clyro
from clyro.constants import APP_URL, ISSUE_TRACKER_URL

_ISSUE_TRACKER = ISSUE_TRACKER_URL


# ---------------------------------------------------------------------------
# CLYRO_QUIET gate — shared logic with local_logger.py
# ---------------------------------------------------------------------------

def is_quiet() -> bool:
    """Check if ``CLYRO_QUIET`` suppresses terminal output.

    Recognises ``true``, ``1``, ``yes`` (case-insensitive).
    Exported so other MCP modules can check before writing to stderr.
    """
    return os.environ.get("CLYRO_QUIET", "").lower() in ("true", "1", "yes")


def write_stderr(text: str) -> None:
    """Write a line to stderr with silent failure on closed fd.

    Governance output must never interfere with the JSON-RPC channel
    (stdout), so all human-readable output goes to stderr.
    """
    try:
        print(text, file=sys.stderr)
    except Exception:
        pass  # stderr closed or broken — silently ignore


# ---------------------------------------------------------------------------
# Session-end governance summary
# ---------------------------------------------------------------------------

class McpTerminalLogger:
    """
    Terminal logger for MCP wrapper sessions.

    Provides:
    - Session-end governance summary (steps, cost, violations, mode)
    - Error context enrichment (issue tracker URL)

    All methods are no-ops when ``CLYRO_QUIET`` is set.
    All methods are fail-safe — exceptions are caught and silenced.
    """

    def __init__(self, *, is_backend_enabled: bool = False) -> None:
        self._is_backend_enabled = is_backend_enabled

    # ------------------------------------------------------------------
    # Session-end governance summary
    # ------------------------------------------------------------------

    def print_session_summary(
        self,
        *,
        steps: int,
        cost_usd: float,
        violations: list[dict[str, Any]] | None = None,
        controls_triggered: list[str] | None = None,
    ) -> None:
        """Print a governance summary to stderr after session ends.

        Args:
            steps: Total tool calls evaluated.
            cost_usd: Accumulated estimated cost.
            violations: List of violation dicts with ``block_type`` and
                ``tool_name`` keys (from audit log).
            controls_triggered: List of control names that fired
                (e.g. ``["step_limit_exceeded", "budget_exceeded"]``).
        """
        if is_quiet():
            return

        try:
            violations = violations or []
            controls_triggered = controls_triggered or []
            mode = "cloud" if self._is_backend_enabled else "local"

            # Format violations
            if violations:
                violation_parts = []
                for v in violations:
                    block_type = v.get("block_type", "unknown")
                    tool_name = v.get("tool_name", "")
                    label = f'{block_type}: "{tool_name}"' if tool_name else block_type
                    violation_parts.append(label)
                violation_text = f"{len(violations)} ({', '.join(violation_parts)})"
            else:
                violation_text = "0"

            controls_text = (
                ", ".join(controls_triggered) if controls_triggered else "none triggered"
            )

            bar = "\u2501" * 41
            write_stderr(bar)
            write_stderr(" clyro-mcp governance summary")
            write_stderr(bar)
            write_stderr(f" Steps:      {steps}")
            write_stderr(f" Cost:       ${cost_usd:.6f}")
            write_stderr(f" Violations: {violation_text}")
            write_stderr(f" Controls:   {controls_text}")
            write_stderr(f" Mode:       {mode}")
            write_stderr(bar)

            if mode == "local":
                write_stderr(f" View full replay \u2192 {APP_URL}")
                write_stderr(" Connect: set CLYRO_API_KEY to enable cloud")
            else:
                write_stderr(f" Traces synced to cloud \u2192 {APP_URL}")

            write_stderr(bar)

        except Exception:
            pass  # Fail-safe: summary must never crash the wrapper

    # ------------------------------------------------------------------
    # Error context enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def format_error_with_context(error: Exception) -> str:
        """Append issue tracker URL to an error message.

        Returns the enriched string — does NOT modify the exception.
        """
        base = str(error)
        if _ISSUE_TRACKER in base:
            return base  # Already enriched (e.g. ClyroError subclass)
        return f"{base}\n  Report at {_ISSUE_TRACKER}"
