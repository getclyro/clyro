# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Audit Logger
# Implements FRD-HK-010

"""Append-only JSONL audit logging for Claude Code hooks."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clyro.audit import BaseAuditLogger
from clyro.redaction import redact_params  # noqa: F401 — re-exported for evaluator/tracer/tests

from .constants import DEFAULT_REDACT_PARAMETERS


class AuditLogger(BaseAuditLogger):
    """Hooks-specific audit logger.

    FRD-HK-010: Writes one line per event. Fail-open on write errors.
    Extends :class:`clyro.audit.BaseAuditLogger` with hooks event methods.
    """

    def __init__(
        self,
        log_path: str | Path = "~/.clyro/hooks/audit.jsonl",
        redact_patterns: list[str] | None = None,
    ):
        super().__init__(log_path)
        self._patterns = redact_patterns or DEFAULT_REDACT_PARAMETERS

    def _redact(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Redact sensitive parameters before writing."""
        return redact_params(params, self._patterns)

    def log_pre_tool_use(
        self,
        session_id: str,
        tool_name: str,
        decision: str,
        step_number: int,
        accumulated_cost_usd: float,
        tool_input: dict[str, Any] | None = None,
        reason: str | None = None,
        rule_results: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Log a PreToolUse evaluation result."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "pre_tool_use",
            "session_id": session_id,
            "tool_name": tool_name,
            "decision": decision,
            "step_number": step_number,
            "accumulated_cost_usd": accumulated_cost_usd,
            "tool_input": self._redact(tool_input),
        }
        if agent_id:
            entry["agent_id"] = agent_id
        if reason:
            entry["reason"] = reason
        if rule_results:
            entry["rule_results"] = rule_results
        self._write(entry)

    def log_post_tool_use(
        self,
        session_id: str,
        tool_name: str,
        step_number: int,
        accumulated_cost_usd: float,
        tool_input: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Log a PostToolUse trace event."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "tool_call_observe",
            "session_id": session_id,
            "tool_name": tool_name,
            "step_number": step_number,
            "accumulated_cost_usd": accumulated_cost_usd,
            "tool_input": self._redact(tool_input),
        }
        if agent_id:
            entry["agent_id"] = agent_id
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        self._write(entry)

    def log_policy_check(
        self,
        session_id: str,
        tool_name: str,
        decision: str,
        step_number: int,
        accumulated_cost_usd: float,
        tool_input: dict[str, Any] | None = None,
        rule_results: list[dict[str, Any]] | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Log a policy check (governance decision) event."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "policy_check",
            "session_id": session_id,
            "tool_name": tool_name,
            "decision": decision,
            "step_number": step_number,
            "accumulated_cost_usd": accumulated_cost_usd,
            "tool_input": self._redact(tool_input),
        }
        if agent_id:
            entry["agent_id"] = agent_id
        if rule_results:
            entry["rule_results"] = rule_results
        self._write(entry)

    def log_error(
        self,
        session_id: str,
        tool_name: str,
        error_type: str,
        error_message: str,
        step_number: int,
        accumulated_cost_usd: float,
        agent_id: str | None = None,
    ) -> None:
        """Log an error event (blocked call details)."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "error",
            "session_id": session_id,
            "tool_name": tool_name,
            "error_type": error_type,
            "error_message": error_message,
            "step_number": step_number,
            "accumulated_cost_usd": accumulated_cost_usd,
        }
        if agent_id:
            entry["agent_id"] = agent_id
        self._write(entry)

    def log_session_start(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> None:
        """Log a session-start lifecycle event."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "session_start",
            "session_id": session_id,
        }
        if agent_id:
            entry["agent_id"] = agent_id
        self._write(entry)

    def log_session_end(
        self,
        session_id: str,
        total_steps: int,
        total_cost_usd: float,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        """Log a session-end lifecycle event."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "session_end",
            "session_id": session_id,
            "total_steps": total_steps,
            "total_cost_usd": total_cost_usd,
            "duration_seconds": duration_seconds,
        }
        if agent_id:
            entry["agent_id"] = agent_id
        self._write(entry)
