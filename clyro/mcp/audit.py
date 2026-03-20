# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Audit Logger
# Implements FRD-008

"""
JSONL audit logger for every intercepted ``tools/call`` and lifecycle event.

Design invariants:
- Audit write failure NEVER blocks tool call forwarding (fail-open).
- Audit log directory is created on first write if absent.
- Audit file permissions are ``0o600`` (owner read/write only — NFR-005).
- Sensitive parameters matching ``redact_parameters`` globs are replaced
  with ``"[REDACTED]"`` before writing.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from clyro.audit import BaseAuditLogger
from clyro.config import AuditConfig
from clyro.mcp.log import get_logger
from clyro.redaction import redact_dict_deepcopy

logger = get_logger(__name__)


class AuditLogger(BaseAuditLogger):
    """
    MCP-specific audit logger with dual-mode emission (JSONL + backend sync).

    Extends :class:`clyro.audit.BaseAuditLogger` with MCP-specific methods:
    - Tool call logging with backend trace event emission
    - Lifecycle events (session_start, session_end, server_exited)
    - Deferred session_start to avoid empty discovery sessions
    - Policy violation reporting

    Args:
        config: Audit-specific configuration.
        session_id: Session UUID (included in every entry).
    """

    def __init__(self, config: AuditConfig, session_id: UUID) -> None:
        log_path = Path(os.path.expanduser(config.log_path))
        super().__init__(log_path)
        self._redact_patterns = config.redact_parameters
        self._session_id = str(session_id)
        self._sync_manager: Any = None  # FRD-015: optional BackendSyncManager
        self._trace_factory: Any = None  # FRD-015: optional TraceEventFactory
        self._pending_session_start: dict[str, Any] | None = None  # Deferred until first tool call
        self._backend_session_started = False  # True once session_start sent to backend
        self._violation_reporter: Any = None  # FRD-006: callable to enqueue violations
        self._agent_id: str | None = None  # FRD-006: agent UUID string
        self._session_ended = False  # Prevent duplicate session_end events
        self._pending_act_events: dict[
            str | int, tuple[str, int]
        ] = {}  # request_id → (act_event_id, step_number)
        # In-memory accumulators for session summary (terminal output)
        self._violations: list[dict[str, Any]] = []
        self._controls_triggered: list[str] = []

    def set_backend(self, sync_manager: Any, trace_factory: Any) -> None:
        """Attach backend sync components for dual-mode emission (FRD-015)."""
        self._sync_manager = sync_manager
        self._trace_factory = trace_factory

    def set_violation_reporter(self, reporter: Any, agent_id: str) -> None:
        """Attach violation reporter for backend persistence (FRD-006)."""
        self._violation_reporter = reporter
        self._agent_id = agent_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_tool_call(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None,
        decision: str,
        step_number: int,
        accumulated_cost_usd: float,
        block_reason: str | None = None,
        block_details: dict[str, Any] | None = None,
        duration_ms: int = 0,
        rule_results: list[dict[str, Any]] | None = None,
        request_id: str | int | None = None,
    ) -> None:
        """Write a ``tool_call`` audit entry and emit trace event if backend enabled."""
        redacted_params = self._redact(parameters) if parameters else {}
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "event": "tool_call",
            "tool_name": tool_name,
            "parameters": redacted_params,
            "decision": decision,
            "block_reason": block_reason,
            "block_details": block_details,
            "duration_ms": duration_ms,
            "step_number": step_number,
            "accumulated_cost_usd": round(accumulated_cost_usd, 6),
            "cost_estimated": True,
        }
        self._write(entry)

        # Track violations and controls for session summary
        if decision == "blocked" and block_reason:
            summary_entry = {"block_type": block_reason, "tool_name": tool_name}
            self._violations.append(summary_entry)
            if block_reason in (
                "step_limit_exceeded", "budget_exceeded", "loop_detected",
            ):
                if block_reason not in self._controls_triggered:
                    self._controls_triggered.append(block_reason)

        # FRD-015: dual-mode emission — also enqueue trace event for backend sync
        # Wrapped in try/except: backend sync failure MUST NOT block forwarding (NFR-007)
        if self._sync_manager and self._trace_factory:
            try:
                self._flush_pending_session_start()

                act_event = self._trace_factory.tool_call_act(
                    tool_name=tool_name,
                    params=redacted_params,
                    step_number=step_number,
                    duration_ms=duration_ms,
                )
                act_event_id = act_event["event_id"]

                trace_decision = "block" if decision == "blocked" else "allow"
                policy_event = self._trace_factory.policy_check(
                    tool_name=tool_name,
                    params=redacted_params,
                    duration_ms=duration_ms,
                    decision=trace_decision,
                    rule_results=rule_results,
                    parent_event_id=act_event_id,
                )
                self._sync_manager.enqueue(policy_event)

                if decision == "blocked":
                    trace_event = self._trace_factory.blocked_call(
                        tool_name=tool_name,
                        block_type=block_reason or "unknown",
                        block_message=f"Blocked by {block_reason}",
                        block_details=block_details,
                    )
                    self._sync_manager.enqueue(trace_event)
                else:
                    if request_id is not None:
                        self._pending_act_events[request_id] = (act_event_id, step_number)
                    self._sync_manager.enqueue(act_event)
            except Exception as e:
                logger.debug("trace_emission_failed", error=str(e), fail_open=True)

        # FRD-006: report policy violations to backend for persistence
        if (
            self._violation_reporter
            and decision == "blocked"
            and block_reason == "policy_violation"
            and block_details
        ):
            try:
                self._violation_reporter(
                    {
                        "agent_id": self._agent_id,
                        "policy_id": block_details.get("policy_id"),
                        "action_type": tool_name,
                        "rule_id": block_details.get("rule_name", "unknown"),
                        "rule_name": block_details.get("rule_name", "unknown"),
                        "operator": block_details.get("operator", ""),
                        "expected_value": json.dumps(block_details.get("expected"), default=str),
                        "actual_value": json.dumps(block_details.get("actual"), default=str),
                        "decision": "block",
                        "message": (
                            f"Blocked by {block_reason}: {block_details.get('rule_name', '')}"
                        ),
                        "parameters_hash": hashlib.sha256(
                            json.dumps(redacted_params, sort_keys=True, default=str).encode()
                        ).hexdigest(),
                        "session_id": self._session_id,
                        "step_number": step_number,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
            except Exception as e:
                logger.debug("violation_reporting_failed", error=str(e), fail_open=True)

    def log_tool_call_response(
        self,
        tool_name: str,
        request_id: str | int,
        call_cost_usd: float,
        accumulated_cost_usd: float,
        duration_ms: int = 0,
        response_content: str | None = None,
    ) -> None:
        """Write a ``tool_call_response`` audit entry with actual cost after server responds."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "event": "tool_call_response",
            "tool_name": tool_name,
            "request_id": request_id,
            "call_cost_usd": round(call_cost_usd, 8),
            "accumulated_cost_usd": round(accumulated_cost_usd, 8),
            "cost_estimated": True,
        }
        if duration_ms:
            entry["duration_ms"] = duration_ms
        self._write(entry)

        # FRD-015: dual-mode emission — enqueue observe-stage trace event for backend sync
        if self._sync_manager and self._trace_factory:
            try:
                act_context = self._pending_act_events.pop(request_id, None)
                parent_eid = act_context[0] if act_context else None
                act_step = act_context[1] if act_context else None
                trace_event = self._trace_factory.tool_call_observe(
                    tool_name=tool_name,
                    response_content=response_content,
                    cost_usd=call_cost_usd,
                    duration_ms=duration_ms,
                    parent_event_id=parent_eid,
                    step_number=act_step,
                )
                self._sync_manager.enqueue(trace_event)
            except Exception as e:
                logger.debug("trace_emission_failed", error=str(e), fail_open=True)

    def log_lifecycle(
        self,
        event: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a lifecycle event (``session_start``, ``session_end``, ``server_exited``)."""
        # Prevent duplicate session_end (e.g., signal handler + finally block)
        if event == "session_end":
            if self._session_ended:
                return
            self._session_ended = True
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "event": event,
        }
        if extra:
            entry.update(extra)
        self._write(entry)

        # FRD-015: dual-mode emission for lifecycle events
        if self._sync_manager and self._trace_factory:
            try:
                if event == "session_start":
                    self._pending_session_start = self._trace_factory.session_start()
                elif event == "session_end":
                    if self._backend_session_started:
                        trace_event = self._trace_factory.session_end()
                        self._sync_manager.enqueue(trace_event)
                elif event == "server_exited":
                    if self._backend_session_started:
                        trace_event = self._trace_factory.create_trace_event(
                            "session_end",
                            None,
                            metadata={"reason": "server_exited", **(extra or {})},
                        )
                        self._sync_manager.enqueue(trace_event)
            except Exception as e:
                logger.debug("trace_emission_failed", event=event, error=str(e), fail_open=True)

    def log_parse_error(self, raw_bytes: bytes) -> None:
        """Write a ``parse_error`` entry for malformed JSON from the host."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "event": "parse_error",
            "raw_preview": raw_bytes[:200].decode("utf-8", errors="replace"),
        }
        self._write(entry)

    def _flush_pending_session_start(self) -> None:
        """Send deferred session_start to backend on first tool activity."""
        if self._pending_session_start and self._sync_manager:
            self._sync_manager.enqueue(self._pending_session_start)
            self._pending_session_start = None
            self._backend_session_started = True

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Session summary accessors (for McpTerminalLogger)
    # ------------------------------------------------------------------

    def get_violations(self) -> list[dict[str, Any]]:
        """Return list of violation dicts accumulated during the session."""
        return list(self._violations)

    def get_controls_triggered(self) -> list[str]:
        """Return list of execution control names that triggered."""
        return list(self._controls_triggered)

    # ------------------------------------------------------------------
    # Redaction
    # ------------------------------------------------------------------

    def _redact(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Replace matching parameter paths with ``[REDACTED]``."""
        return redact_dict_deepcopy(params, self._redact_patterns)
