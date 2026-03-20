# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Backend Integration
# Implements FRD-HK-007 (agent registration), FRD-HK-008 (traces), FRD-HK-009 (session lifecycle)

"""
Agent registration, circuit breaker, event queue, trace emission,
and policy violation reporting for cloud backend integration.

Design:
- clyro-hook is ephemeral (one process per hook event), so we persist
  agent_id and circuit breaker state in session state JSON files.
- Event queue uses a JSONL file that accumulates trace events during
  the session and flushes at session-end (Stop hook).
- AgentRegistrar reuses clyro-mcp's pattern: file-persisted agent_id
  at ~/.clyro/hooks/agents/hook-agent-{instance_id}.id
- Circuit breaker state is persisted in SessionState to survive across
  ephemeral invocations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

from clyro.backend.agent_registrar import (
    _extract_org_id_from_api_key,
    _generate_deterministic_agent_id,
)
from clyro.backend.circuit_breaker import (
    check_can_execute as circuit_can_execute,
)
from clyro.backend.circuit_breaker import (
    record_failure as circuit_record_failure,
)
from clyro.backend.circuit_breaker import (
    record_success as circuit_record_success,
)
from clyro.backend.http_client import AuthenticationError, HttpSyncClient
from clyro.config import DEFAULT_API_URL

from .constants import (
    AGENT_FRAMEWORK,
    AGENT_ID_DIR,
    BACKEND_TIMEOUT_SECONDS,
    DEFAULT_AGENT_NAME,
    DIR_PERMISSIONS,
    EVENT_QUEUE_DIR,
    FILE_PERMISSIONS,
    MEMORY_FALLBACK_MAX_EVENTS,
    OUTPUT_TRUNCATE_BYTES,
)
from .models import CircuitBreakerSnapshot, SessionState

logger = structlog.get_logger()


# ── Agent Registration ─────────────────────────────────────────────────────


def _agent_id_path(instance_id: str) -> Path:
    """Return path to persisted agent_id file."""
    return AGENT_ID_DIR / f"hook-agent-{instance_id}.id"


def _unconfirmed_path(id_path: Path) -> Path:
    """Return path to unconfirmed marker file."""
    return id_path.with_suffix(id_path.suffix + ".unconfirmed")


def compute_instance_id(agent_name: str) -> str:
    """Derive deterministic instance_id from agent name (SHA-256, first 12 chars)."""
    return hashlib.sha256(agent_name.encode()).hexdigest()[:12]


def _load_persisted_agent_id(instance_id: str) -> str | None:
    """Load agent_id from persisted file, or None if absent/corrupt."""
    path = _agent_id_path(instance_id)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        UUID(raw)  # Validate UUID format
        return raw
    except (ValueError, OSError):
        return None


def _persist_agent_id(instance_id: str, agent_id: str, *, confirmed: bool) -> None:
    """Persist agent_id to file with restricted permissions."""
    path = _agent_id_path(instance_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(path.parent), DIR_PERMISSIONS)
        except OSError:
            pass
        path.write_text(agent_id, encoding="utf-8")
        marker = _unconfirmed_path(path)
        if confirmed:
            marker.unlink(missing_ok=True)
        else:
            marker.write_text("", encoding="utf-8")
    except OSError:
        logger.warning("agent_id_persist_failed")


async def _register_and_close(client: HttpSyncClient, agent_name: str) -> str:
    """Register agent with backend and close client in one event loop."""
    try:
        return await client.register_agent(agent_name, framework=AGENT_FRAMEWORK)
    finally:
        await client.close()


def resolve_agent_id(
    config_backend: Any,
    state: SessionState,
) -> str | None:
    """Resolve agent_id: from state, persisted file, or backend registration.

    FRD-HK-007: agent_id is required for cloud policy fetching and trace emission.
    Returns None if no API key configured (local-only mode).
    """
    api_key = getattr(config_backend, "api_key", None)
    if not api_key:
        # Check env var
        api_key = os.environ.get("CLYRO_API_KEY")
    if not api_key:
        return None

    agent_name = getattr(config_backend, "agent_name", None) or DEFAULT_AGENT_NAME
    instance_id = compute_instance_id(agent_name)

    # 1. Use state's agent_id if present
    if state.agent_id:
        return state.agent_id

    # 2. Try loading from persisted file
    persisted = _load_persisted_agent_id(instance_id)
    if persisted:
        marker = _unconfirmed_path(_agent_id_path(instance_id))
        if not marker.exists():
            # Confirmed — use directly
            state.agent_id = persisted
            return persisted

        # Unconfirmed — try re-registering
        try:
            client = HttpSyncClient(
                api_key=api_key,
                base_url=getattr(config_backend, "api_url", DEFAULT_API_URL),
                timeout=BACKEND_TIMEOUT_SECONDS,
            )
            agent_id = asyncio.run(_register_and_close(client, agent_name))
            _persist_agent_id(instance_id, agent_id, confirmed=True)
            state.agent_id = agent_id
            return agent_id
        except Exception:
            # Keep using unconfirmed ID
            state.agent_id = persisted
            return persisted

    # 3. No persisted ID — register with backend
    try:
        api_url = getattr(config_backend, "api_url", DEFAULT_API_URL)
        client = HttpSyncClient(api_key=api_key, base_url=api_url, timeout=BACKEND_TIMEOUT_SECONDS)
        agent_id = asyncio.run(_register_and_close(client, agent_name))
        _persist_agent_id(instance_id, agent_id, confirmed=True)
        state.agent_id = agent_id
        return agent_id
    except Exception:
        # Fallback: deterministic UUID5 (matches SDK & backend) so identity
        # stays stable even if backend is down — no drift on recovery.
        org_id = _extract_org_id_from_api_key(api_key)
        if org_id:
            local_id = str(_generate_deterministic_agent_id(agent_name, org_id))
        else:
            local_id = str(uuid4())
        _persist_agent_id(instance_id, local_id, confirmed=False)
        state.agent_id = local_id
        logger.warning("agent_registration_failed", local_id=local_id)
        return local_id


# ── Event Queue (File-Based JSONL) ─────────────────────────────────────────


_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9._\-]+$")


def _event_queue_path(session_id: str) -> Path:
    """Return path to session's event queue file.

    Validates session_id with whitelist regex and verifies the resolved
    path stays within EVENT_QUEUE_DIR to prevent path traversal.
    """
    if not session_id or not _SAFE_SESSION_ID.match(session_id):
        session_id = "unknown"
    base = EVENT_QUEUE_DIR
    path = base / f"pending-{session_id}.jsonl"
    if not path.resolve().is_relative_to(base.resolve()):
        path = base / "pending-unknown.jsonl"
    return path


# In-memory fallback when file I/O fails (FRD-HK-008)
_memory_fallback: dict[str, list[dict[str, Any]]] = {}


def enqueue_event(session_id: str, event: dict[str, Any]) -> None:
    """Append a trace event to the session's event queue file. Fail-open.

    Falls back to in-memory list when file I/O fails (max 1000 events).
    """
    path = _event_queue_path(session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(path.parent), DIR_PERMISSIONS)
        except OSError:
            pass
        line = json.dumps(event, default=str) + "\n"
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_PERMISSIONS)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except (OSError, PermissionError) as e:
        logger.warning("event_queue_append_failed", error=str(e), fallback="memory")
        # Memory fallback
        if session_id not in _memory_fallback:
            _memory_fallback[session_id] = []
        if len(_memory_fallback[session_id]) < MEMORY_FALLBACK_MAX_EVENTS:
            _memory_fallback[session_id].append(event)


def load_queued_events(session_id: str) -> list[dict[str, Any]]:
    """Load all queued events from session's event queue file + memory fallback."""
    path = _event_queue_path(session_id)
    events = []
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except (OSError, PermissionError):
            pass
    # Include in-memory fallback events
    events.extend(_memory_fallback.get(session_id, []))
    return events


def clear_event_queue(session_id: str) -> None:
    """Remove the session's event queue file and memory fallback after successful flush."""
    path = _event_queue_path(session_id)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
    _memory_fallback.pop(session_id, None)


# ── Trace Event Factory ───────────────────────────────────────────────────


def _derive_event_name(event_type: str, tool_name: str | None) -> str:
    """Derive human-readable event_name from event_type and tool_name."""
    if tool_name:
        return f"{event_type}:{tool_name}"
    return event_type


def _derive_agent_stage(event_type: str) -> str:
    """Map event_type to agent stage (think/act/observe).

    - think: reasoning/planning — session_start, pre_tool_use, policy_check
    - act: executing action — (reserved for future use)
    - observe: processing results — tool_call_observe, session_end, error
    """
    if event_type in ("tool_call_observe", "session_end", "error"):
        return "observe"
    return "think"


def create_trace_event(
    event_type: str,
    session_id: str,
    agent_id: str | None = None,
    agent_name: str | None = None,
    *,
    tool_name: str | None = None,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    duration_ms: int = 0,
    cost_usd: float = 0.0,
    token_count_input: int = 0,
    token_count_output: int = 0,
    step_number: int | None = None,
    accumulated_cost_usd: float = 0.0,
    error_type: str | None = None,
    error_message: str | None = None,
    parent_event_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an SDK-compatible trace event dict with event_id.

    FRD-HK-008/015: All events include event_id (UUID), parent_event_id,
    framework, token counts, and truncated output_data (10KB max).
    """
    from . import __version__

    # Truncate output_data to 10KB (matching MCP wrapper behavior)
    original_output = output_data
    output_data = truncate_output(output_data)
    truncated = output_data is not original_output and output_data is not None

    merged_meta: dict[str, Any] = {
        **(metadata or {}),
        "_source": "claude_code_hooks",
        "cost_estimated": True,
        "agent_name": agent_name or DEFAULT_AGENT_NAME,
    }
    if tool_name:
        merged_meta["tool_name"] = tool_name
    if truncated:
        merged_meta["output_truncated"] = True

    return {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "event_name": _derive_event_name(event_type, tool_name),
        "session_id": session_id,
        "agent_id": agent_id,
        "parent_event_id": parent_event_id,
        "agent_stage": _derive_agent_stage(event_type),
        "framework": AGENT_FRAMEWORK,
        "framework_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "duration_ms": duration_ms,
        "input_data": input_data,
        "output_data": output_data,
        "token_count_input": token_count_input,
        "token_count_output": token_count_output,
        "cost_usd": cost_usd,
        "step_number": step_number,
        "cumulative_cost": accumulated_cost_usd,
        "error_type": error_type,
        "error_message": error_message,
        "metadata": merged_meta,
    }


def truncate_output(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Truncate output data to OUTPUT_TRUNCATE_BYTES. FRD-HK-008."""
    if data is None:
        return None
    try:
        output_json = json.dumps(data, default=str)
        if len(output_json) > OUTPUT_TRUNCATE_BYTES:
            return {"_truncated": output_json[:OUTPUT_TRUNCATE_BYTES]}
    except Exception:
        return {"_error": "serialization_failed"}
    return data


def estimate_tokens(text_len: int) -> int:
    """Estimate token count from character length (4 chars ≈ 1 token).

    Delegates to the canonical heuristic in clyro.cost.HeuristicCostEstimator.
    """
    from clyro.cost import HeuristicCostEstimator

    # HeuristicCostEstimator uses len // 4 — call with a dummy string of
    # the right length to stay in sync if the formula ever changes.
    _, tokens = HeuristicCostEstimator().estimate_from_payload("x" * text_len)
    return tokens


# ── Trace Emission (Batched at Session-End) ────────────────────────────────


async def _send_batch_and_close(
    client: HttpSyncClient, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Send a batch of trace events to backend and close client in one event loop."""
    try:
        return await client.send_batch(events=events)
    finally:
        await client.close()


def flush_event_queue(
    session_id: str,
    api_key: str,
    api_url: str,
    circuit: CircuitBreakerSnapshot,
) -> None:
    """Flush all queued events to backend in a single batch. Fail-open.

    FRD-HK-008/009: Events are queued during PostToolUse and flushed at session-end.
    """
    if not circuit_can_execute(circuit):
        logger.warning("circuit_open_skip_flush", session_id=session_id)
        return

    events = load_queued_events(session_id)
    if not events:
        return

    try:
        client = HttpSyncClient(api_key=api_key, base_url=api_url, timeout=BACKEND_TIMEOUT_SECONDS)
        asyncio.run(_send_batch_and_close(client, events))
        circuit_record_success(circuit)
        clear_event_queue(session_id)
        logger.info("event_queue_flushed", count=len(events), session_id=session_id)
    except AuthenticationError:
        circuit_record_failure(circuit)
        logger.warning("flush_auth_error", session_id=session_id)
    except Exception as e:
        circuit_record_failure(circuit)
        logger.warning("flush_failed", error=str(e), session_id=session_id)


# ── Policy Violation Reporting ─────────────────────────────────────────────


def report_violation(
    api_key: str,
    api_url: str,
    agent_id: str,
    session_id: str,
    tool_name: str,
    reason: str,
    rule_results: list[dict[str, Any]] | None,
    circuit: CircuitBreakerSnapshot,
    *,
    violation_details: dict[str, Any] | None = None,
    tool_input: dict[str, Any] | None = None,
    step_number: int = 0,
) -> None:
    """Report a policy violation to the backend. Fail-open.

    FRD-HK-006: Violations are reported with rich payload for governance analytics,
    including policy_id, operator, expected/actual values, and parameters_hash.
    """
    if not circuit_can_execute(circuit):
        return

    # Build rich violation payload matching MCP wrapper format (FRD-006)
    violation: dict[str, Any] = {
        "agent_id": agent_id,
        "session_id": session_id,
        "action_type": tool_name,
        "decision": "block",
        "message": reason,
        "step_number": step_number,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if violation_details:
        violation["policy_id"] = violation_details.get("policy_id")
        violation["rule_id"] = violation_details.get("rule_name", "unknown")
        violation["rule_name"] = violation_details.get("rule_name", "unknown")
        violation["operator"] = violation_details.get("operator", "")
        violation["expected_value"] = json.dumps(violation_details.get("expected"), default=str)
        violation["actual_value"] = json.dumps(violation_details.get("actual"), default=str)

    if tool_input:
        violation["parameters_hash"] = hashlib.sha256(
            json.dumps(tool_input, sort_keys=True, default=str).encode()
        ).hexdigest()

    if rule_results:
        violation["rule_results"] = rule_results

    async def _report_and_close() -> None:
        try:
            await client.report_violations([violation])
        finally:
            await client.close()

    try:
        client = HttpSyncClient(api_key=api_key, base_url=api_url, timeout=BACKEND_TIMEOUT_SECONDS)
        asyncio.run(_report_and_close())
        circuit_record_success(circuit)
    except Exception as e:
        circuit_record_failure(circuit)
        logger.warning("violation_report_failed", error=str(e))
