# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — CLI Entrypoint
# Implements FRD-009, FRD-012, FRD-015, FRD-016, FRD-017

"""
Parse ``clyro-mcp wrap <server-command> [--config <path>]``, validate
arguments, load config, register signal handlers, and launch the
asyncio event loop.

v1.1: When backend sync is enabled (API key configured), initializes
AgentRegistrar, CloudPolicyFetcher, and BackendSyncManager between
config load and server spawn (TDD §5.6).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from clyro import __version__
from clyro.config import load_config
from clyro.mcp.audit import AuditLogger
from clyro.mcp.log import get_logger
from clyro.mcp.prevention import PreventionStack
from clyro.mcp.router import MessageRouter
from clyro.mcp.session import McpSession
from clyro.mcp.terminal import McpTerminalLogger
from clyro.mcp.transport import StdioTransport

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clyro-mcp",
        description="MCP Governance Wrapper — apply Clyro Prevention Stack to MCP tool calls",
    )
    parser.add_argument("--version", action="version", version=f"clyro-mcp {__version__}")

    sub = parser.add_subparsers(dest="command")
    wrap_parser = sub.add_parser("wrap", help="Wrap an MCP server with governance")
    wrap_parser.add_argument(
        "server_command",
        nargs=argparse.REMAINDER,
        help="MCP server command and arguments",
    )
    wrap_parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to YAML policy config (default: ~/.clyro/mcp-wrapper/mcp-config.yaml)",
    )
    return parser


_MARKER_DIR = Path(os.path.expanduser("~/.clyro/mcp-wrapper"))


def _marker_path(audit_log_path: str) -> Path:
    """Derive a unique marker file path from the audit log path."""
    key = hashlib.sha256(audit_log_path.encode()).hexdigest()[:12]
    return _MARKER_DIR / f"mcp-active-{key}.json"


def _write_marker(marker: Path, session_id: str, audit_log_path: str) -> None:
    """Write an active-session marker file."""
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "audit_log_path": audit_log_path,
                    "started_at": time.time(),
                }
            )
        )
    except OSError:
        pass  # Best-effort


def _delete_marker(marker: Path) -> None:
    """Remove the active-session marker file."""
    try:
        marker.unlink(missing_ok=True)
    except OSError:
        pass


def _recover_orphaned_session(audit_log_path: str) -> None:
    """If a previous session was killed without writing session_end, write one now."""
    marker = _marker_path(audit_log_path)
    if not marker.exists():
        return
    try:
        data = json.loads(marker.read_text())
        session_id = data["session_id"]
        # Use the marker file's mtime as the approximate session_end time
        end_time = os.path.getmtime(str(marker))
        from datetime import UTC, datetime

        entry = (
            json.dumps(
                {
                    "timestamp": datetime.fromtimestamp(end_time, tz=UTC).isoformat(),
                    "session_id": session_id,
                    "event": "session_end",
                    "reason": "orphan_recovery",
                }
            )
            + "\n"
        )
        log_path = Path(os.path.expanduser(audit_log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(entry)
        logger.info("orphan_session_recovered", session_id=session_id)
    except (OSError, json.JSONDecodeError, KeyError):
        pass  # Best-effort
    finally:
        _delete_marker(marker)


def _derive_instance_id(agent_name: str) -> str:
    """Derive instance_id from agent name: sha256(agent_name)[:12] (FRD-018)."""
    return hashlib.sha256(agent_name.encode()).hexdigest()[:12]


def _derive_agent_name(config_agent_name: str | None, server_command: list[str]) -> str:
    """Derive agent name from config or server command (FRD-016)."""
    if config_agent_name:
        return config_agent_name
    return " ".join(server_command) if server_command else "mcp-agent"


async def _init_backend(config, session, server_command):
    """
    Initialize backend components when API key is configured (FRD-015, FRD-016, FRD-017).

    Returns:
        Tuple of (sync_manager, trace_factory, http_client) or (None, None, None).
    """
    from clyro.backend.agent_registrar import AgentRegistrar
    from clyro.backend.circuit_breaker import CircuitBreaker, ConnectivityDetector
    from clyro.backend.cloud_policy import CloudPolicyFetcher
    from clyro.backend.event_queue import EventQueue
    from clyro.backend.http_client import HttpSyncClient
    from clyro.backend.sync_manager import BackendSyncManager
    from clyro.backend.trace_event_factory import TraceEventFactory

    api_key = config.resolved_api_key
    api_url = config.resolved_api_url
    agent_name = _derive_agent_name(config.backend.agent_name, server_command)
    instance_id = _derive_instance_id(agent_name)

    # Create HTTP client (FRD-015)
    http_client = HttpSyncClient(api_key=api_key, base_url=api_url)

    # 2a. Agent registration (FRD-016)
    registrar = AgentRegistrar(instance_id=instance_id, http_client=http_client, api_key=api_key)
    session.agent_id = await registrar.get_or_register(agent_name)
    session.agent_name = agent_name

    # 2b. Cloud policy fetch + merge (FRD-017)
    fetcher = CloudPolicyFetcher(http_client=http_client)
    merged_policies = await fetcher.fetch_and_merge(
        agent_id=str(session.agent_id) if session.agent_id else None,
        local_policies=config.global_.policies,
        timeout=2.0,
    )
    # Update config with merged policies (cloud + local)
    config.global_.policies = merged_policies

    # Promote cloud policies that map to built-in prevention stages (FRD-017)
    # Cloud rules for "cost" and "step_number" need to feed into CostTracker
    # (Stage 3) and StepLimit (Stage 2) respectively, not just PolicyEvaluator
    # (Stage 4). The built-in stages do pre-call estimation / enforcement that
    # the generic PolicyEvaluator cannot replicate.
    for policy in merged_policies:
        try:
            if (
                policy.parameter == "cost"
                and policy.operator == "max_value"
                and policy.value is not None
            ):
                cloud_cost = float(policy.value)
                if cloud_cost < config.global_.max_cost_usd:
                    config.global_.max_cost_usd = cloud_cost
            elif (
                policy.parameter == "step_number"
                and policy.operator == "max_value"
                and policy.value is not None
            ):
                cloud_steps = int(float(policy.value))
                if cloud_steps < config.global_.max_steps:
                    config.global_.max_steps = cloud_steps
        except (TypeError, ValueError):
            pass

    # 2c. Initialize BackendSyncManager (FRD-015, FRD-018, FRD-019)
    event_queue = EventQueue(
        instance_id=instance_id,
        max_size_mb=config.backend.pending_queue_max_mb,
    )
    circuit_breaker = CircuitBreaker()
    connectivity = ConnectivityDetector()
    sync_manager = BackendSyncManager(
        event_queue=event_queue,
        circuit_breaker=circuit_breaker,
        connectivity=connectivity,
        http_client=http_client,
        sync_interval=config.backend.sync_interval_seconds,
    )
    trace_factory = TraceEventFactory(session=session)

    # Start background sync loop
    sync_manager.start()

    return sync_manager, trace_factory, http_client


async def _async_main(
    server_command: list[str],
    config_path: str | None,
) -> int:
    """Core async entry point — creates all components and runs the router."""
    # 0. Recover any orphaned session from a previous SIGKILL
    config = load_config(config_path)
    _recover_orphaned_session(config.audit.log_path)

    # 1. Load config (already done above)

    # 2. Create session
    session = McpSession()

    # 2a-2c. Backend initialization if enabled (FRD-015, FRD-016, FRD-017)
    sync_manager = None
    trace_factory = None
    http_client = None
    if config.is_backend_enabled:
        try:
            sync_manager, trace_factory, http_client = await _init_backend(
                config, session, server_command
            )
        except Exception as exc:
            logger.warning("backend_init_failed", error=str(exc))
            sync_manager = None
            trace_factory = None
            http_client = None

    # 3. Create components
    transport = StdioTransport(server_command)
    prevention = PreventionStack(config)
    audit = AuditLogger(config.audit, session.session_id)
    terminal = McpTerminalLogger(is_backend_enabled=config.is_backend_enabled)

    # Attach backend to audit for dual-mode emission (FRD-015)
    if sync_manager is not None:
        audit.set_backend(sync_manager, trace_factory)

        # Attach violation reporter for backend persistence (FRD-006)
        if session.agent_id:
            audit.set_violation_reporter(
                reporter=sync_manager.enqueue_violation,
                agent_id=str(session.agent_id),
            )

    # 4. Spawn server
    await transport.start()

    # 5. Audit session start + write marker for orphan detection
    audit.log_lifecycle("session_start")
    marker = _marker_path(config.audit.log_path)
    _write_marker(marker, str(session.session_id), config.audit.log_path)

    # 6. Signal handlers (FRD-012)
    loop = asyncio.get_event_loop()
    router = MessageRouter(config, session, transport, prevention, audit)

    def _handle_sigterm() -> None:
        # Write session_end immediately in signal handler — the process
        # may be SIGKILL'd shortly after SIGTERM with no time for cleanup.
        audit.log_lifecycle(
            "session_end",
            extra={
                "total_steps": session.step_count,
                "total_cost_usd": round(session.accumulated_cost_usd, 6),
            },
        )
        _delete_marker(marker)
        router.request_shutdown()

    def _handle_sighup() -> None:
        # Forward SIGHUP to child (FRD-012)
        proc = transport.process
        if proc and proc.pid:
            import os as _os

            try:
                _os.kill(proc.pid, signal.SIGHUP)
            except (ProcessLookupError, OSError):
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_sigterm)

    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(signal.SIGHUP, _handle_sighup)

    # 7. Run router
    try:
        exit_code = await router.run()
    finally:
        await transport.terminate()
        # Log session_end BEFORE backend shutdown so the trace event is enqueued
        # (duplicate-safe: audit._session_ended flag prevents double writes)
        audit.log_lifecycle(
            "session_end",
            extra={
                "total_steps": session.step_count,
                "total_cost_usd": round(session.accumulated_cost_usd, 6),
            },
        )
        _delete_marker(marker)
        # Print governance summary to stderr (respects CLYRO_QUIET)
        terminal.print_session_summary(
            steps=session.step_count,
            cost_usd=session.accumulated_cost_usd,
            violations=audit.get_violations(),
            controls_triggered=audit.get_controls_triggered(),
        )
        # Flush backend sync after session_end is enqueued (FRD-015)
        if sync_manager is not None:
            await sync_manager.shutdown()
        if http_client is not None:
            await http_client.close()
        audit.close()

    return exit_code


def main() -> None:
    """Synchronous CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "wrap":
        parser.print_help(sys.stderr)
        sys.exit(1)

    server_command = args.server_command
    if not server_command:
        logger.error("server_command_required")
        print("Usage: clyro-mcp wrap <server-command> [args...]", file=sys.stderr)
        sys.exit(1)

    # Strip leading '--' if present (argparse REMAINDER quirk)
    if server_command and server_command[0] == "--":
        server_command = server_command[1:]

    exit_code = asyncio.run(_async_main(server_command, args.config))
    sys.exit(exit_code)
