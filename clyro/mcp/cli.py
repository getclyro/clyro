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
from clyro.mcp.server_transport import TransportError
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
    # Native HTTP transport (FRD-021/022/042/039). Override the YAML config.
    wrap_parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default=None,
        help="Downstream transport (default: stdio, or config value)",
    )
    wrap_parser.add_argument(
        "--url",
        default=None,
        help="Remote MCP server URL (required for --transport http)",
    )
    wrap_parser.add_argument(
        "--allow-plaintext",
        action="store_true",
        help="Permit plaintext/loopback HTTP targets (the single safety-floor relaxation, FRD-039)",
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


def _derive_instance_id(agent_name: str, api_url: str = "") -> str:
    """Derive instance_id from agent name + API URL: sha256(name|url)[:12] (FRD-018).

    Including the API URL ensures different environments (production, staging)
    get separate cached agent IDs and don't cross-contaminate.
    """
    key = f"{agent_name}|{api_url}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


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
    instance_id = _derive_instance_id(agent_name, api_url)

    # Create HTTP client (FRD-015)
    http_client = HttpSyncClient(api_key=api_key, base_url=api_url)

    # 2a. Agent registration (FRD-016)
    registrar = AgentRegistrar(instance_id=instance_id, http_client=http_client, api_key=api_key)
    session.agent_id = await registrar.get_or_register(agent_name)
    session.agent_name = agent_name

    # 2b. Cloud policy fetch + merge (FRD-017)
    fetcher = CloudPolicyFetcher(http_client=http_client)
    merged_policies, resolved_default = await fetcher.fetch_and_merge(
        agent_id=str(session.agent_id) if session.agent_id else None,
        local_policies=config.global_.policies,
        timeout=2.0,
        local_default_action=config.default_action,
    )
    # Update config with merged policies (cloud + local) and the resolved
    # default_action. Reconciliation uses cloud-wins: whenever the cloud
    # declared any default_action, it overrides the local wrapper's
    # default_action. If multiple cloud policies disagree, the most-
    # restrictive among cloud (block) wins. The local default applies
    # only when no cloud policies were fetched.
    config.global_.policies = merged_policies
    config.default_action = resolved_default

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


def _build_transport(config, server_command: list[str]):
    """Build the downstream transport from config (FRD-021/022/042).

    Returns ``(transport, label)``. STDIO is the default; HTTP is used when
    ``config.transport == "http"`` and wires the safety floor, TLS policy, and
    the static-credential provider (D21). Selection errors refuse startup.
    """
    from clyro.mcp.selector import SelectionError, select_transport

    try:
        sel = select_transport(
            transport=config.transport,
            url=config.server.url,
            server_command=server_command,
        )
    except SelectionError as exc:
        logger.error("transport_selection_failed", error=str(exc))
        print(f"clyro-mcp: {exc}", file=sys.stderr)
        sys.exit(1)

    if sel.transport == "stdio":
        return StdioTransport(sel.server_command), "stdio"

    # HTTP (FRD-020/042). Compose the outbound-safety trio. Component
    # construction can reject bad config with a TransportError — e.g. a supplied
    # CA bundle that does not exist (FRD-046). Surface it as a clean refuse-to-
    # start, like a SelectionError above, rather than letting it escape as a
    # traceback.
    from clyro.mcp.auth import CredentialProvider
    from clyro.mcp.http_transport import HttpTransport
    from clyro.mcp.safety import SafetyFloor
    from clyro.mcp.tls import TlsPolicy

    try:
        floor = SafetyFloor(allow_plaintext=config.server.allow_plaintext)
        tls = TlsPolicy(config.server.ca_bundle)
        # FRD-033/034: the credential may live under any header the operator
        # names. Hardcoding "Authorization" here meant a credential configured
        # elsewhere was silently unsent AND left unmasked in records (S2).
        auth = CredentialProvider(
            config.server.headers.get(config.server.auth_header),
            header_name=config.server.auth_header,
        )
        transport = HttpTransport(
            sel.url,
            floor=floor,
            tls=tls,
            auth=auth,
            liveness_secs=config.server.liveness_secs,
            max_reconnect=config.server.reconnect.max_attempts,  # FRD-056
        )
    except TransportError as exc:
        logger.error("transport_setup_failed", error=str(exc))
        print(f"clyro-mcp: {exc}", file=sys.stderr)
        sys.exit(1)
    return transport, "http"


async def _async_main(
    server_command: list[str],
    config_path: str | None,
    *,
    transport: str | None = None,
    url: str | None = None,
    allow_plaintext: bool = False,
) -> int:
    """Core async entry point — creates all components and runs the router."""
    # 0. Recover any orphaned session from a previous SIGKILL
    config = load_config(config_path)
    # CLI flags override YAML config (FRD-021/022/042/039).
    if transport:
        config.transport = transport
    if url:
        config.server.url = url
    if allow_plaintext:
        config.server.allow_plaintext = True
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

    # FRD-041: warn when migrating to HTTP without a pinned agent name, since
    # identity continuity holds only for a pinned name (D3).
    if config.transport == "http" and not config.backend.agent_name:
        logger.warning(
            "agent_name_unpinned",
            hint="pin backend.agent_name to keep agent identity continuous across migration",
        )

    # 3. Create components
    transport, transport_label = _build_transport(config, server_command)
    prevention = PreventionStack(config)
    audit = AuditLogger(config.audit, session.session_id)
    audit.set_transport(transport_label)  # FRD-032 (audit records)
    session.transport = transport_label  # FRD-032 (trace records, via metadata)
    if transport_label == "http":
        # FRD-034: mask the known credential value from any emitted record —
        # whichever header the operator configured it under (S2).
        audit.set_credential_mask(config.server.headers.get(config.server.auth_header))
        # FRD-044: record the endpoint (audit records + trace metadata via session).
        audit.set_endpoint(config.server.url)
        session.endpoint = config.server.url
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

    # 4. Open the downstream leg (spawn child | connect + validate). A transport
    # failure here (safety-floor refusal, unreachable server, TLS failure) is an
    # expected, user-facing outcome — report it cleanly and exit non-zero rather
    # than surfacing a traceback.
    try:
        await transport.open()
    except TransportError as exc:
        logger.error("transport_open_failed", error=str(exc))
        print(f"clyro-mcp: cannot start — {exc}", file=sys.stderr)
        audit.close()
        return 1

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
        # Forward SIGHUP to child (FRD-012); stdio-only (HTTP has no child).
        proc = getattr(transport, "process", None)
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
        await transport.close()
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
        # Flush backend sync after session_end is enqueued (FRD-015)
        sync_ok: bool | None = None
        if sync_manager is not None:
            try:
                await sync_manager.shutdown()
                sync_ok = sync_manager._event_queue.pending_count == 0
            except Exception:
                sync_ok = False
        # Print governance summary to stderr (respects CLYRO_QUIET)
        terminal.print_session_summary(
            steps=session.step_count,
            cost_usd=session.accumulated_cost_usd,
            violations=audit.get_violations(),
            controls_triggered=audit.get_controls_triggered(),
            sync_ok=sync_ok,
        )
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
    # Strip leading '--' if present (argparse REMAINDER quirk)
    if server_command and server_command[0] == "--":
        server_command = server_command[1:]

    # HTTP takes its target from --url or the config file, so a server command
    # is not required in that mode (FRD-042); STDIO still requires one. The
    # transport may be selected in the config file, not only on the CLI — so when
    # no CLI flag settles it, consult the config. Without this, an HTTP-in-config
    # setup (transport: http in the YAML) is wrongly rejected for a missing stdio
    # command, because the config is otherwise only read later in _async_main.
    http_mode = args.transport == "http" or bool(args.url)
    if not server_command and not http_mode and args.transport is None:
        http_mode = load_config(args.config).transport == "http"
    if not server_command and not http_mode:
        logger.error("server_command_required")
        print(
            "Usage: clyro-mcp wrap <server-command>   (stdio)\n"
            "       clyro-mcp wrap --transport http --url <URL>   (http)",
            file=sys.stderr,
        )
        sys.exit(1)

    exit_code = asyncio.run(
        _async_main(
            server_command,
            args.config,
            transport=args.transport,
            url=args.url,
            allow_plaintext=args.allow_plaintext,
        )
    )
    sys.exit(exit_code)
