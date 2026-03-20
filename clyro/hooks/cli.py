# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — CLI Entry Points
# Implements FRD-HK-012

"""CLI commands: `clyro-hook evaluate` and `clyro-hook trace`."""

from __future__ import annotations

import argparse
import json
import sys
import time

import structlog

from clyro.constants import ISSUE_TRACKER_URL

from .audit import AuditLogger
from .backend import resolve_agent_id
from .config import ConfigError, HookConfig, load_hook_config
from .constants import EXIT_FAIL_CLOSED, EXIT_FAIL_OPEN, EXIT_OK
from .evaluator import evaluate
from .models import HookInput
from .state import StateLock, load_state, save_state
from .tracer import handle_session_end, handle_tool_complete

logger = structlog.get_logger()

_ISSUE_TRACKER = ISSUE_TRACKER_URL


def _error_with_context(message: str) -> str:
    """Append issue tracker URL to an error message for hooks stderr output."""
    return f"{message}\n  Report at {_ISSUE_TRACKER}"


def _read_stdin() -> dict:
    """Read and parse JSON from stdin. Raises ValueError on failure."""
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Empty stdin")
    return json.loads(raw)


def _parse_hook_input(data: dict) -> HookInput:
    """Validate stdin JSON into HookInput model."""
    if "session_id" not in data or not data["session_id"]:
        raise ValueError("Missing required field: session_id")
    return HookInput.model_validate(data)


def _create_audit(config: HookConfig) -> AuditLogger:
    """Create AuditLogger from config."""
    return AuditLogger(
        log_path=config.audit.log_path or "~/.clyro/hooks/audit.jsonl",
        redact_patterns=config.audit.redact_parameters,
    )


def _ensure_agent_id(config: HookConfig, session_id: str) -> None:
    """Resolve and persist agent_id in session state. Fail-open.

    FRD-HK-007: agent_id is resolved once per session (first invocation)
    and persisted in the session state file for reuse.
    """
    try:
        state = load_state(session_id)
        if not state.agent_id:
            resolve_agent_id(config.backend, state)
            if state.agent_id:
                save_state(state)
    except Exception as e:
        logger.warning("agent_id_resolution_failed", error=str(e))


def cmd_evaluate(args: argparse.Namespace) -> int:
    """PreToolUse hook handler — runs prevention stack.

    FRD-HK-012: Reads stdin, evaluates, outputs decision to stdout.
    Exit 0 = decision rendered, Exit 1 = fail-open.
    """
    start_time = time.monotonic()

    try:
        data = _read_stdin()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Invalid input: {e}", file=sys.stderr)
        return EXIT_FAIL_OPEN

    try:
        hook_input = _parse_hook_input(data)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return EXIT_FAIL_OPEN

    if not hook_input.tool_name:
        print("Missing required field: tool_name", file=sys.stderr)
        return EXIT_FAIL_OPEN

    try:
        config = load_hook_config(args.config)
    except ConfigError as e:
        # Can't evaluate without valid config — fail-closed
        print(_error_with_context(str(e)), file=sys.stderr)
        return EXIT_FAIL_CLOSED

    # Resolve agent_id on first invocation (fail-open)
    _ensure_agent_id(config, hook_input.session_id)

    audit = _create_audit(config)

    try:
        with StateLock(hook_input.session_id):
            result = evaluate(hook_input, config, audit)
    except TimeoutError:
        # Lock contention — fail-closed: deny rather than skip evaluation
        logger.error("state_lock_timeout", session_id=hook_input.session_id)
        return EXIT_FAIL_CLOSED
    except Exception as e:
        # Governance stack crash — fail-closed: deny unevaluated calls
        logger.error("evaluate_unexpected_error", error=_error_with_context(str(e)))
        return EXIT_FAIL_CLOSED
    finally:
        audit.close()

    elapsed_ms = (time.monotonic() - start_time) * 1000
    if elapsed_ms > 200:
        logger.warning("evaluate_latency_exceeded", elapsed_ms=round(elapsed_ms, 1))

    if result is not None:
        # Block decision — output JSON to stdout
        print(json.dumps(result.model_dump(exclude_none=True)))

    return EXIT_OK


def cmd_trace(args: argparse.Namespace) -> int:
    """PostToolUse and Stop hook handler — emits traces.

    FRD-HK-012: Exit 0 always (trace hooks cannot block).
    """
    start_time = time.monotonic()

    if not args.event:
        print("Error: --event is required (tool-complete or session-end)", file=sys.stderr)
        return EXIT_FAIL_OPEN

    if args.event not in ("tool-complete", "session-end"):
        print(
            f"Error: unknown event '{args.event}'. Use 'tool-complete' or 'session-end'",
            file=sys.stderr,
        )
        return EXIT_FAIL_OPEN

    try:
        data = _read_stdin()
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("trace_invalid_input", error=str(e))
        return EXIT_OK  # PostToolUse must not interfere

    try:
        hook_input = HookInput.model_validate(data)
    except Exception as e:
        logger.warning("trace_validation_error", error=str(e))
        return EXIT_OK

    try:
        config = load_hook_config(args.config)
    except ConfigError as e:
        logger.warning("trace_config_error", error=str(e))
        return EXIT_OK

    audit = _create_audit(config)

    try:
        if args.event == "tool-complete":
            handle_tool_complete(hook_input, config, audit)
        elif args.event == "session-end":
            handle_session_end(hook_input, config, audit)
    except Exception as e:
        logger.warning("trace_handler_error", event=args.event, error=str(e))
    finally:
        audit.close()

    elapsed_ms = (time.monotonic() - start_time) * 1000
    if elapsed_ms > 500:
        logger.warning("trace_latency_exceeded", elapsed_ms=round(elapsed_ms, 1))

    return EXIT_OK


def main() -> None:
    """Main entry point for `clyro-hook` CLI."""
    parser = argparse.ArgumentParser(
        prog="clyro-hook",
        description="Clyro governance hooks for Claude Code",
    )
    subparsers = parser.add_subparsers(dest="command", help="Hook command")

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="PreToolUse hook — evaluate tool call against prevention stack",
    )
    eval_parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML policy config (default: ~/.clyro/hooks/claude-code-policy.yaml)",
    )

    # trace subcommand
    trace_parser = subparsers.add_parser(
        "trace",
        help="PostToolUse/Stop hook — emit trace events",
    )
    trace_parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML policy config",
    )
    trace_parser.add_argument(
        "--event",
        required=True,
        choices=["tool-complete", "session-end"],
        help="Event type: tool-complete or session-end",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_FAIL_OPEN)

    if args.command == "evaluate":
        sys.exit(cmd_evaluate(args))
    elif args.command == "trace":
        sys.exit(cmd_trace(args))


if __name__ == "__main__":
    main()
