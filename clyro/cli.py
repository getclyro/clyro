# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — CLI Entry Point
# Implements FRD-SOF-010

"""
CLI entry point for the ``clyro-sdk`` command.

Provides:
- ``clyro-sdk``              — print help with available subcommands
- ``clyro-sdk feedback``     — submit feedback or report an issue
- ``clyro-sdk status``       — show mode, usage, adapter, policies, sessions
- ``clyro-sdk --version``    — print SDK version

Uses argparse (stdlib, zero dependencies) with add_subparsers().
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.parse
import webbrowser
from typing import Any

import clyro
from clyro.config import DEFAULT_API_URL
from clyro.constants import GITHUB_NEW_ISSUE_URL

# Maximum URL length for browser-based feedback (TDD §13.4 edge case)
_MAX_URL_BODY_LENGTH = 2000

# GitHub issue URL base
_GITHUB_ISSUES_URL = GITHUB_NEW_ISSUE_URL


def _auto_capture_context() -> dict[str, Any]:
    """Auto-capture environment context for feedback.  Implements FRD-SOF-010."""
    return {
        "sdk_version": getattr(clyro, "__version__", "unknown"),
        "python_version": sys.version,
        "platform": platform.system(),
        "platform_version": platform.version(),
    }


def _is_headless() -> bool:
    """Detect headless environment.  Implements FRD-SOF-010."""
    if os.environ.get("CI"):
        return True
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return True
    return False


def _submit_cloud_feedback(
    message: str,
    context: dict[str, Any],
) -> bool:
    """
    Submit feedback via POST /v1/feedback (cloud mode).

    Returns True on success, False on failure.
    """
    api_key = os.environ.get("CLYRO_API_KEY")
    endpoint = os.environ.get("CLYRO_ENDPOINT", DEFAULT_API_URL)

    if not api_key:
        return False

    try:
        import httpx

        url = f"{endpoint.rstrip('/')}/v1/feedback"
        headers = {
            "X-Clyro-API-Key": api_key,
        }
        # Backend expects multipart/form-data (Form fields)
        form_data = {
            "category": "general",
            "title": message[:200],
            "description": message,
            "metadata": json.dumps(context),
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, data=form_data, headers=headers)
            response.raise_for_status()

        print("Feedback submitted. Thank you!", file=sys.stderr)
        return True

    except Exception as exc:
        print(f"[clyro] Cloud feedback failed: {exc}", file=sys.stderr)
        return False


def _open_github_issue(message: str, context: dict[str, Any]) -> None:
    """
    Open a pre-filled GitHub Issue URL in the browser.

    Falls back to printing the URL if browser cannot be opened.
    Implements FRD-SOF-010 local mode behaviour.
    """
    # Build issue body (context without API key — NFR-005/§8.4)
    body_lines = [
        f"**Feedback:** {message}",
        "",
        "**Context:**",
        f"- SDK Version: {context.get('sdk_version', 'unknown')}",
        f"- Python: {context.get('python_version', 'unknown')}",
        f"- Platform: {context.get('platform', 'unknown')}",
    ]
    body = "\n".join(body_lines)

    # Truncate body for URL safety (TDD §13.4)
    if len(body) > _MAX_URL_BODY_LENGTH:
        body = body[:_MAX_URL_BODY_LENGTH] + "\n\n(truncated)"

    params = urllib.parse.urlencode(
        {
            "title": f"SDK Feedback: {message[:100]}",
            "body": body,
        }
    )
    url = f"{_GITHUB_ISSUES_URL}?{params}"

    if not _is_headless():
        print("Opening GitHub issue in browser...", file=sys.stderr)
        try:
            opened = webbrowser.open(url)
            if opened:
                return
        except Exception:
            pass

    # Headless or browser failed: print URL
    print(f"[clyro] Open this URL to submit feedback:\n{url}", file=sys.stderr)


def _handle_feedback(args: argparse.Namespace) -> int:
    """Handle the ``clyro-sdk feedback`` subcommand.  Implements FRD-SOF-010."""
    message = args.message

    # Interactive prompt if no --message and TTY available
    if not message:
        if sys.stdin.isatty():
            try:
                message = input("Enter your feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nFeedback cancelled.", file=sys.stderr)
                return 1
        else:
            print(
                'Usage: clyro feedback --message "your feedback"\n'
                "  (--message is required in non-interactive mode)",
                file=sys.stderr,
            )
            return 1

    if not message:
        print("No feedback provided.", file=sys.stderr)
        return 1

    context = _auto_capture_context()

    # Try cloud first (if API key present)
    if os.environ.get("CLYRO_API_KEY"):
        if _submit_cloud_feedback(message, context):
            return 0
        # Cloud failed — fall through to GitHub

    # Local mode or cloud fallback
    _open_github_issue(message, context)
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    """
    Handle the ``clyro-sdk status`` subcommand.

    Implements FRD-CT-001 (local mode), FRD-CT-002 (cloud mode),
    FRD-CT-003 (error handling).
    """
    try:
        return _status_internal()
    except Exception as exc:
        # FRD-CT-003: unexpected error → print message + issue URL, exit 1
        print(
            f"Error: {exc}. Report at github.com/getclyro/clyro/issues",
            file=sys.stderr,
        )
        return 1


def _status_internal() -> int:
    """Internal status logic. Exceptions propagate to _handle_status."""
    version = getattr(clyro, "__version__", "0.0.0")

    # Detect mode from config
    api_key = os.environ.get("CLYRO_API_KEY")
    mode = "cloud" if api_key else "local"

    # Read local SQLite stats (FRD-CT-001)
    local_stats = _read_local_stats()

    # Read local policy count
    policy_count = _read_policy_count()

    # Print header
    bar = "\u2501" * 40
    _print_stderr(f"Clyro v{version} \u2014 {mode.capitalize()} Mode")
    _print_stderr(bar)

    # Always show: mode, version, adapter, policies
    adapter = local_stats.get("adapter", "unknown") if local_stats else "unknown"
    _print_stderr(f" Mode:      {mode}")
    _print_stderr(f" Adapter:   {adapter}")
    _print_stderr(f" Policies:  {policy_count} rules loaded")

    if local_stats is None:
        # FRD-CT-001 failure: SQLite unavailable
        _print_stderr(" Local data unavailable \u2014 run a session first")
    elif local_stats["session_count"] == 0:
        # FRD-CT-001: zero sessions
        _print_stderr(" Sessions:  0")
        _print_stderr(" No sessions recorded yet")
    else:
        _print_stderr(f" Sessions:  {local_stats['session_count']}")
        if local_stats.get("last_session"):
            _print_stderr(f" Last:      {local_stats['last_session']}")

    # Cloud mode: fetch usage (FRD-CT-002)
    if mode == "cloud" and api_key:
        cloud_data = _fetch_cloud_usage(api_key)
        if cloud_data:
            _print_stderr(bar)
            usage = cloud_data.get("usage", {})
            tier = cloud_data.get("tier", "free")
            _print_stderr(f" Tier:      {tier}")
            _print_stderr(
                f" Traces:    {usage.get('traces_count', 0):,} / "
                f"{usage.get('traces_limit', 0):,} "
                f"({usage.get('traces_percentage', 0)}%)"
            )
            _print_stderr(
                f" Agents:    {usage.get('agents_count', 0)} / {usage.get('agents_limit', 0)}"
            )
            _print_stderr(
                f" Storage:   {usage.get('storage_mb', 0)} MB / "
                f"{usage.get('storage_limit_mb', 0)} MB "
                f"({usage.get('storage_percentage', 0)}%)"
            )
            _print_stderr(
                f" API calls: {usage.get('api_calls', 0):,} / "
                f"{usage.get('api_calls_limit', 0):,} "
                f"({usage.get('api_calls_percentage', 0)}%)"
            )
            # Show alerts if present
            for alert in cloud_data.get("alerts", []):
                icon = "\U0001f6a8" if alert.get("type") == "critical" else "\u26a0"
                _print_stderr(f" {icon} {alert.get('message', '')}")

            # Upgrade CTA for free tier
            if tier == "free":
                _print_stderr(bar)
                # TODO(billing): Update to Stripe Checkout URL when billing integration ships
                _print_stderr(" Upgrade for 10x capacity \u2192 https://clyrohq.com/pricing")
        else:
            _print_stderr(" Cloud unreachable \u2014 showing local data only")

    # Local mode CTA
    if mode == "local":
        _print_stderr(bar)
        _print_stderr(" Connect to cloud for team dashboard \u2192 set CLYRO_API_KEY")

    _print_stderr(bar)
    return 0


def _print_stderr(text: str) -> None:
    """Write to stderr with silent failure."""
    try:
        print(text, file=sys.stderr)
    except Exception:
        pass


def _read_local_stats() -> dict[str, Any] | None:
    """
    Read aggregate stats from local SQLite.

    Implements TDD §5.1 steps 3-7.
    Returns None if SQLite is unavailable.
    """
    try:
        from clyro.config import ClyroConfig
        from clyro.storage.sqlite import LocalStorage

        config = ClyroConfig()
        storage = LocalStorage(config)

        with storage._get_connection() as conn:
            # Session count from sync_status
            cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM sync_status")
            session_count = cursor.fetchone()[0] or 0

            # Last session timestamp from trace_buffer
            cursor = conn.execute("SELECT MAX(timestamp) FROM trace_buffer")
            row = cursor.fetchone()
            last_session = row[0] if row and row[0] else None

            # Detect adapter from last session payload
            adapter = "unknown"
            cursor = conn.execute("SELECT payload FROM trace_buffer ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    import json

                    payload = json.loads(row[0])
                    adapter = payload.get("framework", "unknown")
                except Exception:
                    pass

        return {
            "session_count": session_count,
            "last_session": last_session,
            "adapter": adapter,
        }
    except Exception:
        return None


def _read_policy_count() -> int:
    """Read local policy rule count from YAML config."""
    try:
        from clyro.local_policy import SDKPolicyConfig

        config = SDKPolicyConfig.from_default_path()
        return len(config.rules) if config and config.rules else 0
    except Exception:
        return 0


def _fetch_cloud_usage(api_key: str) -> dict[str, Any] | None:
    """
    Fetch usage data from cloud API.

    Implements TDD §5.2 steps 3-6.
    Returns None on any failure (FRD-CT-002 fallback).
    """
    try:
        from clyro.wrapper import _extract_org_id_from_jwt_api_key

        org_id = _extract_org_id_from_jwt_api_key(api_key)
        if org_id is None:
            return None

        endpoint = os.environ.get("CLYRO_ENDPOINT", DEFAULT_API_URL)
        url = f"{endpoint.rstrip('/')}/v1/organizations/{org_id}/usage"

        import httpx

        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                url,
                headers={"X-Clyro-API-Key": api_key},
            )
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    """
    Main CLI entry point.  Implements FRD-SOF-010.

    Registered in pyproject.toml as ``clyro-sdk = "clyro.cli:main"``.
    """
    parser = argparse.ArgumentParser(
        prog="clyro-sdk",
        description=f"clyro v{clyro.__version__} \u2014 Runtime governance for AI agents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"clyro {clyro.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", title="commands")

    # feedback subcommand
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="Submit feedback or report an issue",
    )
    feedback_parser.add_argument(
        "--message",
        "-m",
        type=str,
        default=None,
        help="Feedback text (prompted if interactive TTY)",
    )

    # status subcommand (FRD-CT-001, FRD-CT-002, FRD-CT-003)
    subparsers.add_parser(
        "status",
        help="Show mode, usage, adapter, policies, and session history",
    )

    args = parser.parse_args(argv)

    # Bare ``clyro-sdk`` command → print help (TDD §4.2)
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(0)

    if args.command == "feedback":
        sys.exit(_handle_feedback(args))

    if args.command == "status":
        sys.exit(_handle_status(args))


if __name__ == "__main__":
    main()
