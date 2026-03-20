# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — CLI Entry Point
# Implements FRD-SOF-010

"""
CLI entry point for the ``clyro`` command.

Provides:
- ``clyro-sdk``              — print help with available subcommands
- ``clyro-sdk feedback``     — submit feedback or report an issue
- ``clyro-sdk --version``    — print SDK version

Uses argparse (stdlib, zero dependencies) with add_subparsers().
Future subcommands (``clyro status``, ``clyro connect``) can be added
without restructuring.
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

    args = parser.parse_args(argv)

    # Bare ``clyro`` command → print help (TDD §4.2)
    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(0)

    if args.command == "feedback":
        sys.exit(_handle_feedback(args))


if __name__ == "__main__":
    main()
