# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Structured Logging
# Implements colored stderr-only logging (consistent with Clyro SDK)

"""
Structured logging configuration for the MCP wrapper.

All output goes to **stderr only** — stdout is reserved for JSON-RPC.
Configuration runs at import time so every module gets stderr logging
regardless of whether it enters through the CLI or tests.

Uses a late-binding stderr factory so that pytest's capsys/capfd
fixture replacements are respected at write time.
"""

from __future__ import annotations

import sys

import structlog


class _StderrLoggerFactory:
    """Logger factory that resolves ``sys.stderr`` at write time.

    Unlike ``PrintLoggerFactory(file=sys.stderr)`` which binds at
    configure time, this creates a new ``PrintLogger`` on each call
    so it always uses the *current* ``sys.stderr`` — important for
    pytest's capsys/capfd which monkeypatch ``sys.stderr`` after import.
    """

    def __call__(self, *args: object, **kwargs: object) -> structlog.PrintLogger:
        return structlog.PrintLogger(file=sys.stderr)


# Configure at import time — ensures stderr output even in tests.
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=_StderrLoggerFactory(),
)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)
