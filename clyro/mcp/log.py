# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Structured Logging
# Implements colored stderr-only logging (consistent with Clyro SDK)

"""
Logging re-export for the MCP wrapper.

The shared structlog configuration (level filtering, stderr output) lives in
``clyro.config`` and runs at import time. This module re-exports ``get_logger``
so existing MCP modules that do ``from clyro.mcp.log import get_logger``
continue to work unchanged.
"""

from __future__ import annotations

import structlog

# Ensure shared config is applied (no-op if already imported via another path).
import clyro.config  # noqa: F401


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)
