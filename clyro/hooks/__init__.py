# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Implements FRD-002: Claude Code hooks subpackage
"""
clyro-hook: CLI tool invoked by Claude Code's hook system to enforce
governance policies on all tool calls (Bash, Edit, Write, Read, MCP, etc.).

Reuses the Prevention Stack from clyro.mcp (loop detection, step limits,
cost budgets, policy rules) adapted for the hook lifecycle.
"""

__version__ = "0.1.0"
