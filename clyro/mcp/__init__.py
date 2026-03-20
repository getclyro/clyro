# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Implements FRD-001: MCP wrapper subpackage
"""
Clyro MCP Wrapper — stdio JSON-RPC proxy applying the Clyro Prevention Stack
to MCP tool calls.
"""

__all__ = [
    "AuditLogger",
    "McpSession",
    "MessageRouter",
    "PreventionStack",
    "StdioTransport",
    "WrapperConfig",
]

from clyro.config import WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.prevention import PreventionStack
from clyro.mcp.router import MessageRouter
from clyro.mcp.session import McpSession
from clyro.mcp.transport import StdioTransport
