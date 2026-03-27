# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — AI Agent Governance
# Implements PRD-001, PRD-002, FRD-009 (unified package), NFR-001 (lazy imports)

"""
Clyro SDK module exports.

This module provides the core SDK functionality for wrapping AI agents
with tracing, execution controls, and policy enforcement. The MCP wrapper
and Claude Code hooks subpackages are lazily loaded to avoid import
overhead when only the SDK core is needed.
"""

__version__ = "0.2.6"

from clyro.adapters.claude_agent_sdk import instrument_claude_agent
from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    AuthenticationError,
    BackendUnavailableError,
    ClyroConfigError,
    ClyroError,
    ClyroWrapError,
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    LoopDetectedError,
    PolicyViolationError,
    RateLimitExhaustedError,
    StepLimitExceededError,
)
from clyro.model_selector import COST_OPTIMIZATION_GUIDE, ModelSelector
from clyro.policy import ConsoleApprovalHandler
from clyro.session import Session
from clyro.trace import AgentStage, EventType, Framework, TraceEvent
from clyro.wrapper import WrappedAgent, configure, wrap

__all__ = [
    # Core functions
    "wrap",
    "configure",
    "instrument_claude_agent",
    # Classes
    "WrappedAgent",
    "ClyroConfig",
    "ExecutionControls",
    "Session",
    "ModelSelector",
    "ConsoleApprovalHandler",
    # Trace models
    "TraceEvent",
    "EventType",
    "Framework",
    "AgentStage",
    # Exceptions (SDK + consolidated from MCP)
    "ClyroError",
    "ClyroConfigError",
    "ClyroWrapError",
    "FrameworkVersionError",
    "ExecutionControlError",
    "StepLimitExceededError",
    "CostLimitExceededError",
    "LoopDetectedError",
    "PolicyViolationError",
    "AuthenticationError",
    "RateLimitExhaustedError",
    "BackendUnavailableError",
    # Documentation
    "COST_OPTIMIZATION_GUIDE",
    # Version
    "__version__",
]


# NFR-001: Lazy import strategy for subpackages.
# clyro.mcp and clyro.hooks are NOT eagerly loaded — they are loaded
# only when accessed via attribute lookup. This preserves the <200ms
# hook import budget and prevents cross-loading.
def __getattr__(name: str):
    if name == "mcp":
        from clyro import mcp as _mcp

        return _mcp
    if name == "hooks":
        from clyro import hooks as _hooks

        return _hooks
    raise AttributeError(f"module 'clyro' has no attribute {name!r}")
