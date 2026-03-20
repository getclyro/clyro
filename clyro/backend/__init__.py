# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Backend Integration (v1.1)
# Implements FRD-015, FRD-016, FRD-017, FRD-018, FRD-019

"""
Backend integration subpackage for dashboard trace sync, cloud policy
fetch, agent auto-registration, and offline event persistence.

Activated only when an API key is configured (dual-mode operation).
"""

__all__ = [
    "AgentRegistrar",
    "BackendSyncManager",
    "CircuitBreaker",
    "CloudPolicyFetcher",
    "ConnectivityDetector",
    "EventQueue",
    "HttpSyncClient",
    "RateLimitExhaustedError",
    "TraceEventFactory",
]
