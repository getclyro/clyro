# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Exception Hierarchy
# Implements PRD-001, PRD-002

"""
Exception classes for the Clyro SDK.

This module defines the complete exception hierarchy used throughout
the SDK for error handling and control flow.
"""

from typing import Any

from clyro.constants import ISSUE_TRACKER_URL


class ClyroError(Exception):
    """
    Base exception for all Clyro SDK errors.

    All Clyro-specific exceptions inherit from this class, allowing
    users to catch all SDK errors with a single except clause.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    # Implements FRD-SOF-009: error context enrichment
    _ISSUE_TRACKER = ISSUE_TRACKER_URL

    def __str__(self) -> str:
        base = f"{self.message} (details: {self.details})" if self.details else self.message
        return f"{base}\n  Report at {self._ISSUE_TRACKER}"


class ClyroConfigError(ClyroError):
    """
    Configuration is invalid or missing required values.

    Raised when:
    - Required configuration fields are missing
    - Configuration values fail validation
    - Invalid combination of configuration options
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, details)
        self.field = field
        self.value = value


class ClyroWrapError(ClyroError):
    """
    Agent wrapping failed.

    Raised when:
    - The agent is not callable
    - The agent type is not supported
    - Wrapping initialization fails
    """

    def __init__(
        self,
        message: str,
        agent_type: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if agent_type:
            details["agent_type"] = agent_type
        super().__init__(message, details)
        self.agent_type = agent_type


class FrameworkVersionError(ClyroError):
    """
    Framework version is not supported.

    Raised when the detected framework version falls outside
    the supported version range for the adapter.
    """

    def __init__(
        self,
        framework: str,
        version: str,
        supported: str,
        details: dict[str, Any] | None = None,
    ):
        message = f"{framework} {version} is not supported. Supported versions: {supported}"
        details = details or {}
        details.update(
            {
                "framework": framework,
                "detected_version": version,
                "supported_versions": supported,
            }
        )
        super().__init__(message, details)
        self.framework = framework
        self.version = version
        self.supported = supported


class ExecutionControlError(ClyroError):
    """
    Base class for execution control limit errors.

    Execution controls terminate agent execution when safety
    boundaries are exceeded (steps, cost, loops).
    """

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        step_number: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if session_id:
            details["session_id"] = session_id
        if step_number is not None:
            details["step_number"] = step_number
        super().__init__(message, details)
        self.session_id = session_id
        self.step_number = step_number


class StepLimitExceededError(ExecutionControlError):
    """
    Agent exceeded maximum step count.

    Raised when the configured step limit is reached to prevent
    runaway agent execution.
    """

    def __init__(
        self,
        limit: int,
        current_step: int,
        session_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        message = f"Step limit exceeded: {current_step} steps (limit: {limit})"
        details = details or {}
        details.update({"limit": limit, "current_step": current_step})
        super().__init__(message, session_id, current_step, details)
        self.limit = limit
        self.current_step = current_step


class CostLimitExceededError(ExecutionControlError):
    """
    Agent exceeded maximum cost bound.

    Raised when cumulative token costs exceed the configured
    budget to prevent unexpected expenses.
    """

    def __init__(
        self,
        limit_usd: float,
        current_cost_usd: float,
        session_id: str | None = None,
        step_number: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        message = f"Cost limit exceeded: ${current_cost_usd:.4f} (limit: ${limit_usd:.2f})"
        details = details or {}
        details.update({"limit_usd": limit_usd, "current_cost_usd": current_cost_usd})
        super().__init__(message, session_id, step_number, details)
        self.limit_usd = limit_usd
        self.current_cost_usd = current_cost_usd


class LoopDetectedError(ExecutionControlError):
    """
    Agent entered an infinite loop.

    Raised when the same state hash is detected repeatedly,
    indicating a repetitive execution pattern.
    """

    def __init__(
        self,
        iterations: int,
        state_hash: str,
        session_id: str | None = None,
        step_number: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        message = f"Loop detected: state repeated {iterations} times (hash: {state_hash[:8]}...)"
        details = details or {}
        details.update({"iterations": iterations, "state_hash": state_hash})
        super().__init__(message, session_id, step_number, details)
        self.iterations = iterations
        self.state_hash = state_hash


class PolicyViolationError(ClyroError):
    """
    Action violates a policy rule.

    Raised when an agent action is blocked by a policy rule
    to enforce business constraints.
    """

    def __init__(
        self,
        rule_id: str,
        rule_name: str,
        message: str,
        action_type: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        details.update({"rule_id": rule_id, "rule_name": rule_name})
        if action_type:
            details["action_type"] = action_type
        super().__init__(message, details)
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.action_type = action_type


class TraceError(ClyroError):
    """
    Trace capture or storage failed.

    This is typically a non-fatal error that is logged but
    does not interrupt agent execution (fail-open behavior).
    """

    def __init__(
        self,
        message: str,
        event_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if event_id:
            details["event_id"] = event_id
        super().__init__(message, details)
        self.event_id = event_id


class TransportError(ClyroError):
    """
    Network transport error occurred.

    Raised when communication with the backend fails after
    all retry attempts are exhausted.
    """

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if endpoint:
            details["endpoint"] = endpoint
        if status_code is not None:
            details["status_code"] = status_code
        super().__init__(message, details)
        self.endpoint = endpoint
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Consolidated from clyro_mcp.backend.http_client — Implements FRD-014
# ---------------------------------------------------------------------------


class AuthenticationError(ClyroError):
    """
    Raised when backend API authentication fails.

    Moved from ``clyro_mcp.backend.http_client``.
    """

    def __init__(
        self,
        message: str | int = "Authentication failed",
        details: dict[str, Any] | None = None,
    ):
        # Support AuthenticationError(status_code) shorthand used by http_client
        if isinstance(message, int):
            self.status_code = message
            super().__init__(f"Authentication failed (HTTP {message})", details)
        else:
            self.status_code = None
            super().__init__(message, details)


class RateLimitExhaustedError(ClyroError):
    """
    Raised when backend API rate limit is exceeded.

    Moved from ``clyro_mcp.backend.http_client``.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, details)
        self.retry_after = retry_after


class BackendUnavailableError(ClyroError):
    """
    Raised when the backend is unreachable (circuit breaker open).

    New in FRD-014.
    """

    def __init__(
        self,
        message: str = "Backend unavailable (circuit breaker open)",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
