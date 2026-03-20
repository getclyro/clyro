# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Policy Enforcement
# Implements PRD-011, PRD-012

"""
Policy enforcement client for the Clyro SDK.

This module provides:
- PolicyClient: HTTP client for calling the backend policy evaluation endpoint
- PolicyEvaluator: High-level enforcement logic with fail-open/closed behavior
- PolicyDecision: Typed result from policy evaluation

The SDK calls POST /v1/policies/evaluate on the backend API, which handles
policy storage, Redis caching, and rule evaluation internally. The SDK
does not cache policies or access Redis directly.

Architecture:
    SDK (PolicyClient) → HTTP → Backend API → Redis Cache → PostgreSQL
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

import httpx
import structlog

from clyro.exceptions import PolicyViolationError
from clyro.trace import EventType, TraceEvent

if TYPE_CHECKING:
    from clyro.config import ClyroConfig


class ApprovalHandler(Protocol):
    """
    Protocol for handling require_approval policy decisions.

    Implementations receive the policy decision details and return
    True to approve (allow the action) or False to deny (block it).

    The SDK provides a built-in ConsoleApprovalHandler for interactive use.
    Users can implement custom handlers for webhooks, Slack, etc.
    """

    def __call__(self, decision: PolicyDecision, action_type: str) -> bool:
        """
        Handle an approval request.

        Args:
            decision: The policy evaluation result with rule details
            action_type: Type of action needing approval (e.g., "tool_call")

        Returns:
            True to approve the action, False to deny it
        """
        ...


class ConsoleApprovalHandler:
    """
    Interactive console-based approval handler.

    Prompts the user via stdin to approve or deny actions that
    require human approval. Suitable for local development and
    interactive agent sessions.

    Example:
        ```python
        config = ClyroConfig(
            api_key="cly_live_...",
            agent_name="my-agent",
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        wrapped = clyro.wrap(
            my_agent,
            approval_handler=ConsoleApprovalHandler(),
        )
        ```
    """

    def __call__(self, decision: PolicyDecision, action_type: str) -> bool:
        """Prompt the user for approval via console input."""
        print("\n" + "=" * 60)
        print("POLICY: Action requires human approval")
        print("=" * 60)
        print(f"  Action type : {action_type}")
        if decision.rule_name:
            print(f"  Rule        : {decision.rule_name}")
        if decision.message:
            print(f"  Reason      : {decision.message}")
        print("=" * 60)

        while True:
            response = input("Approve this action? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            print("Please enter 'y' or 'n'.")

logger = structlog.get_logger(__name__)

# Tighter timeouts than transport — policy checks are latency-sensitive
POLICY_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=10.0,
    write=5.0,
    pool=5.0,
)


@dataclass(frozen=True)
class PolicyDecision:
    """
    Result of a policy evaluation from the backend.

    Attributes:
        decision: One of "allow", "block", "require_approval"
        rule_id: ID of the rule that caused the decision (if not allow)
        rule_name: Name of the rule that caused the decision (if not allow)
        message: Human-readable explanation (if not allow)
        evaluated_rules: Number of rules the backend evaluated
        evaluation_time_ms: Backend evaluation latency in milliseconds
        rule_results: Per-rule evaluation details (verbose mode, FRD-BE-001)
    """

    decision: str
    rule_id: str | None = None
    rule_name: str | None = None
    message: str | None = None
    evaluated_rules: int = 0
    evaluation_time_ms: float = 0.0
    rule_results: list[dict[str, Any]] | None = None

    @property
    def is_allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def is_blocked(self) -> bool:
        return self.decision == "block"

    @property
    def requires_approval(self) -> bool:
        return self.decision == "require_approval"

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> PolicyDecision:
        """Create a PolicyDecision from the backend API response."""
        return cls(
            decision=data.get("decision", "block"),
            rule_id=data.get("rule_id"),
            rule_name=data.get("rule_name"),
            message=data.get("message"),
            evaluated_rules=data.get("evaluated_rules", 0),
            evaluation_time_ms=data.get("evaluation_time_ms", 0.0),
            rule_results=data.get("rule_results"),
        )

    @classmethod
    def allow(cls) -> PolicyDecision:
        """Create an allow decision (used for fail-open fallback)."""
        return cls(decision="allow")


class PolicyClient:
    """
    HTTP client for the backend policy evaluation endpoint.

    Provides both sync and async interfaces using separate httpx clients.
    Auth uses the same X-Clyro-API-Key header as the transport layer.
    """

    def __init__(self, config: ClyroConfig):
        self._config = config
        self._async_client: httpx.AsyncClient | None = None
        self._async_client_loop_id: int | None = None  # id() of the loop the client was created in
        self._sync_client: httpx.Client | None = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers with API key auth."""
        headers = {
            "User-Agent": "clyro-sdk/0.1.0",
            "Content-Type": "application/json",
        }
        if self._config.api_key:
            headers["X-Clyro-API-Key"] = self._config.api_key
        return headers

    def _build_payload(
        self,
        agent_id: UUID,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> dict[str, Any]:
        """Build the evaluation request payload matching backend schema."""
        payload: dict[str, Any] = {
            "agent_id": str(agent_id),
            "action": {
                "type": action_type,
                "parameters": parameters,
            },
        }

        if session_id is not None or step_number is not None:
            payload["context"] = {}
            if session_id is not None:
                payload["context"]["session_id"] = str(session_id)
            if step_number is not None:
                payload["context"]["step_number"] = step_number

        return payload

    async def evaluate_async(
        self,
        agent_id: UUID,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate an action against policies (async).

        Args:
            agent_id: Agent to evaluate policies for
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            session_id: Current session ID for audit trail
            step_number: Current step number for audit trail

        Returns:
            PolicyDecision with the backend's evaluation result

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        # Recreate client if closed or bound to a different event loop.
        # The Claude Agent SDK runs hooks in a different event loop than the
        # wrapper, and httpx.AsyncClient's connection pool is bound to the
        # loop it was created in. Reusing a client across loops causes
        # "Event loop is closed" errors.
        current_loop_id = id(asyncio.get_running_loop())
        needs_new_client = (
            self._async_client is None
            or self._async_client.is_closed
            or self._async_client_loop_id != current_loop_id
        )

        if needs_new_client:
            # Don't try to aclose() the old client — its event loop may be
            # closed, making aclose() itself throw. Just drop the reference.
            self._async_client = httpx.AsyncClient(
                timeout=POLICY_TIMEOUT,
                headers=self._get_headers(),
            )
            self._async_client_loop_id = current_loop_id

        payload = self._build_payload(
            agent_id, action_type, parameters, session_id, step_number
        )

        url = f"{self._config.endpoint}/v1/policies/evaluate"
        response = await self._async_client.post(url, json=payload)
        response.raise_for_status()

        return PolicyDecision.from_response(response.json())

    def evaluate_sync(
        self,
        agent_id: UUID,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate an action against policies (sync).

        Same as evaluate_async but uses synchronous httpx.Client.
        Used by LangGraph callbacks and sync wrapper execution path.

        Args:
            agent_id: Agent to evaluate policies for
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            session_id: Current session ID for audit trail
            step_number: Current step number for audit trail

        Returns:
            PolicyDecision with the backend's evaluation result

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                timeout=POLICY_TIMEOUT,
                headers=self._get_headers(),
            )

        payload = self._build_payload(
            agent_id, action_type, parameters, session_id, step_number
        )

        url = f"{self._config.endpoint}/v1/policies/evaluate"
        response = self._sync_client.post(url, json=payload)
        response.raise_for_status()

        return PolicyDecision.from_response(response.json())

    async def close_async(self) -> None:
        """Close the async HTTP client."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    def close_sync(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None


class PolicyEvaluator:
    """
    High-level policy enforcement logic.

    Calls PolicyClient, interprets the decision, emits audit events,
    and raises PolicyViolationError when actions are blocked.

    Respects config.fail_open for network error handling:
    - fail_open=True (default): Network errors → allow action
    - fail_open=False: Network errors → block action
    """

    # Sentinel to distinguish "not passed" from "explicitly None"
    _NO_HANDLER = object()

    def __init__(
        self,
        config: ClyroConfig,
        agent_id: UUID | None = None,
        org_id: UUID | None = None,
        approval_handler: ApprovalHandler | None | object = _NO_HANDLER,
    ):
        self._config = config
        self._agent_id = agent_id
        self._org_id = org_id
        self._client = PolicyClient(config)
        self._events: list[TraceEvent] = []
        self._disabled_reason: str | None = None

        # Auto-detect approval handler:
        # - Explicit handler passed → use it
        # - Explicitly None → no handler (block on require_approval)
        # - Not passed (default) + interactive terminal → ConsoleApprovalHandler
        # - Not passed (default) + non-interactive → no handler (block)
        if approval_handler is self._NO_HANDLER:
            if sys.stdin.isatty():
                self._approval_handler: ApprovalHandler | None = ConsoleApprovalHandler()
                logger.debug("approval_handler_auto_detected", handler="ConsoleApprovalHandler")
            else:
                self._approval_handler = None
        else:
            self._approval_handler = approval_handler  # type: ignore[assignment]

    @property
    def is_enabled(self) -> bool:
        """Check if policy enforcement is active."""
        return (
            self._config.controls.enable_policy_enforcement
            and self._config.api_key is not None
            and self._disabled_reason is None
        )

    def evaluate_sync(
        self,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate an action synchronously and enforce the decision.

        Args:
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            session_id: Current session ID for audit trail
            step_number: Current step number for audit trail

        Returns:
            PolicyDecision (only returned if allowed)

        Raises:
            PolicyViolationError: If action is blocked or requires approval
        """
        if not self.is_enabled:
            return PolicyDecision.allow()

        start_time = time.perf_counter()

        try:
            decision = self._client.evaluate_sync(
                agent_id=self._agent_id,
                action_type=action_type,
                parameters=parameters,
                session_id=session_id,
                step_number=step_number,
            )
        except Exception as e:
            return self._handle_error(e, action_type, session_id, step_number)

        sdk_latency_ms = (time.perf_counter() - start_time) * 1000

        self._log_decision(decision, action_type, sdk_latency_ms, session_id, step_number)
        self._emit_policy_event(decision, action_type, parameters, session_id, step_number)
        self._enforce_decision(decision, action_type)
        return decision

    async def evaluate_async(
        self,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate an action asynchronously and enforce the decision.

        Args:
            action_type: Type of action (e.g., "tool_call", "llm_call")
            parameters: Action parameters for rule evaluation
            session_id: Current session ID for audit trail
            step_number: Current step number for audit trail

        Returns:
            PolicyDecision (only returned if allowed)

        Raises:
            PolicyViolationError: If action is blocked or requires approval
        """
        if not self.is_enabled:
            return PolicyDecision.allow()

        start_time = time.perf_counter()

        try:
            decision = await self._client.evaluate_async(
                agent_id=self._agent_id,
                action_type=action_type,
                parameters=parameters,
                session_id=session_id,
                step_number=step_number,
            )
        except Exception as e:
            return self._handle_error(e, action_type, session_id, step_number)

        sdk_latency_ms = (time.perf_counter() - start_time) * 1000

        self._log_decision(decision, action_type, sdk_latency_ms, session_id, step_number)
        self._emit_policy_event(decision, action_type, parameters, session_id, step_number)
        self._enforce_decision(decision, action_type)
        return decision

    def _handle_error(
        self,
        error: Exception,
        action_type: str,
        session_id: UUID | None,
        step_number: int | None,
    ) -> PolicyDecision:
        """
        Handle policy evaluation errors with appropriate severity.

        HTTP 401/403 errors are auth/permission failures — the API key
        doesn't have policy:read scope. These are configuration errors,
        not policy violations. We auto-disable enforcement for this
        evaluator instance and log an error so the user can fix their
        API key. The agent continues running without policy checks.

        Network/transient errors (connection refused, timeouts, 5xx) are
        infrastructure failures where fail_open behavior applies:
        - fail_open=True (default): Log warning, allow action to proceed
        - fail_open=False: Raise PolicyViolationError to block the action
        """
        # HTTP 401/403 = auth/permission config error → disable enforcement
        is_auth_error = (
            isinstance(error, httpx.HTTPStatusError)
            and error.response.status_code in (401, 403)
        )

        if is_auth_error:
            status = error.response.status_code
            self._disabled_reason = f"HTTP {status}"
            logger.error(
                "policy_enforcement_disabled",
                reason=f"API key lacks policy:read scope (HTTP {status})",
                error=str(error),
                status_code=status,
                action_type=action_type,
                agent_id=str(self._agent_id),
                session_id=str(session_id) if session_id else None,
                hint="Policy enforcement auto-disabled for this session. "
                     "Fix API key permissions to re-enable.",
            )
            return PolicyDecision.allow()

        # Other HTTP client errors (400, 422, etc.) → respect fail_open
        # Network/transient errors → respect fail_open config
        logger.warning(
            "policy_evaluation_error",
            error=str(error),
            error_type=type(error).__name__,
            action_type=action_type,
            agent_id=str(self._agent_id),
            session_id=str(session_id) if session_id else None,
            step_number=step_number,
            fail_open=self._config.fail_open,
        )

        if self._config.fail_open:
            return PolicyDecision.allow()

        raise PolicyViolationError(
            rule_id="system_error",
            rule_name="Policy Evaluation Error",
            message=f"Policy evaluation failed: {error}. Action blocked (fail_open=False).",
            action_type=action_type,
        ) from error

    def _enforce_decision(self, decision: PolicyDecision, action_type: str) -> None:
        """
        Enforce a policy decision.

        - allow: passes through silently
        - block: always raises PolicyViolationError
        - require_approval: calls approval_handler if configured.
          If handler approves → passes through. If handler denies or
          no handler is configured → raises PolicyViolationError.
        """
        if decision.is_blocked:
            raise PolicyViolationError(
                rule_id=decision.rule_id or "unknown",
                rule_name=decision.rule_name or "Unknown Rule",
                message=decision.message or "Action blocked by policy",
                action_type=action_type,
            )

        if decision.requires_approval:
            if self._approval_handler is not None:
                try:
                    approved = self._approval_handler(decision, action_type)
                except Exception as e:
                    logger.warning(
                        "approval_handler_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        action_type=action_type,
                        rule_id=decision.rule_id,
                    )
                    approved = False

                if approved:
                    logger.info(
                        "policy_approval_granted",
                        action_type=action_type,
                        rule_id=decision.rule_id,
                        rule_name=decision.rule_name,
                    )
                    return

                logger.info(
                    "policy_approval_denied",
                    action_type=action_type,
                    rule_id=decision.rule_id,
                    rule_name=decision.rule_name,
                )

            raise PolicyViolationError(
                rule_id=decision.rule_id or "unknown",
                rule_name=decision.rule_name or "Unknown Rule",
                message=decision.message or "Action requires approval",
                action_type=action_type,
                details={"decision": "require_approval"},
            )

    def _log_decision(
        self,
        decision: PolicyDecision,
        action_type: str,
        sdk_latency_ms: float,
        session_id: UUID | None,
        step_number: int | None,
    ) -> None:
        """Log the policy evaluation result for observability."""
        logger.debug(
            "policy_evaluated",
            decision=decision.decision,
            action_type=action_type,
            agent_id=str(self._agent_id),
            rule_id=decision.rule_id,
            rule_name=decision.rule_name,
            evaluated_rules=decision.evaluated_rules,
            backend_latency_ms=round(decision.evaluation_time_ms, 2),
            sdk_latency_ms=round(sdk_latency_ms, 2),
            session_id=str(session_id) if session_id else None,
            step_number=step_number,
        )

    def _emit_policy_event(
        self,
        decision: PolicyDecision,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None,
        step_number: int | None,
    ) -> None:
        """Create and buffer a POLICY_CHECK event for the audit trail."""
        event = self.create_policy_check_event(
            decision, action_type, parameters, session_id, step_number,
        )
        self._events.append(event)

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear all buffered policy check events."""
        events = list(self._events)
        self._events.clear()
        return events

    def create_policy_check_event(
        self,
        decision: PolicyDecision,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> TraceEvent:
        """
        Create a POLICY_CHECK trace event for the audit trail.

        Args:
            decision: Result of policy evaluation
            action_type: Type of action that was evaluated
            parameters: Parameters that were evaluated
            session_id: Current session ID
            step_number: Current step number

        Returns:
            TraceEvent with type POLICY_CHECK
        """
        from uuid import uuid4

        metadata: dict[str, Any] = {
            "decision": decision.decision,
            "action_type": action_type,
            "rule_id": decision.rule_id,
            "rule_name": decision.rule_name,
            "message": decision.message,
            "evaluated_rules": decision.evaluated_rules,
            "evaluation_time_ms": decision.evaluation_time_ms,
            "parameters": parameters,
        }
        if decision.rule_results is not None:
            metadata["rule_results"] = decision.rule_results

        return TraceEvent(
            event_id=uuid4(),
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            session_id=session_id or uuid4(),
            agent_id=self._agent_id,
            step_number=step_number or 0,
            input_data={"action_type": action_type, "parameters": parameters},
            output_data={
                "decision": decision.decision,
                "rule_id": decision.rule_id,
                "rule_name": decision.rule_name,
                "message": decision.message,
            },
            metadata=metadata,
        )

    async def close_async(self) -> None:
        """Close async resources."""
        await self._client.close_async()

    def close_sync(self) -> None:
        """Close sync resources."""
        self._client.close_sync()


# ---------------------------------------------------------------------------
# Consolidated from clyro_mcp.policy_evaluator — Implements FRD-005
# Local YAML-based policy evaluation for MCP/hooks contexts.
# ---------------------------------------------------------------------------


def _resolve_local_parameter(
    arguments: dict[str, Any],
    path: str,
) -> tuple[bool, Any]:
    """
    Resolve a parameter path against tool call arguments.

    Supports simple paths (``"amount"``) and nested paths
    (``"order.quantity"``).
    """
    if path.startswith("*."):
        path = path[2:]

    parts = path.split(".")
    current: Any = arguments
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current


def _evaluate_local_rule(rule: Any, actual: Any) -> bool:
    """
    Evaluate a single local policy rule against a resolved value.

    Returns ``True`` if the rule is **violated** (call should be blocked).
    Supports 8 operators: max_value, min_value, equals, not_equals,
    in_list, not_in_list, contains, not_contains.
    """
    op = rule.operator
    expected = rule.value

    if op == "max_value":
        try:
            return float(actual) > float(expected)
        except (TypeError, ValueError):
            return True

    if op == "min_value":
        try:
            return float(actual) < float(expected)
        except (TypeError, ValueError):
            return True

    if op == "equals":
        return actual != expected

    if op == "not_equals":
        return actual == expected

    if op == "in_list":
        if not isinstance(expected, list):
            return False
        return actual not in expected

    if op == "not_in_list":
        if not isinstance(expected, list):
            return False
        return actual in expected

    if op == "contains":
        try:
            return str(expected) in str(actual)
        except (TypeError, ValueError):
            return False

    if op == "not_contains":
        try:
            return str(expected) not in str(actual)
        except (TypeError, ValueError):
            return False

    return False


class LocalPolicyEvaluator:
    """
    Evaluates tool call arguments against local YAML-defined policy rules.

    Consolidated from ``clyro_mcp.policy_evaluator.PolicyEvaluator``.
    Supports 8 operators: max_value, min_value, equals, not_equals,
    in_list, not_in_list, contains, not_contains.

    Evaluation order:
    1. Per-tool policies (matched by exact tool name).
    2. Global policies.
    A violation from either blocks the call.
    """

    OPERATORS = frozenset({
        "max_value", "min_value", "equals", "not_equals",
        "in_list", "not_in_list", "contains", "not_contains",
    })

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: WrapperConfig with tools and global_ policy rules.
        """
        self._config = config

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
        """
        Evaluate arguments against all applicable rules.

        Returns:
            ``(violated, details, rule_results)``
        """
        args = arguments or {}
        enforcement_violated = False
        enforcement_details: dict[str, Any] = {}
        rule_results: list[dict[str, Any]] = []

        all_rules: list[Any] = []
        tool_cfg = self._config.tools.get(tool_name)
        if tool_cfg:
            all_rules.extend(tool_cfg.policies)
        all_rules.extend(self._config.global_.policies)

        for rule in all_rules:
            try:
                rule_result = self._build_rule_result(rule, tool_name, args)
                rule_results.append(rule_result)

                if not enforcement_violated and rule_result["outcome"] == "triggered":
                    enforcement_violated = True
                    enforcement_details = {
                        "rule_name": rule.name or f"{rule.operator}({rule.parameter})",
                        "tool_name": tool_name,
                        "parameter": rule.parameter,
                        "operator": rule.operator,
                        "expected": rule.value,
                        "actual": rule_result["actual_value"],
                        "policy_id": rule.policy_id,
                    }
            except Exception:
                pass  # Graceful degradation

        return enforcement_violated, enforcement_details, rule_results

    @staticmethod
    def _build_rule_result(
        rule: Any,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a per-rule result dict for trace event enrichment."""
        found, actual = _resolve_local_parameter(arguments, rule.parameter)

        if not found:
            return {
                "policy_id": rule.policy_id,
                "policy_name": None,
                "rule_id": None,
                "rule_name": rule.name or f"{rule.operator}({rule.parameter})",
                "field": rule.parameter,
                "operator": rule.operator,
                "threshold": rule.value,
                "actual_value": None,
                "outcome": "skipped",
                "action": "allow",
                "message": None,
            }

        violated = _evaluate_local_rule(rule, actual)
        outcome = "triggered" if violated else "passed"
        action = "block" if violated else "allow"
        message = (
            f"Blocked: {rule.name or rule.parameter}"
            if violated
            else None
        )

        return {
            "policy_id": rule.policy_id,
            "policy_name": None,
            "rule_id": None,
            "rule_name": rule.name or f"{rule.operator}({rule.parameter})",
            "field": rule.parameter,
            "operator": rule.operator,
            "threshold": rule.value,
            "actual_value": actual,
            "outcome": outcome,
            "action": action,
            "message": message,
        }
