# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Local YAML Policy Evaluation
# Implements FRD-SOF-001, FRD-SOF-002, FRD-SOF-003, NFR-001 through NFR-006

"""
Local YAML-based policy models, loader, and evaluator for the SDK.

This module provides:
- SDKPolicyRule: Extends PolicyRule with SDK-specific ``action`` field
- SDKPolicyConfig: Top-level YAML schema model (version, global, actions)
- load_sdk_policies(): YAML loader with template creation and caching
- SDKLocalPolicyEvaluator: Evaluates action parameters against loaded rules,
  supporting both sync and async paths with identical decisions.

Design principles:
- Zero-touch local mode: first run creates a template YAML automatically
- Fail-open per rule: a single rule exception skips that rule, not the set
- Process-level cache: policies loaded once, no mid-session reload (NFR-006)
- Identical evaluation semantics to MCP wrapper's 8-operator set
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from clyro.config import PolicyRule
from clyro.constants import DOCS_POLICIES_URL
from clyro.exceptions import ClyroConfigError, PolicyViolationError
from clyro.policy import (
    ApprovalHandler,
    ConsoleApprovalHandler,
    PolicyDecision,
    _evaluate_local_rule,
    _resolve_local_parameter,
)
from clyro.trace import EventType, TraceEvent

logger = structlog.get_logger(__name__)


def _warn_stderr(msg: str) -> None:
    """Write a warning to stderr, respecting CLYRO_QUIET. Implements TDD §6.2."""
    if os.environ.get("CLYRO_QUIET", "").lower() in ("true", "1", "yes"):
        return
    try:
        print(msg, file=sys.stderr)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# §3.1  SDKPolicyRule — extends shared PolicyRule with ``action`` field
# Implements FRD-SOF-001
# ---------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({"block", "require_approval"})


class SDKPolicyRule(PolicyRule):
    """
    SDK-specific policy rule with ``action`` field.

    Inherits the 5 base fields (parameter, operator, value, name, policy_id)
    and operator validation from ``PolicyRule``.  Adds ``action`` which
    controls enforcement behaviour when the rule is violated.

    ``model_config = ConfigDict(extra="ignore")`` allows shared YAML files
    to include future MCP-only fields without breaking the SDK.
    """

    model_config = ConfigDict(extra="ignore")

    action: str = Field(
        default="block",
        description="Enforcement action when rule is violated: 'block' or 'require_approval'",
    )

    # Implements FRD-SOF-001: action value validation
    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in _VALID_ACTIONS:
            raise ValueError(f"Unknown policy action: '{v}'. Supported: block, require_approval")
        return v


# ---------------------------------------------------------------------------
# §3.2  SDKPolicyConfig — top-level YAML schema
# Implements FRD-SOF-001, FRD-SOF-003
# ---------------------------------------------------------------------------


class ActionPolicies(BaseModel):
    """Policy list for a single action type (llm_call, tool_call, agent_execution)."""

    model_config = ConfigDict(extra="ignore")
    policies: list[SDKPolicyRule] = Field(default_factory=list)


class GlobalPolicies(BaseModel):
    """Global policies applied to all action types."""

    model_config = ConfigDict(extra="ignore")
    policies: list[SDKPolicyRule] = Field(default_factory=list)


class SDKPolicyConfig(BaseModel):
    """
    Top-level YAML schema model for SDK local policies.

    Implements FRD-SOF-003: per-action-type and global policy sections.
    Unknown action-type keys under ``actions`` are silently ignored
    (forward-compatible).
    """

    model_config = ConfigDict(extra="ignore")

    version: int = Field(description="Schema version, must be 1")
    global_: GlobalPolicies | None = Field(default=None, alias="global")
    actions: dict[str, ActionPolicies] | None = Field(default=None)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported policy file version: {v}. Expected: 1")
        return v


# ---------------------------------------------------------------------------
# YAML loader — template creation + caching
# Implements FRD-SOF-001, NFR-002, NFR-005, NFR-006
# ---------------------------------------------------------------------------

# §4.1  Default template written on first run
_DEFAULT_TEMPLATE = """\
# Clyro SDK Policy Configuration
# Documentation: {DOCS_POLICIES_URL}
#
# Rules are evaluated per action type (llm_call, tool_call, agent_execution)
# and globally. Per-action rules are checked first, then global rules.
#
# Supported operators:
#   max_value, min_value, equals, not_equals,
#   in_list, not_in_list, contains, not_contains
#
# Example:
# actions:
#   llm_call:
#     policies:
#       - parameter: "model"
#         operator: "in_list"
#         value: ["claude-sonnet-4-5-20250514"]
#         name: "allowed_models"

version: 1

global:
  policies: []
"""

# Process-level cache (NFR-006: no mid-session reload)
_loaded_config: SDKPolicyConfig | None = None
_cache_populated: bool = False

# Hardcoded policy file path (NFR-005: no user-configurable path in v1)
_POLICY_DIR = Path.home() / ".clyro" / "sdk"
_POLICY_FILE = _POLICY_DIR / "policies.yaml"


def _get_policy_path() -> Path:
    """Resolve the policy file path with directory verification.

    Implements NFR-005: path traversal prevention.
    The resolved path must reside under the expected policy directory.
    """
    resolved = _POLICY_FILE.resolve()
    expected_dir = _POLICY_DIR.resolve()
    if not resolved.is_relative_to(expected_dir):
        raise ClyroConfigError(
            message=f"Policy file path escapes policy directory: {resolved}",
        )
    return resolved


def load_sdk_policies() -> SDKPolicyConfig:
    """
    Load SDK policies from ``~/.clyro/sdk/policies.yaml``.

    Implements FRD-SOF-001 (YAML loading), NFR-005 (safe_load),
    NFR-006 (process-level cache).

    Returns:
        Parsed SDKPolicyConfig (may have zero rules).

    Raises:
        ClyroConfigError: If version is wrong or Pydantic validation fails.
    """
    global _loaded_config, _cache_populated

    # Return cached config on subsequent calls (NFR-006)
    if _cache_populated:
        return _loaded_config  # type: ignore[return-value]

    empty_config = SDKPolicyConfig(version=1)

    policy_path = _get_policy_path()

    # File doesn't exist → create template, return empty
    if not policy_path.exists():
        try:
            _POLICY_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            policy_path.write_text(
                _DEFAULT_TEMPLATE.format(DOCS_POLICIES_URL=DOCS_POLICIES_URL),
                encoding="utf-8",
            )
            os.chmod(str(policy_path), 0o644)
        except PermissionError:
            _warn_stderr(
                "[clyro] Warning: cannot create policy directory "
                f"{_POLICY_DIR} — proceeding with zero rules"
            )
        except OSError as exc:
            _warn_stderr(f"[clyro] Warning: cannot write policy template: {exc}")
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    # File is a directory (edge case from TDD §13.4)
    if policy_path.is_dir():
        _warn_stderr(
            f"[clyro] Warning: {policy_path} is a directory, not a file "
            "— proceeding with zero rules"
        )
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    # Read file
    try:
        content = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        _warn_stderr(f"[clyro] Warning: cannot read policy file: {exc}")
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    # Empty file → zero rules
    if not content.strip():
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    # Parse YAML (NFR-005: safe_load only)
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        _warn_stderr(
            f"[clyro] Warning: invalid YAML in {policy_path}: {exc} — proceeding with zero rules"
        )
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    if data is None or not isinstance(data, dict):
        _loaded_config = empty_config
        _cache_populated = True
        return empty_config

    # Validate with Pydantic — bad version or invalid fields → ClyroConfigError
    from pydantic import ValidationError

    try:
        config = SDKPolicyConfig.model_validate(data)
    except ValidationError as exc:
        raise ClyroConfigError(
            message=f"Invalid policy configuration in {policy_path}",
            details={"errors": exc.errors()},
        ) from exc

    _loaded_config = config
    _cache_populated = True
    return config


def reset_sdk_policy_cache() -> None:
    """Reset the policy cache (for testing)."""
    global _loaded_config, _cache_populated
    _loaded_config = None
    _cache_populated = False


# ---------------------------------------------------------------------------
# §3.4  LocalPolicyEvaluationResult — internal data class
# Implements FRD-SOF-002
# ---------------------------------------------------------------------------


@dataclass
class LocalPolicyEvaluationResult:
    """Internal result from local policy evaluation."""

    violated: bool
    decision: str  # "allow" or "block"
    violation_details: dict[str, Any] | None = None
    rule_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# §2 C2  SDKLocalPolicyEvaluator
# Implements FRD-SOF-002, NFR-001
# ---------------------------------------------------------------------------


class SDKLocalPolicyEvaluator:
    """
    Evaluates action parameters against locally loaded SDK YAML policies.

    Supports the ``action`` field on rules:
    - ``block``: violated rule immediately blocks (short-circuit)
    - ``require_approval``: invokes ApprovalHandler; denied → block

    Both ``evaluate_sync()`` and ``evaluate_async()`` call the same
    internal ``_evaluate()`` method to guarantee identical decisions
    (FRD-SOF-002 async parity requirement).
    """

    # Sentinel for "not passed" vs "explicitly None"
    _NO_HANDLER = object()

    def __init__(
        self,
        approval_handler: ApprovalHandler | None | object = _NO_HANDLER,
    ) -> None:
        # Auto-detect approval handler (same logic as cloud PolicyEvaluator)
        if approval_handler is self._NO_HANDLER:
            try:
                if sys.stdin.isatty():
                    self._approval_handler: ApprovalHandler | None = ConsoleApprovalHandler()
                else:
                    self._approval_handler = None
            except Exception:
                self._approval_handler = None
        else:
            self._approval_handler = approval_handler  # type: ignore[assignment]

        self._events: list[TraceEvent] = []

    def _evaluate(
        self,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> LocalPolicyEvaluationResult:
        """
        Core evaluation logic shared by sync and async paths.

        Implements FRD-SOF-002 evaluation logic and FRD-SOF-003 ordering:
        per-action-type rules first, then global rules.
        """
        config = load_sdk_policies()

        # Collect applicable rules: per-action first, then global (FRD-SOF-003)
        all_rules: list[SDKPolicyRule] = []
        if config.actions and action_type in config.actions:
            all_rules.extend(config.actions[action_type].policies)
        if config.global_ is not None:
            all_rules.extend(config.global_.policies)

        # Zero rules → allow (FRD-SOF-002)
        if not all_rules:
            result = LocalPolicyEvaluationResult(
                violated=False,
                decision="allow",
            )
            self._emit_policy_event(result, action_type, parameters, session_id, step_number)
            return result

        rule_results: list[dict[str, Any]] = []
        violated = False
        violation_details: dict[str, Any] | None = None

        for rule in all_rules:
            try:
                # Resolve parameter
                found, actual = _resolve_local_parameter(parameters, rule.parameter)
                if not found:
                    rule_results.append(
                        self._build_rule_result(
                            rule,
                            actual=None,
                            outcome="skipped",
                            action_taken="allow",
                        )
                    )
                    continue

                # Evaluate rule
                is_violated = _evaluate_local_rule(rule, actual)

                if not is_violated:
                    rule_results.append(
                        self._build_rule_result(
                            rule,
                            actual=actual,
                            outcome="passed",
                            action_taken="allow",
                        )
                    )
                    continue

                # Implements FRD-SOF-002: action-based enforcement
                should_block = self._enforce_violated_rule(rule, action_type)
                if not should_block:
                    # require_approval was approved → continue to next rule
                    rule_results.append(
                        self._build_rule_result(
                            rule,
                            actual=actual,
                            outcome="approved",
                            action_taken="allow",
                        )
                    )
                    continue

                # Block: record and short-circuit
                violated = True
                violation_details = self._build_violation_details(rule, actual, action_type)
                rule_results.append(
                    self._build_rule_result(
                        rule,
                        actual=actual,
                        outcome="triggered",
                        action_taken="block",
                    )
                )
                break

            except Exception as exc:
                # Fail-open per rule: skip and continue (FRD-SOF-002)
                logger.warning(
                    "clyro_local_policy_rule_error",
                    rule_name=rule.name,
                    error=str(exc),
                    fail_open=True,
                )
                rule_results.append(
                    self._build_rule_result(
                        rule,
                        actual=None,
                        outcome="skipped",
                        action_taken="allow",
                        message=f"Error: {exc}",
                    )
                )
                continue

        result = LocalPolicyEvaluationResult(
            violated=violated,
            decision="block" if violated else "allow",
            violation_details=violation_details,
            rule_results=rule_results,
        )
        self._emit_policy_event(result, action_type, parameters, session_id, step_number)
        return result

    def evaluate_sync(
        self,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate synchronously.  Implements FRD-SOF-002.

        Raises:
            PolicyViolationError: If the action is blocked.
        """
        result = self._evaluate(action_type, parameters, session_id, step_number)

        if result.violated and result.violation_details:
            raise PolicyViolationError(
                rule_id=result.violation_details.get("policy_id", "local"),
                rule_name=result.violation_details.get("rule_name", "unknown"),
                message=(
                    f"Policy violation: rule '{result.violation_details.get('rule_name')}' "
                    f"blocked {action_type}"
                ),
                action_type=action_type,
                details=result.violation_details,
            )

        return PolicyDecision(
            decision="allow",
            evaluated_rules=len(result.rule_results),
            rule_results=result.rule_results,
        )

    async def evaluate_async(
        self,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None = None,
        step_number: int | None = None,
    ) -> PolicyDecision:
        """
        Evaluate asynchronously.  Implements FRD-SOF-002 async parity.

        The local evaluator is purely synchronous (no I/O). The async
        method calls the sync logic directly.

        Raises:
            PolicyViolationError: If the action is blocked.
        """
        # Pure CPU — no await needed (TDD §5 W2)
        return self.evaluate_sync(action_type, parameters, session_id, step_number)

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear all buffered policy check events."""
        events = list(self._events)
        self._events.clear()
        return events

    def close_sync(self) -> None:
        """No-op — local evaluator has no resources to close."""

    async def close_async(self) -> None:
        """No-op — local evaluator has no resources to close."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_violated_rule(self, rule: SDKPolicyRule, action_type: str) -> bool:
        """Decide whether a violated rule should block.

        Returns True if the action should be blocked, False if approved
        (only possible for ``require_approval`` with an approval handler).
        """
        if rule.action != "require_approval":
            return True  # "block" → always block

        if self._approval_handler is None:
            return True  # No handler (non-TTY) → treat as block

        decision_obj = PolicyDecision(
            decision="require_approval",
            rule_id=rule.policy_id or "local",
            rule_name=rule.name or f"{rule.operator}({rule.parameter})",
            message=f"Rule '{rule.name or rule.parameter}' requires approval",
        )
        try:
            approved = self._approval_handler(decision_obj, action_type)
        except Exception:
            approved = False
        return not approved

    @staticmethod
    def _build_rule_result(
        rule: SDKPolicyRule,
        *,
        actual: Any,
        outcome: str,
        action_taken: str,
        message: str | None = None,
    ) -> dict[str, Any]:
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
            "action": action_taken,
            "message": message,
        }

    @staticmethod
    def _build_violation_details(
        rule: SDKPolicyRule,
        actual: Any,
        action_type: str,
    ) -> dict[str, Any]:
        return {
            "rule_name": rule.name or f"{rule.operator}({rule.parameter})",
            "action_type": action_type,
            "parameter": rule.parameter,
            "operator": rule.operator,
            "expected": rule.value,
            "actual": actual,
            "policy_id": rule.policy_id,
        }

    def _emit_policy_event(
        self,
        result: LocalPolicyEvaluationResult,
        action_type: str,
        parameters: dict[str, Any],
        session_id: UUID | None,
        step_number: int | None,
    ) -> None:
        """Create and buffer a POLICY_CHECK trace event for the audit trail."""
        metadata: dict[str, Any] = {
            "decision": result.decision,
            "action_type": action_type,
            "evaluated_rules": len(result.rule_results),
            "parameters": parameters,
            "rule_results": result.rule_results,
        }
        if result.violation_details:
            metadata["rule_name"] = result.violation_details.get("rule_name")
            metadata["rule_id"] = result.violation_details.get("policy_id")

        event = TraceEvent(
            event_id=uuid4(),
            event_type=EventType.POLICY_CHECK,
            event_name="policy_check",
            session_id=session_id or uuid4(),
            step_number=step_number or 0,
            input_data={"action_type": action_type, "parameters": parameters},
            output_data={
                "decision": result.decision,
                "rule_name": metadata.get("rule_name"),
                "evaluated_rules": len(result.rule_results),
            },
            metadata=metadata,
        )
        self._events.append(event)
