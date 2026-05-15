# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Cloud Policy Fetcher
# Implements FRD-017

"""
Fetch policies from the Clyro backend API on startup and merge with
local YAML policies.

Merge rules (FRD-017):
- Local YAML policies override cloud policies with the same ``name``
  (rule-level override by name).
- Cloud policies with names not present in local YAML are added.
- Cloud policies with unsupported operators are skipped with warning.
- ``require_approval`` actions are converted to ``block`` (§10) — MCP /
  hooks do not run an approval handler.
- Each fetched cloud policy carries its own ``default_action``. The
  resolved ``default_action`` for the wrapper uses **cloud-wins**
  precedence: when any cloud policy declares a ``default_action``, the
  cloud's value is authoritative and the local wrapper's ``default_action``
  is ignored. Among multiple cloud defaults, the most-restrictive among
  cloud (``block`` over ``allow``) wins. The local ``default_action``
  applies only when no cloud policies were fetched (none attached to the
  agent, or fetch failed — preserving fail-open behavior).
- Fetch has a hard 2-second timeout (fail-open to local-only).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from clyro.backend.http_client import AuthenticationError, HttpSyncClient
from clyro.config import PolicyRule
from clyro.mcp.log import get_logger

logger = get_logger(__name__)

# Operators supported by the MCP wrapper (TDD §2.8)
_SUPPORTED_OPERATORS = {
    "max_value",
    "min_value",
    "equals",
    "not_equals",
    "in_list",
    "not_in_list",
    "contains",
    "not_contains",
}


class CloudPolicyFetcher:
    """
    Fetch and merge cloud policies with local YAML policies (FRD-017).

    On any failure (network, auth, timeout), returns local policies
    unchanged (fail-open, governance independence — NFR-007).

    Args:
        http_client: HTTP client for backend API calls.
    """

    def __init__(self, http_client: HttpSyncClient) -> None:
        self._http_client = http_client

    async def fetch_and_merge(
        self,
        agent_id: str | None,
        local_policies: list[PolicyRule],
        timeout: float = 2.0,
        local_default_action: str = "allow",
    ) -> tuple[list[PolicyRule], str]:
        """
        Fetch cloud policies, merge with local (FRD-017).

        Args:
            agent_id: Agent UUID for fetching agent-scoped policies.
            local_policies: Locally-defined rules from the wrapper YAML.
            timeout: Hard timeout for the fetch.
            local_default_action: The wrapper's local ``default_action``.
                Used as a floor when reconciling with cloud-policy defaults.

        Returns:
            ``(merged_rule_list, resolved_default_action)``. The resolved
            default uses **cloud-wins** precedence: when the cloud declared
            any ``default_action``, it overrides the local default; if
            multiple cloud policies disagree, the most-restrictive among
            cloud wins. Local default applies only when no cloud defaults
            were fetched. On any fetch failure: returns
            ``(local_policies, local_default_action)`` unchanged.
        """
        try:
            async with asyncio.timeout(timeout):
                if agent_id:
                    cloud_rules, cloud_defaults = await self._fetch_agent_policies(agent_id)
                else:
                    # TODO(v1.2): fetch org-level policies via GET /v1/policies
                    # when the backend API supports it. For v1.1, agent_id=None
                    # means registration failed → fall back to local-only (NFR-007).
                    cloud_rules, cloud_defaults = [], []
        except (TimeoutError, httpx.HTTPError, AuthenticationError) as exc:
            logger.warning("cloud_policy_fetch_failed", error=str(exc), fallback="local_only")
            return local_policies, local_default_action
        except Exception as exc:
            logger.warning(
                "cloud_policy_fetch_unexpected",
                error_type=type(exc).__name__,
                error=str(exc),
                fallback="local_only",
            )
            return local_policies, local_default_action
        if not cloud_rules and not cloud_defaults:
            return local_policies, local_default_action
        return self._merge(cloud_rules, local_policies, cloud_defaults, local_default_action)

    async def _fetch_agent_policies(self, agent_id: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Fetch policies for a specific agent. Returns (rules, defaults)."""
        response = await self._http_client.fetch_policies(agent_id)
        policies = response.get("policies", [])
        return self._extract_rules(policies)

    def _extract_rules(
        self, policies: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Extract rules from backend policy format to flat rule list.

        Also collects each policy's ``default_action`` so the wrapper can
        reconcile its own ``default_action`` against the cloud policies'.
        """
        rules: list[dict[str, Any]] = []
        cloud_defaults: list[str] = []
        for policy in policies:
            policy_name = policy.get("name", "")
            raw_rules = policy.get("rules", {})

            # Backend format: {"version": "1.0", "rules": [...], "default_action": ...}
            if isinstance(raw_rules, dict):
                policy_default = raw_rules.get("default_action")
                if policy_default in ("block", "allow"):
                    cloud_defaults.append(policy_default)
                raw_rules = raw_rules.get("rules", [])

            for rule in raw_rules:
                condition = rule.get("condition", {})
                rules.append(
                    {
                        "name": rule.get("name") or policy_name,
                        "parameter": condition.get("field", ""),
                        "operator": condition.get("operator", ""),
                        "value": condition.get("value"),
                        "action": rule.get("action", "block"),
                        "policy_id": str(policy.get("id", "")),
                    }
                )
        return rules, cloud_defaults

    def _merge(
        self,
        cloud_rules: list[dict[str, Any]],
        local_policies: list[PolicyRule],
        cloud_defaults: list[str],
        local_default_action: str,
    ) -> tuple[list[PolicyRule], str]:
        """
        Merge cloud rules with local policies (FRD-017).

        Local overrides cloud by ``name`` field match for individual rules.

        Resolved ``default_action`` uses **cloud-wins** precedence: when the
        cloud declared any ``default_action`` (one or more cloud policies
        attached to the agent), the cloud's value is authoritative — the
        local wrapper's ``default_action`` is ignored. If multiple cloud
        policies declare conflicting defaults, the most-restrictive among
        cloud wins (``block`` over ``allow``). The local default applies
        only when no cloud defaults were fetched (no policies for this
        agent, or fetch failed — preserving fail-open local-only behavior).
        """
        # Build set of local policy names for override detection
        local_names = {p.name for p in local_policies if p.name}

        merged = list(local_policies)

        for rule in cloud_rules:
            name = rule.get("name")
            operator = rule.get("operator", "")

            # Skip rules whose name matches a local policy (local overrides cloud)
            if name and name in local_names:
                continue

            # Skip unsupported operators with warning
            if operator not in _SUPPORTED_OPERATORS:
                logger.warning(
                    "cloud_policy_unsupported_operator",
                    rule=name,
                    operator=operator,
                )
                continue

            # MCP/hooks do not support approval workflows — downgrade to block.
            action = rule.get("action", "block")
            if action == "require_approval":
                action = "block"
            if action not in {"block", "allow"}:
                action = "block"

            try:
                policy_rule = PolicyRule(
                    parameter=rule.get("parameter", ""),
                    operator=operator,
                    value=rule.get("value"),
                    name=name,
                    policy_id=rule.get("policy_id") or None,
                    action=action,
                )
                merged.append(policy_rule)
            except (ValueError, TypeError) as exc:
                logger.warning("cloud_policy_invalid_rule", rule=name, error=str(exc))

        # Cloud-wins: when the cloud declared any default_action, the cloud's
        # value is authoritative and the local default is ignored. Among
        # multiple cloud defaults, the most-restrictive (block) wins.
        # Fall back to the local default only when no cloud defaults exist.
        if cloud_defaults:
            resolved_default = "block" if "block" in cloud_defaults else "allow"
        else:
            resolved_default = local_default_action

        return merged, resolved_default
