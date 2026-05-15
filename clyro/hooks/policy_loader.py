# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Policy Loader
# Implements FRD-HK-007, FRD-HK-011

"""Load YAML config, merge with cloud policies, manage TTL cache."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import structlog

from clyro.backend.cloud_policy import CloudPolicyFetcher
from clyro.backend.http_client import AuthenticationError, HttpSyncClient
from clyro.config import PolicyRule

from .backend import circuit_can_execute, circuit_record_failure, circuit_record_success
from .config import HookConfig
from .constants import CLOUD_POLICY_TIMEOUT_SECONDS
from .models import PolicyCache, SessionState

logger = structlog.get_logger()


def _cache_is_fresh(cache: PolicyCache) -> bool:
    """Check if the policy cache is still within TTL."""
    if cache.fetched_at is None:
        return False
    now = datetime.now(UTC)
    fetched = cache.fetched_at
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)
    elapsed = (now - fetched).total_seconds()
    return elapsed < cache.ttl_seconds


def _policies_from_cache(cache: PolicyCache) -> list[PolicyRule] | None:
    """Reconstruct PolicyRule objects from cached dicts.

    Fail-closed: if ANY cached policy is corrupt, returns None to signal
    the caller should invalidate the cache and re-fetch. Silently skipping
    corrupt entries could drop security-critical rules.
    """
    rules = []
    for item in cache.merged_policies:
        try:
            rules.append(PolicyRule.model_validate(item))
        except Exception as e:
            logger.error("corrupt_cached_policy", error=str(e), policy=item)
            return None  # Invalidate entire cache
    return rules


async def _fetch_cloud_policies(
    config: HookConfig,
    state: SessionState,
    local_policies: list[PolicyRule],
) -> tuple[list[PolicyRule], str]:
    """Fetch and merge cloud policies. Fail-open on any error.

    Returns ``(merged_rules, resolved_default_action)``. The resolved default
    uses **cloud-wins** precedence: when any cloud policy declares a
    ``default_action``, it overrides the wrapper's local default. Among
    multiple cloud defaults, the most-restrictive wins. The local default
    applies only when no cloud policies were fetched.
    """
    api_key = config.resolved_api_key
    if not api_key:
        return local_policies, config.default_action

    # Check circuit breaker before making API call
    if not circuit_can_execute(state.circuit_breaker):
        logger.warning("circuit_open_skip_policy_fetch")
        return local_policies, config.default_action

    client = HttpSyncClient(
        api_key=api_key,
        base_url=config.resolved_api_url,
        timeout=CLOUD_POLICY_TIMEOUT_SECONDS,
    )
    try:
        fetcher = CloudPolicyFetcher(http_client=client)
        # FRD-HK-007: Use real agent_id for cloud policy fetching
        agent_id = state.agent_id
        if not agent_id:
            logger.warning("no_agent_id_for_policy_fetch", fallback="local_only")
            return local_policies, config.default_action
        merged, resolved_default = await fetcher.fetch_and_merge(
            agent_id=agent_id,
            local_policies=local_policies,
            timeout=CLOUD_POLICY_TIMEOUT_SECONDS,
            local_default_action=config.default_action,
        )
        circuit_record_success(state.circuit_breaker)
        return merged, resolved_default
    except AuthenticationError as e:
        logger.warning("cloud_policy_auth_error", status_code=e.status_code)
        state.cloud_disabled = True
        circuit_record_failure(state.circuit_breaker)
        return local_policies, config.default_action
    except Exception as e:
        logger.warning("cloud_policy_fetch_error", error=str(e))
        circuit_record_failure(state.circuit_breaker)
        return local_policies, config.default_action
    finally:
        await client.close()


def get_merged_policies(config: HookConfig, state: SessionState) -> list[PolicyRule]:
    """Return merged policy list (local YAML + cloud).

    FRD-HK-007: Uses TTL cache from session state. Falls back to local on failure.

    As a side effect, also updates ``config.default_action`` to the resolved
    value. Precedence is **cloud-wins**: the cloud's ``default_action``
    overrides the wrapper's local default whenever any cloud policy declared
    one. The local default only applies when no cloud policies were fetched.
    """
    # Collect all local policies (global + per-tool)
    local_policies: list[PolicyRule] = list(config.global_.policies)
    for tool_config in config.tools.values():
        local_policies.extend(tool_config.policies)

    # If no API key or cloud disabled, use local only
    if not config.resolved_api_key or state.cloud_disabled:
        return local_policies

    # If no agent_id yet, use local only (agent registration happens in CLI init)
    if not state.agent_id:
        return local_policies

    # Check cache freshness — None means cache is corrupt, force re-fetch
    if _cache_is_fresh(state.policy_cache):
        cached = _policies_from_cache(state.policy_cache)
        if cached is not None:
            # Restore the merged default_action from the cache so a fresh
            # config load on each event still honors the cloud's
            # default_action (cloud-wins precedence). Without this, a cache
            # hit would revert to the local-only default and drop the
            # cloud's centrally-mandated value.
            cached_default = state.policy_cache.resolved_default_action
            if cached_default in ("block", "allow"):
                config.default_action = cached_default
            return cached
        logger.warning("cache_invalidated_corrupt_entries", fallback="re-fetch")

    # Fetch from cloud
    try:
        merged, resolved_default = asyncio.run(_fetch_cloud_policies(config, state, local_policies))
        # Apply the resolved default_action to the wrapper config so the
        # local evaluator picks it up. Without this, the cloud policy's
        # default_action would be silently lost.
        config.default_action = resolved_default
        # Update cache in state — store the resolved default alongside the
        # rules so cache-hit on the next event can restore it.
        state.policy_cache = PolicyCache(
            fetched_at=datetime.now(UTC),
            ttl_seconds=config.policy_cache_ttl_seconds,
            merged_policies=[p.model_dump() for p in merged],
            resolved_default_action=resolved_default,
        )
        return merged
    except Exception as e:
        logger.warning("policy_merge_fallback", error=str(e))
        return local_policies
