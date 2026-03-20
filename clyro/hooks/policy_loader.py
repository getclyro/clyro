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
) -> list[PolicyRule]:
    """Fetch and merge cloud policies. Fail-open on any error."""
    api_key = config.resolved_api_key
    if not api_key:
        return local_policies

    # Check circuit breaker before making API call
    if not circuit_can_execute(state.circuit_breaker):
        logger.warning("circuit_open_skip_policy_fetch")
        return local_policies

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
            return local_policies
        merged = await fetcher.fetch_and_merge(
            agent_id=agent_id,
            local_policies=local_policies,
            timeout=CLOUD_POLICY_TIMEOUT_SECONDS,
        )
        circuit_record_success(state.circuit_breaker)
        return merged
    except AuthenticationError as e:
        logger.warning("cloud_policy_auth_error", status_code=e.status_code)
        state.cloud_disabled = True
        circuit_record_failure(state.circuit_breaker)
        return local_policies
    except Exception as e:
        logger.warning("cloud_policy_fetch_error", error=str(e))
        circuit_record_failure(state.circuit_breaker)
        return local_policies
    finally:
        await client.close()


def get_merged_policies(config: HookConfig, state: SessionState) -> list[PolicyRule]:
    """Return merged policy list (local YAML + cloud).

    FRD-HK-007: Uses TTL cache from session state. Falls back to local on failure.
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
            return cached
        logger.warning("cache_invalidated_corrupt_entries", fallback="re-fetch")

    # Fetch from cloud
    try:
        merged = asyncio.run(_fetch_cloud_policies(config, state, local_policies))
        # Update cache in state
        state.policy_cache = PolicyCache(
            fetched_at=datetime.now(UTC),
            ttl_seconds=config.policy_cache_ttl_seconds,
            merged_policies=[p.model_dump() for p in merged],
        )
        return merged
    except Exception as e:
        logger.warning("policy_merge_fallback", error=str(e))
        return local_policies
