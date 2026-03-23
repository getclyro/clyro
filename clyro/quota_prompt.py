# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Quota Prompt Manager
# Implements FRD-CT-004, FRD-CT-005, FRD-CT-006

"""
Upgrade prompts at free-tier quota boundaries.

Shows non-blocking warnings at session start when free-tier usage
reaches 80% (warning) or 95% (critical) of any metered limit.

Suppression (FRD-CT-006):
- CLYRO_QUIET=true/1/yes (case-insensitive) — follows existing SDK convention
- Pro tier users — no upgrade needed
- Local mode — no usage API available

Design (TDD §5.3):
- Single API call per session (check at session start)
- Once per session per metric (dedup via already_shown set)
- Informational only — never blocks SDK operations
- All exceptions swallowed (TDD §6.2)
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import structlog

from clyro.constants import DEFAULT_API_URL
from clyro.local_logger import _is_quiet

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)

# Upgrade URL (TDD §4.2)
# TODO(billing): Update to Stripe Checkout URL when billing integration ships
_UPGRADE_URL = "https://clyrohq.com/pricing"

# Threshold percentages (FRD-CT-004, FRD-CT-005)
_WARNING_THRESHOLD = 80
_CRITICAL_THRESHOLD = 95

# Metrics to check and their display labels
_METRICS = [
    ("traces", "traces_percentage", "traces_count", "traces_limit", "traces"),
    ("storage", "storage_percentage", "storage_mb", "storage_limit_mb", "MB storage"),
    ("api_calls", "api_calls_percentage", "api_calls", "api_calls_limit", "API calls"),
    ("agents", None, "agents_count", "agents_limit", "agents"),
]


def _write_stderr(text: str) -> None:
    """Write to stderr with silent failure."""
    try:
        print(text, file=sys.stderr)
    except Exception:
        pass


def _format_count(count: int) -> str:
    """Format a count for display (e.g., 80000 → 80K)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


class QuotaPromptManager:
    """
    Check usage thresholds and emit upgrade warnings.

    Implements FRD-CT-004 (80% warning), FRD-CT-005 (95% critical),
    FRD-CT-006 (suppression rules).

    Usage:
        qpm = QuotaPromptManager(config)
        qpm.check()  # Call at session start
    """

    def __init__(self, config: ClyroConfig) -> None:
        self._config = config
        self._already_shown: set[str] = set()

    def check(self) -> None:
        """
        Check usage and show upgrade prompts if thresholds are exceeded.

        Implements TDD §5.3 workflow.
        All exceptions swallowed — must never crash user code.
        """
        try:
            self._check_internal()
        except Exception:
            logger.debug("quota_prompt_check_failed", exc_info=True)

    def _check_internal(self) -> None:
        """Internal check logic — exceptions propagate to check() wrapper."""
        # Step 2: Skip in local mode (FRD-CT-006)
        # But if CLYRO_API_KEY is set in env, still check (config may not have it)
        if self._config.is_local_only() and not os.environ.get("CLYRO_API_KEY"):
            return

        # Step 3: Skip if CLYRO_QUIET (FRD-CT-006)
        if _is_quiet():
            return

        # Step 6: Fetch usage from API
        usage_data = self._fetch_usage()
        if usage_data is None:
            return  # API failure — skip silently (step 7)

        # Step 4: Skip if pro tier (FRD-CT-006)
        tier = usage_data.get("tier", "free")
        if tier != "free":
            return

        usage = usage_data.get("usage", {})

        # Step 8: Check each metric
        for metric_key, pct_field, count_field, limit_field, label in _METRICS:
            if metric_key in self._already_shown:
                continue

            current = usage.get(count_field, 0)
            limit = usage.get(limit_field, 0)
            if limit <= 0:
                continue

            # Calculate percentage (agents don't have a pct field in API)
            if pct_field and pct_field in usage:
                percentage = usage[pct_field]
            else:
                percentage = int((current / limit) * 100) if limit > 0 else 0

            if percentage >= 100:
                # FRD-CT-005: reached limit
                _write_stderr(
                    f"\U0001f6a8 Clyro: You've reached your {label} limit "
                    f"({_format_count(current)}/{_format_count(limit)}). "
                    f"Upgrade to continue \u2192 {_UPGRADE_URL}"
                )
                self._already_shown.add(metric_key)
            elif percentage >= _CRITICAL_THRESHOLD:
                # FRD-CT-005: critical warning
                _write_stderr(
                    f"\U0001f6a8 Clyro: You're at {_format_count(current)}/{_format_count(limit)} "
                    f"{label} this month ({percentage}%). "
                    f"You're approaching your limit \u2014 upgrade to avoid disruption "
                    f"\u2192 {_UPGRADE_URL}"
                )
                self._already_shown.add(metric_key)
            elif percentage >= _WARNING_THRESHOLD:
                # FRD-CT-004: warning
                _write_stderr(
                    f"\u26a0 Clyro: You're at {_format_count(current)}/{_format_count(limit)} "
                    f"{label} this month ({percentage}%). "
                    f"Upgrade for 10x capacity \u2192 {_UPGRADE_URL}"
                )
                self._already_shown.add(metric_key)

    def _fetch_usage(self) -> dict[str, Any] | None:
        """
        Fetch usage data from the cloud API.

        Returns parsed JSON response or None on any failure.
        """
        api_key = self._config.api_key or os.environ.get("CLYRO_API_KEY")
        if not api_key:
            return None

        try:
            from clyro.wrapper import _extract_org_id_from_jwt_api_key

            org_id = _extract_org_id_from_jwt_api_key(api_key)
            if org_id is None:
                return None

            endpoint = os.environ.get("CLYRO_ENDPOINT", self._config.endpoint or DEFAULT_API_URL)
            url = f"{endpoint.rstrip('/')}/v1/organizations/{org_id}/usage"

            import httpx

            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    url,
                    headers={"X-Clyro-API-Key": api_key},
                )
                response.raise_for_status()
                return response.json()

        except Exception:
            logger.debug("quota_prompt_fetch_failed", exc_info=True)
            return None
