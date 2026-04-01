# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for Clyro SDK Quota Prompt Manager
# Implements FRD-CT-004, FRD-CT-005, FRD-CT-006

"""
Unit tests for upgrade prompts at quota boundary.

Tests verify:
- 80% warning threshold (FRD-CT-004)
- 95% critical threshold (FRD-CT-005)
- 100% reached limit message (FRD-CT-005)
- CLYRO_QUIET suppression (FRD-CT-006)
- Pro tier skip (FRD-CT-006)
- Local mode skip (FRD-CT-006)
- Once-per-session dedup
- API failure: skip silently
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from clyro.constants import APP_PRICING_URL
from clyro.quota_prompt import QuotaPromptManager, _format_count

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cloud_config():
    """Config in cloud mode with API key."""
    config = MagicMock()
    config.is_local_only.return_value = False
    config.api_key = "cly_test_key"
    config.endpoint = "https://api.test.com"
    return config


@pytest.fixture
def local_config():
    """Config in local mode."""
    config = MagicMock()
    config.is_local_only.return_value = True
    config.api_key = None
    return config


def _usage_response(tier="free", traces_pct=50, agents_count=5, agents_limit=10):
    """Build a mock usage API response."""
    return {
        "tier": tier,
        "usage": {
            "traces_count": traces_pct * 1000,
            "traces_limit": 100000,
            "traces_percentage": traces_pct,
            "storage_mb": 100,
            "storage_limit_mb": 500,
            "storage_percentage": 20,
            "api_calls": 10000,
            "api_calls_limit": 50000,
            "api_calls_percentage": 20,
            "agents_count": agents_count,
            "agents_limit": agents_limit,
        },
        "alerts": [],
    }


# =============================================================================
# Warning Threshold — 80% (FRD-CT-004)
# =============================================================================


class TestWarningThreshold:
    """Tests for 80% warning prompt."""

    def test_warning_at_80_percent(self, cloud_config, capsys):
        """Show warning when traces at 80%."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=80)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "80K/100K" in captured.err
        assert APP_PRICING_URL in captured.err

    def test_warning_at_agents_8_of_10(self, cloud_config, capsys):
        """Show warning when agents at 80% (8/10) — running total, not monthly."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=50, agents_count=8, agents_limit=10)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "8/10" in captured.err
        assert "agents" in captured.err

    def test_no_warning_below_80(self, cloud_config, capsys):
        """No warning when traces below 80%."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=79)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err


# =============================================================================
# Critical Threshold — 95% (FRD-CT-005)
# =============================================================================


class TestCriticalThreshold:
    """Tests for 95% critical prompt."""

    def test_critical_at_95_percent(self, cloud_config, capsys):
        """Show critical warning when traces at 95%."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=95)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "approaching your limit" in captured.err

    def test_reached_limit_at_100(self, cloud_config, capsys):
        """Show 'reached limit' when traces at 100%."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=100)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "reached" in captured.err.lower()


# =============================================================================
# Suppression Rules (FRD-CT-006)
# =============================================================================


class TestSuppression:
    """Tests for prompt suppression."""

    def test_suppressed_by_quiet(self, cloud_config, capsys):
        """No prompt when CLYRO_QUIET=true."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {"CLYRO_QUIET": "true"}), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=95)):
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err

    def test_suppressed_by_quiet_one(self, cloud_config, capsys):
        """No prompt when CLYRO_QUIET=1."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {"CLYRO_QUIET": "1"}), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=95)):
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err

    def test_suppressed_in_local_mode(self, local_config, capsys):
        """No prompt in local mode — no API available."""
        qpm = QuotaPromptManager(local_config)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err

    def test_suppressed_for_pro_tier(self, cloud_config, capsys):
        """No prompt for pro tier users."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(tier="pro", traces_pct=95)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err


# =============================================================================
# Dedup & Failure Handling
# =============================================================================


class TestDedupAndFailures:
    """Tests for once-per-session dedup and failure handling."""

    def test_once_per_session_per_metric(self, cloud_config, capsys):
        """Same metric warning shown only once per session."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=_usage_response(traces_pct=85)):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()
            qpm.check()  # Second call

        captured = capsys.readouterr()
        # Count occurrences — should appear once only
        assert captured.err.count("85K/100K") == 1

    def test_api_failure_skips_silently(self, cloud_config, capsys):
        """No prompt and no error when API call fails."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_fetch_usage", return_value=None):
            os.environ.pop("CLYRO_QUIET", None)
            qpm.check()

        captured = capsys.readouterr()
        assert "Clyro:" not in captured.err

    def test_never_crashes(self, cloud_config):
        """check() never raises exceptions — always swallows."""
        qpm = QuotaPromptManager(cloud_config)

        with patch.dict(os.environ, {}, clear=True), \
             patch.object(qpm, "_check_internal", side_effect=RuntimeError("boom")):
            os.environ.pop("CLYRO_QUIET", None)
            # Should NOT raise
            qpm.check()


# =============================================================================
# Helpers
# =============================================================================


class TestFormatCount:
    """Tests for count formatting helper."""

    def test_format_small(self):
        assert _format_count(500) == "500"

    def test_format_thousands(self):
        assert _format_count(80000) == "80K"

    def test_format_millions(self):
        assert _format_count(1500000) == "1.5M"
