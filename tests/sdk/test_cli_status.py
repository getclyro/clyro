# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for Clyro SDK CLI — Status Subcommand
# Implements FRD-CT-001, FRD-CT-002, FRD-CT-003

"""
Unit tests for ``clyro-sdk status`` subcommand.

Tests verify:
- Local mode output: mode, version, adapter, policies, sessions (FRD-CT-001)
- Cloud mode output: usage metrics from API (FRD-CT-002)
- Cloud fallback: API unreachable → local-only output (FRD-CT-002)
- Zero sessions display (FRD-CT-001)
- SQLite unavailable → degraded output (FRD-CT-001)
- Error handling: unexpected error → exit 1 + issue URL (FRD-CT-003)
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from clyro.cli import _handle_status, _status_internal, main


# =============================================================================
# Local Mode (FRD-CT-001)
# =============================================================================


class TestStatusLocalMode:
    """Tests for clyro-sdk status in local mode."""

    def test_local_mode_shows_mode(self, capsys):
        """Status shows 'local' when no API key."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 5, "last_session": "2026-03-23", "adapter": "langgraph"}), \
             patch("clyro.cli._read_policy_count", return_value=3):
            os.environ.pop("CLYRO_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Local Mode" in captured.err
        assert "local" in captured.err

    def test_local_mode_shows_sessions(self, capsys):
        """Status shows session count from SQLite."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 47, "last_session": "2026-03-23", "adapter": "langgraph"}), \
             patch("clyro.cli._read_policy_count", return_value=4):
            os.environ.pop("CLYRO_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "47" in captured.err
        assert "Policies:  4" in captured.err

    def test_local_mode_zero_sessions(self, capsys):
        """Show 'No sessions recorded yet' when SQLite has zero sessions."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 0, "last_session": None, "adapter": "unknown"}), \
             patch("clyro.cli._read_policy_count", return_value=0):
            os.environ.pop("CLYRO_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Sessions:  0" in captured.err
        assert "No sessions recorded yet" in captured.err

    def test_local_mode_sqlite_unavailable(self, capsys):
        """Show 'Local data unavailable' when SQLite can't be read."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value=None), \
             patch("clyro.cli._read_policy_count", return_value=0):
            os.environ.pop("CLYRO_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Local data unavailable" in captured.err

    def test_local_mode_shows_cloud_cta(self, capsys):
        """Local mode shows 'Connect to cloud' CTA."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 1, "last_session": None, "adapter": "generic"}), \
             patch("clyro.cli._read_policy_count", return_value=0):
            os.environ.pop("CLYRO_API_KEY", None)
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "CLYRO_API_KEY" in captured.err


# =============================================================================
# Cloud Mode (FRD-CT-002)
# =============================================================================


class TestStatusCloudMode:
    """Tests for clyro-sdk status in cloud mode."""

    def test_cloud_mode_shows_usage(self, capsys):
        """Status shows usage metrics when API succeeds."""
        usage_data = {
            "tier": "free",
            "usage": {
                "traces_count": 47231,
                "traces_limit": 100000,
                "traces_percentage": 47,
                "agents_count": 8,
                "agents_limit": 10,
                "storage_mb": 127,
                "storage_limit_mb": 500,
                "storage_percentage": 25,
                "api_calls": 12400,
                "api_calls_limit": 50000,
                "api_calls_percentage": 25,
            },
            "alerts": [],
        }

        with patch.dict(os.environ, {"CLYRO_API_KEY": "cly_test_key"}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 5, "last_session": None, "adapter": "langgraph"}), \
             patch("clyro.cli._read_policy_count", return_value=2), \
             patch("clyro.cli._fetch_cloud_usage", return_value=usage_data):
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Cloud Mode" in captured.err
        assert "47,231" in captured.err
        assert "free" in captured.err

    def test_cloud_mode_api_unreachable_fallback(self, capsys):
        """Show 'Cloud unreachable' when API fails."""
        with patch.dict(os.environ, {"CLYRO_API_KEY": "cly_test_key"}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 5, "last_session": None, "adapter": "generic"}), \
             patch("clyro.cli._read_policy_count", return_value=0), \
             patch("clyro.cli._fetch_cloud_usage", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Cloud unreachable" in captured.err

    def test_cloud_mode_free_tier_shows_upgrade_cta(self, capsys):
        """Free tier shows upgrade CTA."""
        usage_data = {
            "tier": "free",
            "usage": {
                "traces_count": 1000, "traces_limit": 100000, "traces_percentage": 1,
                "agents_count": 1, "agents_limit": 10,
                "storage_mb": 10, "storage_limit_mb": 500, "storage_percentage": 2,
                "api_calls": 100, "api_calls_limit": 50000, "api_calls_percentage": 0,
            },
            "alerts": [],
        }

        with patch.dict(os.environ, {"CLYRO_API_KEY": "cly_test_key"}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 1, "last_session": None, "adapter": "generic"}), \
             patch("clyro.cli._read_policy_count", return_value=0), \
             patch("clyro.cli._fetch_cloud_usage", return_value=usage_data):
            with pytest.raises(SystemExit) as exc_info:
                main(["status"])
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "clyrohq.com/pricing" in captured.err


# =============================================================================
# Error Handling (FRD-CT-003)
# =============================================================================


class TestStatusErrorHandling:
    """Tests for error handling — exit 1 + issue URL."""

    def test_unexpected_error_returns_1(self):
        """Unexpected exception → exit code 1."""
        with patch("clyro.cli._status_internal", side_effect=RuntimeError("boom")):
            args = MagicMock()
            result = _handle_status(args)
            assert result == 1

    def test_unexpected_error_shows_issue_url(self, capsys):
        """Unexpected exception → shows GitHub issue URL."""
        with patch("clyro.cli._status_internal", side_effect=RuntimeError("test error")):
            args = MagicMock()
            _handle_status(args)

        captured = capsys.readouterr()
        assert "github.com" in captured.err
        assert "test error" in captured.err

    def test_status_always_exits_0_on_success(self):
        """Normal execution returns exit code 0."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("clyro.cli._read_local_stats", return_value={"session_count": 1, "last_session": None, "adapter": "generic"}), \
             patch("clyro.cli._read_policy_count", return_value=0):
            os.environ.pop("CLYRO_API_KEY", None)
            result = _status_internal()
            assert result == 0
