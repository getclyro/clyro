# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for Clyro SDK Telemetry Client
# Implements FRD-CT-008, FRD-CT-009, FRD-CT-010, FRD-CT-011

"""
Unit tests for anonymous telemetry client.

Tests verify:
- Opt-in gating: only exact "true" enables telemetry (FRD-CT-011)
- Stderr audit log before sending (FRD-CT-010)
- Payload schema: only specified fields, no PII (FRD-CT-009)
- Network failure handling: swallow all exceptions (FRD-CT-008)
- Timeout handling (FRD-CT-008)
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from clyro.telemetry_client import (
    _collect_telemetry_payload,
    _is_telemetry_enabled,
    submit_telemetry,
)


# =============================================================================
# Opt-In Gating (FRD-CT-011)
# =============================================================================


class TestTelemetryOptIn:
    """Tests for strict case-sensitive opt-in."""

    def test_enabled_exact_true(self):
        """Only exact 'true' enables telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "true"}):
            assert _is_telemetry_enabled() is True

    def test_disabled_by_default(self):
        """Telemetry disabled when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_TELEMETRY", None)
            assert _is_telemetry_enabled() is False

    def test_disabled_capital_true(self):
        """'True' (capital T) does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "True"}):
            assert _is_telemetry_enabled() is False

    def test_disabled_uppercase_true(self):
        """'TRUE' does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "TRUE"}):
            assert _is_telemetry_enabled() is False

    def test_disabled_one(self):
        """'1' does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "1"}):
            assert _is_telemetry_enabled() is False

    def test_disabled_yes(self):
        """'yes' does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "yes"}):
            assert _is_telemetry_enabled() is False

    def test_disabled_on(self):
        """'on' does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "on"}):
            assert _is_telemetry_enabled() is False

    def test_disabled_empty_string(self):
        """Empty string does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": ""}):
            assert _is_telemetry_enabled() is False

    def test_disabled_false(self):
        """'false' does NOT enable telemetry."""
        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "false"}):
            assert _is_telemetry_enabled() is False


# =============================================================================
# Payload Schema (FRD-CT-009)
# =============================================================================


class TestTelemetryPayload:
    """Tests for telemetry payload content."""

    def test_payload_has_required_fields(self):
        """Payload contains all specified fields."""
        config = MagicMock()
        session = MagicMock()
        session.framework = MagicMock(value="langgraph")

        payload = _collect_telemetry_payload(config, session=session, session_count=10, error_count=2)

        assert "sdk_version" in payload
        assert "python_version" in payload
        assert "framework" in payload
        assert "adapter" in payload
        assert "os" in payload
        assert "session_count" in payload
        assert "error_count" in payload
        assert "timestamp" in payload

    def test_payload_has_no_pii(self):
        """Payload must not contain PII fields."""
        config = MagicMock()
        session = MagicMock()
        session.framework = MagicMock(value="langgraph")

        payload = _collect_telemetry_payload(config, session=session, session_count=10, error_count=2)

        # Check no PII fields exist
        for key in payload:
            assert "email" not in key.lower()
            assert "user" not in key.lower()
            assert "org" not in key.lower()
            assert "api_key" not in key.lower()
            assert "ip" not in key.lower()
            assert "name" not in key.lower()

    def test_payload_counts(self):
        """Session and error counts are passed through."""
        config = MagicMock()
        session = MagicMock()
        session.framework = MagicMock(value="crewai")

        payload = _collect_telemetry_payload(config, session=session, session_count=47, error_count=3)

        assert payload["session_count"] == 47
        assert payload["error_count"] == 3
        assert payload["framework"] == "crewai"

    def test_payload_no_session_uses_unknown(self):
        """Framework defaults to 'unknown' when no session provided."""
        config = MagicMock()
        payload = _collect_telemetry_payload(config, session_count=0, error_count=0)
        assert payload["framework"] == "unknown"
        assert payload["adapter"] == "unknown"


# =============================================================================
# Stderr Audit Log (FRD-CT-010)
# =============================================================================


class TestTelemetryAuditLog:
    """Tests for stderr audit logging."""

    def test_payload_logged_before_sending(self, capsys):
        """Payload logged to stderr with [clyro] prefix before HTTP call."""
        config = MagicMock()
        config.framework = "langgraph"
        config.is_local_only.return_value = False

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.post.return_value = mock_response

        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "true"}), \
             patch("clyro.telemetry_client._get_counts_from_sqlite", return_value=(5, 1)), \
             patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            submit_telemetry(config)

        captured = capsys.readouterr()
        assert "[clyro] Telemetry (opt-in): sending" in captured.err
        assert "sdk_version" in captured.err


# =============================================================================
# Failure Handling (FRD-CT-008)
# =============================================================================


class TestTelemetryFailureHandling:
    """Tests for failure handling — telemetry must never crash user code."""

    def test_no_submission_when_disabled(self):
        """No HTTP call when CLYRO_TELEMETRY is not set."""
        config = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_TELEMETRY", None)
            with patch("httpx.Client") as MockClient:
                submit_telemetry(config)
                MockClient.assert_not_called()

    def test_network_failure_swallowed(self):
        """Network failure doesn't raise — swallowed silently."""
        config = MagicMock()
        config.framework = "langgraph"

        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "true"}), \
             patch("clyro.telemetry_client._get_counts_from_sqlite", return_value=(1, 0)), \
             patch("httpx.Client") as MockClient:
            MockClient.side_effect = ConnectionError("refused")

            # Should NOT raise
            submit_telemetry(config)

    def test_timeout_swallowed(self):
        """Timeout doesn't raise — swallowed silently."""
        config = MagicMock()
        config.framework = "langgraph"

        with patch.dict(os.environ, {"CLYRO_TELEMETRY": "true"}), \
             patch("clyro.telemetry_client._get_counts_from_sqlite", return_value=(1, 0)), \
             patch("httpx.Client") as MockClient:
            MockClient.side_effect = TimeoutError("5s exceeded")

            # Should NOT raise
            submit_telemetry(config)
