# Integration tests for SDK Open-Source Foundation
# Implements TDD §13.2 cross-component integration tests

"""
Cross-component integration tests:
- Full local-mode run (C1-C5)
- Quiet mode (NFR-003)
- Network isolation (NFR-004)
- Mode config edge cases (C3)
"""

from __future__ import annotations

import os
import socket
import textwrap
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import clyro
from clyro.config import ClyroConfig
from clyro.constants import ISSUE_TRACKER_URL
from clyro.exceptions import (
    AuthenticationError,
    BackendUnavailableError,
    ClyroConfigError,
    ClyroError,
    ClyroWrapError,
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    LoopDetectedError,
    PolicyViolationError,
    RateLimitExhaustedError,
    StepLimitExceededError,
    TraceError,
    TransportError,
)
from clyro.local_logger import reset_welcome_flag
from clyro.local_policy import reset_sdk_policy_cache


@pytest.fixture(autouse=True)
def _reset():
    reset_sdk_policy_cache()
    reset_welcome_flag()
    yield
    reset_sdk_policy_cache()
    reset_welcome_flag()


@pytest.fixture()
def policy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sdk_dir = tmp_path / ".clyro" / "sdk"
    sdk_dir.mkdir(parents=True)
    monkeypatch.setattr("clyro.local_policy._POLICY_DIR", sdk_dir)
    monkeypatch.setattr("clyro.local_policy._POLICY_FILE", sdk_dir / "policies.yaml")
    return sdk_dir


# ===========================================================================
# C3: Mode configuration
# ===========================================================================


class TestModeConfig:
    """FRD-SOF-004: mode field validation and resolution."""

    def test_auto_resolve_local_no_key(self):
        config = ClyroConfig()
        assert config.mode == "local"

    def test_auto_resolve_cloud_with_key(self):
        config = ClyroConfig(api_key="cly_test_abc", agent_name="test")
        assert config.mode == "cloud"

    def test_explicit_local(self):
        config = ClyroConfig(mode="local")
        assert config.mode == "local"
        assert config.is_local_only() is True

    def test_explicit_cloud_requires_key(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(mode="cloud")

    def test_explicit_cloud_with_key(self):
        config = ClyroConfig(mode="cloud", api_key="cly_test_abc", agent_name="test")
        assert config.mode == "cloud"
        assert config.is_local_only() is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(mode="hybrid")  # type: ignore[arg-type]

    def test_clyro_mode_env_var(self, monkeypatch):
        """FRD-SOF-004: CLYRO_MODE env var."""
        monkeypatch.setenv("CLYRO_MODE", "local")
        config = ClyroConfig.from_env()
        assert config.mode == "local"

    def test_clyro_mode_env_overrides_key(self, monkeypatch):
        """TDD §13.4: CLYRO_MODE=local with CLYRO_API_KEY set → local mode."""
        monkeypatch.setenv("CLYRO_MODE", "local")
        monkeypatch.setenv("CLYRO_API_KEY", "cly_test_abc")
        config = ClyroConfig.from_env()
        assert config.mode == "local"

    def test_is_local_only_backward_compat(self):
        """FRD-SOF-004: is_local_only() returns mode == 'local'."""
        config = ClyroConfig(mode="local")
        assert config.is_local_only() is True
        config2 = ClyroConfig(mode="cloud", api_key="cly_test_x", agent_name="t")
        assert config2.is_local_only() is False


# ===========================================================================
# C6: ClyroError enrichment
# ===========================================================================


class TestClyroErrorEnrichment:
    """FRD-SOF-009: error context enrichment."""

    def test_issue_tracker_url_in_str(self):
        """str() of every ClyroError subclass contains the issue tracker URL."""
        subclasses = [
            ClyroError("test"),
            ClyroConfigError("test"),
            ClyroWrapError("test"),
            FrameworkVersionError("fw", "1.0", ">=2.0"),
            ExecutionControlError("test"),
            StepLimitExceededError(100, 101),
            CostLimitExceededError(10.0, 10.5),
            LoopDetectedError(3, "abc123"),
            PolicyViolationError("r1", "rule", "msg"),
            TraceError("test"),
            TransportError("test"),
            AuthenticationError("test"),
            RateLimitExhaustedError("test"),
            BackendUnavailableError("test"),
        ]

        for exc in subclasses:
            assert ISSUE_TRACKER_URL in str(exc), (
                f"{type(exc).__name__}.__str__() missing issue tracker URL"
            )

    def test_message_does_not_contain_url(self):
        """FRD-SOF-009: .message does NOT contain the URL."""
        exc = ClyroError("test message")
        assert "github.com" not in exc.message

    def test_str_with_details(self):
        exc = ClyroError("test", details={"key": "val"})
        s = str(exc)
        assert "test" in s
        assert "key" in s
        assert ISSUE_TRACKER_URL in s


# ===========================================================================
# NFR-003: Quiet mode full integration
# ===========================================================================


class TestQuietModeIntegration:
    """NFR-003: CLYRO_QUIET=true → zero stderr bytes from clyro output."""

    def test_quiet_mode_zero_stderr(self, policy_dir: Path, monkeypatch, capsys):
        monkeypatch.setenv("CLYRO_QUIET", "true")

        (policy_dir / "policies.yaml").write_text(
            "version: 1\nglobal:\n  policies: []\n", encoding="utf-8",
        )

        config = ClyroConfig(mode="local")

        from clyro.local_logger import LocalTerminalLogger
        from clyro.local_policy import SDKLocalPolicyEvaluator

        logger = LocalTerminalLogger(config)
        logger.print_welcome()

        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        evaluator.evaluate_sync("llm_call", {"model": "gpt-4"})

        session_mock = MagicMock()
        session_mock.step_number = 5
        session_mock.cumulative_cost = Decimal("0.01")
        session_mock.session_id = uuid4()
        session_mock.events = []
        logger.print_session_summary(session_mock)

        captured = capsys.readouterr()
        # Filter out non-clyro output
        clyro_lines = [l for l in captured.err.splitlines() if "[clyro]" in l or "governance" in l.lower()]
        assert len(clyro_lines) == 0, f"Unexpected clyro stderr output: {clyro_lines}"


# ===========================================================================
# NFR-004: Network isolation
# ===========================================================================


class TestNetworkIsolation:
    """NFR-004: local mode makes zero network calls."""

    def test_no_socket_calls_in_local_mode(self, policy_dir: Path, monkeypatch):
        """Patch socket.socket to raise — full local run must succeed."""
        (policy_dir / "policies.yaml").write_text(
            "version: 1\nglobal:\n  policies: []\n", encoding="utf-8",
        )

        original_socket = socket.socket

        def no_network(*args, **kwargs):
            raise AssertionError("Network call detected in local mode!")

        monkeypatch.setattr("socket.socket", no_network)

        try:
            config = ClyroConfig(mode="local")

            from clyro.local_logger import LocalTerminalLogger
            from clyro.local_policy import SDKLocalPolicyEvaluator

            logger = LocalTerminalLogger(config)
            evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

            # These should all work without network
            logger.print_welcome()
            evaluator.evaluate_sync("llm_call", {"model": "gpt-4"})

            session_mock = MagicMock()
            session_mock.step_number = 1
            session_mock.cumulative_cost = Decimal("0")
            session_mock.session_id = uuid4()
            session_mock.events = []
            logger.print_session_summary(session_mock)

        finally:
            monkeypatch.setattr("socket.socket", original_socket)
