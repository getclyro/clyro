"""
Unit tests for CircuitBreaker and ConnectivityDetector — TDD §11.1 v1.1 tests.

FRD-019: Circuit breaker prevents resource waste during sustained
backend outages and enables fast recovery on reconnection.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from clyro.backend.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitState,
    ConnectivityDetector,
    ConnectivityStatus,
)


class TestCircuitBreakerInitial:
    """Initial state is CLOSED (FRD-019)."""

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_can_execute_when_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_get_state_returns_snapshot(self) -> None:
        cb = CircuitBreaker()
        state = cb.get_state()
        assert isinstance(state, CircuitBreakerState)
        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0
        assert state.total_trips == 0


class TestCircuitBreakerClosedToOpen:
    """CLOSED → OPEN after FAILURE_THRESHOLD consecutive failures (FRD-019)."""

    def test_stays_closed_below_threshold(self) -> None:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD - 1):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_total_trips_increments(self) -> None:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            cb.record_failure()
        assert cb.get_state().total_trips == 1

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Failure count reset — need full threshold again
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD - 1):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerOpenToHalfOpen:
    """OPEN → HALF_OPEN after OPEN_TIMEOUT_SECONDS elapses (FRD-019)."""

    def test_blocked_while_open(self) -> None:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            cb.record_failure()
        assert cb.can_execute() is False

    def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            cb.record_failure()

        with patch("clyro.backend.circuit_breaker.time") as mock_time:
            # Initial opened_at
            opened_at = cb._opened_at
            mock_time.monotonic.return_value = opened_at + CircuitBreaker.OPEN_TIMEOUT_SECONDS + 1
            assert cb.can_execute() is True
            assert cb.state == CircuitState.HALF_OPEN


class TestCircuitBreakerHalfOpen:
    """HALF_OPEN → CLOSED or OPEN (FRD-019)."""

    def _make_half_open(self) -> CircuitBreaker:
        cb = CircuitBreaker()
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            cb.record_failure()
        # Force transition to HALF_OPEN
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_successes = 0
        return cb

    def test_allows_probe_requests(self) -> None:
        cb = self._make_half_open()
        assert cb.can_execute() is True

    def test_closes_after_success_threshold(self) -> None:
        cb = self._make_half_open()
        for _ in range(CircuitBreaker.SUCCESS_THRESHOLD):
            cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure(self) -> None:
        cb = self._make_half_open()
        cb.record_success()  # One success
        cb.record_failure()  # Then failure
        assert cb.state == CircuitState.OPEN

    def test_half_open_success_count_accumulates(self) -> None:
        cb = self._make_half_open()
        cb.record_success()
        assert cb.get_state().half_open_successes == 1


class TestConnectivityDetectorInitial:
    """Initial state is UNKNOWN."""

    def test_starts_unknown(self) -> None:
        cd = ConnectivityDetector()
        assert cd.status == ConnectivityStatus.UNKNOWN

    def test_success_returns_status(self) -> None:
        cd = ConnectivityDetector()
        status = cd.record_success()
        assert isinstance(status, ConnectivityStatus)


class TestConnectivityDetectorTransitions:
    """CONNECTED and DISCONNECTED transitions (FRD-019)."""

    def test_connected_after_threshold(self) -> None:
        cd = ConnectivityDetector()
        for _ in range(ConnectivityDetector.CONNECTED_THRESHOLD):
            cd.record_success()
        assert cd.status == ConnectivityStatus.CONNECTED

    def test_disconnected_after_threshold(self) -> None:
        cd = ConnectivityDetector()
        for _ in range(ConnectivityDetector.DISCONNECTED_THRESHOLD):
            cd.record_failure()
        assert cd.status == ConnectivityStatus.DISCONNECTED

    def test_success_resets_failure_count(self) -> None:
        cd = ConnectivityDetector()
        cd.record_failure()
        cd.record_failure()
        cd.record_success()
        # Not disconnected yet — success reset the counter
        assert cd.status != ConnectivityStatus.DISCONNECTED

    def test_reconnection_logged(self, capsys) -> None:
        """DISCONNECTED → CONNECTED logs reconnection message (FRD-019)."""
        cd = ConnectivityDetector()
        for _ in range(ConnectivityDetector.DISCONNECTED_THRESHOLD):
            cd.record_failure()
        for _ in range(ConnectivityDetector.CONNECTED_THRESHOLD):
            cd.record_success()
        captured = capsys.readouterr()
        assert "backend_connectivity_restored" in captured.err

    def test_disconnection_logged(self, capsys) -> None:
        """CONNECTED → DISCONNECTED logs warning (FRD-019)."""
        cd = ConnectivityDetector()
        # First get to CONNECTED
        for _ in range(ConnectivityDetector.CONNECTED_THRESHOLD):
            cd.record_success()
        # Then disconnect
        for _ in range(ConnectivityDetector.DISCONNECTED_THRESHOLD):
            cd.record_failure()
        captured = capsys.readouterr()
        assert "backend_connectivity_lost" in captured.err
