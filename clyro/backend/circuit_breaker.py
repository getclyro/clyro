# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Circuit Breaker + Connectivity Detector
# Implements FRD-019

"""
Three-state circuit breaker for backend HTTP calls with connectivity
detection for immediate sync on reconnection.

State transitions (FRD-019):
    CLOSED → OPEN:     After FAILURE_THRESHOLD consecutive failures
    OPEN → HALF_OPEN:  After OPEN_TIMEOUT_SECONDS elapses
    HALF_OPEN → CLOSED: After SUCCESS_THRESHOLD consecutive successes
    HALF_OPEN → OPEN:  On any failure
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


# ── Shared thresholds ──────────────────────────────────────────────────────
# Used by both the in-memory CircuitBreaker class (MCP) and the stateless
# functions (hooks) to guarantee identical transition behaviour.
FAILURE_THRESHOLD: int = 5
SUCCESS_THRESHOLD: int = 2
OPEN_TIMEOUT_SECONDS: float = 30.0


# ── Protocol for any state container ──────────────────────────────────────
@runtime_checkable
class CircuitBreakerStateProtocol(Protocol):
    """Minimal contract for a circuit breaker state container.

    Any object (dataclass, Pydantic model, dict-wrapper) that exposes these
    mutable attributes can be used with the stateless helper functions below.
    """

    state: str
    failure_count: int
    half_open_successes: int
    opened_at: float | None
    total_trips: int


# ── Stateless helper functions ────────────────────────────────────────────
# These operate on *any* object satisfying CircuitBreakerStateProtocol,
# making them usable with both the MCP in-memory CircuitBreaker and the
# hooks' file-persisted CircuitBreakerSnapshot.


def check_can_execute(snapshot: CircuitBreakerStateProtocol) -> bool:
    """Check if a backend request is allowed given the current state."""
    if snapshot.state in ("closed", CircuitState.CLOSED):
        return True
    if snapshot.state in ("open", CircuitState.OPEN):
        if snapshot.opened_at and (time.monotonic() - snapshot.opened_at) >= OPEN_TIMEOUT_SECONDS:
            snapshot.state = "half_open"
            snapshot.half_open_successes = 0
            return True
        return False
    # half_open: allow probe
    return True


def record_success(snapshot: CircuitBreakerStateProtocol) -> None:
    """Record a successful backend request."""
    if snapshot.state in ("half_open", CircuitState.HALF_OPEN):
        snapshot.half_open_successes += 1
        if snapshot.half_open_successes >= SUCCESS_THRESHOLD:
            snapshot.state = "closed"
            snapshot.failure_count = 0
    elif snapshot.state in ("closed", CircuitState.CLOSED):
        snapshot.failure_count = 0


def record_failure(snapshot: CircuitBreakerStateProtocol) -> None:
    """Record a failed backend request."""
    if snapshot.state in ("half_open", CircuitState.HALF_OPEN):
        snapshot.state = "open"
        snapshot.opened_at = time.monotonic()
    elif snapshot.state in ("closed", CircuitState.CLOSED):
        snapshot.failure_count += 1
        if snapshot.failure_count >= FAILURE_THRESHOLD:
            snapshot.state = "open"
            snapshot.opened_at = time.monotonic()
            snapshot.total_trips += 1


class CircuitState(Enum):
    """Circuit breaker states (FRD-019).

    Used by all circuit breaker implementations:
    - Sync ``CircuitBreaker`` (MCP wrapper, backend/sync_manager)
    - Async ``AsyncCircuitBreaker`` (SDK sync worker)
    - Stateless helpers (hooks)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ConnectivityStatus(Enum):
    """Backend connectivity status.

    Used by both sync ``ConnectivityDetector`` (MCP) and
    async ``ConnectivityDetector`` (SDK sync worker).
    """

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker thresholds.

    Shared by all circuit breaker implementations. Module-level
    constants (``FAILURE_THRESHOLD``, etc.) provide the defaults.
    """

    failure_threshold: int = FAILURE_THRESHOLD
    success_threshold: int = SUCCESS_THRESHOLD
    timeout_seconds: float = OPEN_TIMEOUT_SECONDS
    half_open_max_requests: int = 3


@dataclass
class CircuitBreakerState:
    """Observable state for monitoring and testing (TDD §3.8)."""

    state: CircuitState
    failure_count: int
    half_open_successes: int
    opened_at: float | None
    total_trips: int


class CircuitBreaker:
    """
    Three-state circuit breaker for backend sync requests.

    Prevents resource waste during sustained backend outages
    and enables fast recovery on reconnection (FRD-019).

    Thresholds are hardcoded per PRD — not user-configurable in v1.1.
    """

    # FRD-019: use module-level shared thresholds
    FAILURE_THRESHOLD: int = FAILURE_THRESHOLD
    SUCCESS_THRESHOLD: int = SUCCESS_THRESHOLD
    OPEN_TIMEOUT_SECONDS: float = OPEN_TIMEOUT_SECONDS

    def __init__(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
        self._opened_at: float | None = None
        self._total_trips = 0

    def can_execute(self) -> bool:
        """Check if a sync request is allowed in current state (FRD-019)."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self.OPEN_TIMEOUT_SECONDS:
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                return True  # Allow probe request
            return False
        # HALF_OPEN: allow probe requests
        return True

    def record_success(self) -> None:
        """Record a successful backend request (FRD-019)."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.SUCCESS_THRESHOLD:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed backend request (FRD-019)."""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.FAILURE_THRESHOLD:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._total_trips += 1

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def get_state(self) -> CircuitBreakerState:
        """Return observable state snapshot for monitoring."""
        return CircuitBreakerState(
            state=self._state,
            failure_count=self._failure_count,
            half_open_successes=self._half_open_successes,
            opened_at=self._opened_at,
            total_trips=self._total_trips,
        )


class ConnectivityDetector:
    """
    Track backend connectivity via consecutive success/failure counts.

    Triggers immediate sync on DISCONNECTED → CONNECTED transition
    and logs warnings on CONNECTED → DISCONNECTED (FRD-019).
    """

    CONNECTED_THRESHOLD: int = 2
    DISCONNECTED_THRESHOLD: int = 3

    def __init__(self) -> None:
        self._status = ConnectivityStatus.UNKNOWN
        self._consecutive_successes = 0
        self._consecutive_failures = 0

    def record_success(self) -> ConnectivityStatus:
        """Record a successful request. Returns new status."""
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        old_status = self._status
        if self._consecutive_successes >= self.CONNECTED_THRESHOLD:
            self._status = ConnectivityStatus.CONNECTED

        if (
            old_status == ConnectivityStatus.DISCONNECTED
            and self._status == ConnectivityStatus.CONNECTED
        ):
            logger.info("backend_connectivity_restored")

        return self._status

    def record_failure(self) -> ConnectivityStatus:
        """Record a failed request. Returns new status."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        old_status = self._status
        if self._consecutive_failures >= self.DISCONNECTED_THRESHOLD:
            self._status = ConnectivityStatus.DISCONNECTED

        if (
            old_status == ConnectivityStatus.CONNECTED
            and self._status == ConnectivityStatus.DISCONNECTED
        ):
            logger.warning("backend_connectivity_lost", action="queuing_locally")

        return self._status

    @property
    def status(self) -> ConnectivityStatus:
        """Current connectivity status."""
        return self._status
