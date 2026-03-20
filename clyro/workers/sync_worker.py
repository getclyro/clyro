# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Sync Worker
# Implements PRD-005, PRD-006

"""
Background sync worker for the Clyro SDK.

This module provides a dedicated sync orchestrator that handles:
- Background synchronization of trace events to the backend
- Connectivity detection and automatic recovery
- Circuit breaker pattern for repeated failures
- Event prioritization for efficient syncing
- Comprehensive metrics and observability
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from clyro.backend.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitState,
    ConnectivityStatus,
)

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.storage.sqlite import LocalStorage
    from clyro.trace import TraceEvent

logger = structlog.get_logger(__name__)


# Sync worker constants
FINAL_FLUSH_MAX_ATTEMPTS = 3  # Maximum batch sync attempts during shutdown


@dataclass
class SyncMetrics:
    """Metrics for sync operations."""

    total_events_synced: int = 0
    total_events_failed: int = 0
    total_sync_attempts: int = 0
    last_sync_time: datetime | None = None
    last_successful_sync: datetime | None = None
    last_failure_time: datetime | None = None
    last_failure_reason: str | None = None
    average_sync_latency_ms: float = 0.0
    circuit_breaker_trips: int = 0
    connectivity_changes: int = 0

    # Rolling window for latency calculation
    _latency_samples: list[float] = field(default_factory=list)
    _max_samples: int = 100

    def record_sync_latency(self, latency_ms: float) -> None:
        """Record a sync latency sample."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples.pop(0)
        self.average_sync_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_events_synced": self.total_events_synced,
            "total_events_failed": self.total_events_failed,
            "total_sync_attempts": self.total_sync_attempts,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "last_successful_sync": self.last_successful_sync.isoformat()
            if self.last_successful_sync
            else None,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "last_failure_reason": self.last_failure_reason,
            "average_sync_latency_ms": round(self.average_sync_latency_ms, 2),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "connectivity_changes": self.connectivity_changes,
            "success_rate": self._calculate_success_rate(),
        }

    def _calculate_success_rate(self) -> float:
        """Calculate sync success rate."""
        total = self.total_events_synced + self.total_events_failed
        if total == 0:
            return 100.0
        return round((self.total_events_synced / total) * 100, 2)


class EventSender(Protocol):
    """Protocol for sending events to backend."""

    async def send_batch(self, events: list[TraceEvent]) -> dict[str, Any]:
        """
        Send a batch of events to the backend.

        Args:
            events: List of events to send

        Returns:
            Response dict with 'accepted', 'rejected', 'errors' keys

        Raises:
            Exception: On network or server errors
        """
        ...


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_requests = 0
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock in the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED

    async def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if blocked
        """
        async with self._get_lock():
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        # Transition to half-open
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_requests = 0
                        self._success_count = 0
                        logger.info(
                            "circuit_breaker_half_open",
                            elapsed_seconds=round(elapsed, 2),
                        )
                        return True
                return False

            # HALF_OPEN state
            if self._half_open_requests < self.config.half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._get_lock():
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    # Recover to closed state
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        "circuit_breaker_closed",
                        success_count=self._success_count,
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self) -> bool:
        """
        Record a failed request.

        Returns:
            True if circuit tripped open
        """
        async with self._get_lock():
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open returns to open
                self._state = CircuitState.OPEN
                logger.warning(
                    "circuit_breaker_reopened",
                    failure_count=self._failure_count,
                )
                return True

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "circuit_breaker_opened",
                        failure_count=self._failure_count,
                        threshold=self.config.failure_threshold,
                    )
                    return True

            return False

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_requests = 0


class ConnectivityDetector:
    """
    Detects network connectivity status and triggers recovery.

    Uses a combination of:
    - Sync success/failure tracking
    - Periodic health checks
    - Immediate recovery attempts on status change
    """

    def __init__(
        self,
        health_check_fn: Callable[[], bool] | None = None,
        check_interval_seconds: float = 30.0,
    ):
        """
        Initialize connectivity detector.

        Args:
            health_check_fn: Optional function to check connectivity
            check_interval_seconds: Interval between health checks
        """
        self._health_check_fn = health_check_fn
        self._check_interval = check_interval_seconds
        self._status = ConnectivityStatus.UNKNOWN
        self._last_check_time: float | None = None
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._status_change_callbacks: list[Callable[[ConnectivityStatus], None]] = []

    @property
    def status(self) -> ConnectivityStatus:
        """Get current connectivity status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._status == ConnectivityStatus.CONNECTED

    def on_status_change(self, callback: Callable[[ConnectivityStatus], None]) -> None:
        """Register a callback for status changes."""
        self._status_change_callbacks.append(callback)

    def record_success(self) -> None:
        """Record a successful network operation."""
        self._consecutive_failures = 0
        self._consecutive_successes += 1

        if self._consecutive_successes >= 2 and self._status != ConnectivityStatus.CONNECTED:
            self._update_status(ConnectivityStatus.CONNECTED)

    def record_failure(self) -> None:
        """Record a failed network operation."""
        self._consecutive_successes = 0
        self._consecutive_failures += 1

        if self._consecutive_failures >= 3 and self._status != ConnectivityStatus.DISCONNECTED:
            self._update_status(ConnectivityStatus.DISCONNECTED)

    def _update_status(self, new_status: ConnectivityStatus) -> None:
        """Update connectivity status and notify callbacks."""
        old_status = self._status
        self._status = new_status

        logger.info(
            "connectivity_status_changed",
            old_status=old_status.value,
            new_status=new_status.value,
        )

        for callback in self._status_change_callbacks:
            try:
                callback(new_status)
            except Exception as e:
                logger.error("connectivity_callback_error", error=str(e))

    async def check_connectivity(self) -> ConnectivityStatus:
        """
        Perform a connectivity check.

        Returns:
            Current connectivity status
        """
        if self._health_check_fn is None:
            return self._status

        try:
            is_connected = self._health_check_fn()
            if is_connected:
                self.record_success()
            else:
                self.record_failure()
        except Exception:
            self.record_failure()

        self._last_check_time = time.monotonic()
        return self._status


class SyncWorker:
    """
    Background sync worker for reliable event synchronization.

    Features:
    - Periodic background sync with configurable interval
    - Circuit breaker for failure protection
    - Connectivity detection and auto-recovery
    - Event prioritization (newer events first, errors prioritized)
    - Comprehensive metrics and observability
    - Graceful shutdown with final flush

    Implements PRD-005, PRD-006
    """

    def __init__(
        self,
        config: ClyroConfig,
        storage: LocalStorage,
        sender: EventSender,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize sync worker.

        Args:
            config: SDK configuration
            storage: Local storage instance
            sender: Event sender for backend communication
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self.config = config
        self._storage = storage
        self._sender = sender

        # Circuit breaker for failure protection
        self._circuit_breaker = CircuitBreaker(circuit_breaker_config)

        # Connectivity detection
        self._connectivity = ConnectivityDetector()
        self._connectivity.on_status_change(self._on_connectivity_change)

        # Metrics
        self._metrics = SyncMetrics()

        # Worker state — asyncio primitives are created lazily in start()
        # to bind to the correct event loop.
        self._running = False
        self._task: asyncio.Task | None = None
        self._immediate_sync_event: asyncio.Event | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._lock: asyncio.Lock | None = None

        # Sync state
        self._last_sync_attempt: datetime | None = None
        self._pending_immediate_sync = False

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    @property
    def metrics(self) -> SyncMetrics:
        """Get sync metrics."""
        return self._metrics

    @property
    def circuit_state(self) -> CircuitState:
        """Get circuit breaker state."""
        return self._circuit_breaker.state

    @property
    def connectivity_status(self) -> ConnectivityStatus:
        """Get connectivity status."""
        return self._connectivity.status

    def record_sync_success(self) -> None:
        """Record a successful sync operation for connectivity tracking."""
        self._connectivity.record_success()

    def record_sync_failure(self) -> None:
        """Record a failed sync operation for connectivity tracking."""
        self._connectivity.record_failure()

    async def start(self) -> None:
        """Start the background sync worker."""
        if self._running:
            logger.warning("sync_worker_already_running")
            return

        # Create asyncio primitives in the running event loop
        self._immediate_sync_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._sync_loop())

        logger.info(
            "sync_worker_started",
            interval_seconds=self.config.sync_interval_seconds,
            batch_size=self.config.batch_size,
        )

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the background sync worker gracefully.

        Args:
            timeout: Maximum time to wait for final flush
        """
        if not self._running:
            return

        self._running = False
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        if self._task:
            try:
                # Wait for task to complete with timeout
                await asyncio.wait_for(self._task, timeout=timeout)
            except TimeoutError:
                logger.warning("sync_worker_stop_timeout", timeout=timeout)
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        # Final flush attempt
        await self._final_flush()

        logger.info("sync_worker_stopped", metrics=self._metrics.to_dict())

    async def trigger_immediate_sync(self) -> None:
        """Trigger an immediate sync attempt."""
        self._pending_immediate_sync = True
        if self._immediate_sync_event is not None:
            self._immediate_sync_event.set()

    async def sync_now(self) -> dict[str, Any]:
        """
        Perform an immediate synchronous sync.

        Returns:
            Sync result with counts
        """
        return await self._perform_sync()

    def _on_connectivity_change(self, status: ConnectivityStatus) -> None:
        """Handle connectivity status change."""
        self._metrics.connectivity_changes += 1

        if status == ConnectivityStatus.CONNECTED:
            # Trigger immediate sync on reconnection
            logger.info("connectivity_restored_triggering_sync")
            self._pending_immediate_sync = True
            if self._immediate_sync_event is not None:
                self._immediate_sync_event.set()

            # Reset circuit breaker on connectivity restore
            self._circuit_breaker.reset()

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                # Wait for interval or immediate sync trigger
                try:
                    await asyncio.wait_for(
                        self._wait_for_sync_trigger(),
                        timeout=self.config.sync_interval_seconds,
                    )
                except TimeoutError:
                    pass  # Normal timeout, proceed with sync

                # Check if shutdown requested
                if self._shutdown_event.is_set():
                    break

                # Perform sync
                await self._perform_sync()

                # Enforce storage size limit
                self._storage.enforce_size_limit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("sync_loop_error", error=str(e))
                await asyncio.sleep(1.0)  # Brief pause on error

    async def _wait_for_sync_trigger(self) -> None:
        """Wait for sync trigger (immediate sync event)."""
        await self._immediate_sync_event.wait()
        self._immediate_sync_event.clear()

    async def _perform_sync(self) -> dict[str, Any]:
        """
        Perform a sync operation.

        Returns:
            Sync result with counts
        """
        self._metrics.total_sync_attempts += 1
        self._metrics.last_sync_time = datetime.now(UTC)
        self._last_sync_attempt = self._metrics.last_sync_time

        result = {
            "synced": 0,
            "failed": 0,
            "skipped": 0,
            "circuit_open": False,
        }

        # Check circuit breaker
        if not await self._circuit_breaker.can_execute():
            logger.debug("sync_skipped_circuit_open")
            result["circuit_open"] = True
            result["skipped"] = self._storage.get_event_count().get("unsynced", 0)
            return result

        # Get events to sync with prioritization
        events = await self._get_prioritized_events()

        if not events:
            return result

        # Attempt to sync
        start_time = time.monotonic()

        try:
            response = await self._sender.send_batch(events)
            latency_ms = (time.monotonic() - start_time) * 1000

            # Process successful response
            await self._process_sync_success(response, events, latency_ms, result)

        except Exception as e:
            # Process sync failure
            latency_ms = (time.monotonic() - start_time) * 1000
            await self._process_sync_failure(e, events, latency_ms, result)

        return result

    async def _process_sync_success(
        self,
        response: dict[str, Any],
        events: list[TraceEvent],
        latency_ms: float,
        result: dict[str, Any],
    ) -> None:
        """
        Process a successful sync response.

        Args:
            response: Backend response
            events: Events that were synced
            latency_ms: Sync latency in milliseconds
            result: Result dict to update
        """
        # Record timing
        self._metrics.record_sync_latency(latency_ms)

        # Process response
        accepted = response.get("accepted", 0)
        rejected = response.get("rejected", 0)

        result["synced"] = accepted
        result["failed"] = rejected

        self._metrics.total_events_synced += accepted
        self._metrics.total_events_failed += rejected
        self._metrics.last_successful_sync = datetime.now(UTC)

        # Update storage based on response
        await self._update_storage_after_sync(accepted, rejected, events)

        # Record success for circuit breaker and connectivity
        await self._circuit_breaker.record_success()
        self._connectivity.record_success()

        logger.debug(
            "sync_completed",
            synced=accepted,
            failed=rejected,
            latency_ms=round(latency_ms, 2),
        )

    async def _update_storage_after_sync(
        self,
        accepted: int,
        rejected: int,
        events: list[TraceEvent],
    ) -> None:
        """
        Update local storage based on sync response.

        Args:
            accepted: Number of events accepted by backend
            rejected: Number of events rejected by backend
            events: Events that were sent
        """
        # Mark synced events when backend confirms full acceptance
        if accepted == len(events) and rejected == 0:
            synced_ids = [str(e.event_id) for e in events]
            self._storage.mark_events_synced(synced_ids)
        elif accepted > 0 or rejected > 0:
            # Partial sync - increment attempts for retry
            logger.warning(
                "partial_sync_response",
                accepted=accepted,
                rejected=rejected,
                event_count=len(events),
            )
            event_ids = [str(e.event_id) for e in events]
            self._storage.increment_sync_attempts(event_ids)

    async def _process_sync_failure(
        self,
        error: Exception,
        events: list[TraceEvent],
        latency_ms: float,
        result: dict[str, Any],
    ) -> None:
        """
        Process a sync failure.

        Args:
            error: Exception that occurred
            events: Events that failed to sync
            latency_ms: Time spent before failure
            result: Result dict to update
        """
        # Record failure metrics
        self._metrics.record_sync_latency(latency_ms)
        self._metrics.last_failure_time = datetime.now(UTC)
        self._metrics.last_failure_reason = str(error)
        self._metrics.total_events_failed += len(events)

        result["failed"] = len(events)

        # Update circuit breaker and connectivity
        tripped = await self._circuit_breaker.record_failure()
        if tripped:
            self._metrics.circuit_breaker_trips += 1
        self._connectivity.record_failure()

        # Increment sync attempts for failed events
        event_ids = [str(e.event_id) for e in events]
        self._storage.increment_sync_attempts(event_ids)

        logger.warning(
            "sync_failed",
            error=str(error),
            event_count=len(events),
            circuit_state=self._circuit_breaker.state.value,
        )

    async def _get_prioritized_events(self) -> list[TraceEvent]:
        """
        Get events to sync with prioritization.

        Priority order:
        1. Session end events (need to be synced to complete sessions)
        2. Error events (important for debugging)
        3. Recent events (more likely to be relevant)
        4. Older events (backfill)

        Returns:
            Prioritized list of events to sync
        """
        from clyro.trace import EventType

        # Get unsynced events
        all_events = self._storage.get_unsynced_events(limit=self.config.batch_size * 2)

        if not all_events:
            return []

        # Categorize events
        high_priority: list[TraceEvent] = []
        normal_priority: list[TraceEvent] = []

        for event in all_events:
            if event.event_type in (EventType.SESSION_END, EventType.ERROR):
                high_priority.append(event)
            else:
                normal_priority.append(event)

        # Sort normal priority by timestamp (newer first for relevance)
        normal_priority.sort(key=lambda e: e.timestamp, reverse=True)

        # Combine and limit to batch size
        prioritized = high_priority + normal_priority
        return prioritized[: self.config.batch_size]

    async def _final_flush(self) -> None:
        """Perform final flush on shutdown."""
        # Try up to FINAL_FLUSH_MAX_ATTEMPTS batch syncs
        for attempt in range(FINAL_FLUSH_MAX_ATTEMPTS):
            events = self._storage.get_unsynced_events(limit=self.config.batch_size)
            if not events:
                break

            try:
                await self._sender.send_batch(events)
                synced_ids = [str(e.event_id) for e in events]
                self._storage.mark_events_synced(synced_ids)
            except Exception as e:
                logger.warning(
                    "final_flush_attempt_failed",
                    attempt=attempt + 1,
                    error=str(e),
                )
                break

    def get_status(self) -> dict[str, Any]:
        """
        Get comprehensive sync worker status.

        Returns:
            Status dictionary with all relevant information
        """
        unsynced_count = self._storage.get_event_count().get("unsynced", 0)

        return {
            "running": self._running,
            "circuit_state": self._circuit_breaker.state.value,
            "connectivity_status": self._connectivity.status.value,
            "unsynced_events": unsynced_count,
            "last_sync_attempt": (
                self._last_sync_attempt.isoformat() if self._last_sync_attempt else None
            ),
            "metrics": self._metrics.to_dict(),
            "config": {
                "sync_interval_seconds": self.config.sync_interval_seconds,
                "batch_size": self.config.batch_size,
                "retry_max_attempts": self.config.retry_max_attempts,
            },
        }


class SyncWorkerFactory:
    """Factory for creating sync workers with proper dependencies."""

    @staticmethod
    def create(
        config: ClyroConfig,
        storage: LocalStorage,
        sender: EventSender,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ) -> SyncWorker:
        """
        Create a sync worker instance.

        Args:
            config: SDK configuration
            storage: Local storage instance
            sender: Event sender for backend communication
            circuit_breaker_config: Optional circuit breaker configuration

        Returns:
            Configured SyncWorker instance
        """
        return SyncWorker(
            config=config,
            storage=storage,
            sender=sender,
            circuit_breaker_config=circuit_breaker_config,
        )
