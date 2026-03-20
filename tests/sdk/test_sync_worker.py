# Tests for Clyro SDK Sync Worker
# Implements PRD-005, PRD-006

"""
Comprehensive unit tests for the SyncWorker class.

Tests cover:
- Background sync orchestration
- Circuit breaker functionality
- Connectivity detection and auto-recovery
- Event prioritization
- Metrics collection
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig
from clyro.storage.sqlite import LocalStorage
from clyro.workers.sync_worker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ConnectivityDetector,
    ConnectivityStatus,
    SyncMetrics,
    SyncWorker,
)
from clyro.trace import EventType, TraceEvent, create_session_end_event, create_step_event


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    """Create test configuration."""
    return ClyroConfig(
        local_storage_path=str(temp_dir / "test.db"),
        local_storage_max_mb=10,
        sync_interval_seconds=1,  # Minimum valid sync interval
        batch_size=5,
        retry_max_attempts=2,
    )


@pytest.fixture
def storage(config):
    """Create local storage instance."""
    storage = LocalStorage(config)
    storage.initialize()
    yield storage
    storage.close()


@pytest.fixture
def mock_sender():
    """Create mock event sender."""
    sender = AsyncMock()
    sender.send_batch = AsyncMock(return_value={"accepted": 5, "rejected": 0, "errors": []})
    return sender


@pytest.fixture
def sync_worker(config, storage, mock_sender):
    """Create sync worker instance."""
    return SyncWorker(
        config=config,
        storage=storage,
        sender=mock_sender,
    )


@pytest.fixture
def session_id():
    """Generate a session ID for tests."""
    return uuid4()


def create_test_events(session_id, count: int = 5) -> list[TraceEvent]:
    """Create test events for a session."""
    return [
        create_step_event(
            session_id=session_id,
            step_number=i,
            event_name=f"step_{i}",
        )
        for i in range(count)
    ]


# -----------------------------------------------------------------------------
# Circuit Breaker Tests
# -----------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True

    @pytest.mark.asyncio
    async def test_can_execute_when_closed(self):
        """Test requests allowed when circuit is closed."""
        cb = CircuitBreaker()
        assert await cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        # Record failures up to threshold
        for i in range(3):
            tripped = await cb.record_failure()
            if i < 2:
                assert tripped is False
                assert cb.state == CircuitState.CLOSED
            else:
                assert tripped is True
                assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_blocks_requests_when_open(self):
        """Test requests blocked when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)

        await cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert await cb.can_execute() is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker(config)

        await cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open on next check
        assert await cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config)

        # Open the circuit
        await cb.record_failure()
        await asyncio.sleep(0.15)
        await cb.can_execute()  # Transition to half-open

        # Record successes
        await cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        await cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config)

        # Open and transition to half-open
        await cb.record_failure()
        await asyncio.sleep(0.15)
        await cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        tripped = await cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitState.OPEN

    def test_reset_returns_to_closed(self):
        """Test reset returns circuit to closed state."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        asyncio.get_event_loop().run_until_complete(cb.record_failure())

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


# -----------------------------------------------------------------------------
# Connectivity Detector Tests
# -----------------------------------------------------------------------------


class TestConnectivityDetector:
    """Tests for connectivity detection."""

    def test_initial_status_is_unknown(self):
        """Test initial connectivity status is unknown."""
        detector = ConnectivityDetector()
        assert detector.status == ConnectivityStatus.UNKNOWN

    def test_becomes_connected_after_successes(self):
        """Test status becomes connected after consecutive successes."""
        detector = ConnectivityDetector()

        detector.record_success()
        assert detector.status == ConnectivityStatus.UNKNOWN

        detector.record_success()
        assert detector.status == ConnectivityStatus.CONNECTED

    def test_becomes_disconnected_after_failures(self):
        """Test status becomes disconnected after consecutive failures."""
        detector = ConnectivityDetector()
        detector.record_success()  # Start as connected
        detector.record_success()

        detector.record_failure()
        detector.record_failure()
        assert detector.status == ConnectivityStatus.CONNECTED

        detector.record_failure()
        assert detector.status == ConnectivityStatus.DISCONNECTED

    def test_status_change_callback_called(self):
        """Test callback is called on status change."""
        detector = ConnectivityDetector()
        callback = MagicMock()
        detector.on_status_change(callback)

        # Trigger status change to connected
        detector.record_success()
        detector.record_success()

        callback.assert_called_once_with(ConnectivityStatus.CONNECTED)

    def test_is_connected_property(self):
        """Test is_connected property."""
        detector = ConnectivityDetector()

        assert detector.is_connected is False

        detector.record_success()
        detector.record_success()

        assert detector.is_connected is True


# -----------------------------------------------------------------------------
# Sync Metrics Tests
# -----------------------------------------------------------------------------


class TestSyncMetrics:
    """Tests for sync metrics."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = SyncMetrics()

        assert metrics.total_events_synced == 0
        assert metrics.total_events_failed == 0
        assert metrics.total_sync_attempts == 0
        assert metrics.last_sync_time is None

    def test_record_sync_latency(self):
        """Test latency recording and averaging."""
        metrics = SyncMetrics()

        metrics.record_sync_latency(10.0)
        assert metrics.average_sync_latency_ms == 10.0

        metrics.record_sync_latency(20.0)
        assert metrics.average_sync_latency_ms == 15.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = SyncMetrics()

        # No data
        assert metrics._calculate_success_rate() == 100.0

        # 80% success rate
        metrics.total_events_synced = 80
        metrics.total_events_failed = 20
        assert metrics._calculate_success_rate() == 80.0

    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = SyncMetrics()
        metrics.total_events_synced = 100
        metrics.total_sync_attempts = 10

        result = metrics.to_dict()

        assert "total_events_synced" in result
        assert "total_sync_attempts" in result
        assert "success_rate" in result
        assert result["total_events_synced"] == 100


# -----------------------------------------------------------------------------
# Sync Worker Tests
# -----------------------------------------------------------------------------


class TestSyncWorkerBasic:
    """Basic sync worker tests."""

    def test_initial_state(self, sync_worker):
        """Test initial worker state."""
        assert sync_worker.is_running is False
        assert sync_worker.circuit_state == CircuitState.CLOSED
        assert sync_worker.connectivity_status == ConnectivityStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_start_and_stop(self, sync_worker):
        """Test starting and stopping the worker."""
        await sync_worker.start()
        assert sync_worker.is_running is True

        await sync_worker.stop()
        assert sync_worker.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_ignored(self, sync_worker):
        """Test that double start is ignored."""
        await sync_worker.start()
        await sync_worker.start()  # Should not raise

        assert sync_worker.is_running is True
        await sync_worker.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, sync_worker):
        """Test stop without start doesn't raise."""
        await sync_worker.stop()  # Should not raise
        assert sync_worker.is_running is False


class TestSyncWorkerSync:
    """Sync operation tests."""

    @pytest.mark.asyncio
    async def test_sync_now_sends_events(self, sync_worker, storage, session_id, mock_sender):
        """Test sync_now sends unsynced events."""
        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Sync now
        result = await sync_worker.sync_now()

        # Verify sender was called
        mock_sender.send_batch.assert_called()
        assert result["synced"] > 0 or result["failed"] > 0

    @pytest.mark.asyncio
    async def test_sync_with_empty_storage(self, sync_worker, mock_sender):
        """Test sync with no events doesn't call sender."""
        result = await sync_worker.sync_now()

        mock_sender.send_batch.assert_not_called()
        assert result["synced"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_sync_respects_circuit_breaker(self, config, storage, session_id):
        """Test sync respects circuit breaker state."""
        # Create sender that always fails
        failing_sender = AsyncMock()
        failing_sender.send_batch = AsyncMock(side_effect=Exception("Network error"))

        # Create worker with low failure threshold
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=failing_sender,
            circuit_breaker_config=cb_config,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Trigger failures to open circuit
        await worker.sync_now()
        await worker.sync_now()

        assert worker.circuit_state == CircuitState.OPEN

        # Next sync should be skipped
        result = await worker.sync_now()
        assert result["circuit_open"] is True


class TestSyncWorkerPrioritization:
    """Event prioritization tests."""

    @pytest.mark.asyncio
    async def test_high_priority_events_synced_first(self, sync_worker, storage, session_id, mock_sender):
        """Test high priority events are synced before normal events."""
        # Store normal events first
        normal_events = create_test_events(session_id, count=3)
        storage.store_events(normal_events)

        # Store high priority event (session end)
        high_priority_event = create_session_end_event(
            session_id=session_id,
            agent_id=uuid4(),
        )
        storage.store_event(high_priority_event)

        # Sync
        await sync_worker.sync_now()

        # Verify sender was called
        assert mock_sender.send_batch.called

        # Get the events that were sent
        call_args = mock_sender.send_batch.call_args
        sent_events = call_args[0][0]

        # First event should be high priority (session end)
        if sent_events:
            assert sent_events[0].event_type in (EventType.SESSION_END, EventType.ERROR)


class TestSyncWorkerMetrics:
    """Metrics collection tests."""

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self, sync_worker, storage, session_id, mock_sender):
        """Test metrics are updated on successful sync."""
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        await sync_worker.sync_now()

        metrics = sync_worker.metrics
        assert metrics.total_sync_attempts >= 1
        assert metrics.total_events_synced > 0
        assert metrics.last_successful_sync is not None

    @pytest.mark.asyncio
    async def test_metrics_updated_on_failure(self, config, storage, session_id):
        """Test metrics are updated on failed sync."""
        failing_sender = AsyncMock()
        failing_sender.send_batch = AsyncMock(side_effect=Exception("Network error"))

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=failing_sender,
        )

        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        await worker.sync_now()

        metrics = worker.metrics
        assert metrics.total_events_failed > 0
        assert metrics.last_failure_time is not None
        assert metrics.last_failure_reason is not None


class TestSyncWorkerBackgroundSync:
    """Background sync tests."""

    @pytest.mark.asyncio
    async def test_background_sync_runs_periodically(self, config, storage, session_id, mock_sender):
        """Test background sync runs at configured interval."""
        # Use very short interval for testing
        config.sync_interval_seconds = 0.05

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=mock_sender,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Start worker
        await worker.start()

        # Wait for a few sync cycles
        await asyncio.sleep(0.2)

        await worker.stop()

        # Sender should have been called multiple times
        assert mock_sender.send_batch.call_count >= 1

    @pytest.mark.asyncio
    async def test_immediate_sync_trigger(self, sync_worker, storage, session_id, mock_sender):
        """Test immediate sync trigger."""
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        await sync_worker.start()

        # Trigger immediate sync
        await sync_worker.trigger_immediate_sync()

        # Give it a moment to process
        await asyncio.sleep(0.1)

        await sync_worker.stop()

        assert mock_sender.send_batch.called


class TestSyncWorkerConnectivityRecovery:
    """Connectivity recovery tests."""

    @pytest.mark.asyncio
    async def test_triggers_sync_on_connectivity_restore(self, config, storage, session_id):
        """Test sync is triggered when connectivity is restored."""
        sender = AsyncMock()
        sender.send_batch = AsyncMock(return_value={"accepted": 3, "rejected": 0})

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=sender,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Simulate connectivity loss and restore
        worker._connectivity.record_failure()
        worker._connectivity.record_failure()
        worker._connectivity.record_failure()
        assert worker.connectivity_status == ConnectivityStatus.DISCONNECTED

        # Start worker
        await worker.start()

        # Restore connectivity (this should trigger sync)
        worker._connectivity.record_success()
        worker._connectivity.record_success()

        # Give it a moment
        await asyncio.sleep(0.2)

        await worker.stop()

        # Verify sync was attempted
        assert sender.send_batch.called


class TestSyncWorkerStatus:
    """Status reporting tests."""

    def test_get_status(self, sync_worker):
        """Test get_status returns comprehensive information."""
        status = sync_worker.get_status()

        assert "running" in status
        assert "circuit_state" in status
        assert "connectivity_status" in status
        assert "unsynced_events" in status
        assert "metrics" in status
        assert "config" in status

    @pytest.mark.asyncio
    async def test_status_reflects_running_state(self, sync_worker):
        """Test status reflects running state."""
        assert sync_worker.get_status()["running"] is False

        await sync_worker.start()
        assert sync_worker.get_status()["running"] is True

        await sync_worker.stop()
        assert sync_worker.get_status()["running"] is False


class TestSyncWorkerGracefulShutdown:
    """Graceful shutdown tests."""

    @pytest.mark.asyncio
    async def test_final_flush_on_stop(self, sync_worker, storage, session_id, mock_sender):
        """Test final flush is attempted on stop."""
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        await sync_worker.start()
        await sync_worker.stop()

        # Final flush should have been attempted
        assert mock_sender.send_batch.call_count >= 1

    @pytest.mark.asyncio
    async def test_stop_with_timeout(self, config, storage, session_id):
        """Test stop respects timeout."""
        # Create a slow sender
        slow_sender = AsyncMock()

        async def slow_send(events):
            await asyncio.sleep(0.5)
            return {"accepted": len(events), "rejected": 0}

        slow_sender.send_batch = slow_send

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=slow_sender,
        )

        await worker.start()

        # Stop with short timeout
        await worker.stop(timeout=0.1)

        assert worker.is_running is False
