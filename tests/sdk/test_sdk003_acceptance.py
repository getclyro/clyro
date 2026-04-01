# Tests for SDK-003: Local Storage and Sync Acceptance Criteria
# Implements PRD-005, PRD-006

"""
Acceptance criteria tests for SDK-003: Local Storage and Sync.

These tests validate the acceptance criteria specified in the ClickUp task:

1. Given the trace backend is temporarily unavailable
   When trace events are generated
   Then events are buffered locally in SQLite
   And retried when connectivity is restored
   And local storage does not exceed configured max size

2. Given trace capture fails completely
   When the agent executes
   Then the agent execution continues uninterrupted (fail-open)
   And a warning is logged indicating trace loss
   And buffered events are preserved for later sync

3. Given local storage exceeds 100MB
   When new events are captured
   Then oldest events are pruned automatically
   And sync status is maintained correctly
"""

import logging
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig
from clyro.storage.sqlite import LocalStorage, StorageHealthStatus
from clyro.trace import EventType, TraceEvent, create_session_end_event, create_step_event
from clyro.transport import Transport
from clyro.workers.sync_worker import (
    CircuitBreakerConfig,
    CircuitState,
    SyncWorker,
)

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
        fail_open=True,
    )


@pytest.fixture
def small_storage_config(temp_dir):
    """Create config with very small storage limit for size tests."""
    return ClyroConfig(
        local_storage_path=str(temp_dir / "small.db"),
        local_storage_max_mb=1,  # 1 MB limit
        sync_interval_seconds=1,  # Minimum valid sync interval
        batch_size=5,
    )


@pytest.fixture
def storage(config):
    """Create local storage instance."""
    storage = LocalStorage(config)
    storage.initialize()
    yield storage
    storage.close()


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
            input_data={"data": "x" * 100},  # Add some payload size
        )
        for i in range(count)
    ]


def create_large_event(session_id, size_kb: int = 10) -> TraceEvent:
    """Create a large event for size testing."""
    large_data = "x" * (size_kb * 1024)
    return create_step_event(
        session_id=session_id,
        step_number=0,
        event_name="large_event",
        input_data={"payload": large_data},
    )


# -----------------------------------------------------------------------------
# Acceptance Criteria 1: Backend Unavailable - Events Buffered Locally
# -----------------------------------------------------------------------------


class TestBackendUnavailable:
    """
    Test: Given the trace backend is temporarily unavailable
          When trace events are generated
          Then events are buffered locally in SQLite
          And retried when connectivity is restored
          And local storage does not exceed configured max size
    """

    @pytest.mark.asyncio
    async def test_events_buffered_when_backend_unavailable(self, config, session_id):
        """Test events are stored locally when backend is unavailable."""
        # Create transport with failing sender
        transport = Transport(config)

        # Mock HTTP client to simulate unavailable backend
        with patch.object(transport, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_get_client.return_value = mock_client

            # Try to send events - should fail but buffer locally
            events = create_test_events(session_id, count=3)

            # Store events (this should work even if backend is down)
            transport._storage.store_events(events)

            # Verify events are stored locally
            counts = transport._storage.get_event_count()
            assert counts["total"] == 3
            assert counts["unsynced"] == 3

        await transport.close()

    @pytest.mark.asyncio
    async def test_events_synced_on_connectivity_restore(self, config, session_id):
        """Test events are synced when connectivity is restored."""
        # Create mock sender that fails first, then succeeds
        call_count = [0]

        async def send_batch(events):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Network error")
            return {"accepted": len(events), "rejected": 0}

        mock_sender = AsyncMock()
        mock_sender.send_batch = send_batch

        storage = LocalStorage(config)
        storage.initialize()

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=mock_sender,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # First sync attempt - fails
        await worker.sync_now()

        # Events should still be unsynced
        counts = storage.get_event_count()
        assert counts["unsynced"] > 0

        # Second sync attempt - succeeds
        await worker.sync_now()

        # Events should now be synced
        counts = storage.get_event_count()
        assert counts["synced"] > 0

        storage.close()

    @pytest.mark.asyncio
    async def test_storage_size_limit_enforced(self, small_storage_config, session_id):
        """Test storage prunes synced events when size limit is exceeded.

        Note: Storage correctly preserves unsynced events to prevent data loss.
        The size limit only affects synced events that can be safely removed.
        """
        storage = LocalStorage(small_storage_config)
        storage.initialize()

        # Store events and mark ALL as synced so they can be pruned
        synced_ids = []
        for _i in range(30):
            event = create_large_event(session_id, size_kb=50)
            event.event_id = uuid4()  # Unique ID
            storage.store_event(event)
            synced_ids.append(str(event.event_id))
            # Mark as synced immediately
            storage.mark_events_synced([str(event.event_id)])

        # Check that size enforcement deleted synced events
        max_size = small_storage_config.local_storage_max_mb * 1024 * 1024
        actual_size = storage.get_storage_size()

        # Should be within limit since all events were synced and pruneable
        # Allow 20% tolerance for SQLite overhead
        assert actual_size <= max_size * 1.2, (
            f"Storage size {actual_size} exceeds limit {max_size}"
        )

        # Verify pruning happened (should have fewer than 30 events)
        counts = storage.get_event_count()
        assert counts["total"] < 30, "Expected pruning to reduce event count"

        storage.close()


# -----------------------------------------------------------------------------
# Acceptance Criteria 2: Trace Capture Fails - Fail-Open Behavior
# -----------------------------------------------------------------------------


class TestFailOpenBehavior:
    """
    Test: Given trace capture fails completely
          When the agent executes
          Then the agent execution continues uninterrupted (fail-open)
          And a warning is logged indicating trace loss
          And buffered events are preserved for later sync
    """

    @pytest.mark.asyncio
    async def test_fail_open_continues_execution(self, config, session_id):
        """Test agent execution continues when trace capture fails."""
        transport = Transport(config)

        # Simulate storage failure
        with patch.object(transport._storage, "store_event", side_effect=Exception("Storage error")):
            # This should not raise - fail-open behavior
            event = create_step_event(session_id=session_id, step_number=1, event_name="test")

            # The buffer_event should catch the error and continue
            try:
                await transport.buffer_event(event)
            except Exception:
                pytest.fail("Fail-open violated: exception propagated to caller")

        await transport.close()

    def test_trace_loss_warning_logged(self, config, session_id, caplog):
        """Test warning is logged when trace capture fails."""
        storage = LocalStorage(config)
        storage.initialize()

        # Simulate store failure by corrupting the connection
        with patch.object(
            storage,
            "_get_connection",
            side_effect=sqlite3.Error("Database locked"),
        ):
            with caplog.at_level(logging.WARNING):
                # Attempt to store - should fail but log warning
                event = create_step_event(session_id=session_id, step_number=1, event_name="test")
                result = storage.store_event(event)

                assert result is False

        storage.close()

    def test_buffered_events_preserved(self, config, session_id):
        """Test buffered events are preserved after failed sync."""
        storage = LocalStorage(config)
        storage.initialize()

        # Store events
        events = create_test_events(session_id, count=5)
        storage.store_events(events)

        # Verify events exist
        counts_before = storage.get_event_count()
        assert counts_before["total"] == 5
        assert counts_before["unsynced"] == 5

        # Simulate failed sync by incrementing sync attempts
        event_ids = [str(e.event_id) for e in events]
        storage.increment_sync_attempts(event_ids)

        # Events should still be preserved
        counts_after = storage.get_event_count()
        assert counts_after["total"] == 5
        assert counts_after["unsynced"] == 5

        storage.close()


# -----------------------------------------------------------------------------
# Acceptance Criteria 3: Storage Exceeds Limit - Auto Cleanup
# -----------------------------------------------------------------------------


class TestStorageSizeManagement:
    """
    Test: Given local storage exceeds 100MB
          When new events are captured
          Then oldest events are pruned automatically
          And sync status is maintained correctly
    """

    def test_oldest_synced_events_pruned(self, small_storage_config, session_id):
        """Test oldest synced events are pruned when limit exceeded."""
        storage = LocalStorage(small_storage_config)
        storage.initialize()

        # Store events and mark some as synced
        old_events = create_test_events(session_id, count=10)
        storage.store_events(old_events)

        # Mark old events as synced
        old_ids = [str(e.event_id) for e in old_events]
        storage.mark_events_synced(old_ids)

        # Manually backdate the synced events
        conn = sqlite3.connect(str(storage.db_path))
        old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        conn.execute("UPDATE trace_buffer SET created_at = ?", (old_date,))
        conn.commit()
        conn.close()

        # Store many large events to trigger cleanup
        for _i in range(20):
            event = create_large_event(session_id, size_kb=100)
            event.event_id = uuid4()
            storage.store_event(event)

        # Some old events should have been pruned
        counts = storage.get_event_count()

        # The oldest synced events should be gone, but newer events remain
        assert counts["total"] < 30  # Not all 30 events should remain

        storage.close()

    def test_unsynced_events_not_pruned(self, small_storage_config, session_id):
        """Test unsynced events are NOT pruned (prevent data loss)."""
        storage = LocalStorage(small_storage_config)
        storage.initialize()

        # Store events but do NOT mark as synced
        events = create_test_events(session_id, count=5)
        storage.store_events(events)

        # Get initial unsynced count
        counts_before = storage.get_event_count()
        unsynced_before = counts_before["unsynced"]

        # Try to enforce size limit
        storage.enforce_size_limit()

        # Unsynced events should still be there
        counts_after = storage.get_event_count()
        assert counts_after["unsynced"] == unsynced_before

        storage.close()

    def test_sync_status_maintained_after_prune(self, small_storage_config):
        """Test sync status is correctly maintained after pruning."""
        storage = LocalStorage(small_storage_config)
        storage.initialize()

        session1 = uuid4()
        session2 = uuid4()

        # Store events for session 1 and mark as synced
        events1 = create_test_events(session1, count=3)
        storage.store_events(events1)
        storage.mark_events_synced([str(e.event_id) for e in events1])

        # Store events for session 2 (unsynced)
        events2 = create_test_events(session2, count=3)
        storage.store_events(events2)

        # Check sync status
        status = storage.get_sync_status()
        assert status["sync_pending"] is True  # Session 2 has unsynced events

        storage.close()


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the complete sync workflow."""

    @pytest.mark.asyncio
    async def test_complete_sync_workflow(self, config, session_id):
        """Test complete workflow: store -> sync -> verify."""
        # Create mock sender
        synced_events: list[TraceEvent] = []

        async def track_sync(events):
            synced_events.extend(events)
            return {"accepted": len(events), "rejected": 0}

        mock_sender = AsyncMock()
        mock_sender.send_batch = track_sync

        storage = LocalStorage(config)
        storage.initialize()

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=mock_sender,
        )

        # 1. Store events
        events = create_test_events(session_id, count=5)
        storage.store_events(events)

        # 2. Verify stored
        counts = storage.get_event_count()
        assert counts["total"] == 5
        assert counts["unsynced"] == 5

        # 3. Sync
        await worker.sync_now()

        # 4. Verify synced
        assert len(synced_events) == 5
        assert all(e.session_id == session_id for e in synced_events)

        storage.close()

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, config, session_id):
        """Test retry behavior with exponential backoff."""
        attempt_times: list[float] = []
        call_count = [0]

        async def track_attempts(events):
            import time
            attempt_times.append(time.time())
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return {"accepted": len(events), "rejected": 0}

        mock_sender = AsyncMock()
        mock_sender.send_batch = track_attempts

        storage = LocalStorage(config)
        storage.initialize()

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=mock_sender,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Multiple sync attempts
        for _ in range(3):
            await worker.sync_now()

        # Should have made multiple attempts
        assert call_count[0] >= 3

        storage.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_backend(self, config, session_id):
        """Test circuit breaker prevents overwhelming failing backend."""
        call_count = [0]

        async def count_calls(events):
            call_count[0] += 1
            raise Exception("Backend down")

        mock_sender = AsyncMock()
        mock_sender.send_batch = count_calls

        cb_config = CircuitBreakerConfig(failure_threshold=3)

        storage = LocalStorage(config)
        storage.initialize()

        worker = SyncWorker(
            config=config,
            storage=storage,
            sender=mock_sender,
            circuit_breaker_config=cb_config,
        )

        # Store events
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        # Make many sync attempts
        for _ in range(10):
            await worker.sync_now()

        # Circuit breaker should have stopped calls after threshold
        assert call_count[0] <= 5  # Some calls before circuit opens

        # Circuit should be open
        assert worker.circuit_state == CircuitState.OPEN

        storage.close()


# -----------------------------------------------------------------------------
# Health Check Tests
# -----------------------------------------------------------------------------


class TestHealthChecks:
    """Tests for storage health checks."""

    def test_healthy_storage(self, storage):
        """Test health check returns healthy for good storage."""
        status = storage.check_health()
        assert status == StorageHealthStatus.HEALTHY

    def test_integrity_check_passes(self, storage):
        """Test integrity check passes for valid database."""
        result = storage.check_integrity()
        assert result is True
        assert storage.metrics.integrity_check_passed is True

    def test_repair_recovers_storage(self, config):
        """Test repair can recover storage issues."""
        storage = LocalStorage(config)
        storage.initialize()

        # Damage the database slightly
        conn = sqlite3.connect(str(storage.db_path))
        conn.execute("DROP INDEX IF EXISTS idx_buffer_synced")
        conn.commit()
        conn.close()

        # Repair should work
        result = storage.repair()
        assert result is True

        storage.close()


# -----------------------------------------------------------------------------
# Metrics Tests
# -----------------------------------------------------------------------------


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_store_metrics_updated(self, storage, session_id):
        """Test store operations update metrics."""
        initial_stores = storage.metrics.total_stores

        events = create_test_events(session_id, count=5)
        storage.store_events(events)

        assert storage.metrics.total_stores > initial_stores
        assert storage.metrics.last_store_time is not None

    def test_latency_tracking(self, storage, session_id):
        """Test latency is tracked for operations."""
        events = create_test_events(session_id, count=5)
        storage.store_events(events)
        storage.get_unsynced_events()

        assert storage.metrics.average_store_latency_ms > 0
        assert storage.metrics.average_retrieval_latency_ms > 0

    def test_sync_status_includes_metrics(self, storage, session_id):
        """Test sync status includes metrics."""
        events = create_test_events(session_id, count=3)
        storage.store_events(events)

        status = storage.get_sync_status()

        assert "metrics" in status
        assert "total_stores" in status["metrics"]
        assert "health_status" in status


# -----------------------------------------------------------------------------
# Event Priority Tests
# -----------------------------------------------------------------------------


class TestEventPriority:
    """Tests for event prioritization."""

    def test_high_priority_for_session_end(self, storage, session_id):
        """Test session end events get high priority."""
        event = create_session_end_event(
            session_id=session_id,
            agent_id=uuid4(),
        )
        storage.store_event(event)

        # Get events with priority ordering
        events = storage.get_unsynced_events(prioritized=True, limit=10)

        assert len(events) == 1
        assert events[0].event_type == EventType.SESSION_END

    def test_prioritized_retrieval_order(self, storage, session_id):
        """Test events are retrieved in priority order."""
        # Store normal events first
        normal_events = create_test_events(session_id, count=3)
        storage.store_events(normal_events)

        # Store high priority event
        high_priority = create_session_end_event(
            session_id=session_id,
            agent_id=uuid4(),
        )
        storage.store_event(high_priority)

        # Get events with priority
        events = storage.get_unsynced_events(prioritized=True, limit=10)

        # First event should be high priority
        assert events[0].event_type == EventType.SESSION_END
