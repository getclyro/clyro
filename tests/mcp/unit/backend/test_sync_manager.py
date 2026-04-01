"""
Unit tests for BackendSyncManager — TDD §11.1 v1.1 tests.

FRD-015: Background trace sync.
FRD-018: Event queue coordination.
FRD-019: Circuit breaker integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from clyro.backend.circuit_breaker import CircuitBreaker, ConnectivityDetector
from clyro.backend.event_queue import EventQueue
from clyro.backend.http_client import (
    AuthenticationError,
    HttpSyncClient,
    RateLimitExhaustedError,
)
from clyro.backend.sync_manager import BackendSyncManager


@pytest.fixture
def event_queue(tmp_path) -> EventQueue:
    q = EventQueue(instance_id="test12345678", max_size_mb=1)
    q._path = tmp_path / "mcp-pending-test12345678.jsonl"
    return q


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker()


@pytest.fixture
def connectivity() -> ConnectivityDetector:
    return ConnectivityDetector()


@pytest.fixture
def http_client() -> HttpSyncClient:
    client = MagicMock(spec=HttpSyncClient)
    client.send_batch = AsyncMock(return_value={"accepted": 5, "rejected": 0})
    return client


@pytest.fixture
def manager(event_queue, circuit_breaker, connectivity, http_client) -> BackendSyncManager:
    return BackendSyncManager(
        event_queue=event_queue,
        circuit_breaker=circuit_breaker,
        connectivity=connectivity,
        http_client=http_client,
        sync_interval=1,
    )


def _make_event() -> dict:
    return {"event_id": str(uuid4()), "event_type": "tool_call"}


class TestSyncManagerEnqueue:
    """Enqueue events for sync (FRD-015)."""

    def test_enqueue_adds_to_queue(self, manager: BackendSyncManager, event_queue: EventQueue) -> None:
        manager.enqueue(_make_event())
        assert event_queue.pending_count == 1

    def test_enqueue_ignored_when_disabled(self, manager: BackendSyncManager, event_queue: EventQueue) -> None:
        manager._disabled = True
        manager.enqueue(_make_event())
        assert event_queue.pending_count == 0


class TestSyncManagerPerformSync:
    """Batch sync execution (FRD-015, FRD-019)."""

    @pytest.mark.asyncio
    async def test_syncs_pending_events(
        self, manager: BackendSyncManager, event_queue: EventQueue, http_client
    ) -> None:
        event = _make_event()
        event_queue.append(event)
        await manager._perform_sync()
        http_client.send_batch.assert_called_once()
        assert event_queue.pending_count == 0

    @pytest.mark.asyncio
    async def test_skips_when_circuit_open(
        self, manager: BackendSyncManager, circuit_breaker: CircuitBreaker, event_queue: EventQueue
    ) -> None:
        event_queue.append(_make_event())
        # Force circuit open
        for _ in range(CircuitBreaker.FAILURE_THRESHOLD):
            circuit_breaker.record_failure()
        await manager._perform_sync()
        assert event_queue.pending_count == 1  # Not synced

    @pytest.mark.asyncio
    async def test_skips_when_no_events(
        self, manager: BackendSyncManager, http_client
    ) -> None:
        await manager._perform_sync()
        http_client.send_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_disables_on_auth_error(
        self, manager: BackendSyncManager, event_queue: EventQueue, http_client
    ) -> None:
        event_queue.append(_make_event())
        http_client.send_batch = AsyncMock(side_effect=AuthenticationError(401))
        await manager._perform_sync()
        assert manager.is_disabled is True

    @pytest.mark.asyncio
    async def test_records_failure_on_error(
        self, manager: BackendSyncManager, event_queue: EventQueue,
        http_client, circuit_breaker: CircuitBreaker
    ) -> None:
        event_queue.append(_make_event())
        http_client.send_batch = AsyncMock(side_effect=Exception("network"))
        await manager._perform_sync()
        # Circuit breaker should have recorded a failure
        assert circuit_breaker.get_state().failure_count == 1
        # Event should still be in queue
        assert event_queue.pending_count == 1


class TestSyncManagerRateLimitHandling:
    """Rate limit exhaustion preserves events in queue (M5 fix)."""

    @pytest.mark.asyncio
    async def test_events_preserved_on_rate_limit(
        self, manager: BackendSyncManager, event_queue: EventQueue, http_client
    ) -> None:
        event_queue.append(_make_event())
        http_client.send_batch = AsyncMock(side_effect=RateLimitExhaustedError())
        await manager._perform_sync()
        # Events should still be in queue (not removed)
        assert event_queue.pending_count == 1


class TestSyncManagerStartStop:
    """Start and stop the sync loop (FRD-015)."""

    @pytest.mark.asyncio
    async def test_start_creates_task(self, manager: BackendSyncManager) -> None:
        manager.start()
        assert manager._sync_task is not None
        assert manager._running is True
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_stops_loop(self, manager: BackendSyncManager) -> None:
        manager.start()
        await manager.shutdown()
        assert manager._running is False
        assert manager._sync_task is None


class TestSyncManagerRecovery:
    """Startup recovery of pending events (FRD-018)."""

    @pytest.mark.asyncio
    async def test_logs_recovered_events(
        self, manager: BackendSyncManager, event_queue: EventQueue, capsys
    ) -> None:
        event_queue.append(_make_event())
        event_queue.append(_make_event())
        manager.start()
        captured = capsys.readouterr()
        assert "recovered_pending_events" in captured.err
        await manager.shutdown()


class TestSyncManagerShutdownFlush:
    """Shutdown flush attempts to sync remaining events (TDD §2.12)."""

    @pytest.mark.asyncio
    async def test_flushes_on_shutdown(
        self, manager: BackendSyncManager, event_queue: EventQueue, http_client
    ) -> None:
        event_queue.append(_make_event())
        manager.start()
        await manager.shutdown()
        # The sync should have been attempted during shutdown
        http_client.send_batch.assert_called()
