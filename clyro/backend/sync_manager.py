# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Backend Sync Manager
# Implements FRD-015, FRD-018, FRD-019

"""
Orchestrate the complete backend sync lifecycle:
- Startup recovery of pending events from previous sessions.
- Periodic batch uploads to ``POST /v1/traces``.
- Shutdown flush with bounded timeout.
- Coordination between EventQueue, CircuitBreaker, and HttpSyncClient.

The sync loop runs as an asyncio task on the same event loop as the
message router — cooperative scheduling ensures sync never blocks
message processing (TDD §2.12).
"""

from __future__ import annotations

import asyncio
from typing import Any

from clyro.backend.circuit_breaker import (
    CircuitBreaker,
    ConnectivityDetector,
    ConnectivityStatus,
)
from clyro.backend.event_queue import EventQueue
from clyro.backend.http_client import (
    AuthenticationError,
    HttpSyncClient,
    RateLimitExhaustedError,
)
from clyro.mcp.log import get_logger

logger = get_logger(__name__)


_MAX_BATCH_SIZE = 100  # FRD-015: max events per sync batch
_SHUTDOWN_TIMEOUT_SECONDS = 3.0  # TDD §2.12: max wait during shutdown flush
_SHUTDOWN_MAX_ATTEMPTS = 3  # TDD §2.12: max batch syncs during shutdown


class BackendSyncManager:
    """
    Orchestrate background trace sync to Clyro backend (FRD-015).

    Manages EventQueue persistence, CircuitBreaker state, and
    HttpSyncClient batch uploads on an asyncio background task.

    Args:
        event_queue: File-based event persistence.
        circuit_breaker: Circuit breaker for backend calls.
        connectivity: Connectivity state tracker.
        http_client: HTTP client for backend API.
        sync_interval: Seconds between sync cycles.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        circuit_breaker: CircuitBreaker,
        connectivity: ConnectivityDetector,
        http_client: HttpSyncClient,
        sync_interval: int = 5,
    ) -> None:
        self._event_queue = event_queue
        self._circuit_breaker = circuit_breaker
        self._connectivity = connectivity
        self._http_client = http_client
        self._sync_interval = sync_interval
        self._running = False
        self._sync_trigger = asyncio.Event()
        self._sync_task: asyncio.Task[None] | None = None
        self._disabled = False  # Set on auth failure
        self._pending_violations: list[dict[str, Any]] = []  # FRD-006

    def start(self) -> None:
        """Start the background sync loop (FRD-015)."""
        if self._running:
            return
        self._running = True
        self._sync_task = asyncio.ensure_future(self._sync_loop())

        # Schedule immediate sync if there are recovered events
        if self._event_queue.pending_count > 0:
            logger.info(
                "recovered_pending_events",
                count=self._event_queue.pending_count,
            )
            self._sync_trigger.set()

    def enqueue(self, trace_event: dict[str, Any]) -> None:
        """
        Add a trace event to the sync queue (FRD-015).

        Non-blocking. Called by TraceEventFactory for each event.
        Triggers immediate sync if batch size threshold is reached.
        """
        if self._disabled:
            return
        self._event_queue.append(trace_event)
        if self._event_queue.pending_count >= _MAX_BATCH_SIZE:
            self._sync_trigger.set()

    def enqueue_violation(self, violation: dict[str, Any]) -> None:
        """
        Add a policy violation to the sync queue (FRD-006).

        Non-blocking. Called by AuditLogger on blocked calls with
        block_reason == "policy_violation".
        """
        if self._disabled:
            return
        self._pending_violations.append(violation)

    async def shutdown(self) -> None:
        """
        Flush remaining events and stop sync loop (TDD §2.12).

        Attempts up to 3 batch syncs within 3-second timeout.
        Remaining events persist in EventQueue file for cross-session recovery.
        """
        self._running = False
        self._sync_trigger.set()

        if self._sync_task is not None:
            try:
                async with asyncio.timeout(_SHUTDOWN_TIMEOUT_SECONDS):
                    # Attempt final sync batches
                    for _ in range(_SHUTDOWN_MAX_ATTEMPTS):
                        if self._event_queue.pending_count == 0:
                            break
                        await self._perform_sync()
            except TimeoutError:
                remaining = self._event_queue.pending_count
                if remaining > 0:
                    logger.warning(
                        "shutdown_timeout",
                        remaining_events=remaining,
                        action="persisted_for_next_session",
                    )

            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        """Background sync loop — runs on asyncio event loop (TDD §2.12)."""
        while self._running:
            try:
                await asyncio.wait_for(
                    self._sync_trigger.wait(),
                    timeout=self._sync_interval,
                )
                self._sync_trigger.clear()
            except TimeoutError:
                pass  # Normal interval timeout — proceed with sync

            if not self._running:
                break

            await self._perform_sync()

    async def _perform_sync(self) -> None:
        """Execute a single sync batch (FRD-015, FRD-019)."""
        if self._disabled:
            return

        # Check circuit breaker (FRD-019)
        if not self._circuit_breaker.can_execute():
            return

        # Read batch from queue, prioritizing session_end/error events (TDD §2.12)
        pending = self._event_queue.load_pending()
        if not pending:
            # Even if no trace events, attempt violation sync
            await self._sync_violations()
            return

        # Priority sort: session_end and error events first
        _PRIORITY_TYPES = {"session_end", "error"}
        priority = [e for e in pending if e.get("event_type") in _PRIORITY_TYPES]
        rest = [e for e in pending if e.get("event_type") not in _PRIORITY_TYPES]
        sorted_pending = priority + rest

        batch = sorted_pending[:_MAX_BATCH_SIZE]
        event_ids = {e.get("event_id") for e in batch if e.get("event_id")}

        # Strip internal queue metadata before sending (API uses extra="forbid")
        clean_batch = [{k: v for k, v in e.items() if k != "queued_at"} for e in batch]

        try:
            await self._http_client.send_batch(clean_batch)

            # Success
            self._event_queue.remove_synced(event_ids)
            self._circuit_breaker.record_success()
            status = self._connectivity.record_success()

            # Trigger immediate re-sync if reconnected and more events pending
            if status == ConnectivityStatus.CONNECTED and self._event_queue.pending_count > 0:
                self._sync_trigger.set()

        except AuthenticationError as exc:
            # Disable sync for this session (FRD-015)
            self._disabled = True
            logger.error(
                "backend_auth_failed",
                status_code=exc.status_code,
                action="sync_disabled",
            )
        except RateLimitExhaustedError:
            # Rate limited — events stay in queue, no circuit breaker trip
            logger.warning("backend_rate_limited", action="events_queued")
        except Exception as exc:
            # Network/server error — events stay in queue
            logger.warning(
                "backend_sync_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._circuit_breaker.record_failure()
            self._connectivity.record_failure()

        # FRD-006: sync pending violations after trace events
        await self._sync_violations()

    async def _sync_violations(self) -> None:
        """Sync pending policy violations to backend (FRD-006)."""
        if not self._pending_violations or self._disabled:
            return

        batch = self._pending_violations[:_MAX_BATCH_SIZE]
        try:
            await self._http_client.report_violations(batch)
            self._pending_violations = self._pending_violations[len(batch) :]
        except AuthenticationError:
            self._disabled = True
        except Exception as e:
            logger.debug("violation_sync_failed", error=str(e), fail_open=True)

    @property
    def is_disabled(self) -> bool:
        """True if sync was disabled due to auth failure."""
        return self._disabled
