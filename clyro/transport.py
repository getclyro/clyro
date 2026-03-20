# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Transport Layer
# Implements PRD-005, PRD-006

"""
HTTP transport layer for the Clyro SDK.

This module handles communication with the backend API, including:
- Batched event uploads with retry and exponential backoff
- Local buffering on failure for offline operation
- Background sync via SyncWorker integration
- Circuit breaker for failure protection
- Connectivity detection and auto-recovery
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from clyro.exceptions import TransportError
from clyro.storage.sqlite import LocalStorage
from clyro.workers.sync_worker import (
    CircuitBreakerConfig,
    ConnectivityStatus,
    EventSender,
    SyncWorker,
)
from clyro.trace import TraceEvent

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.otlp_exporter import OTLPExporter

logger = structlog.get_logger(__name__)

# HTTP client timeout settings
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=30.0,
    write=10.0,
    pool=5.0,
)


class HttpEventSender(EventSender):
    """
    HTTP-based event sender implementing the EventSender protocol.

    This class handles the actual HTTP communication with the backend,
    used by SyncWorker for batch event uploads.
    """

    def __init__(
        self,
        config: ClyroConfig,
        get_client: Any,  # Callable returning httpx.AsyncClient
    ):
        """
        Initialize HTTP event sender.

        Args:
            config: SDK configuration
            get_client: Async function to get HTTP client
        """
        self.config = config
        self._get_client = get_client

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
        if not events:
            return {"accepted": 0, "rejected": 0, "errors": []}

        # Exclude org_id from events - the API determines it from the authenticated API key
        payload = {
            "events": [
                {k: v for k, v in event.to_dict().items() if k != "org_id"} for event in events
            ]
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.retry_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            reraise=True,
        ):
            with attempt:
                client = await self._get_client()
                response = await client.post(
                    f"{self.config.endpoint}/v1/traces",
                    json=payload,
                )
                # Handle rate limiting
                if response.status_code == 429:
                    try:
                        retry_after = int(response.headers.get("Retry-After", "5"))
                        retry_after = max(1, min(retry_after, 60))
                    except (ValueError, TypeError):
                        retry_after = 5
                    logger.warning("rate_limited", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    raise httpx.NetworkError("Rate limited")

                response.raise_for_status()
                return response.json()

        # Should not reach here, but return empty result as fallback
        return {"accepted": 0, "rejected": 0, "errors": []}


class Transport:
    """
    HTTP transport for communicating with the Clyro backend.

    Handles:
    - Batched event uploads with retry logic
    - Local buffering on failure (fail-open behavior)
    - Background sync via SyncWorker
    - Circuit breaker for failure protection
    - Connectivity detection and auto-recovery

    Implements PRD-005, PRD-006
    """

    def __init__(
        self,
        config: ClyroConfig,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize transport.

        Args:
            config: SDK configuration
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self.config = config
        self._storage = LocalStorage(config)
        self._client: httpx.AsyncClient | None = None
        self._running = False
        self._event_buffer: list[TraceEvent] = []
        self._buffer_lock = asyncio.Lock()

        # Create event sender for SyncWorker
        self._sender = HttpEventSender(config, self._get_client)

        # Create SyncWorker for background sync
        self._sync_worker = SyncWorker(
            config=config,
            storage=self._storage,
            sender=self._sender,
            circuit_breaker_config=circuit_breaker_config,
        )

        # Create OTLP exporter if endpoint configured (FRD-S001)
        self._otlp_exporter: OTLPExporter | None = None
        if getattr(config, "otlp_export_endpoint", None):
            from clyro.otlp_exporter import OTLPExporter

            self._otlp_exporter = OTLPExporter(config)

    @property
    def endpoint(self) -> str:
        """Get the backend endpoint URL."""
        return self.config.endpoint

    @property
    def is_local_only(self) -> bool:
        """Check if operating in local-only mode."""
        return self.config.is_local_only()

    @property
    def connectivity_status(self) -> ConnectivityStatus:
        """Get current connectivity status."""
        return self._sync_worker.connectivity_status

    @property
    def storage(self) -> LocalStorage:
        """Get the local storage instance."""
        return self._storage

    @property
    def sync_worker(self) -> SyncWorker:
        """Get the sync worker instance."""
        return self._sync_worker

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {
                "User-Agent": "clyro-sdk/0.1.0",
                "Content-Type": "application/json",
            }
            if self.config.api_key:
                headers["X-Clyro-API-Key"] = self.config.api_key

            self._client = httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT,
                headers=headers,
            )
        return self._client

    async def send_events(self, events: list[TraceEvent]) -> dict[str, Any]:
        """
        Send trace events to the backend with retry and buffering.

        Retry Strategy:
        - Retries up to max_attempts times with exponential backoff (1s, 2s, 4s, ...)
        - Only retries on network/timeout errors, not on client errors (4xx)
        - Honors 429 rate limit responses with Retry-After header
        - Falls back to local storage if all retries fail (fail-open)

        Args:
            events: List of trace events to send

        Returns:
            Response from the backend with accepted/rejected counts

        Raises:
            TransportError: If all retry attempts fail
        """
        # Early return for empty batch
        if not events:
            return {"accepted": 0, "rejected": 0, "errors": []}

        # Local-only mode: store events locally without sending to backend
        if self.is_local_only:
            stored = self._storage.store_events(events)
            return {
                "accepted": stored,
                "rejected": len(events) - stored,
                "errors": [],
                "local_only": True,
            }

        # Delegate HTTP call to HttpEventSender (single source of truth
        # for payload format, retry logic, rate-limit handling, org_id stripping)
        try:
            result = await self._sender.send_batch(events)

            # Mark events as successfully synced in local storage
            # This prevents duplicate uploads on next sync attempt
            event_ids = [str(e.event_id) for e in events]
            self._storage.mark_events_synced(event_ids)

            # Update connectivity status on success
            self._sync_worker.record_sync_success()

            # Dispatch to OTLP exporter in parallel (FRD-S004, non-blocking)
            if self._otlp_exporter is not None:
                self._otlp_exporter.dispatch(events)

            logger.debug(
                "events_sent",
                accepted=result.get("accepted", 0),
                rejected=result.get("rejected", 0),
            )
            return result

        except RetryError as e:
            # All retry attempts exhausted - fallback to local buffering
            logger.warning("send_failed_buffering", error=str(e.last_attempt.exception()))
            self._storage.store_events(events)

            # Update connectivity status on failure
            self._sync_worker.record_sync_failure()

            raise TransportError(
                message="Failed to send events after retries",
                endpoint=f"{self.endpoint}/v1/traces",
            ) from e

        except httpx.HTTPStatusError as e:
            # HTTP error response (4xx or 5xx)
            # Only buffer on server errors (5xx), not client errors (4xx)
            if e.response.status_code >= 500:
                # Server error - buffer locally for later retry
                self._storage.store_events(events)
                self._sync_worker.record_sync_failure()
            raise TransportError(
                message=f"HTTP error: {e.response.status_code}",
                endpoint=f"{self.endpoint}/v1/traces",
                status_code=e.response.status_code,
            ) from e

        except (httpx.TimeoutException, httpx.NetworkError) as e:
            # Retries exhausted with reraise=True - raw httpx exceptions
            # propagate instead of RetryError. Fallback to local buffering.
            logger.warning("send_failed_buffering", error=str(e))
            self._storage.store_events(events)
            self._sync_worker.record_sync_failure()

            raise TransportError(
                message="Failed to send events after retries",
                endpoint=f"{self.endpoint}/v1/traces",
            ) from e

    async def buffer_event(self, event: TraceEvent) -> None:
        """
        Buffer an event for later batch upload with overflow protection.

        Implements fail-open behavior: storage errors are logged but do not
        propagate to the caller, allowing agent execution to continue.

        Prevents memory exhaustion by limiting buffer size to 10x batch size.
        Drops oldest events when buffer overflows to maintain system stability.

        Note: Events are NOT stored in local SQLite storage immediately.
        They are only stored if the send from memory buffer fails. This
        prevents duplicate sends from both the memory buffer and the
        background sync worker.

        Args:
            event: Event to buffer
        """
        # Check if we need to flush (before acquiring lock)
        should_flush = False

        async with self._buffer_lock:
            # Prevent memory exhaustion with buffer size limit
            MAX_BUFFER_SIZE = self.config.batch_size * 10  # 10x batch size
            if len(self._event_buffer) >= MAX_BUFFER_SIZE:
                logger.warning(
                    "event_buffer_overflow",
                    buffer_size=len(self._event_buffer),
                    max_size=MAX_BUFFER_SIZE,
                    dropping_oldest=True,
                )
                self._event_buffer.pop(0)  # Drop oldest event to prevent OOM

            self._event_buffer.append(event)

            # Check if buffer is full (don't flush while holding lock)
            should_flush = len(self._event_buffer) >= self.config.batch_size

        # Flush outside the lock to avoid deadlock
        if should_flush:
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush buffered events to the backend."""
        async with self._buffer_lock:
            if not self._event_buffer:
                return

            events = self._event_buffer.copy()
            self._event_buffer.clear()

        if not self.is_local_only:
            try:
                await self.send_events(events)
            except TransportError:
                # Events already stored locally on failure (fail-open)
                pass
            except Exception as e:
                # Unexpected error — store events locally to prevent data loss
                logger.warning("flush_unexpected_error_buffering", error=str(e))
                try:
                    self._storage.store_events(events)
                except Exception:
                    logger.error("flush_local_storage_failed", events_lost=len(events))

    async def start_background_sync(self) -> None:
        """
        Start background sync via SyncWorker.

        The SyncWorker handles:
        - Periodic sync of unsynced events
        - Circuit breaker for failure protection
        - Connectivity detection and auto-recovery
        """
        if self._running:
            return

        self._running = True
        await self._sync_worker.start()

        # Start OTLP exporter background loop (FRD-S004)
        if self._otlp_exporter is not None:
            await self._otlp_exporter.start()

        logger.debug("background_sync_started")

    async def stop_background_sync(self) -> None:
        """Stop background sync."""
        self._running = False
        await self._sync_worker.stop()

        # Final flush of memory buffer
        await self._flush_buffer()
        logger.debug("background_sync_stopped")

    async def trigger_sync(self) -> None:
        """Trigger an immediate sync attempt."""
        await self._sync_worker.trigger_immediate_sync()

    async def flush(self) -> None:
        """
        Flush all pending events from in-memory buffer.

        Note: This only flushes the in-memory buffer. Any events that failed
        to send will remain in local storage and will be synced by the
        background sync worker.
        """
        await self._flush_buffer()

    async def close(self) -> None:
        """Close transport and cleanup resources."""
        await self.stop_background_sync()

        # Stop OTLP exporter (drains pending queue — FRD-S008)
        if self._otlp_exporter is not None:
            await self._otlp_exporter.stop()

        if self._client:
            await self._client.aclose()
            self._client = None

        self._storage.close()

    def get_sync_status(self) -> dict[str, Any]:
        """
        Get comprehensive sync status information.

        Returns:
            Dictionary with transport, storage, and worker status
        """
        return {
            "transport": {
                "endpoint": self.endpoint,
                "local_only": self.is_local_only,
                "running": self._running,
                "buffer_size": len(self._event_buffer),
                "connectivity_status": self._sync_worker.connectivity_status.value,
                "circuit_state": self._sync_worker.circuit_state.value,
            },
            "storage": self._storage.get_sync_status(),
            "worker": self._sync_worker.get_status(),
        }

    def check_health(self) -> dict[str, Any]:
        """
        Perform health check on transport and storage.

        Returns:
            Health status dictionary
        """
        storage_health = self._storage.check_health()
        connectivity = self._sync_worker.connectivity_status

        # Determine overall health
        if storage_health.value == "corrupted":
            overall = "unhealthy"
        elif storage_health.value == "unhealthy":
            overall = "unhealthy"
        elif connectivity == ConnectivityStatus.DISCONNECTED:
            overall = "degraded"
        elif storage_health.value == "degraded":
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "storage": storage_health.value,
            "connectivity": connectivity.value,
            "circuit_state": self._sync_worker.circuit_state.value,
            "sync_worker_running": self._sync_worker.is_running,
        }


class SyncTransport:
    """
    Synchronous wrapper for Transport.

    Provides a synchronous interface for use in non-async contexts.
    """

    def __init__(self, config: ClyroConfig):
        """Initialize sync transport."""
        self.config = config
        self._transport = Transport(config)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get or create an event loop for synchronous operation.

        This method handles three distinct cases:
        1. Running loop exists → Error (can't use sync wrapper in async context)
        2. Stopped loop exists → Reuse it
        3. No loop exists → Create new loop

        Event Loop Lifecycle:
        - asyncio.get_running_loop() raises RuntimeError if no loop is running
        - If a loop exists but is stopped, we can reuse it safely
        - If we're inside an active event loop, we must error out to prevent deadlocks
          (run_until_complete() would block forever inside a running loop)

        Returns:
            Event loop for running async operations synchronously

        Raises:
            RuntimeError: If called from within an active event loop
        """
        try:
            # Try to get the currently running event loop
            # This will succeed if we're in an async context
            loop = asyncio.get_running_loop()

            # If we reach here, a loop exists. Check if it's running.
            if loop.is_running():
                # We're inside an active event loop - cannot use run_until_complete()
                # This would cause a deadlock, so we error out immediately
                raise RuntimeError(
                    "SyncTransport cannot be used inside a running event loop. "
                    "Use the async Transport or wrap async agents instead."
                )

            # Loop exists but is stopped - safe to reuse
            self._loop = loop

        except RuntimeError as exc:
            # RuntimeError can mean two things:
            # 1. Our custom error from above (loop is running)
            # 2. No event loop exists (asyncio.get_running_loop() failed)

            # Check if it's our custom error and re-raise it
            if str(exc).startswith("SyncTransport cannot be used"):
                raise

            # No event loop exists - create a new one
            # This is the normal case for synchronous code
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        return self._loop

    @property
    def storage(self) -> LocalStorage:
        """Get the local storage instance."""
        return self._transport.storage

    def send_events(self, events: list[TraceEvent]) -> dict[str, Any]:
        """Send events synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self._transport.send_events(events))

    def buffer_event(self, event: TraceEvent) -> None:
        """Buffer an event synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._transport.buffer_event(event))

    def start_background_sync(self) -> None:
        """
        Start background sync synchronously.

        Uses a background thread with its own event loop to run
        the async background sync worker without blocking the main thread.

        Fail-open behavior: Errors are logged but not raised.
        """
        import threading

        def _start_sync_in_thread():
            """Run background sync in a separate thread with its own event loop."""
            try:
                # Create new event loop for this thread
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)

                # Start background sync
                thread_loop.run_until_complete(self._transport.start_background_sync())

                # Keep the loop running to maintain background tasks
                # The loop will stop when transport.stop_background_sync() is called
                thread_loop.run_forever()
            except Exception as e:
                # Fail-open: log error but don't crash
                logger.warning(
                    "background_sync_start_failed",
                    error=str(e),
                    fail_open=True,
                )
            finally:
                try:
                    thread_loop.close()
                except Exception:
                    pass

        # Start background sync in daemon thread (won't block process exit)
        sync_thread = threading.Thread(
            target=_start_sync_in_thread,
            daemon=True,
            name="clyro-background-sync",
        )
        sync_thread.start()
        logger.debug("background_sync_thread_started")

    def flush(self) -> None:
        """Flush all pending events synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._transport.flush())

    def close(self) -> None:
        """Close transport synchronously."""
        loop = self._get_loop()
        loop.run_until_complete(self._transport.close())

    def get_sync_status(self) -> dict[str, Any]:
        """Get sync status."""
        return self._transport.get_sync_status()

    def check_health(self) -> dict[str, Any]:
        """Perform health check."""
        return self._transport.check_health()
