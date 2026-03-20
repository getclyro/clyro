# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK OTLP Exporter
# Implements FRD-S001, FRD-S002, FRD-S003, FRD-S004, FRD-S005, FRD-S008

"""
OTLP exporter for sending Clyro traces to a secondary destination.

This module translates Clyro TraceEvent records into OTLP format and
sends them to a user-configured OTLP/HTTP endpoint alongside the native
Clyro ingest path. The OTLP export path is strictly isolated from
native ingest — all exceptions are caught and never propagated.

Reference: TDD §2.2 C7, §3.3 (Clyro → OTLP mapping), §5.5, §5.6
"""

from __future__ import annotations

import asyncio
import gzip
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
import structlog

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

from clyro.trace import EventType, TraceEvent

logger = structlog.get_logger(__name__)

# Status mapping: Clyro → OTLP StatusCode (TDD §3.3)
_STATUS_MAP = {
    "success": 1,  # STATUS_CODE_OK
    "error": 2,  # STATUS_CODE_ERROR
    "unknown": 0,  # STATUS_CODE_UNSET
}


class OTLPExporter:
    """
    Async OTLP exporter that sends Clyro events to a secondary destination.

    Implements FRD-S001 (OTLP/HTTP export), FRD-S002 (co-existence with native),
    FRD-S004 (non-blocking async), FRD-S005 (failure isolation),
    FRD-S008 (graceful shutdown).

    Reference: TDD §2.2 C7
    """

    def __init__(self, config: ClyroConfig) -> None:
        """
        Initialize the OTLP exporter.

        Only created when otlp_export_endpoint is configured (FRD-S006).

        Args:
            config: SDK configuration with OTLP export settings.
        """
        self._endpoint = config.otlp_export_endpoint
        self._headers = dict(config.otlp_export_headers)
        self._timeout_ms = config.otlp_export_timeout_ms
        self._compression = config.otlp_export_compression
        self._queue_size = config.otlp_export_queue_size
        self._agent_name = config.agent_name or "unknown"

        self._queue: asyncio.Queue[list[TraceEvent]] = asyncio.Queue(maxsize=self._queue_size)
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

        # Metrics counters (in-process, for SDK-side tracking)
        self._dispatched_count = 0
        self._dropped_count = 0
        self._error_count = 0

    async def start(self) -> None:
        """Start the background export worker."""
        if self._running:
            return

        self._running = True
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_ms / 1000.0),
            headers={
                "User-Agent": "clyro-sdk/0.1.0",
                **self._headers,
            },
        )
        self._worker_task = asyncio.create_task(self._export_loop())
        logger.debug("otlp_exporter_started", endpoint=self._get_host())

    async def stop(self) -> None:
        """
        Gracefully stop the exporter. Implements FRD-S008.

        Drains the queue up to timeout, then drops remaining batches.
        """
        if not self._running:
            return

        self._running = False

        # Signal the worker to drain and stop
        if self._worker_task:
            # Give the worker time to drain (up to timeout)
            try:
                await asyncio.wait_for(
                    self._worker_task,
                    timeout=self._timeout_ms / 1000.0,
                )
            except TimeoutError:
                remaining = self._queue.qsize()
                if remaining > 0:
                    logger.warning(
                        "otlp_exporter_shutdown_timeout",
                        remaining_batches=remaining,
                    )
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.debug(
            "otlp_exporter_stopped",
            dispatched=self._dispatched_count,
            dropped=self._dropped_count,
            errors=self._error_count,
        )

    def dispatch(self, events: list[TraceEvent]) -> None:
        """
        Non-blocking dispatch of events to the OTLP export queue.

        Implements FRD-S004: adds ≤1ms latency. If queue is full,
        the batch is dropped with a warning log (no exception).

        Args:
            events: List of TraceEvent records to export.
        """
        try:
            self._queue.put_nowait(events)
            self._dispatched_count += 1
        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning(
                "otlp_export_queue_full",
                dropped_batch_size=len(events),
                queue_size=self._queue_size,
            )

    async def _export_loop(self) -> None:
        """Background worker that exports batches from the queue."""
        while self._running or not self._queue.empty():
            try:
                # Use timeout to allow checking self._running periodically
                events = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._send_batch(events)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # FRD-S005: catch everything — never propagate
                logger.warning("otlp_export_loop_error", error=str(e), fail_open=True)

    async def _send_batch(self, events: list[TraceEvent]) -> None:
        """
        Translate and send a batch of events to the secondary OTLP endpoint.

        Implements FRD-S003 (Clyro → OTLP translation), FRD-S005 (failure isolation).
        All exceptions are caught, logged, and swallowed.
        """
        try:
            payload = self._translate_batch(events)

            # Apply compression (FRD-S001)
            content_type = "application/x-protobuf"
            headers = {"Content-Type": content_type}

            if self._compression == "gzip":
                body = gzip.compress(payload)
                headers["Content-Encoding"] = "gzip"
            else:
                body = payload

            # Send to secondary destination
            response = await self._client.post(
                self._endpoint,
                content=body,
                headers=headers,
            )

            if response.status_code >= 400:
                self._error_count += 1
                logger.warning(
                    "otlp_export_http_error",
                    host=self._get_host(),
                    status_code=response.status_code,
                    batch_size=len(events),
                )

        except Exception as e:
            # FRD-S005: Catch ALL exceptions — never propagate to SDK caller
            self._error_count += 1
            logger.warning(
                "otlp_export_error",
                host=self._get_host(),
                error=type(e).__name__,
                batch_size=len(events),
            )

    def _translate_batch(self, events: list[TraceEvent]) -> bytes:
        """
        Translate Clyro events to OTLP ExportTraceServiceRequest.

        Implements FRD-S003 (Clyro → OTLP span attribute mapping).
        Uses the mapping table from TDD §3.3.

        Returns:
            Serialized Protobuf bytes.
        """
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from opentelemetry.proto.common.v1.common_pb2 import (
            AnyValue,
            KeyValue,
        )
        from opentelemetry.proto.resource.v1.resource_pb2 import Resource
        from opentelemetry.proto.trace.v1.trace_pb2 import (
            ResourceSpans,
            ScopeSpans,
        )

        # Group events by session_id (each session → one ResourceSpans)
        sessions: dict[str, list[TraceEvent]] = {}
        for event in events:
            sid = str(event.session_id)
            sessions.setdefault(sid, []).append(event)

        resource_spans_list = []
        for _session_id, session_events in sessions.items():
            # Build Resource with agent name attributes (FRD-S003)
            resource = Resource(
                attributes=[
                    KeyValue(
                        key="clyro.agent.name",
                        value=AnyValue(string_value=self._agent_name),
                    ),
                    KeyValue(
                        key="service.name",
                        value=AnyValue(string_value=self._agent_name),
                    ),
                ]
            )

            spans = []
            for event in session_events:
                try:
                    span = self._translate_event(event)
                    spans.append(span)
                except Exception as e:
                    # FRD-S005: per-span translation error → skip span, log warning
                    logger.warning(
                        "otlp_export_span_translation_error",
                        event_id=str(event.event_id),
                        error=str(e),
                    )

            if spans:
                resource_spans_list.append(
                    ResourceSpans(
                        resource=resource,
                        scope_spans=[ScopeSpans(spans=spans)],
                    )
                )

        request = ExportTraceServiceRequest(resource_spans=resource_spans_list)
        return request.SerializeToString()

    def _translate_event(self, event: TraceEvent) -> Any:
        """
        Translate a single Clyro event to an OTLP Span.

        Implements TDD §3.3 field mapping.
        """
        from opentelemetry.proto.common.v1.common_pb2 import (
            AnyValue,
            KeyValue,
        )
        from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

        # session_id → trace_id: UUID hex to 16 bytes (FRD-S003)
        trace_id = UUID(str(event.session_id)).bytes

        # event_id → span_id: lower 8 bytes of UUID (FRD-S003)
        span_id = UUID(str(event.event_id)).bytes[8:]

        # parent_event_id → parent_span_id (FRD-S003)
        parent_span_id = b""
        if event.parent_event_id:
            parent_span_id = UUID(str(event.parent_event_id)).bytes[8:]

        # Timing: timestamp → start_time_unix_nano (FRD-S003)
        start_time_ns = int(event.timestamp.timestamp() * 1_000_000_000)
        end_time_ns = start_time_ns + (event.duration_ms * 1_000_000)

        # Status mapping (FRD-S003)
        status_value = (
            event.metadata.get("_otlp_status", "unknown") if event.metadata else "unknown"
        )
        # Derive status from event_type if no _otlp_status
        if event.event_type == EventType.ERROR:
            status_code = 2  # ERROR
        elif status_value == "success":
            status_code = 1  # OK
        else:
            status_code = 0  # UNSET

        # Build span attributes
        attributes = []

        # Add loop stage attribute (FRD-S003)
        attributes.append(
            KeyValue(
                key="clyro.loop.stage",
                value=AnyValue(string_value=event.agent_stage.value),
            )
        )

        # Add metadata as span attributes (preserving all keys)
        if event.metadata:
            for key, value in event.metadata.items():
                if key.startswith("_"):
                    continue  # Skip internal keys
                attributes.append(
                    KeyValue(
                        key=key,
                        value=_python_to_any_value(value),
                    )
                )

        return Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=event.event_name or event.event_type.value,
            kind=Span.SPAN_KIND_INTERNAL,
            start_time_unix_nano=start_time_ns,
            end_time_unix_nano=end_time_ns,
            attributes=attributes,
            status=Status(code=status_code),
        )

    def _get_host(self) -> str:
        """Extract host from endpoint URL for safe logging (FRD-S005)."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(self._endpoint)
            return parsed.hostname or "unknown"
        except Exception:
            return "unknown"

    @property
    def stats(self) -> dict[str, int]:
        """Get export statistics."""
        return {
            "dispatched": self._dispatched_count,
            "dropped": self._dropped_count,
            "errors": self._error_count,
            "queue_size": self._queue.qsize(),
        }


def _python_to_any_value(value: Any) -> Any:
    """Convert a Python value to OTLP AnyValue."""
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue

    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    elif isinstance(value, int):
        return AnyValue(int_value=value)
    elif isinstance(value, float):
        return AnyValue(double_value=value)
    elif isinstance(value, str):
        return AnyValue(string_value=value)
    else:
        return AnyValue(string_value=str(value))
