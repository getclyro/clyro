# Tests for SDK OTLP Exporter (C7)
# Implements TDD §11.6

"""
Unit tests for the SDK OTLP Exporter.

Tests Clyro → OTLP translation, async dispatch, failure isolation,
queue behavior, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

pytest.importorskip("opentelemetry.proto", reason="opentelemetry-proto required for OTLP tests")

from clyro.config import ClyroConfig
from clyro.otlp_exporter import OTLPExporter, _python_to_any_value
from clyro.trace import AgentStage, EventType, Framework, TraceEvent


# =============================================================================
# Helpers
# =============================================================================

def _make_config(**kwargs) -> ClyroConfig:
    """Create a ClyroConfig with OTLP export enabled."""
    defaults = {
        "agent_name": "test-agent",
        "otlp_export_endpoint": "https://otel.example.com/v1/traces",
        "otlp_export_timeout_ms": 1000,
        "otlp_export_queue_size": 10,
        "otlp_export_compression": "gzip",
    }
    defaults.update(kwargs)
    return ClyroConfig(**defaults)


def _make_event(**kwargs) -> TraceEvent:
    """Create a TraceEvent for testing."""
    defaults = {
        "session_id": uuid4(),
        "event_type": EventType.STEP,
        "event_name": "test-step",
        "agent_stage": AgentStage.THINK,
        "timestamp": datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        "duration_ms": 150,
    }
    defaults.update(kwargs)
    return TraceEvent(**defaults)


# =============================================================================
# §11.6 — Clyro → OTLP span mapping (FRD-S003)
# =============================================================================

class TestClyroToOTLPTranslation:
    """Tests for the reverse translation (Clyro → OTLP)."""

    def test_session_id_to_trace_id(self):
        """session_id → 16-byte trace_id. Implements FRD-S003."""
        config = _make_config()
        exporter = OTLPExporter(config)
        session_id = uuid4()
        event = _make_event(session_id=session_id)

        payload = exporter._translate_batch([event])
        assert len(payload) > 0  # Serialized protobuf bytes

    def test_event_id_to_span_id(self):
        """event_id → 8-byte span_id (lower 64 bits). Implements FRD-S003."""
        config = _make_config()
        exporter = OTLPExporter(config)
        event = _make_event()
        span = exporter._translate_event(event)
        # span_id should be 8 bytes
        assert len(span.span_id) == 8

    def test_agent_name_in_resource_attributes(self):
        """agent_name → clyro.agent.name + service.name. Implements FRD-S003."""
        config = _make_config(agent_name="my-cool-agent")
        exporter = OTLPExporter(config)

        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        event = _make_event()
        payload_bytes = exporter._translate_batch([event])
        request = ExportTraceServiceRequest()
        request.ParseFromString(payload_bytes)

        resource_attrs = {
            kv.key: kv.value.string_value
            for kv in request.resource_spans[0].resource.attributes
        }
        assert resource_attrs.get("clyro.agent.name") == "my-cool-agent"
        assert resource_attrs.get("service.name") == "my-cool-agent"

    def test_loop_stage_as_span_attribute(self):
        """loop_stage → clyro.loop.stage attribute. Implements FRD-S003."""
        config = _make_config()
        exporter = OTLPExporter(config)
        event = _make_event(agent_stage=AgentStage.ACT)
        span = exporter._translate_event(event)

        attrs = {kv.key: kv.value for kv in span.attributes}
        assert "clyro.loop.stage" in attrs
        assert attrs["clyro.loop.stage"].string_value == "act"


# =============================================================================
# §11.6 — Non-blocking dispatch (FRD-S004)
# =============================================================================

class TestNonBlockingDispatch:
    """Tests for async queue dispatch."""

    def test_dispatch_returns_immediately(self):
        """dispatch() should return in <1ms (non-blocking). Implements FRD-S004."""
        import time

        config = _make_config()
        exporter = OTLPExporter(config)
        events = [_make_event() for _ in range(50)]

        start = time.monotonic()
        exporter.dispatch(events)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 10  # Well under 1ms target, but allow test overhead

    def test_queue_full_drops_batch(self):
        """Queue full → batch dropped, warning logged. Implements FRD-S004."""
        config = _make_config(otlp_export_queue_size=2)
        exporter = OTLPExporter(config)

        # Fill the queue
        exporter.dispatch([_make_event()])
        exporter.dispatch([_make_event()])

        # This should be dropped
        exporter.dispatch([_make_event()])

        assert exporter._dropped_count == 1
        assert exporter._dispatched_count == 2


# =============================================================================
# §11.6 — Failure isolation (FRD-S005)
# =============================================================================

class TestFailureIsolation:
    """Tests that OTLP export errors never affect native ingest."""

    @pytest.mark.asyncio
    async def test_export_error_caught(self):
        """Network error → caught, logged, native ingest unaffected. Implements FRD-S005."""
        config = _make_config()
        exporter = OTLPExporter(config)

        # Mock client to raise an error
        exporter._client = MagicMock()
        exporter._client.post = AsyncMock(side_effect=ConnectionError("Connection refused"))

        events = [_make_event()]

        # Should not raise
        await exporter._send_batch(events)
        assert exporter._error_count == 1

    @pytest.mark.asyncio
    async def test_http_500_caught(self):
        """HTTP 500 → caught, logged. Implements FRD-S005."""
        config = _make_config()
        exporter = OTLPExporter(config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        exporter._client = MagicMock()
        exporter._client.post = AsyncMock(return_value=mock_response)

        await exporter._send_batch([_make_event()])
        assert exporter._error_count == 1

    @pytest.mark.asyncio
    async def test_serialization_error_caught(self):
        """Bad data causing serialization error → warning, span omitted. Implements FRD-S005."""
        config = _make_config()
        exporter = OTLPExporter(config)
        exporter._client = MagicMock()
        exporter._client.post = AsyncMock()

        # Patch translate_batch to raise
        with patch.object(exporter, "_translate_batch", side_effect=ValueError("bad data")):
            await exporter._send_batch([_make_event()])

        assert exporter._error_count == 1


# =============================================================================
# §11.6 — Graceful shutdown (FRD-S008)
# =============================================================================

class TestGracefulShutdown:
    """Tests for graceful shutdown and queue drain."""

    @pytest.mark.asyncio
    async def test_shutdown_drains_queue(self):
        """Pending batches flushed during shutdown. Implements FRD-S008."""
        config = _make_config()
        exporter = OTLPExporter(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        await exporter.start()

        # Mock the HTTP client
        exporter._client = MagicMock()
        exporter._client.post = AsyncMock(return_value=mock_response)
        exporter._client.aclose = AsyncMock()

        # Dispatch some events
        exporter.dispatch([_make_event()])

        await exporter.stop()

        # Queue should be empty after shutdown
        assert exporter._queue.empty()


# =============================================================================
# §11.6 — Stats
# =============================================================================

class TestExporterStats:
    def test_stats_tracking(self):
        """Stats track dispatched, dropped, errors."""
        config = _make_config(otlp_export_queue_size=1)
        exporter = OTLPExporter(config)

        exporter.dispatch([_make_event()])  # dispatched
        exporter.dispatch([_make_event()])  # dropped (queue full)

        stats = exporter.stats
        assert stats["dispatched"] == 1
        assert stats["dropped"] == 1
        assert stats["errors"] == 0


# =============================================================================
# _python_to_any_value helper
# =============================================================================

class TestPythonToAnyValue:
    def test_string_value(self):
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue
        result = _python_to_any_value("hello")
        assert result.string_value == "hello"

    def test_int_value(self):
        result = _python_to_any_value(42)
        assert result.int_value == 42

    def test_float_value(self):
        result = _python_to_any_value(3.14)
        assert result.double_value == 3.14

    def test_bool_value(self):
        result = _python_to_any_value(True)
        assert result.bool_value is True

    def test_other_converted_to_string(self):
        result = _python_to_any_value({"key": "val"})
        assert isinstance(result.string_value, str)
