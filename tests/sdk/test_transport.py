# Tests for Clyro SDK Transport Layer
# Implements PRD-005, PRD-006

"""Unit tests for the transport layer."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig
from clyro.constants import DEFAULT_API_URL
from clyro.exceptions import TransportError
from clyro.trace import TraceEvent, EventType, create_session_start_event
from clyro.transport import Transport, SyncTransport, DEFAULT_TIMEOUT


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def local_only_config(temp_dir):
    """Create a local-only config (no API key)."""
    return ClyroConfig(
        local_storage_path=str(temp_dir / "traces.db"),
    )


@pytest.fixture
def backend_config(temp_dir):
    """Create a config with backend connection."""
    return ClyroConfig(
        api_key="cly_test_key",
        endpoint="https://api.example.com",
        local_storage_path=str(temp_dir / "traces.db"),
    )


@pytest.fixture
def sample_event():
    """Create a sample trace event."""
    return create_session_start_event(
        session_id=uuid4(),
        agent_id=uuid4(),
    )


class TestTransportInit:
    """Tests for Transport initialization."""

    def test_init_with_config(self, local_only_config):
        """Test transport initialization."""
        transport = Transport(local_only_config)

        assert transport.config == local_only_config
        assert transport.endpoint == DEFAULT_API_URL
        assert transport.is_local_only is True

    def test_init_with_backend_config(self, backend_config):
        """Test transport with backend configuration."""
        transport = Transport(backend_config)

        assert transport.is_local_only is False
        assert transport.endpoint == "https://api.example.com"


class TestTransportLocalOnly:
    """Tests for local-only mode."""

    @pytest.mark.asyncio
    async def test_send_events_local_only(self, local_only_config, sample_event):
        """Test sending events in local-only mode stores locally."""
        transport = Transport(local_only_config)

        result = await transport.send_events([sample_event])

        assert result["local_only"] is True
        assert result["accepted"] == 1
        assert result["rejected"] == 0

    @pytest.mark.asyncio
    async def test_flush_in_local_only_mode(self, local_only_config, sample_event):
        """Test flush in local-only mode doesn't call backend."""
        transport = Transport(local_only_config)

        await transport.buffer_event(sample_event)
        await transport.flush()

        # No errors means success
        sync_status = transport.get_sync_status()
        assert sync_status["transport"]["local_only"] is True


class TestTransportSendEvents:
    """Tests for send_events method."""

    @pytest.mark.asyncio
    async def test_send_empty_events_returns_zero(self, backend_config):
        """Test sending empty event list."""
        transport = Transport(backend_config)

        result = await transport.send_events([])

        assert result["accepted"] == 0
        assert result["rejected"] == 0

    @pytest.mark.asyncio
    async def test_send_events_success(self, backend_config, sample_event):
        """Test successful event sending."""
        transport = Transport(backend_config)

        # Mock the _get_client method on the HttpEventSender, not on Transport
        with patch.object(transport._sender, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"accepted": 1, "rejected": 0}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await transport.send_events([sample_event])

            assert result["accepted"] == 1
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_events_rate_limited(self, backend_config, sample_event):
        """Test handling of rate limiting (429)."""
        transport = Transport(backend_config)

        # Mock the _get_client method on the HttpEventSender, not on Transport
        with patch.object(transport._sender, "_get_client") as mock_get_client:
            mock_client = AsyncMock()

            # First call returns 429, second call succeeds
            mock_response_429 = MagicMock()
            mock_response_429.status_code = 429
            mock_response_429.headers = {"Retry-After": "1"}

            mock_response_ok = MagicMock()
            mock_response_ok.status_code = 200
            mock_response_ok.json.return_value = {"accepted": 1, "rejected": 0}
            mock_response_ok.raise_for_status = MagicMock()

            mock_client.post.side_effect = [mock_response_429, mock_response_ok]
            mock_get_client.return_value = mock_client

            # This should retry after rate limit
            result = await transport.send_events([sample_event])

            assert result["accepted"] == 1


class TestTransportBuffering:
    """Tests for event buffering."""

    @pytest.mark.asyncio
    async def test_buffer_adds_to_list(self, backend_config, sample_event):
        """Test that buffer_event adds event to buffer."""
        transport = Transport(backend_config)

        await transport.buffer_event(sample_event)

        assert len(transport._event_buffer) == 1
        assert transport._event_buffer[0] == sample_event


class TestTransportSyncStatus:
    """Tests for sync status."""

    def test_get_sync_status(self, local_only_config):
        """Test get_sync_status returns expected format."""
        transport = Transport(local_only_config)

        status = transport.get_sync_status()

        assert "transport" in status
        assert "storage" in status
        assert status["transport"]["endpoint"] == DEFAULT_API_URL
        assert status["transport"]["local_only"] is True
        assert status["transport"]["running"] is False
        assert status["transport"]["buffer_size"] == 0


class TestTransportClose:
    """Tests for transport cleanup."""

    @pytest.mark.asyncio
    async def test_close_stops_background_sync(self, local_only_config):
        """Test close stops background sync."""
        transport = Transport(local_only_config)
        await transport.start_background_sync()

        assert transport._running is True
        assert transport._sync_worker.is_running is True

        await transport.close()

        assert transport._running is False
        assert transport._sync_worker.is_running is False


class TestSyncTransport:
    """Tests for synchronous transport wrapper."""

    def test_init_with_config(self, local_only_config):
        """Test SyncTransport initialization."""
        transport = SyncTransport(local_only_config)

        assert transport.config == local_only_config

    def test_flush_sync(self, local_only_config, sample_event):
        """Test synchronous flush."""
        transport = SyncTransport(local_only_config)

        transport.buffer_event(sample_event)
        transport.flush()

        # No errors means success
        status = transport.get_sync_status()
        assert status is not None

    def test_get_sync_status(self, local_only_config):
        """Test get_sync_status on SyncTransport."""
        transport = SyncTransport(local_only_config)

        status = transport.get_sync_status()

        assert "transport" in status
        assert "storage" in status

    def test_close_sync(self, local_only_config):
        """Test synchronous close."""
        transport = SyncTransport(local_only_config)

        # Should not raise
        transport.close()

    @pytest.mark.asyncio
    async def test_sync_transport_in_running_loop_raises(self, local_only_config, sample_event):
        """Test that SyncTransport raises in a running event loop."""
        transport = SyncTransport(local_only_config)

        with pytest.raises(RuntimeError, match="running event loop"):
            transport.buffer_event(sample_event)


class TestDefaultTimeout:
    """Tests for default timeout settings."""

    def test_default_timeout_values(self):
        """Test DEFAULT_TIMEOUT has expected values."""
        assert DEFAULT_TIMEOUT.connect == 5.0
        assert DEFAULT_TIMEOUT.read == 30.0
        assert DEFAULT_TIMEOUT.write == 10.0
        assert DEFAULT_TIMEOUT.pool == 5.0
