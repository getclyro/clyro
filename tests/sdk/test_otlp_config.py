# Tests for SDK OTLP Configuration (C8)
# Implements TDD §11.7

"""
Unit tests for OTLP export configuration validation.

Tests endpoint validation, compression validation, header filtering,
and default-off behavior.
"""

from __future__ import annotations

import pytest

from clyro.config import ClyroConfig
from clyro.exceptions import ClyroConfigError

# =============================================================================
# §11.7 — Valid HTTPS endpoint (FRD-S001)
# =============================================================================

class TestValidHTTPSEndpoint:
    def test_valid_https_endpoint_accepted(self):
        config = ClyroConfig(otlp_export_endpoint="https://otel.example.com/v1/traces")
        assert config.otlp_export_endpoint == "https://otel.example.com/v1/traces"

    def test_trailing_slash_stripped(self):
        config = ClyroConfig(otlp_export_endpoint="https://otel.example.com/")
        assert config.otlp_export_endpoint == "https://otel.example.com"


# =============================================================================
# §11.7 — Valid localhost HTTP (FRD-S007)
# =============================================================================

class TestLocalhostHTTP:
    def test_localhost_http_accepted(self):
        config = ClyroConfig(otlp_export_endpoint="http://localhost:4318/v1/traces")
        assert config.otlp_export_endpoint == "http://localhost:4318/v1/traces"

    def test_127_0_0_1_http_accepted(self):
        config = ClyroConfig(otlp_export_endpoint="http://127.0.0.1:4318")
        assert config.otlp_export_endpoint == "http://127.0.0.1:4318"

    def test_ipv6_localhost_http_accepted(self):
        config = ClyroConfig(otlp_export_endpoint="http://[::1]:4318")
        assert config.otlp_export_endpoint == "http://[::1]:4318"


# =============================================================================
# §11.7 — Invalid HTTP (non-localhost) (FRD-S001)
# =============================================================================

class TestInvalidHTTPEndpoint:
    def test_http_non_localhost_rejected(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(otlp_export_endpoint="http://external.example.com/v1/traces")


# =============================================================================
# §11.7 — No endpoint → disabled (FRD-S006)
# =============================================================================

class TestDefaultOff:
    def test_none_endpoint_disabled(self):
        config = ClyroConfig(otlp_export_endpoint=None)
        assert config.otlp_export_endpoint is None

    def test_missing_endpoint_disabled(self):
        config = ClyroConfig()
        assert config.otlp_export_endpoint is None

    def test_empty_string_endpoint_disabled(self):
        config = ClyroConfig(otlp_export_endpoint="")
        assert config.otlp_export_endpoint is None


# =============================================================================
# §11.7 — Invalid timeout (FRD-S007)
# =============================================================================

class TestInvalidTimeout:
    def test_zero_timeout_rejected(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(
                otlp_export_endpoint="https://otel.example.com",
                otlp_export_timeout_ms=0,
            )

    def test_negative_timeout_rejected(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(
                otlp_export_endpoint="https://otel.example.com",
                otlp_export_timeout_ms=-1,
            )


# =============================================================================
# §11.7 — Invalid compression (FRD-S007)
# =============================================================================

class TestInvalidCompression:
    def test_unknown_compression_rejected(self):
        with pytest.raises(ClyroConfigError):
            ClyroConfig(
                otlp_export_endpoint="https://otel.example.com",
                otlp_export_compression="snappy",
            )

    def test_gzip_accepted(self):
        config = ClyroConfig(
            otlp_export_endpoint="https://otel.example.com",
            otlp_export_compression="gzip",
        )
        assert config.otlp_export_compression == "gzip"

    def test_none_compression_accepted(self):
        config = ClyroConfig(
            otlp_export_endpoint="https://otel.example.com",
            otlp_export_compression="none",
        )
        assert config.otlp_export_compression == "none"


# =============================================================================
# §11.7 — Reserved header filtering (FRD-S007)
# =============================================================================

class TestReservedHeaders:
    def test_reserved_header_ignored(self):
        """Content-Type header silently ignored with warning."""
        config = ClyroConfig(
            otlp_export_endpoint="https://otel.example.com",
            otlp_export_headers={
                "Content-Type": "text/plain",
                "Authorization": "Bearer token123",
            },
        )
        assert "Content-Type" not in config.otlp_export_headers
        assert "Authorization" in config.otlp_export_headers

    def test_content_encoding_ignored(self):
        config = ClyroConfig(
            otlp_export_endpoint="https://otel.example.com",
            otlp_export_headers={
                "Content-Encoding": "br",
                "X-Custom": "value",
            },
        )
        assert "Content-Encoding" not in config.otlp_export_headers
        assert "X-Custom" in config.otlp_export_headers


# =============================================================================
# Default values
# =============================================================================

class TestDefaultValues:
    def test_default_timeout(self):
        config = ClyroConfig()
        assert config.otlp_export_timeout_ms == 5000

    def test_default_queue_size(self):
        config = ClyroConfig()
        assert config.otlp_export_queue_size == 100

    def test_default_compression(self):
        config = ClyroConfig()
        assert config.otlp_export_compression == "gzip"

    def test_default_headers_empty(self):
        config = ClyroConfig()
        assert config.otlp_export_headers == {}
