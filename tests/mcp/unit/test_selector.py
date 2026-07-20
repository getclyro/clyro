# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for transport selection + the new config fields —
FRD-021, FRD-022, FRD-042.
"""

from __future__ import annotations

import pytest

from clyro.config import ServerConfig, WrapperConfig
from clyro.mcp.selector import SelectionError, select_transport


class TestSelection:
    def test_unset_defaults_to_stdio(self) -> None:
        # FRD-022
        sel = select_transport(transport=None, url=None, server_command=["npx", "srv"])
        assert sel.transport == "stdio"
        assert sel.server_command == ["npx", "srv"]

    def test_http_with_url(self) -> None:
        # FRD-042
        sel = select_transport(
            transport="http", url="https://s.internal/mcp", server_command=[]
        )
        assert sel.transport == "http"
        assert sel.url == "https://s.internal/mcp"
        assert sel.server_command is None

    def test_unknown_transport_refused(self) -> None:
        # FRD-021
        with pytest.raises(SelectionError):
            select_transport(transport="grpc", url=None, server_command=["x"])

    def test_http_without_url_refused(self) -> None:
        # FRD-042 failure
        with pytest.raises(SelectionError):
            select_transport(transport="http", url=None, server_command=[])

    def test_stdio_without_command_refused(self) -> None:
        with pytest.raises(SelectionError):
            select_transport(transport="stdio", url=None, server_command=[])

    def test_case_insensitive(self) -> None:
        assert select_transport(transport="HTTP", url="https://a/", server_command=[]).transport == "http"


class TestConfigDefaults:
    """New config fields default to preserve stdio behaviour."""

    def test_wrapperconfig_defaults_stdio(self) -> None:
        cfg = WrapperConfig(default_action="allow")
        assert cfg.transport == "stdio"  # FRD-022
        assert isinstance(cfg.server, ServerConfig)

    def test_server_config_defaults(self) -> None:
        s = ServerConfig()
        assert s.url is None
        assert s.allow_plaintext is False  # FRD-039 relaxation off by default
        assert s.liveness_secs == 60  # FRD-049 (D15)
        assert s.reconnect.max_attempts == 5  # FRD-056

    def test_invalid_transport_value_rejected_by_model(self) -> None:
        with pytest.raises(Exception):
            WrapperConfig(default_action="allow", transport="ftp")
