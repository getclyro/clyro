"""
Unit tests for CLI async entry point and backend initialization.

Covers:
- _derive_instance_id determinism
- _derive_agent_name precedence
- _async_main config loading
- _async_main backend init failure (fail-graceful)
- _init_backend component wiring
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clyro.mcp.cli import _derive_agent_name, _derive_instance_id


class TestDeriveInstanceId:
    """Instance ID derivation from agent name."""

    def test_deterministic(self) -> None:
        """Same agent name always produces same instance_id."""
        id1 = _derive_instance_id("npx server-fs /path")
        id2 = _derive_instance_id("npx server-fs /path")
        assert id1 == id2

    def test_different_names_different_ids(self) -> None:
        id1 = _derive_instance_id("server-a")
        id2 = _derive_instance_id("server-b")
        assert id1 != id2

    def test_length_is_12(self) -> None:
        result = _derive_instance_id("test-agent")
        assert len(result) == 12

    def test_uses_sha256(self) -> None:
        name = "test-agent"
        expected = hashlib.sha256(name.encode()).hexdigest()[:12]
        assert _derive_instance_id(name) == expected


class TestDeriveAgentName:
    """Agent name derivation from config or server command."""

    def test_config_name_takes_precedence(self) -> None:
        result = _derive_agent_name("my-agent", ["npx", "server"])
        assert result == "my-agent"

    def test_falls_back_to_server_command(self) -> None:
        result = _derive_agent_name(None, ["npx", "server-fs", "/path"])
        assert result == "npx server-fs /path"

    def test_empty_server_command_uses_default(self) -> None:
        result = _derive_agent_name(None, [])
        assert result == "mcp-agent"

    def test_none_config_with_command(self) -> None:
        result = _derive_agent_name(None, ["echo", "hello"])
        assert result == "echo hello"


class TestAsyncMainBackendFailGraceful:
    """_async_main handles backend init failure gracefully."""

    @pytest.mark.asyncio
    async def test_backend_init_failure_continues_without_sync(self) -> None:
        """If _init_backend raises, wrapper should continue without sync."""
        from clyro.mcp.cli import _async_main

        mock_config = MagicMock()
        mock_config.is_backend_enabled = True
        mock_config.audit = MagicMock()
        mock_config.audit.log_path = "/tmp/test-audit.jsonl"
        mock_config.audit.redact_parameters = []
        mock_config.global_ = MagicMock()

        with patch("clyro.mcp.cli.load_config", return_value=mock_config), \
             patch("clyro.mcp.cli._init_backend", side_effect=RuntimeError("backend down")), \
             patch("clyro.mcp.cli.StdioTransport") as mock_transport_cls, \
             patch("clyro.mcp.cli.PreventionStack"), \
             patch("clyro.mcp.cli.AuditLogger") as mock_audit_cls, \
             patch("clyro.mcp.cli.MessageRouter") as mock_router_cls:

            mock_transport = AsyncMock()
            mock_transport.start = AsyncMock()
            mock_transport.terminate = AsyncMock(return_value=0)
            mock_transport_cls.return_value = mock_transport

            mock_audit = MagicMock()
            mock_audit_cls.return_value = mock_audit

            mock_router = AsyncMock()
            mock_router.run = AsyncMock(return_value=0)
            mock_router_cls.return_value = mock_router

            # Should NOT raise — backend failure is graceful
            code = await _async_main(["echo", "hello"], None)
            assert code == 0

            # Audit should NOT have backend attached
            mock_audit.set_backend.assert_not_called()
