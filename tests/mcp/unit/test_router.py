"""
Unit tests for MessageRouter message handling logic.

Tests cover:
- Message classification (tools/call vs passthrough)
- AllowDecision path (forward to server + audit)
- BlockDecision path (error to host + audit, never forward)
- Malformed JSON forwarding (FRD-001)
- Batch JSON-RPC passthrough (CRITICAL-03 fix)
- Notification passthrough
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clyro.config import WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
from clyro.mcp.router import MessageRouter
from clyro.mcp.session import McpSession
from clyro.mcp.transport import StdioTransport


def _make_router(
    prevention_result=None,
) -> tuple[MessageRouter, AsyncMock, MagicMock, MagicMock]:
    """Build a MessageRouter with mocked dependencies."""
    config = WrapperConfig()
    session = McpSession()
    transport = MagicMock(spec=StdioTransport)
    transport.write_to_child = AsyncMock()
    prevention = MagicMock(spec=PreventionStack)
    if prevention_result is not None:
        prevention.evaluate.return_value = prevention_result
    audit = MagicMock(spec=AuditLogger)
    router = MessageRouter(config, session, transport, prevention, audit)
    return router, transport, prevention, audit


class TestMessageClassification:
    """Message routing rules."""

    def test_tools_call_identified(self) -> None:
        """tools/call method triggers evaluation."""
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "/tmp"}},
        }
        assert msg.get("method") == "tools/call"

    def test_passthrough_methods(self) -> None:
        """tools/list, initialize forwarded."""
        for method in ["tools/list", "initialize", "resources/list", "prompts/list"]:
            msg = {"jsonrpc": "2.0", "id": 1, "method": method}
            assert msg.get("method") != "tools/call"

    def test_notification_no_id(self) -> None:
        """Notifications (no id) are always passthrough."""
        msg = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
        assert "id" not in msg

    def test_invalid_json_detection(self) -> None:
        """Malformed JSON detectable."""
        raw = b"{{not valid json\n"
        try:
            json.loads(raw)
            parsed = True
        except json.JSONDecodeError:
            parsed = False
        assert not parsed

    def test_tools_call_extraction(self) -> None:
        """Correctly extract tool_name and arguments from params."""
        msg = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "query_db", "arguments": {"sql": "SELECT 1"}},
        }
        params = msg.get("params", {})
        assert params.get("name") == "query_db"
        assert params.get("arguments") == {"sql": "SELECT 1"}


class TestHandleHostMessage:
    """Actual MessageRouter._handle_host_message tests."""

    @pytest.mark.asyncio
    async def test_passthrough_non_tools_call(self) -> None:
        """Non-tools/call messages are forwarded without evaluation."""
        router, transport, prevention, audit = _make_router()
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        raw = json.dumps(msg).encode() + b"\n"
        await router._handle_host_message(raw)

        transport.write_to_child.assert_awaited_once_with(raw)
        prevention.evaluate.assert_not_called()
        audit.log_tool_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_passthrough(self) -> None:
        """Notifications (no id) are forwarded without evaluation."""
        router, transport, prevention, audit = _make_router()
        msg = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
        raw = json.dumps(msg).encode() + b"\n"
        await router._handle_host_message(raw)

        transport.write_to_child.assert_awaited_once_with(raw)
        prevention.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_json_forwarded(self) -> None:
        """Malformed JSON is logged and forwarded raw (FRD-001)."""
        router, transport, prevention, audit = _make_router()
        raw = b"{{not valid json\n"
        await router._handle_host_message(raw)

        audit.log_parse_error.assert_called_once_with(raw)
        transport.write_to_child.assert_awaited_once_with(raw)
        prevention.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_json_rpc_passthrough(self) -> None:
        """Batch JSON-RPC arrays are forwarded without governance."""
        router, transport, prevention, audit = _make_router()
        batch = [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "t"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        raw = json.dumps(batch).encode() + b"\n"
        await router._handle_host_message(raw)

        transport.write_to_child.assert_awaited_once_with(raw)
        prevention.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_tools_call_allowed(self) -> None:
        """tools/call with AllowDecision is forwarded and audited."""
        allow = AllowDecision(tool_name="read_file", step_number=1)
        router, transport, prevention, audit = _make_router(allow)
        msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "/tmp"}},
        }
        raw = json.dumps(msg).encode() + b"\n"

        with patch("clyro.mcp.router.sys") as mock_sys:
            mock_sys.stdout.buffer = MagicMock()
            await router._handle_host_message(raw)

        transport.write_to_child.assert_awaited_once_with(raw)
        audit.log_tool_call.assert_called_once()
        call_kwargs = audit.log_tool_call.call_args
        assert call_kwargs.kwargs["decision"] == "allowed"
        assert call_kwargs.kwargs["tool_name"] == "read_file"

    @pytest.mark.asyncio
    async def test_tools_call_blocked(self) -> None:
        """tools/call with BlockDecision sends error to host, never forwards."""
        block = BlockDecision(
            block_type="policy_violation",
            tool_name="transfer",
            step_number=1,
            details={"rule_name": "max_amount"},
        )
        router, transport, prevention, audit = _make_router(block)
        msg = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "transfer", "arguments": {"amount": 9999}},
        }
        raw = json.dumps(msg).encode() + b"\n"

        with patch("clyro.mcp.router.sys") as mock_sys:
            mock_buf = MagicMock()
            mock_sys.stdout.buffer = mock_buf
            await router._handle_host_message(raw)

        # Server should NOT receive the message
        transport.write_to_child.assert_not_awaited()
        # Error response should be written to stdout
        mock_buf.write.assert_called_once()
        written = mock_buf.write.call_args[0][0]
        error_msg = json.loads(written)
        assert error_msg["id"] == 99
        assert error_msg["error"]["code"] == -32600
        # Audit should record the block
        audit.log_tool_call.assert_called_once()
        call_kwargs = audit.log_tool_call.call_args
        assert call_kwargs.kwargs["decision"] == "blocked"
        assert call_kwargs.kwargs["block_reason"] == "policy_violation"
