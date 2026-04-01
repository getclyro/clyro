"""
End-to-end tests — TDD §11.3.

These tests use a real child process (a simple echo server)
to verify the wrapper's stdio piping and governance end-to-end.

Note: Tests requiring external MCP servers (like @modelcontextprotocol/
server-filesystem) are skipped if unavailable (CI-friendly).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from clyro.config import WrapperConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.transport import StdioTransport

# A minimal echo MCP server for testing
ECHO_SERVER_SCRIPT = '''\
import json, sys
while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        msg = json.loads(line)
        result = {"content": [{"type": "text", "text": "echo"}]}
        resp = {"jsonrpc": "2.0", "id": msg.get("id"), "result": result}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    except Exception:
        sys.stdout.write(line)
        sys.stdout.flush()
'''


@pytest.fixture
def echo_server_path(tmp_path: Path) -> str:
    """Write the echo server script to a temp file."""
    script = tmp_path / "echo_server.py"
    script.write_text(ECHO_SERVER_SCRIPT)
    return str(script)


class TestStdioTransport:
    """TDD §11.3 — verify transport can spawn and communicate with a child."""

    @pytest.mark.asyncio
    async def test_spawn_and_communicate(self, echo_server_path: str) -> None:
        """Spawn echo server, send a message, read the response."""
        transport = StdioTransport([sys.executable, echo_server_path])
        proc = await transport.start()
        assert proc is not None

        # Send a tools/call
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"text": "hello"}},
        }
        await transport.write_to_child(json.dumps(msg).encode() + b"\n")

        # Read response
        line = await transport.read_line_from_child()
        assert line is not None
        resp = json.loads(line)
        assert resp["id"] == 1
        assert resp["result"]["content"][0]["text"] == "echo"

        # Terminate
        exit_code = await transport.terminate()
        # Process may have already exited due to stdin closing
        assert exit_code is not None

    @pytest.mark.asyncio
    async def test_server_not_found_exits(self) -> None:
        """TDD §11.3 — non-existent command -> exit(1)."""
        transport = StdioTransport(["nonexistent_binary_xyz"])
        with pytest.raises(SystemExit) as exc_info:
            await transport.start()
        assert exc_info.value.code == 1


class TestE2ELoopDetection:
    """TDD §11.3 #3 — loop detection with echo server."""

    @pytest.mark.asyncio
    async def test_loop_blocked(self, echo_server_path: str, tmp_path: Path) -> None:
        """Send 3 identical calls with threshold:3 -> 3rd blocked."""
        from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig.model_validate(
            {
                "global": {"loop_detection": {"threshold": 3, "window": 10}},
                "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()

        results = []
        for _ in range(3):
            results.append(ps.evaluate("echo", {"text": "same"}, s))

        assert isinstance(results[0], AllowDecision)
        assert isinstance(results[1], AllowDecision)
        assert isinstance(results[2], BlockDecision)
        assert results[2].block_type == "loop_detected"


class TestE2EPolicyEnforcement:
    """TDD §11.3 #4 — policy enforcement with real config."""

    def test_contains_blocks_drop(self) -> None:
        cfg = WrapperConfig.model_validate(
            {
                "tools": {
                    "query_database": {
                        "policies": [
                            {"parameter": "sql", "operator": "contains", "value": "DROP"}
                        ]
                    }
                }
            }
        )
        from clyro.mcp.prevention import BlockDecision, PreventionStack
        from clyro.mcp.session import McpSession

        ps = PreventionStack(cfg)
        s = McpSession()

        result = ps.evaluate("query_database", {"sql": "DROP TABLE users"}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "policy_violation"


class TestE2EAuditLogWritten:
    """TDD §11.3 #5 — audit JSONL file created with entries."""

    def test_audit_file_written(self, tmp_path: Path) -> None:
        from clyro.mcp.prevention import PreventionStack
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig.model_validate(
            {"audit": {"log_path": str(tmp_path / "audit.jsonl")}}
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        audit = AuditLogger(cfg.audit, s.session_id)

        audit.log_lifecycle("session_start")
        ps.evaluate("t", {"x": 1}, s)
        audit.log_tool_call("t", {"x": 1}, "allowed", 1, 0.0)
        audit.log_lifecycle("session_end")
        audit.close()

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        events = [json.loads(line)["event"] for line in lines]
        assert events == ["session_start", "tool_call", "session_end"]


class TestE2ENoConfigPermissive:
    """TDD §11.3 #7 — no config file -> permissive mode with defaults."""

    def test_permissive_mode(self) -> None:
        from clyro.config import load_config
        from clyro.mcp.prevention import AllowDecision, PreventionStack
        from clyro.mcp.session import McpSession

        cfg = load_config("/nonexistent/path.yaml")
        ps = PreventionStack(cfg)
        s = McpSession()

        result = ps.evaluate("any_tool", {"any": "params"}, s)
        assert isinstance(result, AllowDecision)
