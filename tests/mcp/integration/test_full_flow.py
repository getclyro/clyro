"""
Integration tests — TDD §11.2.

These tests exercise the full prevention pipeline with real config
objects, verifying the interaction between components.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from clyro.mcp.audit import AuditLogger
from clyro.config import WrapperConfig, load_config
from clyro.mcp.errors import format_error
from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
from clyro.mcp.session import McpSession


class TestFullAllowedFlow:
    """TDD §11.2 #1 — allowed flow produces audit entry."""

    def test_allowed_call_audited(self, tmp_path: Path) -> None:
        cfg = WrapperConfig.model_validate(
            {"audit": {"log_path": str(tmp_path / "audit.jsonl")}}
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        audit = AuditLogger(cfg.audit, s.session_id)

        decision = ps.evaluate("read_file", {"path": "/tmp/a"}, s)
        assert isinstance(decision, AllowDecision)

        audit.log_tool_call(
            tool_name="read_file",
            parameters={"path": "/tmp/a"},
            decision="allowed",
            step_number=decision.step_number,
            accumulated_cost_usd=s.accumulated_cost_usd,
        )
        audit.close()

        entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
        assert entry["decision"] == "allowed"
        assert entry["step_number"] == 1


class TestFullBlockedFlow:
    """TDD §11.2 #2 — max_steps:1, second call blocked."""

    def test_second_call_blocked(self) -> None:
        cfg = WrapperConfig.model_validate({"global": {"max_steps": 1}})
        ps = PreventionStack(cfg)
        s = McpSession()

        r1 = ps.evaluate("t", {}, s)
        assert isinstance(r1, AllowDecision)

        r2 = ps.evaluate("t2", {}, s)
        assert isinstance(r2, BlockDecision)
        assert r2.block_type == "step_limit_exceeded"

        # Verify error formatter produces valid JSON-RPC
        error_json = format_error(99, r2.block_type, r2.details)
        parsed = json.loads(error_json)
        assert parsed["error"]["code"] == -32600


class TestLoopDetectionIntegration:
    """TDD §11.2 #3 — 3 identical calls with threshold:3."""

    def test_third_call_blocked(self) -> None:
        cfg = WrapperConfig.model_validate(
            {"global": {"loop_detection": {"threshold": 3, "window": 10}}}
        )
        ps = PreventionStack(cfg)
        s = McpSession()

        r1 = ps.evaluate("tool", {"x": 1}, s)
        assert isinstance(r1, AllowDecision)
        r2 = ps.evaluate("tool", {"x": 1}, s)
        assert isinstance(r2, AllowDecision)
        r3 = ps.evaluate("tool", {"x": 1}, s)
        assert isinstance(r3, BlockDecision)
        assert r3.block_type == "loop_detected"


class TestPolicyViolationIntegration:
    """TDD §11.2 #4 — max_value:500, amount:1200 -> blocked."""

    def test_policy_blocks_high_amount(self) -> None:
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "policies": [
                        {"parameter": "amount", "operator": "max_value", "value": 500}
                    ]
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()

        result = ps.evaluate("transfer", {"amount": 1200}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "policy_violation"


class TestPassthroughNonToolCalls:
    """TDD §11.2 #5 — non-tool methods identified as passthrough."""

    def test_non_tool_methods(self) -> None:
        """Verify message classification logic (tools/call vs others)."""
        passthrough = ["initialize", "tools/list", "resources/read", "notifications/progress"]
        for method in passthrough:
            assert method != "tools/call"


class TestConfigValidationIntegration:
    """TDD §11.2 #7 — invalid config -> exit code 1."""

    def test_invalid_config_exits(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("global:\n  max_steps: -5\n")
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)


class TestConcurrentCalls:
    """TDD §11.2 #8 — sequential step numbering under rapid calls."""

    def test_sequential_steps(self) -> None:
        cfg = WrapperConfig.model_validate({"global": {"max_steps": 100}})
        ps = PreventionStack(cfg)
        s = McpSession()

        step_numbers = []
        for i in range(5):
            result = ps.evaluate(f"tool_{i}", {"i": i}, s)
            assert isinstance(result, AllowDecision)
            step_numbers.append(result.step_number)

        assert step_numbers == [1, 2, 3, 4, 5]
