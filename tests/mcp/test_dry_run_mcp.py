# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""MCP-wrapper dry-run tests (A10) — FRD-004/008/017/018.

Verifies that in dry_run the MCP audit path emits exactly one distinct
``would_block`` event, emits NO enforced sibling (no ``policy_check(block)``,
no ``error``/``error_type='policy_violation'``), never fires the violation
reporter, and stamps the FRD-018 session marker on every event.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from clyro.backend.trace_event_factory import TraceEventFactory
from clyro.config import AuditConfig
from clyro.mcp.audit import AuditLogger


def _mcp_session():
    s = MagicMock()
    s.session_id = uuid4()
    s.agent_id = uuid4()
    s.step_count = 1
    s.accumulated_cost_usd = 0.0
    s.agent_name = "agent"
    return s


def test_factory_would_block_shape():
    f = TraceEventFactory(session=_mcp_session(), dry_run=True)
    wb = f.would_block("refund", {"amt": 500}, "policy_violation", {"policy_id": "p1"})
    assert wb["event_type"] == "would_block"
    assert wb["error_type"] is None  # FRD-017
    assert wb["metadata"]["dry_run"] is True  # FRD-018
    assert wb["metadata"]["would_block"]["check_type"] == "policy"
    assert wb["metadata"]["would_block"]["rule_id"] == "p1"


def test_factory_maps_exec_block_types():
    f = TraceEventFactory(session=_mcp_session(), dry_run=True)
    assert (
        f.would_block("t", None, "budget_exceeded")["metadata"]["would_block"]["check_type"]
        == "cost"
    )
    assert (
        f.would_block("t", None, "step_limit_exceeded")["metadata"]["would_block"]["check_type"]
        == "step"
    )
    assert (
        f.would_block("t", None, "loop_detected")["metadata"]["would_block"]["check_type"] == "loop"
    )


def test_factory_stamps_dry_run_on_all_events():
    f = TraceEventFactory(session=_mcp_session(), dry_run=True)
    assert f.tool_call_act("t", {}, 1)["metadata"]["dry_run"] is True
    assert f.policy_check("t", {}, decision="allow")["metadata"]["dry_run"] is True
    # non-dry-run factory: no marker
    f2 = TraceEventFactory(session=_mcp_session())
    assert "dry_run" not in f2.tool_call_act("t", {}, 1)["metadata"]


def test_audit_would_block_emits_no_enforced_sibling(tmp_path):
    sess = _mcp_session()
    audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), sess.session_id)
    sync = MagicMock()
    reporter = MagicMock()
    audit.set_backend(sync, TraceEventFactory(session=sess, dry_run=True))
    audit.set_violation_reporter(reporter, str(sess.agent_id))

    audit.log_tool_call(
        tool_name="refund",
        parameters={"amt": 500},
        decision="would_block",
        step_number=1,
        accumulated_cost_usd=0.0,
        block_reason="policy_violation",
        block_details={"policy_id": "p1"},
        request_id="req-1",
    )

    enqueued = [c.args[0] for c in sync.enqueue.call_args_list]
    types = [e["event_type"] for e in enqueued]
    # exactly one would_block; the tool ran so the act event is present too
    assert types.count("would_block") == 1
    assert "tool_call" in types
    # FRD-017: NO enforced sibling
    assert "error" not in types
    assert not any(
        e["event_type"] == "policy_check" and e["metadata"].get("decision") == "block"
        for e in enqueued
    )
    assert not any(e.get("error_type") == "policy_violation" for e in enqueued)
    # FRD-010: violation reporter never fires for a would-block
    reporter.assert_not_called()


class TestWouldBlockLatch:
    """FRD-022 on MCP: the prevention stack re-evaluates EVERY tools/call, so a
    tripped limit (sticky) or a repeatedly-violated rule would emit one marker
    per call — unbounded, burning the org's trace quota."""

    def _router(self, dry_run=True):
        from clyro.config import WrapperConfig
        from clyro.mcp.router import MessageRouter

        sess = _mcp_session()
        return MessageRouter(
            config=WrapperConfig(default_action="allow"),
            session=sess,
            transport=MagicMock(),
            prevention=MagicMock(),
            audit=MagicMock(),
            dry_run=dry_run,
        )

    def _decision(self, block_type, details=None):
        d = MagicMock()
        d.block_type = block_type
        d.details = details or {}
        return d

    def test_sticky_checks_key_once_per_session(self):
        # step/cost trip on every subsequent call → one marker per session.
        r = self._router()
        for check in ("step", "cost"):
            keys = {
                r._would_block_key(check, None, f"tool_{i}", self._decision("x")) for i in range(50)
            }
            assert len(keys) == 1, f"{check} must collapse to one key"

    def test_loop_keyed_per_signature(self):
        r = self._router()
        a = r._would_block_key(
            "loop", None, "t", self._decision("loop_detected", {"pattern_hash": "aaa"})
        )
        b = r._would_block_key(
            "loop", None, "t", self._decision("loop_detected", {"pattern_hash": "bbb"})
        )
        assert a != b  # distinct loops each record
        assert a == r._would_block_key(
            "loop", None, "t", self._decision("loop_detected", {"pattern_hash": "aaa"})
        )

    def test_policy_keyed_per_rule_and_tool(self):
        r = self._router()
        k1 = r._would_block_key("policy", "p1", "refund", self._decision("policy_violation"))
        k2 = r._would_block_key("policy", "p1", "refund", self._decision("policy_violation"))
        k3 = r._would_block_key("policy", "p2", "refund", self._decision("policy_violation"))
        k4 = r._would_block_key("policy", "p1", "delete", self._decision("policy_violation"))
        assert k1 == k2  # same rule+tool → one marker
        assert len({k1, k3, k4}) == 3  # distinct rule / tool → own markers

    def test_latch_emits_once_across_many_calls(self):
        r = self._router()
        key = r._would_block_key("policy", "p1", "refund", self._decision("policy_violation"))
        firsts = [r._wb_latch.record(key) for _ in range(500)]
        assert firsts.count(True) == 1  # 500 blocked calls → 1 marker
        assert r._wb_latch.occurrences(key) == 500  # frequency still known


def test_audit_repeat_emits_no_further_marker(tmp_path):
    # FRD-022: emit_marker=False suppresses the backend marker but the tool still
    # ran, so the act event is still enqueued (and the JSONL audit still written).
    sess = _mcp_session()
    audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), sess.session_id)
    sync = MagicMock()
    audit.set_backend(sync, TraceEventFactory(session=sess, dry_run=True))
    audit.log_tool_call(
        tool_name="refund",
        parameters={},
        decision="would_block",
        step_number=2,
        accumulated_cost_usd=0.0,
        block_reason="policy_violation",
        block_details={"policy_id": "p1"},
        request_id="r2",
        emit_marker=False,
    )
    types = [c.args[0]["event_type"] for c in sync.enqueue.call_args_list]
    assert "would_block" not in types  # repeat → no further marker
    assert "tool_call" in types  # the tool ran, act event still recorded


def test_audit_enforce_block_still_emits_sibling(tmp_path):
    # Regression: enforce-mode block still writes the policy_check(block) + error.
    sess = _mcp_session()
    audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), sess.session_id)
    sync = MagicMock()
    audit.set_backend(sync, TraceEventFactory(session=sess))
    audit.log_tool_call(
        tool_name="refund",
        parameters={"amt": 500},
        decision="blocked",
        step_number=1,
        accumulated_cost_usd=0.0,
        block_reason="policy_violation",
        block_details={"policy_id": "p1"},
    )
    types = [c.args[0]["event_type"] for c in sync.enqueue.call_args_list]
    assert "error" in types  # enforce path unchanged
    assert "would_block" not in types


@pytest.mark.asyncio
async def test_would_block_recorded_before_send_failure(tmp_path):
    """A10 ordering regression (FRD-020): the would_block marker + audit must be
    written BEFORE the downstream send — mirroring the allow branch — so a
    mid-exchange send failure cannot drop the would_block and leave an orphan
    unresolved response for a call that was never recorded. Before the fix the
    dry-run branch sent first, so a failing send() dropped the would_block
    entirely (the exact finding dry-run exists to capture)."""
    from clyro.config import WrapperConfig
    from clyro.mcp.prevention import PreventionStack
    from clyro.mcp.router import MessageRouter
    from clyro.mcp.server_transport import TransportError
    from clyro.mcp.session import McpSession

    class _FailingTransport:
        # Minimal ServerTransport whose send() always fails mid-exchange.
        async def send(self, raw):
            raise TransportError("connection reset mid-exchange")

    config = WrapperConfig(default_action="block")  # no rule matches => block
    session = McpSession()
    audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), session.session_id)
    sync = MagicMock()
    audit.set_backend(sync, TraceEventFactory(session=session, dry_run=True))
    router = MessageRouter(
        config, session, _FailingTransport(), PreventionStack(config), audit, dry_run=True
    )

    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "delete_everything", "arguments": {}},
        }
    ).encode()

    with pytest.raises(TransportError):
        await router._handle_host_message(raw)

    types = [c.args[0]["event_type"] for c in sync.enqueue.call_args_list]
    assert "would_block" in types, (
        "would_block must be audited before the send, so a mid-exchange send "
        "failure cannot drop the dry-run finding"
    )


class TestMcpAbsoluteCeiling:
    """FRD-021 parity on MCP: a hard, non-suppressable ceiling that blocks even in
    dry_run — the one thing dry_run cannot forward/allow, matching the SDK."""

    def test_prevention_returns_absolute_step_block(self):
        from clyro.config import GlobalConfig, WrapperConfig
        from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig(default_action="allow", global_=GlobalConfig(absolute_max_steps=2))
        stack = PreventionStack(cfg)
        sess = McpSession()
        assert isinstance(stack.evaluate("t", {}, sess), AllowDecision)  # step 1
        assert isinstance(stack.evaluate("t", {}, sess), AllowDecision)  # step 2
        d = stack.evaluate("t", {}, sess)  # step 3 > absolute_max_steps=2
        assert isinstance(d, BlockDecision)
        assert d.absolute is True
        assert d.block_type == "absolute_step_ceiling"

    def test_prevention_returns_absolute_cost_block(self):
        from clyro.config import GlobalConfig, WrapperConfig
        from clyro.mcp.prevention import BlockDecision, PreventionStack
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig(default_action="allow", global_=GlobalConfig(absolute_max_cost_usd=1.0))
        stack = PreventionStack(cfg)
        sess = McpSession()
        sess.add_cost(5.0)  # accumulated cost over the ceiling
        d = stack.evaluate("t", {}, sess)
        assert isinstance(d, BlockDecision)
        assert d.absolute is True
        assert d.block_type == "absolute_cost_ceiling"

    def test_normal_use_does_not_trip_ceiling(self):
        # Default ceiling (1M / $100k) never fires in ordinary use.
        from clyro.config import WrapperConfig
        from clyro.mcp.prevention import AllowDecision, PreventionStack
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig(default_action="allow")
        stack = PreventionStack(cfg)
        sess = McpSession()
        # Vary the call each iteration so loop detection doesn't fire — we're
        # isolating the ceiling, which must stay silent under the huge default.
        for i in range(20):
            assert isinstance(stack.evaluate(f"tool_{i}", {"i": i}, sess), AllowDecision)

    @pytest.mark.asyncio
    async def test_router_hard_blocks_ceiling_even_in_dry_run(self, tmp_path):
        # In dry_run an absolute-ceiling block must NOT be forwarded — the host gets
        # a real JSON-RPC error and the server never sees the call.
        from clyro.config import GlobalConfig, WrapperConfig
        from clyro.mcp.prevention import PreventionStack
        from clyro.mcp.router import MessageRouter
        from clyro.mcp.session import McpSession

        cfg = WrapperConfig(default_action="allow", global_=GlobalConfig(absolute_max_steps=1))
        session = McpSession()
        audit = AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), session.session_id)
        sent = []

        class _T:
            async def send(self, raw):
                sent.append(raw)

        router = MessageRouter(cfg, session, _T(), PreventionStack(cfg), audit, dry_run=True)

        def call(i):
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {"name": "t", "arguments": {}},
                }
            ).encode()

        await router._handle_host_message(call(1))  # step 1: allowed & forwarded
        await router._handle_host_message(
            call(2)
        )  # step 2 > 1: ceiling → hard block, not forwarded
        assert len(sent) == 1, "the absolute-ceiling call must NOT be forwarded, even in dry_run"
