# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for global dry-run (monitor) mode — A10.

Covers the SDK surface of the 4×3 conformance matrix (NFR-002) and the
functional criteria C1–C14 that live in the SDK: mode resolution + fail-safe
default (FRD-001/002), record-then-allow for all four checks (FRD-003/008/020),
de-dup (FRD-022), the absolute ceiling (FRD-021), disabled-stays-silent
(FRD-006), require_approval no-hang (FRD-013), the marker shape + FRD-017
invariant, and the FRD-018 session marker.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

import clyro.dry_run as dry_run
import clyro.local_policy as lp
from clyro.config import ClyroConfig, ExecutionControls, WrapperConfig
from clyro.dry_run import DRY_RUN_TOKEN, normalize_enforcement_mode, resolve_dry_run_env
from clyro.exceptions import (
    AbsoluteCeilingExceededError,
    PolicyViolationError,
    StepLimitExceededError,
)
from clyro.local_policy import SDKLocalPolicyEvaluator
from clyro.policy import PolicyDecision, PolicyEvaluator
from clyro.session import Session
from clyro.trace import EventType, create_step_event


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CLYRO_DRY_RUN", raising=False)
    yield


def _step(session):
    return create_step_event(session_id=session.session_id, step_number=0, event_name="s")


def _would_blocks(session):
    return [e for e in session.events if e.event_type == EventType.WOULD_BLOCK]


# ---------------------------------------------------------------------------
# Mode resolution + fail-safe default (FRD-001/002, SC4)
# ---------------------------------------------------------------------------


class TestModeResolution:
    def test_default_is_enforce(self):
        assert ExecutionControls().enforcement_mode == "enforce"
        assert ExecutionControls().is_dry_run is False

    def test_explicit_dry_run(self):
        assert ExecutionControls(enforcement_mode="dry_run").is_dry_run is True

    def test_normalize_only_dry_run_string_enables(self):
        assert normalize_enforcement_mode("dry_run") == "dry_run"
        assert normalize_enforcement_mode("DRY_RUN ") == "dry_run"
        for bad in ("enforce", "nonsense", "", None, 1, "drrun"):
            assert normalize_enforcement_mode(bad) == "enforce"

    def test_env_resolver_truthy_falsy_malformed(self, monkeypatch):
        assert resolve_dry_run_env() is None
        for truthy in ("1", "true", "yes", "on", "dry_run", "TRUE"):
            monkeypatch.setenv("CLYRO_DRY_RUN", truthy)
            assert resolve_dry_run_env() is True
        for other in ("0", "false", "no", "garbage", ""):
            monkeypatch.setenv("CLYRO_DRY_RUN", other)
            assert resolve_dry_run_env() is False, other

    def test_env_override_wins_over_config(self, monkeypatch):
        # SC4/FRD-002: a well-formed override wins over the in-code config.
        monkeypatch.setenv("CLYRO_DRY_RUN", "1")
        cfg = ClyroConfig(controls=ExecutionControls(enforcement_mode="enforce"))
        assert cfg.controls.is_dry_run is True

    def test_malformed_env_never_enables_dry_run(self, monkeypatch):
        # SC4: absent / unknown value => 0 accidental dry-run.
        monkeypatch.setenv("CLYRO_DRY_RUN", "garbage")
        cfg = ClyroConfig(controls=ExecutionControls(enforcement_mode="dry_run"))
        # explicit dry_run config, but malformed env override => fail-safe enforce
        assert cfg.controls.is_dry_run is False

    def test_wrapper_config_resolved_flag(self, monkeypatch):
        wc = WrapperConfig(default_action="allow")
        assert wc.resolved_is_dry_run is False
        monkeypatch.setenv("CLYRO_DRY_RUN", "yes")
        assert wc.resolved_is_dry_run is True


# ---------------------------------------------------------------------------
# SDK session — step / cost / loop record-then-allow (C1, C2, C7, FRD-003/008)
# ---------------------------------------------------------------------------


class TestSessionExecutionControls:
    def test_enforce_step_limit_raises(self):
        # C2: enforce path unchanged.
        s = Session(config=ClyroConfig(controls=ExecutionControls(max_steps=2)))
        with pytest.raises(StepLimitExceededError):
            for _ in range(5):
                s.record_event(_step(s))

    def test_dry_run_step_records_would_block_not_raise(self):
        # C1 + C7: no raise; exactly one would_block marker on the stream.
        s = Session(
            config=ClyroConfig(controls=ExecutionControls(max_steps=2, enforcement_mode="dry_run"))
        )
        for _ in range(5):
            s.record_event(_step(s))  # must not raise
        wb = _would_blocks(s)
        assert len(wb) == 1
        assert wb[0].metadata["would_block"]["check_type"] == "step"

    def test_dry_run_dedup_one_per_reason(self):
        # C12 / FRD-022: crossing the cost limit then running many steps => 1 record.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    max_steps=100000, max_cost_usd=0.0001, enforcement_mode="dry_run"
                )
            )
        )
        for _ in range(200):
            ev = _step(s)
            ev.cost_usd = ev.cost_usd.__class__("0.01")
            s.record_event(ev)
        cost_wb = [e for e in _would_blocks(s) if e.metadata["would_block"]["check_type"] == "cost"]
        assert len(cost_wb) == 1

    def test_loop_would_block_deduped_per_distinct_signature(self):
        # FRD-022: the same loop signature records once; a *different* loop
        # records again (keys must not collapse distinct loops onto one).
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(loop_detection_threshold=3, enforcement_mode="dry_run")
            )
        )
        for _ in range(6):  # state A repeated → 1 loop would_block
            s._check_loop_detection({"phase": "A"}, action="a")
        for _ in range(6):  # distinct state B repeated → a 2nd, distinct one
            s._check_loop_detection({"phase": "B"}, action="b")
        loop_wb = [e for e in _would_blocks(s) if e.metadata["would_block"]["check_type"] == "loop"]
        assert len(loop_wb) == 2

    def test_marker_shape_no_error_type(self):
        # FRD-008/017/018: check_type, would_be_outcome, dry_run marker, NO error_type.
        s = Session(
            config=ClyroConfig(controls=ExecutionControls(max_steps=1, enforcement_mode="dry_run"))
        )
        for _ in range(3):
            s.record_event(_step(s))
        wb = _would_blocks(s)[0]
        assert wb.error_type is None
        assert wb.metadata["dry_run"] is True
        assert wb.metadata["would_block"]["would_be_outcome"] == "block"


class TestAbsoluteCeiling:
    def test_ceiling_fires_even_in_dry_run_with_soft_disabled(self):
        # C11 / FRD-021: hard stop even in dry_run and even with soft limit off.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    max_steps=2,
                    enable_step_limit=False,
                    enforcement_mode="dry_run",
                    absolute_max_steps=10,
                )
            )
        )
        with pytest.raises(AbsoluteCeilingExceededError) as exc:
            for _ in range(50):
                s.record_event(_step(s))
        assert exc.value.dimension == "step"

    def test_ceiling_not_reached_in_normal_use(self):
        s = Session(
            config=ClyroConfig(controls=ExecutionControls(max_steps=3, enforcement_mode="dry_run"))
        )
        # 20 steps is well under the default 100k ceiling → no ceiling raise.
        for _ in range(20):
            s.record_event(_step(s))
        assert _would_blocks(s)  # step would-blocks recorded, no ceiling

    def test_ceiling_fires_via_record_step_in_dry_run(self):
        # FRD-021 regression: record_step bypasses record_event, so it must guard
        # the ceiling itself — otherwise a dry_run loop through the wrapper step
        # path (wrapper.record_step) would never hard-stop.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    enable_step_limit=False,
                    enforcement_mode="dry_run",
                    absolute_max_steps=10,
                )
            )
        )
        with pytest.raises(AbsoluteCeilingExceededError) as exc:
            for _ in range(50):
                s.record_step("s")
        assert exc.value.dimension == "step"

    def test_ceiling_fires_via_record_llm_call_in_dry_run(self):
        # FRD-021 regression: record_llm_call (LLM step/cost path) bypasses
        # record_event and must guard the ceiling itself.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    enable_step_limit=False,
                    enforcement_mode="dry_run",
                    absolute_max_steps=10,
                )
            )
        )
        with pytest.raises(AbsoluteCeilingExceededError) as exc:
            for _ in range(50):
                s.record_llm_call("gpt-4-turbo", {"prompt": "x"})
        assert exc.value.dimension == "step"

    def test_ceiling_fires_via_add_cost_in_dry_run(self):
        # FRD-021 regression: Session.add_cost (public cost-tracking API) bypasses
        # record_event and must guard the cost ceiling itself. NOTE: this is the SDK
        # Session; the MCP wrapper uses a separate McpSession with no ceiling concept.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    enable_cost_limit=False,
                    enforcement_mode="dry_run",
                    absolute_max_cost_usd=1.0,
                )
            )
        )
        with pytest.raises(AbsoluteCeilingExceededError) as exc:
            for _ in range(50):
                s.add_cost(Decimal("0.10"))
        assert exc.value.dimension == "cost"

    def test_ceiling_propagates_through_policy_event_drain(self):
        # FRD-021 regression: record_event raises the ceiling while draining a
        # policy would_block event once cumulative steps cross it. check_policy's
        # drain must PROPAGATE it (hard stop), not swallow it as a fail-open
        # `clyro_policy_event_drain_failed` (the bug seen with a low ceiling).
        from unittest.mock import MagicMock

        from clyro.dry_run import build_would_block_event

        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    enforcement_mode="dry_run", enable_step_limit=False, absolute_max_steps=1
                )
            )
        )
        s._step_number = 5  # already past the ceiling of 1
        evaluator = MagicMock()
        evaluator.evaluate_sync.return_value = None
        evaluator.drain_events.return_value = [
            build_would_block_event(session_id=s.session_id, check_type="policy", action="x")
        ]
        s._policy_evaluator = evaluator

        with pytest.raises(AbsoluteCeilingExceededError) as exc:
            s.check_policy("tool_call", {"x": 1})
        assert exc.value.dimension == "step"


class TestDisabledStaysSilent:
    def test_disabled_check_no_would_block(self):
        # C8 / FRD-006: a disabled check produces no would_block in dry_run.
        s = Session(
            config=ClyroConfig(
                controls=ExecutionControls(
                    max_steps=1, enable_step_limit=False, enforcement_mode="dry_run"
                )
            )
        )
        for _ in range(10):
            s.record_event(_step(s))
        assert _would_blocks(s) == []


# ---------------------------------------------------------------------------
# Policy — cloud + local record-then-allow, approval no-hang (C1, FRD-013)
# ---------------------------------------------------------------------------


def _cloud_evaluator(mode):
    cfg = ClyroConfig(
        api_key="cly_test",
        endpoint="http://x",
        controls=ExecutionControls(enable_policy_enforcement=True, enforcement_mode=mode),
    )
    return PolicyEvaluator(config=cfg, agent_id=uuid4())


class TestCloudPolicy:
    def test_enforce_block_raises(self):
        pe = _cloud_evaluator("enforce")
        with pytest.raises(PolicyViolationError):
            pe._enforce_decision(PolicyDecision(decision="block", rule_id="r1"), "tool_call")

    def test_dry_run_block_becomes_would_block_event(self):
        pe = _cloud_evaluator("dry_run")
        ev = pe.create_policy_check_event(
            PolicyDecision(decision="block", rule_id="r1", rule_name="No refunds"),
            "tool_call",
            {"x": 1},
            uuid4(),
            5,
        )
        assert ev.event_type == EventType.WOULD_BLOCK
        assert ev.error_type is None
        assert ev.metadata["would_block"]["would_be_outcome"] == "block"

    def test_dry_run_allow_stays_policy_check(self):
        pe = _cloud_evaluator("dry_run")
        ev = pe.create_policy_check_event(
            PolicyDecision(decision="allow"), "tool_call", {}, uuid4(), 1
        )
        assert ev.event_type == EventType.POLICY_CHECK

    def test_dry_run_enforce_decision_no_raise_no_handler(self):
        pe = _cloud_evaluator("dry_run")
        # Neither block nor require_approval raises, and the approval handler
        # (which would block on input()) is never touched.
        pe._enforce_decision(PolicyDecision(decision="block", rule_id="r1"), "tool_call")
        pe._enforce_decision(PolicyDecision(decision="require_approval", rule_id="r2"), "tool_call")

    def test_local_preflight_and_backend_block_emit_exactly_one(self, monkeypatch):
        # FRD-008: a matched local YAML rule must short-circuit the backend so a
        # single action never records TWO would_blocks (local + backend).
        rule = SimpleNamespace(
            parameter="amount",
            operator="max_value",
            value=100,
            action="block",
            name="big",
            policy_id="local1",
        )
        monkeypatch.setattr(
            lp,
            "load_sdk_policies",
            lambda: SimpleNamespace(
                actions={"tool_call": SimpleNamespace(policies=[rule])},
                global_=None,
                default_action="allow",
            ),
        )
        pe = _cloud_evaluator("dry_run")
        pe._client.evaluate_sync = lambda **kw: PolicyDecision(
            decision="block", rule_id="cloud1", rule_name="Cloud"
        )
        pe.evaluate_sync("tool_call", {"amount": 500}, uuid4(), 3)
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 1
        assert wb[0].metadata["would_block"]["rule_id"] == "local1"  # local short-circuits

    def test_dry_run_backend_error_allows_and_latches(self):
        # A10 regression: with fail_open=False a policy-backend error must NOT
        # hard-block in dry_run — record then allow. A persistent outage must record
        # exactly ONE would_block (latched), not one stdout line/event per call.
        cfg = ClyroConfig(
            api_key="cly_test",
            endpoint="http://x",
            fail_open=False,
            controls=ExecutionControls(enable_policy_enforcement=True, enforcement_mode="dry_run"),
        )
        pe = PolicyEvaluator(config=cfg, agent_id=uuid4())

        def boom(**kw):
            raise RuntimeError("policy backend down")

        pe._client.evaluate_sync = boom
        sid = uuid4()
        for _ in range(20):
            decision = pe.evaluate_sync("tool_call", {"amount": 5}, sid, 1)
            assert decision.is_allowed  # never hard-blocks in dry_run

        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 1  # latched: one marker for the persistent outage
        assert wb[0].metadata["would_block"]["rule_id"] == "system_error"


def _local_cfg(action):
    rule = SimpleNamespace(
        parameter="amount",
        operator="max_value",
        value=100,
        action=action,
        name="big",
        policy_id="p1",
    )
    return SimpleNamespace(
        actions={"tool_call": SimpleNamespace(policies=[rule])},
        global_=None,
        default_action="allow",
    )


class TestLocalPolicy:
    def test_enforce_raises(self, monkeypatch):
        monkeypatch.setattr(lp, "load_sdk_policies", lambda: _local_cfg("block"))
        ev = SDKLocalPolicyEvaluator(approval_handler=None, dry_run=False)
        with pytest.raises(PolicyViolationError):
            ev.evaluate_sync("tool_call", {"amount": 500}, uuid4(), 3)

    def test_dry_run_records_would_block_and_allows(self, monkeypatch):
        monkeypatch.setattr(lp, "load_sdk_policies", lambda: _local_cfg("block"))
        ev = SDKLocalPolicyEvaluator(approval_handler=None, dry_run=True)
        dec = ev.evaluate_sync("tool_call", {"amount": 500}, uuid4(), 3)
        assert dec.decision == "allow"
        wb = [e for e in ev.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 1 and wb[0].error_type is None

    def test_dry_run_require_approval_never_calls_handler(self, monkeypatch):
        # C13 / FRD-013: the blocking approval handler must NOT run in dry_run.
        monkeypatch.setattr(lp, "load_sdk_policies", lambda: _local_cfg("require_approval"))

        def boom(*a):
            raise AssertionError("approval handler must not be invoked in dry_run")

        ev = SDKLocalPolicyEvaluator(approval_handler=boom, dry_run=True)
        dec = ev.evaluate_sync("tool_call", {"amount": 500}, uuid4(), 3)
        assert dec.decision == "allow"
        wb = [e for e in ev.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert wb[0].metadata["would_block"]["would_be_outcome"] == "require_approval"


# ---------------------------------------------------------------------------
# Policy would-block volume is bounded at the source (FRD-022 for policy)
# ---------------------------------------------------------------------------


class TestPolicyWouldBlockAggregation:
    """Policy is NOT sticky — every action is a new decision — so without a latch
    a rule tripping on thousands of actions emits thousands of near-identical
    markers, burning the org's trace quota and bloating the report."""

    def _evaluator(self, rule_id="r1"):
        pe = _cloud_evaluator("dry_run")
        pe._client.evaluate_sync = lambda **kw: PolicyDecision(
            decision="block", rule_id=rule_id, rule_name="Rule"
        )
        return pe

    def test_one_marker_per_rule_per_session_not_one_per_action(self):
        pe = self._evaluator()
        sid = uuid4()
        for i in range(500):
            pe.evaluate_sync("tool_call", {"n": i}, sid, i)
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 1  # was 500

    def test_repeats_are_counted_for_the_report(self):
        pe = self._evaluator()
        sid = uuid4()
        for i in range(500):
            pe.evaluate_sync("tool_call", {"n": i}, sid, i)
        key = dry_run.WouldBlockLatch.policy_key(sid, "r1", "tool_call")
        assert pe._wb_latch.occurrences(key) == 500  # frequency still known
        assert pe._wb_latch.suppressed_total == 499

    def test_distinct_would_be_outcome_records_its_own_marker(self):
        # "would block" and "would require approval" are different findings —
        # collapsing them onto one key would silently drop the second.
        pe = self._evaluator()
        sid = uuid4()
        pe.evaluate_sync("tool_call", {}, sid, 1)  # block
        pe._client.evaluate_sync = lambda **kw: PolicyDecision(
            decision="require_approval", rule_id="r1", rule_name="Rule"
        )
        pe.evaluate_sync("tool_call", {}, sid, 2)  # require_approval, same rule
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        outcomes = {e.metadata["would_block"]["would_be_outcome"] for e in wb}
        assert outcomes == {"block", "require_approval"}

    def test_distinct_rule_records_its_own_marker(self):
        pe = self._evaluator()
        sid = uuid4()
        pe.evaluate_sync("tool_call", {"n": 1}, sid, 1)
        pe._client.evaluate_sync = lambda **kw: PolicyDecision(
            decision="block", rule_id="r2", rule_name="Other"
        )
        pe.evaluate_sync("tool_call", {"n": 2}, sid, 2)
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 2  # one per distinct rule

    def test_distinct_action_type_records_its_own_marker(self):
        pe = self._evaluator()
        sid = uuid4()
        pe.evaluate_sync("tool_call", {"n": 1}, sid, 1)
        pe.evaluate_sync("llm_call", {"n": 2}, sid, 2)
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 2

    def test_new_session_records_again(self):
        pe = self._evaluator()
        pe.evaluate_sync("tool_call", {"n": 1}, uuid4(), 1)
        pe.evaluate_sync("tool_call", {"n": 2}, uuid4(), 2)
        wb = [e for e in pe.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 2  # per-session, so each run reports its findings

    def test_latch_memory_is_bounded(self):
        latch = dry_run.WouldBlockLatch()
        latch._MAX_KEYS = 50
        for i in range(500):
            latch.record(f"session-{i}:policy:r:{i}")
        assert len(latch._counts) <= 50

    def test_local_evaluator_also_latches(self, monkeypatch):
        monkeypatch.setattr(lp, "load_sdk_policies", lambda: _local_cfg("block"))
        ev = SDKLocalPolicyEvaluator(approval_handler=None, dry_run=True)
        sid = uuid4()
        for i in range(100):
            ev.evaluate_sync("tool_call", {"amount": 500 + i}, sid, i)
        wb = [e for e in ev.drain_events() if e.event_type == EventType.WOULD_BLOCK]
        assert len(wb) == 1  # was 100

    def test_enforce_mode_policy_check_events_are_not_latched(self):
        # Regression: only dry_run would-blocks latch. Enforce must still emit a
        # policy_check per decision (AGS counts depend on it).
        pe = _cloud_evaluator("enforce")
        pe._client.evaluate_sync = lambda **kw: PolicyDecision(decision="allow")
        sid = uuid4()
        for i in range(10):
            pe.evaluate_sync("tool_call", {"n": i}, sid, i)
        checks = [e for e in pe.drain_events() if e.event_type == EventType.POLICY_CHECK]
        assert len(checks) == 10  # one per decision, unchanged


# ---------------------------------------------------------------------------
# The event sink must never silently drop a would-block (FRD-008/NFR-004)
# ---------------------------------------------------------------------------


class TestSinkNeverSilentlyDrops:
    """`session._event_sink` is the ONLY delivery path for would_block events
    (lifecycle/adapter events are buffered directly by the wrapper from outside
    the agent's loop). The checks run *during* execution, so a sync-wrapped agent
    whose framework runs an internal loop (LangGraph) reaches
    SyncTransport.buffer_event from inside a running loop, which raises by design.
    That used to be swallowed — the would-block vanished, and only "sometimes"."""

    def _wrapper(self):
        from clyro.transport import SyncTransport
        from clyro.wrapper import WrappedAgent

        cfg = ClyroConfig(
            api_key="cly_test_x",
            endpoint="http://x",
            controls=ExecutionControls(max_steps=2, enforcement_mode="dry_run"),
        )
        w = WrappedAgent.__new__(WrappedAgent)
        w._config = cfg
        w._sink_tasks = set()
        w._transport = SyncTransport(cfg)
        stored: list = []
        w._transport._transport._storage.store_event = lambda e, priority=None: (
            stored.append(e),
            True,
        )[1]
        return w, stored

    def _event(self):
        from clyro.trace import create_step_event

        return create_step_event(session_id=uuid4(), step_number=0, event_name="s")

    def test_would_block_survives_when_sink_called_inside_running_loop(self):
        import asyncio

        w, stored = self._wrapper()
        ev = self._event()

        async def inside_agent_loop():
            w._buffer_event_sink(ev)  # SyncTransport + running loop => raises

        asyncio.run(inside_agent_loop())
        assert len(stored) == 1, "event was dropped instead of falling back to storage"

    def test_fallback_is_logged_not_silent(self, capsys):
        import asyncio

        w, _ = self._wrapper()
        ev = self._event()

        async def inside_agent_loop():
            w._buffer_event_sink(ev)

        asyncio.run(inside_agent_loop())
        err = capsys.readouterr().err
        assert "clyro_event_sink_fallback_to_storage" in err
        assert "warning" in err.lower()  # a lost governance event must be visible

    def test_normal_path_still_buffers_without_fallback(self):
        # No running loop => the normal SyncTransport path works, no fallback.
        w, stored = self._wrapper()
        w._transport.buffer_event = lambda e: None  # succeeds
        w._buffer_event_sink(self._event())
        assert stored == []


# ---------------------------------------------------------------------------
# Report integrity under trace quota (FRD-023)
# ---------------------------------------------------------------------------


class TestQuotaDropSignal:
    """The backend returns `dropped: N` on a 2xx when the quota gate sheds
    events. The client MUST surface it, so a dry-run report declares itself
    incomplete instead of silently reading as a complete record."""

    def _transport(self):
        from clyro.transport import Transport

        return Transport(ClyroConfig(api_key="cly_test", endpoint="http://x"))

    def test_no_drop_leaves_report_complete(self):
        t = self._transport()
        t._record_dropped({"accepted": 5, "rejected": 0})
        assert t.dropped_count == 0
        assert t.report_incomplete is False

    def test_drop_marks_report_incomplete_and_accumulates(self):
        t = self._transport()
        t._record_dropped({"accepted": 0, "rejected": 7, "dropped": 7})
        assert t.dropped_count == 7
        assert t.report_incomplete is True
        t._record_dropped({"accepted": 1, "dropped": 3})  # cumulative across batches
        assert t.dropped_count == 10

    def test_drop_is_logged_at_warning_not_debug(self, capsys):
        t = self._transport()
        t._record_dropped({"accepted": 0, "dropped": 4})
        err = capsys.readouterr().err
        assert "clyro_traces_dropped_quota" in err
        assert "warning" in err.lower()  # never silent/debug

    def test_malformed_dropped_value_is_ignored(self):
        t = self._transport()
        t._record_dropped({"dropped": "not-a-number"})
        t._record_dropped({})  # legacy backend with no dropped field
        assert t.dropped_count == 0
        assert t.report_incomplete is False


# ---------------------------------------------------------------------------
# Adapters thread dry_run into their own local evaluator (FRD-020 A5 regression)
# ---------------------------------------------------------------------------


class TestOpenAIAdapterDryRunThreading:
    """The OpenAI standalone client builds its own local policy evaluator; it
    must inherit the dry-run mode or dry-run silently still blocks on OpenAI."""

    def _adapter(self, mode_str):
        from clyro.adapters.openai import OpenAIAdapter

        cfg = ClyroConfig(
            mode="local",
            controls=ExecutionControls(enable_policy_enforcement=True, enforcement_mode=mode_str),
        )
        ad = OpenAIAdapter.__new__(OpenAIAdapter)
        ad._config = cfg
        ad._approval_handler = None
        return ad

    def test_local_evaluator_inherits_dry_run(self):
        assert self._adapter("dry_run")._build_policy_evaluator()._is_dry_run is True

    def test_local_evaluator_enforce_unchanged(self):
        assert self._adapter("enforce")._build_policy_evaluator()._is_dry_run is False


# ---------------------------------------------------------------------------
# Loop-detector memory bound (FRD-021)
# ---------------------------------------------------------------------------


def test_loop_detector_state_hash_dict_is_bounded():
    from clyro.loop_detector import LoopDetector

    det = LoopDetector(threshold=3)
    det._MAX_STATE_HASHES = 100  # shrink for the test
    # Distinct states each visited once (no action → no action-sequence loop);
    # raise_on_loop=False so a would-be loop never raises during the sweep.
    for i in range(1000):
        det.check(state={"i": i}, raise_on_loop=False)
    # Once-seen hashes are pruned below threshold; the dict stays bounded.
    assert len(det._state.state_hash_counts) <= det._MAX_STATE_HASHES


# ---------------------------------------------------------------------------
# Log token + banner (FRD-011/012)
# ---------------------------------------------------------------------------


def test_token_is_the_fixed_greppable_string():
    assert DRY_RUN_TOKEN == "CLYRO-DRYRUN"


def test_banner_only_fires_in_dry_run(capsys):
    dry_run.log_mode_banner("sdk", is_dry_run=False)
    # enforce is silent
    dry_run.log_mode_banner("sdk", is_dry_run=True)
    err = capsys.readouterr().err
    assert DRY_RUN_TOKEN in err
