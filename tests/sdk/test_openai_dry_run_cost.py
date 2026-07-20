# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI adapter dry-run cost tests (A10) — FRD-017/020/022.

Regression: the OpenAI adapter enforces the cost limit at TWO sites —
``_check_prevention_stack`` (pre-call) and ``_enforce_cost_limit_post_call``.
Only the pre-call one was gated for dry_run. The post-call site is the one that
actually fires: cost crosses the limit while recording *this* call's usage, so
it trips before the next turn's pre-call check ever runs.

The consequence was the worst kind for A10 — dry_run still BLOCKED the agent,
and emitted an enforced ``error`` sibling that FRD-017 forbids. "Monitor mode"
that halts production traffic is the exact footgun dry-run exists to remove.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from clyro.adapters.openai import TracedCompletions
from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import CostLimitExceededError


def _adapter(enforcement_mode: str, max_cost_usd: float = 0.01):
    cfg = ClyroConfig(
        agent_name="test-agent",
        controls=ExecutionControls(
            enable_cost_limit=True,
            max_cost_usd=max_cost_usd,
            enforcement_mode=enforcement_mode,
        ),
    )
    a = TracedCompletions.__new__(TracedCompletions)
    a._config = cfg
    a._agent_id = uuid4()
    a._org_id = None
    a._approval_handler = None
    return a


def _session(*, dry_run: bool, cumulative_cost: str = "5.00"):
    """Session double: over-budget, so both cost sites should trip."""
    s = MagicMock()
    s.session_id = uuid4()
    s.agent_id = uuid4()
    s.is_dry_run = dry_run
    s.cumulative_cost = Decimal(cumulative_cost)
    s.step_number = 3
    s._emit_would_block = MagicMock()
    return s


class TestPostCallCostLimit:
    """_enforce_cost_limit_post_call — the site that actually fires."""

    def test_dry_run_records_instead_of_raising(self):
        """The bug: dry_run raised CostLimitExceededError and halted the agent."""
        a = _adapter("dry_run")
        s = _session(dry_run=True)
        a._emit_error_event = MagicMock()
        a._auto_flush = MagicMock()

        a._enforce_cost_limit_post_call(s, {}, duration_ms=10)  # must NOT raise

        s._emit_would_block.assert_called_once()
        assert s._emit_would_block.call_args.args[0] == "cost"

    def test_dry_run_emits_no_enforced_error_sibling(self):
        """FRD-017: a would-block must not also produce an ``error`` event."""
        a = _adapter("dry_run")
        s = _session(dry_run=True)
        a._emit_error_event = MagicMock()
        a._auto_flush = MagicMock()

        a._enforce_cost_limit_post_call(s, {}, duration_ms=10)

        a._emit_error_event.assert_not_called()

    def test_enforce_mode_still_raises(self):
        """Regression: enforce must be unchanged — dry-run must not disarm it."""
        a = _adapter("enforce")
        s = _session(dry_run=False)
        a._emit_error_event = MagicMock()
        a._auto_flush = MagicMock()

        with pytest.raises(CostLimitExceededError):
            a._enforce_cost_limit_post_call(s, {}, duration_ms=10)
        a._emit_error_event.assert_called_once()

    def test_under_budget_is_a_noop_in_both_modes(self):
        for mode, dry in (("dry_run", True), ("enforce", False)):
            a = _adapter(mode, max_cost_usd=100.0)
            s = _session(dry_run=dry, cumulative_cost="0.01")
            a._emit_error_event = MagicMock()
            a._auto_flush = MagicMock()
            a._enforce_cost_limit_post_call(s, {}, duration_ms=10)
            s._emit_would_block.assert_not_called()
            a._emit_error_event.assert_not_called()

    def test_cost_limit_disabled_is_a_noop(self):
        cfg = ClyroConfig(
            agent_name="t",
            controls=ExecutionControls(
                enable_cost_limit=False, max_cost_usd=0.01, enforcement_mode="dry_run"
            ),
        )
        a = TracedCompletions.__new__(TracedCompletions)
        a._config = cfg
        s = _session(dry_run=True)
        a._enforce_cost_limit_post_call(s, {}, duration_ms=10)
        s._emit_would_block.assert_not_called()


class TestBothCostSitesAgree:
    """Pre-call and post-call must behave identically — the divergence WAS the bug."""

    def test_both_sites_record_in_dry_run(self):
        a = _adapter("dry_run")
        a._emit_error_event = MagicMock()
        a._auto_flush = MagicMock()

        post = _session(dry_run=True)
        a._enforce_cost_limit_post_call(post, {}, duration_ms=10)

        pre = _session(dry_run=True)
        pre.step_number = 1
        a._config.controls.enable_step_limit = False
        a._config.controls.enable_loop_detection = False
        a._check_prevention_stack(pre, {})

        for s in (post, pre):
            s._emit_would_block.assert_called_once()
            assert s._emit_would_block.call_args.args[0] == "cost"

    def test_both_sites_share_a_dedup_key(self):
        """FRD-022: one marker per session, not one per enforcement site."""
        a = _adapter("dry_run")
        a._emit_error_event = MagicMock()
        a._auto_flush = MagicMock()
        a._config.controls.enable_step_limit = False
        a._config.controls.enable_loop_detection = False

        s = _session(dry_run=True)
        a._enforce_cost_limit_post_call(s, {}, duration_ms=10)
        a._check_prevention_stack(s, {})

        keys = {c.kwargs.get("dedup_key") for c in s._emit_would_block.call_args_list}
        assert keys == {"cost"}, f"sites must share dedup_key='cost', got {keys}"
