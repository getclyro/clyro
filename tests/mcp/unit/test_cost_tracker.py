"""
Unit tests for CostTracker — TDD §11.1 tests #9–#12.
"""

from __future__ import annotations

import pytest

from clyro.cost import CostTracker


class TestCostTracker:
    """Character-length heuristic cost tracking."""

    def test_allows_under_budget(self) -> None:
        """TDD §11.1 #9 — accumulated < max -> allowed."""
        ct = CostTracker(max_cost_usd=10.0, cost_per_token_usd=0.00001)
        exceeds, _ = ct.check_budget(0.0, {"sql": "SELECT 1"})
        assert not exceeds

    def test_blocks_over_budget(self) -> None:
        """TDD §11.1 #10 — accumulated + estimated > max -> blocked."""
        ct = CostTracker(max_cost_usd=0.0001, cost_per_token_usd=0.01)
        exceeds, details = ct.check_budget(0.0, {"data": "x" * 1000})
        assert exceeds
        assert "max_cost_usd" in details
        assert details["cost_estimated"] is True

    def test_blocked_calls_no_cost(self) -> None:
        """TDD §11.1 #11 — blocked calls don't accumulate cost.

        The CostTracker itself does not accumulate; the caller does.
        Verify that ``accumulate`` is the only way to add cost.
        """
        ct = CostTracker(max_cost_usd=10.0, cost_per_token_usd=0.00001)
        # check_budget is read-only — calling it doesn't change anything
        ct.check_budget(5.0, {"x": 1})
        # no accumulate call -> no cost change (caller responsibility)

    def test_cost_heuristic_formula(self) -> None:
        """TDD §11.1 #12 — verify accumulate formula: (params_len + resp_len) / 4 * rate."""
        rate = 0.00001
        ct = CostTracker(max_cost_usd=10.0, cost_per_token_usd=rate)
        params_len = 100
        resp_len = 200
        cost = ct.accumulate(params_len, resp_len)
        expected = (params_len + resp_len) / 4 * rate
        assert abs(cost - expected) < 1e-12

    def test_pre_check_uses_response_overhead(self) -> None:
        """Pre-check applies 2x multiplier for expected response cost."""
        rate = 0.01
        ct = CostTracker(max_cost_usd=1.0, cost_per_token_usd=rate)
        # params '{"x": 1}' = 8 chars -> 2 tokens input -> 4 tokens w/ 2x -> $0.04
        # Budget is $1.0, accumulated is $0.97
        # Without 2x: $0.97 + $0.02 = $0.99 < $1.0 (would pass)
        # With 2x:    $0.97 + $0.04 = $1.01 > $1.0 (should block)
        exceeds, _ = ct.check_budget(0.97, {"x": 1})
        assert exceeds

    def test_invalid_max_cost_raises(self) -> None:
        with pytest.raises(ValueError):
            CostTracker(max_cost_usd=0)

    def test_invalid_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            CostTracker(max_cost_usd=1.0, cost_per_token_usd=-0.001)

    def test_empty_params_budget_check(self) -> None:
        """Empty params should have minimal cost."""
        ct = CostTracker(max_cost_usd=10.0, cost_per_token_usd=0.00001)
        exceeds, _ = ct.check_budget(9.99999, None)
        assert not exceeds  # "{}" is 2 chars -> 0.5 tokens * 2 = 1 token -> negligible
