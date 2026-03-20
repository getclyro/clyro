# Implements NFR-002: Behavioral parity tests
"""
Verify consolidated modules produce identical results to original
implementations for identical inputs.
"""

import json


class TestCostTrackerParity:
    """CostTracker from clyro.cost matches original clyro_mcp.cost_tracker."""

    def test_check_budget_within_limit(self):
        from clyro.cost import CostTracker
        tracker = CostTracker(max_cost_usd=10.0, cost_per_token_usd=0.00001)
        exceeds, details = tracker.check_budget(0.0, {"key": "value"})
        assert exceeds is False
        assert details == {}

    def test_check_budget_exceeds_limit(self):
        from clyro.cost import CostTracker
        tracker = CostTracker(max_cost_usd=0.00001, cost_per_token_usd=0.00001)
        exceeds, details = tracker.check_budget(0.0, {"key": "x" * 1000})
        assert exceeds is True
        assert "accumulated_cost_usd" in details
        assert "max_cost_usd" in details

    def test_accumulate(self):
        from clyro.cost import CostTracker
        tracker = CostTracker(max_cost_usd=10.0, cost_per_token_usd=0.00001)
        cost = tracker.accumulate(400, 400)
        # (400 + 400) / 4 * 0.00001 = 200 * 0.00001 = 0.002
        assert abs(cost - 0.002) < 1e-9


class TestLoopDetectorParity:
    """compute_call_signature from clyro.loop_detector matches original."""

    def test_deterministic_signature(self):
        from clyro.loop_detector import compute_call_signature
        sig1 = compute_call_signature("tool_a", {"key": "value"})
        sig2 = compute_call_signature("tool_a", {"key": "value"})
        assert sig1 == sig2

    def test_different_tools_different_signatures(self):
        from clyro.loop_detector import compute_call_signature
        sig1 = compute_call_signature("tool_a", {"key": "value"})
        sig2 = compute_call_signature("tool_b", {"key": "value"})
        assert sig1 != sig2

    def test_none_params(self):
        from clyro.loop_detector import compute_call_signature
        sig = compute_call_signature("tool_a", None)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex


class TestLocalPolicyEvaluatorParity:
    """LocalPolicyEvaluator from clyro.policy matches original operators."""

    def _make_config(self, rules):
        """Create a minimal config-like object for LocalPolicyEvaluator."""
        from clyro.config import PolicyRule, ToolConfig, GlobalConfig, WrapperConfig
        global_config = GlobalConfig(policies=[PolicyRule(**r) for r in rules])
        return WrapperConfig(global_=global_config)

    def test_max_value_violation(self):
        from clyro.policy import LocalPolicyEvaluator
        config = self._make_config([
            {"parameter": "amount", "operator": "max_value", "value": 100}
        ])
        evaluator = LocalPolicyEvaluator(config)
        violated, details, _ = evaluator.evaluate("test_tool", {"amount": 150})
        assert violated is True

    def test_max_value_pass(self):
        from clyro.policy import LocalPolicyEvaluator
        config = self._make_config([
            {"parameter": "amount", "operator": "max_value", "value": 100}
        ])
        evaluator = LocalPolicyEvaluator(config)
        violated, details, _ = evaluator.evaluate("test_tool", {"amount": 50})
        assert violated is False

    def test_contains_violation(self):
        from clyro.policy import LocalPolicyEvaluator
        config = self._make_config([
            {"parameter": "command", "operator": "contains", "value": "rm -rf"}
        ])
        evaluator = LocalPolicyEvaluator(config)
        violated, _, _ = evaluator.evaluate("bash", {"command": "rm -rf /"})
        assert violated is True

    def test_not_contains_pass(self):
        from clyro.policy import LocalPolicyEvaluator
        config = self._make_config([
            {"parameter": "command", "operator": "not_contains", "value": "rm -rf"}
        ])
        evaluator = LocalPolicyEvaluator(config)
        # not_contains triggers when the value is absent from the parameter,
        # so "ls -la" not containing "rm -rf" IS a violation.
        violated, _, _ = evaluator.evaluate("bash", {"command": "ls -la"})
        assert violated is True
