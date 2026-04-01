"""
Unit tests for MCP governance enrichment (FRD-BE-011).

Covers:
- PolicyEvaluator dual-path: enforcement + reporting rule_results
- PolicyEvaluator rule_results structure and outcomes
- TraceEventFactory.policy_check() with decision and rule_results
- Graceful degradation on rule_result serialization failure
"""

from __future__ import annotations

from uuid import uuid4

from clyro.backend.trace_event_factory import TraceEventFactory
from clyro.config import WrapperConfig
from clyro.mcp.session import McpSession
from clyro.policy import LocalPolicyEvaluator as PolicyEvaluator


def _config_with_rules(rules: list[dict]) -> WrapperConfig:
    return WrapperConfig.model_validate({"global": {"policies": rules}})


def _config_with_tool_rules(tool_name: str, rules: list[dict]) -> WrapperConfig:
    return WrapperConfig.model_validate({"tools": {tool_name: {"policies": rules}}})


# =============================================================================
# PolicyEvaluator — Dual-Path Evaluation (FRD-BE-011)
# =============================================================================


class TestDualPathEvaluation:
    """Verify enforcement + reporting paths work correctly."""

    def test_all_rules_evaluated_even_after_violation(self) -> None:
        """FRD-BE-011: All rules must be evaluated for rule_results."""
        cfg = _config_with_rules([
            {"parameter": "amount", "operator": "max_value", "value": 100, "name": "max_amount"},
            {"parameter": "currency", "operator": "equals", "value": "USD", "name": "currency_check"},
        ])
        pe = PolicyEvaluator(cfg)
        violated, details, rule_results = pe.evaluate("transfer", {"amount": 500, "currency": "USD"})

        assert violated
        assert details["rule_name"] == "max_amount"
        # Both rules should be in rule_results
        assert len(rule_results) == 2
        assert rule_results[0]["outcome"] == "triggered"
        assert rule_results[1]["outcome"] == "passed"

    def test_no_violation_returns_all_passed(self) -> None:
        """When no rules violated, all rule_results should be 'passed'."""
        cfg = _config_with_rules([
            {"parameter": "amount", "operator": "max_value", "value": 1000},
            {"parameter": "currency", "operator": "equals", "value": "USD"},
        ])
        pe = PolicyEvaluator(cfg)
        violated, details, rule_results = pe.evaluate("t", {"amount": 50, "currency": "USD"})

        assert not violated
        assert details == {}
        assert len(rule_results) == 2
        assert all(r["outcome"] == "passed" for r in rule_results)

    def test_missing_parameter_produces_skipped(self) -> None:
        """Parameters not present → outcome='skipped'."""
        cfg = _config_with_rules([
            {"parameter": "nonexistent", "operator": "max_value", "value": 5},
        ])
        pe = PolicyEvaluator(cfg)
        violated, _, rule_results = pe.evaluate("t", {"other": 100})

        assert not violated
        assert len(rule_results) == 1
        assert rule_results[0]["outcome"] == "skipped"
        assert rule_results[0]["actual_value"] is None

    def test_empty_rules_returns_empty_results(self) -> None:
        """No rules → empty rule_results (FRD-BE-011 §C8)."""
        cfg = _config_with_rules([])
        pe = PolicyEvaluator(cfg)
        violated, _, rule_results = pe.evaluate("t", {"x": 1})

        assert not violated
        assert rule_results == []

    def test_none_arguments_returns_skipped(self) -> None:
        """None arguments → all rules skipped."""
        cfg = _config_with_rules([
            {"parameter": "x", "operator": "max_value", "value": 5},
        ])
        pe = PolicyEvaluator(cfg)
        violated, _, rule_results = pe.evaluate("t", None)

        assert not violated
        assert len(rule_results) == 1
        assert rule_results[0]["outcome"] == "skipped"


class TestRuleResultStructure:
    """Verify rule_result dict structure matches TDD §3.3."""

    def test_triggered_rule_result_fields(self) -> None:
        """Triggered rule result has all required fields."""
        policy_id = str(uuid4())
        cfg = _config_with_rules([
            {
                "parameter": "amount",
                "operator": "max_value",
                "value": 100,
                "name": "Max Amount",
                "policy_id": policy_id,
            }
        ])
        pe = PolicyEvaluator(cfg)
        _, _, rule_results = pe.evaluate("transfer", {"amount": 500})

        r = rule_results[0]
        assert r["policy_id"] == policy_id
        assert r["policy_name"] is None  # MCP doesn't have policy names
        assert r["rule_id"] is None  # YAML rules have no ID
        assert r["rule_name"] == "Max Amount"
        assert r["field"] == "amount"
        assert r["operator"] == "max_value"
        assert r["threshold"] == 100
        assert r["actual_value"] == 500
        assert r["outcome"] == "triggered"
        assert r["action"] == "block"
        assert r["message"] is not None

    def test_passed_rule_result_fields(self) -> None:
        """Passed rule result has null message."""
        cfg = _config_with_rules([
            {"parameter": "amount", "operator": "max_value", "value": 1000, "name": "Limit"},
        ])
        pe = PolicyEvaluator(cfg)
        _, _, rule_results = pe.evaluate("t", {"amount": 50})

        r = rule_results[0]
        assert r["outcome"] == "passed"
        assert r["action"] == "allow"
        assert r["message"] is None
        assert r["actual_value"] == 50

    def test_per_tool_and_global_rules_combined(self) -> None:
        """Both per-tool and global rules appear in rule_results."""
        cfg = WrapperConfig.model_validate({
            "global": {"policies": [
                {"parameter": "x", "operator": "max_value", "value": 999},
            ]},
            "tools": {"mytool": {"policies": [
                {"parameter": "x", "operator": "max_value", "value": 5, "name": "tool_limit"},
            ]}},
        })
        pe = PolicyEvaluator(cfg)
        violated, _, rule_results = pe.evaluate("mytool", {"x": 10})

        assert violated
        assert len(rule_results) == 2
        # Per-tool first, then global
        assert rule_results[0]["rule_name"] == "tool_limit"
        assert rule_results[0]["outcome"] == "triggered"
        assert rule_results[1]["outcome"] == "passed"  # x=10 < 999 (global rule passes)


# =============================================================================
# TraceEventFactory — Enriched policy_check (FRD-BE-011)
# =============================================================================


class TestTraceEventFactoryEnrichment:
    """Verify TraceEventFactory.policy_check() includes decision and rule_results."""

    def _make_factory(self) -> TraceEventFactory:
        session = McpSession()
        return TraceEventFactory(session)

    def test_policy_check_without_enrichment(self) -> None:
        """Backward compat: policy_check without decision/rule_results works."""
        factory = self._make_factory()
        event = factory.policy_check("read_file", {"path": "/tmp"})

        assert event["event_type"] == "policy_check"
        # No decision or rule_results in metadata (only _source and cost_estimated)
        assert "decision" not in event["metadata"]
        assert "rule_results" not in event["metadata"]

    def test_policy_check_with_decision_and_rule_results(self) -> None:
        """FRD-BE-011: policy_check includes decision and rule_results in metadata."""
        factory = self._make_factory()
        rule_results = [
            {"policy_id": "p1", "outcome": "passed"},
            {"policy_id": "p2", "outcome": "triggered"},
        ]
        event = factory.policy_check(
            "transfer",
            {"amount": 500},
            decision="block",
            rule_results=rule_results,
        )

        assert event["event_type"] == "policy_check"
        assert event["metadata"]["decision"] == "block"
        assert event["metadata"]["rule_results"] == rule_results
        assert event["metadata"]["_source"] == "mcp"  # still present

    def test_policy_check_with_allow_decision(self) -> None:
        """Allowed calls include decision='allow' in metadata."""
        factory = self._make_factory()
        event = factory.policy_check(
            "read_file",
            {"path": "/tmp"},
            decision="allow",
            rule_results=[],
        )

        assert event["metadata"]["decision"] == "allow"
        assert event["metadata"]["rule_results"] == []
