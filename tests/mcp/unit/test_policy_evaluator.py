"""
Unit tests for PolicyEvaluator — TDD §11.1 tests #13–#18.
"""

from __future__ import annotations

from clyro.config import WrapperConfig
from clyro.policy import LocalPolicyEvaluator as PolicyEvaluator


def _config_with_global_policies(rules: list[dict]) -> WrapperConfig:
    return WrapperConfig.model_validate(
        {"global": {"policies": rules}}
    )


def _config_with_tool_policies(tool_name: str, rules: list[dict]) -> WrapperConfig:
    return WrapperConfig.model_validate(
        {"tools": {tool_name: {"policies": rules}}}
    )


class TestPolicyMaxValue:
    """Operator: max_value."""

    def test_blocks_over_max(self) -> None:
        """TDD §11.1 #13 — amount=1200 with max_value:500 → violated."""
        cfg = _config_with_global_policies(
            [{"parameter": "amount", "operator": "max_value", "value": 500}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 1200})
        assert violated
        assert details["operator"] == "max_value"
        assert details["actual"] == 1200

    def test_allows_under_max(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "amount", "operator": "max_value", "value": 500}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("transfer", {"amount": 100})
        assert not violated


class TestPolicyNotContains:
    """Operator: not_contains — blocks when expected value is ABSENT."""

    def test_allows_when_substring_present(self) -> None:
        """not_contains:'DROP' allows when 'DROP' IS present in value."""
        cfg = _config_with_tool_policies(
            "query_database",
            [{"parameter": "sql", "operator": "not_contains", "value": "DROP"}],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("query_database", {"sql": "DROP TABLE users"})
        assert not violated

    def test_blocks_when_substring_absent(self) -> None:
        """not_contains:'DROP' blocks when 'DROP' is NOT present in value."""
        cfg = _config_with_tool_policies(
            "query_database",
            [{"parameter": "sql", "operator": "not_contains", "value": "DROP"}],
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rr = pe.evaluate("query_database", {"sql": "SELECT * FROM users"})
        assert violated
        assert details["operator"] == "not_contains"


class TestPolicyInList:
    """Operator: in_list."""

    def test_allows_in_list(self) -> None:
        """TDD §11.1 #15 — category='books' in ['books','music'] → allowed."""
        cfg = _config_with_global_policies(
            [{"parameter": "category", "operator": "in_list", "value": ["books", "music"]}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("search", {"category": "books"})
        assert not violated

    def test_blocks_not_in_list(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "category", "operator": "in_list", "value": ["books", "music"]}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("search", {"category": "weapons"})
        assert violated


class TestPolicyNotInList:
    """Operator: not_in_list."""

    def test_blocks_in_blocked_list(self) -> None:
        """TDD §11.1 #16 — to='ceo@...' in not_in_list → violated."""
        cfg = _config_with_tool_policies(
            "send_email",
            [
                {
                    "parameter": "to",
                    "operator": "not_in_list",
                    "value": ["ceo@company.com", "board@company.com"],
                }
            ],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("send_email", {"to": "ceo@company.com"})
        assert violated

    def test_allows_not_in_blocked_list(self) -> None:
        cfg = _config_with_tool_policies(
            "send_email",
            [
                {
                    "parameter": "to",
                    "operator": "not_in_list",
                    "value": ["ceo@company.com"],
                }
            ],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("send_email", {"to": "alice@company.com"})
        assert not violated


class TestPolicyEvalOrder:
    """Evaluation ordering."""

    def test_per_tool_before_global(self) -> None:
        """TDD §11.1 #17 — per-tool rules evaluated before global."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "policies": [
                        {"parameter": "x", "operator": "max_value", "value": 999},
                    ]
                },
                "tools": {
                    "mytool": {
                        "policies": [
                            {"parameter": "x", "operator": "max_value", "value": 5},
                        ]
                    }
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        # x=10 violates per-tool (max_value:5) but not global (max_value:999)
        violated, details, _rule_results = pe.evaluate("mytool", {"x": 10})
        assert violated
        assert details["expected"] == 5  # per-tool rule triggered

    def test_wildcard_parameter(self) -> None:
        """TDD §11.1 #18 — *.amount matches any tool's amount param."""
        cfg = _config_with_global_policies(
            [{"parameter": "*.amount", "operator": "max_value", "value": 100}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("any_tool", {"amount": 500})
        assert violated


class TestPolicyOperators:
    """Cover remaining operators: equals, not_equals, min_value, contains."""

    def test_equals_blocks_mismatch(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "mode", "operator": "equals", "value": "safe"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"mode": "unsafe"})
        assert violated

    def test_equals_allows_match(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "mode", "operator": "equals", "value": "safe"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"mode": "safe"})
        assert not violated

    def test_not_equals_blocks_match(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "env", "operator": "not_equals", "value": "production"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"env": "production"})
        assert violated

    def test_min_value_blocks_below(self) -> None:
        cfg = _config_with_global_policies(
            [{"parameter": "count", "operator": "min_value", "value": 10}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"count": 3})
        assert violated

    def test_contains_blocks_when_present(self) -> None:
        """contains:'DANGER' blocks when 'DANGER' IS found in the value."""
        cfg = _config_with_global_policies(
            [{"parameter": "text", "operator": "contains", "value": "DANGER"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"text": "this is DANGER zone"})
        assert violated

    def test_contains_allows_when_absent(self) -> None:
        """contains:'DANGER' allows when 'DANGER' is NOT found in the value."""
        cfg = _config_with_global_policies(
            [{"parameter": "text", "operator": "contains", "value": "DANGER"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"text": "this is OK"})
        assert not violated

    def test_missing_parameter_no_violation(self) -> None:
        """Rule does not apply if parameter is absent."""
        cfg = _config_with_global_policies(
            [{"parameter": "nonexistent", "operator": "max_value", "value": 5}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"other": 100})
        assert not violated

    def test_none_arguments(self) -> None:
        """None arguments treated as empty dict."""
        cfg = _config_with_global_policies(
            [{"parameter": "x", "operator": "max_value", "value": 5}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", None)
        assert not violated

    def test_non_numeric_violates_max_value(self) -> None:
        """Non-numeric value with max_value -> violation (can't satisfy numeric bound)."""
        cfg = _config_with_global_policies(
            [{"parameter": "amount", "operator": "max_value", "value": 500}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("t", {"amount": "not-a-number"})
        assert violated
        assert details["operator"] == "max_value"

    def test_non_numeric_violates_min_value(self) -> None:
        """Non-numeric value with min_value -> violation."""
        cfg = _config_with_global_policies(
            [{"parameter": "count", "operator": "min_value", "value": 1}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"count": "abc"})
        assert violated


class TestPolicyIdInViolationDetails:
    """policy_id propagated in violation details (FRD-006)."""

    def test_violation_includes_policy_id(self) -> None:
        """Violation details should include policy_id from the rule."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "policies": [
                        {
                            "parameter": "amount",
                            "operator": "max_value",
                            "value": 100,
                            "name": "max_amount",
                            "policy_id": "00000000-1111-2222-3333-444444444444",
                        }
                    ]
                }
            }
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 500})
        assert violated
        assert details["policy_id"] == "00000000-1111-2222-3333-444444444444"

    def test_violation_policy_id_none_for_local(self) -> None:
        """Local YAML rules without policy_id should have None."""
        cfg = _config_with_global_policies(
            [{"parameter": "amount", "operator": "max_value", "value": 100}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 500})
        assert violated
        assert details["policy_id"] is None

    def test_no_violation_no_policy_id(self) -> None:
        """Non-violated rules should return empty details (no policy_id key)."""
        cfg = _config_with_global_policies(
            [{"parameter": "amount", "operator": "max_value", "value": 1000}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 50})
        assert not violated
        assert "policy_id" not in details
