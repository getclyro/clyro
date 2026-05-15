"""
Unit tests for PolicyEvaluator — TDD §11.1 tests #13–#18.
"""

from __future__ import annotations

from clyro.config import WrapperConfig
from clyro.policy import LocalPolicyEvaluator as PolicyEvaluator


def _config_with_global_policies(
    rules: list[dict], default_action: str = "allow"
) -> WrapperConfig:
    return WrapperConfig.model_validate(
        {"global": {"policies": rules}, "default_action": default_action}
    )


def _config_with_tool_policies(
    tool_name: str, rules: list[dict], default_action: str = "allow"
) -> WrapperConfig:
    return WrapperConfig.model_validate(
        {
            "tools": {tool_name: {"policies": rules}},
            "default_action": default_action,
        }
    )


class TestPolicyMaxValue:
    """Operator: max_value."""

    def test_blocks_over_max(self) -> None:
        """TDD §11.1 #13 — amount=1200 with max_value:500 → violated."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "amount", "operator": "max_value", "value": 500}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 1200})
        assert violated
        assert details["operator"] == "max_value"
        assert details["actual"] == 1200

    def test_allows_under_max(self) -> None:
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "amount", "operator": "max_value", "value": 500}]
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
            [{"action": "block", "parameter": "sql", "operator": "not_contains", "value": "DROP"}],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("query_database", {"sql": "DROP TABLE users"})
        assert not violated

    def test_blocks_when_substring_absent(self) -> None:
        """not_contains:'DROP' blocks when 'DROP' is NOT present in value."""
        cfg = _config_with_tool_policies(
            "query_database",
            [{"action": "block", "parameter": "sql", "operator": "not_contains", "value": "DROP"}],
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rr = pe.evaluate("query_database", {"sql": "SELECT * FROM users"})
        assert violated
        assert details["operator"] == "not_contains"


class TestPolicyInList:
    """Operator: in_list — matches when actual IS in list (default action=block)."""

    def test_blocks_when_in_list(self) -> None:
        """in_list ['books','music'] matches when category='books' → BLOCK."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "category", "operator": "in_list", "value": ["books", "music"]}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("search", {"category": "books"})
        assert violated

    def test_allows_when_not_in_list(self) -> None:
        """in_list ['books','music'] does not match when category='weapons' → ALLOW (default)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "category", "operator": "in_list", "value": ["books", "music"]}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("search", {"category": "weapons"})
        assert not violated


class TestPolicyNotInList:
    """Operator: not_in_list — matches when actual is NOT in list (default action=block)."""

    def test_allows_when_in_list(self) -> None:
        """not_in_list does not match when value IS in list → ALLOW (default)."""
        cfg = _config_with_tool_policies(
            "send_email",
            [
                {"action": "block", "parameter": "to",
                    "operator": "not_in_list",
                    "value": ["ceo@company.com", "board@company.com"],
                }
            ],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("send_email", {"to": "ceo@company.com"})
        assert not violated

    def test_blocks_when_not_in_list(self) -> None:
        """not_in_list matches when value is NOT in list → BLOCK."""
        cfg = _config_with_tool_policies(
            "send_email",
            [
                {"action": "block", "parameter": "to",
                    "operator": "not_in_list",
                    "value": ["ceo@company.com"],
                }
            ],
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("send_email", {"to": "alice@company.com"})
        assert violated


class TestPolicyEvalOrder:
    """Evaluation ordering."""

    def test_per_tool_before_global(self) -> None:
        """TDD §11.1 #17 — per-tool rules evaluated before global."""
        cfg = WrapperConfig.model_validate(
            {"default_action": "allow",
                "global": {
                    "policies": [
                        {"action": "block", "parameter": "x", "operator": "max_value", "value": 999},
                    ]
                },
                "tools": {
                    "mytool": {
                        "policies": [
                            {"action": "block", "parameter": "x", "operator": "max_value", "value": 5},
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
            [{"action": "block", "parameter": "*.amount", "operator": "max_value", "value": 100}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("any_tool", {"amount": 500})
        assert violated


class TestPolicyOperators:
    """Cover remaining operators: equals, not_equals, min_value, contains."""

    def test_equals_blocks_when_equal(self) -> None:
        """equals 'safe' matches when mode=='safe' → BLOCK (default action)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "mode", "operator": "equals", "value": "safe"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"mode": "safe"})
        assert violated

    def test_equals_allows_when_unequal(self) -> None:
        """equals 'safe' does not match when mode!='safe' → ALLOW (default)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "mode", "operator": "equals", "value": "safe"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"mode": "unsafe"})
        assert not violated

    def test_not_equals_allows_when_equal(self) -> None:
        """not_equals 'production' does not match when env=='production' → ALLOW."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "env", "operator": "not_equals", "value": "production"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"env": "production"})
        assert not violated

    def test_min_value_blocks_below(self) -> None:
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "count", "operator": "min_value", "value": 10}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"count": 3})
        assert violated

    def test_contains_blocks_when_present(self) -> None:
        """contains:'DANGER' blocks when 'DANGER' IS found in the value."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "text", "operator": "contains", "value": "DANGER"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"text": "this is DANGER zone"})
        assert violated

    def test_contains_allows_when_absent(self) -> None:
        """contains:'DANGER' allows when 'DANGER' is NOT found in the value."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "text", "operator": "contains", "value": "DANGER"}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"text": "this is OK"})
        assert not violated

    def test_missing_parameter_no_violation(self) -> None:
        """Rule does not apply if parameter is absent."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "nonexistent", "operator": "max_value", "value": 5}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"other": 100})
        assert not violated

    def test_none_arguments(self) -> None:
        """None arguments treated as empty dict."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "x", "operator": "max_value", "value": 5}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", None)
        assert not violated

    def test_non_numeric_matches_max_value(self) -> None:
        """Non-numeric value with max_value → match=True → action fires (fail-closed block)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "amount", "operator": "max_value", "value": 500}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("t", {"amount": "not-a-number"})
        assert violated
        assert details["operator"] == "max_value"

    def test_non_numeric_matches_min_value(self) -> None:
        """Non-numeric value with min_value → match=True → action fires (fail-closed block)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "count", "operator": "min_value", "value": 1}]
        )
        pe = PolicyEvaluator(cfg)
        violated, _, _rr = pe.evaluate("t", {"count": "abc"})
        assert violated


class TestPolicyIdInViolationDetails:
    """policy_id propagated in violation details (FRD-006)."""

    def test_violation_includes_policy_id(self) -> None:
        """Violation details should include policy_id from the rule."""
        cfg = WrapperConfig.model_validate(
            {"default_action": "allow",
                "global": {
                    "policies": [
                        {"action": "block", "parameter": "amount",
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
            [{"action": "block", "parameter": "amount", "operator": "max_value", "value": 100}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 500})
        assert violated
        assert details["policy_id"] is None

    def test_no_violation_no_policy_id(self) -> None:
        """Non-violated rules should return empty details (no policy_id key)."""
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "amount", "operator": "max_value", "value": 1000}]
        )
        pe = PolicyEvaluator(cfg)
        violated, details, _rule_results = pe.evaluate("transfer", {"amount": 50})
        assert not violated
        assert "policy_id" not in details


# ===========================================================================
# New-semantic behavior tests
# ===========================================================================


class TestActionAllowInMCP:
    """MCP/hooks respects action: allow on matched rules."""

    def test_action_allow_matched_yields_no_block(self) -> None:
        cfg = _config_with_global_policies(
            [
                {
                    "parameter": "cluster",
                    "operator": "in_list",
                    "value": ["c1"],
                    "action": "allow",
                }
            ]
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("t", {"cluster": "c1"})
        assert not blocked

    def test_action_allow_short_circuits_later_block(self) -> None:
        cfg = _config_with_global_policies(
            [
                {
                    "parameter": "cluster",
                    "operator": "in_list",
                    "value": ["c1"],
                    "action": "allow",
                },
                {
                    "action": "block",
                    "parameter": "cluster",
                    "operator": "in_list",
                    "value": ["c1"],
                },
            ]
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("t", {"cluster": "c1"})
        assert not blocked


class TestDefaultActionInMCP:
    """default_action covers the no-match path in MCP/hooks."""

    def test_default_action_allow_no_match(self) -> None:
        cfg = _config_with_global_policies(
            [{"action": "block", "parameter": "cluster", "operator": "in_list", "value": ["c1"]}]
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("t", {"cluster": "c2"})
        assert not blocked

    def test_default_action_block_no_match(self) -> None:
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {"action": "block", "parameter": "cluster", "operator": "in_list", "value": ["c1"]}
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        blocked, details, _ = pe.evaluate("t", {"cluster": "c2"})
        assert blocked
        # Synthetic details so consumers see something actionable
        assert details["rule_name"] == "default_action"
        assert details["operator"] == "default_action"

    def test_default_action_block_skipped_when_rule_matches(self) -> None:
        """A matched action: allow rule short-circuits before default_action block fires."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {
                            "parameter": "cluster",
                            "operator": "in_list",
                            "value": ["c1"],
                            "action": "allow",
                        }
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("t", {"cluster": "c1"})
        assert not blocked


class TestFirstMatchWinsInMCP:
    """First matching rule's action wins; later rules don't override."""

    def test_block_then_allow_first_wins(self) -> None:
        cfg = _config_with_global_policies(
            [
                {
                    "action": "block",
                    "parameter": "cluster",
                    "operator": "in_list",
                    "value": ["c1"],
                },
                {
                    "parameter": "cluster",
                    "operator": "in_list",
                    "value": ["c1"],
                    "action": "allow",
                },
            ]
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("t", {"cluster": "c1"})
        assert blocked

    def test_per_tool_evaluated_before_global(self) -> None:
        cfg = WrapperConfig.model_validate(
            {"default_action": "allow",
                "global": {
                    "policies": [
                        {"action": "block", "parameter": "cluster",
                            "operator": "in_list",
                            "value": ["c1"],
                            "name": "global_block",
                        }
                    ]
                },
                "tools": {
                    "mytool": {
                        "policies": [
                            {
                                "parameter": "cluster",
                                "operator": "in_list",
                                "value": ["c1"],
                                "action": "allow",
                                "name": "tool_allow",
                            }
                        ]
                    }
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        # Per-tool runs first; allow short-circuits → not blocked
        blocked, _, _ = pe.evaluate("mytool", {"cluster": "c1"})
        assert not blocked


class TestFixBAllowlistShape:
    """Fix B: ``default_action`` bypassed when every rule was skipped because
    its parameter wasn't present in the tool's arguments. Lets the allowlist
    shape (``default_action: block`` + ``action: allow`` on a field-scoped
    rule) work on tools that don't carry the rule's parameter."""

    def test_field_absent_bypasses_default_action_block(self) -> None:
        """User's reported case: rmq_cluster rule, default_action: block.
        A tool call without rmq_cluster should NOT be blocked."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {
                            "parameter": "rmq_cluster",
                            "operator": "in_list",
                            "value": ["cluster1", "cluster3"],
                            "name": "allowed_clusters",
                            "action": "allow",
                        }
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        # Tool call without rmq_cluster — rule skipped, default_action bypassed
        blocked, _, _ = pe.evaluate("some_other_tool", {"foo": "bar"})
        assert not blocked

    def test_field_present_and_matches_allows(self) -> None:
        """Allowlist hit: rule matches → action: allow short-circuits."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {
                            "parameter": "rmq_cluster",
                            "operator": "in_list",
                            "value": ["cluster1", "cluster3"],
                            "name": "allowed_clusters",
                            "action": "allow",
                        }
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("rmq_tool", {"rmq_cluster": "cluster1"})
        assert not blocked

    def test_field_present_but_no_match_fires_default_action(self) -> None:
        """Allowlist miss: field present, doesn't match → default_action fires."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {
                            "parameter": "rmq_cluster",
                            "operator": "in_list",
                            "value": ["cluster1", "cluster3"],
                            "name": "allowed_clusters",
                            "action": "allow",
                        }
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        blocked, details, _ = pe.evaluate("rmq_tool", {"rmq_cluster": "cluster2"})
        assert blocked
        assert details["rule_name"] == "default_action"

    def test_zero_rules_with_default_block_still_blocks(self) -> None:
        """Preserved: zero rules + default_action: block → BLOCK. The
        all-skipped bypass requires rules to exist."""
        cfg = WrapperConfig.model_validate(
            {"default_action": "block", "global": {"policies": []}}
        )
        pe = PolicyEvaluator(cfg)
        blocked, _, _ = pe.evaluate("any_tool", {"any": "args"})
        assert blocked

    def test_mixed_skipped_and_no_match_fires_default_action(self) -> None:
        """If at least one rule was actually evaluated (no_match), the policy
        is applicable and default_action fires normally."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "block",
                "global": {
                    "policies": [
                        {
                            "parameter": "rmq_cluster",
                            "operator": "in_list",
                            "value": ["cluster1"],
                            "name": "rmq_allowlist",
                            "action": "allow",
                        },
                        {
                            "parameter": "endpoint",
                            "operator": "in_list",
                            "value": ["safe.example.com"],
                            "name": "endpoint_allowlist",
                            "action": "allow",
                        },
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        # endpoint IS present (rule 2 evaluated as no_match — value not in
        # list); rmq_cluster is absent (rule 1 skipped). Not all skipped →
        # default_action: block fires.
        blocked, details, _ = pe.evaluate(
            "other_tool", {"endpoint": "evil.example.com"}
        )
        assert blocked
        assert details["rule_name"] == "default_action"

    def test_denylist_shape_unaffected(self) -> None:
        """The denylist shape never relied on default_action firing on skipped
        rules, so Fix B doesn't change its behavior."""
        cfg = WrapperConfig.model_validate(
            {
                "default_action": "allow",
                "global": {
                    "policies": [
                        {
                            "parameter": "rmq_cluster",
                            "operator": "not_in_list",
                            "value": ["cluster1", "cluster3"],
                            "name": "rmq_denylist",
                            "action": "block",
                        }
                    ]
                },
            }
        )
        pe = PolicyEvaluator(cfg)
        # Tool without rmq_cluster → rule skipped → default_action: allow
        blocked, _, _ = pe.evaluate("other_tool", {"foo": "bar"})
        assert not blocked
        # Tool with disallowed cluster → matches → blocks
        blocked, _, _ = pe.evaluate("rmq_tool", {"rmq_cluster": "cluster2"})
        assert blocked
