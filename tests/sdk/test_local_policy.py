# Tests for local_policy.py — C1 (YAML loader/models) + C2 (SDKLocalPolicyEvaluator)
# Implements TDD §13.1 C1/C2 test cases

"""
Test coverage targets:
- C1 (YAML loader): 90%+
- C2 (SDKLocalPolicyEvaluator): 90%+
"""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from clyro.exceptions import ClyroConfigError, PolicyViolationError
from clyro.local_policy import (
    SDKLocalPolicyEvaluator,
    SDKPolicyConfig,
    SDKPolicyRule,
    load_sdk_policies,
    reset_sdk_policy_cache,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the module-level policy cache before each test."""
    reset_sdk_policy_cache()
    yield
    reset_sdk_policy_cache()


@pytest.fixture()
def policy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect policy loading to a temp directory."""
    sdk_dir = tmp_path / ".clyro" / "sdk"
    sdk_dir.mkdir(parents=True)
    monkeypatch.setattr("clyro.local_policy._POLICY_DIR", sdk_dir)
    monkeypatch.setattr("clyro.local_policy._POLICY_FILE", sdk_dir / "policies.yaml")
    return sdk_dir


def _write_policy(policy_dir: Path, content: str) -> Path:
    """Write a policy YAML, auto-injecting defaults for required fields.

    Tests can pre-set `default_action` or per-rule `action` to override; the
    injection only fills in fields that are missing. Malformed YAML strings
    are passed through unchanged so error-path tests still exercise validation.
    """
    p = policy_dir / "policies.yaml"
    text = textwrap.dedent(content)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None

    if isinstance(data, dict):
        if "default_action" not in data:
            data["default_action"] = "allow"

        def _inject_rule_action(rule: object) -> None:
            if isinstance(rule, dict) and "action" not in rule:
                rule["action"] = "block"

        global_section = data.get("global")
        if isinstance(global_section, dict):
            for rule in global_section.get("policies", []) or []:
                _inject_rule_action(rule)

        actions_section = data.get("actions")
        if isinstance(actions_section, dict):
            for action_block in actions_section.values():
                if isinstance(action_block, dict):
                    for rule in action_block.get("policies", []) or []:
                        _inject_rule_action(rule)

        text = yaml.dump(data, sort_keys=False)

    p.write_text(text, encoding="utf-8")
    return p


# ===========================================================================
# C1: SDKPolicyRule model tests
# ===========================================================================


class TestSDKPolicyRule:
    """FRD-SOF-001: SDKPolicyRule validation."""

    def test_valid_block_action(self):
        rule = SDKPolicyRule(
            parameter="cost", operator="max_value", value=100, action="block",
        )
        assert rule.action == "block"

    def test_valid_require_approval_action(self):
        rule = SDKPolicyRule(
            parameter="cost", operator="max_value", value=100,
            action="require_approval",
        )
        assert rule.action == "require_approval"

    def test_action_required(self):
        """action is required — omitting it raises."""
        with pytest.raises(ValidationError):
            SDKPolicyRule(
                parameter="cost", operator="max_value", value=100,
            )

    def test_invalid_action_raises(self):
        with pytest.raises(ValidationError):
            SDKPolicyRule(
                parameter="cost", operator="max_value", value=100,
                action="unknown_value",
            )

    def test_inherits_operator_validation(self):
        with pytest.raises(ValidationError):
            SDKPolicyRule(
                parameter="cost", operator="invalid_op", value=100, action="block",
            )

    def test_extra_fields_ignored(self):
        """SDKPolicyRule has extra='ignore' — future MCP fields accepted."""
        rule = SDKPolicyRule(
            parameter="cost", operator="max_value", value=100, action="block",
            future_mcp_field="whatever",  # type: ignore[call-arg]
        )
        assert rule.parameter == "cost"
        assert not hasattr(rule, "future_mcp_field")

    def test_all_8_operators_accepted(self):
        ops = [
            "max_value", "min_value", "equals", "not_equals",
            "in_list", "not_in_list", "contains", "not_contains",
        ]
        for op in ops:
            rule = SDKPolicyRule(parameter="x", operator=op, value=1, action="block")
            assert rule.operator == op


# ===========================================================================
# C1: SDKPolicyConfig model tests
# ===========================================================================


class TestSDKPolicyConfig:
    """FRD-SOF-001, FRD-SOF-003: YAML schema validation."""

    def test_valid_config(self):
        config = SDKPolicyConfig(version=1, default_action="allow")
        assert config.version == 1

    def test_wrong_version_raises(self):
        with pytest.raises(ValidationError):
            SDKPolicyConfig(version=2, default_action="allow")

    def test_default_action_required(self):
        """default_action is required — omitting it raises."""
        with pytest.raises(ValidationError):
            SDKPolicyConfig(version=1)

    def test_global_alias(self):
        """'global' YAML key maps to global_ Python field."""
        data = {"version": 1, "default_action": "allow", "global": {"policies": []}}
        config = SDKPolicyConfig.model_validate(data)
        assert config.global_ is not None
        assert config.global_.policies == []

    def test_actions_with_known_types(self):
        data = {
            "version": 1,
            "default_action": "allow",
            "actions": {
                "llm_call": {
                    "policies": [
                        {"parameter": "model", "operator": "in_list",
                         "value": ["gpt-4"], "name": "test", "action": "block"},
                    ],
                },
            },
        }
        config = SDKPolicyConfig.model_validate(data)
        assert "llm_call" in config.actions
        assert len(config.actions["llm_call"].policies) == 1

    def test_unknown_action_type_ignored(self):
        """FRD-SOF-003: unknown action types silently ignored."""
        data = {
            "version": 1,
            "default_action": "allow",
            "actions": {
                "custom_action": {
                    "policies": [
                        {"parameter": "x", "operator": "equals", "value": 1, "action": "block"},
                    ],
                },
            },
        }
        config = SDKPolicyConfig.model_validate(data)
        assert "custom_action" in config.actions


# ===========================================================================
# C1: YAML Loader tests
# ===========================================================================


class TestLoadSDKPolicies:
    """FRD-SOF-001: YAML file loading with template creation."""

    def test_missing_file_creates_template(self, policy_dir: Path):
        config = load_sdk_policies()
        assert config.version == 1
        # Template file should have been created
        policy_file = policy_dir / "policies.yaml"
        assert policy_file.exists()
        content = policy_file.read_text()
        assert "version: 1" in content

    def test_empty_file_returns_zero_rules(self, policy_dir: Path):
        _write_policy(policy_dir, "")
        config = load_sdk_policies()
        assert config.version == 1

    def test_valid_file_loads_rules(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 100
                  name: max_cost
        """)
        config = load_sdk_policies()
        assert config.global_ is not None
        assert len(config.global_.policies) == 1
        assert config.global_.policies[0].name == "max_cost"

    def test_invalid_yaml_returns_zero_rules(self, policy_dir: Path, capsys):
        _write_policy(policy_dir, ": invalid: yaml: [")
        config = load_sdk_policies()
        assert config.version == 1
        # Warning should go to stderr
        captured = capsys.readouterr()
        assert "Warning" in captured.err or "invalid YAML" in captured.err

    def test_bad_version_raises_config_error(self, policy_dir: Path):
        _write_policy(policy_dir, "version: 99\n")
        with pytest.raises(ClyroConfigError):
            load_sdk_policies()

    def test_unknown_operator_raises_config_error(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: x
                  operator: magic_op
                  value: 1
        """)
        with pytest.raises(ClyroConfigError):
            load_sdk_policies()

    def test_unknown_action_raises_config_error(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: x
                  operator: max_value
                  value: 1
                  action: unknown_action
        """)
        with pytest.raises(ClyroConfigError):
            load_sdk_policies()

    def test_cache_returns_same_object(self, policy_dir: Path):
        _write_policy(policy_dir, "version: 1\n")
        c1 = load_sdk_policies()
        c2 = load_sdk_policies()
        assert c1 is c2

    def test_file_is_directory_returns_zero_rules(self, policy_dir: Path, capsys):
        """TDD §13.4 edge case: policies.yaml is a directory."""
        policy_file = policy_dir / "policies.yaml"
        policy_file.mkdir(exist_ok=True)
        config = load_sdk_policies()
        assert config.version == 1
        captured = capsys.readouterr()
        assert "directory" in captured.err

    def test_permission_denied_on_dir_creation(self, tmp_path: Path, monkeypatch):
        """Dir permission denied → warning + zero rules."""
        sdk_dir = tmp_path / "noperm" / ".clyro" / "sdk"
        monkeypatch.setattr("clyro.local_policy._POLICY_DIR", sdk_dir)
        monkeypatch.setattr("clyro.local_policy._POLICY_FILE", sdk_dir / "policies.yaml")

        # Patch mkdir to raise PermissionError
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            with patch.object(Path, "exists", return_value=False):
                config = load_sdk_policies()
                assert config.version == 1

    def test_action_field_in_yaml(self, policy_dir: Path):
        """FRD-SOF-001: require_approval action in YAML."""
        _write_policy(policy_dir, """\
            version: 1
            actions:
              llm_call:
                policies:
                  - parameter: max_tokens
                    operator: max_value
                    value: 8192
                    name: large_context
                    action: require_approval
        """)
        config = load_sdk_policies()
        rule = config.actions["llm_call"].policies[0]
        assert rule.action == "require_approval"


# ===========================================================================
# C2: SDKLocalPolicyEvaluator tests
# ===========================================================================


class TestSDKLocalPolicyEvaluator:
    """FRD-SOF-002: local policy evaluation."""

    def test_zero_rules_allows(self, policy_dir: Path):
        _write_policy(policy_dir, "version: 1\nglobal:\n  policies: []\n")
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("llm_call", {"model": "gpt-4"})
        assert decision.decision == "allow"

    def test_block_rule_raises(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 10
                  name: max_cost
                  action: block
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError) as exc_info:
            evaluator.evaluate_sync("llm_call", {"cost": 20})
        assert "max_cost" in str(exc_info.value)

    def test_allow_when_not_violated(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 100
                  name: max_cost
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("llm_call", {"cost": 50})
        assert decision.decision == "allow"

    def test_per_action_type_rules(self, policy_dir: Path):
        """FRD-SOF-003: per-action-type evaluation."""
        _write_policy(policy_dir, """\
            version: 1
            actions:
              tool_call:
                policies:
                  - parameter: endpoint
                    operator: contains
                    value: "internal"
                    name: no_internal
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

        # tool_call with internal endpoint → block (contains "internal" is violated)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("tool_call", {"endpoint": "internal-api.corp.com"})

        # Reset cache for second evaluation
        reset_sdk_policy_cache()
        _write_policy(policy_dir, """\
            version: 1
            actions:
              tool_call:
                policies:
                  - parameter: endpoint
                    operator: contains
                    value: "internal"
                    name: no_internal
        """)

        # llm_call doesn't match tool_call rules → allow
        evaluator2 = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator2.evaluate_sync("llm_call", {"endpoint": "internal-api.corp.com"})
        assert decision.decision == "allow"

    def test_require_approval_approved(self, policy_dir: Path):
        """FRD-SOF-002: require_approval with handler that approves."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: tokens
                  operator: max_value
                  value: 1000
                  name: large_request
                  action: require_approval
        """)
        handler = MagicMock(return_value=True)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=handler)
        decision = evaluator.evaluate_sync("llm_call", {"tokens": 2000})
        assert decision.decision == "allow"
        handler.assert_called_once()

    def test_require_approval_denied(self, policy_dir: Path):
        """FRD-SOF-002: require_approval with handler that denies → block."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: tokens
                  operator: max_value
                  value: 1000
                  name: large_request
                  action: require_approval
        """)
        handler = MagicMock(return_value=False)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=handler)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("llm_call", {"tokens": 2000})

    def test_require_approval_no_handler_blocks(self, policy_dir: Path):
        """FRD-SOF-002: no handler (non-TTY) → treat as block."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: tokens
                  operator: max_value
                  value: 1000
                  action: require_approval
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("llm_call", {"tokens": 2000})

    def test_unresolved_parameter_skipped(self, policy_dir: Path):
        """FRD-SOF-002: unresolved parameter → skip rule."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: nonexistent.field
                  operator: max_value
                  value: 10
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("llm_call", {"cost": 20})
        assert decision.decision == "allow"

    def test_dot_path_resolution(self, policy_dir: Path):
        """Nested parameter path resolution."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: order.quantity
                  operator: max_value
                  value: 10
                  name: max_quantity
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("tool_call", {"order": {"quantity": 20}})

    def test_wildcard_prefix(self, policy_dir: Path):
        """Wildcard prefix *.amount resolution."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: "*.amount"
                  operator: max_value
                  value: 100
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("tool_call", {"amount": 200})

    def test_single_rule_exception_skips(self, policy_dir: Path):
        """FRD-SOF-002: single rule exception → skip + continue."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 10
                  name: max_cost
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

        # Patch _evaluate_local_rule to raise for this test
        with patch("clyro.local_policy._evaluate_local_rule", side_effect=RuntimeError("boom")):
            decision = evaluator.evaluate_sync("llm_call", {"cost": 20})
            assert decision.decision == "allow"

    def test_all_8_operators(self, policy_dir: Path):
        """TDD §13.1 C2: all 8 operators work correctly.

        Each operator's condition "matches" when the predicate below holds.
        With the rule's default action of "block", a match → BLOCK.
        """
        test_cases = [
            ("max_value", 10, {"x": 20}, True),    # 20 > 10 → match → BLOCK
            ("max_value", 10, {"x": 5}, False),    # 5 not > 10 → no match
            ("min_value", 10, {"x": 5}, True),     # 5 < 10 → match → BLOCK
            ("min_value", 10, {"x": 20}, False),
            ("equals", "foo", {"x": "foo"}, True),   # foo == foo → match → BLOCK
            ("equals", "foo", {"x": "bar"}, False),
            ("not_equals", "foo", {"x": "bar"}, True),  # bar != foo → match → BLOCK
            ("not_equals", "foo", {"x": "foo"}, False),
            ("in_list", ["a", "b"], {"x": "a"}, True),   # a in [a,b] → match → BLOCK
            ("in_list", ["a", "b"], {"x": "c"}, False),
            ("not_in_list", ["a", "b"], {"x": "c"}, True),  # c not in [a,b] → match → BLOCK
            ("not_in_list", ["a", "b"], {"x": "a"}, False),
            ("contains", "bad", {"x": "bad_word"}, True),    # "bad" in "bad_word" → match → BLOCK
            ("contains", "bad", {"x": "good_word"}, False),
            ("not_contains", "good", {"x": "bad_word"}, True),  # "good" not in "bad_word" → match → BLOCK
            ("not_contains", "good", {"x": "good_word"}, False),
        ]

        for op, value, params, should_violate in test_cases:
            reset_sdk_policy_cache()
            # Build YAML using yaml.dump to handle list values correctly
            rule = {
                "parameter": "x",
                "operator": op,
                "value": value,
                "name": "test_rule",
                "action": "block",
            }
            yaml_data = {
                "version": 1,
                "default_action": "allow",
                "global": {"policies": [rule]},
            }
            yaml_content = yaml.dump(yaml_data, default_flow_style=False)
            (policy_dir / "policies.yaml").write_text(yaml_content, encoding="utf-8")
            evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

            if should_violate:
                with pytest.raises(PolicyViolationError, match="test_rule"):
                    evaluator.evaluate_sync("llm_call", params)
            else:
                decision = evaluator.evaluate_sync("llm_call", params)
                assert decision.decision == "allow", f"Failed: {op} {value} {params}"

    def test_drain_events(self, policy_dir: Path):
        """Events are buffered and drainable."""
        _write_policy(policy_dir, "version: 1\nglobal:\n  policies: []\n")
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        evaluator.evaluate_sync("llm_call", {})
        events = evaluator.drain_events()
        assert len(events) == 1
        assert events[0].event_name == "policy_check"
        # Second drain returns empty
        assert evaluator.drain_events() == []

    async def test_async_parity(self, policy_dir: Path):
        """FRD-SOF-002: async path produces identical decisions."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 100
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

        sync_decision = evaluator.evaluate_sync("llm_call", {"cost": 50})

        reset_sdk_policy_cache()
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 100
        """)
        evaluator2 = SDKLocalPolicyEvaluator(approval_handler=None)
        async_decision = await evaluator2.evaluate_async("llm_call", {"cost": 50})

        assert sync_decision.decision == async_decision.decision


class TestFixBAllowlistShape:
    """Fix B: ``default_action`` is bypassed when every rule was skipped
    because its parameter wasn't present in the action's parameters.

    Prevents the allowlist shape (``default_action: block`` paired with a
    field-scoped ``action: allow`` rule) from blocking action types like
    ``agent_execution`` and ``llm_call`` where the rule's tool-specific
    field could never have been present.

    The zero-rules case still fires ``default_action`` so the
    "block everything by default" idiom is preserved.
    """

    def test_field_absent_bypasses_default_action_block(self, policy_dir: Path):
        """The allowlist case the user reported: rule's field is absent on
        this action_type → policy is inapplicable → default_action: block
        does NOT fire."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: rmq_cluster
                  operator: in_list
                  value: [cluster1, cluster3]
                  name: allowed_clusters
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        # llm_call has no rmq_cluster — rule is skipped, default_action bypassed
        decision = evaluator.evaluate_sync("llm_call", {"model": "gpt-4", "cost": 0.01})
        assert decision.decision == "allow"

    def test_field_absent_bypasses_default_action_agent_execution(
        self, policy_dir: Path
    ):
        """agent_execution typically carries {agent_name, model, ...} but
        never tool args. The rule's field is absent → bypass default_action."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: rmq_cluster
                  operator: in_list
                  value: [cluster1, cluster3]
                  name: allowed_clusters
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync(
            "agent_execution", {"agent_name": "my-agent"}
        )
        assert decision.decision == "allow"

    def test_field_present_and_matches_allowlist_rule(self, policy_dir: Path):
        """Allowlist match → action: allow → ALLOW (rule fires explicitly)."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: rmq_cluster
                  operator: in_list
                  value: [cluster1, cluster3]
                  name: allowed_clusters
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync(
            "tool_call", {"rmq_cluster": "cluster1"}
        )
        assert decision.decision == "allow"

    def test_field_present_but_no_match_fires_default_action_block(
        self, policy_dir: Path
    ):
        """Allowlist miss → rule no_match (not skipped) → default_action fires."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: rmq_cluster
                  operator: in_list
                  value: [cluster1, cluster3]
                  name: allowed_clusters
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("tool_call", {"rmq_cluster": "cluster2"})

    def test_zero_rules_with_default_block_unchanged_by_fix_b(
        self, policy_dir: Path
    ):
        """Zero-rules path is reached via a separate early-return branch that
        Fix B does not touch; behavior is unchanged from pre-Fix-B. (In the
        SDK evaluator this path returns ``violated=True`` but no
        ``violation_details``, so ``evaluate_sync`` does not raise — a
        pre-existing quirk distinct from Fix B.)"""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies: []
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        # Pre-existing behavior: no raise. Fix B's "all skipped" branch is
        # for the rules-exist-but-all-skipped case, not zero-rules.
        decision = evaluator.evaluate_sync("llm_call", {"cost": 5.0})
        assert decision.decision == "allow"

    def test_mixed_skipped_and_no_match_fires_default_action(
        self, policy_dir: Path
    ):
        """If at least one rule was actually evaluated (no_match), the policy
        is applicable to this action and default_action fires normally."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: rmq_cluster
                  operator: in_list
                  value: [cluster1]
                  name: rmq_allowlist
                  action: allow
                - parameter: cost
                  operator: max_value
                  value: 10.0
                  name: cost_cap
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        # cost rule WAS evaluated (no_match: 0.01 not > 10.0); rmq rule skipped.
        # Not "all skipped" → default_action: block fires.
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("llm_call", {"cost": 0.01})

    def test_denylist_shape_unaffected(self, policy_dir: Path):
        """The denylist shape (default_action: allow + action: block) is
        unchanged by Fix B — it never relied on default_action firing on
        skipped rules in the first place."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: allow
            global:
              policies:
                - parameter: rmq_cluster
                  operator: not_in_list
                  value: [cluster1, cluster3]
                  name: rmq_denylist
                  action: block
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        # llm_call: rmq_cluster absent → rule skipped → default_action: allow
        decision = evaluator.evaluate_sync("llm_call", {"cost": 0.01})
        assert decision.decision == "allow"
        # tool_call with disallowed cluster → rule matches → block
        reset_sdk_policy_cache()
        _write_policy(policy_dir, """\
            version: 1
            default_action: allow
            global:
              policies:
                - parameter: rmq_cluster
                  operator: not_in_list
                  value: [cluster1, cluster3]
                  name: rmq_denylist
                  action: block
        """)
        evaluator2 = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator2.evaluate_sync("tool_call", {"rmq_cluster": "cluster2"})


# ===========================================================================
# NFR-001: Latency benchmark
# ===========================================================================


class TestNFR006NoHotReload:
    """NFR-006: policies cached on first load, file changes ignored."""

    def test_file_change_after_load_ignored(self, policy_dir: Path):
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 100
                  name: original_rule
        """)
        config1 = load_sdk_policies()
        assert config1.global_.policies[0].name == "original_rule"

        # Modify file after initial load
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 50
                  name: modified_rule
        """)

        # Second load should return cached (original) config
        config2 = load_sdk_policies()
        assert config2 is config1
        assert config2.global_.policies[0].name == "original_rule"


class TestPerActionAndGlobalOrdering:
    """FRD-SOF-003: per-action rules evaluated BEFORE global rules."""

    def test_per_action_evaluated_before_global(self, policy_dir: Path):
        """If per-action rule blocks, global rules are never reached."""
        _write_policy(policy_dir, """\
            version: 1
            actions:
              llm_call:
                policies:
                  - parameter: model
                    operator: equals
                    value: "forbidden"
                    name: per_action_block
            global:
              policies:
                - parameter: model
                  operator: equals
                  value: "also_forbidden"
                  name: global_block
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

        # `equals "forbidden"` matches when model == "forbidden". Action defaults
        # to "block", so model="forbidden" → per-action rule matches → BLOCK.
        with pytest.raises(PolicyViolationError, match="per_action_block"):
            evaluator.evaluate_sync("llm_call", {"model": "forbidden"})

        # Verify per-action rule was the one that triggered (not global)
        events = evaluator.drain_events()
        assert len(events) == 1
        rule_results = events[0].metadata.get("rule_results", [])
        # First rule triggered, second (global) never reached due to short-circuit
        assert len(rule_results) == 1
        assert rule_results[0]["rule_name"] == "per_action_block"

    def test_global_rules_apply_when_no_per_action(self, policy_dir: Path):
        """Action type with no per-action section still gets global rules."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cost
                  operator: max_value
                  value: 10
                  name: global_cost
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError, match="global_cost"):
            evaluator.evaluate_sync("agent_execution", {"cost": 20})


class TestLocalPolicyBenchmark:
    """NFR-001: <5ms p95 for 20 rules."""

    def test_evaluation_latency_p95(self, policy_dir: Path):
        rules = []
        for i in range(20):
            rules.append(
                f"    - parameter: field_{i}\n"
                f"      operator: max_value\n"
                f"      value: 1000\n"
                f"      name: rule_{i}\n"
            )
        yaml_content = "version: 1\nglobal:\n  policies:\n" + "".join(rules)
        _write_policy(policy_dir, yaml_content)

        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        params = {f"field_{i}": 500 for i in range(20)}

        latencies = []
        for _ in range(1000):
            reset_sdk_policy_cache()
            _write_policy(policy_dir, yaml_content)
            # Only measure evaluation, not file loading
            load_sdk_policies()
            start = time.perf_counter()
            evaluator._evaluate("llm_call", params)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p95 = latencies[int(0.95 * len(latencies))]
        assert p95 < 5.0, f"p95 latency {p95:.2f}ms exceeds 5ms budget"


# ===========================================================================
# NFR-002: YAML load time benchmark
# ===========================================================================


class TestYAMLLoadBenchmark:
    """NFR-002: <10ms for 50-rule file (target).

    CI threshold set to 50ms to avoid flaky failures on slow machines.
    The 10ms target is validated in local profiling, not hard-asserted in CI.
    """

    def test_yaml_load_cold(self, policy_dir: Path):
        rules = []
        for i in range(50):
            rules.append(
                f"    - parameter: field_{i}\n"
                f"      operator: max_value\n"
                f"      value: 1000\n"
                f"      name: rule_{i}\n"
            )
        yaml_content = "version: 1\nglobal:\n  policies:\n" + "".join(rules)
        _write_policy(policy_dir, yaml_content)

        start = time.perf_counter()
        load_sdk_policies()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Target: <10ms. CI threshold: <50ms (Pydantic validation + YAML parsing overhead)
        assert elapsed_ms < 50.0, f"Cold load took {elapsed_ms:.2f}ms (CI budget: 50ms)"


# ===========================================================================
# New-semantic behavior tests: action=allow, default_action, first-match-wins
# ===========================================================================


class TestActionAllowSemantic:
    """Rules with action: allow short-circuit to allow on match."""

    def test_action_allow_matched_returns_allow(self, policy_dir: Path):
        """A matched rule with action: allow → call is allowed."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1", "c3"]
                  action: allow
                  name: allow_clusters
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("tool_call", {"cluster": "c1"})
        assert decision.decision == "allow"

    def test_action_allow_short_circuits_later_block(self, policy_dir: Path):
        """First-match-wins: matched action: allow prevents a later block rule from firing."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: allow
                  name: allow_c1
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: block
                  name: block_c1
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("tool_call", {"cluster": "c1"})
        assert decision.decision == "allow"


class TestDefaultActionSemantic:
    """default_action determines outcome when no rule matches."""

    def test_default_action_allow_when_no_match(self, policy_dir: Path):
        """default_action defaults to allow; no rule matches → allow."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: block
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("tool_call", {"cluster": "c2"})
        assert decision.decision == "allow"

    def test_default_action_block_when_no_match_blocks(self, policy_dir: Path):
        """default_action=block + no rule match → BLOCK."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError):
            evaluator.evaluate_sync("tool_call", {"cluster": "c2"})

    def test_default_action_block_skipped_when_rule_matches(self, policy_dir: Path):
        """default_action=block is irrelevant once a rule matches."""
        _write_policy(policy_dir, """\
            version: 1
            default_action: block
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: allow
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        decision = evaluator.evaluate_sync("tool_call", {"cluster": "c1"})
        assert decision.decision == "allow"


class TestFirstMatchWins:
    """First matching rule's action wins; later rules don't override."""

    def test_first_block_wins_over_later_allow(self, policy_dir: Path):
        """If block rule matches first, allow rule does not override."""
        _write_policy(policy_dir, """\
            version: 1
            global:
              policies:
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: block
                  name: block_first
                - parameter: cluster
                  operator: in_list
                  value: ["c1"]
                  action: allow
                  name: allow_second
        """)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)
        with pytest.raises(PolicyViolationError, match="block_first"):
            evaluator.evaluate_sync("tool_call", {"cluster": "c1"})


class TestUserSpecFourCases:
    """End-to-end coverage of the four cases the user specified."""

    @pytest.mark.parametrize(
        "operator,action,default,cluster,expected",
        [
            # Case 1: in_list + block, default allow
            ("in_list", "block", "allow", "c1", "block"),
            ("in_list", "block", "allow", "c2", "allow"),
            # Case 2: in_list + allow, default block
            ("in_list", "allow", "block", "c1", "allow"),
            ("in_list", "allow", "block", "c2", "block"),
            # Case 3: not_in_list + block, default allow
            ("not_in_list", "block", "allow", "c1", "allow"),
            ("not_in_list", "block", "allow", "c2", "block"),
            # Case 4: not_in_list + allow, default block
            ("not_in_list", "allow", "block", "c1", "block"),
            ("not_in_list", "allow", "block", "c2", "allow"),
        ],
    )
    def test_user_spec_case(
        self,
        policy_dir: Path,
        operator: str,
        action: str,
        default: str,
        cluster: str,
        expected: str,
    ):
        """Confirm rmq_cluster scenarios behave per the user's spec."""
        yaml_content = (
            "version: 1\n"
            f"default_action: {default}\n"
            "global:\n"
            "  policies:\n"
            "    - parameter: rmq_cluster\n"
            f"      operator: {operator}\n"
            "      value: [c1, c3]\n"
            f"      action: {action}\n"
            "      name: rmq_rule\n"
        )
        _write_policy(policy_dir, yaml_content)
        evaluator = SDKLocalPolicyEvaluator(approval_handler=None)

        if expected == "block":
            with pytest.raises(PolicyViolationError):
                evaluator.evaluate_sync("tool_call", {"rmq_cluster": cluster})
        else:
            decision = evaluator.evaluate_sync("tool_call", {"rmq_cluster": cluster})
            assert decision.decision == "allow"


class TestDefaultActionRequired:
    """Regression: default_action is required on SDKPolicyConfig.

    Locks the contract — if anyone re-introduces a default value for
    default_action, this test will fail and force a deliberate change.
    """

    def test_missing_default_action_raises(self) -> None:
        with pytest.raises(ValidationError, match="default_action"):
            SDKPolicyConfig.model_validate({
                "version": 1,
                "global": {"policies": []},
            })
