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
    p = policy_dir / "policies.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
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

    def test_default_action_is_block(self):
        rule = SDKPolicyRule(
            parameter="cost", operator="max_value", value=100,
        )
        assert rule.action == "block"

    def test_invalid_action_raises(self):
        with pytest.raises(ValidationError):
            SDKPolicyRule(
                parameter="cost", operator="max_value", value=100,
                action="unknown_value",
            )

    def test_inherits_operator_validation(self):
        with pytest.raises(ValidationError):
            SDKPolicyRule(
                parameter="cost", operator="invalid_op", value=100,
            )

    def test_extra_fields_ignored(self):
        """SDKPolicyRule has extra='ignore' — future MCP fields accepted."""
        rule = SDKPolicyRule(
            parameter="cost", operator="max_value", value=100,
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
            rule = SDKPolicyRule(parameter="x", operator=op, value=1)
            assert rule.operator == op


# ===========================================================================
# C1: SDKPolicyConfig model tests
# ===========================================================================


class TestSDKPolicyConfig:
    """FRD-SOF-001, FRD-SOF-003: YAML schema validation."""

    def test_valid_config(self):
        config = SDKPolicyConfig(version=1)
        assert config.version == 1

    def test_wrong_version_raises(self):
        with pytest.raises(ValidationError):
            SDKPolicyConfig(version=2)

    def test_global_alias(self):
        """'global' YAML key maps to global_ Python field."""
        data = {"version": 1, "global": {"policies": []}}
        config = SDKPolicyConfig.model_validate(data)
        assert config.global_ is not None
        assert config.global_.policies == []

    def test_actions_with_known_types(self):
        data = {
            "version": 1,
            "actions": {
                "llm_call": {
                    "policies": [
                        {"parameter": "model", "operator": "in_list",
                         "value": ["gpt-4"], "name": "test"},
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
            "actions": {
                "custom_action": {
                    "policies": [
                        {"parameter": "x", "operator": "equals", "value": 1},
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
        """TDD §13.1 C2: all 8 operators work correctly."""
        test_cases = [
            ("max_value", 10, {"x": 20}, True),   # 20 > 10 → violated
            ("max_value", 10, {"x": 5}, False),    # 5 <= 10 → ok
            ("min_value", 10, {"x": 5}, True),     # 5 < 10 → violated
            ("min_value", 10, {"x": 20}, False),
            ("equals", "foo", {"x": "bar"}, True), # bar != foo → violated
            ("equals", "foo", {"x": "foo"}, False),
            ("not_equals", "foo", {"x": "foo"}, True),  # foo == foo → violated
            ("not_equals", "foo", {"x": "bar"}, False),
            ("in_list", ["a", "b"], {"x": "c"}, True),  # c not in [a,b] → violated
            ("in_list", ["a", "b"], {"x": "a"}, False),
            ("not_in_list", ["a", "b"], {"x": "a"}, True),  # a in [a,b] → violated
            ("not_in_list", ["a", "b"], {"x": "c"}, False),
            ("contains", "bad", {"x": "bad_word"}, True),    # "bad" in "bad_word" → violated
            ("contains", "bad", {"x": "good_word"}, False),
            ("not_contains", "good", {"x": "bad_word"}, True),  # "good" not in "bad_word" → violated
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
            }
            yaml_data = {
                "version": 1,
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

        # "forbidden" triggers per-action rule (not_equals: "forbidden" != model -> violated? No.
        # equals: model != "forbidden" -> violated. So model="forbidden" -> not violated.
        # Actually equals returns True (violated) when actual != expected.
        # So model="forbidden" with equals/"forbidden" -> actual == expected -> NOT violated.
        # model="other" with equals/"forbidden" -> actual != expected -> violated.

        # Use a model that violates the per-action rule
        with pytest.raises(PolicyViolationError, match="per_action_block"):
            evaluator.evaluate_sync("llm_call", {"model": "other"})

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
