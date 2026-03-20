"""
Unit tests for PreventionStack — TDD §11.1 tests #6–#8, #35.
"""

from __future__ import annotations

from clyro.config import WrapperConfig
from clyro.mcp.prevention import AllowDecision, BlockDecision, PreventionStack
from clyro.mcp.session import McpSession


class TestStepLimit:
    """Step limit enforcement — TDD §11.1 #6, #7."""

    def test_allows_under_limit(self) -> None:
        """TDD §11.1 #6 — calls 1-50 with max_steps=50 -> all allowed."""
        cfg = WrapperConfig.model_validate({"global": {"max_steps": 50}})
        ps = PreventionStack(cfg)
        s = McpSession()
        for i in range(50):
            result = ps.evaluate(f"tool_{i}", {"i": i}, s)
            assert isinstance(result, AllowDecision)

    def test_blocks_at_limit(self) -> None:
        """TDD §11.1 #7 — call 51 with max_steps=50 -> blocked."""
        cfg = WrapperConfig.model_validate({"global": {"max_steps": 50}})
        ps = PreventionStack(cfg)
        s = McpSession()
        for i in range(50):
            ps.evaluate(f"tool_{i}", {"i": i}, s)
        result = ps.evaluate("tool_51", {}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "step_limit_exceeded"
        assert result.details["step_count"] == 51
        assert result.details["max_steps"] == 50


class TestStepCounterIncludesBlocked:
    """TDD §11.1 #8 — blocked call still increments step counter."""

    def test_blocked_increments_step(self) -> None:
        cfg = WrapperConfig.model_validate({"global": {"max_steps": 1}})
        ps = PreventionStack(cfg)
        s = McpSession()
        # First call allowed (step=1, which is <= 1)
        r1 = ps.evaluate("t", {}, s)
        assert isinstance(r1, AllowDecision)
        assert s.step_count == 1
        # Second call blocked (step=2 > 1)
        r2 = ps.evaluate("t", {}, s)
        assert isinstance(r2, BlockDecision)
        assert s.step_count == 2  # still incremented


class TestPreventionShortCircuit:
    """TDD §11.1 #35 — loop detected -> policy eval never called."""

    def test_short_circuit_on_loop(self) -> None:
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "loop_detection": {"threshold": 2, "window": 10},
                    "policies": [
                        {"parameter": "x", "operator": "max_value", "value": 999},
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        # First call
        ps.evaluate("t", {"x": 1}, s)
        # Second call triggers loop before policy can fire
        result = ps.evaluate("t", {"x": 1}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "loop_detected"


class TestPreventionFullPipeline:
    """Full pipeline: allow through all 4 stages."""

    def test_allow_all_stages(self) -> None:
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_steps": 100,
                    "max_cost_usd": 10.0,
                    "loop_detection": {"threshold": 3, "window": 10},
                    "policies": [
                        {"parameter": "amount", "operator": "max_value", "value": 1000},
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        result = ps.evaluate("transfer", {"amount": 100}, s)
        assert isinstance(result, AllowDecision)
        assert result.step_number == 1

    def test_budget_block(self) -> None:
        """CostTracker blocks when budget is tight."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_cost_usd": 0.0001,
                    "cost_per_token_usd": 0.01,
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        result = ps.evaluate("t", {"data": "x" * 1000}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "budget_exceeded"


class TestSessionContextEnrichment:
    """Session context fields injected into policy evaluation arguments."""

    def test_cost_policy_triggers_on_session_cost(self) -> None:
        """Policy with field=cost blocks when session cost exceeds threshold."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_steps": 100,
                    "max_cost_usd": 10.0,
                    "policies": [
                        {
                            "parameter": "cost",
                            "operator": "max_value",
                            "value": 0.00005,
                            "name": "cost_limit",
                        },
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        s.accumulated_cost_usd = 0.0001  # exceeds 0.00005

        result = ps.evaluate("some_tool", {"site_name": "test"}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "policy_violation"
        assert result.details["rule_name"] == "cost_limit"
        assert result.details["actual"] == 0.0001

    def test_step_number_policy_triggers(self) -> None:
        """Policy with field=step_number and min_value blocks when step < threshold."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_steps": 100,
                    "max_cost_usd": 10.0,
                    "policies": [
                        {
                            "parameter": "step_number",
                            "operator": "min_value",
                            "value": 20,
                            "name": "steps_threshold",
                        },
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()

        # First call: step=1 < 20 → blocked
        result = ps.evaluate("tool_a", {}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "policy_violation"
        assert result.details["rule_name"] == "steps_threshold"

    def test_tool_argument_not_overwritten(self) -> None:
        """Real tool argument takes priority over session context."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_steps": 100,
                    "max_cost_usd": 10.0,
                    "policies": [
                        {
                            "parameter": "cost",
                            "operator": "max_value",
                            "value": 50,
                            "name": "cost_cap",
                        },
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        s.accumulated_cost_usd = 0.001  # session cost is low

        # Tool argument "cost" = 100, which exceeds max_value of 50
        result = ps.evaluate("tool_a", {"cost": 100}, s)
        assert isinstance(result, BlockDecision)
        assert result.block_type == "policy_violation"
        assert result.details["actual"] == 100  # tool arg, not session cost

    def test_session_context_available_when_arg_missing(self) -> None:
        """Session context fills in when tool args don't have the field."""
        cfg = WrapperConfig.model_validate(
            {
                "global": {
                    "max_steps": 100,
                    "max_cost_usd": 10.0,
                    "policies": [
                        {
                            "parameter": "cost",
                            "operator": "max_value",
                            "value": 0.01,
                            "name": "cost_limit",
                        },
                    ],
                }
            }
        )
        ps = PreventionStack(cfg)
        s = McpSession()
        s.accumulated_cost_usd = 0.005  # under 0.01

        # Tool args don't contain "cost" — session context fills it in
        result = ps.evaluate("tool_a", {"site_name": "test"}, s)
        assert isinstance(result, AllowDecision)  # 0.005 <= 0.01 → allowed
