# Integration Tests for Clyro SDK Execution Controls
# Implements PRD-009, PRD-010, PRD-NFR-001

"""
Integration tests for execution controls including step limits, cost limits,
and loop detection working together with the session and wrapper.
"""

from dataclasses import dataclass
from decimal import Decimal

import pytest

from clyro import wrap
from clyro.config import ClyroConfig, ExecutionControls
from conftest import TEST_ORG_ID
from clyro.cost import CostCalculator, TokenUsage
from clyro.exceptions import (
    CostLimitExceededError,
    LoopDetectedError,
    StepLimitExceededError,
)
from clyro.session import Session, get_current_session


class TestSessionWithCostCalculator:
    """Integration tests for Session with cost calculator."""

    def test_record_llm_call_with_response(self):
        """Test recording LLM call with automatic cost extraction."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()

        # Simulate OpenAI response
        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi there!"},
            llm_response=response,
            duration_ms=150,
        )

        # Verify cost was calculated
        # gpt-4-turbo: (1000 * 0.01 + 500 * 0.03) / 1000 = 0.025
        assert session.cumulative_cost == Decimal("0.025")
        assert event.token_count_input == 1000
        assert event.token_count_output == 500
        assert event.cost_usd == Decimal("0.025")

    def test_record_llm_call_with_explicit_tokens(self):
        """Test recording LLM call with explicit token counts."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()

        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi there!"},
            input_tokens=500,
            output_tokens=250,
            duration_ms=100,
        )

        # Verify cost was calculated
        # gpt-4-turbo: (500 * 0.01 + 250 * 0.03) / 1000 = 0.0125
        assert session.cumulative_cost == Decimal("0.0125")
        assert event.token_count_input == 500
        assert event.token_count_output == 250

    def test_record_llm_call_no_cost_data(self):
        """Test recording LLM call without explicit cost data (uses estimation)."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()

        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            duration_ms=100,
        )

        # SDK estimates cost from input text when cost limit is enabled
        # "Hello" is ~6 tokens for gpt-4-turbo at $0.01/1k input tokens = $0.00006
        assert session.cumulative_cost > Decimal("0")
        assert event.cost_usd > Decimal("0")
        assert event.token_count_input > 0

    def test_multiple_llm_calls_accumulate_cost(self):
        """Test that multiple LLM calls accumulate cost correctly."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()

        # First call
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "First"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Second call
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Second"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Total cost should be 2 * 0.025 = 0.05
        assert session.cumulative_cost == Decimal("0.05")
        assert session.step_number == 2


class TestCostLimitEnforcement:
    """Integration tests for cost limit enforcement."""

    def test_cost_limit_triggered_by_llm_calls(self):
        """Test that cost limit stops execution after LLM calls."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(max_cost_usd=0.04),  # $0.04 limit
        )
        session = Session(config=config)
        session.start()

        # First call: ~$0.025
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "First"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Second call: should trigger limit (~$0.05 total, which is >$0.04)
        with pytest.raises(CostLimitExceededError) as exc_info:
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Second"},
                input_tokens=1000,
                output_tokens=500,
            )

        assert exc_info.value.limit_usd == 0.04
        assert exc_info.value.current_cost_usd > 0.04

    def test_cost_limit_with_expensive_model(self):
        """Test cost limit with expensive model (Claude-3-Opus)."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(max_cost_usd=0.10),
        )
        session = Session(config=config)
        session.start()

        # Claude-3-Opus is expensive: $0.015/$0.075 per 1K tokens
        # 1000 input + 500 output = (1000 * 0.015 + 500 * 0.075) / 1000 = 0.0525
        session.record_llm_call(
            model="claude-3-opus",
            input_data={"prompt": "Test"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Second call should exceed limit
        with pytest.raises(CostLimitExceededError):
            session.record_llm_call(
                model="claude-3-opus",
                input_data={"prompt": "Test 2"},
                input_tokens=1000,
                output_tokens=500,
            )

    def test_cost_limit_disabled(self):
        """Test that disabled cost limit allows unlimited spending."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_cost_usd=0.01,  # Very low limit
                enable_cost_limit=False,  # But disabled
            ),
        )
        session = Session(config=config)
        session.start()

        # Should not raise even with high cost
        for _ in range(10):
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Test"},
                input_tokens=1000,
                output_tokens=500,
            )

        assert session.cumulative_cost == Decimal("0.25")


class TestStepLimitWithLLMCalls:
    """Integration tests for step limit with LLM calls."""

    def test_step_limit_counts_llm_calls(self):
        """Test that LLM calls count toward step limit."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(max_steps=3),
        )
        session = Session(config=config)
        session.start()

        # Three calls should succeed
        for i in range(3):
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": f"Call {i}"},
            )

        # Fourth should fail
        with pytest.raises(StepLimitExceededError) as exc_info:
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Call 4"},
            )

        assert exc_info.value.limit == 3
        assert exc_info.value.current_step == 4


class TestWrappedAgentExecutionControls:
    """Integration tests for wrapped agent execution controls."""

    def test_wrapped_agent_respects_step_limit(self):
        """Test that wrapped agent respects step limit."""
        # Note: The wrapper records 1 step when it calls the agent,
        # so we need to account for wrapper overhead in our limit
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=True,
            controls=ExecutionControls(max_steps=6),  # 1 for wrapper + 5 for agent
        )

        step_count = 0

        @wrap(config=config, org_id=TEST_ORG_ID)
        def counting_agent(limit: int) -> int:
            nonlocal step_count
            session = get_current_session()
            for i in range(limit):
                session.record_step(event_name=f"step_{i}")
                step_count = i + 1
            return step_count

        # Should complete within limit (5 agent steps + 1 wrapper step = 6)
        result = counting_agent(limit=5)
        assert result == 5

        step_count = 0

        # Configure for fewer steps to trigger limit
        config2 = ClyroConfig(
            agent_name="test-agent",
            fail_open=True,
            controls=ExecutionControls(max_steps=3),  # 1 for wrapper + 2 allowed
        )

        @wrap(config=config2, org_id=TEST_ORG_ID)
        def counting_agent_limited(limit: int) -> int:
            nonlocal step_count
            session = get_current_session()
            for i in range(limit):
                session.record_step(event_name=f"step_{i}")
                step_count = i + 1
            return step_count

        # Should fail when exceeding limit
        with pytest.raises(StepLimitExceededError):
            counting_agent_limited(limit=10)

    def test_wrapped_agent_cost_tracking(self):
        """Test that wrapped agent tracks costs correctly."""
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=True,
            controls=ExecutionControls(max_cost_usd=1.0),
        )

        @wrap(config=config, org_id=TEST_ORG_ID)
        def llm_calling_agent(calls: int) -> Decimal:
            session = get_current_session()

            for i in range(calls):
                session.record_llm_call(
                    model="gpt-4-turbo",
                    input_data={"prompt": f"Call {i}"},
                    input_tokens=100,
                    output_tokens=50,
                )

            return session.cumulative_cost

        total_cost = llm_calling_agent(calls=5)

        # Each call: (100 * 0.01 + 50 * 0.03) / 1000 = 0.0025
        # Total: 5 * 0.0025 = 0.0125
        assert total_cost == Decimal("0.0125")


class TestMixedExecutionControls:
    """Integration tests for multiple execution controls together."""

    def test_step_and_cost_limits_together(self):
        """Test that both step and cost limits are enforced."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_steps=10,
                max_cost_usd=0.05,
            ),
        )
        session = Session(config=config)
        session.start()

        # Make expensive calls - cost limit should trigger first
        with pytest.raises(CostLimitExceededError):
            for _ in range(10):
                session.record_llm_call(
                    model="gpt-4-turbo",
                    input_data={"prompt": "Test"},
                    input_tokens=1000,
                    output_tokens=500,
                )

    def test_loop_detection_with_cost_tracking(self):
        """Test loop detection works alongside cost tracking."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_steps=100,
                max_cost_usd=10.0,
                loop_detection_threshold=3,
            ),
        )
        session = Session(config=config)
        session.start()

        stuck_state = {"processing": True, "items": []}

        # Should detect loop and raise
        with pytest.raises(LoopDetectedError):
            for _ in range(10):
                session.record_step(
                    event_name="process",
                    state_snapshot=stuck_state,
                    cost_usd=Decimal("0.001"),
                )


class TestCustomPricingIntegration:
    """Integration tests for custom pricing models."""

    def test_custom_pricing_in_cost_calculation(self):
        """Test that custom pricing is used in cost calculations."""
        config = ClyroConfig(
            agent_name="test-agent",
            pricing={
                "custom-llm": {"input": 0.001, "output": 0.002},
            },
        )
        session = Session(config=config)
        session.start()

        session.record_llm_call(
            model="custom-llm",
            input_data={"prompt": "Test"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Cost = (1000 * 0.001 + 500 * 0.002) / 1000 = 0.002
        assert session.cumulative_cost == Decimal("0.002")

    def test_register_pricing_at_runtime(self):
        """Test registering custom pricing at runtime."""
        config = ClyroConfig(agent_name="test-agent")
        config.register_model_pricing("new-model", 0.005, 0.010)

        session = Session(config=config)
        session.start()

        session.record_llm_call(
            model="new-model",
            input_data={"prompt": "Test"},
            input_tokens=1000,
            output_tokens=1000,
        )

        # Cost = (1000 * 0.005 + 1000 * 0.01) / 1000 = 0.015
        assert session.cumulative_cost == Decimal("0.015")


class TestRealWorldScenarios:
    """Integration tests simulating real-world usage scenarios."""

    def test_multi_step_agent_workflow(self):
        """Test a multi-step agent workflow with various events."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_steps=20,
                max_cost_usd=1.0,
                loop_detection_threshold=5,
            ),
        )
        session = Session(config=config)
        session.start(input_data={"query": "What's the weather?"})

        # Step 1: Parse query
        session.record_step(
            event_name="parse_query",
            input_data={"raw_query": "What's the weather?"},
            output_data={"intent": "weather", "location": None},
        )

        # Step 2: LLM call to get location
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Extract location from: What's the weather?"},
            output_data={"location": "San Francisco"},
            input_tokens=50,
            output_tokens=20,
        )

        # Step 3: Tool call (no cost)
        session.record_step(
            event_name="weather_api_call",
            input_data={"location": "San Francisco"},
            output_data={"temp": 72, "conditions": "sunny"},
        )

        # Step 4: Format response with LLM
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Format weather data"},
            output_data={"response": "It's 72°F and sunny in San Francisco."},
            input_tokens=100,
            output_tokens=50,
        )

        # End session
        session.end(output_data={"response": "It's 72°F and sunny in San Francisco."})

        # Verify session state
        assert session.step_number == 4
        assert session.cumulative_cost > Decimal("0")
        assert len(session.events) == 6  # start + 4 steps + end

    def test_agent_hitting_budget(self):
        """Test an agent that hits its budget mid-workflow."""
        config = ClyroConfig(
            agent_name="test-agent",
            controls=ExecutionControls(
                max_cost_usd=0.03,  # Low budget
            ),
        )
        session = Session(config=config)
        session.start()

        # First call uses ~$0.025
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "First"},
            input_tokens=1000,
            output_tokens=500,
        )

        # Second call should hit budget
        with pytest.raises(CostLimitExceededError) as exc_info:
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Second"},
                input_tokens=500,
                output_tokens=200,
            )

        error = exc_info.value
        assert error.limit_usd == 0.03
        assert error.session_id == str(session.session_id)

    def test_session_summary_includes_cost(self):
        """Test that session summary includes cost information."""
        config = ClyroConfig(agent_name="test-agent")
        session = Session(config=config)
        session.start()

        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Test"},
            input_tokens=1000,
            output_tokens=500,
        )

        session.end()

        summary = session.get_summary()
        assert summary["cumulative_cost_usd"] == 0.025
        assert summary["step_count"] == 1
        assert summary["is_active"] is False
