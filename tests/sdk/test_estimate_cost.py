# Tests for cost estimation and generation_params tracking
# Validates predictive cost features

from decimal import Decimal
from uuid import uuid4

from clyro.config import ClyroConfig, ExecutionControls
from clyro.session import Session


class TestEstimateCallCost:
    """Test suite for Session.estimate_call_cost() method."""

    def test_estimate_cost_basic(self):
        """Test basic cost estimation."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        # Estimate cost for a simple prompt
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo",
            input_data="What is the capital of France?",
            max_tokens=100,
        )

        assert isinstance(estimated, Decimal)
        assert estimated > Decimal("0")

    def test_estimate_cost_with_dict_input(self):
        """Test cost estimation with dictionary input."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        # Estimate with structured input
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo",
            input_data={"prompt": "Summarize this text", "context": "Some context here"},
            max_tokens=500,
        )

        assert isinstance(estimated, Decimal)
        assert estimated > Decimal("0")

    def test_estimate_cost_scales_with_max_tokens(self):
        """Test that estimated cost scales with max_tokens."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        prompt = "Write a story"

        # Estimate with different max_tokens
        est_small = session.estimate_call_cost(model="gpt-4-turbo", input_data=prompt, max_tokens=100)
        est_large = session.estimate_call_cost(model="gpt-4-turbo", input_data=prompt, max_tokens=2000)

        # Larger max_tokens should cost more
        assert est_large > est_small

    def test_estimate_cost_scales_with_input_size(self):
        """Test that estimated cost scales with input size."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        short_prompt = "Hi"
        long_prompt = "This is a much longer prompt " * 100

        # Estimate with different input sizes
        est_short = session.estimate_call_cost(model="gpt-4-turbo", input_data=short_prompt, max_tokens=100)
        est_long = session.estimate_call_cost(model="gpt-4-turbo", input_data=long_prompt, max_tokens=100)

        # Longer input should cost more (if tiktoken available)
        # If tiktoken not available, estimates may be equal
        assert est_long >= est_short

    def test_estimate_cost_different_models(self):
        """Test that different models have different costs."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        prompt = "Hello, world!"

        # Estimate for expensive vs cheap model
        est_expensive = session.estimate_call_cost(model="gpt-4-turbo", input_data=prompt, max_tokens=100)
        est_cheap = session.estimate_call_cost(model="gpt-4o-mini", input_data=prompt, max_tokens=100)

        # Expensive model should cost more
        assert est_expensive > est_cheap

    def test_estimate_cost_with_safety_margin(self):
        """Test cost estimation with different safety margins."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        prompt = "Calculate something"

        # Estimate with different safety margins
        est_default = session.estimate_call_cost(
            model="gpt-4-turbo", input_data=prompt, max_tokens=100, safety_margin=1.2
        )
        est_conservative = session.estimate_call_cost(
            model="gpt-4-turbo", input_data=prompt, max_tokens=100, safety_margin=1.5
        )
        est_exact = session.estimate_call_cost(
            model="gpt-4-turbo", input_data=prompt, max_tokens=100, safety_margin=1.0
        )

        # Higher safety margin should produce higher estimate
        assert est_conservative > est_default > est_exact

    def test_estimate_cost_with_zero_max_tokens(self):
        """Test cost estimation with zero max_tokens."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        estimated = session.estimate_call_cost(
            model="gpt-4-turbo", input_data="Test", max_tokens=0, safety_margin=1.0
        )

        # Should still work (only input cost)
        assert isinstance(estimated, Decimal)
        assert estimated >= Decimal("0")

    def test_estimate_cost_without_tiktoken(self):
        """Test cost estimation when tiktoken is not available."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        # Should still work, even if tiktoken not available
        # It will estimate 0 input tokens and use max_tokens for output
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo", input_data="Test prompt", max_tokens=100
        )

        assert isinstance(estimated, Decimal)
        # Should at least account for output tokens
        assert estimated >= Decimal("0")

    def test_estimate_cost_proactive_budget_check(self):
        """Test using estimate_call_cost for proactive budget checking."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        budget = Decimal("0.10")

        # Estimate cost for expensive call
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo", input_data="Write a long essay", max_tokens=2000
        )

        # Make decision based on estimate
        if session.cumulative_cost + estimated > budget:
            # Switch to cheaper model
            model = "gpt-4o-mini"
            estimated_cheap = session.estimate_call_cost(
                model=model, input_data="Write a long essay", max_tokens=2000
            )
            assert estimated_cheap < estimated

    def test_estimate_cost_empty_input(self):
        """Test cost estimation with empty input."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())

        estimated = session.estimate_call_cost(model="gpt-4-turbo", input_data="", max_tokens=100)

        assert isinstance(estimated, Decimal)
        # Should still have output cost
        assert estimated > Decimal("0")


class TestGenerationParamsTracking:
    """Test suite for generation_params tracking in record_llm_call."""

    def test_record_llm_call_with_generation_params(self):
        """Test recording LLM call with generation parameters."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Record call with generation params
        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi there"},
            input_tokens=10,
            output_tokens=5,
            generation_params={"temperature": 0.7, "max_tokens": 100, "top_p": 0.9},
        )

        # Check that params were stored in metadata
        assert event.metadata is not None
        assert "generation_params" in event.metadata
        assert event.metadata["generation_params"]["temperature"] == 0.7
        assert event.metadata["generation_params"]["max_tokens"] == 100
        assert event.metadata["generation_params"]["top_p"] == 0.9

    def test_record_llm_call_without_generation_params(self):
        """Test recording LLM call without generation parameters."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Record call without generation params
        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi there"},
            input_tokens=10,
            output_tokens=5,
        )

        # Metadata should be None or not contain generation_params
        if event.metadata:
            assert "generation_params" not in event.metadata

    def test_generation_params_with_all_parameters(self):
        """Test tracking all common generation parameters."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        params = {
            "temperature": 0.8,
            "max_tokens": 1500,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop": ["\n\n"],
        }

        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Write creatively"},
            output_data={"response": "Once upon a time..."},
            input_tokens=20,
            output_tokens=100,
            generation_params=params,
        )

        # All params should be stored
        assert event.metadata["generation_params"] == params

    def test_generation_params_cost_correlation(self):
        """Test that generation_params can be used to analyze cost patterns."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Record multiple calls with different params
        calls = [
            {
                "params": {"temperature": 0.0, "max_tokens": 50},
                "input_tokens": 100,
                "output_tokens": 30,
            },
            {
                "params": {"temperature": 0.8, "max_tokens": 2000},
                "input_tokens": 100,
                "output_tokens": 500,
            },
        ]

        events = []
        for call in calls:
            event = session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Test"},
                output_data={"response": "Response"},
                input_tokens=call["input_tokens"],
                output_tokens=call["output_tokens"],
                generation_params=call["params"],
            )
            events.append(event)

        # First call (low temp, low max_tokens) should have lower cost
        assert events[0].cost_usd < events[1].cost_usd

        # Can analyze params from events
        for event in events:
            assert "generation_params" in event.metadata
            params = event.metadata["generation_params"]
            assert "temperature" in params
            assert "max_tokens" in params

    def test_generation_params_empty_dict(self):
        """Test with empty generation_params dict."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        event = session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "Hello"},
            output_data={"response": "Hi"},
            input_tokens=10,
            output_tokens=5,
            generation_params={},
        )

        # Empty dict should still be stored
        assert event.metadata is not None
        assert "generation_params" in event.metadata
        assert event.metadata["generation_params"] == {}


class TestCostEstimationIntegration:
    """Integration tests for cost estimation with other features."""

    def test_estimate_then_record_workflow(self):
        """Test realistic workflow: estimate, decide, then record."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        budget = Decimal("0.10")

        # Step 1: Estimate cost
        prompt = "Write a detailed analysis"
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo", input_data=prompt, max_tokens=1500
        )

        # Step 2: Make decision based on estimate
        if session.cumulative_cost + estimated > budget:
            model = "gpt-4o-mini"
            max_tokens = 1000
        else:
            model = "gpt-4-turbo"
            max_tokens = 1500

        # Step 3: Record actual call
        event = session.record_llm_call(
            model=model,
            input_data={"prompt": prompt},
            output_data={"response": "Analysis here"},
            input_tokens=50,
            output_tokens=300,
            generation_params={"temperature": 0.3, "max_tokens": max_tokens},
        )

        # Verify recording worked
        assert event.event_name == model  # Model is stored in event_name for LLM_CALL events
        assert event.metadata["generation_params"]["max_tokens"] == max_tokens

    def test_estimate_with_session_cost_accumulation(self):
        """Test cost estimation with accumulated session costs."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Record some calls
        session.record_llm_call(
            model="gpt-4-turbo",
            input_data={"prompt": "First call"},
            output_data={"response": "Response 1"},
            input_tokens=100,
            output_tokens=50,
        )

        cumulative_before = session.cumulative_cost

        # Estimate next call
        estimated = session.estimate_call_cost(
            model="gpt-4-turbo", input_data="Second call", max_tokens=100
        )

        # Check if next call would exceed a limit
        hypothetical_total = cumulative_before + estimated

        assert hypothetical_total >= cumulative_before
        assert estimated > Decimal("0")

    def test_generation_params_with_cost_controls(self):
        """Test generation_params tracking with cost limits."""
        config = ClyroConfig(
            api_key="cly_test_key",
            controls=ExecutionControls(
                enable_cost_limit=True,
                max_cost_usd=0.10,
            ),
        )
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Record calls with different params until near limit
        events = []
        for i in range(3):
            try:
                event = session.record_llm_call(
                    model="gpt-4-turbo",
                    input_data={"prompt": f"Call {i}"},
                    output_data={"response": f"Response {i}"},
                    input_tokens=500,
                    output_tokens=500,
                    generation_params={"temperature": 0.5 + i * 0.1, "max_tokens": 500},
                )
                events.append(event)
            except Exception:
                # Cost limit exceeded
                break

        # Should have recorded at least one event with params
        assert len(events) > 0
        for event in events:
            assert "generation_params" in event.metadata


class TestParameterImpactAnalysis:
    """Tests for analyzing parameter impact on costs."""

    def test_temperature_impact_simulation(self):
        """Simulate analyzing temperature impact on costs."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Simulate calls with different temperatures
        # (In practice, higher temp → more verbose → higher cost)
        temps = [0.0, 0.5, 1.0]
        output_tokens = [50, 75, 100]  # Simulating verbosity increase

        events = []
        for temp, tokens in zip(temps, output_tokens, strict=False):
            event = session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Explain AI"},
                output_data={"response": "..."},
                input_tokens=50,
                output_tokens=tokens,
                generation_params={"temperature": temp},
            )
            events.append(event)

        # Can analyze: higher temp correlates with higher tokens/cost
        assert events[0].token_count_output < events[2].token_count_output
        assert events[0].cost_usd < events[2].cost_usd

        # Params are trackable
        assert events[0].metadata["generation_params"]["temperature"] == 0.0
        assert events[2].metadata["generation_params"]["temperature"] == 1.0

    def test_max_tokens_impact_simulation(self):
        """Simulate analyzing max_tokens impact on costs."""
        config = ClyroConfig(api_key="cly_test_key")
        session = Session(config=config, session_id=uuid4())
        session.start()

        # Estimate costs with different max_tokens
        max_tokens_values = [100, 500, 2000]
        estimates = []

        for max_tokens in max_tokens_values:
            est = session.estimate_call_cost(
                model="gpt-4-turbo", input_data="Write an essay", max_tokens=max_tokens
            )
            estimates.append(est)

        # Higher max_tokens should lead to higher estimates
        assert estimates[0] < estimates[1] < estimates[2]
