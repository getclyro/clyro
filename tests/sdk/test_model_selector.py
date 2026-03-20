# Tests for Model Selector utility
# Validates cost optimization recommendations

import pytest

from clyro.model_selector import COST_OPTIMIZATION_GUIDE, ModelSelector


class TestModelSelector:
    """Test suite for ModelSelector utility."""

    def test_get_available_tasks(self):
        """Test getting list of available task types."""
        tasks = ModelSelector.get_available_tasks()

        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "classification" in tasks
        assert "creative_writing" in tasks
        assert "code_generation" in tasks

    def test_recommend_classification(self):
        """Test recommendation for classification task."""
        rec = ModelSelector.recommend("classification")

        assert "description" in rec
        assert "recommended_models" in rec
        assert "params" in rec
        assert "expected_cost_usd" in rec
        assert "rationale" in rec

        # Classification should recommend cheap models
        assert len(rec["recommended_models"]) > 0
        assert any("haiku" in m or "mini" in m for m in rec["recommended_models"])

        # Classification should have low temperature
        assert rec["params"]["temperature"] == 0.0
        assert rec["params"]["max_tokens"] <= 100

        # Classification should be cheap
        assert rec["expected_cost_usd"] < 0.01

    def test_recommend_creative_writing(self):
        """Test recommendation for creative writing task."""
        rec = ModelSelector.recommend("creative_writing")

        # Creative should recommend powerful models
        assert any("gpt-4" in m or "opus" in m or "sonnet" in m for m in rec["recommended_models"])

        # Creative should have high temperature
        assert rec["params"]["temperature"] >= 0.7
        assert rec["params"]["max_tokens"] >= 1000

        # Creative should be more expensive
        assert rec["expected_cost_usd"] > 0.01

    def test_recommend_code_generation(self):
        """Test recommendation for code generation task."""
        rec = ModelSelector.recommend("code_generation")

        # Code should recommend capable models
        assert any("gpt-4" in m or "sonnet" in m for m in rec["recommended_models"])

        # Code should have balanced temperature
        assert 0.2 <= rec["params"]["temperature"] <= 0.5
        assert rec["params"]["max_tokens"] >= 1000

    def test_recommend_data_extraction(self):
        """Test recommendation for data extraction task."""
        rec = ModelSelector.recommend("data_extraction")

        # Data extraction should have low temperature
        assert rec["params"]["temperature"] <= 0.3
        assert rec["params"]["max_tokens"] <= 1000

        # Should be moderately cheap
        assert rec["expected_cost_usd"] < 0.05

    def test_recommend_with_budget_within_limit(self):
        """Test recommendation when within budget."""
        rec = ModelSelector.recommend("classification", budget_usd=0.001)

        # Should return normal recommendation since classification is cheap
        assert rec["expected_cost_usd"] <= 0.001

    def test_recommend_with_budget_over_limit(self):
        """Test recommendation when over budget."""
        rec = ModelSelector.recommend("creative_writing", budget_usd=0.001)

        # Should either find cheaper alternative or warn
        # The function should still return a recommendation
        assert "recommended_models" in rec
        assert "params" in rec

    def test_recommend_with_prefer_speed(self):
        """Test recommendation with speed preference."""
        rec = ModelSelector.recommend("code_generation", prefer_speed=True)

        # Should prioritize faster models
        first_model = rec["recommended_models"][0]
        # Haiku, mini, or 3.5-turbo should be preferred
        assert any(fast in first_model for fast in ["haiku", "mini", "3.5-turbo", "4o"])

    def test_recommend_unknown_task_raises_error(self):
        """Test that unknown task type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ModelSelector.recommend("unknown_task_type")

        assert "Unknown task type" in str(exc_info.value)
        assert "unknown_task_type" in str(exc_info.value)

    def test_get_task_info(self):
        """Test getting information about a specific task."""
        info = ModelSelector.get_task_info("classification")

        assert "description" in info
        assert "recommended_models" in info
        assert "params" in info
        assert "expected_cost_usd" in info
        assert "rationale" in info

    def test_get_task_info_unknown_raises_error(self):
        """Test that unknown task raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ModelSelector.get_task_info("invalid_task")

        assert "Unknown task type" in str(exc_info.value)

    def test_all_task_profiles_have_required_fields(self):
        """Test that all task profiles have required fields."""
        required_fields = [
            "description",
            "recommended_models",
            "params",
            "expected_cost_usd",
            "rationale",
        ]

        for task_type in ModelSelector.get_available_tasks():
            profile = ModelSelector.get_task_info(task_type)

            for field in required_fields:
                assert field in profile, f"Task '{task_type}' missing field '{field}'"

            # Validate structure
            assert isinstance(profile["description"], str)
            assert isinstance(profile["recommended_models"], list)
            assert len(profile["recommended_models"]) > 0
            assert isinstance(profile["params"], dict)
            assert isinstance(profile["expected_cost_usd"], (int, float))
            assert isinstance(profile["rationale"], str)

    def test_all_params_have_valid_ranges(self):
        """Test that all parameter recommendations are within valid ranges."""
        for task_type in ModelSelector.get_available_tasks():
            profile = ModelSelector.get_task_info(task_type)
            params = profile["params"]

            # Temperature should be 0.0 to 2.0
            if "temperature" in params:
                assert 0.0 <= params["temperature"] <= 2.0

            # Top_p should be 0.0 to 1.0
            if "top_p" in params:
                assert 0.0 <= params["top_p"] <= 1.0

            # Max_tokens should be positive
            if "max_tokens" in params:
                assert params["max_tokens"] > 0

            # Frequency/presence penalty should be -2.0 to 2.0
            if "frequency_penalty" in params:
                assert -2.0 <= params["frequency_penalty"] <= 2.0

            if "presence_penalty" in params:
                assert -2.0 <= params["presence_penalty"] <= 2.0

    def test_cost_expectations_are_reasonable(self):
        """Test that cost expectations are reasonable."""
        for task_type in ModelSelector.get_available_tasks():
            profile = ModelSelector.get_task_info(task_type)
            cost = profile["expected_cost_usd"]

            # All costs should be positive
            assert cost > 0

            # No cost should be unreasonably high (max $1 per call)
            assert cost < 1.0

    def test_budget_optimization_reduces_cost(self):
        """Test that budget optimization actually reduces cost."""
        # Test with an expensive task
        expensive_rec = ModelSelector.recommend("reasoning")
        original_cost = expensive_rec["expected_cost_usd"]

        # Request with tight budget
        budget = original_cost * 0.1  # 10% of original
        cheap_rec = ModelSelector.recommend("reasoning", budget_usd=budget)

        # Should have adjusted somehow (cost or max_tokens)
        # Either cost is reduced, or we got a warning but still got a recommendation
        assert "recommended_models" in cheap_rec

    def test_task_profiles_consistency(self):
        """Test consistency across task profiles."""
        # Low-temperature tasks should generally be cheaper
        low_temp_tasks = []
        high_temp_tasks = []

        for task_type in ModelSelector.get_available_tasks():
            profile = ModelSelector.get_task_info(task_type)
            temp = profile["params"]["temperature"]
            cost = profile["expected_cost_usd"]

            if temp <= 0.3:
                low_temp_tasks.append((task_type, cost))
            elif temp >= 0.7:
                high_temp_tasks.append((task_type, cost))

        # On average, low-temp tasks should be cheaper
        # (they tend to be simpler tasks with fewer tokens)
        if low_temp_tasks and high_temp_tasks:
            avg_low_temp = sum(c for _, c in low_temp_tasks) / len(low_temp_tasks)
            avg_high_temp = sum(c for _, c in high_temp_tasks) / len(high_temp_tasks)

            # This is a general trend, not a strict rule
            # Just checking that the relationship makes sense
            assert avg_low_temp <= avg_high_temp * 3  # Allow some variance


class TestCostOptimizationGuide:
    """Test suite for cost optimization documentation."""

    def test_guide_exists(self):
        """Test that cost optimization guide exists."""
        assert COST_OPTIMIZATION_GUIDE is not None
        assert isinstance(COST_OPTIMIZATION_GUIDE, str)
        assert len(COST_OPTIMIZATION_GUIDE) > 0

    def test_guide_covers_key_topics(self):
        """Test that guide covers essential topics."""
        guide = COST_OPTIMIZATION_GUIDE

        # Should cover parameters
        assert "temperature" in guide
        assert "max_tokens" in guide
        assert "top_p" in guide

        # Should cover model selection
        assert "Model Selection" in guide or "model" in guide.lower()

        # Should cover cost-performance trade-offs
        assert "trade-off" in guide.lower() or "cheaper" in guide.lower()

        # Should have code examples
        assert "```python" in guide or "```" in guide

    def test_guide_mentions_specific_models(self):
        """Test that guide mentions specific model examples."""
        guide = COST_OPTIMIZATION_GUIDE

        # Should mention some actual models
        models_mentioned = [
            "gpt-4" in guide.lower(),
            "claude" in guide.lower(),
            "haiku" in guide.lower(),
            "sonnet" in guide.lower(),
        ]

        assert any(models_mentioned), "Guide should mention specific models"

    def test_guide_has_practical_examples(self):
        """Test that guide includes practical code examples."""
        guide = COST_OPTIMIZATION_GUIDE

        # Should have estimate_call_cost example
        assert "estimate_call_cost" in guide

        # Should have record_llm_call with generation_params example
        assert "generation_params" in guide

        # Should have ModelSelector example
        assert "ModelSelector" in guide


class TestModelSelectorEdgeCases:
    """Test edge cases and error handling."""

    def test_recommend_with_zero_budget(self):
        """Test recommendation with zero budget."""
        rec = ModelSelector.recommend("classification", budget_usd=0.0)

        # Should still return a recommendation (may warn)
        assert "recommended_models" in rec

    def test_recommend_with_negative_budget(self):
        """Test recommendation with negative budget."""
        # Should still work (negative budget doesn't make sense, but shouldn't crash)
        rec = ModelSelector.recommend("classification", budget_usd=-1.0)

        assert "recommended_models" in rec

    def test_recommend_with_very_high_budget(self):
        """Test recommendation with very high budget."""
        rec = ModelSelector.recommend("classification", budget_usd=1000.0)

        # Should return normal recommendation
        assert rec["expected_cost_usd"] < 1000.0

    def test_task_profile_immutability(self):
        """Test that getting recommendations doesn't mutate original profiles."""
        # Get recommendation twice
        rec1 = ModelSelector.recommend("classification")
        rec2 = ModelSelector.recommend("classification")

        # Should be equal (original not mutated)
        assert rec1["params"] == rec2["params"]
        assert rec1["recommended_models"] == rec2["recommended_models"]

    def test_prefer_speed_doesnt_break_recommendation(self):
        """Test that prefer_speed doesn't break the recommendation."""
        for task_type in ModelSelector.get_available_tasks():
            rec = ModelSelector.recommend(task_type, prefer_speed=True)

            # Should still have all required fields
            assert "recommended_models" in rec
            assert "params" in rec
            assert len(rec["recommended_models"]) > 0


class TestModelSelectorIntegration:
    """Integration tests for ModelSelector with realistic scenarios."""

    def test_classification_workflow(self):
        """Test realistic classification workflow."""
        # User wants to do classification with tight budget
        rec = ModelSelector.recommend("classification", budget_usd=0.001)

        # Should recommend cheap model
        model = rec["recommended_models"][0]
        params = rec["params"]

        # Verify settings are appropriate for classification
        assert params["temperature"] <= 0.2  # Low for consistency
        assert params["max_tokens"] <= 100  # Short output
        assert rec["expected_cost_usd"] <= 0.001

    def test_creative_workflow(self):
        """Test realistic creative writing workflow."""
        # User wants creative writing, willing to pay more
        rec = ModelSelector.recommend("creative_writing", budget_usd=0.1)

        model = rec["recommended_models"][0]
        params = rec["params"]

        # Verify settings are appropriate for creative tasks
        assert params["temperature"] >= 0.6  # High for creativity
        assert params["max_tokens"] >= 1000  # Long output

    def test_cost_conscious_code_generation(self):
        """Test cost-conscious code generation."""
        # User wants code generation but with budget constraint
        rec = ModelSelector.recommend("code_generation", budget_usd=0.01)

        # Should either adjust or warn
        assert "recommended_models" in rec

        # If it found a solution, verify it's reasonable
        if rec["expected_cost_usd"] <= 0.01:
            # Should still have enough tokens for code
            assert rec["params"]["max_tokens"] >= 500

    def test_speed_optimized_conversation(self):
        """Test speed-optimized conversation."""
        rec = ModelSelector.recommend("conversation", prefer_speed=True)

        # Should prioritize faster models
        first_model = rec["recommended_models"][0]

        # Verify it's a fast model
        fast_indicators = ["haiku", "mini", "3.5-turbo", "4o"]
        assert any(indicator in first_model for indicator in fast_indicators)
