# Tests for Clyro SDK Cost Calculator
# Implements PRD-009, PRD-010

"""Unit tests for cost calculation and token extraction."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from clyro.config import ClyroConfig
from clyro.cost import (
    AnthropicTokenExtractor,
    CostCalculator,
    OpenAITokenExtractor,
    TiktokenEstimator,
    TokenUsage,
    calculate_cost,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_creation(self):
        """Test creating token usage."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, model="gpt-4")

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "gpt-4"

    def test_total_tokens(self):
        """Test total_tokens property."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_immutable(self):
        """Test that TokenUsage is immutable."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)

        with pytest.raises(AttributeError):
            usage.input_tokens = 200  # type: ignore


class TestOpenAITokenExtractor:
    """Tests for OpenAI token extraction."""

    def test_can_extract_from_object(self):
        """Test detection of OpenAI response objects."""
        extractor = OpenAITokenExtractor()

        @dataclass
        class Usage:
            prompt_tokens: int = 100
            completion_tokens: int = 50

        @dataclass
        class Response:
            usage: Usage = None
            model: str = "gpt-4"

            def __post_init__(self):
                self.usage = Usage()

        response = Response()
        assert extractor.can_extract(response) is True

    def test_can_extract_from_dict(self):
        """Test detection of dict-style OpenAI responses."""
        extractor = OpenAITokenExtractor()

        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model": "gpt-4-turbo",
        }

        assert extractor.can_extract(response) is True

    def test_cannot_extract_invalid_response(self):
        """Test rejection of non-OpenAI responses."""
        extractor = OpenAITokenExtractor()

        assert extractor.can_extract(None) is False
        assert extractor.can_extract({}) is False
        assert extractor.can_extract({"usage": None}) is False
        assert extractor.can_extract("string") is False

    def test_extract_from_object(self):
        """Test extraction from OpenAI response object."""
        extractor = OpenAITokenExtractor()

        @dataclass
        class Usage:
            prompt_tokens: int = 100
            completion_tokens: int = 50

        @dataclass
        class Response:
            usage: Usage = None
            model: str = "gpt-4"

            def __post_init__(self):
                self.usage = Usage()

        response = Response()
        usage = extractor.extract(response)

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.model == "gpt-4"

    def test_extract_from_dict(self):
        """Test extraction from dict-style response."""
        extractor = OpenAITokenExtractor()

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        usage = extractor.extract(response)

        assert usage is not None
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.model == "gpt-4-turbo"

    def test_extract_handles_missing_fields(self):
        """Test graceful handling of missing fields."""
        extractor = OpenAITokenExtractor()

        # Missing completion_tokens
        response = {
            "usage": {"prompt_tokens": 100},
            "model": "gpt-4",
        }

        usage = extractor.extract(response)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 0

    def test_extract_handles_none_values(self):
        """Test handling of None values in usage."""
        extractor = OpenAITokenExtractor()

        response = {
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": 50,
            },
        }

        usage = extractor.extract(response)
        assert usage is not None
        assert usage.input_tokens == 0
        assert usage.output_tokens == 50


class TestAnthropicTokenExtractor:
    """Tests for Anthropic token extraction."""

    def test_can_extract_from_object(self):
        """Test detection of Anthropic response objects."""
        extractor = AnthropicTokenExtractor()

        @dataclass
        class Usage:
            input_tokens: int = 100
            output_tokens: int = 50

        @dataclass
        class Response:
            usage: Usage = None
            model: str = "claude-3-sonnet"

            def __post_init__(self):
                self.usage = Usage()

        response = Response()
        assert extractor.can_extract(response) is True

    def test_can_extract_from_dict(self):
        """Test detection of dict-style Anthropic responses."""
        extractor = AnthropicTokenExtractor()

        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
            "model": "claude-3-sonnet-20240229",
        }

        assert extractor.can_extract(response) is True

    def test_cannot_extract_invalid_response(self):
        """Test rejection of non-Anthropic responses."""
        extractor = AnthropicTokenExtractor()

        assert extractor.can_extract(None) is False
        assert extractor.can_extract({}) is False
        assert extractor.can_extract({"usage": {}}) is False

    def test_extract_from_object(self):
        """Test extraction from Anthropic response object."""
        extractor = AnthropicTokenExtractor()

        @dataclass
        class Usage:
            input_tokens: int = 200
            output_tokens: int = 100

        @dataclass
        class Response:
            usage: Usage = None
            model: str = "claude-3-opus"

            def __post_init__(self):
                self.usage = Usage()

        response = Response()
        usage = extractor.extract(response)

        assert usage is not None
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.model == "claude-3-opus"

    def test_extract_from_dict(self):
        """Test extraction from dict-style response."""
        extractor = AnthropicTokenExtractor()

        response = {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 250,
            },
            "model": "claude-3-haiku-20240307",
        }

        usage = extractor.extract(response)

        assert usage is not None
        assert usage.input_tokens == 500
        assert usage.output_tokens == 250
        assert usage.model == "claude-3-haiku-20240307"


class TestTiktokenEstimator:
    """Tests for tiktoken-based token estimation."""

    def test_is_available(self):
        """Test availability check."""
        # This test will pass whether tiktoken is installed or not
        result = TiktokenEstimator.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not TiktokenEstimator.is_available(),
        reason="tiktoken not installed",
    )
    def test_count_tokens(self):
        """Test token counting with tiktoken."""
        count = TiktokenEstimator.count_tokens("Hello, world!")
        assert count is not None
        assert count > 0

    @pytest.mark.skipif(
        not TiktokenEstimator.is_available(),
        reason="tiktoken not installed",
    )
    def test_estimate_from_text(self):
        """Test estimation from input/output text."""
        usage = TiktokenEstimator.estimate_from_text(
            input_text="What is the capital of France?",
            output_text="The capital of France is Paris.",
            model="gpt-4",
        )

        assert usage is not None
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.model == "gpt-4"

    @pytest.mark.skipif(
        not TiktokenEstimator.is_available(),
        reason="tiktoken not installed",
    )
    def test_estimate_with_empty_text(self):
        """Test estimation with empty text."""
        # Both empty returns None
        usage = TiktokenEstimator.estimate_from_text(
            input_text="",
            output_text="",
        )
        assert usage is None

        # Only input
        usage = TiktokenEstimator.estimate_from_text(
            input_text="Hello",
            output_text=None,
        )
        assert usage is not None
        assert usage.input_tokens > 0
        assert usage.output_tokens == 0


class TestCostCalculator:
    """Tests for CostCalculator."""

    def test_calculate_basic(self):
        """Test basic cost calculation."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost = calculator.calculate(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4-turbo",
        )

        # gpt-4-turbo: input=$0.01/1K, output=$0.03/1K
        # Cost = (1000 * 0.01 + 500 * 0.03) / 1000 = 0.01 + 0.015 = 0.025
        assert cost == Decimal("0.025")

    def test_calculate_zero_tokens(self):
        """Test calculation with zero tokens."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost = calculator.calculate(
            input_tokens=0,
            output_tokens=0,
            model="gpt-4-turbo",
        )

        assert cost == Decimal("0")

    def test_calculate_different_models(self):
        """Test calculation for different models."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # GPT-4-turbo (expensive)
        cost_gpt4 = calculator.calculate(1000, 1000, "gpt-4-turbo")

        # GPT-3.5-turbo (cheaper)
        cost_gpt35 = calculator.calculate(1000, 1000, "gpt-3.5-turbo")

        # Claude-3-haiku (very cheap)
        cost_haiku = calculator.calculate(1000, 1000, "claude-3-haiku")

        # Verify relative costs make sense
        assert cost_gpt4 > cost_gpt35
        assert cost_gpt35 > cost_haiku

    def test_calculate_unknown_model_uses_fallback(self):
        """Test that unknown models use fallback pricing."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost = calculator.calculate(1000, 1000, "unknown-model-xyz")

        # Should use fallback pricing (0.01, 0.03)
        # Cost = (1000 * 0.01 + 1000 * 0.03) / 1000 = 0.04
        assert cost == Decimal("0.04")

    def test_calculate_partial_model_match(self):
        """Test partial model name matching."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # "gpt-4-turbo-preview" should match "gpt-4-turbo" pricing
        cost1 = calculator.calculate(1000, 1000, "gpt-4-turbo")
        cost2 = calculator.calculate(1000, 1000, "gpt-4-turbo-preview")

        assert cost1 == cost2

    def test_extract_tokens_openai(self):
        """Test token extraction from OpenAI response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model": "gpt-4",
        }

        usage = calculator.extract_tokens(response)

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_tokens_anthropic(self):
        """Test token extraction from Anthropic response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "input_tokens": 200,
                "output_tokens": 100,
            },
            "model": "claude-3-sonnet",
        }

        usage = calculator.extract_tokens(response)

        assert usage is not None
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100

    def test_extract_tokens_none_response(self):
        """Test extraction returns None for invalid response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        assert calculator.extract_tokens(None) is None
        assert calculator.extract_tokens({}) is None
        assert calculator.extract_tokens("invalid") is None

    def test_calculate_from_response_openai(self):
        """Test full cost calculation from OpenAI response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        cost, usage = calculator.calculate_from_response(response)

        assert usage is not None
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert cost == Decimal("0.025")

    def test_calculate_from_response_anthropic(self):
        """Test full cost calculation from Anthropic response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
            },
            "model": "claude-3-sonnet",
        }

        cost, usage = calculator.calculate_from_response(response)

        assert usage is not None
        # claude-3-sonnet: input=$0.003/1K, output=$0.015/1K
        # Cost = (1000 * 0.003 + 500 * 0.015) / 1000 = 0.003 + 0.0075 = 0.0105
        assert cost == Decimal("0.0105")

    def test_calculate_from_response_invalid(self):
        """Test calculation from invalid response returns zero."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost, usage = calculator.calculate_from_response({"invalid": True})

        assert cost == Decimal("0")
        assert usage is None

    def test_calculate_from_response_with_fallback_model(self):
        """Test calculation with fallback model when model not in response."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            # No model field
        }

        cost, usage = calculator.calculate_from_response(
            response, fallback_model="gpt-4-turbo"
        )

        assert usage is not None
        assert cost == Decimal("0.025")

    def test_custom_pricing(self):
        """Test custom pricing configuration."""
        config = ClyroConfig(
            pricing={
                "custom-model": {"input": 0.001, "output": 0.002},
            }
        )
        calculator = CostCalculator(config)

        cost = calculator.calculate(1000, 1000, "custom-model")

        # Cost = (1000 * 0.001 + 1000 * 0.002) / 1000 = 0.003
        assert cost == Decimal("0.003")


class TestCalculateCostFunction:
    """Tests for convenience calculate_cost function."""

    def test_calculate_from_response(self):
        """Test calculation from response."""
        config = ClyroConfig()
        response = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model": "gpt-4-turbo",
        }

        cost, usage = calculate_cost(config, response=response)

        assert cost > Decimal("0")
        assert usage is not None

    def test_calculate_from_explicit_tokens(self):
        """Test calculation from explicit token counts."""
        config = ClyroConfig()

        cost, usage = calculate_cost(
            config,
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4-turbo",
        )

        assert cost == Decimal("0.025")
        assert usage is not None
        assert usage.input_tokens == 1000

    def test_calculate_no_data(self):
        """Test calculation with no data."""
        config = ClyroConfig()

        cost, usage = calculate_cost(config)

        assert cost == Decimal("0")
        assert usage is None


class TestCostCalculatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_large_token_counts(self):
        """Test with very large token counts."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost = calculator.calculate(
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=500_000,
            model="gpt-4-turbo",
        )

        # Should calculate without overflow
        assert cost > Decimal("0")
        # gpt-4-turbo: (1M * 0.01 + 500K * 0.03) / 1000 = 10 + 15 = 25
        assert cost == Decimal("25")

    def test_decimal_precision(self):
        """Test decimal precision is maintained."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # Very small number of tokens
        cost = calculator.calculate(
            input_tokens=1,
            output_tokens=1,
            model="claude-3-haiku",  # Very cheap
        )

        # claude-3-haiku: input=$0.00025/1K, output=$0.00125/1K
        # Cost = (1 * 0.00025 + 1 * 0.00125) / 1000 = 0.0000015
        assert cost == Decimal("0.0000015")

    def test_response_with_extra_fields(self):
        """Test extraction ignores extra fields."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,  # Extra field
            },
            "model": "gpt-4",
        }

        usage = calculator.extract_tokens(response)

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_nested_response_structure(self):
        """Test handling of nested response structures."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # Some wrappers nest the response
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model": "gpt-4",
            "data": {
                "nested": "value",
            },
        }

        usage = calculator.extract_tokens(response)
        assert usage is not None
        assert usage.input_tokens == 100

    def test_register_custom_extractor(self):
        """Test registering a custom token extractor.  # Tests cost.py:353-360"""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # Create a custom extractor
        class CustomExtractor:
            def can_extract(self, response):
                return isinstance(response, dict) and "custom_tokens" in response

            def extract(self, response):
                if not isinstance(response, dict) or "custom_tokens" not in response:
                    return None
                return TokenUsage(
                    input_tokens=response["custom_tokens"]["in"],
                    output_tokens=response["custom_tokens"]["out"],
                    model=response.get("model"),
                )

        calculator.register_extractor(CustomExtractor())

        # Test custom extractor is used
        response = {
            "custom_tokens": {"in": 150, "out": 75},
            "model": "custom-llm",
        }

        usage = calculator.extract_tokens(response)
        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 75
        assert usage.model == "custom-llm"

    def test_register_extractor_priority(self):
        """Test custom extractors take priority over built-in ones."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # Custom extractor that overrides OpenAI
        class OverrideExtractor:
            def can_extract(self, response):
                return (
                    isinstance(response, dict)
                    and "usage" in response
                    and "prompt_tokens" in response.get("usage", {})
                )

            def extract(self, response):
                # Return modified values to prove this extractor was used
                return TokenUsage(
                    input_tokens=999,
                    output_tokens=888,
                    model="overridden",
                )

        calculator.register_extractor(OverrideExtractor())

        # Standard OpenAI response
        response = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model": "gpt-4",
        }

        usage = calculator.extract_tokens(response)
        assert usage is not None
        # Should get values from custom extractor
        assert usage.input_tokens == 999
        assert usage.output_tokens == 888
        assert usage.model == "overridden"

    def test_extractor_exception_handling(self):
        """Test that extractor errors are handled gracefully.  # Tests cost.py:387-393"""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # Create a faulty extractor that raises exceptions
        class FaultyExtractor:
            def can_extract(self, response):
                return True  # Claims to handle everything

            def extract(self, response):
                raise RuntimeError("Extractor failure!")

        calculator.register_extractor(FaultyExtractor())

        # Standard OpenAI response - should fall back to built-in extractors
        response = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model": "gpt-4",
        }

        # Should not raise, should fall back to next extractor
        usage = calculator.extract_tokens(response)
        # Falls through to OpenAITokenExtractor
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_calculate_from_text_without_tiktoken(self):
        """Test calculate_from_text returns zero without tiktoken.  # Tests cost.py:490-496"""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost, usage = calculator.calculate_from_text(
            input_text="Hello world",
            output_text="Hi there",
            model="gpt-4",
        )

        # If tiktoken is not installed, should return zero/None
        if not TiktokenEstimator.is_available():
            assert cost == Decimal("0")
            assert usage is None
        else:
            # If tiktoken is installed, should return estimated values
            assert cost > Decimal("0")
            assert usage is not None

    @pytest.mark.skipif(
        not TiktokenEstimator.is_available(),
        reason="tiktoken not installed",
    )
    def test_calculate_from_text_with_tiktoken(self):
        """Test calculate_from_text with tiktoken available."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        cost, usage = calculator.calculate_from_text(
            input_text="What is the capital of France?",
            output_text="The capital of France is Paris.",
            model="gpt-4-turbo",
        )

        assert cost > Decimal("0")
        assert usage is not None
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0


class TestOpenAIExtractorErrorPaths:
    """Tests for OpenAI extractor error handling paths.  # Tests cost.py:110,120,128-130"""

    def test_extract_returns_none_when_usage_is_none(self):
        """Test extraction when usage is None.  # Tests cost.py:110"""
        extractor = OpenAITokenExtractor()

        @dataclass
        class Response:
            usage: None = None
            model: str = "gpt-4"

        response = Response()
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_with_non_dict_non_object_usage(self):
        """Test extraction with usage that's neither dict nor has attributes.  # Tests cost.py:120"""
        extractor = OpenAITokenExtractor()

        # Usage is a string (unusual but possible)
        response = {"usage": "not_a_dict_or_object", "model": "gpt-4"}
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_handles_attribute_error(self):
        """Test extraction handles AttributeError gracefully.  # Tests cost.py:128-130"""
        extractor = OpenAITokenExtractor()

        # Create object that raises AttributeError
        class BrokenUsage:
            @property
            def prompt_tokens(self):
                raise AttributeError("Broken!")

        @dataclass
        class Response:
            model: str = "gpt-4"

            @property
            def usage(self):
                return BrokenUsage()

        response = Response()
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_handles_type_error(self):
        """Test extraction handles TypeError gracefully."""
        extractor = OpenAITokenExtractor()

        # Create object where int() conversion fails
        class BadTokens:
            prompt_tokens = "not_a_number"
            completion_tokens = 50

        @dataclass
        class Response:
            usage: Any = None
            model: str = "gpt-4"

            def __post_init__(self):
                self.usage = BadTokens()

        response = Response()
        # Should handle ValueError during int conversion
        usage = extractor.extract(response)
        # Will try to convert "not_a_number" to int, which raises ValueError
        # The exception handler catches this
        assert usage is None


class TestAnthropicExtractorErrorPaths:
    """Tests for Anthropic extractor error handling paths.  # Tests cost.py:178,188,196-198"""

    def test_extract_returns_none_when_usage_is_none(self):
        """Test extraction when usage is None.  # Tests cost.py:178"""
        extractor = AnthropicTokenExtractor()

        @dataclass
        class Response:
            usage: None = None
            model: str = "claude-3-sonnet"

        response = Response()
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_with_non_dict_non_object_usage(self):
        """Test extraction with usage that's neither dict nor has attributes.  # Tests cost.py:188"""
        extractor = AnthropicTokenExtractor()

        # Usage is a string
        response = {"usage": "not_a_dict", "model": "claude-3-sonnet"}
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_handles_attribute_error(self):
        """Test extraction handles AttributeError gracefully.  # Tests cost.py:196-198"""
        extractor = AnthropicTokenExtractor()

        class BrokenUsage:
            @property
            def input_tokens(self):
                raise AttributeError("Broken!")

        @dataclass
        class Response:
            model: str = "claude-3-sonnet"

            @property
            def usage(self):
                return BrokenUsage()

        response = Response()
        usage = extractor.extract(response)
        assert usage is None

    def test_extract_handles_type_error(self):
        """Test extraction handles TypeError gracefully."""
        extractor = AnthropicTokenExtractor()

        class BadTokens:
            input_tokens = "invalid"
            output_tokens = 50

        @dataclass
        class Response:
            usage: Any = None
            model: str = "claude-3-sonnet"

            def __post_init__(self):
                self.usage = BadTokens()

        response = Response()
        usage = extractor.extract(response)
        assert usage is None
