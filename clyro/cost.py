# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Cost Calculator
# Implements PRD-009, PRD-010

"""
Cost calculation module for tracking LLM token usage and expenses.

This module provides utilities to extract token counts from various LLM
provider responses and calculate costs using a configurable pricing table.

Supported Providers:
- OpenAI: Extracts from usage.prompt_tokens, usage.completion_tokens
- Anthropic: Extracts from usage.input_tokens, usage.output_tokens
- Generic: Fallback estimation via tiktoken (optional dependency)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """
    Token usage information extracted from an LLM response.

    Attributes:
        input_tokens: Number of tokens in the prompt/input
        output_tokens: Number of tokens in the completion/output
        model: Model identifier (e.g., "gpt-4-turbo", "claude-3-sonnet")
    """

    input_tokens: int
    output_tokens: int
    model: str | None = None

    def __post_init__(self) -> None:
        """Validate token counts are non-negative."""
        if self.input_tokens < 0:
            raise ValueError(f"input_tokens must be non-negative, got {self.input_tokens}")
        if self.output_tokens < 0:
            raise ValueError(f"output_tokens must be non-negative, got {self.output_tokens}")

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


@runtime_checkable
class TokenExtractor(Protocol):
    """Protocol for token extractors that parse LLM responses."""

    def can_extract(self, response: Any) -> bool:
        """Check if this extractor can handle the given response."""
        ...

    def extract(self, response: Any) -> TokenUsage | None:
        """Extract token usage from the response."""
        ...


class OpenAITokenExtractor:
    """
    Extract token usage from OpenAI API responses.

    Handles both the standard OpenAI response format and LangChain
    wrappers that preserve the usage information.

    Expected formats:
    - response.usage.prompt_tokens / completion_tokens
    - response["usage"]["prompt_tokens"] / ["completion_tokens"]
    """

    def can_extract(self, response: Any) -> bool:
        """Check if response looks like an OpenAI response."""
        # Object attribute access
        if hasattr(response, "usage"):
            usage = getattr(response, "usage", None)
            if usage is not None:
                return hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens")

        # Dict-style access
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                return "prompt_tokens" in usage or "completion_tokens" in usage

        return False

    def extract(self, response: Any) -> TokenUsage | None:
        """Extract token usage from OpenAI response."""
        try:
            usage = None
            model = None

            # Try object attribute access first
            if hasattr(response, "usage"):
                usage = getattr(response, "usage", None)
                model = getattr(response, "model", None)
            elif isinstance(response, dict):
                usage = response.get("usage")
                model = response.get("model")

            if usage is None:
                return None

            # Extract token counts - try object attributes first, then dict access
            input_tokens = 0
            output_tokens = 0

            if hasattr(usage, "prompt_tokens"):
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                return TokenUsage(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    model=str(model) if model else None,
                )

            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or 0
                return TokenUsage(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    model=str(model) if model else None,
                )

            return None

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("openai_token_extraction_failed", error=str(e))
            return None


class AnthropicTokenExtractor:
    """
    Extract token usage from Anthropic API responses.

    Handles both the standard Anthropic response format and variations
    in the response structure.

    Expected formats:
    - response.usage.input_tokens / output_tokens
    - response["usage"]["input_tokens"] / ["output_tokens"]
    """

    def can_extract(self, response: Any) -> bool:
        """Check if response looks like an Anthropic response."""
        # Object attribute access
        if hasattr(response, "usage"):
            usage = getattr(response, "usage", None)
            if usage is not None:
                return hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens")

        # Dict-style access
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                return "input_tokens" in usage or "output_tokens" in usage

        return False

    def extract(self, response: Any) -> TokenUsage | None:
        """Extract token usage from Anthropic response."""
        try:
            usage = None
            model = None

            # Try object attribute access first
            if hasattr(response, "usage"):
                usage = getattr(response, "usage", None)
                model = getattr(response, "model", None)
            elif isinstance(response, dict):
                usage = response.get("usage")
                model = response.get("model")

            if usage is None:
                return None

            # Extract token counts - try object attributes first, then dict access
            input_tokens = 0
            output_tokens = 0

            if hasattr(usage, "input_tokens"):
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                return TokenUsage(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    model=str(model) if model else None,
                )

            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                return TokenUsage(
                    input_tokens=int(input_tokens),
                    output_tokens=int(output_tokens),
                    model=str(model) if model else None,
                )

            return None

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("anthropic_token_extraction_failed", error=str(e))
            return None


class TiktokenEstimator:
    """
    Estimate token counts using tiktoken when provider doesn't report usage.

    This is a fallback mechanism for cases where the LLM response doesn't
    include token usage information. Uses the cl100k_base encoder by default,
    which is compatible with GPT-3.5 and GPT-4 models.

    Note: tiktoken is an optional dependency. If not installed, estimation
    will return None.
    """

    _encoder: Any = None
    _encoder_name: str = "cl100k_base"
    _available: bool | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if tiktoken is installed and available."""
        if cls._available is not None:
            return cls._available

        try:
            import tiktoken  # noqa: F401

            cls._available = True
        except ImportError:
            cls._available = False
            logger.debug("tiktoken_not_available")

        return cls._available

    @classmethod
    def _get_encoder(cls) -> Any:
        """Get or create the tiktoken encoder (lazy initialization)."""
        if cls._encoder is not None:
            return cls._encoder

        if not cls.is_available():
            return None

        try:
            import tiktoken

            cls._encoder = tiktoken.get_encoding(cls._encoder_name)
            return cls._encoder
        except Exception as e:
            logger.warning("tiktoken_encoder_init_failed", error=str(e))
            return None

    @classmethod
    def count_tokens(cls, text: str) -> int | None:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count, or None if tiktoken is not available
        """
        encoder = cls._get_encoder()
        if encoder is None:
            return None

        try:
            return len(encoder.encode(text))
        except Exception as e:
            logger.debug("tiktoken_count_failed", error=str(e))
            return None

    @classmethod
    def estimate_from_text(
        cls,
        input_text: str | None = None,
        output_text: str | None = None,
        model: str | None = None,
    ) -> TokenUsage | None:
        """
        Estimate token usage from input/output text.

        Args:
            input_text: The prompt/input text
            output_text: The completion/output text
            model: Optional model identifier

        Returns:
            TokenUsage with estimated counts, or None if estimation fails
        """
        if not cls.is_available():
            return None

        input_tokens = 0
        output_tokens = 0

        if input_text:
            count = cls.count_tokens(input_text)
            if count is not None:
                input_tokens = count

        if output_text:
            count = cls.count_tokens(output_text)
            if count is not None:
                output_tokens = count

        if input_tokens == 0 and output_tokens == 0:
            return None

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )


class CostCalculator:
    """
    Calculate LLM costs from token usage and pricing configuration.

    This class combines token extraction from multiple providers with
    cost calculation using the configured pricing table.

    Example:
        ```python
        config = ClyroConfig()
        calculator = CostCalculator(config)

        # From an LLM response
        cost = calculator.calculate_from_response(openai_response)

        # From explicit token counts
        cost = calculator.calculate(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4-turbo"
        )
        ```
    """

    def __init__(self, config: ClyroConfig):
        """
        Initialize the cost calculator.

        Args:
            config: ClyroConfig instance containing pricing information
        """
        self.config = config
        self._extractors: list[TokenExtractor] = [
            OpenAITokenExtractor(),
            AnthropicTokenExtractor(),
        ]

    def register_extractor(self, extractor: TokenExtractor) -> None:
        """
        Register a custom token extractor.

        Args:
            extractor: Token extractor implementing the TokenExtractor protocol
        """
        self._extractors.insert(0, extractor)  # Custom extractors take priority

    def extract_tokens(self, response: Any) -> TokenUsage | None:
        """
        Extract token usage from an LLM response.

        Tries each registered extractor in order until one succeeds.

        Args:
            response: LLM response object or dict

        Returns:
            TokenUsage if extraction succeeds, None otherwise
        """
        for extractor in self._extractors:
            try:
                if extractor.can_extract(response):
                    usage = extractor.extract(response)
                    if usage is not None:
                        logger.debug(
                            "tokens_extracted",
                            extractor=type(extractor).__name__,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            model=usage.model,
                        )
                        return usage
            except Exception as e:
                logger.debug(
                    "extractor_failed",
                    extractor=type(extractor).__name__,
                    error=str(e),
                )
                continue

        return None

    def calculate(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> Decimal:
        """
        Calculate cost from token counts.

        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model identifier for pricing lookup

        Returns:
            Cost in USD as Decimal
        """
        if input_tokens == 0 and output_tokens == 0:
            return Decimal("0")

        # Get pricing for the model
        model_name = model or "unknown"
        input_price, output_price = self.config.get_model_pricing(model_name)

        # Calculate cost: (tokens * price_per_1k) / 1000
        input_cost = (Decimal(input_tokens) * input_price) / Decimal("1000")
        output_cost = (Decimal(output_tokens) * output_price) / Decimal("1000")

        total_cost = input_cost + output_cost

        logger.debug(
            "cost_calculated",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=str(input_cost),
            output_cost=str(output_cost),
            total_cost=str(total_cost),
        )

        return total_cost

    def calculate_from_response(
        self,
        response: Any,
        fallback_model: str | None = None,
    ) -> tuple[Decimal, TokenUsage | None]:
        """
        Calculate cost from an LLM response.

        Extracts tokens from the response and calculates cost. Returns
        both the cost and the extracted token usage for tracking.

        Args:
            response: LLM response object or dict
            fallback_model: Model to use if not detectable from response

        Returns:
            Tuple of (cost_usd, token_usage)
            - cost_usd is Decimal("0") if extraction fails
            - token_usage is None if extraction fails
        """
        usage = self.extract_tokens(response)

        if usage is None:
            logger.debug(
                "token_extraction_failed_for_cost",
                response_type=type(response).__name__,
            )
            return Decimal("0"), None

        model = usage.model or fallback_model
        cost = self.calculate(usage.input_tokens, usage.output_tokens, model)

        return cost, usage

    def calculate_from_text(
        self,
        input_text: str | None = None,
        output_text: str | None = None,
        model: str | None = None,
    ) -> tuple[Decimal, TokenUsage | None]:
        """
        Estimate cost from input/output text using tiktoken.

        This is a fallback when LLM responses don't include token usage.

        Args:
            input_text: The prompt/input text
            output_text: The completion/output text
            model: Model identifier for pricing lookup

        Returns:
            Tuple of (estimated_cost_usd, estimated_token_usage)
            - Both are None/zero if tiktoken is not available
        """
        usage = TiktokenEstimator.estimate_from_text(input_text, output_text, model)

        if usage is None:
            return Decimal("0"), None

        cost = self.calculate(usage.input_tokens, usage.output_tokens, model)
        return cost, usage


def calculate_cost(
    config: ClyroConfig,
    response: Any | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    model: str | None = None,
) -> tuple[Decimal, TokenUsage | None]:
    """
    Convenience function to calculate cost.

    Can calculate from either an LLM response or explicit token counts.

    Args:
        config: ClyroConfig with pricing information
        response: Optional LLM response to extract tokens from
        input_tokens: Optional explicit input token count
        output_tokens: Optional explicit output token count
        model: Optional model identifier

    Returns:
        Tuple of (cost_usd, token_usage)
    """
    calculator = CostCalculator(config)

    # If response provided, try to extract from it
    if response is not None:
        return calculator.calculate_from_response(response, fallback_model=model)

    # If explicit tokens provided, calculate directly
    if input_tokens is not None or output_tokens is not None:
        usage = TokenUsage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            model=model,
        )
        cost = calculator.calculate(usage.input_tokens, usage.output_tokens, model)
        return cost, usage

    return Decimal("0"), None


# ---------------------------------------------------------------------------
# Consolidated from clyro_mcp.cost_tracker — Implements FRD-003
# Heuristic cost tracking for MCP/hooks contexts where token metadata
# is unavailable.
# ---------------------------------------------------------------------------


class HeuristicCostEstimator:
    """
    Heuristic cost estimator for MCP/hooks contexts.

    Estimates token counts from JSON payload character length when
    actual token metadata is unavailable (e.g., MCP tool calls).

    The heuristic: ``estimated_tokens = len(json_payload) / 4``
    """

    def __init__(self, cost_per_token_usd: float = 0.00001) -> None:
        if cost_per_token_usd <= 0:
            raise ValueError(f"cost_per_token_usd must be > 0, got {cost_per_token_usd}")
        self._cost_per_token_usd = cost_per_token_usd

    @property
    def cost_per_token_usd(self) -> float:
        return self._cost_per_token_usd

    def estimate_from_payload(
        self, json_payload: str, model: str | None = None
    ) -> tuple[float, int]:
        """
        Estimate cost from a JSON payload string.

        Returns:
            Tuple of (estimated_cost_usd, estimated_tokens).
        """
        estimated_tokens = len(json_payload) // 4
        cost = estimated_tokens * self._cost_per_token_usd
        return cost, estimated_tokens

    def estimate_round_trip(self, params_json_len: int, response_content_len: int) -> float:
        """
        Compute estimated cost for a completed call (params + response).

        Returns:
            Estimated cost in USD for this single call.
        """
        estimated_tokens = (params_json_len + response_content_len) / 4
        return estimated_tokens * self._cost_per_token_usd


class CostTracker:
    """
    Unified cost tracker used by MCP/hooks sessions.

    Provides budget checking and cost accumulation using heuristic
    estimation. Consolidated from ``clyro_mcp.cost_tracker.CostTracker``.
    """

    def __init__(
        self,
        max_cost_usd: float = 10.0,
        cost_per_token_usd: float = 0.00001,
    ) -> None:
        if max_cost_usd <= 0:
            raise ValueError(f"max_cost_usd must be > 0, got {max_cost_usd}")
        self._max_cost_usd = max_cost_usd
        self._estimator = HeuristicCostEstimator(cost_per_token_usd)

    @property
    def max_cost_usd(self) -> float:
        return self._max_cost_usd

    @property
    def cost_per_token_usd(self) -> float:
        return self._estimator.cost_per_token_usd

    def check_budget(
        self,
        accumulated_cost_usd: float,
        params: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check whether forwarding this call would exceed the budget.

        Applies a 2x response overhead multiplier to the input estimate.

        Returns:
            ``(exceeds_budget, details)`` — details populated when blocked.
        """
        import json as _json

        params_json = _json.dumps(params or {}, default=str)
        estimated_input_tokens = len(params_json) / 4
        estimated_round_trip_cost = estimated_input_tokens * 2 * self._estimator.cost_per_token_usd

        if accumulated_cost_usd + estimated_round_trip_cost > self._max_cost_usd:
            return True, {
                "accumulated_cost_usd": round(accumulated_cost_usd, 6),
                "max_cost_usd": self._max_cost_usd,
                "cost_estimated": True,
            }

        return False, {}

    def accumulate(self, params_json_len: int, response_content_len: int) -> float:
        """
        Compute estimated cost for a completed call.

        Returns:
            Estimated cost in USD for this single call.
        """
        return self._estimator.estimate_round_trip(params_json_len, response_content_len)
