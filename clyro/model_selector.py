# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Model Selector
# Cost Optimization Utility

"""
Model selection utility for cost-performance optimization.

This module provides intelligent model and parameter recommendations
based on task types, helping users make informed cost-performance
trade-offs.

Example:
    ```python
    from clyro.model_selector import ModelSelector

    # Get recommendation for a classification task
    rec = ModelSelector.recommend("classification", budget_usd=0.001)

    # Use recommended settings
    session.record_llm_call(
        model=rec["recommended_models"][0],
        input_data=prompt,
        generation_params=rec["params"]
    )
    ```
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ModelSelector:
    """
    Helper for choosing cost-optimal models based on task type.

    This class provides curated recommendations for common LLM use cases,
    balancing cost, quality, and appropriate parameter settings.
    """

    # Task profiles with recommended models and parameters
    TASK_PROFILES: dict[str, dict[str, Any]] = {
        "classification": {
            "description": "Binary or multi-class classification, sentiment analysis, categorization",
            "recommended_models": [
                "claude-3-haiku",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
            ],
            "params": {
                "temperature": 0.0,
                "max_tokens": 50,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.0002,
            "rationale": "Classification requires deterministic, concise outputs. Low temperature (0.0) ensures consistency. Minimal tokens needed.",
        },
        "data_extraction": {
            "description": "Structured data extraction, entity recognition, information retrieval",
            "recommended_models": [
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "claude-3-haiku",
            ],
            "params": {
                "temperature": 0.1,
                "max_tokens": 500,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.001,
            "rationale": "Data extraction benefits from low temperature (0.1) for accuracy. Moderate token budget for structured output.",
        },
        "summarization": {
            "description": "Text summarization, content condensation",
            "recommended_models": [
                "gpt-4o-mini",
                "claude-3-5-sonnet",
                "gpt-4-turbo",
            ],
            "params": {
                "temperature": 0.3,
                "max_tokens": 500,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.003,
            "rationale": "Summarization needs balance between accuracy and fluency. Low-medium temperature (0.3) works well.",
        },
        "qa": {
            "description": "Question answering, information lookup, fact retrieval",
            "recommended_models": [
                "gpt-4o-mini",
                "claude-3-5-sonnet",
                "gpt-4-turbo",
            ],
            "params": {
                "temperature": 0.2,
                "max_tokens": 300,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.002,
            "rationale": "Q&A requires factual accuracy with some fluency. Low temperature (0.2) minimizes hallucinations.",
        },
        "creative_writing": {
            "description": "Story generation, creative content, marketing copy",
            "recommended_models": [
                "gpt-4-turbo",
                "claude-3-5-sonnet",
                "claude-3-opus",
            ],
            "params": {
                "temperature": 0.8,
                "max_tokens": 2000,
                "top_p": 0.95,
            },
            "expected_cost_usd": 0.07,
            "rationale": "Creative tasks benefit from high temperature (0.8) and top_p (0.95) for diversity. Large token budget for long-form content.",
        },
        "code_generation": {
            "description": "Code writing, code completion, debugging assistance",
            "recommended_models": [
                "gpt-4-turbo",
                "claude-3-5-sonnet",
                "gpt-4o",
            ],
            "params": {
                "temperature": 0.3,
                "max_tokens": 1500,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.055,
            "rationale": "Code generation needs balance: low enough temperature (0.3) for correctness, high enough for multiple valid solutions.",
        },
        "code_review": {
            "description": "Code analysis, bug detection, security review",
            "recommended_models": [
                "gpt-4-turbo",
                "claude-3-5-sonnet",
                "claude-3-opus",
            ],
            "params": {
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.045,
            "rationale": "Code review requires analytical precision. Low temperature (0.2) for consistent, thorough analysis.",
        },
        "translation": {
            "description": "Language translation, localization",
            "recommended_models": [
                "gpt-4o",
                "gpt-4-turbo",
                "claude-3-5-sonnet",
            ],
            "params": {
                "temperature": 0.3,
                "max_tokens": 1000,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.025,
            "rationale": "Translation needs low-medium temperature (0.3) for accuracy while preserving natural fluency.",
        },
        "conversation": {
            "description": "Chatbot, customer support, interactive dialogue",
            "recommended_models": [
                "gpt-4o-mini",
                "gpt-4o",
                "claude-3-5-sonnet",
            ],
            "params": {
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.015,
            "rationale": "Conversation benefits from medium-high temperature (0.7) for natural, varied responses.",
        },
        "reasoning": {
            "description": "Complex reasoning, multi-step problem solving, chain-of-thought",
            "recommended_models": [
                "gpt-4-turbo",
                "claude-3-opus",
                "claude-3-5-sonnet",
            ],
            "params": {
                "temperature": 0.4,
                "max_tokens": 2000,
                "top_p": 0.9,
            },
            "expected_cost_usd": 0.08,
            "rationale": "Reasoning tasks need powerful models with low-medium temperature (0.4) for logical consistency. Large token budget for step-by-step explanations.",
        },
    }

    @classmethod
    def recommend(
        cls,
        task_type: str,
        budget_usd: float | None = None,
        prefer_speed: bool = False,
    ) -> dict[str, Any]:
        """
        Get recommended model and parameters for a task type.

        Args:
            task_type: Type of task (see TASK_PROFILES keys)
            budget_usd: Optional budget constraint per call
            prefer_speed: If True, prioritize faster/cheaper models

        Returns:
            Dictionary with:
                - description: Task description
                - recommended_models: List of model IDs (best first)
                - params: Recommended generation parameters
                - expected_cost_usd: Expected cost per call
                - rationale: Explanation of recommendations

        Raises:
            ValueError: If task_type is not recognized

        Example:
            ```python
            # Get recommendation with budget constraint
            rec = ModelSelector.recommend("classification", budget_usd=0.001)

            # Check if within budget
            if rec["expected_cost_usd"] <= 0.001:
                model = rec["recommended_models"][0]
            ```
        """
        if task_type not in cls.TASK_PROFILES:
            available = ", ".join(cls.TASK_PROFILES.keys())
            raise ValueError(
                f"Unknown task type: '{task_type}'. Available types: {available}"
            )

        profile = cls.TASK_PROFILES[task_type].copy()

        # Apply budget filtering if specified
        if budget_usd is not None:
            if profile["expected_cost_usd"] > budget_usd:
                # Try to find a cheaper alternative
                cheaper = cls._find_cheaper_alternative(task_type, budget_usd)
                if cheaper:
                    logger.info(
                        "model_selector_budget_adjusted",
                        task_type=task_type,
                        budget_usd=budget_usd,
                        original_cost=profile["expected_cost_usd"],
                        adjusted_cost=cheaper["expected_cost_usd"],
                    )
                    return cheaper
                else:
                    logger.warning(
                        "model_selector_over_budget",
                        task_type=task_type,
                        budget_usd=budget_usd,
                        expected_cost=profile["expected_cost_usd"],
                        message="No cheaper alternative found, returning original recommendation",
                    )

        # Re-order models if speed is preferred
        if prefer_speed:
            # Move cheaper/faster models to front
            speed_priority = [
                "haiku",
                "mini",
                "3.5-turbo",
                "4o",
                "sonnet",
                "opus",
                "4-turbo",
            ]
            models = profile["recommended_models"]
            sorted_models = sorted(
                models,
                key=lambda m: next(
                    (i for i, p in enumerate(speed_priority) if p in m), 999
                ),
            )
            profile["recommended_models"] = sorted_models

        return profile

    @classmethod
    def _find_cheaper_alternative(
        cls,
        task_type: str,
        budget_usd: float,
    ) -> dict[str, Any] | None:
        """
        Find a cheaper model configuration for the task type.

        This method attempts to:
        1. Use the cheapest model from the recommended list
        2. Reduce max_tokens if still over budget
        3. Fall back to absolute cheapest models

        Args:
            task_type: Type of task
            budget_usd: Budget constraint

        Returns:
            Modified profile dictionary, or None if no solution found
        """
        profile = cls.TASK_PROFILES[task_type].copy()

        # Model cost order (cheapest to most expensive)
        cheap_models = [
            "claude-3-haiku",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-4o",
            "claude-3-5-sonnet",
            "gpt-4-turbo",
            "claude-3-opus",
        ]

        # Try each cheap model
        for cheap_model in cheap_models:
            # Estimate if this model would fit budget
            # Using rough cost heuristics (actual cost depends on pricing table)
            if "haiku" in cheap_model:
                est_cost = 0.0002
            elif "mini" in cheap_model or "3.5-turbo" in cheap_model:
                est_cost = 0.0005
            elif "4o" in cheap_model:
                est_cost = 0.005
            elif "sonnet" in cheap_model or "4-turbo" in cheap_model:
                est_cost = 0.03
            else:
                est_cost = 0.05

            if est_cost <= budget_usd:
                profile["recommended_models"] = [cheap_model]
                profile["expected_cost_usd"] = est_cost
                profile["rationale"] += (
                    f" (Budget-optimized: switched to {cheap_model})"
                )
                return profile

        # If still over budget, try reducing max_tokens
        params = profile["params"].copy()
        original_max_tokens = params.get("max_tokens", 1000)

        # Try 50%, 25%, 10% of original max_tokens
        for reduction in [0.5, 0.25, 0.1]:
            reduced_tokens = int(original_max_tokens * reduction)
            if reduced_tokens < 10:
                continue

            # Rough cost is proportional to max_tokens
            reduced_cost = profile["expected_cost_usd"] * reduction

            if reduced_cost <= budget_usd:
                params["max_tokens"] = reduced_tokens
                profile["params"] = params
                profile["expected_cost_usd"] = reduced_cost
                profile["rationale"] += (
                    f" (Budget-optimized: reduced max_tokens to {reduced_tokens})"
                )
                return profile

        return None

    @classmethod
    def get_available_tasks(cls) -> list[str]:
        """
        Get list of supported task types.

        Returns:
            List of task type identifiers
        """
        return list(cls.TASK_PROFILES.keys())

    @classmethod
    def get_task_info(cls, task_type: str) -> dict[str, Any]:
        """
        Get information about a task type.

        Args:
            task_type: Type of task

        Returns:
            Dictionary with task information

        Raises:
            ValueError: If task_type is not recognized
        """
        if task_type not in cls.TASK_PROFILES:
            available = ", ".join(cls.TASK_PROFILES.keys())
            raise ValueError(
                f"Unknown task type: '{task_type}'. Available types: {available}"
            )

        return cls.TASK_PROFILES[task_type].copy()


# Cost optimization best practices documentation
COST_OPTIMIZATION_GUIDE = """
Cost Optimization Guide
======================

LLM Parameter Impact on Cost:

1. **temperature** (0.0-2.0):
   - Low (0.0-0.3): Deterministic, concise → Lower cost
   - Medium (0.4-0.7): Balanced → Moderate cost
   - High (0.8-1.0): Creative, verbose → +10-30% cost

   Recommendation: Use 0.0 for classification, 0.8 for creative tasks

2. **max_tokens**:
   - Directly caps output tokens (50-75% of typical cost)
   - Set as tight as possible for your use case
   - Example: Use 50 for yes/no, 500 for summaries, 2000 for essays

3. **top_p** (nucleus sampling):
   - Low (0.5-0.7): Focused → Shorter output
   - High (0.9-1.0): Diverse → Longer output (+5-15% cost)

4. **frequency_penalty** / **presence_penalty**:
   - Can reduce repetition → Potentially shorter outputs
   - Usually minimal impact on cost

Model Selection Guidelines:

1. **For Routine Tasks** (classification, data extraction):
   - Use: claude-3-haiku, gpt-4o-mini
   - Cost: ~$0.0002-0.001 per call
   - Savings: 96-99% vs premium models

2. **For Complex Reasoning** (analysis, problem solving):
   - Use: gpt-4-turbo, claude-3-opus
   - Cost: ~$0.05-0.08 per call
   - Quality: Highest accuracy

3. **For Code** (generation, review):
   - Use: gpt-4-turbo, claude-3-5-sonnet
   - Cost: ~$0.04-0.06 per call
   - Balance: Good quality at moderate cost

4. **For Creative Content**:
   - Use: gpt-4-turbo, claude-3-5-sonnet
   - Cost: ~$0.05-0.07 per call
   - Benefit: Better creativity with high temperature

Cost-Performance Trade-offs:

- GPT-4o-mini: 96% cheaper than GPT-4, ~85% accuracy
- Claude-3-Haiku: 99% cheaper than Opus, ~80% accuracy
- Use cheaper models for high-volume routine tasks
- Reserve expensive models for critical decisions

Best Practices:

1. **Estimate Before Calling**:
   ```python
   estimated = session.estimate_call_cost("gpt-4-turbo", prompt, max_tokens=1500)
   if session.cumulative_cost + estimated > budget:
       model = "gpt-4o-mini"  # Switch to cheaper
   ```

2. **Track Parameter Impact**:
   ```python
   session.record_llm_call(
       model=model,
       input_data=prompt,
       generation_params={
           "temperature": 0.7,
           "max_tokens": 1000,
       }
   )
   ```

3. **Use Task-Based Selection**:
   ```python
   from clyro.model_selector import ModelSelector

   rec = ModelSelector.recommend("classification", budget_usd=0.001)
   session.record_llm_call(
       model=rec["recommended_models"][0],
       input_data=prompt,
       generation_params=rec["params"]
   )
   ```

4. **Monitor and Optimize**:
   - Review session summaries to identify cost hotspots
   - Compare generation_params across high-cost calls
   - Experiment with parameter tuning for your use case
"""
