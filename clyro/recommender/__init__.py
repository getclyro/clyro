# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender (SDK engine)
# Implements policy-recommender FRD-PR-001..016

"""Static policy recommender for wrapped agents.

Point ``clyro suggest <import-path>`` at an existing agent and the recommender
introspects it (tools, prompt, topology), maps it to the catalogue
(agent_type / concerns / kits), optionally enriches via an LLM, and emits a
recommendation payload for the Agent Setup Wizard.
"""

from clyro.recommender.recommender import Recommender, SuggestResult
from clyro.recommender.types import (
    Recommendation,
    RecommendationPayload,
    ToolSpec,
    ToolSurface,
)

__all__ = [
    "Recommender",
    "SuggestResult",
    "Recommendation",
    "RecommendationPayload",
    "ToolSpec",
    "ToolSurface",
]
