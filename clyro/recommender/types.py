# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — shared types
# Implements policy-recommender FRD-PR-001, FRD-PR-010 (TDD §3.4)

"""Dataclasses for the recommender pipeline.

The wire shape of ``RecommendationPayload`` mirrors the backend
``routes/schemas/policy_recommender.py`` and ``FRD_frontend.md`` §6 — the single
source of truth for the payload contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Confidence = Literal["high", "medium", "low"]
Framework = Literal["langgraph", "crewai", "anthropic", "claude_agent_sdk", "generic"]


@dataclass(frozen=True)
class ToolSpec:
    """A single tool exposed to the agent."""

    name: str
    description: str | None = None
    args_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class TopologyMeta:
    """Adapter-specific structural metadata (FRD-PR-001)."""

    node_count: int = 0
    multi_agent: bool = False
    has_rag: bool = False
    has_mcp: bool = False


@dataclass(frozen=True)
class ToolSurface:
    """The introspected shape of a wrapped agent (FRD-PR-001)."""

    framework: Framework
    tools: list[ToolSpec] = field(default_factory=list)
    topology: TopologyMeta = field(default_factory=TopologyMeta)


@dataclass
class Recommendation:
    """One recommended kit or inferred concern with rationale + confidence."""

    id: str
    rationale: str
    confidence: Confidence = "medium"
    coverage_pct: int | None = None
    partial_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "rationale": self.rationale, "confidence": self.confidence}
        if self.coverage_pct is not None:
            out["coverage_pct"] = self.coverage_pct
        if self.partial_match is not None:
            out["partial_match"] = self.partial_match
        return out


@dataclass
class RecommendationPayload:
    """The full recommender output (TDD §3.4 / FRD-PR-010)."""

    agent_fingerprint: str
    detected_agent_type: str
    alternative_agent_types: list[str] = field(default_factory=list)
    recommended_kits: list[Recommendation] = field(default_factory=list)
    inferred_concerns: list[Recommendation] = field(default_factory=list)
    sector_hint: str | None = None
    llm_enriched: bool = False
    transport_used: str = "rule-based"
    catalogue_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recommended_kits"] = [r.to_dict() for r in self.recommended_kits]
        data["inferred_concerns"] = [r.to_dict() for r in self.inferred_concerns]
        return data
