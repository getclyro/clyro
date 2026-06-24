# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — orchestrator
# Implements policy-recommender FRD-PR-001..016 (composition)

"""End-to-end recommender: introspect → fingerprint → cache → rules (+ optional LLM)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from clyro.constants import DEFAULT_API_URL
from clyro.recommender import mappers
from clyro.recommender.cache import FingerprintCache
from clyro.recommender.catalogue_client import CatalogueClient
from clyro.recommender.fingerprint import compute_fingerprint
from clyro.recommender.introspection import AgentIntrospector
from clyro.recommender.proposer import LlmProposer, LlmValidationError
from clyro.recommender.transport import (
    TransportError,
    TransportUnavailable,
    resolve_transport,
)
from clyro.recommender.types import Recommendation, RecommendationPayload

logger = logging.getLogger("clyro.recommender")


@dataclass
class SuggestResult:
    payload: RecommendationPayload
    cache: str  # hit | miss | bypassed
    transport: str  # claude-code | anthropic-api | rule-based
    model_id: str


class Recommender:
    """Compose the recommender pipeline (TDD §5.1)."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        cache: FingerprintCache | None = None,
        introspector: AgentIntrospector | None = None,
    ):
        self._base_url = base_url
        self._cache = cache if cache is not None else FingerprintCache()
        self._introspector = introspector or AgentIntrospector()

    def suggest(
        self,
        agent: Any,
        *,
        llm_transport: str = "auto",
        api_key: str | None = None,
        deployment_mode: str = "self-hosted",
        use_cache: bool = True,
    ) -> SuggestResult:
        """Run the pipeline and return a recommendation (TDD §5.1)."""
        intro = self._introspector.introspect(agent)
        snapshot = CatalogueClient(self._base_url).fetch()
        fingerprint = compute_fingerprint(intro.surface, intro.system_prompt, snapshot.version)

        # Cache lookup (FRD-PR-016).
        if use_cache:
            cached = self._cache.get(fingerprint)
            if cached is not None:
                return SuggestResult(
                    payload=_payload_from_dict(cached),
                    cache="hit",
                    transport=cached.get("transport_used", "rule-based"),
                    model_id=intro.model_id,
                )

        # Rule-based backbone (FRD-PR-005..009).
        agent_type, alternatives, _scores, _conf = mappers.detect_agent_type(
            intro.surface, intro.system_prompt
        )
        concerns = mappers.infer_concerns(intro.surface, intro.system_prompt)
        kits = mappers.rollup_kits(agent_type, [c.id for c in concerns], snapshot.kits)
        sector = mappers.sector_hint(intro.system_prompt)

        # Optional LLM enrichment (FRD-PR-010..015).
        transport = resolve_transport(
            llm_transport, deployment_mode=deployment_mode, api_key=api_key
        )
        llm_enriched = False
        transport_name = "rule-based"
        if transport is not None:
            transport_name = transport.name
            try:
                proposal = LlmProposer(transport, snapshot).propose(
                    intro.surface, intro.system_prompt
                )
                agent_type = proposal.get("detected_agent_type", agent_type)
                alternatives = proposal.get("alternative_agent_types", alternatives)
                kits = _recs_from_llm(proposal.get("recommended_kits", [])) or kits
                concerns = _recs_from_llm(proposal.get("inferred_concerns", [])) or concerns
                llm_enriched = True
            except (LlmValidationError, TransportError, TransportUnavailable) as exc:
                if llm_transport != "auto":
                    raise  # explicit mode fails loud (FRD-PR-015)
                logger.warning("clyro.recommender.llm_fell_back: %s", type(exc).__name__)
                transport_name = "rule-based"

        payload = RecommendationPayload(
            agent_fingerprint=fingerprint,
            detected_agent_type=agent_type,
            alternative_agent_types=list(alternatives),
            recommended_kits=kits,
            inferred_concerns=concerns,
            sector_hint=sector,
            llm_enriched=llm_enriched,
            transport_used=transport_name,
            catalogue_version=snapshot.version,
        )

        cache_status = "miss"
        if use_cache:
            self._cache.put(fingerprint, payload.to_dict())
        else:
            cache_status = "bypassed"

        return SuggestResult(
            payload=payload,
            cache=cache_status,
            transport=transport_name,
            model_id=intro.model_id,
        )


def _recs_from_llm(items: list[dict[str, Any]]) -> list[Recommendation]:
    out: list[Recommendation] = []
    for it in items or []:
        if isinstance(it, dict) and it.get("id"):
            out.append(
                Recommendation(
                    id=it["id"],
                    rationale=it.get("rationale", ""),
                    confidence=it.get("confidence", "medium"),
                    coverage_pct=it.get("coverage_pct"),
                )
            )
    return out


def _payload_from_dict(data: dict[str, Any]) -> RecommendationPayload:
    return RecommendationPayload(
        agent_fingerprint=data.get("agent_fingerprint", ""),
        detected_agent_type=data.get("detected_agent_type", ""),
        alternative_agent_types=data.get("alternative_agent_types", []),
        recommended_kits=_recs_from_llm(data.get("recommended_kits", [])),
        inferred_concerns=_recs_from_llm(data.get("inferred_concerns", [])),
        sector_hint=data.get("sector_hint"),
        llm_enriched=data.get("llm_enriched", False),
        transport_used=data.get("transport_used", "rule-based"),
        catalogue_version=data.get("catalogue_version"),
    )
