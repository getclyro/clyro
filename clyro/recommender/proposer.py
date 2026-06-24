# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — LLM proposer
# Implements policy-recommender FRD-PR-010

"""LLM-backed proposer, strictly gated by catalogue-id schema validation.

The proposer enriches the rule-based backbone with nuance + rationale. Its output
is validated against the live catalogue ids; an unknown id triggers exactly one
re-prompt, then the caller falls back to rule-based (FRD-PR-010 failure clause).
The prompt template ships with the SDK (Q4).
"""

from __future__ import annotations

import json
from typing import Any

from clyro.recommender.catalogue_client import CatalogueSnapshot
from clyro.recommender.transport import Transport
from clyro.recommender.types import ToolSurface

_PROMPT_TEMPLATE = """You are a policy-governance assistant for AI agents.
Given an agent's introspected shape and the Clyro catalogue, recommend the best
agent_type, the kits, and the concerns. You MUST only use ids from the catalogue.

CATALOGUE
  agent_types: {agent_types}
  concerns: {concerns}
  kits: {kits}

AGENT SHAPE
  framework: {framework}
  tools: {tools}
  system_prompt (truncated): {system_prompt}
  topology: {topology}

Return ONLY a JSON object, no prose, matching exactly:
{{
  "detected_agent_type": "agent_type.<id>",
  "alternative_agent_types": ["agent_type.<id>"],
  "recommended_kits": [{{"id": "kit.<id>", "rationale": "<short>", "confidence": "high|medium|low"}}],
  "inferred_concerns": [{{"id": "concern.<id>", "rationale": "<short>", "confidence": "high|medium|low"}}]
}}
"""


class LlmValidationError(Exception):
    """The LLM output failed catalogue-id validation twice (FRD-PR-010)."""


def _truncate(text: str, limit: int = 2000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "…"


def build_prompt(surface: ToolSurface, system_prompt: str, snapshot: CatalogueSnapshot) -> str:
    """Render the proposer prompt (FRD-PR-010)."""
    return _PROMPT_TEMPLATE.format(
        agent_types=sorted(snapshot.agent_type_ids),
        concerns=sorted(snapshot.concern_ids),
        kits=sorted(snapshot.kit_ids),
        framework=surface.framework,
        tools=[{"name": t.name, "description": t.description} for t in surface.tools],
        system_prompt=_truncate(system_prompt),
        topology={
            "node_count": surface.topology.node_count,
            "multi_agent": surface.topology.multi_agent,
            "has_rag": surface.topology.has_rag,
            "has_mcp": surface.topology.has_mcp,
        },
    )


def _parse_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction — handles ```json fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _validate_ids(payload: dict[str, Any], snapshot: CatalogueSnapshot) -> str | None:
    """Return the first id that does not resolve against the catalogue, else None."""
    candidates: list[str] = []
    candidates.append(payload.get("detected_agent_type", ""))
    candidates.extend(payload.get("alternative_agent_types", []) or [])
    candidates.extend(
        (k.get("id") if isinstance(k, dict) else k)
        for k in payload.get("recommended_kits", []) or []
    )
    candidates.extend(
        (c.get("id") if isinstance(c, dict) else c)
        for c in payload.get("inferred_concerns", []) or []
    )
    for cid in candidates:
        if cid and not snapshot.is_valid_id(cid):
            return cid
    return None


class LlmProposer:
    """Invoke an LLM transport and return a schema-validated proposal."""

    def __init__(self, transport: Transport, snapshot: CatalogueSnapshot):
        self._transport = transport
        self._snapshot = snapshot

    def propose(self, surface: ToolSurface, system_prompt: str) -> dict[str, Any]:
        """Return a validated proposal dict.

        Raises ``LlmValidationError`` when the output can't be parsed as JSON or
        carries an unknown catalogue id *after one retry* (FRD-PR-010). Transport
        errors (network/auth) propagate to the caller. A non-JSON reply is treated
        as a validation failure — it must never crash ``Recommender.suggest``.
        """
        prompt = build_prompt(surface, system_prompt, self._snapshot)
        payload = self._invoke_and_parse(prompt)
        problem = self._problem(payload)
        if problem is None:
            return payload  # type: ignore[return-value]

        # Re-prompt exactly once with the error context (FRD-PR-010).
        retry_prompt = (
            prompt
            + f"\n\nYour previous answer was invalid ({problem}). Return ONLY a JSON "
            + "object using ids strictly from the catalogue lists above."
        )
        payload = self._invoke_and_parse(retry_prompt)
        problem = self._problem(payload)
        if problem is not None:
            raise LlmValidationError(f"invalid proposer output after retry: {problem}")
        return payload  # type: ignore[return-value]

    def _invoke_and_parse(self, prompt: str) -> dict[str, Any] | None:
        """Invoke the transport and parse JSON; ``None`` on unparseable output.

        Transport errors are *not* caught here — they propagate so the caller can
        distinguish "LLM unreachable" from "LLM returned garbage".
        """
        raw = self._transport.invoke(prompt)
        try:
            return _parse_json(raw)
        except (ValueError, TypeError):
            return None

    def _problem(self, payload: dict[str, Any] | None) -> str | None:
        """Return a description of why ``payload`` is unusable, or None if good."""
        if payload is None:
            return "not valid JSON"
        bad = _validate_ids(payload, self._snapshot)
        return f"unknown id '{bad}'" if bad is not None else None
