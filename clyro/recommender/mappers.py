# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — rule-based mappers
# Implements policy-recommender FRD-PR-005, 006, 007, 008, 009

"""Deterministic rule-based detectors — the high-confidence backbone.

These run with no LLM and emit only catalogue ids. The keyword/signature tables
encode scope §2.2 (S4), §2.3 (S6/S7), and §2.4 (S8). The LLM proposer (when
enabled) refines these; the rule-based output stands alone otherwise.
"""

from __future__ import annotations

import re
from typing import Any

from clyro.recommender.types import Confidence, Recommendation, ToolSurface

# --- Catalogue ids (stable PKs from clyro_api/data/catalogue) -----------------
AT_CONVERSATIONAL = "agent_type.conversational"
AT_TRANSACTIONAL = "agent_type.transactional"
AT_DECISIONING = "agent_type.decisioning"
AT_RETRIEVAL = "agent_type.retrieval"
AT_CODE = "agent_type.code-assistant"
AT_WORKFLOW = "agent_type.workflow-automation"

C_PII = "concern.pii-protection"
C_REVERSIBILITY = "concern.reversibility"
C_APPROVAL = "concern.approval-gates"
C_COST = "concern.cost-governance"
C_TOOL_SCOPE = "concern.tool-scope"
C_CREDENTIAL = "concern.credential-hygiene"
C_HALLUCINATION = "concern.hallucination-guardrails"
C_REG_TAG = "concern.regulatory-tag-coverage"
C_SOURCE = "concern.source-faithfulness"
C_CROSS_BORDER = "concern.cross-border-data"
C_INJECTION = "concern.prompt-injection-defense"

# --- agent_type signatures (FRD-PR-005 / scope S4) ----------------------------
_TOOL_VERBS = {
    AT_TRANSACTIONAL: (
        "pay",
        "refund",
        "charge",
        "transfer",
        "send",
        "book",
        "cancel",
        "create_order",
    ),
    AT_DECISIONING: ("approve", "deny", "score", "recommend", "rank"),
    AT_RETRIEVAL: ("search", "get", "list", "read", "retrieve"),
    AT_CODE: ("execute", "run", "edit_file", "shell", "git"),
    AT_CONVERSATIONAL: ("respond", "reply", "lookup", "answer"),
}
_PROMPT_KEYWORDS = {
    AT_CONVERSATIONAL: ("assistant", "support", "chat", "respond"),
    AT_DECISIONING: ("decide", "score", "approve"),
}
# Concrete RAG signals (matched at word boundaries against tool names). Excludes
# the bare verb "retrieve" / word "knowledge", which fire on ordinary CRUD prose.
_RAG_SIGNALS = (
    "vectorstore",
    "vector_search",
    "embedding",
    "retriever",
    "rag",
    "knowledge_base",
    "semantic_search",
)


def _verb_in(verb: str, name: str) -> bool:
    """Match a verb at a word boundary in a tool name, so ``get`` matches
    ``get_total`` / ``getQueue`` but not ``budget``/``target``."""
    return re.search(rf"(?:^|[^a-z0-9]){re.escape(verb)}", name, re.I) is not None


def _tool_names_lower(surface: ToolSurface) -> list[str]:
    return [t.name.lower() for t in surface.tools]


def _arg_names_lower(surface: ToolSurface) -> list[str]:
    names: list[str] = []
    for t in surface.tools:
        schema = t.args_schema or {}
        props = schema.get("properties", schema) if isinstance(schema, dict) else {}
        if isinstance(props, dict):
            names.extend(str(k).lower() for k in props)
    return names


def detect_agent_type(
    surface: ToolSurface, system_prompt: str
) -> tuple[str, list[str], dict[str, float], Confidence]:
    """Score the six agent_types; return (top, alternatives, scores, confidence).

    Implements FRD-PR-005/006. Defaults to conversational (lowest blast radius)
    when there is no signal.
    """
    prompt = (system_prompt or "").lower()
    tools = _tool_names_lower(surface)
    scores: dict[str, float] = {}

    for at, verbs in _TOOL_VERBS.items():
        scores[at] = scores.get(at, 0.0) + sum(
            1.0 for tool in tools for verb in verbs if _verb_in(verb, tool)
        )
    for at, kws in _PROMPT_KEYWORDS.items():
        scores[at] = scores.get(at, 0.0) + sum(1.0 for kw in kws if kw in prompt)

    # Topology signals.
    if surface.topology.has_rag or any(_verb_in(s, t) for t in tools for s in _RAG_SIGNALS):
        scores[AT_RETRIEVAL] = scores.get(AT_RETRIEVAL, 0.0) + 2.0
    if surface.topology.multi_agent or surface.topology.node_count > 5:
        scores[AT_WORKFLOW] = scores.get(AT_WORKFLOW, 0.0) + 2.0

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    nonzero = [(at, s) for at, s in ranked if s > 0]

    if not nonzero:
        return AT_CONVERSATIONAL, [], {}, "low"

    top_choice, top_score = nonzero[0]
    runner_up = nonzero[1][1] if len(nonzero) > 1 else 0.0
    alternatives = [at for at, _ in nonzero[1:]]

    if top_score >= 2 * max(runner_up, 0.5):
        confidence: Confidence = "high"
    elif top_score > runner_up:
        confidence = "medium"
    else:
        confidence = "low"

    return top_choice, alternatives, dict(nonzero), confidence


# --- concern inference (FRD-PR-007 / scope S6) --------------------------------
_PII_ARGS = re.compile(r"email|ssn|card|phone|address|dob|account_number", re.I)
_SIDE_EFFECT = re.compile(r"pay|refund|charge|transfer|delete|cancel", re.I)
_POLICY_WORDS = re.compile(r"\bpolicy\b|won't|must not|do not", re.I)
# Free-form user input interpolated into the system prompt (template markers like
# {user_input}, {query}, {{question}}) → prompt-injection exposure (scope S6).
_TEMPLATE_MARKER = re.compile(r"\{\{?\s*[a-z_][a-z0-9_]*\s*\}?\}", re.I)


def infer_concerns(surface: ToolSurface, system_prompt: str) -> list[Recommendation]:
    """Map tool/arg/prompt signals → concern recommendations (FRD-PR-007)."""
    prompt = system_prompt or ""
    tools = _tool_names_lower(surface)
    args = _arg_names_lower(surface)
    out: dict[str, Recommendation] = {}

    def add(cid: str, rationale: str, confidence: Confidence) -> None:
        if cid not in out:
            out[cid] = Recommendation(id=cid, rationale=rationale, confidence=confidence)

    pii_arg = next((a for a in args if _PII_ARGS.search(a)), None)
    if pii_arg:
        add(C_PII, f"Tool argument `{pii_arg}` is PII.", "high")

    se_tool = next((t for t in tools if _SIDE_EFFECT.search(t)), None)
    if se_tool:
        add(C_REVERSIBILITY, f"Tool `{se_tool}` performs an irreversible action.", "high")
        add(C_APPROVAL, f"Tool `{se_tool}` warrants an approval gate.", "medium")

    if surface.topology.multi_agent or surface.topology.node_count > 5:
        add(
            C_COST,
            "Multi-agent / multi-node topology increases cost-runaway blast radius.",
            "medium",
        )

    if surface.topology.has_mcp:
        add(C_TOOL_SCOPE, "MCP server attached — tool scope should be allow-listed.", "medium")
        add(
            C_CREDENTIAL,
            "MCP integration handles credentials — enforce credential hygiene.",
            "medium",
        )

    if _POLICY_WORDS.search(prompt):
        add(
            C_HALLUCINATION,
            "System prompt asserts policy constraints — guard against hallucinating them.",
            "low",
        )
        add(C_REG_TAG, "Prompt-asserted constraints benefit from regulatory tagging.", "low")

    if surface.topology.has_rag:
        add(
            C_SOURCE,
            "RAG pipeline detected — every claim should cite a retrievable source.",
            "medium",
        )
        add(C_CROSS_BORDER, "Retrieval may move personal data across jurisdictions.", "low")

    if _TEMPLATE_MARKER.search(prompt):
        add(
            C_INJECTION,
            "System prompt interpolates free-form input (template marker) — injection risk.",
            "medium",
        )

    return list(out.values())


# --- kit roll-up (FRD-PR-008 / scope S7) --------------------------------------
def rollup_kits(
    agent_type: str,
    inferred_concern_ids: list[str],
    kits_catalogue: list[dict[str, Any]],
    threshold: float = 0.7,
) -> list[Recommendation]:
    """Bundle inferred concerns into kits applicable to the agent_type (FRD-PR-008).

    ``kits_catalogue`` items: ``{"id", "applies_to": [agent_type...],
    "concerns": [concern_id...]}`` (as returned by ``GET /v1/kits``).
    Returns kits whose member concerns cover ≥ ``threshold`` of the inferred set,
    ordered by coverage; falls back to the best-fit kit (``partial_match``) when
    none clears the bar.
    """
    inferred = set(inferred_concern_ids)
    candidates: list[Recommendation] = []
    best_fit: Recommendation | None = None

    for kit in kits_catalogue:
        applies_to = set(kit.get("applies_to") or kit.get("agent_types") or [])
        if agent_type not in applies_to:
            continue
        members = {c["id"] if isinstance(c, dict) else c for c in (kit.get("concerns") or [])}
        if not inferred:
            coverage = 1.0 if members else 0.0
        else:
            coverage = len(inferred & members) / len(inferred)
        pct = int(round(coverage * 100))
        rec = Recommendation(
            id=kit["id"],
            rationale=f"Covers {pct}% of the inferred concerns for {agent_type}.",
            confidence="high" if coverage >= threshold else "medium",
            coverage_pct=pct,
        )
        if coverage >= threshold:
            candidates.append(rec)
        if best_fit is None or pct > (best_fit.coverage_pct or -1):
            best_fit = rec

    if candidates:
        return sorted(candidates, key=lambda r: r.coverage_pct or 0, reverse=True)
    if best_fit is not None:
        best_fit.partial_match = True
        return [best_fit]
    return []


# --- sector hint (FRD-PR-009 / scope S8) --------------------------------------
_SECTOR_KEYWORDS = {
    "bfsi": ("bank", "banking", "trader", "fca", "dora", "pci"),
    "pharma": ("patient", "hipaa", "ehr", "clinical"),
    "retail": ("checkout", "order", "shopper", "cart"),
}


def sector_hint(system_prompt: str) -> str | None:
    """Emit at most one high-confidence sector hint (≥2 keywords); else None.

    Ambiguous prompts (matching ≥2 sectors) yield None (FRD-PR-009 failure clause).
    """
    prompt = (system_prompt or "").lower()
    hits = {
        sector: sum(1 for kw in kws if kw in prompt) for sector, kws in _SECTOR_KEYWORDS.items()
    }
    strong = [sector for sector, n in hits.items() if n >= 2]
    if len(strong) == 1:
        return strong[0]
    return None
