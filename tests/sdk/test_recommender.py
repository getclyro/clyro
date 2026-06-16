# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Unit tests for the policy-recommender SDK engine (FRD-PR-001..016).

from __future__ import annotations

import pytest

from clyro.recommender import mappers
from clyro.recommender.cache import FingerprintCache
from clyro.recommender.catalogue_client import CatalogueSnapshot, _version_digest
from clyro.recommender.fingerprint import compute_fingerprint
from clyro.recommender.introspection import AgentIntrospector
from clyro.recommender.proposer import LlmProposer, LlmValidationError, _parse_json, _validate_ids
from clyro.recommender.transport import (
    RecommenderConfigError,
    TransportUnavailable,
    resolve_transport,
)
from clyro.recommender.types import ToolSpec, ToolSurface, TopologyMeta


def _surface(tools=None, **topo):
    return ToolSurface(
        framework="langgraph",
        tools=tools or [],
        topology=TopologyMeta(**topo),
    )


# --- agent_type detection (FRD-PR-005/006) ------------------------------------
class TestAgentType:
    def test_transactional_from_side_effect_tools(self):
        s = _surface([ToolSpec(name="refund_customer"), ToolSpec(name="transfer_funds")])
        at, _alts, _scores, conf = mappers.detect_agent_type(s, "")
        assert at == mappers.AT_TRANSACTIONAL
        assert conf in ("high", "medium")

    def test_retrieval_from_rag_topology(self):
        s = _surface([ToolSpec(name="search_docs")], has_rag=True)
        at, *_ = mappers.detect_agent_type(s, "")
        assert at == mappers.AT_RETRIEVAL

    def test_workflow_from_multi_agent(self):
        s = _surface([], multi_agent=True, node_count=8)
        at, *_ = mappers.detect_agent_type(s, "")
        assert at == mappers.AT_WORKFLOW

    def test_no_signal_defaults_conversational_low(self):
        at, alts, scores, conf = mappers.detect_agent_type(_surface(), "")
        assert at == mappers.AT_CONVERSATIONAL and conf == "low" and scores == {}


# --- concern inference (FRD-PR-007) -------------------------------------------
class TestConcerns:
    def test_pii_arg_and_side_effect(self):
        s = _surface([ToolSpec(name="refund", args_schema={"properties": {"account_number": {}}})])
        ids = {c.id for c in mappers.infer_concerns(s, "")}
        assert mappers.C_PII in ids
        assert mappers.C_REVERSIBILITY in ids and mappers.C_APPROVAL in ids

    def test_mcp_and_rag(self):
        s = _surface([ToolSpec(name="x")], has_mcp=True, has_rag=True)
        ids = {c.id for c in mappers.infer_concerns(s, "")}
        assert mappers.C_TOOL_SCOPE in ids and mappers.C_CREDENTIAL in ids
        assert mappers.C_SOURCE in ids

    def test_empty_agent_no_concerns(self):
        assert mappers.infer_concerns(_surface(), "") == []


# --- kit roll-up (FRD-PR-008) -------------------------------------------------
class TestKits:
    _KITS = [
        {"id": "kit.customer-facing", "applies_to": [mappers.AT_CONVERSATIONAL],
         "concerns": [mappers.C_PII, mappers.C_HALLUCINATION]},
        {"id": "kit.regulated-starter", "applies_to": [mappers.AT_CONVERSATIONAL],
         "concerns": [mappers.C_PII]},
    ]

    def test_full_coverage_kit_chosen(self):
        kits = mappers.rollup_kits(mappers.AT_CONVERSATIONAL, [mappers.C_PII], self._KITS)
        assert any(k.id == "kit.regulated-starter" and not k.partial_match for k in kits)

    def test_best_fit_when_below_threshold(self):
        kits = mappers.rollup_kits(
            mappers.AT_CONVERSATIONAL,
            [mappers.C_PII, mappers.C_COST, mappers.C_INJECTION],
            self._KITS,
        )
        assert len(kits) == 1 and kits[0].partial_match is True

    def test_no_kit_for_agent_type(self):
        assert mappers.rollup_kits(mappers.AT_CODE, [mappers.C_PII], self._KITS) == []


# --- sector hint (FRD-PR-009) -------------------------------------------------
class TestSector:
    def test_single_sector(self):
        assert mappers.sector_hint("our bank follows DORA and PCI") == "bfsi"

    def test_ambiguous_returns_none(self):
        assert mappers.sector_hint("patient checkout at the clinical shopper bank DORA") is None

    def test_weak_signal_none(self):
        assert mappers.sector_hint("a bank") is None  # only 1 keyword


# --- fingerprint (FRD-PR-003) -------------------------------------------------
class TestFingerprint:
    def test_deterministic(self):
        s = _surface([ToolSpec(name="a"), ToolSpec(name="b")])
        assert compute_fingerprint(s, "p", "v1") == compute_fingerprint(s, "p", "v1")

    def test_changes_with_catalogue_version(self):
        s = _surface([ToolSpec(name="a")])
        assert compute_fingerprint(s, "p", "v1") != compute_fingerprint(s, "p", "v2")

    def test_tool_order_independent(self):
        a = _surface([ToolSpec(name="a"), ToolSpec(name="b")])
        b = _surface([ToolSpec(name="b"), ToolSpec(name="a")])
        assert compute_fingerprint(a, "p", "v") == compute_fingerprint(b, "p", "v")


# --- transport resolution (FRD-PR-012/013/014/015) ----------------------------
class TestTransport:
    def test_rule_based_returns_none(self):
        assert resolve_transport("rule-based") is None

    def test_invalid_raises_config_error(self):
        with pytest.raises(RecommenderConfigError):
            resolve_transport("bogus")

    def test_explicit_claude_code_missing_fails_loud(self, monkeypatch):
        monkeypatch.setattr("clyro.recommender.transport.shutil.which", lambda _: None)
        with pytest.raises(TransportUnavailable):
            resolve_transport("claude-code")

    def test_auto_with_nothing_available_is_rule_based(self, monkeypatch):
        monkeypatch.setattr("clyro.recommender.transport.shutil.which", lambda _: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert resolve_transport("auto") is None

    def test_cloud_forces_anthropic(self, monkeypatch):
        monkeypatch.setattr("clyro.recommender.transport.shutil.which", lambda _: "/usr/bin/claude")
        # cloud ignores claude-code availability and requires a key → unavailable here
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(TransportUnavailable):
            resolve_transport("auto", deployment_mode="cloud")


# --- proposer schema validation (FRD-PR-010) ----------------------------------
class _FakeTransport:
    name = "fake"

    def __init__(self, responses):
        self._responses = list(responses)

    def is_available(self):
        return True

    def invoke(self, prompt):
        return self._responses.pop(0)


def _snapshot():
    return CatalogueSnapshot(
        agent_type_ids={"agent_type.conversational"},
        concern_ids={"concern.pii-protection"},
        kit_ids={"kit.customer-facing"},
    )


class TestProposer:
    def test_parse_json_handles_fences(self):
        assert _parse_json('```json\n{"a": 1}\n```')["a"] == 1

    def test_validate_ids_flags_unknown(self):
        snap = _snapshot()
        bad = {"detected_agent_type": "agent_type.nope"}
        assert _validate_ids(bad, snap) == "agent_type.nope"

    def test_propose_valid(self):
        snap = _snapshot()
        good = '{"detected_agent_type":"agent_type.conversational","alternative_agent_types":[],"recommended_kits":[{"id":"kit.customer-facing"}],"inferred_concerns":[{"id":"concern.pii-protection"}]}'
        out = LlmProposer(_FakeTransport([good]), snap).propose(_surface(), "")
        assert out["detected_agent_type"] == "agent_type.conversational"

    def test_propose_invalid_retries_then_raises(self):
        snap = _snapshot()
        bad = '{"detected_agent_type":"agent_type.nope"}'
        with pytest.raises(LlmValidationError):
            LlmProposer(_FakeTransport([bad, bad]), snap).propose(_surface(), "")


# --- catalogue snapshot (FRD-PR-003/010) --------------------------------------
class TestCatalogue:
    def test_version_digest_deterministic(self):
        items = {"concerns": [{"id": "concern.x", "version": 2}]}
        assert _version_digest(items) == _version_digest(items)

    def test_is_valid_id(self):
        snap = _snapshot()
        assert snap.is_valid_id("concern.pii-protection")
        assert not snap.is_valid_id("concern.unknown")


# --- cache (FRD-PR-016) -------------------------------------------------------
class TestCache:
    def test_put_get_roundtrip(self, tmp_path):
        cache = FingerprintCache(path=tmp_path / "c.db")
        cache.put("fp1", {"x": 1})
        assert cache.get("fp1") == {"x": 1}

    def test_miss_returns_none(self, tmp_path):
        cache = FingerprintCache(path=tmp_path / "c.db")
        assert cache.get("absent") is None

    def test_expired_returns_none(self, tmp_path):
        cache = FingerprintCache(path=tmp_path / "c.db", ttl_days=0)
        cache.put("fp", {"x": 1})
        assert cache.get("fp") is None  # ttl 0 → immediately expired


# --- introspection (FRD-PR-001/002/004) ---------------------------------------
class _FakeTool:
    def __init__(self, name, description=None):
        self.name = name
        self.description = description


class _FakeAgent:
    def __init__(self):
        self.tools = [_FakeTool("refund_customer", "Refund a customer")]
        self.system_prompt = "You are a support assistant"
        self.model = "claude-opus-4-8"


class TestIntrospection:
    def test_extracts_tools_prompt_model(self):
        result = AgentIntrospector().introspect(_FakeAgent())
        assert any(t.name == "refund_customer" for t in result.surface.tools)
        assert "support assistant" in result.system_prompt
        assert result.model_id == "claude-opus-4-8"

    def test_never_raises_on_opaque_agent(self):
        result = AgentIntrospector().introspect(object())
        assert result.surface.tools == [] and result.system_prompt == ""
        assert result.model_id == "unknown"
