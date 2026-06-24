# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# End-to-end / integration tests for the recommender orchestrator, catalogue
# client, and `clyro suggest` CLI (policy-recommender). Complements the unit
# tests in test_recommender.py.

from __future__ import annotations

import json
import urllib.error

import pytest

from clyro.recommender import catalogue_client as cc_mod
from clyro.recommender.cache import FingerprintCache
from clyro.recommender.catalogue_client import CatalogueClient, CatalogueSnapshot
from clyro.recommender.recommender import Recommender, SuggestResult
from clyro.recommender.types import RecommendationPayload


class _FakeAgent:
    def __init__(self):
        self.tools = [type("T", (), {"name": "refund_customer", "description": "Refund"})()]
        # Neutral prompt (no conversational/decisioning keywords) so the side-effect
        # tool makes this unambiguously transactional.
        self.system_prompt = "Process refunds and money transfers"
        self.model = "claude-opus-4-8"


def _snapshot():
    return CatalogueSnapshot(
        agent_type_ids={"agent_type.conversational", "agent_type.transactional"},
        concern_ids={"concern.pii-protection", "concern.reversibility", "concern.approval-gates"},
        kit_ids={"kit.customer-facing"},
        kits=[{"id": "kit.customer-facing", "applies_to": ["agent_type.transactional"],
               "concerns": ["concern.reversibility"]}],
        version="v1",
    )


@pytest.fixture
def _patched_catalogue(monkeypatch):
    monkeypatch.setattr(CatalogueClient, "fetch", lambda self: _snapshot())


# --- Recommender.suggest orchestration (FRD-PR-001..016) ----------------------
class TestRecommenderSuggest:
    def test_rule_based_happy_path(self, tmp_path, _patched_catalogue):
        rec = Recommender(cache=FingerprintCache(path=tmp_path / "c.db"))
        result = rec.suggest(_FakeAgent(), llm_transport="rule-based")
        assert isinstance(result, SuggestResult)
        assert result.transport == "rule-based"
        assert result.payload.llm_enriched is False
        assert result.payload.detected_agent_type == "agent_type.transactional"  # refund verb
        assert result.cache == "miss"
        assert result.payload.catalogue_version == "v1"

    def test_cache_hit_on_second_run(self, tmp_path, _patched_catalogue):
        cache = FingerprintCache(path=tmp_path / "c.db")
        rec = Recommender(cache=cache)
        first = rec.suggest(_FakeAgent(), llm_transport="rule-based")
        assert first.cache == "miss"
        second = rec.suggest(_FakeAgent(), llm_transport="rule-based")
        assert second.cache == "hit"
        assert second.payload.detected_agent_type == first.payload.detected_agent_type

    def test_no_cache_bypasses(self, tmp_path, _patched_catalogue):
        rec = Recommender(cache=FingerprintCache(path=tmp_path / "c.db"))
        result = rec.suggest(_FakeAgent(), llm_transport="rule-based", use_cache=False)
        assert result.cache == "bypassed"

    def test_llm_failure_falls_back_in_auto(self, tmp_path, monkeypatch, _patched_catalogue):
        # auto mode: a transport that errors at invoke → fall back to rule-based.
        from clyro.recommender import recommender as rec_mod
        from clyro.recommender.transport import TransportError

        class _BoomTransport:
            name = "claude-code"

            def invoke(self, prompt):
                raise TransportError("claude-code", "boom")

        monkeypatch.setattr(rec_mod, "resolve_transport", lambda *a, **k: _BoomTransport())
        rec = Recommender(cache=FingerprintCache(path=tmp_path / "c.db"))
        result = rec.suggest(_FakeAgent(), llm_transport="auto")
        assert result.transport == "rule-based"  # fell back
        assert result.payload.llm_enriched is False


# --- CatalogueClient fetch + offline fallback (FRD-PR-010/016) -----------------
class TestCatalogueClient:
    def test_fetch_builds_snapshot(self, monkeypatch, tmp_path):
        items = {
            "agent-types": [{"id": "agent_type.conversational", "version": 1}],
            "concerns": [{"id": "concern.pii-protection", "version": 2}],
            "kits": [{"id": "kit.customer-facing", "version": 1,
                      "applies_to": ["agent_type.conversational"], "concerns": []}],
        }
        monkeypatch.setattr(
            cc_mod, "_http_get_json", lambda url, t: {"items": items[url.rsplit("/", 1)[1]]}
        )
        monkeypatch.setattr(cc_mod, "_SNAPSHOT_PATH", tmp_path / "snap.json")
        snap = CatalogueClient("https://x").fetch()
        assert snap.source == "remote"
        assert "concern.pii-protection" in snap.concern_ids
        assert snap.version  # digest computed

    def test_offline_falls_back_to_cache(self, monkeypatch, tmp_path):
        # Pre-seed a cached snapshot, then make the network fail.
        snap_path = tmp_path / "snap.json"
        snap_path.write_text(json.dumps({
            "agent-types": [{"id": "agent_type.conversational", "version": 1}],
            "concerns": [], "kits": [],
        }))
        monkeypatch.setattr(cc_mod, "_SNAPSHOT_PATH", snap_path)

        def _boom(url, t):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(cc_mod, "_http_get_json", _boom)
        snap = CatalogueClient("https://x").fetch()
        assert snap.source == "cache"
        assert "agent_type.conversational" in snap.agent_type_ids

    def test_offline_no_cache_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cc_mod, "_SNAPSHOT_PATH", tmp_path / "absent.json")

        def _boom(url, t):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(cc_mod, "_http_get_json", _boom)
        with pytest.raises(urllib.error.URLError):
            CatalogueClient("https://x").fetch()


# --- CLI handle_suggest (FRD-PR-FE-001..005) ----------------------------------
class TestHandleSuggest:
    def _args(self, **over):
        import argparse

        ns = argparse.Namespace(
            agent="x:y", llm_transport="rule-based", json=False, out=None,
            apply=False, yes=False, no_cache=False,
        )
        for k, v in over.items():
            setattr(ns, k, v)
        return ns

    def _result(self):
        return SuggestResult(
            payload=RecommendationPayload(
                agent_fingerprint="a" * 64, detected_agent_type="agent_type.conversational",
                catalogue_version="v1",
            ),
            cache="miss", transport="rule-based", model_id="m",
        )

    def test_json_output(self, monkeypatch, capsys):
        from clyro.recommender import cli as cli_mod

        monkeypatch.setattr(cli_mod, "_resolve_agent", lambda p: object())
        result = self._result()
        monkeypatch.setattr(cli_mod.Recommender, "suggest", lambda self, agent, **k: result)
        rc = cli_mod.handle_suggest(self._args(json=True))
        out = capsys.readouterr().out
        assert rc == 0
        assert json.loads(out)["detected_agent_type"] == "agent_type.conversational"

    def test_urlerror_is_handled(self, monkeypatch):
        from clyro.recommender import cli as cli_mod
        from clyro.recommender.transport import EXIT_CONFIG_ERROR

        monkeypatch.setattr(cli_mod, "_resolve_agent", lambda p: object())

        def _boom(self, agent, **k):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(cli_mod.Recommender, "suggest", _boom)
        rc = cli_mod.handle_suggest(self._args())
        assert rc == EXIT_CONFIG_ERROR  # not an unhandled traceback

    def test_bad_import_path_exits_2(self, monkeypatch):
        from clyro.recommender import cli as cli_mod
        from clyro.recommender.transport import EXIT_CONFIG_ERROR

        rc = cli_mod.handle_suggest(self._args(agent="nonexistent.module:thing"))
        assert rc == EXIT_CONFIG_ERROR
