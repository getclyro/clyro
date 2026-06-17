# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for the SDK pricing-catalog consumer (C6) + cost precedence

"""
Unit tests for clyro/pricing_catalog.py and the custom > catalog > static >
default precedence in ClyroConfig.get_model_pricing.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from clyro.config import ClyroConfig
from clyro.pricing_catalog import (
    PricingCatalogCache,
    _normalize,
    pricing_catalog_cache,
    refresh_pricing_catalog,
)


@pytest.fixture(autouse=True)
def _reset_singleton_catalog():
    """Keep the module-singleton catalog empty around each test."""
    pricing_catalog_cache._prices = {}
    pricing_catalog_cache._loaded = False
    yield
    pricing_catalog_cache._prices = {}
    pricing_catalog_cache._loaded = False


def _payload(*records) -> dict:
    return {
        "records": [
            {"canonical_key": k, "input_per_1k": i, "output_per_1k": o, "source": "catalog"}
            for (k, i, o) in records
        ],
        "generated_at": "2026-06-12T09:00:00Z",
    }


# ---------------------------------------------------------------------------
# Normalization (parity with the backend C2)
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_strips_provider_prefix(self):
        assert _normalize("openai/gpt-4o") == "gpt-4o"
        assert _normalize("anthropic/claude-3.5-sonnet") == "claude-3.5-sonnet"

    def test_strips_iso_date_suffix(self):
        assert _normalize("anthropic/claude-3.5-sonnet-20241022") == "claude-3.5-sonnet"
        assert _normalize("gpt-4o-2024-08-06") == "gpt-4o"

    def test_strips_price_equivalent_channel_suffixes(self):
        # ':beta'/':extended'/'-latest' are the same model at the same price -> stripped.
        assert _normalize("some-model:extended") == "some-model"
        assert _normalize("some-model:beta") == "some-model"
        assert _normalize("gpt-4o-latest") == "gpt-4o"

    def test_keeps_free_channel(self):
        # ':free' is a DISTINCT product priced $0 -> NOT stripped, keeps its own key
        # (else it collapses into the paid base and gets billed).
        assert _normalize("mistralai/devstral-small:free") == "devstral-small:free"
        assert _normalize("openai/gpt-oss-120b:free") == "gpt-oss-120b:free"

    def test_lowercases_and_trims(self):
        assert _normalize("  OpenAI/GPT-4o  ") == "gpt-4o"


# ---------------------------------------------------------------------------
# Cache parsing + resilience
# ---------------------------------------------------------------------------


class TestCacheUpdate:
    def test_parses_records(self):
        cache = PricingCatalogCache()
        assert cache.update_from_payload(_payload(("gpt-4o", "0.0025", "0.01"))) is True
        assert cache.get("gpt-4o") == (Decimal("0.0025"), Decimal("0.01"))
        assert cache.loaded is True

    def test_lookup_normalizes_raw_id(self):
        cache = PricingCatalogCache()
        cache.update_from_payload(_payload(("claude-3.5-sonnet", "0.003", "0.015")))
        assert cache.get("anthropic/claude-3.5-sonnet-20241022") == (Decimal("0.003"), Decimal("0.015"))

    def test_empty_payload_keeps_last_known(self):
        cache = PricingCatalogCache()
        cache.update_from_payload(_payload(("gpt-4o", "0.0025", "0.01")))
        assert cache.update_from_payload({"records": []}) is False  # not swapped in
        assert cache.get("gpt-4o") == (Decimal("0.0025"), Decimal("0.01"))  # retained

    def test_non_dict_payload_rejected(self):
        cache = PricingCatalogCache()
        assert cache.update_from_payload(["not", "a", "dict"]) is False
        assert cache.update_from_payload({"records": None}) is False

    def test_malformed_records_skipped_others_kept(self):
        cache = PricingCatalogCache()
        payload = {
            "records": [
                {"canonical_key": "gpt-4o", "input_per_1k": "0.0025", "output_per_1k": "0.01"},
                {"canonical_key": "", "input_per_1k": "1", "output_per_1k": "1"},  # empty key
                {"canonical_key": "bad", "input_per_1k": "abc", "output_per_1k": "1"},  # non-numeric
                {"canonical_key": "neg", "input_per_1k": "-1", "output_per_1k": "1"},  # negative
                {"input_per_1k": "1", "output_per_1k": "1"},  # no key
            ]
        }
        cache.update_from_payload(payload)
        assert cache.get("gpt-4o") is not None
        assert cache.get("bad") is None and cache.get("neg") is None

    def test_get_returns_none_when_empty(self):
        assert PricingCatalogCache().get("gpt-4o") is None


# ---------------------------------------------------------------------------
# refresh_pricing_catalog — fetch + fail-open
# ---------------------------------------------------------------------------


class _FakeHttp:
    def __init__(self, payload=None, error: Exception | None = None):
        self._payload = payload
        self._error = error

    async def fetch_pricing(self):
        if self._error is not None:
            raise self._error
        return self._payload


class TestRefresh:
    async def test_refresh_populates_singleton(self):
        ok = await refresh_pricing_catalog(_FakeHttp(payload=_payload(("gpt-4o", "0.0025", "0.01"))))
        assert ok is True
        assert pricing_catalog_cache.get("gpt-4o") == (Decimal("0.0025"), Decimal("0.01"))

    async def test_refresh_fetch_failure_is_fail_open(self):
        ok = await refresh_pricing_catalog(_FakeHttp(error=ConnectionError("backend down")))
        assert ok is False
        assert pricing_catalog_cache.loaded is False  # nothing wiped, no raise


# ---------------------------------------------------------------------------
# Precedence: custom > catalog > static > default
# ---------------------------------------------------------------------------


class TestPricingPrecedence:
    def test_static_used_when_no_catalog(self):
        # gpt-4o is in the bundled DEFAULT_PRICING (static) — unchanged when offline.
        cfg = ClyroConfig(agent_name="t")
        assert cfg.get_model_pricing("gpt-4o") == (Decimal("0.005"), Decimal("0.015"))

    def test_catalog_beats_static(self):
        cfg = ClyroConfig(agent_name="t")
        pricing_catalog_cache.update_from_payload(_payload(("gpt-4o", "0.999", "1.5")))
        assert cfg.get_model_pricing("gpt-4o") == (Decimal("0.999"), Decimal("1.5"))
        # and it normalizes the raw id to the canonical key
        assert cfg.get_model_pricing("openai/gpt-4o") == (Decimal("0.999"), Decimal("1.5"))

    def test_registered_custom_beats_catalog(self):
        cfg = ClyroConfig(agent_name="t")
        pricing_catalog_cache.update_from_payload(_payload(("gpt-4o", "0.999", "1.5")))
        cfg.register_model_pricing("gpt-4o", 0.111, 0.222)
        assert cfg.get_model_pricing("gpt-4o") == (Decimal("0.111"), Decimal("0.222"))

    def test_passed_custom_pricing_beats_catalog(self):
        cfg = ClyroConfig(agent_name="t", pricing={"gpt-4o": {"input": 0.222, "output": 0.444}})
        pricing_catalog_cache.update_from_payload(_payload(("gpt-4o", "0.999", "1.5")))
        assert cfg.get_model_pricing("gpt-4o") == (Decimal("0.222"), Decimal("0.444"))

    def test_unknown_model_falls_to_flat_default(self):
        cfg = ClyroConfig(agent_name="t")
        assert cfg.get_model_pricing("totally-unknown-xyz") == (Decimal("0.01"), Decimal("0.03"))

    def test_catalog_used_for_model_absent_from_static(self):
        cfg = ClyroConfig(agent_name="t")
        pricing_catalog_cache.update_from_payload(_payload(("llama-3.1-405b", "0.002", "0.004")))
        assert cfg.get_model_pricing("meta/llama-3.1-405b") == (Decimal("0.002"), Decimal("0.004"))


# ---------------------------------------------------------------------------
# refresh_pricing_catalog_from_config — provider-agnostic transport entry point
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, api_key=None, local=True, endpoint="http://127.0.0.1:1"):
        self.api_key = api_key
        self.endpoint = endpoint
        self._local = local

    def is_local_only(self):
        return self._local


class TestRefreshFromConfig:
    async def test_noop_without_api_key(self):
        from clyro.pricing_catalog import refresh_pricing_catalog_from_config

        assert await refresh_pricing_catalog_from_config(_Cfg(api_key=None, local=False)) is False

    async def test_noop_in_local_mode(self):
        from clyro.pricing_catalog import refresh_pricing_catalog_from_config

        assert await refresh_pricing_catalog_from_config(_Cfg(api_key="cly_x", local=True)) is False

    async def test_noop_when_already_loaded(self):
        from clyro.pricing_catalog import refresh_pricing_catalog_from_config

        pricing_catalog_cache._loaded = True
        assert await refresh_pricing_catalog_from_config(_Cfg(api_key="cly_x", local=False)) is False

    async def test_unreachable_backend_is_fail_open(self):
        # eligible (cloud + api_key) but backend unreachable -> fail-open, no raise
        from clyro.pricing_catalog import refresh_pricing_catalog_from_config

        assert await refresh_pricing_catalog_from_config(_Cfg(api_key="cly_x", local=False)) is False
        assert pricing_catalog_cache.loaded is False


# ---------------------------------------------------------------------------
# Transport hook — the shared, provider-agnostic trigger (+ GC-safe task)
# ---------------------------------------------------------------------------


class TestTransportHook:
    async def test_hook_routes_to_shared_entry(self, monkeypatch):
        from clyro.transport import Transport

        captured = {}

        async def _fake(config):
            captured["config"] = config
            return True

        monkeypatch.setattr("clyro.pricing_catalog.refresh_pricing_catalog_from_config", _fake)
        t = Transport(ClyroConfig(agent_name="t"))
        await t._refresh_pricing_catalog()
        assert captured.get("config") is t.config  # routed to the shared C6 entry point

    async def test_hook_is_fail_open(self, monkeypatch):
        from clyro.transport import Transport

        async def _boom(config):
            raise RuntimeError("boom")

        monkeypatch.setattr("clyro.pricing_catalog.refresh_pricing_catalog_from_config", _boom)
        t = Transport(ClyroConfig(agent_name="t"))
        await t._refresh_pricing_catalog()  # must not raise

    async def test_start_background_sync_retains_task(self, monkeypatch):
        from clyro.transport import Transport

        t = Transport(ClyroConfig(agent_name="t"))

        async def _noop():
            return None

        monkeypatch.setattr(t._sync_worker, "start", _noop)
        monkeypatch.setattr(t, "_refresh_pricing_catalog", lambda: _noop())
        await t.start_background_sync()
        assert t._pricing_task is not None  # retained so it can't be GC'd mid-flight
        if not t._pricing_task.done():
            t._pricing_task.cancel()
