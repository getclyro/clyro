# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Pricing Catalog Cache (C6 consumer)
# Implements the SDK side of the C6 Pricing Distribution API

"""
SDK-side cache of the backend pricing catalog (C6 consumer).

The backend serves resolved pricing at ``GET /v1/pricing`` (C6); this module
pulls that set once per process (refreshed on the catalog cadence) and caches it
in memory, so every adapter's cost computation can price models from the live
catalog instead of the SDK's small bundled table. The cache is the **catalog**
tier of the full precedence ``custom > catalog > static > default`` (the resolver
lives in ``ClyroConfig.get_model_pricing``).

Resilience: a failed/empty refresh retains the last-known set (never wiped), and
``get`` returns ``None`` when nothing is loaded so the caller falls back to the
static/default tiers — pricing never blocks or raises.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Canonical-key normalization (parity with the backend C2 normalizer): strip the
# provider prefix, then a channel suffix, then a trailing ISO date.
_CHANNEL_SUFFIXES = (":free", ":beta", ":extended", "-latest")
_DATE_SUFFIX_RE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")


def _normalize(model: str) -> str:
    """Reduce a raw model id to the backend's canonical key (lowercase bare name).

    Close-parity (not byte-exact) with the backend C2 normalizer: it strips the
    provider segment by ``rsplit('/')`` rather than a known-prefix list, which
    matches real model ids but may differ on exotic ones.
    """
    n = model.strip().lower()
    if "/" in n:
        n = n.rsplit("/", 1)[-1]  # drop provider prefix(es), e.g. anthropic/claude -> claude
    for suffix in _CHANNEL_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return _DATE_SUFFIX_RE.sub("", n)


class PricingCatalogCache:
    """In-memory cache of backend catalog prices, keyed by canonical model key.  # Implements C6"""

    def __init__(self) -> None:
        self._prices: dict[str, tuple[Decimal, Decimal]] = {}
        self._loaded = False

    def update_from_payload(self, payload: Any) -> bool:
        """Replace the cache from a ``GET /v1/pricing`` response. Returns success.

        Per-record validation skips malformed entries; an empty/garbage payload is
        rejected so the last-known set is retained (never wiped with nothing).
        """
        if not isinstance(payload, dict):
            return False
        records = payload.get("records")
        if not isinstance(records, list):
            return False

        prices: dict[str, tuple[Decimal, Decimal]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            key = record.get("canonical_key")
            if not isinstance(key, str) or not key:
                continue
            try:
                input_per_1k = Decimal(str(record["input_per_1k"]))
                output_per_1k = Decimal(str(record["output_per_1k"]))
            except (KeyError, TypeError, ValueError, InvalidOperation):
                continue
            if input_per_1k < 0 or output_per_1k < 0:
                continue
            prices[key] = (input_per_1k, output_per_1k)

        if not prices:
            logger.warning("pricing_catalog_empty_payload", fail_open=True)
            return False  # keep last-known rather than wipe it

        self._prices = prices
        self._loaded = True
        logger.debug("pricing_catalog_loaded", model_count=len(prices))
        return True

    def get(self, raw_model: str) -> tuple[Decimal, Decimal] | None:
        """Return (input_per_1k, output_per_1k) for a raw model id, or None if absent."""
        if not self._prices or not raw_model:
            return None
        return self._prices.get(_normalize(raw_model))

    @property
    def loaded(self) -> bool:
        return self._loaded


# Module singleton — the SDK's shared catalog cache (consulted by get_model_pricing).
pricing_catalog_cache = PricingCatalogCache()


async def refresh_pricing_catalog(http_client: Any) -> bool:
    """Pull ``GET /v1/pricing`` into the cache (C6 consumer). Fail-open.

    ``http_client`` must expose ``fetch_pricing()``. Any failure is swallowed and
    the last-known set is retained, so refreshing pricing can never break the agent.
    """
    try:
        payload = await http_client.fetch_pricing()
    except Exception as e:
        logger.warning("pricing_catalog_fetch_failed", error=str(e), fail_open=True)
        return False
    return pricing_catalog_cache.update_from_payload(payload)


async def refresh_pricing_catalog_from_config(config: Any) -> bool:
    """Pull the C6 catalog using a config's backend credentials. Idempotent, fail-open.

    Provider-agnostic entry point called from the transport's shared background-sync
    startup (the common backend-startup path for every adapter), so all adapters
    price from the live catalog. No-op in local mode, without an api_key, or once
    the catalog is already loaded.
    """
    if config is None or pricing_catalog_cache.loaded:
        return False
    if not getattr(config, "api_key", None) or getattr(config, "is_local_only", lambda: True)():
        return False

    from clyro.backend.http_client import HttpSyncClient

    client = HttpSyncClient(api_key=config.api_key, base_url=config.endpoint)
    try:
        return await refresh_pricing_catalog(client)
    finally:
        try:
            await client.close()
        except Exception:
            pass
