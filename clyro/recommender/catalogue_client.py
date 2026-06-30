# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — catalogue client
# Implements policy-recommender FRD-PR-003, FRD-PR-010 (TDD §2.1 C8, §4.5/F-9)

"""Fetch the public catalogue and cache a local snapshot.

The catalogue endpoints (``/v1/agent-types``, ``/v1/concerns``, ``/v1/kits``) are
public (no api_key — F-9), so ``clyro suggest`` works with no Clyro account. The
fetched snapshot is cached at ``~/.clyro/catalogue-snapshot.json`` so offline
re-runs still work; only the first run needs connectivity.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENDPOINTS = ("agent-types", "concerns", "kits")
_SNAPSHOT_PATH = Path.home() / ".clyro" / "catalogue-snapshot.json"


@dataclass
class CatalogueSnapshot:
    """Ids + kit definitions + a deterministic version digest."""

    agent_type_ids: set[str]
    concern_ids: set[str]
    kit_ids: set[str]
    kits: list[dict[str, Any]] = field(default_factory=list)
    version: str = ""
    source: str = "remote"  # remote | cache

    def is_valid_id(self, value: str) -> bool:
        return value in self.agent_type_ids or value in self.concern_ids or value in self.kit_ids


def _coerce_version(value: Any) -> int:
    """Best-effort int version; non-numeric values fall back to 1 (defensive)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _version_digest(items_by_entity: dict[str, list[dict[str, Any]]]) -> str:
    """Deterministic digest of per-record versions (FRD-PR-003 — no global semver)."""
    triples = sorted(
        (entity, str(item.get("id", "")), _coerce_version(item.get("version", 1)))
        for entity, items in items_by_entity.items()
        for item in items
        if isinstance(item, dict)
    )
    return hashlib.sha256(json.dumps(triples).encode()).hexdigest()[:16]


def _build_snapshot(
    items_by_entity: dict[str, list[dict[str, Any]]], source: str
) -> CatalogueSnapshot:
    return CatalogueSnapshot(
        agent_type_ids={i["id"] for i in items_by_entity.get("agent-types", [])},
        concern_ids={i["id"] for i in items_by_entity.get("concerns", [])},
        kit_ids={i["id"] for i in items_by_entity.get("kits", [])},
        kits=items_by_entity.get("kits", []),
        version=_version_digest(items_by_entity),
        source=source,
    )


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    from clyro import __version__  # local import avoids a circular import at module load

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            # urllib's default "Python-urllib/x" UA is banned by Cloudflare's default
            # bot rules (403, error code 1010) — identify as the SDK, same as the
            # prefill POST does. Without this the first-run catalogue fetch is blocked.
            "User-Agent": f"clyro-sdk/{__version__}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted base_url)
        return json.loads(resp.read().decode("utf-8"))


class CatalogueClient:
    """Fetch the public catalogue with an offline snapshot fallback."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def fetch(self) -> CatalogueSnapshot:
        """Return a snapshot from the network, falling back to the local cache."""
        try:
            items_by_entity: dict[str, list[dict[str, Any]]] = {}
            for entity in _ENDPOINTS:
                data = _http_get_json(f"{self._base_url}/v1/{entity}", self._timeout)
                items_by_entity[entity] = data.get("items", [])
            snapshot = _build_snapshot(items_by_entity, source="remote")
            self._write_cache(items_by_entity)
            return snapshot
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            cached = self._read_cache()
            if cached is not None:
                return cached
            raise

    # -- local snapshot cache -------------------------------------------------

    def _write_cache(self, items_by_entity: dict[str, list[dict[str, Any]]]) -> None:
        try:
            _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _SNAPSHOT_PATH.write_text(json.dumps(items_by_entity))
        except OSError:
            pass  # cache is best-effort; never fail the fetch on a write error

    def _read_cache(self) -> CatalogueSnapshot | None:
        try:
            items_by_entity = json.loads(_SNAPSHOT_PATH.read_text())
        except (OSError, ValueError):
            return None
        return _build_snapshot(items_by_entity, source="cache")
