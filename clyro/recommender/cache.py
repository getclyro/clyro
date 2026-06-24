# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — fingerprint cache (self-hosted SQLite)
# Implements policy-recommender FRD-PR-016

"""Local SQLite cache of recommendation payloads keyed by agent fingerprint.

A cache hit skips the (uncached) LLM proposer call entirely. Failures never block
the recommendation flow — a broken cache simply degrades to recompute.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path.home() / ".clyro" / "proposer-cache.db"


class FingerprintCache:
    """SQLite-backed cache; ``get``/``put`` are best-effort and never raise."""

    def __init__(self, path: Path | None = None, ttl_days: int = 7):
        self._path = path or _DEFAULT_PATH
        self._ttl_seconds = ttl_days * 86400
        self._available = self._init_db()

    def _init_db(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS proposer_cache ("
                    "fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                    "computed_at REAL NOT NULL)"
                )
            return True
        except (sqlite3.Error, OSError):
            return False

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        if not self._available:
            return None
        try:
            with sqlite3.connect(self._path) as conn:
                row = conn.execute(
                    "SELECT payload, computed_at FROM proposer_cache WHERE fingerprint = ?",
                    (fingerprint,),
                ).fetchone()
            if row is None:
                return None
            payload_json, computed_at = row
            if time.time() - computed_at > self._ttl_seconds:
                return None  # expired
            return json.loads(payload_json)
        except (sqlite3.Error, OSError, ValueError):
            return None

    def put(self, fingerprint: str, payload: dict[str, Any]) -> None:
        if not self._available:
            return
        try:
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO proposer_cache "
                    "(fingerprint, payload, computed_at) VALUES (?, ?, ?)",
                    (fingerprint, json.dumps(payload), time.time()),
                )
        except (sqlite3.Error, OSError, TypeError):
            pass
