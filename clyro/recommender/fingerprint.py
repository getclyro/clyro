# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — agent fingerprint
# Implements policy-recommender FRD-PR-003

"""Deterministic SHA-256 fingerprint of an agent's introspected shape.

Re-running ``clyro suggest`` on unchanged agent code yields the same fingerprint,
so the cache (FRD-PR-016) is a hit. A catalogue revision changes the
``catalogue_version`` digest and therefore the fingerprint, forcing recomputation
(freshness wins — TDD §7.1).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from clyro.recommender.types import ToolSurface


def _canonical_default(obj: Any) -> str:
    """Coerce a non-JSON-serializable node via ``repr`` (FRD-PR-003 failure clause).

    The fingerprint MUST always be computable — it is on the cache-lookup path.
    """
    return repr(obj)


def compute_fingerprint(
    surface: ToolSurface,
    system_prompt: str,
    catalogue_version: str,
) -> str:
    """Return a 64-char lowercase hex SHA-256 of the canonical agent shape."""
    canonical = {
        "framework": surface.framework,
        "tools": sorted(
            (
                {
                    "name": t.name,
                    "description": t.description,
                    "args_schema": t.args_schema,
                }
                for t in surface.tools
            ),
            key=lambda t: t["name"],
        ),
        "system_prompt": (system_prompt or "").strip(),
        "topology": {
            "node_count": surface.topology.node_count,
            "multi_agent": surface.topology.multi_agent,
            "has_rag": surface.topology.has_rag,
            "has_mcp": surface.topology.has_mcp,
        },
        "catalogue_version": catalogue_version,
    }
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        default=_canonical_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
