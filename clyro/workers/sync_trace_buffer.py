# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Kick off background sync by wrapping a dummy agent and waiting.

This leverages existing SDK behavior to drain trace_buffer in the background
when the backend is reachable.

"""

from __future__ import annotations

import time

import httpx
from dotenv import load_dotenv

from clyro.config import ClyroConfig
from clyro.wrapper import wrap

load_dotenv()


def _backend_available(endpoint: str) -> bool:
    try:
        response = httpx.get(f"{endpoint.rstrip('/')}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def main() -> int:
    config = ClyroConfig.from_env()
    if not config.api_key or not config.api_key.strip():
        print("api_key is required.")
        return 2

    if not _backend_available(config.endpoint):
        print(f"Backend not reachable at {config.endpoint}")
        return 2

    if config.is_local_only():
        print("Local-only mode: api_key is required to sync.")
        return 2

    def _dummy_agent() -> str:
        return "ok"

    wrapped = wrap(_dummy_agent, config=config)
    wrapped()

    time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
