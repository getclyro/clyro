# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — HTTP Sync Client
# Implements FRD-015, FRD-016

"""
HTTP client for backend API communication with retry logic.

Endpoints:
- ``POST /v1/traces`` — Send trace event batches (FRD-015)
- ``POST /v1/agents/register`` — Auto-register MCP agent (FRD-016)

Retry strategy: exponential backoff 1s → 2s → 4s, max 3 retries.
Supports ``Retry-After`` header for 429 responses.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

# Implements FRD-014: exceptions moved to clyro.exceptions
from clyro.config import DEFAULT_API_URL
from clyro.exceptions import AuthenticationError, RateLimitExhaustedError  # noqa: F401


class HttpSyncClient:
    """
    HTTP client for Clyro backend API communication.

    API key is stored only in this instance — never logged, never
    included in trace events or audit entries (NFR-010).

    Args:
        api_key: Clyro API key for authentication.
        base_url: Backend API base URL (no trailing slash).
        timeout: Per-request timeout in seconds.
    """

    _BACKOFF_DELAYS = [1, 2, 4]  # FRD-019: exponential backoff

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_URL,
        timeout: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"X-Clyro-API-Key": api_key},
        )

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        handle_429: bool = True,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry.

        Retries on network errors, timeouts, and 5xx responses.
        Raises :class:`AuthenticationError` on 401/403.
        Optionally handles 429 rate-limit responses with Retry-After.

        Args:
            method: HTTP method (``"POST"`` or ``"GET"``).
            path: URL path (appended to ``base_url``).
            body: JSON request body (for POST).
            params: Query parameters (for GET).
            handle_429: If ``True``, retry on 429 with Retry-After header.

        Returns:
            Successful :class:`httpx.Response`.
        """
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt, delay in enumerate(self._BACKOFF_DELAYS):
            try:
                if method == "GET":
                    response = await self._client.get(url, params=params)
                else:
                    response = await self._client.post(url, json=body)

                if handle_429 and response.status_code == 429:
                    try:
                        retry_after = int(response.headers.get("Retry-After", str(delay)))
                        retry_after = max(1, min(retry_after, 60))
                    except (ValueError, TypeError):
                        retry_after = delay
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code in (401, 403):
                    raise AuthenticationError(response.status_code)

                response.raise_for_status()
                return response

            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < len(self._BACKOFF_DELAYS) - 1:
                    await asyncio.sleep(delay)
                    continue
                raise

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code >= 500:
                    if attempt < len(self._BACKOFF_DELAYS) - 1:
                        await asyncio.sleep(delay)
                        continue
                raise

        # All retries exhausted (429 loop or unexpected)
        if last_exc:
            raise last_exc
        raise RateLimitExhaustedError()

    async def send_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Send trace events to backend (FRD-015).

        Retries on network errors and 5xx. Raises AuthenticationError
        on 401/403. Supports Retry-After for 429 responses.

        Returns:
            Response payload: ``{"accepted": int, "rejected": int, "errors": []}``
        """
        response = await self._request_with_retry(
            "POST", "/v1/traces", body={"events": events},
        )
        return response.json()

    async def register_agent(self, agent_name: str, framework: str = "mcp") -> str:
        """
        Register MCP agent with backend (FRD-016).

        Retries on network errors and 5xx with same backoff as send_batch.
        Raises AuthenticationError on 401/403.

        Returns:
            Agent ID as string (UUID).
        """
        response = await self._request_with_retry(
            "POST",
            "/v1/agents/register",
            body={
                "agent_name": agent_name,
                "framework": framework,
                "description": "MCP-wrapped server",
            },
            handle_429=False,
        )
        return response.json()["agent_id"]

    async def fetch_policies(self, agent_id: str) -> dict[str, Any]:
        """
        Fetch cloud policies for an agent (FRD-017).

        Returns:
            Response payload with ``policies`` key.
        """
        response = await self._request_with_retry(
            "GET",
            f"/v1/agents/{agent_id}/policies",
            params={"enabled": "true"},
            handle_429=False,
        )
        return response.json()

    async def report_violations(self, violations: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Report policy violations to backend (FRD-006).

        Same retry strategy as send_batch: exponential backoff 1s → 2s → 4s.
        Raises AuthenticationError on 401/403.

        Returns:
            Response payload: ``{"accepted": int, "rejected": int, "errors": []}``
        """
        response = await self._request_with_retry(
            "POST", "/v1/policy-violations", body={"violations": violations},
        )
        return response.json()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
