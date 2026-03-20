"""
Unit tests for HttpSyncClient — TDD §11.1 v1.1 tests.

FRD-015: HTTP client for POST /v1/traces with retry.
FRD-016: HTTP client for POST /v1/agents/register.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from clyro.backend.http_client import (
    AuthenticationError,
    HttpSyncClient,
    RateLimitExhaustedError,
)


@pytest.fixture
def client() -> HttpSyncClient:
    return HttpSyncClient(
        api_key="test-api-key",
        base_url="https://api.test.dev",
        timeout=1.0,
    )


class TestHttpSyncClientInit:
    """Client initialization."""

    def test_creates_client_with_headers(self, client: HttpSyncClient) -> None:
        assert client._api_key == "test-api-key"
        assert client._base_url == "https://api.test.dev"

    def test_strips_trailing_slash(self) -> None:
        c = HttpSyncClient(api_key="k", base_url="https://api.test.dev/")
        assert c._base_url == "https://api.test.dev"


class TestSendBatch:
    """POST /v1/traces with retry logic (FRD-015, FRD-019)."""

    @pytest.mark.asyncio
    async def test_success(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"accepted": 5, "rejected": 0, "errors": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.send_batch([{"event_id": "1"}])

        assert result["accepted"] == 5

    @pytest.mark.asyncio
    async def test_auth_error_401(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(AuthenticationError) as exc_info:
                await client.send_batch([])
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_error_403(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(AuthenticationError):
                await client.send_batch([])

    @pytest.mark.asyncio
    async def test_retries_on_network_error(self, client: HttpSyncClient) -> None:
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"accepted": 1}
        success_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                httpx.NetworkError("conn refused"),
                success_response,
            ]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.send_batch([{"event_id": "1"}])
        assert result["accepted"] == 1

    @pytest.mark.asyncio
    async def test_retries_on_5xx(self, client: HttpSyncClient) -> None:
        err_response = MagicMock()
        err_response.status_code = 503
        err_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=err_response
        )

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"accepted": 1}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [err_response, ok_response]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.send_batch([{"event_id": "1"}])
        assert result["accepted"] == 1

    @pytest.mark.asyncio
    async def test_retries_429_with_retry_after(self, client: HttpSyncClient) -> None:
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "2"}

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"accepted": 1}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [rate_limited, ok_response]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await client.send_batch([{"event_id": "1"}])
                mock_sleep.assert_called_once_with(2)
        assert result["accepted"] == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self, client: HttpSyncClient) -> None:
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.NetworkError("conn refused")
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.NetworkError):
                    await client.send_batch([])

    @pytest.mark.asyncio
    async def test_429_exhaustion_raises_rate_limit_error(self, client: HttpSyncClient) -> None:
        """All 3 retries getting 429 should raise RateLimitExhaustedError, not return {}."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = rate_limited
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RateLimitExhaustedError):
                    await client.send_batch([{"event_id": "1"}])


class TestRegisterAgent:
    """POST /v1/agents/register (FRD-016)."""

    @pytest.mark.asyncio
    async def test_register_success(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agent_id": "00000000-0000-0000-0000-000000000001",
            "agent_name": "test-agent",
            "created": True,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            agent_id = await client.register_agent("test-agent")
        assert agent_id == "00000000-0000-0000-0000-000000000001"

    @pytest.mark.asyncio
    async def test_register_auth_error(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(AuthenticationError):
                await client.register_agent("test-agent")

    @pytest.mark.asyncio
    async def test_register_retries_on_network_error(self, client: HttpSyncClient) -> None:
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"agent_id": "agent-123"}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                httpx.NetworkError("conn refused"),
                ok_response,
            ]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                agent_id = await client.register_agent("test-agent")
        assert agent_id == "agent-123"

    @pytest.mark.asyncio
    async def test_register_retries_on_5xx(self, client: HttpSyncClient) -> None:
        err_response = MagicMock()
        err_response.status_code = 502
        err_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "502", request=MagicMock(), response=err_response
        )

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {"agent_id": "agent-456"}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [err_response, ok_response]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                agent_id = await client.register_agent("test-agent")
        assert agent_id == "agent-456"


class TestFetchPolicies:
    """GET /v1/agents/{id}/policies (FRD-017)."""

    @pytest.mark.asyncio
    async def test_fetch_policies_success(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "policies": [{"name": "p1", "rules": {"version": "1.0", "rules": []}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.fetch_policies("agent-id-123")
        assert len(result["policies"]) == 1


class TestReportViolations:
    """POST /v1/policy-violations (FRD-006)."""

    @pytest.mark.asyncio
    async def test_success(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"accepted": 1, "rejected": 0, "errors": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.report_violations([{"rule_id": "r1"}])

        assert result["accepted"] == 1
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert "/v1/policy-violations" in call_url

    @pytest.mark.asyncio
    async def test_auth_error_401(self, client: HttpSyncClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(AuthenticationError):
                await client.report_violations([])

    @pytest.mark.asyncio
    async def test_retries_on_network_error(self, client: HttpSyncClient) -> None:
        ok_response = MagicMock()
        ok_response.status_code = 201
        ok_response.json.return_value = {"accepted": 1}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                httpx.NetworkError("conn refused"),
                ok_response,
            ]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.report_violations([{"rule_id": "r1"}])
        assert result["accepted"] == 1

    @pytest.mark.asyncio
    async def test_429_retries(self, client: HttpSyncClient) -> None:
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        ok_response = MagicMock()
        ok_response.status_code = 201
        ok_response.json.return_value = {"accepted": 1}
        ok_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [rate_limited, ok_response]
            with patch("clyro.backend.http_client.asyncio.sleep", new_callable=AsyncMock):
                result = await client.report_violations([{"rule_id": "r1"}])
        assert result["accepted"] == 1


class TestClientClose:
    """Close the HTTP client."""

    @pytest.mark.asyncio
    async def test_close(self, client: HttpSyncClient) -> None:
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()
