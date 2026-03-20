"""
Unit tests for CloudPolicyFetcher — TDD §11.1 v1.1 tests.

FRD-017: Fetch cloud policies and merge with local YAML policies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clyro.backend.cloud_policy import CloudPolicyFetcher
from clyro.backend.http_client import HttpSyncClient
from clyro.config import PolicyRule


@pytest.fixture
def http_client() -> HttpSyncClient:
    return MagicMock(spec=HttpSyncClient)


@pytest.fixture
def fetcher(http_client) -> CloudPolicyFetcher:
    return CloudPolicyFetcher(http_client=http_client)


def _make_local_policy(name: str, param: str = "amount", op: str = "max_value", val=1000) -> PolicyRule:
    return PolicyRule(parameter=param, operator=op, value=val, name=name)


class TestCloudPolicyFetchSuccess:
    """Fetch and merge cloud policies (FRD-017)."""

    @pytest.mark.asyncio
    async def test_returns_local_when_no_cloud(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        http_client.fetch_policies = AsyncMock(return_value={"policies": []})
        local = [_make_local_policy("local-rule")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        assert len(result) == 1
        assert result[0].name == "local-rule"

    @pytest.mark.asyncio
    async def test_merges_cloud_rules(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "name": "cloud-policy",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "cloud-rule",
                        "condition": {"field": "price", "operator": "max_value", "value": 500},
                        "action": "block",
                    }],
                },
            }],
        })
        local = [_make_local_policy("local-rule")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        assert len(result) == 2
        names = {r.name for r in result}
        assert "local-rule" in names
        assert "cloud-rule" in names


class TestCloudPolicyLocalOverride:
    """Local policies override cloud policies with same name (FRD-017)."""

    @pytest.mark.asyncio
    async def test_local_overrides_cloud_by_name(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "name": "shared-policy",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "shared-rule",
                        "condition": {"field": "amount", "operator": "max_value", "value": 9999},
                        "action": "block",
                    }],
                },
            }],
        })
        local = [_make_local_policy("shared-rule", val=100)]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        # Only local rule should be present (cloud skipped because same name)
        assert len(result) == 1
        assert result[0].value == 100


class TestCloudPolicyUnsupportedOperator:
    """Unsupported operators skipped with warning (FRD-017)."""

    @pytest.mark.asyncio
    async def test_skips_unsupported_operator(self, fetcher: CloudPolicyFetcher, http_client, capsys) -> None:
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "name": "p1",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "unsupported-rule",
                        "condition": {"field": "x", "operator": "requires_approval", "value": True},
                        "action": "require_approval",
                    }],
                },
            }],
        })
        result = await fetcher.fetch_and_merge("agent-1", [], timeout=2.0)
        assert len(result) == 0
        captured = capsys.readouterr()
        assert "cloud_policy_unsupported_operator" in captured.err


class TestCloudPolicyFailOpen:
    """Fail-open: return local policies on any failure (FRD-017, NFR-007)."""

    @pytest.mark.asyncio
    async def test_returns_local_on_network_error(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        http_client.fetch_policies = AsyncMock(side_effect=Exception("network error"))
        local = [_make_local_policy("local")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        assert len(result) == 1
        assert result[0].name == "local"

    @pytest.mark.asyncio
    async def test_returns_local_on_timeout(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        async def slow_fetch(*args, **kwargs):
            import asyncio
            await asyncio.sleep(10)  # Way beyond timeout
            return {"policies": []}

        http_client.fetch_policies = slow_fetch
        local = [_make_local_policy("local")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=0.01)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_local_when_no_agent_id(self, fetcher: CloudPolicyFetcher) -> None:
        local = [_make_local_policy("local")]
        result = await fetcher.fetch_and_merge(None, local, timeout=2.0)
        assert result == local


class TestCloudPolicyUnexpectedError:
    """Unexpected exceptions still fail-open but log distinctly (NFR-007, M1 fix)."""

    @pytest.mark.asyncio
    async def test_unexpected_error_logs_type_name(self, fetcher: CloudPolicyFetcher, http_client, capsys) -> None:
        http_client.fetch_policies = AsyncMock(side_effect=RuntimeError("unexpected"))
        local = [_make_local_policy("local")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        assert len(result) == 1
        assert result[0].name == "local"
        captured = capsys.readouterr()
        assert "cloud_policy_fetch_unexpected" in captured.err
        assert "RuntimeError" in captured.err


class TestCloudPolicyIdPropagation:
    """policy_id propagated from backend through extract/merge (FRD-006)."""

    @pytest.mark.asyncio
    async def test_cloud_rule_has_policy_id(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        """Cloud policy ID should be propagated to PolicyRule.policy_id."""
        policy_uuid = "00000000-1111-2222-3333-444444444444"
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "id": policy_uuid,
                "name": "cloud-policy",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "cloud-rule",
                        "condition": {"field": "amount", "operator": "max_value", "value": 500},
                        "action": "block",
                    }],
                },
            }],
        })
        result = await fetcher.fetch_and_merge("agent-1", [], timeout=2.0)
        assert len(result) == 1
        assert result[0].policy_id == policy_uuid

    @pytest.mark.asyncio
    async def test_local_rule_has_no_policy_id(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        """Local YAML policies should have policy_id=None."""
        http_client.fetch_policies = AsyncMock(return_value={"policies": []})
        local = [_make_local_policy("local-rule")]
        result = await fetcher.fetch_and_merge("agent-1", local, timeout=2.0)
        assert len(result) == 1
        assert result[0].policy_id is None

    @pytest.mark.asyncio
    async def test_missing_id_produces_empty_string_policy_id(
        self, fetcher: CloudPolicyFetcher, http_client
    ) -> None:
        """Policy without 'id' field should produce empty string → None policy_id."""
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "name": "no-id-policy",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "no-id-rule",
                        "condition": {"field": "x", "operator": "max_value", "value": 10},
                        "action": "block",
                    }],
                },
            }],
        })
        result = await fetcher.fetch_and_merge("agent-1", [], timeout=2.0)
        assert len(result) == 1
        # Empty string from str(policy.get("id", "")) → filtered to None by `or None`
        assert result[0].policy_id is None


class TestCloudPolicyApprovalConversion:
    """require_approval actions converted to block (§10)."""

    @pytest.mark.asyncio
    async def test_converts_approval_to_block(self, fetcher: CloudPolicyFetcher, http_client) -> None:
        http_client.fetch_policies = AsyncMock(return_value={
            "policies": [{
                "name": "p1",
                "rules": {
                    "version": "1.0",
                    "rules": [{
                        "name": "approval-rule",
                        "condition": {"field": "amount", "operator": "max_value", "value": 500},
                        "action": "require_approval",
                    }],
                },
            }],
        })
        result = await fetcher.fetch_and_merge("agent-1", [], timeout=2.0)
        # Rule should be included (operator is valid — max_value), action converted
        assert len(result) == 1
        assert result[0].operator == "max_value"
