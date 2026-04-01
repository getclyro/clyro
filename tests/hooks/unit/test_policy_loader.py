"""Unit tests for policy loader."""

from datetime import UTC, datetime, timedelta

from clyro.hooks.config import HookConfig
from clyro.hooks.models import PolicyCache, SessionState
from clyro.hooks.policy_loader import _cache_is_fresh, _policies_from_cache, get_merged_policies


class TestCacheIsFresh:
    def test_empty_cache_is_stale(self):
        cache = PolicyCache()
        assert not _cache_is_fresh(cache)

    def test_recent_cache_is_fresh(self):
        cache = PolicyCache(
            fetched_at=datetime.now(UTC),
            ttl_seconds=300,
        )
        assert _cache_is_fresh(cache)

    def test_old_cache_is_stale(self):
        cache = PolicyCache(
            fetched_at=datetime.now(UTC) - timedelta(seconds=600),
            ttl_seconds=300,
        )
        assert not _cache_is_fresh(cache)


class TestPoliciesFromCache:
    def test_reconstructs_rules(self):
        cache = PolicyCache(
            merged_policies=[
                {"parameter": "command", "operator": "contains", "value": "rm -rf"},
            ],
        )
        rules = _policies_from_cache(cache)
        assert len(rules) == 1
        assert rules[0].parameter == "command"

    def test_corrupt_entry_invalidates_cache(self):
        """A corrupt entry should invalidate the entire cache (fail-closed)."""
        cache = PolicyCache(
            merged_policies=[
                {"invalid": "entry"},
                {"parameter": "command", "operator": "contains", "value": "rm"},
            ],
        )
        result = _policies_from_cache(cache)
        assert result is None  # Entire cache invalidated


class TestGetMergedPolicies:
    def test_local_only_no_api_key(self):
        config = HookConfig.model_validate({
            "global": {
                "policies": [
                    {"parameter": "command", "operator": "contains", "value": "rm -rf"},
                ],
            },
            "audit": {},
            "backend": {"api_key": None},
        })
        state = SessionState(session_id="test")

        policies = get_merged_policies(config, state)
        assert len(policies) == 1
        assert policies[0].value == "rm -rf"

    def test_local_only_cloud_disabled(self):
        config = HookConfig.model_validate({
            "global": {
                "policies": [
                    {"parameter": "command", "operator": "contains", "value": "rm"},
                ],
            },
            "audit": {},
            "backend": {"api_key": "test-key"},
        })
        state = SessionState(session_id="test", cloud_disabled=True)

        policies = get_merged_policies(config, state)
        assert len(policies) == 1

    def test_uses_cache_when_fresh(self):
        config = HookConfig.model_validate({
            "global": {"policies": []},
            "audit": {},
            "backend": {"api_key": "test-key"},
        })
        state = SessionState(
            session_id="test",
            agent_id="test-agent-id",
            policy_cache=PolicyCache(
                fetched_at=datetime.now(UTC),
                ttl_seconds=300,
                merged_policies=[
                    {"parameter": "command", "operator": "contains",
                     "value": "cached-value"},
                ],
            ),
        )

        policies = get_merged_policies(config, state)
        assert len(policies) == 1
        assert policies[0].value == "cached-value"

    def test_no_agent_id_falls_back_to_local(self):
        """Without agent_id, should use local policies even with API key."""
        config = HookConfig.model_validate({
            "global": {
                "policies": [
                    {"parameter": "command", "operator": "contains", "value": "rm"},
                ],
            },
            "audit": {},
            "backend": {"api_key": "test-key"},
        })
        state = SessionState(session_id="test", agent_id=None)

        policies = get_merged_policies(config, state)
        assert len(policies) == 1
        assert policies[0].value == "rm"
