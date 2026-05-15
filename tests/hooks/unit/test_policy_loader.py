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
                {"action": "block", "parameter": "command", "operator": "contains", "value": "rm -rf"},
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
                {"action": "block", "parameter": "command", "operator": "contains", "value": "rm"},
            ],
        )
        result = _policies_from_cache(cache)
        assert result is None  # Entire cache invalidated


class TestGetMergedPolicies:
    def test_local_only_no_api_key(self):
        config = HookConfig.model_validate({"default_action": "allow",
            "global": {
                "policies": [
                    {"action": "block", "parameter": "command", "operator": "contains", "value": "rm -rf"},
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
        config = HookConfig.model_validate({"default_action": "allow",
            "global": {
                "policies": [
                    {"action": "block", "parameter": "command", "operator": "contains", "value": "rm"},
                ],
            },
            "audit": {},
            "backend": {"api_key": "test-key"},
        })
        state = SessionState(session_id="test", cloud_disabled=True)

        policies = get_merged_policies(config, state)
        assert len(policies) == 1

    def test_uses_cache_when_fresh(self):
        config = HookConfig.model_validate({"default_action": "allow",
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
                    {"action": "block", "parameter": "command", "operator": "contains",
                     "value": "cached-value"},
                ],
            ),
        )

        policies = get_merged_policies(config, state)
        assert len(policies) == 1
        assert policies[0].value == "cached-value"

    def test_no_agent_id_falls_back_to_local(self):
        """Without agent_id, should use local policies even with API key."""
        config = HookConfig.model_validate({"default_action": "allow",
            "global": {
                "policies": [
                    {"action": "block", "parameter": "command", "operator": "contains", "value": "rm"},
                ],
            },
            "audit": {},
            "backend": {"api_key": "test-key"},
        })
        state = SessionState(session_id="test", agent_id=None)

        policies = get_merged_policies(config, state)
        assert len(policies) == 1
        assert policies[0].value == "rm"


class TestCacheDefaultActionPersistence:
    """Cache must preserve the resolved default_action across event boundaries.

    Hooks server loads a fresh HookConfig on every Claude Code event. On the
    first event we fetch from the cloud and resolve `default_action` via
    cloud-wins precedence; subsequent events within the cache TTL must restore
    that same resolved value — otherwise the cloud's centrally-mandated
    `default_action` would silently revert to the local YAML's default on
    every cache hit.
    """

    def test_cache_hit_restores_resolved_default_action(self):
        """Cached resolved_default_action overrides the freshly-loaded config default."""
        config = HookConfig.model_validate({
            "default_action": "allow",        # local YAML default
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
                    {"action": "allow", "parameter": "file_path",
                     "operator": "in_list", "value": [".env"]},
                ],
                resolved_default_action="block",  # cloud said block
            ),
        )

        get_merged_policies(config, state)
        # Cache hit must propagate the cloud's stricter default
        assert config.default_action == "block"

    def test_cache_hit_keeps_local_default_when_resolved_is_allow(self):
        """Cached resolved 'allow' still overrides local — symmetric to above."""
        config = HookConfig.model_validate({
            "default_action": "block",
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
                merged_policies=[],
                resolved_default_action="allow",
            ),
        )

        get_merged_policies(config, state)
        assert config.default_action == "allow"

    def test_cache_hit_without_resolved_default_leaves_local_unchanged(self):
        """Legacy cache entries (pre-fix) have no resolved_default_action; don't crash."""
        config = HookConfig.model_validate({
            "default_action": "allow",
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
                merged_policies=[],
                resolved_default_action=None,  # legacy cache, no field stored
            ),
        )

        get_merged_policies(config, state)
        # Falls back to local YAML's default; no error
        assert config.default_action == "allow"

    def test_cache_hit_ignores_invalid_resolved_value(self):
        """A corrupted cache value (not 'block'/'allow') is treated as missing."""
        config = HookConfig.model_validate({
            "default_action": "allow",
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
                merged_policies=[],
                resolved_default_action="garbage",
            ),
        )

        get_merged_policies(config, state)
        assert config.default_action == "allow"
