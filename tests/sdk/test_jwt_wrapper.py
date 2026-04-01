# Tests for JWT API Key Integration in SDK Wrapper
# Tests org_id extraction from JWT keys and agent_id generation

"""
Tests for SDK wrapper JWT API key integration.

These tests verify:
- org_id extraction from JWT API keys
- agent_id generation with org_id namespace
- Proper error handling when org_id is missing
"""

from uuid import UUID, uuid4, uuid5

import pytest

from clyro import wrap
from clyro.config import ClyroConfig
from clyro.exceptions import ClyroWrapError
from clyro.wrapper import _extract_org_id_from_jwt_api_key, _generate_agent_id_from_name


def _create_mock_jwt_key(org_id: UUID) -> str:
    """
    Create a mock JWT API key with embedded org_id for testing.

    This creates a minimal JWT-like key that can be parsed by the SDK.
    """
    import base64
    import json

    payload = {
        "org_id": str(org_id),
        "key_id": str(uuid4()),
        "env": "test",
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")

    # Mock signature (SDK doesn't verify)
    signature = "a" * 32

    return f"cly_test_{payload_b64}.{signature}"


class TestExtractOrgIDFromJWTKey:
    """Test org_id extraction from JWT API keys."""

    def test_extract_org_id_from_valid_jwt_key(self):
        """Test extracting org_id from valid JWT key."""
        org_id = uuid4()
        jwt_key = _create_mock_jwt_key(org_id)

        extracted_org_id = _extract_org_id_from_jwt_api_key(jwt_key)

        assert extracted_org_id == org_id

    def test_extract_org_id_from_invalid_key_returns_none(self):
        """Test that invalid key returns None."""
        assert _extract_org_id_from_jwt_api_key("invalid") is None
        assert _extract_org_id_from_jwt_api_key("cly_test_notjwt") is None
        assert _extract_org_id_from_jwt_api_key("") is None


class TestGenerateAgentIDWithOrgID:
    """Test agent_id generation with org_id namespace."""

    def test_generate_agent_id_is_deterministic(self):
        """Test that same inputs produce same agent_id."""
        org_id = uuid4()
        agent_name = "test-agent"

        agent_id1 = _generate_agent_id_from_name(agent_name, org_id)
        agent_id2 = _generate_agent_id_from_name(agent_name, org_id)

        assert agent_id1 == agent_id2

    def test_different_org_ids_produce_different_agent_ids(self):
        """Test that different org_ids produce different agent_ids."""
        org_id1 = uuid4()
        org_id2 = uuid4()
        agent_name = "test-agent"

        agent_id1 = _generate_agent_id_from_name(agent_name, org_id1)
        agent_id2 = _generate_agent_id_from_name(agent_name, org_id2)

        assert agent_id1 != agent_id2

    def test_generate_agent_id_uses_uuid5(self):
        """Test that agent_id generation uses UUID5."""
        org_id = uuid4()
        agent_name = "test-agent"

        agent_id = _generate_agent_id_from_name(agent_name, org_id)
        expected_id = uuid5(org_id, agent_name.lower())

        assert agent_id == expected_id


class TestWrapWithJWTKey:
    """Test wrapping agents with JWT API keys."""

    def test_wrap_with_jwt_key_extracts_org_id(self):
        """Test that wrapping with JWT key extracts org_id automatically."""
        org_id = uuid4()
        jwt_key = _create_mock_jwt_key(org_id)

        config = ClyroConfig(
            agent_name="test-agent",
            api_key=jwt_key,
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped = wrap(my_agent, config=config)

        # org_id should be extracted from JWT key
        assert wrapped._org_id == org_id

        # agent_id should be generated with org_id namespace
        expected_agent_id = uuid5(org_id, "test-agent".lower())
        assert wrapped._agent_id == expected_agent_id

    def test_wrap_with_explicit_org_id_overrides_key(self):
        """Test that explicit org_id overrides JWT key org_id."""
        key_org_id = uuid4()
        explicit_org_id = uuid4()
        jwt_key = _create_mock_jwt_key(key_org_id)

        config = ClyroConfig(
            agent_name="test-agent",
            api_key=jwt_key,
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped = wrap(my_agent, config=config, org_id=explicit_org_id)

        # Explicit org_id should take precedence
        assert wrapped._org_id == explicit_org_id


class TestWrapWithoutOrgID:
    """Test wrapping behavior when org_id cannot be determined."""

    def test_wrap_with_agent_name_but_no_org_id_succeeds_local_mode(self):
        """Test that wrapping with agent_name but no org_id works in local mode."""
        config = ClyroConfig(
            agent_name="test-agent",
            api_key=None,  # No API key → auto-resolves to local mode
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        # In local mode, agent_name works without org_id
        wrapped = wrap(my_agent, config=config)
        assert wrapped._agent_id is not None

    def test_wrap_with_agent_name_and_non_jwt_key_raises_error(self):
        """Test that non-JWT key without org_id raises error."""
        config = ClyroConfig(
            agent_name="test-agent",
            api_key="cly_test_not_a_jwt_key",  # Not a JWT key
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        with pytest.raises(ClyroWrapError) as exc_info:
            wrap(my_agent, config=config)

        assert "org_id is required" in str(exc_info.value)

    def test_wrap_with_agent_id_doesnt_need_org_id(self):
        """Test that providing agent_id directly doesn't require org_id."""
        agent_id = uuid4()

        config = ClyroConfig(
            api_key=None,  # No API key
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        # Should work fine with agent_id (no org_id needed)
        wrapped = wrap(my_agent, config=config, agent_id=agent_id)

        assert wrapped._agent_id == agent_id


class TestWrapWithExplicitOrgID:
    """Test wrapping with explicit org_id parameter."""

    def test_wrap_with_explicit_org_id_generates_agent_id(self):
        """Test that explicit org_id is used for agent_id generation."""
        org_id = uuid4()

        config = ClyroConfig(
            agent_name="test-agent",
        )

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped = wrap(my_agent, config=config, org_id=org_id)

        expected_agent_id = uuid5(org_id, "test-agent".lower())
        assert wrapped._agent_id == expected_agent_id
        assert wrapped._org_id == org_id


class TestAgentIDCollisionPrevention:
    """Test that org_id namespace prevents agent_id collisions."""

    def test_same_agent_name_different_orgs_different_ids(self):
        """Test that same agent name in different orgs produces different IDs."""
        org_id1 = uuid4()
        org_id2 = uuid4()
        agent_name = "my-agent"

        jwt_key1 = _create_mock_jwt_key(org_id1)
        jwt_key2 = _create_mock_jwt_key(org_id2)

        config1 = ClyroConfig(agent_name=agent_name, api_key=jwt_key1)
        config2 = ClyroConfig(agent_name=agent_name, api_key=jwt_key2)

        def my_agent(query: str) -> str:
            return f"Response: {query}"

        wrapped1 = wrap(my_agent, config=config1)
        wrapped2 = wrap(my_agent, config=config2)

        # Same agent_name but different org_id should produce different agent_id
        assert wrapped1._agent_id != wrapped2._agent_id

        # Verify they match expected UUIDs
        assert wrapped1._agent_id == uuid5(org_id1, agent_name.lower())
        assert wrapped2._agent_id == uuid5(org_id2, agent_name.lower())
