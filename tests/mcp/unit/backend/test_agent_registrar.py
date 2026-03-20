"""
Unit tests for AgentRegistrar — TDD §11.1 v1.1 tests.

FRD-016: Auto-register MCP agent and persist agent_id locally.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from clyro.backend.agent_registrar import AgentRegistrar
from clyro.backend.http_client import HttpSyncClient

TEST_AGENT_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def http_client() -> HttpSyncClient:
    return MagicMock(spec=HttpSyncClient)


@pytest.fixture
def registrar(tmp_path, http_client) -> AgentRegistrar:
    r = AgentRegistrar(instance_id="test12345678", http_client=http_client)
    r._id_path = tmp_path / "mcp-agent-test12345678.id"
    r._unconfirmed_path = r._id_path.with_suffix(r._id_path.suffix + ".unconfirmed")
    return r


class TestAgentRegistrarLoadPersisted:
    """Load persisted agent_id from file (FRD-016)."""

    @pytest.mark.asyncio
    async def test_loads_persisted_id(self, registrar: AgentRegistrar) -> None:
        registrar._id_path.write_text(TEST_AGENT_ID)
        agent_id = await registrar.get_or_register("test-agent")
        assert agent_id == UUID(TEST_AGENT_ID)

    @pytest.mark.asyncio
    async def test_ignores_corrupted_file(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        registrar._id_path.write_text("not-a-uuid")
        http_client.register_agent = AsyncMock(return_value=TEST_AGENT_ID)

        agent_id = await registrar.get_or_register("test-agent")
        assert agent_id == UUID(TEST_AGENT_ID)
        http_client.register_agent.assert_called_once()


class TestAgentRegistrarRegister:
    """Register new agent via HTTP (FRD-016)."""

    @pytest.mark.asyncio
    async def test_registers_and_persists(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        http_client.register_agent = AsyncMock(return_value=TEST_AGENT_ID)

        agent_id = await registrar.get_or_register("test-agent")
        assert agent_id == UUID(TEST_AGENT_ID)
        assert registrar._id_path.exists()
        assert registrar._id_path.read_text().strip() == TEST_AGENT_ID
        # Confirmed registration — no unconfirmed marker
        assert not registrar._unconfirmed_path.exists()

    @pytest.mark.asyncio
    async def test_passes_agent_name(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        http_client.register_agent = AsyncMock(return_value=TEST_AGENT_ID)
        await registrar.get_or_register("my-custom-agent")
        http_client.register_agent.assert_called_once_with("my-custom-agent")


class TestAgentRegistrarFallback:
    """Fallback to local UUID on registration failure (FRD-016)."""

    @pytest.mark.asyncio
    async def test_falls_back_to_local_id(
        self, registrar: AgentRegistrar, http_client, capsys
    ) -> None:
        http_client.register_agent = AsyncMock(
            side_effect=Exception("network error")
        )
        agent_id = await registrar.get_or_register("test-agent")

        # Should get a valid UUID (local)
        assert isinstance(agent_id, UUID)
        captured = capsys.readouterr()
        assert "agent_registration_failed" in captured.err

    @pytest.mark.asyncio
    async def test_fallback_persists_local_id(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        """Local fallback UUID must be persisted so subsequent sessions reuse it."""
        http_client.register_agent = AsyncMock(
            side_effect=Exception("network error")
        )
        agent_id = await registrar.get_or_register("test-agent")

        assert registrar._id_path.exists()
        assert registrar._id_path.read_text().strip() == str(agent_id)
        # Should be marked as unconfirmed
        assert registrar._unconfirmed_path.exists()

    @pytest.mark.asyncio
    async def test_fallback_reuses_persisted_id_on_next_call(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        """When backend is still down, the same local ID should be reused."""
        http_client.register_agent = AsyncMock(
            side_effect=Exception("network error")
        )
        first_id = await registrar.get_or_register("test-agent")
        second_id = await registrar.get_or_register("test-agent")

        assert first_id == second_id


class TestAgentRegistrarRecovery:
    """Recovery: register with backend after local fallback (FRD-016)."""

    @pytest.mark.asyncio
    async def test_unconfirmed_id_replaced_when_backend_comes_online(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        """When an unconfirmed ID exists and backend becomes available,
        the backend-assigned ID should replace the local one."""
        # Simulate previous offline session: local ID + unconfirmed marker
        local_id = "11111111-1111-1111-1111-111111111111"
        registrar._id_path.write_text(local_id)
        registrar._unconfirmed_path.write_text("")

        # Backend is now online
        http_client.register_agent = AsyncMock(return_value=TEST_AGENT_ID)

        agent_id = await registrar.get_or_register("test-agent")

        assert agent_id == UUID(TEST_AGENT_ID)
        assert registrar._id_path.read_text().strip() == TEST_AGENT_ID
        assert not registrar._unconfirmed_path.exists()

    @pytest.mark.asyncio
    async def test_unconfirmed_id_kept_when_backend_still_down(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        """When an unconfirmed ID exists and backend is still down,
        the local ID should be kept as-is."""
        local_id = "11111111-1111-1111-1111-111111111111"
        registrar._id_path.write_text(local_id)
        registrar._unconfirmed_path.write_text("")

        # Backend still offline
        http_client.register_agent = AsyncMock(
            side_effect=Exception("still offline")
        )

        agent_id = await registrar.get_or_register("test-agent")

        assert agent_id == UUID(local_id)
        # Marker should still exist
        assert registrar._unconfirmed_path.exists()

    @pytest.mark.asyncio
    async def test_confirmed_id_not_re_registered(
        self, registrar: AgentRegistrar, http_client
    ) -> None:
        """A confirmed (backend-registered) ID should be used directly
        without calling register_agent again."""
        registrar._id_path.write_text(TEST_AGENT_ID)
        # No .unconfirmed marker = confirmed

        agent_id = await registrar.get_or_register("test-agent")

        assert agent_id == UUID(TEST_AGENT_ID)
        # register_agent should NOT have been called
        assert not hasattr(http_client, "register_agent") or \
            not getattr(http_client.register_agent, "called", False)


class TestAgentRegistrarPersistence:
    """Persistence edge cases (FRD-016)."""

    @pytest.mark.asyncio
    async def test_id_path_property(self, registrar: AgentRegistrar) -> None:
        assert "mcp-agent-test12345678.id" in str(registrar.id_path)
