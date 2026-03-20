# Tests for refactored wrapper execution methods
# Verifies that the extracted helper methods work correctly

"""
Tests for the refactored execution flow in WrappedAgent.

This module ensures that the refactored helper methods
(_create_session, _run_sync_with_tracing, _run_async_with_tracing,
_cleanup_session_sync, _cleanup_session_async) work correctly
and maintain the same behavior as the original implementation.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from clyro import wrap
from clyro.config import ClyroConfig, ExecutionControls, reset_config
from conftest import TEST_ORG_ID
from clyro.exceptions import StepLimitExceededError
from clyro.session import get_current_session
from clyro.storage.sqlite import LocalStorage


class TestRefactoredExecutionFlow:
    """Tests for the refactored execution methods."""

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_config()

    def test_session_created_and_cleaned_up_sync(self):
        """Test that session is properly created and cleaned up in sync execution."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)
        session_ids_seen = []

        @wrap(config=config, org_id=TEST_ORG_ID)
        def test_agent(value: int) -> int:
            # Session should be active during execution
            session = get_current_session()
            assert session is not None, "Session should be active during execution"
            session_ids_seen.append(str(session.session_id))
            return value * 2

        # Execute agent
        result = test_agent(5)
        assert result == 10

        # Session should be cleaned up after execution
        assert get_current_session() is None, "Session should be cleaned up after execution"

        # Verify a session was created
        assert len(session_ids_seen) == 1

    @pytest.mark.asyncio
    async def test_session_created_and_cleaned_up_async(self):
        """Test that session is properly created and cleaned up in async execution."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)
        session_ids_seen = []

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def async_test_agent(value: int) -> int:
            # Session should be active during execution
            session = get_current_session()
            assert session is not None, "Session should be active during execution"
            session_ids_seen.append(str(session.session_id))
            await asyncio.sleep(0.01)  # Simulate async work
            return value * 2

        # Execute agent
        result = await async_test_agent(5)
        assert result == 10

        # Session should be cleaned up after execution
        assert get_current_session() is None, "Session should be cleaned up after execution"

        # Verify a session was created
        assert len(session_ids_seen) == 1

    def test_session_cleaned_up_on_error_sync(self):
        """Test that session is cleaned up even when sync agent raises error."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        def failing_agent(value: int) -> int:
            session = get_current_session()
            assert session is not None
            raise ValueError("Intentional error")

        # Execute agent and expect error
        with pytest.raises(ValueError, match="Intentional error"):
            failing_agent(5)

        # Session should still be cleaned up
        assert get_current_session() is None, "Session should be cleaned up even on error"

    @pytest.mark.asyncio
    async def test_session_cleaned_up_on_error_async(self):
        """Test that session is cleaned up even when async agent raises error."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def failing_agent(value: int) -> int:
            session = get_current_session()
            assert session is not None
            await asyncio.sleep(0.01)
            raise ValueError("Intentional error")

        # Execute agent and expect error
        with pytest.raises(ValueError, match="Intentional error"):
            await failing_agent(5)

        # Session should still be cleaned up
        assert get_current_session() is None, "Session should be cleaned up even on error"

    def test_execution_controls_enforced_sync(self):
        """Test that execution controls are enforced in refactored sync execution."""
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=False,
            controls=ExecutionControls(
                enable_step_limit=True,
                max_steps=2,
            ),
        )

        call_count = [0]

        @wrap(config=config, org_id=TEST_ORG_ID)
        def test_agent(value: int) -> int:
            call_count[0] += 1
            session = get_current_session()
            assert session is not None

            # Record multiple steps to exceed limit
            for i in range(5):
                session.record_step(
                    event_name=f"step_{i}",
                    input_data={"value": value},
                )
            return value

        # Should raise StepLimitExceededError
        with pytest.raises(StepLimitExceededError):
            test_agent(5)

        # Verify agent was called
        assert call_count[0] == 1

        # Session should be cleaned up even on control error
        assert get_current_session() is None

    @pytest.mark.asyncio
    async def test_execution_controls_enforced_async(self):
        """Test that execution controls are enforced in refactored async execution."""
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=False,
            controls=ExecutionControls(
                enable_step_limit=True,
                max_steps=2,
            ),
        )

        call_count = [0]

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def async_test_agent(value: int) -> int:
            call_count[0] += 1
            session = get_current_session()
            assert session is not None

            # Record multiple steps to exceed limit
            for i in range(5):
                session.record_step(
                    event_name=f"step_{i}",
                    input_data={"value": value},
                )
            await asyncio.sleep(0.01)
            return value

        # Should raise StepLimitExceededError
        with pytest.raises(StepLimitExceededError):
            await async_test_agent(5)

        # Verify agent was called
        assert call_count[0] == 1

        # Session should be cleaned up even on control error
        assert get_current_session() is None
