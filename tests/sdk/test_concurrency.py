# Concurrency tests for thread-safe session management
# Tests for issue identified in code review: global session state

"""
Tests for concurrent agent execution to ensure thread-safety.

This module verifies that the SDK can handle multiple concurrent agent
executions without race conditions or session state corruption.
"""

import asyncio
import concurrent.futures
import time
from uuid import uuid4

import pytest

from clyro import wrap
from clyro.config import ClyroConfig, ExecutionControls
from clyro.session import get_current_session, set_current_session
from conftest import TEST_ORG_ID


class TestConcurrentExecution:
    """Test concurrent agent execution with contextvars-based session isolation."""

    def test_concurrent_sync_agents_isolated_sessions(self):
        """Test that concurrent synchronous agent calls maintain isolated sessions."""
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=True,
            controls=ExecutionControls(max_steps=5),
        )

        @wrap(config=config, org_id=TEST_ORG_ID)
        def test_agent(agent_id: int, delay: float = 0.1) -> dict:
            """Test agent that returns its agent ID and session ID."""
            session = get_current_session()
            assert session is not None, f"Agent {agent_id}: No active session"

            session_id = str(session.session_id)
            time.sleep(delay)  # Simulate work

            # Verify session is still the same after delay
            current_session = get_current_session()
            assert current_session is not None
            assert str(current_session.session_id) == session_id

            return {
                "agent_id": agent_id,
                "session_id": session_id,
            }

        # Execute multiple agents concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(test_agent, agent_id=i, delay=0.05)
                for i in range(10)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all agents completed
        assert len(results) == 10

        # Verify each agent got a unique session
        session_ids = [r["session_id"] for r in results]
        assert len(session_ids) == len(set(session_ids)), "Sessions were not isolated"

        # Verify agent IDs match
        agent_ids = sorted([r["agent_id"] for r in results])
        assert agent_ids == list(range(10))

    def test_concurrent_sync_agents_no_session_leak(self):
        """Test that sessions don't leak between concurrent executions."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        def agent_with_error(should_fail: bool) -> str:
            session = get_current_session()
            assert session is not None

            if should_fail:
                raise ValueError("Intentional error")

            return "success"

        def run_agent(should_fail: bool) -> tuple[bool, str | None]:
            """Run agent and return (success, session_id)."""
            try:
                agent_with_error(should_fail=should_fail)
                session = get_current_session()
                return True, str(session.session_id) if session else None
            except ValueError:
                session = get_current_session()
                return False, str(session.session_id) if session else None

        # Run mix of successful and failing agents
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(run_agent, should_fail=(i % 2 == 0))
                for i in range(8)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify we got expected results
        successful = [r for r in results if r[0]]
        failed = [r for r in results if not r[0]]

        assert len(successful) == 4
        assert len(failed) == 4

        # Verify no session is active after all executions
        assert get_current_session() is None

    @pytest.mark.asyncio
    async def test_concurrent_async_agents_isolated_sessions(self):
        """Test that concurrent async agent calls maintain isolated sessions."""
        config = ClyroConfig(
            agent_name="test-agent",
            fail_open=True,
            controls=ExecutionControls(max_steps=5),
        )

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def async_test_agent(agent_id: int, delay: float = 0.1) -> dict:
            """Async test agent that returns its agent ID and session ID."""
            session = get_current_session()
            assert session is not None, f"Agent {agent_id}: No active session"

            session_id = str(session.session_id)
            await asyncio.sleep(delay)  # Simulate async work

            # Verify session is still the same after delay
            current_session = get_current_session()
            assert current_session is not None
            assert str(current_session.session_id) == session_id

            return {
                "agent_id": agent_id,
                "session_id": session_id,
            }

        # Execute multiple async agents concurrently
        tasks = [async_test_agent(agent_id=i, delay=0.05) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all agents completed
        assert len(results) == 10

        # Verify each agent got a unique session
        session_ids = [r["session_id"] for r in results]
        assert len(session_ids) == len(set(session_ids)), "Sessions were not isolated"

        # Verify agent IDs match
        agent_ids = sorted([r["agent_id"] for r in results])
        assert agent_ids == list(range(10))

    @pytest.mark.asyncio
    async def test_concurrent_async_agents_with_errors(self):
        """Test error handling in concurrent async agents."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def async_agent_with_error(agent_id: int) -> str:
            await asyncio.sleep(0.01)
            if agent_id % 3 == 0:
                raise ValueError(f"Error in agent {agent_id}")
            return f"success-{agent_id}"

        # Run agents, some will fail
        results = []
        for i in range(9):
            try:
                result = await async_agent_with_error(agent_id=i)
                results.append(("success", result))
            except ValueError as e:
                results.append(("error", str(e)))

        # Verify expected number of successes and failures
        successes = [r for r in results if r[0] == "success"]
        errors = [r for r in results if r[0] == "error"]

        assert len(successes) == 6  # agents 1,2,4,5,7,8
        assert len(errors) == 3  # agents 0,3,6

        # Verify no session is active
        assert get_current_session() is None

    def test_session_context_isolation(self):
        """Test that session context is properly isolated between threads."""
        results = {}

        def thread_worker(thread_id: int):
            """Worker that sets a session and verifies it stays isolated."""
            from clyro.session import Session

            # Create and set a session
            session = Session(
                config=ClyroConfig(agent_name="test-agent"),
                agent_id=uuid4(),
            )
            set_current_session(session)

            # Simulate some work
            time.sleep(0.05)

            # Verify session is still the same
            current = get_current_session()
            assert current is not None
            assert current.session_id == session.session_id

            results[thread_id] = str(session.session_id)

            # Clean up
            set_current_session(None)

        # Run multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(10)]
            for f in concurrent.futures.as_completed(futures):
                f.result()  # Raise any exceptions

        # Verify all threads completed with unique sessions
        assert len(results) == 10
        session_ids = list(results.values())
        assert len(session_ids) == len(set(session_ids)), "Session IDs were not unique"

    def test_rapid_sequential_executions(self):
        """Test rapid sequential executions don't leak state."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        def quick_agent(value: int) -> int:
            return value * 2

        session_ids = []

        # Run many quick sequential executions
        for i in range(50):
            result = quick_agent(value=i)
            assert result == i * 2

            # Session should be cleared after each execution
            session = get_current_session()
            assert session is None, f"Session leaked after execution {i}"

    @pytest.mark.asyncio
    async def test_mixed_sync_async_context_isolation(self):
        """Test that sync and async agents maintain separate contexts."""
        config = ClyroConfig(agent_name="test-agent", fail_open=True)

        @wrap(config=config, org_id=TEST_ORG_ID)
        def sync_agent(value: int) -> dict:
            session = get_current_session()
            return {"value": value, "session_id": str(session.session_id)}

        @wrap(config=config, org_id=TEST_ORG_ID)
        async def async_agent(value: int) -> dict:
            await asyncio.sleep(0.01)
            session = get_current_session()
            return {"value": value, "session_id": str(session.session_id)}

        # Run async agents
        async_results = await asyncio.gather(*[async_agent(i) for i in range(5)])

        # Run sync agents in executor to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            sync_futures = [
                loop.run_in_executor(executor, sync_agent, i) for i in range(5)
            ]
            sync_results = await asyncio.gather(*sync_futures)

        # Verify all completed with unique sessions
        all_session_ids = [r["session_id"] for r in async_results + sync_results]
        assert len(all_session_ids) == 10
        assert len(set(all_session_ids)) == 10, "Sessions were not isolated"
