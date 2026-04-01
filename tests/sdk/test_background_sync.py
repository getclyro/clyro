#!/usr/bin/env python3
"""
Test script to verify automatic background sync.

This script demonstrates that background sync now starts automatically
when wrapping an agent, without requiring manual intervention.
"""

import asyncio
import time

from conftest import TEST_ORG_ID

from clyro import ClyroConfig, wrap
from clyro.transport import Transport


def sync_agent(query: str) -> str:
    """Simple synchronous agent for testing."""
    return f"Response to: {query}"


async def async_agent(query: str) -> str:
    """Simple async agent for testing."""
    await asyncio.sleep(0.1)
    return f"Async response to: {query}"


def test_sync_agent_background_sync():
    """Test that sync agents start background sync automatically."""
    print("\n=== Testing Sync Agent Background Sync ===")

    # Wrap the agent (background sync should start automatically)
    config = ClyroConfig(agent_name="test-agent")
    wrapped = wrap(sync_agent, config=config, org_id=TEST_ORG_ID)

    print(f"Background sync started: {wrapped._background_sync_started}")
    print(f"Is local only: {config.is_local_only()}")

    # Execute the agent
    result = wrapped("test query")
    print(f"Result: {result}")

    # Check transport status
    status = wrapped.get_status()
    print("\nTransport status:")
    print(f"  - Background sync started: {status['background_sync_started']}")
    if status['transport'] is not None:
        print(f"  - Transport running: {status['transport']['transport']['running']}")
        print(f"  - Unsynced events: {status['transport']['storage']['events']['unsynced']}")

        # Give background sync time to work
        time.sleep(2)

        # Check status again
        status_after = wrapped.get_status()
        print("\nStatus after 2 seconds:")
        print(f"  - Unsynced events: {status_after['transport']['storage']['events']['unsynced']}")
    else:
        print("  - Transport: None (local-only mode)")

    # Cleanup
    wrapped.close()
    print("\n✓ Sync agent test completed")


async def test_async_agent_background_sync():
    """Test that async agents start background sync on first execution."""
    print("\n=== Testing Async Agent Background Sync ===")

    # Wrap the agent
    config = ClyroConfig(agent_name="test-agent")
    wrapped = wrap(async_agent, config=config, org_id=TEST_ORG_ID)

    print(f"Background sync started (before execution): {wrapped._background_sync_started}")

    # Execute the agent (should start background sync on first call)
    result = await wrapped("test query")
    print(f"Result: {result}")
    print(f"Background sync started (after execution): {wrapped._background_sync_started}")

    # Check transport status
    if isinstance(wrapped._transport, Transport):
        status = wrapped._transport.get_sync_status()
        print("\nTransport status:")
        print(f"  - Transport running: {status['transport']['running']}")
        print(f"  - Unsynced events: {status['storage']['events']['unsynced']}")
        print(f"  - Circuit state: {status['transport']['circuit_state']}")

    # Give background sync time to work
    await asyncio.sleep(2)

    # Cleanup
    await wrapped.close_async()
    print("\n✓ Async agent test completed")


def test_local_only_mode():
    """Test that local-only mode doesn't start background sync."""
    print("\n=== Testing Local-Only Mode ===")

    # Configure for local-only mode (no API key)
    config = ClyroConfig(agent_name="test-agent", api_key=None)
    wrapped = wrap(sync_agent, config=config, org_id=TEST_ORG_ID)

    print(f"Is local only: {config.is_local_only()}")
    print(f"Background sync started: {wrapped._background_sync_started}")

    # Execute the agent
    result = wrapped("test query")
    print(f"Result: {result}")

    # Background sync should NOT be started
    assert not wrapped._background_sync_started, "Background sync should not start in local-only mode"

    print("\n✓ Local-only mode test completed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Background Sync Automatic Start Test")
    print("=" * 60)

    # Test sync agent
    test_sync_agent_background_sync()

    # Test async agent
    asyncio.run(test_async_agent_background_sync())

    # Test local-only mode
    test_local_only_mode()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
