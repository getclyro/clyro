"""
Shared test fixtures and constants for Clyro SDK tests.
"""

from uuid import UUID

import pytest

# Test organization ID for use across all tests
# This is a fixed UUID to ensure consistent test behavior
TEST_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def test_org_id() -> UUID:
    """
    Fixture providing a test organization ID.

    Returns a consistent UUID for use in tests that require org_id
    for agent auto-registration.

    Returns:
        UUID: Test organization ID
    """
    return TEST_ORG_ID
