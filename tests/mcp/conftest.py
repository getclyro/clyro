"""
Shared test fixtures for clyro-mcp tests.
"""

from __future__ import annotations

import tempfile
from uuid import UUID

import pytest

from clyro.config import (
    WrapperConfig,
)
from clyro.mcp.session import McpSession

TEST_SESSION_ID = UUID("00000000-0000-0000-0000-000000000099")


@pytest.fixture
def default_config() -> WrapperConfig:
    """WrapperConfig with sensible defaults."""
    return WrapperConfig()


@pytest.fixture
def strict_config(tmp_path) -> WrapperConfig:
    """Config with low limits for testing blocking behaviour."""
    return WrapperConfig.model_validate(
        {
            "global": {
                "max_steps": 2,
                "max_cost_usd": 0.001,
                "cost_per_token_usd": 0.001,
                "loop_detection": {"threshold": 3, "window": 10},
                "policies": [
                    {"parameter": "*.amount", "operator": "max_value", "value": 500},
                ],
            },
            "tools": {
                "query_database": {
                    "policies": [
                        {
                            "parameter": "sql",
                            "operator": "contains",
                            "value": "DROP",
                            "name": "no-drop",
                        },
                    ],
                },
            },
            "audit": {
                "log_path": str(tmp_path / "clyro-mcp-test-audit.jsonl"),
                "redact_parameters": ["*.password", "*.secret"],
            },
        }
    )


@pytest.fixture
def session() -> McpSession:
    """Fresh session."""
    return McpSession()


@pytest.fixture
def session_with_id() -> McpSession:
    """Session with a deterministic ID for assertions."""
    return McpSession(session_id=TEST_SESSION_ID)
