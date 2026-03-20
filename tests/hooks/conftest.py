"""Shared test fixtures for clyro-hook tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from clyro.constants import DEFAULT_API_URL
from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import HookConfig, load_hook_config
from clyro.hooks.models import HookInput, SessionState


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test state/audit files."""
    return tmp_path


@pytest.fixture
def sessions_dir(tmp_dir):
    """Provide a temporary sessions directory."""
    d = tmp_dir / "sessions"
    d.mkdir()
    return d


@pytest.fixture
def audit_path(tmp_dir):
    """Provide a temporary audit log path."""
    return tmp_dir / "audit.jsonl"


@pytest.fixture
def audit_logger(audit_path):
    """Provide an AuditLogger writing to a temp file."""
    al = AuditLogger(log_path=audit_path, redact_patterns=["*password*", "*secret*"])
    yield al
    al.close()


@pytest.fixture
def sample_hook_input() -> HookInput:
    """Provide a basic HookInput for testing."""
    return HookInput(
        session_id="test-session-001",
        tool_name="Bash",
        tool_input={"command": "ls -la", "description": "list files"},
    )


@pytest.fixture
def sample_config(tmp_dir) -> HookConfig:
    """Provide a basic HookConfig with permissive defaults."""
    config_data = {
        "global": {
            "max_steps": 50,
            "max_cost_usd": 10.0,
            "cost_per_token_usd": 0.00001,
            "loop_detection": {"threshold": 3, "window": 10},
            "policies": [],
        },
        "tools": {},
        "backend": {
            "api_key": None,
            "api_url": DEFAULT_API_URL,
            "agent_name": "test-agent",
        },
        "audit": {
            "log_path": str(tmp_dir / "audit.jsonl"),
            "redact_parameters": ["*password*", "*secret*", "*token*", "*api_key*"],
        },
    }
    return HookConfig.model_validate(config_data)


@pytest.fixture
def config_with_policies(tmp_dir) -> HookConfig:
    """Provide a config with policy rules."""
    config_data = {
        "global": {
            "max_steps": 50,
            "max_cost_usd": 10.0,
            "cost_per_token_usd": 0.00001,
            "loop_detection": {"threshold": 3, "window": 10},
            "policies": [
                {
                    "parameter": "command",
                    "operator": "contains",
                    "value": "rm -rf",
                    "name": "Block recursive force delete",
                },
            ],
        },
        "tools": {
            "Bash": {
                "policies": [
                    {
                        "parameter": "command",
                        "operator": "contains",
                        "value": "sudo",
                        "name": "Block sudo commands",
                    },
                ],
            },
        },
        "backend": {"api_key": None},
        "audit": {
            "log_path": str(tmp_dir / "audit.jsonl"),
            "redact_parameters": ["*password*", "*secret*"],
        },
    }
    return HookConfig.model_validate(config_data)


@pytest.fixture
def sample_state() -> SessionState:
    """Provide a basic session state."""
    return SessionState(session_id="test-session-001")


def write_config_file(tmp_dir: Path, config_data: dict) -> Path:
    """Write a YAML config file and return its path."""
    config_path = tmp_dir / "claude-code-policy.yaml"
    config_path.write_text(yaml.dump(config_data))
    return config_path
