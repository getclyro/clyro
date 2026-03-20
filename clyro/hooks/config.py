# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Configuration
# Implements FRD-HK-011

"""YAML configuration loading with defaults and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from clyro.config import (
    DEFAULT_API_URL,
    WrapperConfig,
)

from .constants import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_COST_PER_TOKEN_USD,
    DEFAULT_LOOP_THRESHOLD,
    DEFAULT_LOOP_WINDOW,
    DEFAULT_MAX_COST_USD,
    DEFAULT_MAX_STEPS,
    DEFAULT_POLICY_CACHE_TTL_SECONDS,
    DEFAULT_REDACT_PARAMETERS,
)

logger = structlog.get_logger()


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class HookConfig(WrapperConfig):
    """Extends WrapperConfig with hook-specific defaults.

    Reuses the same schema so PolicyEvaluator, CostTracker, etc.
    work without adaptation.
    """

    model_config = {"extra": "ignore", "populate_by_name": True}

    policy_cache_ttl_seconds: int = DEFAULT_POLICY_CACHE_TTL_SECONDS


def _hook_defaults() -> dict[str, Any]:
    """Return permissive defaults for when no config file exists."""
    return {
        "global": {
            "max_steps": DEFAULT_MAX_STEPS,
            "max_cost_usd": DEFAULT_MAX_COST_USD,
            "cost_per_token_usd": DEFAULT_COST_PER_TOKEN_USD,
            "loop_detection": {
                "threshold": DEFAULT_LOOP_THRESHOLD,
                "window": DEFAULT_LOOP_WINDOW,
            },
            "policies": [],
        },
        "tools": {},
        "backend": {
            "api_key": None,
            "api_url": DEFAULT_API_URL,
            "agent_name": "claude-code-session",
        },
        "audit": {
            "log_path": str(DEFAULT_AUDIT_PATH),
            "redact_parameters": DEFAULT_REDACT_PARAMETERS,
        },
    }


def load_hook_config(config_path: str | None = None) -> HookConfig:
    """Load and validate hook configuration from YAML.

    FRD-HK-011:
    - Missing file: use permissive defaults, warn to stderr
    - Invalid YAML: exit 1
    - Schema validation failure: exit 1
    - CLYRO_API_KEY env var overrides backend.api_key
    - CLYRO_API_URL env var overrides backend.api_url
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    path = path.expanduser()

    if not path.exists():
        logger.warning("config_file_missing", path=str(path), using="permissive defaults")
        data = _hook_defaults()
    else:
        try:
            raw = path.read_text()
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in config file {path}: {e}") from e

    # Merge with defaults for missing keys
    defaults = _hook_defaults()
    for key in defaults:
        if key not in data:
            data[key] = defaults[key]

    # Env var overrides
    if env_key := os.environ.get("CLYRO_API_KEY"):
        data.setdefault("backend", {})["api_key"] = env_key
    if env_url := os.environ.get("CLYRO_API_URL"):
        data.setdefault("backend", {})["api_url"] = env_url

    try:
        config = HookConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"Config validation error: {e}") from e

    return config
