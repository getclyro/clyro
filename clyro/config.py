# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Configuration
# Implements PRD-002

"""
Configuration models for the Clyro SDK.

This module provides Pydantic models for SDK configuration with
validation, defaults, and type safety.
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from clyro.constants import DEFAULT_API_URL
from clyro.exceptions import ClyroConfigError

logger = structlog.get_logger(__name__)


# Default pricing table (per 1K tokens)
# Users can extend this by passing custom pricing to ClyroConfig
# or by calling config.register_model_pricing() after initialization
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI models
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic models
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
}


# Implements PRD-009, PRD-010
class ExecutionControls(BaseModel):
    """
    Execution safety controls configuration.

    These controls prevent runaway agent execution by enforcing
    limits on steps, cost, and detecting infinite loops.
    """

    max_steps: int = Field(
        default=100,
        ge=1,
        le=100000,
        description="Maximum execution steps before termination",
    )
    max_cost_usd: float = Field(
        default=10.0,
        ge=0.0,
        le=10000.0,
        description="Maximum cumulative cost in USD",
    )
    loop_detection_threshold: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Number of identical state iterations before loop detection triggers",
    )
    enable_step_limit: bool = Field(
        default=True,
        description="Enable step limit enforcement",
    )
    enable_cost_limit: bool = Field(
        default=True,
        description="Enable cost limit enforcement",
    )
    enable_loop_detection: bool = Field(
        default=True,
        description="Enable infinite loop detection",
    )
    enable_policy_enforcement: bool = Field(
        default=False,
        description="Enable policy enforcement via backend evaluation",
    )

    model_config = {"extra": "forbid"}


class ClyroConfig(BaseModel):
    """
    SDK configuration model.

    Provides configuration for SDK behavior including connection settings,
    execution controls, local storage, and operational behavior.

    Example:
        ```python
        config = ClyroConfig(
            api_key="cly_live_...",
            agent_name="my-agent",
            controls=ExecutionControls(max_steps=50),
        )
        ```
    """

    # Implements FRD-SOF-004: SDK operating mode
    mode: Literal["local", "cloud"] | None = Field(
        default=None,
        description="Operating mode: 'local' (YAML policies, no network) or 'cloud' "
        "(backend API). Auto-resolved from api_key if not set.",
    )

    # Connection settings
    api_key: str | None = Field(
        default=None,
        description="API key for cloud backend authentication",
    )
    endpoint: str = Field(
        default=DEFAULT_API_URL,
        description="Backend API endpoint URL. Override via CLYRO_ENDPOINT env var.",
    )

    # Agent identification
    agent_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Human-readable agent name (used to auto-generate agent_id if not provided)",
    )
    agent_id: str | None = Field(
        default=None,
        description="UUID of agent (auto-generated from agent_name if not provided)",
    )

    # Execution controls
    controls: ExecutionControls = Field(
        default_factory=ExecutionControls,
        description="Execution safety controls",
    )

    # Local storage settings
    local_storage_path: str | None = Field(
        default=None,
        description="Path for local SQLite storage (default: ~/.clyro)",
    )
    local_storage_max_mb: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum local storage size in MB",
    )

    # Sync and transport settings
    sync_interval_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=300.0,
        description="Interval between background sync attempts",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum events per batch upload",
    )
    retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed requests",
    )

    # Behavior settings
    fail_open: bool = Field(
        default=True,
        description="Continue agent execution if trace capture fails",
    )
    capture_inputs: bool = Field(
        default=True,
        description="Capture input data in traces",
    )
    capture_outputs: bool = Field(
        default=True,
        description="Capture output data in traces",
    )
    capture_state: bool = Field(
        default=True,
        description="Capture state snapshots in traces",
    )

    # OTLP Export Configuration (C8 — default-off, FRD-S006, FRD-S007)
    otlp_export_endpoint: str | None = Field(
        default=None,
        description="OTLP/HTTP endpoint for secondary export. HTTPS required "
        "(HTTP allowed for localhost only). Export disabled when absent.",
    )
    otlp_export_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers for OTLP export requests.",
    )
    otlp_export_timeout_ms: int = Field(
        default=5000,
        gt=0,
        description="Per-request timeout for OTLP export in milliseconds.",
    )
    otlp_export_queue_size: int = Field(
        default=100,
        gt=0,
        description="Async queue capacity for OTLP export batches.",
    )
    otlp_export_compression: str = Field(
        default="gzip",
        description="Compression for OTLP export: 'gzip' or 'none'.",
    )

    # Pricing configuration
    pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: DEFAULT_PRICING.copy(),
        description="Token pricing per model (input/output per 1K tokens). "
        "Users can override or extend this dict with custom models.",
    )

    model_config = {"extra": "forbid"}

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise ClyroConfigError(
                message="Invalid configuration",
                details={"errors": exc.errors()},
            ) from exc

    # Implements FRD-SOF-004: mode validation
    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: Literal["local", "cloud"] | None) -> Literal["local", "cloud"] | None:
        """Validate mode value if explicitly provided."""
        if v is not None and v not in ("local", "cloud"):
            raise ValueError(f"Invalid mode: '{v}'. Must be 'local' or 'cloud'")
        return v

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint must start with http:// or https://")
        # Remove trailing slash for consistency
        return v.rstrip("/")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key format if provided."""
        if v is None:
            return v
        if not v.strip():
            raise ValueError("API key cannot be empty or whitespace")
        # API keys should start with known prefixes
        # Custom keys are allowed but we log a debug message for awareness
        valid_prefixes = ("cly_live_", "cly_test_", "cly_dev_")
        if not v.startswith(valid_prefixes):
            # Note: We use structlog at module level, but validators run before
            # full initialization. Log is deferred to runtime when key is used.
            # This is intentionally a no-op during validation; the key format
            # warning will appear in ClyroConfig.__init__ if needed in future.
            pass
        return v.strip()

    @field_validator("local_storage_path")
    @classmethod
    def validate_storage_path(cls, v: str | None) -> str | None:
        """Expand and validate storage path."""
        if v is None:
            return v
        # Expand user home directory
        expanded = os.path.expanduser(v)
        return str(Path(expanded).resolve())

    @field_validator("otlp_export_endpoint")
    @classmethod
    def validate_otlp_endpoint(cls, v: str | None) -> str | None:
        """Validate OTLP export endpoint URL. Implements FRD-S007."""
        if v is None or not v.strip():
            return None
        v = v.strip().rstrip("/")
        from urllib.parse import urlparse

        parsed = urlparse(v)
        if parsed.scheme == "https":
            return v
        if parsed.scheme == "http":
            host = parsed.hostname or ""
            if host in ("localhost", "127.0.0.1", "::1", "[::1]"):
                return v
            raise ValueError(
                f"OTLP export endpoint must use HTTPS for non-localhost hosts. "
                f"Got: {v}"
            )
        raise ValueError(
            f"OTLP export endpoint must use HTTP or HTTPS scheme. Got: {v}"
        )

    @field_validator("otlp_export_compression")
    @classmethod
    def validate_otlp_compression(cls, v: str) -> str:
        """Validate OTLP export compression. Implements FRD-S007."""
        if v not in ("gzip", "none"):
            raise ValueError(
                f"OTLP export compression must be 'gzip' or 'none'. Got: {v}"
            )
        return v

    @field_validator("otlp_export_headers")
    @classmethod
    def validate_otlp_headers(cls, v: dict[str, str]) -> dict[str, str]:
        """Filter reserved headers with warning. Implements FRD-S007."""
        reserved = {"content-type", "content-encoding"}
        filtered = {}
        for key, value in v.items():
            if key.lower() in reserved:
                logger.warning(
                    "otlp_export_reserved_header_ignored",
                    header=key,
                )
                continue
            filtered[key] = value
        return filtered

    @model_validator(mode="after")
    def set_defaults(self) -> ClyroConfig:
        """Set computed defaults after validation.  Implements FRD-SOF-004."""
        # Implements FRD-SOF-004: auto-resolve mode from api_key presence
        # CLYRO_MODE env var is respected even when ClyroConfig is constructed explicitly
        if self.mode is None:
            env_mode = os.getenv("CLYRO_MODE", "").lower().strip()
            if env_mode in ("local", "cloud"):
                object.__setattr__(self, "mode", env_mode)
            elif self.api_key is not None:
                object.__setattr__(self, "mode", "cloud")
            else:
                object.__setattr__(self, "mode", "local")

        # Implements FRD-SOF-004: cloud mode requires an API key
        if self.mode == "cloud" and self.api_key is None:
            raise ValueError(
                "Cloud mode requires an API key. Set CLYRO_API_KEY or use mode='local'"
            )

        # Set default storage path if not provided
        if self.local_storage_path is None:
            default_path = Path.home() / ".clyro" / "traces.db"
            object.__setattr__(self, "local_storage_path", str(default_path))
        return self

    def get_storage_path(self) -> Path:
        """Get the resolved storage path."""
        if self.local_storage_path:
            return Path(self.local_storage_path)
        return Path.home() / ".clyro" / "traces.db"

    def get_model_pricing(self, model: str) -> tuple[Decimal, Decimal]:
        """
        Get pricing for a model.

        Args:
            model: Model name/identifier

        Returns:
            Tuple of (input_price_per_1k, output_price_per_1k)
        """
        # Try exact match first
        if model in self.pricing:
            p = self.pricing[model]
            return Decimal(str(p["input"])), Decimal(str(p["output"]))

        # Try partial match (e.g., "gpt-4-turbo-preview" matches "gpt-4-turbo")
        for known_model, pricing in self.pricing.items():
            if model.startswith(known_model) or known_model in model:
                return Decimal(str(pricing["input"])), Decimal(str(pricing["output"]))

        # Default fallback pricing (conservative estimate)
        return Decimal("0.01"), Decimal("0.03")

    def register_model_pricing(
        self,
        model: str,
        input_price_per_1k: float,
        output_price_per_1k: float,
    ) -> None:
        """
        Register custom pricing for a model.

        This allows users to add pricing for custom or new models not in the default table.

        Args:
            model: Model name/identifier
            input_price_per_1k: Price per 1K input tokens in USD
            output_price_per_1k: Price per 1K output tokens in USD

        Example:
            ```python
            config = ClyroConfig()
            config.register_model_pricing("custom-model", 0.02, 0.06)
            ```
        """
        self.pricing[model] = {
            "input": input_price_per_1k,
            "output": output_price_per_1k,
        }
        logger.debug(
            "model_pricing_registered",
            model=model,
            input=input_price_per_1k,
            output=output_price_per_1k,
        )

    def is_local_only(self) -> bool:
        """Check if SDK is operating in local mode.  Implements FRD-SOF-004 backward compat."""
        return self.mode == "local"

    @classmethod
    def from_env(cls) -> ClyroConfig:
        """
        Create configuration from environment variables.

        Environment variables:
            CLYRO_API_KEY: API key for authentication
            CLYRO_ENDPOINT: Backend endpoint URL
            CLYRO_AGENT_NAME: Agent identifier (for auto-registration flow)
            CLYRO_AGENT_ID: Agent UUID (for manual registration flow)
            CLYRO_MAX_STEPS: Maximum execution steps
            CLYRO_MAX_COST_USD: Maximum cost in USD
            CLYRO_STORAGE_PATH: Local storage path
            CLYRO_FAIL_OPEN: Fail-open behavior (true/false)

        Returns:
            ClyroConfig instance
        """
        config_dict: dict[str, Any] = {}

        # Implements FRD-SOF-004: CLYRO_MODE env var
        if mode := os.getenv("CLYRO_MODE"):
            config_dict["mode"] = mode

        if api_key := os.getenv("CLYRO_API_KEY"):
            config_dict["api_key"] = api_key

        if endpoint := os.getenv("CLYRO_ENDPOINT"):
            config_dict["endpoint"] = endpoint

        if agent_name := os.getenv("CLYRO_AGENT_NAME"):
            config_dict["agent_name"] = agent_name

        if agent_id := os.getenv("CLYRO_AGENT_ID"):
            config_dict["agent_id"] = agent_id

        if storage_path := os.getenv("CLYRO_STORAGE_PATH"):
            config_dict["local_storage_path"] = storage_path

        # Parse execution controls
        controls_dict: dict[str, Any] = {}
        if max_steps := os.getenv("CLYRO_MAX_STEPS"):
            controls_dict["max_steps"] = int(max_steps)
        if max_cost := os.getenv("CLYRO_MAX_COST_USD"):
            controls_dict["max_cost_usd"] = float(max_cost)
        if enable_policies := os.getenv("CLYRO_ENABLE_POLICIES"):
            controls_dict["enable_policy_enforcement"] = enable_policies.lower() in (
                "true",
                "1",
                "yes",
            )
        if controls_dict:
            config_dict["controls"] = ExecutionControls(**controls_dict)

        # Parse boolean settings
        if fail_open := os.getenv("CLYRO_FAIL_OPEN"):
            config_dict["fail_open"] = fail_open.lower() in ("true", "1", "yes")

        # OTLP Export settings (FRD-S006, FRD-S007)
        if otlp_endpoint := os.getenv("CLYRO_OTLP_EXPORT_ENDPOINT"):
            config_dict["otlp_export_endpoint"] = otlp_endpoint
        if otlp_timeout := os.getenv("CLYRO_OTLP_EXPORT_TIMEOUT_MS"):
            config_dict["otlp_export_timeout_ms"] = int(otlp_timeout)
        if otlp_queue := os.getenv("CLYRO_OTLP_EXPORT_QUEUE_SIZE"):
            config_dict["otlp_export_queue_size"] = int(otlp_queue)
        if otlp_compression := os.getenv("CLYRO_OTLP_EXPORT_COMPRESSION"):
            config_dict["otlp_export_compression"] = otlp_compression

        return cls(**config_dict)


# Global configuration instance
_global_config: ClyroConfig | None = None


def get_config() -> ClyroConfig:
    """
    Get the current global configuration.

    Returns:
        Current ClyroConfig instance, or default if not configured.
    """
    global _global_config
    if _global_config is None:
        _global_config = ClyroConfig()
    return _global_config


def set_config(config: ClyroConfig) -> None:
    """
    Set the global configuration.

    Args:
        config: ClyroConfig instance to use globally
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset global configuration to None (for testing)."""
    global _global_config
    _global_config = None


# ---------------------------------------------------------------------------
# Consolidated from clyro_mcp.config — Implements FRD-006
# MCP/hooks configuration models for YAML-based policy and wrapper config.
# ---------------------------------------------------------------------------

MCP_DEFAULT_CONFIG_PATH = "~/.clyro/mcp-wrapper/mcp-config.yaml"


class LoopDetectionConfig(BaseModel):
    """Loop detection tuning knobs for MCP/hooks context."""

    threshold: int = Field(default=3, ge=1, description="Block after N identical calls")
    window: int = Field(default=10, ge=1, description="Check within last M calls")


class PolicyRule(BaseModel):
    """A single policy rule evaluated against tool call arguments."""

    parameter: str = Field(description='Parameter path, e.g. "amount"')
    operator: str = Field(
        description="Comparison operator: max_value, min_value, equals, "
        "not_equals, in_list, not_in_list, contains, not_contains"
    )
    value: Any = Field(description="Threshold / reference value")
    name: str | None = Field(default=None, description="Human-readable rule name")
    policy_id: str | None = Field(
        default=None, description="UUID from cloud policies, None for local YAML"
    )

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        allowed = {
            "max_value", "min_value", "equals", "not_equals",
            "in_list", "not_in_list", "contains", "not_contains",
        }
        if v not in allowed:
            raise ValueError(f"Unknown operator '{v}'. Allowed: {sorted(allowed)}")
        return v


class ToolConfig(BaseModel):
    """Per-tool policy configuration."""

    policies: list[PolicyRule] = Field(default_factory=list)


class AuditConfig(BaseModel):
    """Audit logging configuration."""

    log_path: str = Field(default="~/.clyro/mcp-wrapper/mcp-audit.jsonl")
    redact_parameters: list[str] = Field(default_factory=list)


class GlobalConfig(BaseModel):
    """Top-level global settings for MCP/hooks."""

    max_steps: int = Field(default=50, ge=1)
    max_cost_usd: float = Field(default=10.0, gt=0)
    cost_per_token_usd: float = Field(default=0.00001, gt=0)
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig)
    policies: list[PolicyRule] = Field(default_factory=list)


class BackendConfig(BaseModel):
    """Backend integration config for MCP/hooks."""

    api_key: str | None = Field(default=None, description="Overridden by CLYRO_API_KEY")
    api_url: str = Field(
        default=DEFAULT_API_URL,
        description="Overridden by CLYRO_API_URL",
    )
    agent_name: str | None = Field(default=None)
    sync_interval_seconds: int = Field(default=5, ge=1, le=300)
    sync_enabled: bool | None = Field(default=None)
    pending_queue_max_mb: int = Field(default=10, ge=1, le=100)


class WrapperConfig(BaseModel):
    """
    Root configuration model for MCP Wrapper and hooks.

    Implements YAML schema with Pydantic validation.
    """

    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    tools: dict[str, ToolConfig] = Field(default_factory=dict)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    model_config = {"populate_by_name": True}

    @property
    def is_backend_enabled(self) -> bool:
        """True if backend sync should be active."""
        api_key = os.environ.get("CLYRO_API_KEY") or self.backend.api_key
        if not api_key:
            return False
        if self.backend.sync_enabled is False:
            return False
        return True

    @property
    def resolved_api_key(self) -> str | None:
        """API key with env var override."""
        return os.environ.get("CLYRO_API_KEY") or self.backend.api_key

    @property
    def resolved_api_url(self) -> str:
        """API URL with env var override."""
        return os.environ.get("CLYRO_API_URL") or self.backend.api_url


def load_mcp_config(config_path: str | None = None) -> WrapperConfig:
    """
    Load and validate MCP/hooks configuration from a YAML file.

    Args:
        config_path: Path to config file. ``None`` uses the default.

    Returns:
        Validated ``WrapperConfig``.
    """
    resolved = Path(os.path.expanduser(config_path or MCP_DEFAULT_CONFIG_PATH))

    if not resolved.exists():
        return WrapperConfig()

    raw_text = resolved.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError:
        sys.exit(1)

    if data is None:
        return WrapperConfig()

    if not isinstance(data, dict):
        sys.exit(1)

    # Warn about unknown top-level keys (forward-compat)
    known_keys = {f.alias or name for name, f in WrapperConfig.model_fields.items()}
    known_keys |= set(WrapperConfig.model_fields.keys())
    unknown = set(data.keys()) - known_keys
    if unknown:
        import sys as _sys

        print(
            f"unknown_config_keys: {sorted(unknown)}",
            file=_sys.stderr,
        )

    from pydantic import ValidationError

    try:
        return WrapperConfig.model_validate(data)
    except ValidationError:
        sys.exit(1)


# Backward-compatible alias for MCP code that imports load_config
load_config = load_mcp_config
