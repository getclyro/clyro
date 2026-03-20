# Tests for Clyro SDK Configuration
# Implements PRD-002

"""Unit tests for SDK configuration."""

import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from clyro.constants import DEFAULT_API_URL
from clyro.config import (
    DEFAULT_PRICING,
    ClyroConfig,
    ExecutionControls,
    get_config,
    reset_config,
    set_config,
)
from clyro.exceptions import ClyroConfigError


class TestExecutionControls:
    """Tests for ExecutionControls configuration."""

    def test_default_values(self):
        """Test default execution control values."""
        controls = ExecutionControls()
        assert controls.max_steps == 100
        assert controls.max_cost_usd == 10.0
        assert controls.loop_detection_threshold == 3
        assert controls.enable_step_limit is True
        assert controls.enable_cost_limit is True
        assert controls.enable_loop_detection is True

    def test_custom_values(self):
        """Test custom execution control values."""
        controls = ExecutionControls(
            max_steps=50,
            max_cost_usd=5.0,
            loop_detection_threshold=5,
            enable_loop_detection=False,
        )
        assert controls.max_steps == 50
        assert controls.max_cost_usd == 5.0
        assert controls.loop_detection_threshold == 5
        assert controls.enable_loop_detection is False

    def test_min_max_validation(self):
        """Test min/max value validation."""
        # max_steps must be >= 1
        with pytest.raises(ValidationError):
            ExecutionControls(max_steps=0)

        # max_cost_usd must be >= 0
        with pytest.raises(ValidationError):
            ExecutionControls(max_cost_usd=-1.0)

        # loop_detection_threshold must be >= 2
        with pytest.raises(ValidationError):
            ExecutionControls(loop_detection_threshold=1)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ExecutionControls(unknown_field=True)


class TestClyroConfig:
    """Tests for ClyroConfig configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClyroConfig()
        assert config.api_key is None
        assert config.endpoint == DEFAULT_API_URL
        assert config.agent_name is None
        assert config.local_storage_max_mb == 100
        assert config.fail_open is True
        assert config.sync_interval_seconds == 5.0
        assert config.batch_size == 100
        assert config.capture_inputs is True
        assert config.capture_outputs is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClyroConfig(
            api_key="cly_test_abc123",
            endpoint="https://custom.api.io",
            agent_name="my-agent",
            local_storage_max_mb=50,
            fail_open=False,
            controls=ExecutionControls(max_steps=200),
        )
        assert config.api_key == "cly_test_abc123"
        assert config.endpoint == "https://custom.api.io"
        assert config.agent_name == "my-agent"
        assert config.local_storage_max_mb == 50
        assert config.fail_open is False
        assert config.controls.max_steps == 200

    def test_endpoint_validation(self):
        """Test endpoint URL validation."""
        # Valid endpoints
        config = ClyroConfig(endpoint="http://localhost:8000")
        assert config.endpoint == "http://localhost:8000"

        config = ClyroConfig(endpoint="https://api.example.com/")
        assert config.endpoint == "https://api.example.com"  # Trailing slash removed

        # Invalid endpoint
        with pytest.raises(ClyroConfigError):
            ClyroConfig(endpoint="not-a-url")

    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid API keys
        config = ClyroConfig(api_key="cly_live_abc123")
        assert config.api_key == "cly_live_abc123"

        config = ClyroConfig(api_key="cly_test_xyz789")
        assert config.api_key == "cly_test_xyz789"

        # Empty/whitespace API key should fail
        with pytest.raises(ClyroConfigError):
            ClyroConfig(api_key="   ")

    def test_local_storage_path_expansion(self):
        """Test that storage path is expanded."""
        config = ClyroConfig(local_storage_path="~/clyro/data.db")
        assert "~" not in config.local_storage_path
        assert str(Path.home()) in config.local_storage_path

    def test_default_storage_path(self):
        """Test default storage path is set."""
        config = ClyroConfig()
        path = config.get_storage_path()
        assert path.name == "traces.db"
        assert ".clyro" in str(path)

    def test_is_local_only(self):
        """Test local-only mode detection."""
        # No API key = local only
        config = ClyroConfig()
        assert config.is_local_only() is True

        # With API key = not local only
        config = ClyroConfig(api_key="cly_live_abc")
        assert config.is_local_only() is False

    def test_get_model_pricing_exact_match(self):
        """Test getting pricing for exact model match."""
        config = ClyroConfig()
        input_price, output_price = config.get_model_pricing("gpt-4-turbo")
        assert input_price == Decimal("0.01")
        assert output_price == Decimal("0.03")

    def test_get_model_pricing_partial_match(self):
        """Test getting pricing for partial model match."""
        config = ClyroConfig()
        # "gpt-4-turbo-preview" should match "gpt-4-turbo"
        input_price, output_price = config.get_model_pricing("gpt-4-turbo-preview")
        assert input_price == Decimal("0.01")
        assert output_price == Decimal("0.03")

    def test_get_model_pricing_unknown_model(self):
        """Test getting pricing for unknown model."""
        config = ClyroConfig()
        input_price, output_price = config.get_model_pricing("unknown-model-xyz")
        # Should return default fallback pricing
        assert input_price == Decimal("0.01")
        assert output_price == Decimal("0.03")

    def test_custom_pricing(self):
        """Test custom pricing configuration."""
        config = ClyroConfig(
            pricing={
                "custom-model": {"input": 0.002, "output": 0.006},
            }
        )
        input_price, output_price = config.get_model_pricing("custom-model")
        assert input_price == Decimal("0.002")
        assert output_price == Decimal("0.006")

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ClyroConfigError):
            ClyroConfig(unknown_field="value")


class TestClyroConfigFromEnv:
    """Tests for ClyroConfig.from_env()."""

    def test_from_env_empty(self):
        """Test from_env with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = ClyroConfig.from_env()
            assert config.api_key is None
            assert config.endpoint == DEFAULT_API_URL

    def test_from_env_with_api_key(self):
        """Test from_env with CLYRO_API_KEY."""
        with patch.dict(os.environ, {"CLYRO_API_KEY": "cly_test_env"}):
            config = ClyroConfig.from_env()
            assert config.api_key == "cly_test_env"

    def test_from_env_with_endpoint(self):
        """Test from_env with CLYRO_ENDPOINT."""
        with patch.dict(os.environ, {"CLYRO_ENDPOINT": "http://localhost:8000"}):
            config = ClyroConfig.from_env()
            assert config.endpoint == "http://localhost:8000"

    def test_from_env_with_controls(self):
        """Test from_env with execution control variables."""
        env_vars = {
            "CLYRO_MAX_STEPS": "50",
            "CLYRO_MAX_COST_USD": "5.0",
        }
        with patch.dict(os.environ, env_vars):
            config = ClyroConfig.from_env()
            assert config.controls.max_steps == 50
            assert config.controls.max_cost_usd == 5.0

    def test_from_env_with_fail_open(self):
        """Test from_env with CLYRO_FAIL_OPEN."""
        with patch.dict(os.environ, {"CLYRO_FAIL_OPEN": "false"}):
            config = ClyroConfig.from_env()
            assert config.fail_open is False

        with patch.dict(os.environ, {"CLYRO_FAIL_OPEN": "true"}):
            config = ClyroConfig.from_env()
            assert config.fail_open is True


class TestGlobalConfig:
    """Tests for global configuration management."""

    def setup_method(self):
        """Reset global config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset global config after each test."""
        reset_config()

    def test_get_config_returns_default(self):
        """Test get_config returns default when not set."""
        config = get_config()
        assert config is not None
        assert config.api_key is None
        assert config.endpoint == DEFAULT_API_URL

    def test_set_config(self):
        """Test set_config sets global configuration."""
        custom_config = ClyroConfig(api_key="cly_test_custom")
        set_config(custom_config)

        config = get_config()
        assert config.api_key == "cly_test_custom"

    def test_reset_config(self):
        """Test reset_config clears global configuration."""
        set_config(ClyroConfig(api_key="cly_test_reset"))
        reset_config()

        # Next get_config should create new default
        config = get_config()
        assert config.api_key is None


class TestDefaultPricing:
    """Tests for DEFAULT_PRICING constant."""

    def test_default_pricing_contains_common_models(self):
        """Test that DEFAULT_PRICING contains common models."""
        assert "gpt-4-turbo" in DEFAULT_PRICING
        assert "gpt-4o" in DEFAULT_PRICING
        assert "claude-3-opus" in DEFAULT_PRICING
        assert "claude-3-sonnet" in DEFAULT_PRICING

    def test_default_pricing_structure(self):
        """Test that pricing entries have correct structure."""
        for model, pricing in DEFAULT_PRICING.items():
            assert "input" in pricing, f"{model} missing 'input' price"
            assert "output" in pricing, f"{model} missing 'output' price"
            assert isinstance(pricing["input"], (int, float)), f"{model} input price should be numeric"
            assert isinstance(pricing["output"], (int, float)), f"{model} output price should be numeric"
            assert pricing["input"] >= 0, f"{model} input price should be non-negative"
            assert pricing["output"] >= 0, f"{model} output price should be non-negative"
