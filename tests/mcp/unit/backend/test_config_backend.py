"""
Unit tests for BackendConfig and WrapperConfig backend features — TDD §11.1 v1.1 tests.

FRD-010 Update: Backend configuration with env var overrides.
FRD-015: is_backend_enabled property.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from clyro.config import BackendConfig, WrapperConfig, load_config
from clyro.constants import DEFAULT_API_URL


class TestBackendConfigDefaults:
    """BackendConfig default values (FRD-010 Update)."""

    def test_defaults(self) -> None:
        bc = BackendConfig()
        assert bc.api_key is None
        assert bc.api_url == DEFAULT_API_URL
        assert bc.agent_name is None
        assert bc.sync_interval_seconds == 5
        assert bc.sync_enabled is None
        assert bc.pending_queue_max_mb == 10

    def test_sync_interval_validation(self) -> None:
        bc = BackendConfig(sync_interval_seconds=1)
        assert bc.sync_interval_seconds == 1

        with pytest.raises(ValueError):
            BackendConfig(sync_interval_seconds=0)

        with pytest.raises(ValueError):
            BackendConfig(sync_interval_seconds=301)

    def test_pending_queue_max_mb_validation(self) -> None:
        with pytest.raises(ValueError):
            BackendConfig(pending_queue_max_mb=0)

        with pytest.raises(ValueError):
            BackendConfig(pending_queue_max_mb=101)


class TestWrapperConfigBackend:
    """WrapperConfig backend integration (FRD-015)."""

    def test_backend_defaults_in_wrapper(self) -> None:
        cfg = WrapperConfig()
        assert cfg.backend.api_key is None
        assert cfg.backend.api_url == DEFAULT_API_URL

    def test_backend_from_yaml_dict(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {
                "api_key": "my-key",
                "api_url": "https://custom.api.dev",
                "agent_name": "my-agent",
                "sync_interval_seconds": 10,
                "pending_queue_max_mb": 20,
            }
        })
        assert cfg.backend.api_key == "my-key"
        assert cfg.backend.api_url == "https://custom.api.dev"
        assert cfg.backend.agent_name == "my-agent"
        assert cfg.backend.sync_interval_seconds == 10
        assert cfg.backend.pending_queue_max_mb == 20


class TestIsBackendEnabled:
    """is_backend_enabled property (FRD-015)."""

    def test_disabled_when_no_key(self) -> None:
        cfg = WrapperConfig()
        with patch.dict(os.environ, {}, clear=True):
            # Remove CLYRO_API_KEY if present
            os.environ.pop("CLYRO_API_KEY", None)
            assert cfg.is_backend_enabled is False

    def test_enabled_with_yaml_key(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {"api_key": "test-key"}
        })
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_API_KEY", None)
            assert cfg.is_backend_enabled is True

    def test_enabled_with_env_key(self) -> None:
        cfg = WrapperConfig()
        with patch.dict(os.environ, {"CLYRO_API_KEY": "env-key"}):
            assert cfg.is_backend_enabled is True

    def test_disabled_when_sync_enabled_false(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {"api_key": "test-key", "sync_enabled": False}
        })
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_API_KEY", None)
            assert cfg.is_backend_enabled is False


class TestResolvedApiKey:
    """resolved_api_key with env var override (FRD-010 Update)."""

    def test_returns_yaml_key(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {"api_key": "yaml-key"}
        })
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_API_KEY", None)
            assert cfg.resolved_api_key == "yaml-key"

    def test_env_overrides_yaml(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {"api_key": "yaml-key"}
        })
        with patch.dict(os.environ, {"CLYRO_API_KEY": "env-key"}):
            assert cfg.resolved_api_key == "env-key"

    def test_returns_none_when_no_key(self) -> None:
        cfg = WrapperConfig()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_API_KEY", None)
            assert cfg.resolved_api_key is None


class TestResolvedApiUrl:
    """resolved_api_url with env var override (FRD-010 Update)."""

    def test_returns_yaml_url(self) -> None:
        cfg = WrapperConfig.model_validate({
            "backend": {"api_url": "https://custom.dev"}
        })
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLYRO_API_URL", None)
            assert cfg.resolved_api_url == "https://custom.dev"

    def test_env_overrides_yaml(self) -> None:
        cfg = WrapperConfig()
        with patch.dict(os.environ, {"CLYRO_API_URL": "https://env.dev"}):
            assert cfg.resolved_api_url == "https://env.dev"


class TestLoadConfigWithBackend:
    """load_config handles backend section (FRD-010 Update)."""

    def test_backend_key_not_unknown(self, tmp_path) -> None:
        """'backend' key should not trigger unknown key warning."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("backend:\n  api_key: test\n")
        cfg = load_config(str(config_file))
        assert cfg.backend.api_key == "test"
