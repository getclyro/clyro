"""Unit tests for configuration loading."""

import os
from unittest.mock import patch

import pytest
import yaml

from clyro.hooks.config import ConfigError, HookConfig, load_hook_config


class TestHookConfig:
    def test_default_config(self):
        config = HookConfig.model_validate({
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {},
            "backend": {},
        })
        assert config.global_.max_steps == 50
        assert config.policy_cache_ttl_seconds == 300

    def test_custom_cache_ttl(self):
        config = HookConfig.model_validate({
            "global": {},
            "audit": {},
            "backend": {},
            "policy_cache_ttl_seconds": 600,
        })
        assert config.policy_cache_ttl_seconds == 600


class TestLoadHookConfig:
    def test_missing_file_uses_defaults(self, tmp_path):
        config = load_hook_config(str(tmp_path / "nonexistent.yaml"))
        assert config.global_.max_steps == 50
        assert config.global_.max_cost_usd == 10.0

    def test_valid_yaml(self, tmp_path):
        config_data = {
            "global": {"max_steps": 100, "max_cost_usd": 5.0},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_data))

        config = load_hook_config(str(path))
        assert config.global_.max_steps == 100
        assert config.global_.max_cost_usd == 5.0

    def test_invalid_yaml_raises_config_error(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid: yaml: [[[")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_hook_config(str(path))

    def test_env_var_override_api_key(self, tmp_path):
        config_data = {"global": {}, "backend": {"api_key": "yaml-key"}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {"CLYRO_API_KEY": "env-key"}):
            config = load_hook_config(str(path))
            assert config.backend.api_key == "env-key"

    def test_env_var_override_api_url(self, tmp_path):
        config_data = {"global": {}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {"CLYRO_API_URL": "http://custom:8000"}):
            config = load_hook_config(str(path))
            assert config.backend.api_url == "http://custom:8000"

    def test_invalid_schema_raises_config_error(self, tmp_path):
        """Schema validation errors should raise ConfigError, not SystemExit."""
        config_data = {"global": {"max_steps": "not_a_number"}}
        path = tmp_path / "bad_schema.yaml"
        path.write_text(yaml.dump(config_data))

        with pytest.raises(ConfigError, match="Config validation error"):
            load_hook_config(str(path))

    def test_policies_in_config(self, tmp_path):
        config_data = {
            "global": {
                "policies": [
                    {"parameter": "command", "operator": "contains", "value": "rm -rf",
                     "name": "Block rm -rf"},
                ],
            },
            "tools": {
                "Bash": {
                    "policies": [
                        {"parameter": "command", "operator": "contains", "value": "sudo",
                         "name": "Block sudo"},
                    ],
                },
            },
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_data))

        config = load_hook_config(str(path))
        assert len(config.global_.policies) == 1
        assert config.global_.policies[0].name == "Block rm -rf"
        assert len(config.tools["Bash"].policies) == 1
