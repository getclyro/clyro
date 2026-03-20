"""
Unit tests for ConfigLoader — TDD §11.1 tests #24–#27.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from clyro.config import load_config


class TestConfigDefaults:
    """TDD §11.1 #24 — no config file → defaults applied."""

    def test_defaults_with_no_file(self) -> None:
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg.global_.max_steps == 50
        assert cfg.global_.max_cost_usd == 10.0
        assert cfg.global_.loop_detection.threshold == 3
        assert cfg.global_.loop_detection.window == 10
        assert cfg.global_.cost_per_token_usd == 0.00001
        assert cfg.tools == {}
        assert cfg.global_.policies == []


class TestConfigParseError:
    """TDD §11.1 #25 — malformed YAML → exit(1)."""

    def test_malformed_yaml_exits(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{{invalid yaml: [")
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)


class TestConfigInvalidValues:
    """TDD §11.1 #26 — invalid field values → exit(1)."""

    def test_invalid_threshold_exits(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("global:\n  loop_detection:\n    threshold: 0\n")
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)

    def test_negative_cost_exits(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("global:\n  max_cost_usd: -5\n")
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)


class TestConfigUnknownKeys:
    """TDD §11.1 #27 — unknown top-level key → warning, not error."""

    def test_unknown_keys_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("global:\n  max_steps: 10\nfuture_section:\n  x: 1\n")
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.global_.max_steps == 10
            captured = capsys.readouterr()
            assert "unknown_config_keys" in captured.err
        finally:
            os.unlink(path)


class TestConfigValidLoad:
    """Full config load with tools and audit sections."""

    def test_full_config(self) -> None:
        yaml_text = """\
global:
  max_steps: 25
  max_cost_usd: 5.0
  loop_detection:
    threshold: 5
    window: 20
  policies:
    - parameter: "*.amount"
      operator: max_value
      value: 1000

tools:
  query_database:
    policies:
      - parameter: sql
        operator: contains
        value: DROP
        name: no-drop

audit:
  log_path: /tmp/test-audit.jsonl
  redact_parameters:
    - "*.password"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.global_.max_steps == 25
            assert cfg.global_.max_cost_usd == 5.0
            assert cfg.global_.loop_detection.threshold == 5
            assert len(cfg.global_.policies) == 1
            assert "query_database" in cfg.tools
            assert cfg.tools["query_database"].policies[0].name == "no-drop"
            assert cfg.audit.redact_parameters == ["*.password"]
        finally:
            os.unlink(path)

    def test_empty_file_returns_defaults(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = load_config(path)
            assert cfg.global_.max_steps == 50
        finally:
            os.unlink(path)

    def test_invalid_operator_exits(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "global:\n  policies:\n"
                "    - parameter: x\n      operator: invalid_op\n      value: 1\n"
            )
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)
