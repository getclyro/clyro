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


class TestFixedFloorsCannotBeWeakened:
    """TDD §3 — liveness and reconnect are *fixed floors*, not free knobs.

    Regression: both were exposed as unbounded config values, so an operator
    could set ``liveness_secs: 99999`` (a dead connection is never detected —
    FRD-049 void) or ``max_attempts: 1000000`` (a reconnect storm — FRD-056's
    DoS bound, TDD §12, void). They may now only be tightened.
    """

    def test_liveness_cannot_exceed_the_guarantee(self) -> None:
        from pydantic import ValidationError

        from clyro.config import ServerConfig

        with pytest.raises(ValidationError):
            ServerConfig(liveness_secs=99999)  # would void FRD-049

    def test_liveness_may_be_tightened(self) -> None:
        from clyro.config import ServerConfig

        assert ServerConfig(liveness_secs=3).liveness_secs == 3  # stricter is fine

    def test_reconnect_cap_cannot_be_raised(self) -> None:
        from pydantic import ValidationError

        from clyro.config import ReconnectConfig

        with pytest.raises(ValidationError):
            ReconnectConfig(max_attempts=1000)  # would void FRD-056's DoS bound

    def test_reconnect_cap_may_be_lowered(self) -> None:
        from clyro.config import ReconnectConfig

        assert ReconnectConfig(max_attempts=2).max_attempts == 2  # fewer is fine


class TestConfigErrorsAreExplained:
    """A bad config must exit(1) WITH a message, never silently (operator UX).

    Regression: load_mcp_config used to ``sys.exit(1)`` with no output on YAML
    errors, non-dict configs, and validation errors, so an operator with a
    typo (e.g. transport: htttp) got a silent vanish.
    """

    def _write(self, text: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(text)
            return f.name

    def test_invalid_transport_prints_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._write("transport: htttp\n")
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_config(path)
            assert exc_info.value.code == 1
            err = capsys.readouterr().err
            assert "invalid config" in err
            assert "htttp" in err  # names the actual bad value
        finally:
            os.unlink(path)

    def test_malformed_yaml_prints_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._write("{{invalid yaml: [")
        try:
            with pytest.raises(SystemExit):
                load_config(path)
            assert "not valid YAML" in capsys.readouterr().err
        finally:
            os.unlink(path)

    def test_non_dict_config_prints_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._write("- just\n- a\n- list\n")
        try:
            with pytest.raises(SystemExit):
                load_config(path)
            assert "expected a mapping" in capsys.readouterr().err
        finally:
            os.unlink(path)


class TestConfigUnknownKeys:
    """TDD §11.1 #27 — unknown top-level key → warning, not error."""

    def test_unknown_keys_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("default_action: allow\nglobal:\n  max_steps: 10\nfuture_section:\n  x: 1\n")
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
default_action: allow
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
      action: block

tools:
  query_database:
    policies:
      - parameter: sql
        operator: contains
        value: DROP
        name: no-drop
        action: block

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


class TestDefaultActionRequired:
    """Regression: default_action is required on WrapperConfig.

    Locks the contract — if anyone re-introduces a default value for
    default_action, this test will fail and force a deliberate change.
    """

    def test_missing_default_action_raises(self) -> None:
        from pydantic import ValidationError

        from clyro.config import WrapperConfig

        with pytest.raises(ValidationError, match="default_action"):
            WrapperConfig.model_validate({"global": {"max_steps": 50}})

    def test_bare_construction_raises(self) -> None:
        from pydantic import ValidationError

        from clyro.config import WrapperConfig

        with pytest.raises(ValidationError, match="default_action"):
            WrapperConfig()
