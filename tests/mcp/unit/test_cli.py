"""
Unit tests for CLI argument parsing — FRD-009.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from clyro.mcp.cli import _build_parser, main


class TestCliArgumentParsing:
    """CLI parser behaviour."""

    def test_wrap_subcommand_required(self) -> None:
        """No subcommand → exit 1."""
        with patch("sys.argv", ["clyro-mcp"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_wrap_missing_server_command(self) -> None:
        """wrap with no server → exit 1."""
        with patch("sys.argv", ["clyro-mcp", "wrap"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_version_flag(self) -> None:
        """--version prints version and exits 0."""
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_parser_wrap_with_server(self) -> None:
        """wrap <command> parses correctly."""
        parser = _build_parser()
        args = parser.parse_args(["wrap", "npx", "server-fs", "/path"])
        assert args.command == "wrap"
        assert args.server_command == ["npx", "server-fs", "/path"]

    def test_parser_wrap_with_config(self) -> None:
        """--config flag parsed correctly."""
        parser = _build_parser()
        args = parser.parse_args(["wrap", "--config", "/my/config.yaml", "npx", "server"])
        assert args.config == "/my/config.yaml"
        assert args.server_command == ["npx", "server"]


class TestBuildTransportBadConfig:
    """A bad HTTP config must refuse cleanly (exit 1 + message), not traceback.

    Regression: a missing CA bundle raised TransportError from TlsPolicy that
    escaped _build_transport uncaught, dumping a Python traceback (bug #4).
    """

    def test_missing_ca_bundle_exits_cleanly(self, tmp_path, capsys) -> None:
        from clyro.config import load_config
        from clyro.mcp.cli import _build_transport

        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "default_action: allow\n"
            "transport: http\n"
            "server:\n"
            "  url: https://example.com/mcp\n"
            "  ca_bundle: /tmp/definitely-missing-bundle.pem\n"
        )
        config = load_config(str(cfg_file))
        with pytest.raises(SystemExit) as exc_info:
            _build_transport(config, [])
        assert exc_info.value.code == 1
        assert "TLS CA bundle not found" in capsys.readouterr().err


class TestHttpSelectableFromConfig:
    """HTTP can be selected in the config file alone — no CLI --transport/--url.

    Regression: the ``server_command_required`` guard decided from CLI flags only
    (``args.transport``/``args.url``), so ``transport: http`` in the config file
    was rejected before the config was ever read. An operator could not configure
    HTTP purely via config.
    """

    def test_http_in_config_is_not_rejected(self, tmp_path) -> None:
        cfg = tmp_path / "http.yaml"
        cfg.write_text(
            "default_action: allow\n"
            "transport: http\n"
            "server:\n"
            "  url: http://localhost:3001/mcp\n"
            "  allow_plaintext: true\n"
        )
        # Stub asyncio.run so we don't actually connect; we only assert the guard
        # let us through to the async entry instead of exiting with code 1.
        with patch("sys.argv", ["clyro-mcp", "wrap", "--config", str(cfg)]), \
             patch("clyro.mcp.cli.asyncio.run", return_value=0) as run:
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0  # got past the server_command_required guard
        run.assert_called_once()         # reached _async_main

    def test_stdio_in_config_still_requires_command(self, tmp_path) -> None:
        # The fix must not weaken the stdio requirement.
        cfg = tmp_path / "stdio.yaml"
        cfg.write_text("default_action: allow\n")  # transport defaults to stdio
        with patch("sys.argv", ["clyro-mcp", "wrap", "--config", str(cfg)]), \
             patch("clyro.mcp.cli.asyncio.run", return_value=0) as run:
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1  # still refused
        run.assert_not_called()          # never reached the async entry
