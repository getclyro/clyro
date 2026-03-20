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
