# Tests for cli.py — C7 (clyro feedback CLI)
# Implements TDD §13.1 C7 test cases

"""
Test coverage target: 85%+
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from clyro.constants import DEFAULT_API_URL, GITHUB_NEW_ISSUE_URL
from clyro.cli import (
    _auto_capture_context,
    _is_headless,
    _open_github_issue,
    main,
)


# ===========================================================================
# Context capture
# ===========================================================================


class TestAutoCapture:
    def test_captures_required_fields(self):
        ctx = _auto_capture_context()
        assert "sdk_version" in ctx
        assert "python_version" in ctx
        assert "platform" in ctx


# ===========================================================================
# Headless detection
# ===========================================================================


class TestHeadlessDetection:
    def test_ci_env_is_headless(self, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert _is_headless() is True

    def test_no_display_linux_is_headless(self, monkeypatch):
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        if sys.platform == "linux":
            assert _is_headless() is True


# ===========================================================================
# Bare command
# ===========================================================================


class TestBareCommand:
    def test_bare_command_prints_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "clyro" in captured.out


# ===========================================================================
# Feedback subcommand
# ===========================================================================


class TestFeedbackCommand:
    def test_message_flag(self, capsys, monkeypatch):
        """--message flag works in non-interactive mode."""
        monkeypatch.delenv("CLYRO_API_KEY", raising=False)

        with patch("clyro.cli._open_github_issue") as mock_gh:
            with pytest.raises(SystemExit) as exc_info:
                main(["feedback", "--message", "test feedback"])
            assert exc_info.value.code == 0
            mock_gh.assert_called_once()
            # First arg is the message
            assert mock_gh.call_args[0][0] == "test feedback"

    def test_no_message_no_tty_exits_1(self, monkeypatch):
        """No --message and not a TTY → exit 1."""
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=MagicMock(return_value=False)))

        with pytest.raises(SystemExit) as exc_info:
            main(["feedback"])
        assert exc_info.value.code == 1

    def test_cloud_mode_submits_to_api(self, monkeypatch, capsys):
        """With CLYRO_API_KEY set, tries cloud submission first."""
        monkeypatch.setenv("CLYRO_API_KEY", "cly_test_123")

        with patch("clyro.cli._submit_cloud_feedback", return_value=True) as mock_cloud:
            with pytest.raises(SystemExit) as exc_info:
                main(["feedback", "-m", "test"])
            assert exc_info.value.code == 0
            mock_cloud.assert_called_once()

    def test_cloud_failure_falls_back_to_github(self, monkeypatch, capsys):
        """Cloud API failure → fall back to GitHub issue."""
        monkeypatch.setenv("CLYRO_API_KEY", "cly_test_123")

        with patch("clyro.cli._submit_cloud_feedback", return_value=False):
            with patch("clyro.cli._open_github_issue") as mock_gh:
                with pytest.raises(SystemExit) as exc_info:
                    main(["feedback", "-m", "test"])
                assert exc_info.value.code == 0
                mock_gh.assert_called_once()


# ===========================================================================
# GitHub issue URL
# ===========================================================================


class TestCloudFeedback:
    """Coverage for _submit_cloud_feedback (lines 65-95)."""

    def test_cloud_submit_success(self, monkeypatch, capsys):
        """Exercise _submit_cloud_feedback with mocked httpx."""
        monkeypatch.setenv("CLYRO_API_KEY", "cly_test_abc")
        monkeypatch.setenv("CLYRO_ENDPOINT", DEFAULT_API_URL)

        from clyro.cli import _submit_cloud_feedback

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_instance.__exit__ = MagicMock(return_value=False)
        mock_client_instance.post = MagicMock(return_value=mock_response)

        with patch("httpx.Client", return_value=mock_client_instance):
            result = _submit_cloud_feedback("test message", {"sdk_version": "0.2.0"})

        assert result is True
        captured = capsys.readouterr()
        assert "Feedback submitted" in captured.err

    def test_cloud_submit_no_api_key(self, monkeypatch):
        """No API key → returns False immediately."""
        monkeypatch.delenv("CLYRO_API_KEY", raising=False)
        from clyro.cli import _submit_cloud_feedback
        assert _submit_cloud_feedback("test", {}) is False

    def test_cloud_submit_network_error(self, monkeypatch, capsys):
        """Network error → returns False and prints warning."""
        monkeypatch.setenv("CLYRO_API_KEY", "cly_test_abc")
        from clyro.cli import _submit_cloud_feedback

        with patch("httpx.Client", side_effect=ConnectionError("offline")):
            result = _submit_cloud_feedback("test", {})

        assert result is False
        captured = capsys.readouterr()
        assert "Cloud feedback failed" in captured.err


class TestInteractiveFeedback:
    """Coverage for interactive TTY prompt (lines 146-150) and empty message (159-161)."""

    def test_interactive_prompt_success(self, monkeypatch, capsys):
        """TTY prompt with valid input."""
        monkeypatch.delenv("CLYRO_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=MagicMock(return_value=True)))

        with patch("builtins.input", return_value="my feedback"):
            with patch("clyro.cli._open_github_issue") as mock_gh:
                with pytest.raises(SystemExit) as exc_info:
                    main(["feedback"])
                assert exc_info.value.code == 0
                mock_gh.assert_called_once()

    def test_interactive_prompt_empty(self, monkeypatch, capsys):
        """TTY prompt with empty input → exit 1."""
        monkeypatch.delenv("CLYRO_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=MagicMock(return_value=True)))

        with patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                main(["feedback"])
            assert exc_info.value.code == 1

    def test_interactive_prompt_keyboard_interrupt(self, monkeypatch, capsys):
        """Ctrl+C during prompt → exit 1."""
        monkeypatch.delenv("CLYRO_API_KEY", raising=False)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=MagicMock(return_value=True)))

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exc_info:
                main(["feedback"])
            assert exc_info.value.code == 1


class TestGitHubIssue:
    def test_url_construction(self, capsys, monkeypatch):
        """URL includes message and context."""
        monkeypatch.setenv("CI", "true")  # Force headless
        _open_github_issue("test msg", {"sdk_version": "0.2.0"})
        captured = capsys.readouterr()
        assert GITHUB_NEW_ISSUE_URL in captured.err
        assert "test+msg" in captured.err or "test%20msg" in captured.err

    def test_long_message_truncated(self, capsys, monkeypatch):
        """TDD §13.4: 10KB message truncated to 2000 chars."""
        monkeypatch.setenv("CI", "true")
        long_msg = "x" * 10000
        _open_github_issue(long_msg, {})
        captured = capsys.readouterr()
        assert "truncated" in captured.err

    def test_browser_open_attempted(self, monkeypatch):
        """In non-headless mode, webbrowser.open is called."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")

        with patch("webbrowser.open", return_value=True) as mock_open:
            _open_github_issue("test", {"sdk_version": "0.2.0"})
            mock_open.assert_called_once()

    def test_browser_failure_prints_url(self, capsys, monkeypatch):
        """Browser open failure → print URL to stderr."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")

        with patch("webbrowser.open", side_effect=OSError("no browser")):
            _open_github_issue("test", {})
            captured = capsys.readouterr()
            assert "github.com" in captured.err
