"""Unit tests for CLI entry points."""

import argparse
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from clyro.hooks.cli import _ensure_agent_id, cmd_evaluate, cmd_trace, main
from clyro.hooks.models import HookOutput, SessionState


class TestCmdEvaluate:
    def _make_args(self, config=None):
        return argparse.Namespace(config=config)

    def test_empty_stdin_returns_fail_open(self):
        args = self._make_args()
        with patch("sys.stdin", StringIO("")):
            result = cmd_evaluate(args)
        assert result == 1  # EXIT_FAIL_OPEN

    def test_invalid_json_returns_fail_open(self):
        args = self._make_args()
        with patch("sys.stdin", StringIO("NOT JSON")):
            result = cmd_evaluate(args)
        assert result == 1

    def test_missing_session_id_returns_fail_open(self):
        args = self._make_args()
        data = json.dumps({"tool_name": "Bash", "tool_input": {}})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_evaluate(args)
        assert result == 1

    def test_missing_tool_name_returns_fail_open(self):
        args = self._make_args()
        data = json.dumps({"session_id": "s1", "tool_input": {}})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_evaluate(args)
        assert result == 1

    @patch("clyro.hooks.cli.StateLock")
    @patch("clyro.hooks.cli.evaluate")
    @patch("clyro.hooks.cli.load_hook_config")
    def test_allow_returns_exit_0_empty_stdout(
        self, mock_config, mock_eval, mock_lock, tmp_path, capsys
    ):
        mock_config.return_value = MagicMock()
        mock_config.return_value.audit.log_path = str(tmp_path / "audit.jsonl")
        mock_config.return_value.audit.redact_parameters = []
        mock_eval.return_value = None  # Allow
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        args = self._make_args()
        data = json.dumps({"session_id": "s1", "tool_name": "Bash", "tool_input": {"command": "ls"}})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_evaluate(args)

        assert result == 0
        captured = capsys.readouterr()
        assert captured.out.strip() == ""  # Empty stdout = allow

    @patch("clyro.hooks.cli.StateLock")
    @patch("clyro.hooks.cli.evaluate")
    @patch("clyro.hooks.cli.load_hook_config")
    def test_block_returns_exit_0_with_json(
        self, mock_config, mock_eval, mock_lock, tmp_path, capsys
    ):
        mock_config.return_value = MagicMock()
        mock_config.return_value.audit.log_path = str(tmp_path / "audit.jsonl")
        mock_config.return_value.audit.redact_parameters = []
        mock_eval.return_value = HookOutput(decision="block", reason="Step limit exceeded")
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        args = self._make_args()
        data = json.dumps({"session_id": "s1", "tool_name": "Bash", "tool_input": {}})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_evaluate(args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["decision"] == "block"

    @patch("clyro.hooks.cli.StateLock")
    @patch("clyro.hooks.cli.load_hook_config")
    def test_lock_timeout_returns_fail_closed(self, mock_config, mock_lock):
        """Lock timeout should fail-closed (exit 2) — deny unevaluated calls."""
        mock_config.return_value = MagicMock()
        mock_config.return_value.audit.log_path = "/dev/null"
        mock_config.return_value.audit.redact_parameters = []
        mock_lock.return_value.__enter__ = MagicMock(side_effect=TimeoutError)

        args = self._make_args()
        data = json.dumps({"session_id": "s1", "tool_name": "Bash", "tool_input": {}})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_evaluate(args)
        assert result == 2  # EXIT_FAIL_CLOSED


class TestCmdTrace:
    def _make_args(self, event=None, config=None):
        return argparse.Namespace(event=event, config=config)

    def test_missing_event_returns_fail_open(self):
        args = self._make_args(event=None)
        result = cmd_trace(args)
        assert result == 1

    def test_unknown_event_returns_fail_open(self):
        args = self._make_args(event="unknown-event")
        result = cmd_trace(args)
        assert result == 1

    def test_invalid_stdin_returns_exit_0(self):
        args = self._make_args(event="tool-complete")
        with patch("sys.stdin", StringIO("NOT JSON")):
            result = cmd_trace(args)
        assert result == 0  # PostToolUse must not interfere

    @patch("clyro.hooks.cli.handle_tool_complete")
    @patch("clyro.hooks.cli.load_hook_config")
    def test_tool_complete_calls_handler(self, mock_config, mock_handler, tmp_path):
        mock_config.return_value = MagicMock()
        mock_config.return_value.audit.log_path = str(tmp_path / "audit.jsonl")
        mock_config.return_value.audit.redact_parameters = []

        args = self._make_args(event="tool-complete")
        data = json.dumps({
            "session_id": "s1", "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_result": {"stdout": "file.txt"},
        })
        with patch("sys.stdin", StringIO(data)):
            result = cmd_trace(args)

        assert result == 0
        mock_handler.assert_called_once()

    @patch("clyro.hooks.cli.handle_session_end")
    @patch("clyro.hooks.cli.load_hook_config")
    def test_session_end_calls_handler(self, mock_config, mock_handler, tmp_path):
        mock_config.return_value = MagicMock()
        mock_config.return_value.audit.log_path = str(tmp_path / "audit.jsonl")
        mock_config.return_value.audit.redact_parameters = []

        args = self._make_args(event="session-end")
        data = json.dumps({"session_id": "s1"})
        with patch("sys.stdin", StringIO(data)):
            result = cmd_trace(args)

        assert result == 0
        mock_handler.assert_called_once()


class TestEnsureAgentId:
    @patch("clyro.hooks.cli.save_state")
    @patch("clyro.hooks.cli.load_state")
    @patch("clyro.hooks.cli.resolve_agent_id")
    def test_resolves_and_persists_agent_id(self, mock_resolve, mock_load, mock_save):
        state = SessionState(session_id="s1")
        mock_load.return_value = state

        def set_agent_id(config, st):
            st.agent_id = "resolved-id"
            return "resolved-id"

        mock_resolve.side_effect = set_agent_id
        config = MagicMock()

        _ensure_agent_id(config, "s1")
        mock_save.assert_called_once_with(state)
        assert state.agent_id == "resolved-id"

    @patch("clyro.hooks.cli.save_state")
    @patch("clyro.hooks.cli.load_state")
    @patch("clyro.hooks.cli.resolve_agent_id")
    def test_skips_if_agent_id_already_set(self, mock_resolve, mock_load, mock_save):
        state = SessionState(session_id="s1", agent_id="existing")
        mock_load.return_value = state

        config = MagicMock()
        _ensure_agent_id(config, "s1")
        mock_resolve.assert_not_called()

    @patch("clyro.hooks.cli.load_state", side_effect=Exception("disk error"))
    def test_fails_open_on_error(self, mock_load):
        config = MagicMock()
        # Should not raise
        _ensure_agent_id(config, "s1")


class TestMain:
    def test_no_command_exits(self):
        with patch("sys.argv", ["clyro-hook"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
