"""Unit tests for session state management."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from clyro.hooks.models import SessionState
from clyro.hooks.state import (
    StateLock,
    cleanup_stale_sessions,
    load_state,
    save_state,
    state_path,
)


@pytest.fixture(autouse=True)
def mock_sessions_dir(tmp_path):
    """Redirect all state operations to a temp directory."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions):
        yield sessions


class TestLoadState:
    def test_new_session_returns_fresh_state(self):
        state = load_state("new-session")
        assert state.session_id == "new-session"
        assert state.step_count == 0
        assert state.accumulated_cost_usd == 0.0
        assert state.loop_history == []

    def test_existing_state_loaded(self, mock_sessions_dir):
        path = mock_sessions_dir / "existing.json"
        data = SessionState(session_id="existing", step_count=10).model_dump(mode="json")
        path.write_text(json.dumps(data, default=str))

        state = load_state("existing")
        assert state.session_id == "existing"
        assert state.step_count == 10

    def test_corrupt_json_fails_closed(self, mock_sessions_dir):
        """Corrupt JSON should raise CorruptStateError (fail-closed)."""
        from clyro.hooks.state import CorruptStateError

        path = mock_sessions_dir / "corrupt.json"
        path.write_text("NOT VALID JSON {{{")

        with pytest.raises(CorruptStateError):
            load_state("corrupt")

    def test_invalid_schema_fails_closed(self, mock_sessions_dir):
        """Invalid schema should raise CorruptStateError (fail-closed)."""
        from clyro.hooks.state import CorruptStateError

        path = mock_sessions_dir / "bad-schema.json"
        path.write_text(json.dumps({"session_id": "bad-schema", "step_count": "not_an_int"}))

        with pytest.raises(CorruptStateError):
            load_state("bad-schema")

    def test_negative_step_count_fails_closed(self, mock_sessions_dir):
        """Negative step_count should raise CorruptStateError (fail-closed)."""
        from clyro.hooks.state import CorruptStateError

        path = mock_sessions_dir / "neg.json"
        data = SessionState(session_id="neg").model_dump(mode="json")
        data["step_count"] = -5
        path.write_text(json.dumps(data, default=str))

        with pytest.raises(CorruptStateError, match="Invalid step_count"):
            load_state("neg")


class TestSaveState:
    def test_atomic_write(self, mock_sessions_dir):
        state = SessionState(session_id="atomic-test", step_count=7)
        save_state(state)

        path = mock_sessions_dir / "atomic-test.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["step_count"] == 7

    def test_no_temp_file_left(self, mock_sessions_dir):
        state = SessionState(session_id="clean-test")
        save_state(state)

        tmp = mock_sessions_dir / "clean-test.tmp"
        assert not tmp.exists()


class TestStateLock:
    def test_lock_acquire_and_release(self, mock_sessions_dir):
        with StateLock("lock-test", timeout=2):
            # Inside lock — should be able to read/write state
            state = SessionState(session_id="lock-test")
            save_state(state)
        # After release — should still work
        loaded = load_state("lock-test")
        assert loaded.session_id == "lock-test"

    def test_lock_timeout_raises(self, mock_sessions_dir):
        # Acquire lock, then try to acquire again with short timeout
        lock_path = mock_sessions_dir / "timeout-test.lock"
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)

        with pytest.raises(TimeoutError):
            with StateLock("timeout-test", timeout=0.2):
                pass

        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class TestStatePath:
    def test_path_traversal_sanitized(self, mock_sessions_dir):
        """Session IDs with path traversal chars must not escape sessions dir."""
        path = state_path("../../etc/passwd")
        assert mock_sessions_dir in path.parents or path.parent == mock_sessions_dir
        assert ".." not in str(path.name)

    def test_slash_stripped(self, mock_sessions_dir):
        path = state_path("session/with/slashes")
        assert "/" not in path.stem

    def test_valid_session_id(self, mock_sessions_dir):
        path = state_path("valid-session-123.abc")
        assert path.stem == "valid-session-123.abc"


class TestCleanupStaleSessions:
    def test_removes_old_files(self, mock_sessions_dir):
        # Create an old file
        old_path = mock_sessions_dir / "old-session.json"
        old_path.write_text("{}")
        # Set mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        os.utime(old_path, (old_time, old_time))

        # Create a fresh file
        fresh_path = mock_sessions_dir / "fresh-session.json"
        fresh_path.write_text("{}")

        cleanup_stale_sessions()

        assert not old_path.exists()
        assert fresh_path.exists()

    def test_ignores_non_json_files(self, mock_sessions_dir):
        lock_path = mock_sessions_dir / "session.lock"
        lock_path.write_text("")
        old_time = time.time() - 48 * 3600
        os.utime(lock_path, (old_time, old_time))

        cleanup_stale_sessions()
        assert lock_path.exists()
