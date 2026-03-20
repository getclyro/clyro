# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Session State Manager
# Implements FRD-HK-002

"""Session state persistence with file locking and atomic writes."""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
from pathlib import Path

import structlog
from pydantic import ValidationError

from .constants import (
    DIR_PERMISSIONS,
    FILE_PERMISSIONS,
    SESSIONS_DIR,
    STALE_SESSION_AGE_HOURS,
    STATE_LOCK_TIMEOUT_SECONDS,
)
from .models import SessionState

logger = structlog.get_logger()


def _sessions_dir() -> Path:
    """Return sessions directory, creating if needed with 0o700 permissions."""
    sessions = SESSIONS_DIR
    if not sessions.exists():
        sessions.mkdir(parents=True, exist_ok=True)
        os.chmod(sessions, DIR_PERMISSIONS)
    return sessions


_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9._\-]+$")


def state_path(session_id: str) -> Path:
    """Return the path to a session's state file.

    FRD-HK-002: session_id is validated to prevent path traversal.
    Uses whitelist-first approach: reject invalid IDs before any processing,
    then verify the resolved path stays within the sessions directory.
    """
    if not session_id or not _SAFE_SESSION_ID.match(session_id):
        session_id = "invalid-session"
    base = _sessions_dir()
    path = base / f"{session_id}.json"
    # Ensure resolved path is within sessions directory
    if not path.resolve().is_relative_to(base.resolve()):
        path = base / "invalid-session.json"
    return path


class CorruptStateError(Exception):
    """Raised when session state file is corrupt and cannot be loaded safely."""


def load_state(session_id: str) -> SessionState:
    """Load session state from disk, creating fresh state if missing.

    FRD-HK-002: Fail-closed on corruption — raises CorruptStateError rather
    than silently resetting to step=0/cost=0 (which would bypass step limits
    and cost budgets).  Fresh state is only created when no state file exists.
    """
    path = state_path(session_id)
    if not path.exists():
        return SessionState(session_id=session_id)
    try:
        data = json.loads(path.read_text())
        state = SessionState.model_validate(data)
        # Validate field types per FRD-HK-003/004/005
        if not isinstance(state.step_count, int) or state.step_count < 0:
            raise CorruptStateError(f"Invalid step_count: {state.step_count!r}")
        if (
            not isinstance(state.accumulated_cost_usd, (int, float))
            or state.accumulated_cost_usd < 0
        ):
            raise CorruptStateError(f"Invalid accumulated_cost_usd: {state.accumulated_cost_usd!r}")
        if not isinstance(state.loop_history, list):
            raise CorruptStateError(
                f"Invalid loop_history type: {type(state.loop_history).__name__}"
            )
        return state
    except CorruptStateError:
        raise  # Re-raise our own error
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("corrupt_state_fail_closed", session_id=session_id, error=str(e))
        raise CorruptStateError(f"Session state corrupt for {session_id}: {e}") from e


def save_state(state: SessionState) -> None:
    """Atomically write session state to disk.

    FRD-HK-002: write to temp file, then rename.
    """
    path = state_path(state.session_id)
    tmp_path = path.with_suffix(".tmp")
    data = state.model_dump(mode="json")

    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_PERMISSIONS)
    try:
        os.write(fd, json.dumps(data, default=str).encode())
    finally:
        os.close(fd)
    os.rename(str(tmp_path), str(path))


class StateLock:
    """Context manager for exclusive file lock on session state.

    FRD-HK-002: fcntl.flock with LOCK_EX, 5s timeout.
    """

    def __init__(self, session_id: str, timeout: float = STATE_LOCK_TIMEOUT_SECONDS):
        self._path = state_path(session_id).with_suffix(".lock")
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> StateLock:
        _sessions_dir()
        self._fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT, FILE_PERMISSIONS)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise TimeoutError(
                        f"Could not acquire state lock within {self._timeout}s"
                    ) from None
                time.sleep(0.05)

    def __exit__(self, *args) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def cleanup_stale_sessions() -> None:
    """Delete session state files and pending event queues older than 24 hours.

    FRD-HK-002: Only called at session-end (Stop hook), not on every PreToolUse.
    Errors are silently ignored (fail-open).
    """
    cutoff = time.time() - (STALE_SESSION_AGE_HOURS * 3600)

    # Clean stale session state files
    try:
        sessions_dir = _sessions_dir()
        for entry in os.scandir(sessions_dir):
            if entry.name.endswith(".json") and entry.stat().st_mtime < cutoff:
                try:
                    os.unlink(entry.path)
                except OSError:
                    pass
    except Exception:
        pass

    # Clean stale pending event queue files
    try:
        from .constants import EVENT_QUEUE_DIR

        if EVENT_QUEUE_DIR.exists():
            for entry in os.scandir(EVENT_QUEUE_DIR):
                if entry.name.endswith(".jsonl") and entry.stat().st_mtime < cutoff:
                    try:
                        os.unlink(entry.path)
                    except OSError:
                        pass
    except Exception:
        pass
