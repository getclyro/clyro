# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code hooks dry-run tests (A10) — FRD-005/014/017/018.

In dry_run a would-be block returns allow (None) instead of ``decision=block``,
emits a distinct ``would_block`` trace event with NO error sibling (FRD-017),
never reports a policy violation (FRD-010), stamps the FRD-018 session marker on
every event, and relaxes the fail-closed governance paths to allow (FRD-014).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from clyro.hooks.audit import AuditLogger
from clyro.hooks.config import HookConfig
from clyro.hooks.evaluator import evaluate
from clyro.hooks.models import HookInput, SessionState


@pytest.fixture(autouse=True)
def mock_sessions_dir(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    pending = tmp_path / "pending"
    pending.mkdir()
    with patch("clyro.hooks.state.SESSIONS_DIR", sessions), \
         patch("clyro.hooks.backend.EVENT_QUEUE_DIR", pending), \
         patch("clyro.hooks.evaluator.load_state") as mock_load, \
         patch("clyro.hooks.evaluator.save_state"):
        mock_load.return_value = SessionState(session_id="test-session")
        yield {"load_state": mock_load, "pending": pending}


@pytest.fixture(autouse=True)
def _reset_dry_run():
    # The process-level FRD-018 flag is set by evaluate(); reset around each test.
    from clyro.hooks.backend import set_dry_run

    set_dry_run(False)
    yield
    set_dry_run(False)


def _config(tmp_path, dry_run: bool):
    return HookConfig.model_validate({
        "default_action": "allow",
        "global": {"max_steps": 50, "max_cost_usd": 10.0},
        "audit": {"log_path": str(tmp_path / "audit.jsonl")},
        "backend": {},
        "dry_run": dry_run,
    })


def _enqueued_events(pending):
    """Read all events written to the pending queue files."""
    import json

    events = []
    for f in pending.rglob("*"):
        if f.is_file():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


class TestHooksStepLimitDryRun:
    def test_enforce_blocks(self, mock_sessions_dir, tmp_path):
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=50
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        result = evaluate(
            HookInput(session_id="test-session", tool_name="Bash", tool_input={"command": "ls"}),
            _config(tmp_path, dry_run=False),
            audit,
        )
        assert result is not None and result.decision == "block"
        audit.close()

    def test_dry_run_allows(self, mock_sessions_dir, tmp_path):
        # C6/FRD-005: a would-be block returns allow (None) in dry_run.
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=50
        )
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        result = evaluate(
            HookInput(session_id="test-session", tool_name="Bash", tool_input={"command": "ls"}),
            _config(tmp_path, dry_run=True),
            audit,
        )
        assert result is None  # allow
        audit.close()


class TestHooksWouldBlockEvent:
    def test_dry_run_emits_would_block_no_error_sibling(self, mock_sessions_dir, tmp_path):
        # FRD-017: would_block event present; NO policy_check(block)/error sibling.
        # FRD-018: every event carries the dry_run marker.
        mock_sessions_dir["load_state"].return_value = SessionState(
            session_id="test-session", step_count=50, agent_id="agent-1",
        )
        cfg = HookConfig.model_validate({
            "default_action": "allow",
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "cly_test", "agent_name": "a"},
            "dry_run": True,
        })
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        with patch.object(HookConfig, "resolved_api_key", "cly_test"):
            evaluate(
                HookInput(session_id="test-session", tool_name="Bash", tool_input={"command": "ls"}),
                cfg,
                audit,
            )
        audit.close()

        events = _enqueued_events(mock_sessions_dir["pending"])
        types = [e["event_type"] for e in events]
        assert "would_block" in types
        assert "error" not in types  # FRD-017
        assert not any(
            e["event_type"] == "policy_check" and e["metadata"].get("decision") == "block"
            for e in events
        )
        # FRD-018: session marker on every event
        assert all(e["metadata"].get("dry_run") is True for e in events)
        wb = next(e for e in events if e["event_type"] == "would_block")
        assert wb["error_type"] is None
        assert wb["metadata"]["would_block"]["check_type"] == "step"


class TestHooksWouldBlockLatch:
    """FRD-022 on hooks: the prevention stack runs on EVERY tool call, so a
    tripped limit would emit one marker per call — unbounded. Hooks run as a
    short-lived CLI per call, so the de-dup keys must persist in session state."""

    def _run(self, tmp_path, state, n):
        audit = AuditLogger(log_path=tmp_path / "audit.jsonl")
        cfg = HookConfig.model_validate({
            "default_action": "allow",
            "global": {"max_steps": 50, "max_cost_usd": 10.0},
            "audit": {"log_path": str(tmp_path / "audit.jsonl")},
            "backend": {"api_key": "cly_test", "agent_name": "a"},
            "dry_run": True,
        })
        for i in range(n):
            evaluate(
                HookInput(
                    session_id="test-session", tool_name="Bash",
                    tool_input={"command": f"ls {i}"},
                ),
                cfg, audit,
            )
        audit.close()

    def test_repeated_step_limit_records_one_marker(self, mock_sessions_dir, tmp_path):
        # Same persisted state object across calls == the same session, which is
        # exactly what the real per-call CLI reloads from disk.
        state = SessionState(session_id="test-session", step_count=50, agent_id="agent-1")
        mock_sessions_dir["load_state"].return_value = state
        self._run(tmp_path, state, n=25)

        events = _enqueued_events(mock_sessions_dir["pending"])
        wb = [e for e in events if e["event_type"] == "would_block"]
        assert len(wb) == 1, f"25 blocked calls emitted {len(wb)} markers (expected 1)"
        assert wb[0]["metadata"]["would_block"]["check_type"] == "step"

    def test_latch_keys_persist_in_session_state(self, mock_sessions_dir, tmp_path):
        # The key must be written to the state that gets saved to disk — an
        # in-memory latch would not survive the next CLI invocation.
        state = SessionState(session_id="test-session", step_count=50, agent_id="agent-1")
        mock_sessions_dir["load_state"].return_value = state
        self._run(tmp_path, state, n=3)
        assert "step" in state.would_block_keys

    def test_state_key_list_is_bounded(self):
        from clyro.hooks.evaluator import _WOULD_BLOCK_KEYS_MAX

        assert _WOULD_BLOCK_KEYS_MAX > 0  # a long session cannot grow state unbounded


class TestHooksFailClosedRelaxation:
    def test_lock_timeout_relaxes_to_allow_in_dry_run(self, tmp_path):
        # FRD-014: a governance-stack failure resolves to allow in dry_run.
        from clyro.hooks import cli as hooks_cli

        args = type("A", (), {"config": None})()
        cfg = _config(tmp_path, dry_run=True)
        with patch("clyro.hooks.cli.load_hook_config", return_value=cfg), \
             patch("clyro.hooks.cli.StateLock") as lock, \
             patch("clyro.hooks.cli._read_stdin", return_value={
                 "session_id": "s1", "tool_name": "Bash", "tool_input": {}}):
            lock.return_value.__enter__ = lambda *a: (_ for _ in ()).throw(TimeoutError())
            result = hooks_cli.cmd_evaluate(args)
        assert result == hooks_cli.EXIT_FAIL_OPEN  # allow, not fail-closed

    def test_lock_timeout_fails_closed_in_enforce(self, tmp_path):
        from clyro.hooks import cli as hooks_cli

        args = type("A", (), {"config": None})()
        cfg = _config(tmp_path, dry_run=False)
        with patch("clyro.hooks.cli.load_hook_config", return_value=cfg), \
             patch("clyro.hooks.cli.StateLock") as lock, \
             patch("clyro.hooks.cli._read_stdin", return_value={
                 "session_id": "s1", "tool_name": "Bash", "tool_input": {}}):
            lock.return_value.__enter__ = lambda *a: (_ for _ in ()).throw(TimeoutError())
            result = hooks_cli.cmd_evaluate(args)
        assert result == hooks_cli.EXIT_FAIL_CLOSED
