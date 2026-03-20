"""
Unit tests for EventQueue — TDD §11.1 v1.1 tests.

FRD-018: File-based JSONL persistence of unsynced trace events.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from clyro.backend.event_queue import EventQueue


@pytest.fixture
def queue(tmp_path) -> EventQueue:
    """EventQueue using a temp directory."""
    q = EventQueue(instance_id="test12345678", max_size_mb=1)
    q._path = tmp_path / "mcp-pending-test12345678.jsonl"
    return q


def _make_event(event_id: str | None = None) -> dict:
    return {
        "event_id": event_id or str(uuid4()),
        "event_type": "tool_call",
        "tool_name": "test_tool",
    }


class TestEventQueueAppend:
    """Append events to queue file (FRD-018)."""

    def test_append_creates_file(self, queue: EventQueue) -> None:
        event = _make_event()
        queue.append(event)
        assert queue.file_path.exists()

    def test_append_adds_queued_at(self, queue: EventQueue) -> None:
        event = _make_event()
        queue.append(event)
        loaded = queue.load_pending()
        assert len(loaded) == 1
        assert "queued_at" in loaded[0]

    def test_append_multiple(self, queue: EventQueue) -> None:
        for _ in range(5):
            queue.append(_make_event())
        assert queue.pending_count == 5


class TestEventQueueLoadPending:
    """Load pending events from queue file (FRD-018)."""

    def test_empty_file(self, queue: EventQueue) -> None:
        assert queue.load_pending() == []

    def test_skips_corrupted_lines(self, queue: EventQueue, capsys) -> None:
        queue.append(_make_event())
        # Manually inject a corrupted line
        with open(queue.file_path, "a") as f:
            f.write("not valid json\n")
        queue.append(_make_event())

        loaded = queue.load_pending()
        assert len(loaded) == 2  # Skipped the corrupted line
        captured = capsys.readouterr()
        assert "corrupted_queue_line" in captured.err

    def test_skips_empty_lines(self, queue: EventQueue) -> None:
        queue.append(_make_event())
        with open(queue.file_path, "a") as f:
            f.write("\n\n")
        queue.append(_make_event())
        assert len(queue.load_pending()) == 2


class TestEventQueueRemoveSynced:
    """Remove synced events via atomic temp-file + rename (FRD-018)."""

    def test_removes_by_event_id(self, queue: EventQueue) -> None:
        e1 = _make_event("id-1")
        e2 = _make_event("id-2")
        e3 = _make_event("id-3")
        queue.append(e1)
        queue.append(e2)
        queue.append(e3)

        queue.remove_synced({"id-1", "id-3"})
        remaining = queue.load_pending()
        assert len(remaining) == 1
        assert remaining[0]["event_id"] == "id-2"

    def test_remove_all(self, queue: EventQueue) -> None:
        e1 = _make_event("id-1")
        queue.append(e1)
        queue.remove_synced({"id-1"})
        assert queue.pending_count == 0


class TestEventQueuePendingCount:
    """Count of unsynced events (FRD-018)."""

    def test_zero_initially(self, queue: EventQueue) -> None:
        assert queue.pending_count == 0

    def test_increments_on_append(self, queue: EventQueue) -> None:
        queue.append(_make_event())
        assert queue.pending_count == 1
        queue.append(_make_event())
        assert queue.pending_count == 2


class TestEventQueueSizeLimit:
    """Prune oldest events if file exceeds max size (FRD-018)."""

    def test_enforces_size_limit(self, tmp_path) -> None:
        # Create a queue with a very small max size (1KB)
        q = EventQueue(instance_id="size_test", max_size_mb=1)
        q._path = tmp_path / "mcp-pending-size_test.jsonl"
        # Set absurdly small limit to test pruning
        q._max_size_bytes = 200  # ~200 bytes

        # Write enough events to exceed limit
        for i in range(20):
            q.append({"event_id": f"evt-{i}", "data": "x" * 10})

        # File should have been pruned
        remaining = q.load_pending()
        assert len(remaining) < 20


class TestEventQueueClear:
    """Clear the queue entirely."""

    def test_clear_removes_file(self, queue: EventQueue) -> None:
        queue.append(_make_event())
        queue.clear()
        assert not queue.file_path.exists()
        assert queue.pending_count == 0


class TestEventQueueMemoryFallback:
    """Memory fallback when file I/O fails (FRD-018)."""

    def test_falls_back_to_memory(self, queue: EventQueue, capsys) -> None:
        from unittest.mock import patch

        event = _make_event()
        with patch("builtins.open", side_effect=IOError("disk full")):
            queue.append(event)

        # Should be in memory fallback
        assert len(queue._memory_fallback) == 1
        captured = capsys.readouterr()
        assert "event_queue_write_failed" in captured.err

        # Memory fallback events should appear in load_pending
        loaded = queue.load_pending()
        assert any(e["event_id"] == event["event_id"] for e in loaded)
