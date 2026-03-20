# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Event Queue (File-Based JSONL Persistence)
# Implements FRD-018

"""
File-based JSONL persistence of unsynced trace events.

Design invariants (FRD-018):
- Append-only writes using standard file append mode.
- Atomic removal of synced events via temp-file + rename.
- Size limit enforcement by pruning oldest events.
- Cross-session recovery: pending events survive process crashes.
- Write failure falls back to in-memory list (max 1000 events).
- Corrupted lines are skipped with per-line warning.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clyro.mcp.log import get_logger

logger = get_logger(__name__)


class EventQueue:
    """
    File-based JSONL queue for unsynced trace events.

    Queue file: ``~/.clyro/mcp-wrapper/mcp-pending-{instance_id}.jsonl``

    Args:
        instance_id: Unique identifier derived from agent name
            (``sha256(agent_name)[:12]``).
        max_size_mb: Maximum queue file size in MB (FRD-018).
    """

    _MAX_MEMORY_EVENTS = 1000  # Fallback capacity when file I/O fails

    def __init__(self, instance_id: str, max_size_mb: int = 10) -> None:
        self._path = Path.home() / ".clyro" / "mcp-wrapper" / f"mcp-pending-{instance_id}.jsonl"
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._memory_fallback: list[dict[str, Any]] = []
        self._file_failed = False

    def append(self, event: dict[str, Any]) -> None:
        """Append event to queue file (FRD-018). Falls back to memory on failure."""
        event["queued_at"] = datetime.now(UTC).isoformat()
        line = json.dumps(event, default=str) + "\n"
        try:
            self._ensure_dir()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
            self._file_failed = False
            self._enforce_size_limit()
        except OSError as exc:
            if not self._file_failed:
                logger.warning("event_queue_write_failed", error=str(exc), fallback="memory")
                self._file_failed = True
            if len(self._memory_fallback) < self._MAX_MEMORY_EVENTS:
                self._memory_fallback.append(event)

    def _load_file_events(self) -> list[dict[str, Any]]:
        """Load events from queue file only (not memory fallback). Skips corrupted lines."""
        events: list[dict[str, Any]] = []
        if not self._path.exists():
            return events

        try:
            with open(self._path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("corrupted_queue_line", line_no=line_no)
        except OSError as exc:
            logger.error("pending_queue_read_failed", error=str(exc))
        return events

    def load_pending(self) -> list[dict[str, Any]]:
        """Load all pending events from queue file + memory fallback (FRD-018)."""
        events = self._load_file_events()
        # Include in-memory fallback events
        events.extend(self._memory_fallback)
        return events

    def remove_synced(self, event_ids: set[str]) -> None:
        """Remove synced events via atomic temp-file + rename (FRD-018)."""
        # Read file-only events (not memory fallback) to avoid duplication
        file_events = self._load_file_events()
        remaining_file = [e for e in file_events if e.get("event_id") not in event_ids]
        # Filter memory fallback separately
        self._memory_fallback = [
            e for e in self._memory_fallback if e.get("event_id") not in event_ids
        ]

        try:
            self._ensure_dir()
            tmp_path = self._path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for event in remaining_file:
                    f.write(json.dumps(event, default=str) + "\n")
            tmp_path.rename(self._path)  # Atomic on POSIX
        except OSError as exc:
            logger.error("pending_queue_update_failed", error=str(exc))

    @property
    def pending_count(self) -> int:
        """Count of unsynced events (line count of queue file + memory fallback)."""
        count = len(self._memory_fallback)
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    count += sum(1 for line in f if line.strip())
            except OSError:
                pass
        return count

    def clear(self) -> None:
        """Remove the queue file entirely."""
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass
        self._memory_fallback.clear()

    @property
    def file_path(self) -> Path:
        """Path to the queue file."""
        return self._path

    def _ensure_dir(self) -> None:
        """Create ~/.clyro/mcp-wrapper/ with 0o700 permissions if absent (FRD-018)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(self._path.parent), 0o700)
        except OSError:
            pass  # Best-effort

    def _enforce_size_limit(self) -> None:
        """Prune oldest events if file exceeds max size (FRD-018)."""
        try:
            if not self._path.exists():
                return
            file_size = self._path.stat().st_size
            if file_size <= self._max_size_bytes:
                return

            # HIGH-2 fix: read file events ONLY (not memory fallback) to avoid
            # duplication when memory events get written to file and remain in
            # self._memory_fallback.
            events = self._load_file_events()
            if not events:
                return

            # Remove oldest events (those at the start of the list)
            total_size = file_size
            while events and total_size > self._max_size_bytes:
                removed = events.pop(0)
                total_size -= len(json.dumps(removed, default=str)) + 1  # +1 for newline

            # Rewrite file with remaining events
            tmp_path = self._path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event, default=str) + "\n")
            tmp_path.rename(self._path)

            logger.info(
                "queue_pruned",
                max_mb=self._max_size_bytes // (1024 * 1024),
            )
        except OSError as exc:
            logger.error("queue_size_limit_failed", error=str(exc))
