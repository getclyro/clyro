# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Local SQLite Storage
# Implements PRD-005, PRD-006

"""
Local SQLite storage for offline trace buffering.

This module provides persistent local storage for trace events,
enabling offline operation and reliable sync with the backend.

Features:
- SQLite-based persistence with WAL mode for concurrent access
- Schema versioning and migration support
- Health checks and corruption detection
- Comprehensive metrics and observability
- Size limit enforcement with smart cleanup
- Event prioritization support for sync ordering
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from clyro.trace import TraceEvent

if TYPE_CHECKING:
    from clyro.config import ClyroConfig

logger = structlog.get_logger(__name__)

# Current schema version - increment when schema changes
SCHEMA_VERSION = 3

# Storage management constants
SIZE_LIMIT_BATCH_DELETE = 100  # Events to delete per batch when enforcing size limit
SIZE_LIMIT_THRESHOLD_PERCENT = 0.9  # Target 90% of max size (10% buffer for incoming)

# Schema definition - V3 adds agent_stage column
SCHEMA_SQL = """
-- Local trace buffer
CREATE TABLE IF NOT EXISTS trace_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    synced INTEGER DEFAULT 0,
    sync_attempts INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 2,
    agent_stage TEXT DEFAULT 'think',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_sync_attempt TEXT
);

-- Local configuration
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Sync status per session
CREATE TABLE IF NOT EXISTS sync_status (
    session_id TEXT PRIMARY KEY,
    last_synced_event_id TEXT,
    last_sync_at TEXT,
    sync_status TEXT DEFAULT 'pending',
    event_count INTEGER DEFAULT 0,
    synced_count INTEGER DEFAULT 0
);

-- Storage metrics history
CREATE TABLE IF NOT EXISTS storage_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_events INTEGER DEFAULT 0,
    synced_events INTEGER DEFAULT 0,
    unsynced_events INTEGER DEFAULT 0,
    storage_size_bytes INTEGER DEFAULT 0,
    sync_success_count INTEGER DEFAULT 0,
    sync_failure_count INTEGER DEFAULT 0
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_buffer_synced ON trace_buffer(synced);
CREATE INDEX IF NOT EXISTS idx_buffer_session ON trace_buffer(session_id);
CREATE INDEX IF NOT EXISTS idx_buffer_timestamp ON trace_buffer(timestamp);
CREATE INDEX IF NOT EXISTS idx_buffer_created ON trace_buffer(created_at);
CREATE INDEX IF NOT EXISTS idx_buffer_priority ON trace_buffer(priority, timestamp);
CREATE INDEX IF NOT EXISTS idx_buffer_sync_attempts ON trace_buffer(sync_attempts);
CREATE INDEX IF NOT EXISTS idx_buffer_agent_stage ON trace_buffer(agent_stage);
"""

# SQL query templates for dynamic operations
SQL_MARK_EVENTS_SYNCED = """
    UPDATE trace_buffer
    SET synced = 1, last_sync_attempt = ?
    WHERE event_id IN ({placeholders})
"""

SQL_INCREMENT_SYNC_ATTEMPTS = """
    UPDATE trace_buffer
    SET sync_attempts = sync_attempts + 1,
        last_sync_attempt = ?
    WHERE event_id IN ({placeholders})
"""

# Migration definitions: version -> (description, SQL statements)
MIGRATIONS: dict[int, tuple[str, list[str]]] = {
    2: (
        "Add priority column and metrics table",
        [
            "ALTER TABLE trace_buffer ADD COLUMN priority INTEGER DEFAULT 2",
            """CREATE TABLE IF NOT EXISTS storage_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_events INTEGER DEFAULT 0,
                synced_events INTEGER DEFAULT 0,
                unsynced_events INTEGER DEFAULT 0,
                storage_size_bytes INTEGER DEFAULT 0,
                sync_success_count INTEGER DEFAULT 0,
                sync_failure_count INTEGER DEFAULT 0
            )""",
            "CREATE INDEX IF NOT EXISTS idx_buffer_priority ON trace_buffer(priority, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_buffer_sync_attempts ON trace_buffer(sync_attempts)",
        ],
    ),
    3: (
        "Add agent_stage column for Think/Act/Observe visualization",
        [
            "ALTER TABLE trace_buffer ADD COLUMN agent_stage TEXT DEFAULT 'think'",
            "CREATE INDEX IF NOT EXISTS idx_buffer_agent_stage ON trace_buffer(agent_stage)",
        ],
    ),
}


class StorageHealthStatus(Enum):
    """Storage health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CORRUPTED = "corrupted"


class EventPriority(Enum):
    """Event sync priority levels."""

    HIGH = 1  # Session end events, errors - sync first
    NORMAL = 2  # Regular events
    LOW = 3  # Old buffered events, can wait


@dataclass
class StorageMetrics:
    """Metrics for storage operations."""

    total_stores: int = 0
    total_store_failures: int = 0
    total_retrievals: int = 0
    total_retrieval_failures: int = 0
    total_marks_synced: int = 0
    total_events_pruned: int = 0
    last_store_time: datetime | None = None
    last_error_time: datetime | None = None
    last_error_message: str | None = None
    average_store_latency_ms: float = 0.0
    average_retrieval_latency_ms: float = 0.0
    integrity_check_passed: bool = True
    last_integrity_check: datetime | None = None

    # Rolling window for latency calculation
    _store_latencies: list[float] = field(default_factory=list)
    _retrieval_latencies: list[float] = field(default_factory=list)
    _max_samples: int = 100

    def record_store_latency(self, latency_ms: float) -> None:
        """Record a store operation latency sample."""
        self._store_latencies.append(latency_ms)
        if len(self._store_latencies) > self._max_samples:
            self._store_latencies.pop(0)
        self.average_store_latency_ms = sum(self._store_latencies) / len(self._store_latencies)

    def record_retrieval_latency(self, latency_ms: float) -> None:
        """Record a retrieval operation latency sample."""
        self._retrieval_latencies.append(latency_ms)
        if len(self._retrieval_latencies) > self._max_samples:
            self._retrieval_latencies.pop(0)
        self.average_retrieval_latency_ms = sum(self._retrieval_latencies) / len(
            self._retrieval_latencies
        )

    def record_error(self, message: str) -> None:
        """Record an error occurrence."""
        self.last_error_time = datetime.now(UTC)
        self.last_error_message = message

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_stores": self.total_stores,
            "total_store_failures": self.total_store_failures,
            "total_retrievals": self.total_retrievals,
            "total_retrieval_failures": self.total_retrieval_failures,
            "total_marks_synced": self.total_marks_synced,
            "total_events_pruned": self.total_events_pruned,
            "last_store_time": self.last_store_time.isoformat() if self.last_store_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_error_message": self.last_error_message,
            "average_store_latency_ms": round(self.average_store_latency_ms, 2),
            "average_retrieval_latency_ms": round(self.average_retrieval_latency_ms, 2),
            "integrity_check_passed": self.integrity_check_passed,
            "last_integrity_check": (
                self.last_integrity_check.isoformat() if self.last_integrity_check else None
            ),
            "store_success_rate": self._calculate_store_success_rate(),
        }

    def _calculate_store_success_rate(self) -> float:
        """Calculate store success rate percentage."""
        total = self.total_stores + self.total_store_failures
        if total == 0:
            return 100.0
        return round((self.total_stores / total) * 100, 2)


class LocalStorage:
    """
    SQLite-based local storage for trace events.

    Provides persistent storage for traces when the backend is
    unavailable, with automatic sync when connectivity is restored.

    Features:
    - WAL mode for better concurrent access
    - Schema versioning with automatic migrations
    - Health checks and integrity verification
    - Comprehensive metrics and observability
    - Size limit enforcement with smart cleanup
    - Event prioritization for sync ordering

    Implements PRD-005, PRD-006
    """

    def __init__(self, config: ClyroConfig):
        """
        Initialize local storage.

        Args:
            config: SDK configuration
        """
        self.config = config
        self._db_path = config.get_storage_path()
        self._max_size_bytes = config.local_storage_max_mb * 1024 * 1024
        self._initialized = False
        self._lock = threading.Lock()  # Thread safety for initialization
        self._metrics = StorageMetrics()
        self._health_status = StorageHealthStatus.HEALTHY
        self._schema_version: int | None = None

    @property
    def db_path(self) -> Path:
        """Get the database file path."""
        return self._db_path

    @property
    def metrics(self) -> StorageMetrics:
        """Get storage metrics."""
        return self._metrics

    @property
    def health_status(self) -> StorageHealthStatus:
        """Get current health status."""
        return self._health_status

    def initialize(self) -> None:
        """
        Initialize the database, creating tables if needed.

        This is called automatically on first use.
        Handles schema migrations for existing databases.
        """
        if self._initialized:
            return

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Set secure file permissions (user read/write only) before creating DB
        # This prevents other users from accessing trace data
        import os
        import stat

        db_exists = self._db_path.exists()

        try:
            with self._get_connection() as conn:
                # Check if database exists and get current version
                current_version = self._get_schema_version(conn)

                if current_version == 0:
                    # Fresh database - create schema
                    conn.executescript(SCHEMA_SQL)
                    conn.execute(
                        "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                        (
                            "schema_version",
                            str(SCHEMA_VERSION),
                            datetime.now(UTC).isoformat(),
                        ),
                    )
                elif current_version < SCHEMA_VERSION:
                    # Existing database - run migrations
                    self._run_migrations(conn, current_version, SCHEMA_VERSION)

                conn.commit()

            self._schema_version = SCHEMA_VERSION
            self._initialized = True

            # Set secure permissions on database file (0600 = user read/write only)
            # This is critical for protecting sensitive trace data
            if not db_exists and self._db_path.exists():
                os.chmod(self._db_path, stat.S_IRUSR | stat.S_IWUSR)
                logger.debug(
                    "database_permissions_set",
                    path=str(self._db_path),
                    mode="0600",
                )

            logger.debug(
                "local_storage_initialized",
                path=str(self._db_path),
                schema_version=SCHEMA_VERSION,
            )

        except sqlite3.Error as e:
            self._health_status = StorageHealthStatus.UNHEALTHY
            self._metrics.record_error(f"Initialization failed: {e}")
            logger.error("local_storage_init_failed", error=str(e))
            raise

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        """
        Get the current schema version from the database.

        Args:
            conn: Database connection

        Returns:
            Schema version number (0 if fresh database)
        """
        try:
            # Check if config table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='config'"
            )
            if cursor.fetchone() is None:
                return 0

            # Get version from config
            cursor = conn.execute("SELECT value FROM config WHERE key = 'schema_version'")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

        except sqlite3.Error:
            return 0

    def _run_migrations(
        self,
        conn: sqlite3.Connection,
        from_version: int,
        to_version: int,
    ) -> None:
        """
        Run database migrations from one version to another.

        Args:
            conn: Database connection
            from_version: Starting schema version
            to_version: Target schema version
        """
        logger.info(
            "running_migrations",
            from_version=from_version,
            to_version=to_version,
        )

        for version in range(from_version + 1, to_version + 1):
            if version not in MIGRATIONS:
                continue

            description, statements = MIGRATIONS[version]
            logger.debug(
                "applying_migration",
                version=version,
                description=description,
            )

            for sql in statements:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError as e:
                    # Handle cases where column/table already exists
                    # SQLite doesn't provide error codes, but we can check the error message more safely
                    error_msg = str(e).lower()
                    if "duplicate column" in error_msg or "already exists" in error_msg:
                        logger.debug("migration_already_applied", sql=sql[:50])
                        continue
                    # Log the error for debugging before re-raising
                    logger.error("migration_failed", sql=sql[:100], error=str(e))
                    raise

        # Update schema version
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", str(to_version), datetime.now(UTC).isoformat()),
        )

        logger.info("migrations_completed", final_version=to_version)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def store_event(self, event: TraceEvent, priority: EventPriority | None = None) -> bool:
        """
        Store a trace event locally.

        Args:
            event: The trace event to store
            priority: Optional sync priority (auto-determined if not specified)

        Returns:
            True if stored successfully, False otherwise
        """
        self.initialize()
        start_time = time.monotonic()

        # Auto-determine priority based on event type
        if priority is None:
            priority = self._determine_event_priority(event)

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO trace_buffer
                    (event_id, session_id, event_type, timestamp, payload, priority, agent_stage, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.event_id),
                        str(event.session_id),
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.to_json(),
                        priority.value,
                        event.agent_stage.value,
                        datetime.now(UTC).isoformat(),
                    ),
                )

                if cursor.rowcount > 0:
                    # Update sync status for session
                    conn.execute(
                        """
                        INSERT INTO sync_status (session_id, event_count, sync_status)
                        VALUES (?, 1, 'pending')
                        ON CONFLICT(session_id) DO UPDATE SET
                            event_count = event_count + 1,
                            sync_status = CASE
                                WHEN sync_status = 'synced' THEN 'pending'
                                ELSE sync_status
                            END
                        """,
                        (str(event.session_id),),
                    )

                conn.commit()

            if cursor.rowcount > 0:
                # Record metrics
                latency_ms = (time.monotonic() - start_time) * 1000
                self._metrics.record_store_latency(latency_ms)
                self._metrics.total_stores += 1
                self._metrics.last_store_time = datetime.now(UTC)

                logger.debug(
                    "event_stored",
                    event_id=str(event.event_id),
                    session_id=str(event.session_id),
                    priority=priority.name,
                    latency_ms=round(latency_ms, 2),
                )
            self.enforce_size_limit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            self._metrics.total_store_failures += 1
            self._metrics.record_error(str(e))
            logger.error("event_store_failed", error=str(e), event_id=str(event.event_id))
            return False

    def _determine_event_priority(self, event: TraceEvent) -> EventPriority:
        """
        Determine sync priority based on event type.

        Args:
            event: The trace event

        Returns:
            Appropriate priority level
        """
        from clyro.trace import EventType

        # High priority: session end and error events
        if event.event_type in (EventType.SESSION_END, EventType.ERROR):
            return EventPriority.HIGH

        # Normal priority: everything else
        return EventPriority.NORMAL

    def store_events(self, events: list[TraceEvent]) -> int:
        """
        Store multiple trace events locally with automatic priority assignment.

        Args:
            events: List of trace events to store

        Returns:
            Number of events stored successfully
        """
        if not events:
            return 0

        self.initialize()
        start_time = time.monotonic()
        stored = 0

        try:
            with self._get_connection() as conn:
                session_counts: dict[str, int] = {}
                for event in events:
                    try:
                        priority = self._determine_event_priority(event)
                        cursor = conn.execute(
                            """
                            INSERT OR IGNORE INTO trace_buffer
                            (event_id, session_id, event_type, timestamp, payload, priority, agent_stage, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(event.event_id),
                                str(event.session_id),
                                event.event_type.value,
                                event.timestamp.isoformat(),
                                event.to_json(),
                                priority.value,
                                event.agent_stage.value,
                                datetime.now(UTC).isoformat(),
                            ),
                        )
                        if cursor.rowcount > 0:
                            stored += 1
                            session_id = str(event.session_id)
                            session_counts[session_id] = session_counts.get(session_id, 0) + 1
                    except sqlite3.IntegrityError:
                        # Event already exists, skip
                        pass

                for session_id, count in session_counts.items():
                    conn.execute(
                        """
                        INSERT INTO sync_status (session_id, event_count, sync_status)
                        VALUES (?, ?, 'pending')
                        ON CONFLICT(session_id) DO UPDATE SET
                            event_count = event_count + ?,
                            sync_status = CASE
                                WHEN sync_status = 'synced' THEN 'pending'
                                ELSE sync_status
                            END
                        """,
                        (session_id, count, count),
                    )

                conn.commit()

            # Record metrics
            latency_ms = (time.monotonic() - start_time) * 1000
            self._metrics.record_store_latency(latency_ms)
            self._metrics.total_stores += stored
            self._metrics.total_store_failures += len(events) - stored
            self._metrics.last_store_time = datetime.now(UTC)

            logger.debug(
                "events_stored",
                count=stored,
                total=len(events),
                latency_ms=round(latency_ms, 2),
            )

        except sqlite3.Error as e:
            self._metrics.total_store_failures += len(events)
            self._metrics.record_error(str(e))
            logger.error("events_store_failed", error=str(e))

        if stored > 0:
            self.enforce_size_limit()
        return stored

    def get_unsynced_events(
        self,
        limit: int = 100,
        prioritized: bool = True,
        max_attempts: int | None = None,
    ) -> list[TraceEvent]:
        """
        Get unsynced events for batch upload.

        Args:
            limit: Maximum number of events to return
            prioritized: If True, order by priority then timestamp
            max_attempts: If set, exclude events with more sync attempts

        Returns:
            List of unsynced trace events ordered for sync
        """
        self.initialize()
        start_time = time.monotonic()
        events = []

        try:
            with self._get_connection() as conn:
                # Build query based on options
                if prioritized:
                    # Order by priority (1=high, 2=normal, 3=low) then by timestamp
                    order_clause = "ORDER BY priority ASC, timestamp ASC"
                else:
                    order_clause = "ORDER BY timestamp ASC"

                # Add attempt filter if specified
                attempt_clause = ""
                params: list[Any] = []
                if max_attempts is not None:
                    attempt_clause = "AND sync_attempts < ?"
                    params.append(max_attempts)

                params.append(limit)

                query = f"""
                    SELECT payload FROM trace_buffer
                    WHERE synced = 0 {attempt_clause}
                    {order_clause}
                    LIMIT ?
                """

                cursor = conn.execute(query, params)

                for row in cursor:
                    try:
                        data = json.loads(row["payload"])
                        event = TraceEvent.from_dict(data)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("invalid_event_payload", error=str(e))

            # Record metrics
            latency_ms = (time.monotonic() - start_time) * 1000
            self._metrics.record_retrieval_latency(latency_ms)
            self._metrics.total_retrievals += 1

        except sqlite3.Error as e:
            self._metrics.total_retrieval_failures += 1
            self._metrics.record_error(str(e))
            logger.error("get_unsynced_failed", error=str(e))

        return events

    def mark_events_synced(self, event_ids: list[str | UUID]) -> int:
        """
        Mark events as synced after successful upload.

        Args:
            event_ids: List of event IDs to mark as synced

        Returns:
            Number of events marked
        """
        if not event_ids:
            return 0

        self.initialize()

        try:
            with self._get_connection() as conn:
                placeholders = ",".join("?" * len(event_ids))
                ids = [str(eid) for eid in event_ids]

                query = SQL_MARK_EVENTS_SYNCED.format(placeholders=placeholders)
                cursor = conn.execute(
                    query,
                    [datetime.now(UTC).isoformat()] + ids,
                )

                # Update session sync status
                self._update_session_sync_counts(conn, ids)

                conn.commit()
                marked = cursor.rowcount

                # Update metrics
                self._metrics.total_marks_synced += marked

                logger.debug("events_marked_synced", count=marked)
                return marked

        except sqlite3.Error as e:
            self._metrics.record_error(str(e))
            logger.error("mark_synced_failed", error=str(e))
            return 0

    def _update_session_sync_counts(
        self,
        conn: sqlite3.Connection,
        event_ids: list[str],
    ) -> None:
        """
        Update sync counts for affected sessions.

        Args:
            conn: Database connection
            event_ids: List of synced event IDs
        """
        if not event_ids:
            return

        try:
            # Get affected sessions
            placeholders = ",".join("?" * len(event_ids))
            cursor = conn.execute(
                f"""
                SELECT DISTINCT session_id FROM trace_buffer
                WHERE event_id IN ({placeholders})
                """,
                event_ids,
            )

            session_ids = [row[0] for row in cursor.fetchall()]

            # Update each session's sync count
            for session_id in session_ids:
                conn.execute(
                    """
                    UPDATE sync_status
                    SET synced_count = (
                        SELECT COUNT(*) FROM trace_buffer
                        WHERE session_id = ? AND synced = 1
                    ),
                    sync_status = CASE
                        WHEN (SELECT COUNT(*) FROM trace_buffer WHERE session_id = ? AND synced = 0) = 0
                        THEN 'synced'
                        ELSE 'pending'
                    END,
                    last_sync_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        session_id,
                        session_id,
                        datetime.now(UTC).isoformat(),
                        session_id,
                    ),
                )

        except sqlite3.Error as e:
            logger.warning("update_session_sync_counts_failed", error=str(e))

    def increment_sync_attempts(self, event_ids: list[str | UUID]) -> None:
        """
        Increment sync attempt counter for failed events.

        Args:
            event_ids: List of event IDs that failed to sync
        """
        if not event_ids:
            return

        self.initialize()

        try:
            with self._get_connection() as conn:
                placeholders = ",".join("?" * len(event_ids))
                ids = [str(eid) for eid in event_ids]

                query = SQL_INCREMENT_SYNC_ATTEMPTS.format(placeholders=placeholders)
                conn.execute(
                    query,
                    [datetime.now(UTC).isoformat()] + ids,
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.error("increment_attempts_failed", error=str(e))

    def get_events_by_session(self, session_id: str | UUID) -> list[TraceEvent]:
        """
        Get all events for a specific session.

        Args:
            session_id: Session ID to filter by

        Returns:
            List of trace events for the session
        """
        self.initialize()
        events = []

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT payload FROM trace_buffer
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    """,
                    (str(session_id),),
                )

                for row in cursor:
                    try:
                        data = json.loads(row["payload"])
                        event = TraceEvent.from_dict(data)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("invalid_event_payload", error=str(e))

        except sqlite3.Error as e:
            logger.error("get_session_events_failed", error=str(e))

        return events

    def get_session_ids(self) -> list[str]:
        """
        Get distinct session IDs present in storage.

        Returns:
            List of session IDs
        """
        self.initialize()
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT session_id
                    FROM trace_buffer
                    ORDER BY session_id ASC
                    """
                )
                return [row["session_id"] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error("get_session_ids_failed", error=str(e))
            return []

    def get_storage_size(self) -> int:
        """
        Get current storage size in bytes.

        Returns:
            Size of the database file in bytes
        """
        if self._db_path.exists():
            return self._db_path.stat().st_size
        return 0

    def get_event_count(self) -> dict[str, int]:
        """
        Get count of events by sync status.

        Returns:
            Dictionary with 'total', 'synced', 'unsynced' counts
        """
        self.initialize()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN synced = 1 THEN 1 ELSE 0 END) as synced,
                        SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as unsynced
                    FROM trace_buffer
                    """
                )
                row = cursor.fetchone()
                return {
                    "total": row["total"] or 0,
                    "synced": row["synced"] or 0,
                    "unsynced": row["unsynced"] or 0,
                }

        except sqlite3.Error as e:
            logger.error("get_event_count_failed", error=str(e))
            return {"total": 0, "synced": 0, "unsynced": 0}

    def prune_old_events(self, keep_days: int = 7) -> int:
        """
        Remove old synced events to manage storage size.

        Args:
            keep_days: Number of days to keep synced events

        Returns:
            Number of events deleted
        """
        self.initialize()
        deleted = 0

        try:
            # Delete old synced events
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM trace_buffer
                    WHERE synced = 1
                    AND datetime(created_at) < datetime('now', ?)
                    """,
                    (f"-{keep_days} days",),
                )
                deleted = cursor.rowcount
                conn.commit()

            # Vacuum in separate connection (VACUUM cannot run inside transaction)
            if deleted > 0:
                self._vacuum()
                logger.info("events_pruned", count=deleted, keep_days=keep_days)

            return deleted

        except sqlite3.Error as e:
            logger.error("prune_failed", error=str(e))
            return 0

    def enforce_size_limit(self) -> int:
        """
        Enforce storage size limit by removing oldest synced events.

        Deletion Strategy:
        - Only deletes events that have been successfully synced to backend
        - Preserves all unsynced events to prevent data loss
        - Deletes in batches of 100 to reduce lock contention
        - Stops when reaching 90% of max size (10% buffer for new events)
        - Runs VACUUM to reclaim disk space after deletion

        Returns:
            Number of events deleted
        """
        # Check if we're over the size limit
        current_size = self.get_storage_size()
        if current_size <= self._max_size_bytes:
            return 0

        self.initialize()
        total_deleted = 0

        # Target 90% of max size to provide buffer for incoming events
        # This prevents constant deletion on every write
        size_threshold = self._max_size_bytes * SIZE_LIMIT_THRESHOLD_PERCENT

        try:
            with self._get_connection() as conn:
                # Delete oldest synced events in batches until under threshold
                # Loop continues until size is acceptable or no more synced events remain
                while self.get_storage_size() > size_threshold:
                    # Delete oldest synced events in batch
                    # Uses subquery to efficiently identify candidates by age
                    cursor = conn.execute(
                        f"""
                        DELETE FROM trace_buffer
                        WHERE id IN (
                            SELECT id FROM trace_buffer
                            WHERE synced = 1
                            ORDER BY created_at ASC
                            LIMIT {SIZE_LIMIT_BATCH_DELETE}
                        )
                        """
                    )
                    deleted = cursor.rowcount

                    if deleted == 0:
                        # No more synced events to delete - stop to preserve unsynced data
                        # This prevents data loss of events that haven't been uploaded yet
                        logger.warning(
                            "size_limit_no_synced_events",
                            current_size=self.get_storage_size(),
                            max_size=self._max_size_bytes,
                        )
                        break

                    total_deleted += deleted
                    conn.commit()

            # Run VACUUM in separate connection to reclaim disk space
            # VACUUM must run outside a transaction, so we do it after commit
            if total_deleted > 0:
                self._vacuum()
                logger.info(
                    "size_limit_enforced",
                    deleted=total_deleted,
                    new_size=self.get_storage_size(),
                    max_size=self._max_size_bytes,
                )

        except sqlite3.Error as e:
            logger.error("enforce_size_limit_failed", error=str(e))

        return total_deleted

    def clear(self) -> None:
        """Clear all data from local storage."""
        self.initialize()

        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM trace_buffer")
                conn.execute("DELETE FROM sync_status")
                conn.execute("DELETE FROM storage_metrics")
                conn.commit()

            # Vacuum in separate connection to reclaim space
            self._vacuum()
            logger.info("local_storage_cleared")

        except sqlite3.Error as e:
            logger.error("clear_failed", error=str(e))

    def _vacuum(self) -> None:
        """
        Run VACUUM to reclaim disk space.

        VACUUM must run outside a transaction, so we use autocommit mode.
        """
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            conn.isolation_level = None  # Enable autocommit for VACUUM
            conn.execute("VACUUM")
            conn.close()
        except sqlite3.Error as e:
            logger.warning("vacuum_failed", error=str(e))

    def close(self) -> None:
        """Close storage and cleanup resources."""
        # SQLite connections are opened/closed per operation
        # This method is for interface compatibility
        pass

    def get_sync_status(self) -> dict[str, Any]:
        """
        Get overall sync status with comprehensive details.

        Returns:
            Dictionary with sync statistics, health info, and metrics
        """
        counts = self.get_event_count()
        storage_size = self.get_storage_size()

        return {
            "storage_path": str(self._db_path),
            "storage_size_bytes": storage_size,
            "storage_size_mb": round(storage_size / (1024 * 1024), 2),
            "max_size_mb": self.config.local_storage_max_mb,
            "usage_percent": round((storage_size / self._max_size_bytes) * 100, 1)
            if self._max_size_bytes > 0
            else 0,
            "events": counts,
            "sync_pending": counts["unsynced"] > 0,
            "health_status": self._health_status.value,
            "schema_version": self._schema_version,
            "metrics": self._metrics.to_dict(),
        }

    # -------------------------------------------------------------------------
    # Health Check Methods
    # -------------------------------------------------------------------------

    def check_health(self) -> StorageHealthStatus:
        """
        Perform comprehensive health check on the storage.

        Checks:
        - Database file accessibility
        - Table integrity
        - Index validity
        - Storage capacity

        Returns:
            Current health status
        """
        try:
            self.initialize()

            # Check 1: Database file exists and is readable
            if not self._db_path.exists():
                self._health_status = StorageHealthStatus.UNHEALTHY
                logger.error("health_check_failed", reason="database_file_missing")
                return self._health_status

            # Check 2: Can connect and query
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()

            # Check 3: Tables exist and are queryable
            if not self._check_tables_exist():
                self._health_status = StorageHealthStatus.DEGRADED
                logger.warning("health_check_degraded", reason="missing_tables")
                return self._health_status

            # Check 4: Storage capacity
            usage_percent = (self.get_storage_size() / self._max_size_bytes) * 100
            if usage_percent > 95:
                self._health_status = StorageHealthStatus.DEGRADED
                logger.warning(
                    "health_check_degraded",
                    reason="storage_nearly_full",
                    usage_percent=round(usage_percent, 1),
                )
                return self._health_status

            self._health_status = StorageHealthStatus.HEALTHY
            return self._health_status

        except sqlite3.Error as e:
            self._health_status = StorageHealthStatus.UNHEALTHY
            self._metrics.record_error(f"Health check failed: {e}")
            logger.error("health_check_failed", error=str(e))
            return self._health_status

    def check_integrity(self) -> bool:
        """
        Run SQLite integrity check on the database.

        This checks for:
        - Corrupted B-tree structures
        - Corrupted indexes
        - Missing or extra entries

        Returns:
            True if database passes integrity check
        """
        self.initialize()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                passed = result[0] == "ok"
                self._metrics.integrity_check_passed = passed
                self._metrics.last_integrity_check = datetime.now(UTC)

                if not passed:
                    self._health_status = StorageHealthStatus.CORRUPTED
                    logger.error(
                        "integrity_check_failed",
                        result=result[0],
                    )
                else:
                    logger.debug("integrity_check_passed")

                return passed

        except sqlite3.Error as e:
            self._metrics.integrity_check_passed = False
            self._metrics.last_integrity_check = datetime.now(UTC)
            self._metrics.record_error(f"Integrity check error: {e}")
            logger.error("integrity_check_error", error=str(e))
            return False

    def _check_tables_exist(self) -> bool:
        """
        Check if required tables exist.

        Returns:
            True if all required tables exist
        """
        required_tables = {"trace_buffer", "config", "sync_status"}

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}

                return required_tables.issubset(existing_tables)

        except sqlite3.Error:
            return False

    def repair(self) -> bool:
        """
        Attempt to repair storage issues.

        Performs:
        - VACUUM to reclaim space and rebuild indexes
        - Recreates missing tables
        - Resets health status

        Returns:
            True if repair was successful
        """
        try:
            self.initialize()

            # Recreate any missing tables
            with self._get_connection() as conn:
                conn.executescript(SCHEMA_SQL)
                conn.commit()

            # Run VACUUM to rebuild and optimize
            self._vacuum()

            # Re-check health
            self.check_health()

            logger.info("storage_repair_completed", health=self._health_status.value)
            return self._health_status == StorageHealthStatus.HEALTHY

        except sqlite3.Error as e:
            self._metrics.record_error(f"Repair failed: {e}")
            logger.error("storage_repair_failed", error=str(e))
            return False

    def get_failed_events(self, min_attempts: int = 3, limit: int = 100) -> list[TraceEvent]:
        """
        Get events that have failed to sync multiple times.

        Useful for identifying problematic events that may need
        manual intervention or removal.

        Args:
            min_attempts: Minimum sync attempts to consider as failed
            limit: Maximum number of events to return

        Returns:
            List of events with high failure counts
        """
        self.initialize()
        events = []

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT payload, sync_attempts FROM trace_buffer
                    WHERE synced = 0 AND sync_attempts >= ?
                    ORDER BY sync_attempts DESC, created_at ASC
                    LIMIT ?
                    """,
                    (min_attempts, limit),
                )

                for row in cursor:
                    try:
                        data = json.loads(row["payload"])
                        event = TraceEvent.from_dict(data)
                        events.append(event)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("invalid_event_payload", error=str(e))

        except sqlite3.Error as e:
            logger.error("get_failed_events_error", error=str(e))

        return events

    def remove_failed_events(self, min_attempts: int = 10) -> int:
        """
        Remove events that have permanently failed to sync.

        Events with too many failed sync attempts are considered
        permanently failed and are removed to prevent infinite retry.

        Args:
            min_attempts: Minimum sync attempts before removal

        Returns:
            Number of events removed
        """
        self.initialize()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM trace_buffer
                    WHERE synced = 0 AND sync_attempts >= ?
                    """,
                    (min_attempts,),
                )
                deleted = cursor.rowcount
                conn.commit()

            if deleted > 0:
                logger.warning(
                    "failed_events_removed",
                    count=deleted,
                    min_attempts=min_attempts,
                )

            return deleted

        except sqlite3.Error as e:
            self._metrics.record_error(str(e))
            logger.error("remove_failed_events_error", error=str(e))
            return 0

    def record_metrics_snapshot(self) -> None:
        """
        Record a metrics snapshot in the storage_metrics table.

        This can be called periodically to build a history of
        storage performance for analysis.
        """
        self.initialize()

        try:
            counts = self.get_event_count()
            storage_size = self.get_storage_size()

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO storage_metrics
                    (timestamp, total_events, synced_events, unsynced_events,
                     storage_size_bytes, sync_success_count, sync_failure_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(UTC).isoformat(),
                        counts["total"],
                        counts["synced"],
                        counts["unsynced"],
                        storage_size,
                        self._metrics.total_marks_synced,
                        self._metrics.total_store_failures,
                    ),
                )
                conn.commit()

                # Cleanup old metrics (keep last 7 days)
                conn.execute(
                    """
                    DELETE FROM storage_metrics
                    WHERE datetime(timestamp) < datetime('now', '-7 days')
                    """
                )
                conn.commit()

        except sqlite3.Error as e:
            logger.warning("record_metrics_snapshot_failed", error=str(e))
