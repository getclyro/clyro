# Tests for Clyro SDK Local Storage
# Implements PRD-005, PRD-006

"""Unit tests for local SQLite storage."""

import sqlite3
import tempfile
from datetime import UTC
from pathlib import Path
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig
from clyro.storage.sqlite import LocalStorage
from clyro.trace import EventType, TraceEvent, create_step_event


@pytest.fixture
def temp_storage():
    """Create a temporary storage instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ClyroConfig(
            local_storage_path=str(Path(tmpdir) / "test.db"),
            local_storage_max_mb=10,
        )
        storage = LocalStorage(config)
        yield storage
        storage.close()


@pytest.fixture
def session_id():
    """Generate a session ID for tests."""
    return uuid4()


class TestLocalStorageInitialization:
    """Tests for storage initialization."""

    def test_initialize_creates_db(self):
        """Test that initialization creates the database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ClyroConfig(local_storage_path=str(Path(tmpdir) / "test.db"))
            storage = LocalStorage(config)
            storage.initialize()

            assert storage.db_path.exists()
            storage.close()

    def test_initialize_creates_tables(self, temp_storage):
        """Test that initialization creates required tables."""
        temp_storage.initialize()

        # Store and retrieve should work
        event = TraceEvent(
            session_id=uuid4(),
            event_type=EventType.STEP,
        )
        result = temp_storage.store_event(event)
        assert result is True

    def test_initialize_idempotent(self, temp_storage):
        """Test that initialization can be called multiple times."""
        temp_storage.initialize()
        temp_storage.initialize()  # Should not fail

        assert temp_storage._initialized is True


class TestEventStorage:
    """Tests for event storage operations."""

    def test_store_event(self, temp_storage, session_id):
        """Test storing a single event."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test_step",
        )

        result = temp_storage.store_event(event)

        assert result is True

    def test_store_event_preserves_data(self, temp_storage, session_id):
        """Test that stored events preserve all data."""
        event = create_step_event(
            session_id=session_id,
            step_number=5,
            event_name="process_data",
            input_data={"key": "value"},
            output_data={"result": 42},
        )

        temp_storage.store_event(event)
        events = temp_storage.get_events_by_session(session_id)

        assert len(events) == 1
        retrieved = events[0]
        assert retrieved.event_id == event.event_id
        assert retrieved.step_number == 5
        assert retrieved.event_name == "process_data"
        assert retrieved.input_data == {"key": "value"}
        assert retrieved.output_data == {"result": 42}

    def test_store_events_batch(self, temp_storage, session_id):
        """Test storing multiple events in a batch."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]

        stored = temp_storage.store_events(events)

        assert stored == 5
        counts = temp_storage.get_event_count()
        assert counts["total"] == 5

    def test_store_events_empty_list(self, temp_storage):
        """Test storing empty event list."""
        stored = temp_storage.store_events([])
        assert stored == 0

    def test_store_duplicate_event(self, temp_storage, session_id):
        """Test that duplicate events are handled."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )

        temp_storage.store_event(event)
        temp_storage.store_event(event)  # Same event again

        counts = temp_storage.get_event_count()
        assert counts["total"] == 1  # Should be deduplicated


class TestEventRetrieval:
    """Tests for event retrieval operations."""

    def test_get_unsynced_events(self, temp_storage, session_id):
        """Test retrieving unsynced events."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        unsynced = temp_storage.get_unsynced_events(limit=10)

        assert len(unsynced) == 5

    def test_get_unsynced_events_respects_limit(self, temp_storage, session_id):
        """Test that limit is respected when getting unsynced events."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(10)
        ]
        temp_storage.store_events(events)

        unsynced = temp_storage.get_unsynced_events(limit=3)

        assert len(unsynced) == 3

    def test_get_events_by_session(self, temp_storage):
        """Test retrieving events by session ID."""
        session1 = uuid4()
        session2 = uuid4()

        # Store events for two sessions
        for i in range(3):
            temp_storage.store_event(
                create_step_event(session_id=session1, step_number=i, event_name=f"s1_{i}")
            )
        for i in range(2):
            temp_storage.store_event(
                create_step_event(session_id=session2, step_number=i, event_name=f"s2_{i}")
            )

        events1 = temp_storage.get_events_by_session(session1)
        events2 = temp_storage.get_events_by_session(session2)

        assert len(events1) == 3
        assert len(events2) == 2

    def test_get_events_ordered_by_timestamp(self, temp_storage, session_id):
        """Test that events are ordered by timestamp."""
        import time

        events = []
        for i in range(3):
            event = create_step_event(
                session_id=session_id,
                step_number=i,
                event_name=f"step{i}",
            )
            events.append(event)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        temp_storage.store_events(events)

        retrieved = temp_storage.get_events_by_session(session_id)

        for i in range(len(retrieved) - 1):
            assert retrieved[i].timestamp <= retrieved[i + 1].timestamp


class TestSyncStatus:
    """Tests for sync status operations."""

    def test_mark_events_synced(self, temp_storage, session_id):
        """Test marking events as synced."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        # Mark first 3 as synced
        event_ids = [str(e.event_id) for e in events[:3]]
        marked = temp_storage.mark_events_synced(event_ids)

        assert marked == 3

        counts = temp_storage.get_event_count()
        assert counts["synced"] == 3
        assert counts["unsynced"] == 2

    def test_mark_events_synced_empty_list(self, temp_storage):
        """Test marking empty event list as synced."""
        marked = temp_storage.mark_events_synced([])
        assert marked == 0

    def test_increment_sync_attempts(self, temp_storage, session_id):
        """Test incrementing sync attempts for failed events."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)

        temp_storage.increment_sync_attempts([str(event.event_id)])
        temp_storage.increment_sync_attempts([str(event.event_id)])

        # Events should still be unsynced
        counts = temp_storage.get_event_count()
        assert counts["unsynced"] == 1


class TestStorageManagement:
    """Tests for storage management operations."""

    def test_get_storage_size(self, temp_storage, session_id):
        """Test getting storage size."""
        # Empty storage should have some size (schema)
        initial_size = temp_storage.get_storage_size()
        assert initial_size >= 0

        # Add events
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(10)
        ]
        temp_storage.store_events(events)

        new_size = temp_storage.get_storage_size()
        assert new_size >= initial_size

    def test_get_event_count(self, temp_storage, session_id):
        """Test getting event counts."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        counts = temp_storage.get_event_count()

        assert counts["total"] == 5
        assert counts["unsynced"] == 5
        assert counts["synced"] == 0

    def test_prune_old_events(self, temp_storage, session_id):
        """Test pruning old synced events."""
        from datetime import datetime, timedelta

        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        # Mark all as synced
        event_ids = [str(e.event_id) for e in events]
        temp_storage.mark_events_synced(event_ids)

        # Manually update created_at to be 10 days ago for testing
        old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        conn = sqlite3.connect(str(temp_storage.db_path))
        conn.execute("UPDATE trace_buffer SET created_at = ?", (old_date,))
        conn.commit()
        conn.close()

        # Prune with 7 days retention (should delete events older than 7 days)
        deleted = temp_storage.prune_old_events(keep_days=7)

        assert deleted == 5
        counts = temp_storage.get_event_count()
        assert counts["total"] == 0

    def test_clear_storage(self, temp_storage, session_id):
        """Test clearing all storage."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        temp_storage.clear()

        counts = temp_storage.get_event_count()
        assert counts["total"] == 0


class TestSyncStatusInfo:
    """Tests for sync status information."""

    def test_get_sync_status(self, temp_storage, session_id):
        """Test getting sync status information."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(3)
        ]
        temp_storage.store_events(events)

        status = temp_storage.get_sync_status()

        assert "storage_path" in status
        assert "storage_size_bytes" in status
        assert "storage_size_mb" in status
        assert "max_size_mb" in status
        assert "events" in status
        assert status["events"]["total"] == 3
        assert status["sync_pending"] is True

    def test_sync_pending_false_when_all_synced(self, temp_storage, session_id):
        """Test sync_pending is False when all events are synced."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)
        temp_storage.mark_events_synced([str(event.event_id)])

        status = temp_storage.get_sync_status()

        assert status["sync_pending"] is False


class TestFailedEvents:
    """Tests for failed event handling."""

    def test_get_failed_events(self, temp_storage, session_id):
        """Test retrieving events that have failed multiple sync attempts."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        # Simulate multiple sync failures for first 2 events
        failed_ids = [str(e.event_id) for e in events[:2]]
        for _ in range(4):  # 4 attempts
            temp_storage.increment_sync_attempts(failed_ids)

        # Get failed events (min_attempts=3)
        failed = temp_storage.get_failed_events(min_attempts=3, limit=10)

        assert len(failed) == 2
        for event in failed:
            assert str(event.event_id) in failed_ids

    def test_get_failed_events_respects_limit(self, temp_storage, session_id):
        """Test that limit is respected when getting failed events."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        # Mark all as failed with 5 attempts
        all_ids = [str(e.event_id) for e in events]
        for _ in range(5):
            temp_storage.increment_sync_attempts(all_ids)

        failed = temp_storage.get_failed_events(min_attempts=3, limit=2)

        assert len(failed) == 2

    def test_get_failed_events_empty_when_none_failed(self, temp_storage, session_id):
        """Test that no events are returned when none have failed."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)

        failed = temp_storage.get_failed_events(min_attempts=3, limit=10)

        assert len(failed) == 0

    def test_remove_failed_events(self, temp_storage, session_id):
        """Test removing events that have permanently failed."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(5)
        ]
        temp_storage.store_events(events)

        # Mark first 2 as permanently failed (10+ attempts)
        failed_ids = [str(e.event_id) for e in events[:2]]
        for _ in range(12):
            temp_storage.increment_sync_attempts(failed_ids)

        # Remove events with 10+ attempts
        removed = temp_storage.remove_failed_events(min_attempts=10)

        assert removed == 2
        counts = temp_storage.get_event_count()
        assert counts["total"] == 3  # Only 3 remaining

    def test_remove_failed_events_none_to_remove(self, temp_storage, session_id):
        """Test that no events are removed when none meet threshold."""
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)
        temp_storage.increment_sync_attempts([str(event.event_id)])  # Only 1 attempt

        removed = temp_storage.remove_failed_events(min_attempts=10)

        assert removed == 0
        counts = temp_storage.get_event_count()
        assert counts["total"] == 1


class TestHealthAndIntegrity:
    """Tests for health check and integrity verification."""

    def test_check_health_healthy(self, temp_storage, session_id):
        """Test health check returns healthy for normal storage."""
        temp_storage.initialize()
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)

        from clyro.storage.sqlite import StorageHealthStatus

        status = temp_storage.check_health()

        assert status == StorageHealthStatus.HEALTHY

    def test_check_integrity_passes(self, temp_storage, session_id):
        """Test integrity check passes for valid database."""
        temp_storage.initialize()
        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)

        result = temp_storage.check_integrity()

        assert result is True
        assert temp_storage.metrics.integrity_check_passed is True

    def test_repair_recovers_storage(self, temp_storage):
        """Test repair operation recovers storage."""
        temp_storage.initialize()

        result = temp_storage.repair()

        assert result is True

    def test_record_metrics_snapshot(self, temp_storage, session_id):
        """Test recording metrics snapshot."""
        events = [
            create_step_event(session_id=session_id, step_number=i, event_name=f"step{i}")
            for i in range(3)
        ]
        temp_storage.store_events(events)

        # Should not raise
        temp_storage.record_metrics_snapshot()

        # Verify snapshot was recorded by checking database

        conn = sqlite3.connect(str(temp_storage.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM storage_metrics")
        count = cursor.fetchone()[0]
        conn.close()

        assert count >= 1


class TestMigrations:
    """Tests for schema migration functionality."""

    def test_migration_v1_to_v3(self):
        """Test migration from schema V1 to V3 (current)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_migration.db"

            # Create V1 schema manually (without priority, agent_stage columns and metrics table)
            conn = sqlite3.connect(str(db_path))
            conn.executescript("""
                CREATE TABLE trace_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    synced INTEGER DEFAULT 0,
                    sync_attempts INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_sync_attempt TEXT
                );

                CREATE TABLE config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE sync_status (
                    session_id TEXT PRIMARY KEY,
                    last_synced_event_id TEXT,
                    last_sync_at TEXT,
                    sync_status TEXT DEFAULT 'pending',
                    event_count INTEGER DEFAULT 0,
                    synced_count INTEGER DEFAULT 0
                );

                INSERT INTO config (key, value) VALUES ('schema_version', '1');
            """)
            conn.commit()
            conn.close()

            # Initialize storage - should trigger migration to latest version
            config = ClyroConfig(local_storage_path=str(db_path))
            storage = LocalStorage(config)
            storage.initialize()

            # Verify migration worked
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Check priority column exists (added in v2)
            cursor = conn.execute("PRAGMA table_info(trace_buffer)")
            columns = {row["name"] for row in cursor.fetchall()}
            assert "priority" in columns

            # Check agent_stage column exists (added in v3)
            assert "agent_stage" in columns

            # Check storage_metrics table exists (added in v2)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='storage_metrics'"
            )
            assert cursor.fetchone() is not None

            # Check schema version updated to latest
            cursor = conn.execute("SELECT value FROM config WHERE key='schema_version'")
            version = cursor.fetchone()["value"]
            from clyro.storage.sqlite import SCHEMA_VERSION

            assert version == str(SCHEMA_VERSION)

            conn.close()
            storage.close()

    def test_migration_v2_to_v3(self):
        """Test migration from schema V2 to V3 (agent_stage column)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_migration_v2.db"

            # Create V2 schema manually (with priority but without agent_stage)
            conn = sqlite3.connect(str(db_path))
            conn.executescript("""
                CREATE TABLE trace_buffer (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    synced INTEGER DEFAULT 0,
                    sync_attempts INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 2,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_sync_attempt TEXT
                );

                CREATE TABLE config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE sync_status (
                    session_id TEXT PRIMARY KEY,
                    last_synced_event_id TEXT,
                    last_sync_at TEXT,
                    sync_status TEXT DEFAULT 'pending',
                    event_count INTEGER DEFAULT 0,
                    synced_count INTEGER DEFAULT 0
                );

                CREATE TABLE storage_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_events INTEGER DEFAULT 0,
                    synced_events INTEGER DEFAULT 0,
                    unsynced_events INTEGER DEFAULT 0,
                    storage_size_bytes INTEGER DEFAULT 0,
                    sync_success_count INTEGER DEFAULT 0,
                    sync_failure_count INTEGER DEFAULT 0
                );

                INSERT INTO config (key, value) VALUES ('schema_version', '2');
            """)
            conn.commit()
            conn.close()

            # Initialize storage - should trigger migration to v3
            config = ClyroConfig(local_storage_path=str(db_path))
            storage = LocalStorage(config)
            storage.initialize()

            # Verify migration worked
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Check agent_stage column exists
            cursor = conn.execute("PRAGMA table_info(trace_buffer)")
            columns = {row["name"] for row in cursor.fetchall()}
            assert "agent_stage" in columns

            # Check agent_stage index exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_buffer_agent_stage'"
            )
            assert cursor.fetchone() is not None

            # Check schema version updated to 3
            cursor = conn.execute("SELECT value FROM config WHERE key='schema_version'")
            version = cursor.fetchone()["value"]
            assert version == "3"

            conn.close()
            storage.close()

    def test_fresh_database_creates_correct_schema(self):
        """Test that a fresh database gets the correct schema version."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fresh.db"

            config = ClyroConfig(local_storage_path=str(db_path))
            storage = LocalStorage(config)
            storage.initialize()

            # Verify schema version
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT value FROM config WHERE key='schema_version'")
            version = cursor.fetchone()[0]
            conn.close()

            from clyro.storage.sqlite import SCHEMA_VERSION

            assert version == str(SCHEMA_VERSION)

            storage.close()


class TestEventPriority:
    """Tests for event priority functionality."""

    def test_session_end_gets_high_priority(self, temp_storage, session_id):
        """Test that session_end events get HIGH priority."""
        from clyro.storage.sqlite import EventPriority

        event = TraceEvent(
            session_id=session_id,
            event_type=EventType.SESSION_END,
        )
        temp_storage.store_event(event)

        # Verify priority in database

        conn = sqlite3.connect(str(temp_storage.db_path))
        cursor = conn.execute(
            "SELECT priority FROM trace_buffer WHERE event_id = ?", (str(event.event_id),)
        )
        priority = cursor.fetchone()[0]
        conn.close()

        assert priority == EventPriority.HIGH.value

    def test_error_gets_high_priority(self, temp_storage, session_id):
        """Test that error events get HIGH priority."""
        from clyro.storage.sqlite import EventPriority

        event = TraceEvent(
            session_id=session_id,
            event_type=EventType.ERROR,
            error_type="TestError",
            error_message="Test error message",
        )
        temp_storage.store_event(event)

        # Verify priority in database

        conn = sqlite3.connect(str(temp_storage.db_path))
        cursor = conn.execute(
            "SELECT priority FROM trace_buffer WHERE event_id = ?", (str(event.event_id),)
        )
        priority = cursor.fetchone()[0]
        conn.close()

        assert priority == EventPriority.HIGH.value

    def test_step_gets_normal_priority(self, temp_storage, session_id):
        """Test that step events get NORMAL priority."""
        from clyro.storage.sqlite import EventPriority

        event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="test",
        )
        temp_storage.store_event(event)

        # Verify priority in database

        conn = sqlite3.connect(str(temp_storage.db_path))
        cursor = conn.execute(
            "SELECT priority FROM trace_buffer WHERE event_id = ?", (str(event.event_id),)
        )
        priority = cursor.fetchone()[0]
        conn.close()

        assert priority == EventPriority.NORMAL.value

    def test_prioritized_retrieval_order(self, temp_storage, session_id):
        """Test that prioritized retrieval returns high priority first."""
        # Store normal event first
        normal_event = create_step_event(
            session_id=session_id,
            step_number=1,
            event_name="normal",
        )
        temp_storage.store_event(normal_event)

        # Store high priority event second
        high_event = TraceEvent(
            session_id=session_id,
            event_type=EventType.ERROR,
            error_type="TestError",
            error_message="Test",
        )
        temp_storage.store_event(high_event)

        # Get events with prioritization
        events = temp_storage.get_unsynced_events(limit=10, prioritized=True)

        # High priority should come first
        assert events[0].event_type == EventType.ERROR
        assert events[1].event_type == EventType.STEP
