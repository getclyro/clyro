# SQLite SDK Migration Tests

"""
Unit tests for SQLite SDK migration manager.

Tests cover:
- Migration execution
- Validation
- Dry-run mode
- Rollback support
- Error handling
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from clyro.storage.migrations.manager import (
    MIGRATIONS,
    SQLiteMigrationManager,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def db_connection(temp_db):
    """Create a database connection with config table."""
    conn = sqlite3.connect(temp_db)

    # Create config table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create trace_buffer table (required for migrations)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trace_buffer (
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
        )
    """)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def migration_manager(temp_db):
    """Create a migration manager."""
    return SQLiteMigrationManager(temp_db)


class TestSQLiteMigrationManager:
    """Test SQLiteMigrationManager class."""

    def test_get_schema_version_no_config_table(self, temp_db):
        """Test get_schema_version when config table doesn't exist."""
        manager = SQLiteMigrationManager(temp_db)
        conn = sqlite3.connect(temp_db)

        version = manager.get_schema_version(conn)
        assert version == 0

        conn.close()

    def test_get_schema_version_no_version_key(self, db_connection, migration_manager):
        """Test get_schema_version when schema_version key doesn't exist."""
        version = migration_manager.get_schema_version(db_connection)
        assert version == 0

    def test_get_schema_version_with_version(self, db_connection, migration_manager):
        """Test get_schema_version when version exists."""
        db_connection.execute(
            "INSERT INTO config (key, value) VALUES ('schema_version', '2')"
        )
        db_connection.commit()

        version = migration_manager.get_schema_version(db_connection)
        assert version == 2

    def test_run_migrations_dry_run(self, db_connection, migration_manager):
        """Test dry-run mode returns SQL without executing."""
        statements = migration_manager.run_migrations(
            db_connection, from_version=0, to_version=2, dry_run=True
        )

        # Should return statements
        assert len(statements) > 0
        assert any("ALTER TABLE" in stmt for stmt in statements)

        # Should not have updated schema version
        cursor = db_connection.execute(
            "SELECT value FROM config WHERE key = 'schema_version'"
        )
        assert cursor.fetchone() is None

    def test_run_migrations_applies_changes(self, db_connection, migration_manager):
        """Test that migrations are actually applied."""
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=2, dry_run=False
        )

        # Check that priority column was added
        cursor = db_connection.execute("PRAGMA table_info(trace_buffer)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "priority" in columns

        # Check that storage_metrics table was created
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='storage_metrics'"
        )
        assert cursor.fetchone() is not None

        # Check schema version was updated
        version = migration_manager.get_schema_version(db_connection)
        assert version == 2

    def test_run_migrations_idempotency(self, db_connection, migration_manager):
        """Test that running migrations multiple times is safe."""
        # First run
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=2
        )

        # Second run should not fail
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=2
        )

        # Schema should still be version 2
        version = migration_manager.get_schema_version(db_connection)
        assert version == 2

    def test_run_migrations_incremental(self, db_connection, migration_manager):
        """Test running migrations incrementally."""
        # Run to version 2
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=2
        )
        assert migration_manager.get_schema_version(db_connection) == 2

        # Run to version 3
        migration_manager.run_migrations(
            db_connection, from_version=2, to_version=3
        )
        assert migration_manager.get_schema_version(db_connection) == 3

        # Check that agent_stage column was added
        cursor = db_connection.execute("PRAGMA table_info(trace_buffer)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "agent_stage" in columns

    def test_rollback(self, db_connection, migration_manager):
        """Test rollback functionality."""
        # Apply migrations
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=3
        )

        # Rollback to version 2
        migration_manager.rollback(db_connection, from_version=3, to_version=2)

        # Check version was updated
        version = migration_manager.get_schema_version(db_connection)
        assert version == 2

        # Check that agent_stage index was dropped
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND name='idx_buffer_agent_stage'"
        )
        assert cursor.fetchone() is None

    def test_rollback_invalid_target(self, db_connection, migration_manager):
        """Test that rollback raises error for invalid target."""
        with pytest.raises(ValueError, match="must be less than"):
            migration_manager.rollback(db_connection, from_version=2, to_version=3)

    def test_validate_migrations(self, migration_manager):
        """Test migration validation."""
        errors = migration_manager.validate_migrations()

        # MIGRATIONS dict should be valid
        assert len(errors) == 0

    def test_validate_migrations_detects_gap(self, migration_manager):
        """Test validation detects gaps in version sequence."""
        # Temporarily modify MIGRATIONS to create a gap
        original_migrations = MIGRATIONS.copy()
        MIGRATIONS.clear()
        MIGRATIONS[1] = ("First", ["SELECT 1"], [])
        MIGRATIONS[3] = ("Third", ["SELECT 3"], [])  # Gap at version 2

        errors = migration_manager.validate_migrations()
        assert len(errors) > 0
        assert any("Gap" in error for error in errors)

        # Restore original
        MIGRATIONS.clear()
        MIGRATIONS.update(original_migrations)

    def test_get_migration_info(self, migration_manager):
        """Test getting migration information."""
        info = migration_manager.get_migration_info(2)

        # Should return info about version 2 - priority column feature
        assert info["version"] == 2
        assert "priority" in info["name"].lower()
        assert info["forward_statements"] > 0
        assert info["rollback_statements"] > 0
        assert info["reversible"] is True

    def test_get_migration_info_nonexistent(self, migration_manager):
        """Test getting info for nonexistent migration."""
        info = migration_manager.get_migration_info(999)
        assert info is None

    def test_get_all_versions(self, migration_manager):
        """Test getting all migration versions."""
        versions = migration_manager.get_all_versions()

        assert isinstance(versions, list)
        assert len(versions) > 0
        assert versions == sorted(versions)  # Should be sorted
        assert 2 in versions
        assert 3 in versions

    def test_preview_migration(self, db_connection, migration_manager):
        """Test previewing migration SQL."""
        preview = migration_manager.preview_migration(
            db_connection, from_version=0, to_version=2
        )

        assert isinstance(preview, str)
        assert "ALTER TABLE" in preview
        assert "CREATE TABLE" in preview
        assert "storage_metrics" in preview


class TestMigrationDefinitions:
    """Test the MIGRATIONS dictionary itself."""

    def test_migrations_exist(self):
        """Test that migrations are defined."""
        assert len(MIGRATIONS) > 0

    def test_migration_structure(self):
        """Test that each migration has correct structure."""
        for version, (name, forward, rollback) in MIGRATIONS.items():
            assert isinstance(version, int)
            assert version > 0
            assert isinstance(name, str)
            assert len(name) > 0
            assert isinstance(forward, list)
            assert len(forward) > 0
            assert isinstance(rollback, list)
            # Rollback can be empty for SQLite limitations

    def test_version_2_migration(self):
        """Test version 2 migration content."""
        name, forward, rollback = MIGRATIONS[2]

        assert "priority" in name.lower()
        assert any("ALTER TABLE" in stmt for stmt in forward)
        assert any("storage_metrics" in stmt for stmt in forward)
        assert len(rollback) > 0  # Version 2 has rollback statements

    def test_version_3_migration(self):
        """Test version 3 migration content."""
        name, forward, rollback = MIGRATIONS[3]

        assert "agent_stage" in name.lower()
        assert any("ALTER TABLE" in stmt for stmt in forward)
        assert any("agent_stage" in stmt for stmt in forward)
        assert len(rollback) > 0  # Version 3 has rollback statements


class TestMigrationErrorHandling:
    """Test error handling in migrations."""

    def test_invalid_sql_raises_error(self, db_connection, migration_manager):
        """Test that invalid SQL raises error."""
        # Temporarily add invalid migration
        original_migrations = MIGRATIONS.copy()
        MIGRATIONS[99] = ("Invalid", ["INVALID SQL SYNTAX"], [])

        with pytest.raises(sqlite3.Error):
            migration_manager.run_migrations(
                db_connection, from_version=0, to_version=99
            )

        # Restore original
        MIGRATIONS.clear()
        MIGRATIONS.update(original_migrations)

    def test_error_doesnt_update_version(self, db_connection, migration_manager):
        """Test that failed migration doesn't update version."""
        # Set initial version
        db_connection.execute(
            "INSERT INTO config (key, value) VALUES ('schema_version', '0')"
        )
        db_connection.commit()

        # Temporarily add failing migration
        original_migrations = MIGRATIONS.copy()
        MIGRATIONS[1] = ("Failing", ["INVALID SQL"], [])

        try:
            migration_manager.run_migrations(
                db_connection, from_version=0, to_version=1
            )
        except sqlite3.Error:
            pass

        # Version should still be 0
        version = migration_manager.get_schema_version(db_connection)
        assert version == 0

        # Restore original
        MIGRATIONS.clear()
        MIGRATIONS.update(original_migrations)


class TestMigrationIntegration:
    """Integration tests for full migration workflow."""

    def test_fresh_database_migration(self, db_connection, migration_manager):
        """Test migrating a fresh database from 0 to latest."""
        # Run all migrations
        migration_manager.run_migrations(
            db_connection,
            from_version=0,
            to_version=migration_manager.current_version,
        )

        # Check final version
        version = migration_manager.get_schema_version(db_connection)
        assert version == migration_manager.current_version

        # Verify all expected columns exist
        cursor = db_connection.execute("PRAGMA table_info(trace_buffer)")
        columns = [row[1] for row in cursor.fetchall()]

        assert "priority" in columns
        assert "agent_stage" in columns

        # Verify storage_metrics table exists
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='storage_metrics'"
        )
        assert cursor.fetchone() is not None

    def test_migration_with_data(self, db_connection, migration_manager):
        """Test migrating database that contains data."""
        # Insert test data
        db_connection.execute("""
            INSERT INTO trace_buffer (event_id, session_id, event_type, timestamp, payload)
            VALUES ('test-1', 'session-1', 'action', '2024-01-01T00:00:00', '{}')
        """)
        db_connection.commit()

        # Run migrations
        migration_manager.run_migrations(
            db_connection, from_version=0, to_version=3
        )

        # Data should still exist
        cursor = db_connection.execute(
            "SELECT event_id FROM trace_buffer WHERE event_id = 'test-1'"
        )
        assert cursor.fetchone() is not None

        # New columns should have default values
        cursor = db_connection.execute(
            "SELECT priority, agent_stage FROM trace_buffer WHERE event_id = 'test-1'"
        )
        row = cursor.fetchone()
        assert row[0] == 2  # Default priority
        assert row[1] == "think"  # Default agent_stage
