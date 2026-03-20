# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# SQLite SDK Migration Manager

"""
Enhanced SQLite migration manager for SDK.

Builds on the existing version-based system with improved:
- Migration validation
- Rollback support (where possible)
- Better error handling
- Dry-run mode for testing
- Migration testing utilities
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import structlog

    logger = structlog.get_logger(__name__)
else:
    try:
        import structlog

        logger = structlog.get_logger(__name__)
    except ImportError:
        # Fallback to print if structlog not available
        class SimpleLogger:
            def info(self, msg, **kwargs):
                print(f"INFO: {msg}", kwargs)

            def debug(self, msg, **kwargs):
                pass  # Suppress debug in simple logger

            def warning(self, msg, **kwargs):
                print(f"WARNING: {msg}", kwargs)

            def error(self, msg, **kwargs):
                print(f"ERROR: {msg}", kwargs)

        logger = SimpleLogger()

# Migration registry: version -> (name, forward_statements, rollback_statements)
# Rollback statements may be empty for migrations that can't be reversed in SQLite
MIGRATIONS: dict[int, tuple[str, list[str], list[str]]] = {
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
        [  # Rollback statements
            "DROP INDEX IF EXISTS idx_buffer_sync_attempts",
            "DROP INDEX IF EXISTS idx_buffer_priority",
            "DROP TABLE IF EXISTS storage_metrics",
            # Note: SQLite doesn't support DROP COLUMN easily, so we can't remove priority column
        ],
    ),
    3: (
        "Add agent_stage column for Think/Act/Observe visualization",
        [
            "ALTER TABLE trace_buffer ADD COLUMN agent_stage TEXT DEFAULT 'think'",
            "CREATE INDEX IF NOT EXISTS idx_buffer_agent_stage ON trace_buffer(agent_stage)",
        ],
        [  # Rollback
            "DROP INDEX IF EXISTS idx_buffer_agent_stage",
            # Note: Can't drop agent_stage column in SQLite
        ],
    ),
}


class SQLiteMigrationManager:
    """Manages SQLite migrations with enhanced features."""

    def __init__(self, db_path: Path):
        """
        Initialize migration manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.current_version = 3  # Track latest version

    def get_schema_version(self, conn: sqlite3.Connection) -> int:
        """
        Get current schema version from database.

        Args:
            conn: Database connection

        Returns:
            Current schema version, or 0 if not found
        """
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='config'"
            )
            if cursor.fetchone() is None:
                return 0

            cursor = conn.execute("SELECT value FROM config WHERE key = 'schema_version'")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return 0

    def run_migrations(
        self,
        conn: sqlite3.Connection,
        from_version: int,
        to_version: int,
        dry_run: bool = False,
    ) -> list[str]:
        """
        Run migrations with validation and reporting.

        Args:
            conn: Database connection
            from_version: Starting version
            to_version: Target version
            dry_run: If True, only return SQL without executing

        Returns:
            List of executed SQL statements

        Raises:
            sqlite3.Error: If migration fails
        """
        executed_statements = []

        logger.info(
            "running_migrations",
            from_version=from_version,
            to_version=to_version,
            dry_run=dry_run,
        )

        for version in range(from_version + 1, to_version + 1):
            if version not in MIGRATIONS:
                continue

            name, forward_stmts, _ = MIGRATIONS[version]

            logger.debug("applying_migration", version=version, description=name)

            for sql in forward_stmts:
                if dry_run:
                    executed_statements.append(sql)
                else:
                    try:
                        conn.execute(sql)
                        executed_statements.append(sql)
                    except sqlite3.OperationalError as e:
                        error_msg = str(e).lower()
                        # Allow idempotent errors (column/table/index already exists)
                        if any(
                            phrase in error_msg
                            for phrase in [
                                "duplicate column",
                                "already exists",
                                "table already exists",
                            ]
                        ):
                            logger.debug("migration_already_applied", sql=sql[:50], error=str(e))
                            continue
                        # Re-raise other errors
                        logger.error(
                            "migration_failed", version=version, sql=sql[:100], error=str(e)
                        )
                        raise

        if not dry_run:
            # Update schema version
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                ("schema_version", str(to_version), datetime.now(UTC).isoformat()),
            )

        logger.info(
            "migrations_completed",
            final_version=to_version,
            statements_executed=len(executed_statements),
        )

        return executed_statements

    def rollback(self, conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
        """
        Rollback migrations (where supported).

        Note: SQLite has limited ALTER TABLE support, so some migrations
        may not be fully reversible (e.g., DROP COLUMN not supported).

        Args:
            conn: Database connection
            from_version: Current version
            to_version: Target version (must be < from_version)

        Raises:
            ValueError: If to_version >= from_version
            sqlite3.Error: If rollback fails
        """
        if to_version >= from_version:
            raise ValueError(
                f"Rollback target version ({to_version}) must be less than "
                f"current version ({from_version})"
            )

        logger.warning("rolling_back_migrations", from_version=from_version, to_version=to_version)

        for version in range(from_version, to_version, -1):
            if version not in MIGRATIONS:
                continue

            name, _, rollback_stmts = MIGRATIONS[version]

            logger.debug("rolling_back_migration", version=version, description=name)

            for sql in rollback_stmts:
                try:
                    conn.execute(sql)
                except sqlite3.Error as e:
                    logger.warning("rollback_failed", version=version, sql=sql[:100], error=str(e))
                    # Continue with other rollback statements

        # Update version
        conn.execute(
            "UPDATE config SET value = ?, updated_at = ? WHERE key = 'schema_version'",
            (str(to_version), datetime.now(UTC).isoformat()),
        )

        logger.info("rollback_complete", final_version=to_version)

    def validate_migrations(self) -> list[str]:
        """
        Validate migration definitions.

        Checks for:
        - Version continuity (no gaps)
        - Missing descriptions
        - Empty forward statements

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check version continuity
        versions = sorted(MIGRATIONS.keys())
        if not versions:
            return ["No migrations defined"]

        for i, version in enumerate(versions):
            if i > 0 and version != versions[i - 1] + 1:
                errors.append(f"Gap in migration versions: {versions[i - 1]} -> {version}")

        # Check migration structure
        for version, (name, forward, _rollback) in MIGRATIONS.items():
            if not name:
                errors.append(f"Migration {version} missing description")
            if not forward:
                errors.append(f"Migration {version} has no forward statements")
            # Note: rollback statements are optional

        return errors

    def get_migration_info(self, version: int) -> dict[str, Any] | None:
        """
        Get information about a specific migration.

        Args:
            version: Migration version number

        Returns:
            Dictionary with migration details, or None if not found
        """
        if version not in MIGRATIONS:
            return None

        name, forward, rollback = MIGRATIONS[version]
        return {
            "version": version,
            "name": name,
            "forward_statements": len(forward),
            "rollback_statements": len(rollback),
            "reversible": len(rollback) > 0,
        }

    def get_all_versions(self) -> list[int]:
        """
        Get all available migration versions.

        Returns:
            Sorted list of migration version numbers
        """
        return sorted(MIGRATIONS.keys())

    def preview_migration(
        self, conn: sqlite3.Connection, from_version: int, to_version: int
    ) -> str:
        """
        Preview SQL that would be executed for a migration.

        Args:
            conn: Database connection (not used, but kept for consistency)
            from_version: Starting version
            to_version: Target version

        Returns:
            Multi-line string with all SQL statements
        """
        statements = self.run_migrations(conn, from_version, to_version, dry_run=True)
        return "\n\n".join(f"-- Statement {i + 1}\n{stmt}" for i, stmt in enumerate(statements))


# Keep reference to MIGRATIONS for backward compatibility
__all__ = ["SQLiteMigrationManager", "MIGRATIONS"]
