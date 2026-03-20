# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# SQLite SDK Migrations

"""
Enhanced migration system for SQLite SDK storage.

Provides improved migration management with validation, dry-run support,
and better error handling.

Usage:
    from clyro.storage.migrations import SQLiteMigrationManager

    manager = SQLiteMigrationManager(db_path)
    manager.run_migrations(conn, from_version=0, to_version=3)
"""

from clyro.storage.migrations.manager import (
    MIGRATIONS,
    SQLiteMigrationManager,
)

__all__ = [
    "SQLiteMigrationManager",
    "MIGRATIONS",
]
