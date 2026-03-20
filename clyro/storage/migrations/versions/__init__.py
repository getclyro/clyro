# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# SQLite SDK Migration Versions

"""
SQLite migration version modules.

Migration versions are defined in the migrations.manager module's
MIGRATIONS dictionary.

Each migration consists of:
- Version number (int)
- Description (str)
- Forward SQL statements (list[str])
- Rollback SQL statements (list[str]) - optional

Current migrations:
- Version 2: Add priority column and metrics table
- Version 3: Add agent_stage column for Think/Act/Observe visualization
"""
