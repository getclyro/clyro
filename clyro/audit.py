# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Base Audit Logger
# Shared by hooks (clyro.hooks.audit) and MCP wrapper (clyro.mcp.audit)

"""
Base class for append-only JSONL audit logging with fail-open semantics.

Design invariants (inherited by all subclasses):
- Audit write failure NEVER blocks tool call forwarding (fail-open).
- Audit log directory is created on first write if absent.
- Audit file permissions are ``0o600`` (owner read/write only — NFR-005).
- Subclasses implement domain-specific ``log_*`` methods; this base
  provides only the I/O layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default file/directory permissions for audit logs (NFR-005).
_DIR_PERMISSIONS = 0o700
_FILE_PERMISSIONS = 0o600


class BaseAuditLogger:
    """Append-only JSONL audit logger — base class.

    Provides:
    - Lazy file open with restricted permissions.
    - Single-line JSON append (``_write``).
    - Permission hardening on existing files.
    - Fail-open: I/O errors are logged, never raised.
    - ``close()`` for resource cleanup.

    Subclasses add domain-specific event methods
    (``log_pre_tool_use``, ``log_tool_call``, etc.).
    """

    def __init__(
        self,
        log_path: str | Path,
        *,
        dir_permissions: int = _DIR_PERMISSIONS,
        file_permissions: int = _FILE_PERMISSIONS,
    ) -> None:
        self._log_path = Path(str(log_path)).expanduser()
        self._dir_permissions = dir_permissions
        self._file_permissions = file_permissions
        self._fd: Any = None
        self._write_failed = False

    # ------------------------------------------------------------------
    # File I/O (shared by all subclasses)
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        """Lazily open (and create) the audit log file with restricted permissions.

        Creates parent directories if absent. Tightens permissions on
        existing files that are more permissive than expected.
        """
        if self._fd is not None:
            return

        # Create parent directory
        parent = self._log_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(str(parent), self._dir_permissions)
            except OSError:
                pass

        # Harden permissions on existing files (NFR-005)
        if self._log_path.exists():
            try:
                mode = os.stat(str(self._log_path)).st_mode & 0o777
                if mode != self._file_permissions:
                    logger.warning(
                        "audit_log_permissions",
                        path=str(self._log_path),
                        current=oct(mode),
                        expected=oct(self._file_permissions),
                        action="tightening",
                    )
                    os.chmod(str(self._log_path), self._file_permissions)
            except OSError:
                pass  # Best-effort; don't block on permission check

        # Open with restricted permissions
        fd = os.open(
            str(self._log_path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            self._file_permissions,
        )
        self._fd = os.fdopen(fd, "a")

    def _write(self, entry: dict[str, Any]) -> None:
        """Append one JSONL line. Fail-open: errors go to stderr only."""
        try:
            self._ensure_open()
            line = json.dumps(entry, default=str) + "\n"
            self._fd.write(line)
            self._fd.flush()
            self._write_failed = False
        except OSError as exc:
            if not self._write_failed:
                logger.error("audit_write_error", error=str(exc))
            self._write_failed = True

    def close(self) -> None:
        """Flush and close the file handle."""
        if self._fd is not None:
            try:
                self._fd.close()
            except OSError:
                pass
            self._fd = None
