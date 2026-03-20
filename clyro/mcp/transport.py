# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Stdio Transport
# Implements FRD-002, FRD-013

"""
Child process lifecycle management and I/O piping.

Spawns the MCP server as a child process using
``asyncio.create_subprocess_exec()`` with stdin/stdout/stderr pipes,
monitors the child for exit, and provides helpers for graceful shutdown.
"""

from __future__ import annotations

import asyncio
import sys

from clyro.mcp.log import get_logger

logger = get_logger(__name__)


class StdioTransport:
    """
    Manage the MCP server child process.

    Args:
        command: Server command and arguments (e.g. ``["npx", "server-fs", "/path"]``).
    """

    def __init__(self, command: list[str]) -> None:
        self._command = command
        self._process: asyncio.subprocess.Process | None = None

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    async def start(self) -> asyncio.subprocess.Process:
        """
        Spawn the child process.

        Returns:
            The ``asyncio.subprocess.Process`` handle.

        Raises:
            SystemExit: If the command is not found (exit 1 — FRD-013).
        """
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as exc:
            logger.error("server_start_failed", command=self._command[0], error=str(exc))
            sys.exit(1)
        return self._process

    async def terminate(self, timeout: float = 5.0) -> int:
        """
        Gracefully terminate the child process.

        Sends SIGTERM, waits *timeout* seconds, then SIGKILL if needed.

        Returns:
            Child exit code.
        """
        if self._process is None:
            return 0

        try:
            self._process.terminate()
        except ProcessLookupError:
            # Already exited
            return self._process.returncode or 0

        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout)
        except TimeoutError:
            # Force kill
            try:
                self._process.kill()
                await self._process.wait()
            except ProcessLookupError:
                pass

        return self._process.returncode or 0

    async def write_to_child(self, data: bytes) -> None:
        """Write raw bytes to the child's stdin.

        Raises:
            BrokenPipeError: If the child's stdin is unavailable (process died).
        """
        if not self._process or not self._process.stdin:
            logger.warning("child_stdin_unavailable", reason="server process may have exited")
            raise BrokenPipeError("Child stdin unavailable")
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def read_line_from_child(self) -> bytes | None:
        """Read one newline-delimited line from the child's stdout."""
        if self._process and self._process.stdout:
            line = await self._process.stdout.readline()
            return line if line else None
        return None

    async def read_stderr_line(self) -> bytes | None:
        """Read one line from the child's stderr."""
        if self._process and self._process.stderr:
            line = await self._process.stderr.readline()
            return line if line else None
        return None
