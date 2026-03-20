# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for hooks CLI error context enrichment.

Verifies that critical hook errors include the issue tracker URL
so developers can report problems.
"""

from __future__ import annotations

from clyro.constants import ISSUE_TRACKER_URL
from clyro.hooks.cli import _ISSUE_TRACKER, _error_with_context


class TestErrorWithContext:
    def test_appends_issue_tracker(self):
        result = _error_with_context("Config file not found")
        assert "Config file not found" in result
        assert _ISSUE_TRACKER in result
        assert "Report at" in result

    def test_includes_full_url(self):
        result = _error_with_context("something went wrong")
        assert ISSUE_TRACKER_URL in result

    def test_empty_message(self):
        result = _error_with_context("")
        assert _ISSUE_TRACKER in result


class TestMcpErrorsIssueTracker:
    """Verify that MCP JSON-RPC error responses include the issue tracker."""

    def test_format_error_includes_issue_tracker(self):
        from clyro.mcp.errors import format_error

        result = format_error("req-1", "policy_violation", {"rule_name": "test"})

        import json
        parsed = json.loads(result.strip())
        assert parsed["error"]["data"]["issue_tracker"] == _ISSUE_TRACKER
