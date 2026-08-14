# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the zero-argument tools/call defect — FRD-030.

A ``tools/call`` with no ``arguments`` member (arguments=None) previously
crashed ``PreventionStack.evaluate`` with a TypeError and killed the session.
"""

from __future__ import annotations

from clyro.config import WrapperConfig
from clyro.mcp.prevention import AllowDecision, PreventionStack
from clyro.mcp.session import McpSession


def _stack() -> PreventionStack:
    return PreventionStack(WrapperConfig(default_action="allow"))


def test_zero_arg_call_does_not_crash() -> None:
    # FRD-030: arguments=None must be governed, not raise.
    decision = _stack().evaluate("list_tables", None, McpSession())
    assert isinstance(decision, AllowDecision)


def test_zero_arg_then_normal_call_both_governed() -> None:
    # The failure previously killed the reader task, dropping later calls.
    stack, session = _stack(), McpSession()
    d1 = stack.evaluate("zero_arg_tool", None, session)
    d2 = stack.evaluate("t", {"a": 1}, session)
    assert isinstance(d1, AllowDecision)
    assert isinstance(d2, AllowDecision)
    assert session.step_count == 2  # both were counted, session survived


def test_empty_dict_equivalent_to_none() -> None:
    stack = _stack()
    assert isinstance(stack.evaluate("t", {}, McpSession()), AllowDecision)
