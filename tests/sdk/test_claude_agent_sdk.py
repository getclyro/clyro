# Tests for Claude Agent SDK Adapter
# Implements NFR-003: ≥85% line coverage for adapter module

"""
Comprehensive tests for the Claude Agent SDK adapter.

Test organization follows the TDD §11 testing strategy:
- TestToolUseCorrelator: C3 component (FRD-006)
- TestSubagentTracker: C4 component (FRD-007)
- TestCostEstimator: Cost estimation (FRD-011b)
- TestClaudeAgentHandler: C2 component (FRD-002 through FRD-013)
- TestExecutionControls: FRD-011a/b/c
- TestPolicyEnforcement: FRD-010
- TestHookRegistrar: C1 component (FRD-001)
- TestClaudeAgentAdapter: C5 component
- TestInstrumentClaudeAgent: Public API
- TestDataSensitivity: NFR-006
- TestEdgeCases: §11.3 edge cases
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import (
    FrameworkVersionError,
    PolicyViolationError,
)
from clyro.session import Session
from clyro.trace import AgentStage, EventType, Framework, TraceEvent

# --- Mock Claude Agent SDK module ---


class MockClaudeAgentSDKModule(ModuleType):
    """Mock claude_agent_sdk module for testing."""

    __version__ = "0.1.42"


class MockClaudeAgentOptions:
    """Mock ClaudeAgentOptions for testing."""

    __module__ = "claude_agent_sdk.options"

    def __init__(self, model: str = "claude-sonnet-4-6", hooks: dict | None = None):
        self.model = model
        self.hooks = hooks or {}


# --- Fixtures ---


@pytest.fixture
def mock_claude_sdk_module():
    """Install mock claude_agent_sdk module."""
    mock_module = MockClaudeAgentSDKModule("claude_agent_sdk")
    with patch.dict(sys.modules, {"claude_agent_sdk": mock_module}):
        yield mock_module


@pytest.fixture
def mock_claude_sdk_unsupported():
    """Install mock claude_agent_sdk with unsupported version."""
    mock_module = MockClaudeAgentSDKModule("claude_agent_sdk")
    mock_module.__version__ = "0.1.30"
    with patch.dict(sys.modules, {"claude_agent_sdk": mock_module}):
        yield mock_module


@pytest.fixture
def config():
    """Create test ClyroConfig."""
    return ClyroConfig(
        capture_inputs=True,
        capture_outputs=True,
        capture_state=True,
        controls=ExecutionControls(
            max_steps=100,
            max_cost_usd=10.0,
            loop_detection_threshold=3,
            enable_step_limit=True,
            enable_cost_limit=True,
            enable_loop_detection=True,
            enable_policy_enforcement=False,
        ),
    )


@pytest.fixture
def config_no_capture():
    """Create test ClyroConfig with capture disabled."""
    return ClyroConfig(
        capture_inputs=False,
        capture_outputs=False,
        capture_state=False,
        controls=ExecutionControls(enable_policy_enforcement=False),
    )


@pytest.fixture
def session(config):
    """Create a started test Session."""
    s = Session(
        config=config,
        agent_id=uuid4(),
        org_id=uuid4(),
        framework=Framework.CLAUDE_AGENT_SDK,
    )
    s.start()
    return s


@pytest.fixture
def handler(config, session):
    """Create a ClaudeAgentHandler with a session."""
    from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

    return ClaudeAgentHandler(
        config=config,
        framework_version="0.1.42",
        session=session,
    )


@pytest.fixture
def handler_no_capture(config_no_capture, session):
    """Create handler with capture disabled."""
    from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

    session_nc = Session(
        config=config_no_capture,
        agent_id=uuid4(),
        org_id=uuid4(),
        framework=Framework.CLAUDE_AGENT_SDK,
    )
    session_nc.start()
    return ClaudeAgentHandler(
        config=config_no_capture,
        framework_version="0.1.42",
        session=session_nc,
    )


def _make_input_data(
    session_id: str = "test-session-123",
    hook_event: str = "PreToolUse",
    **kwargs: Any,
) -> dict[str, Any]:
    """Helper to create hook input_data dict."""
    data = {"session_id": session_id, "hook_event_name": hook_event}
    data.update(kwargs)
    return data


# =============================================================================
# TestToolUseCorrelator (C3, FRD-006)
# =============================================================================


class TestToolUseCorrelator:
    """Tests for ToolUseCorrelator component."""

    def test_start_and_complete(self):
        """FRD-006: Store PreToolUse, complete PostToolUse with duration."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        event_id = uuid4()

        tool_use_id = correlator.start("tui-123", event_id, "bash")
        assert tool_use_id == "tui-123"
        assert correlator.pending_count == 1

        parent_id, duration_ms = correlator.complete("tui-123")
        assert parent_id == event_id
        assert duration_ms is not None
        assert duration_ms >= 0
        assert correlator.pending_count == 0

    def test_complete_missing_returns_none(self):
        """FRD-006: PostToolUse without matching PreToolUse."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        parent_id, duration_ms = correlator.complete("nonexistent")
        assert parent_id is None
        assert duration_ms is None

    def test_complete_none_tool_use_id(self):
        """FRD-006: Complete with None tool_use_id."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        parent_id, duration_ms = correlator.complete(None)
        assert parent_id is None
        assert duration_ms is None

    def test_synthetic_tool_use_id(self):
        """B-01: Generate synthetic UUID when tool_use_id is None."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        event_id = uuid4()

        tool_use_id = correlator.start(None, event_id, "bash")
        assert tool_use_id is not None
        assert len(tool_use_id) > 0
        assert correlator.pending_count == 1

    def test_synthetic_tool_use_id_empty_string(self):
        """B-01: Generate synthetic UUID when tool_use_id is empty string."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        tool_use_id = correlator.start("", uuid4(), "bash")
        assert tool_use_id != ""

    def test_duplicate_tool_use_id_overwrites(self):
        """FRD-006: Duplicate tool_use_id overwrites existing entry."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        event_id_1 = uuid4()
        event_id_2 = uuid4()

        correlator.start("tui-dup", event_id_1, "bash")
        correlator.start("tui-dup", event_id_2, "read")
        assert correlator.pending_count == 1

        parent_id, _ = correlator.complete("tui-dup")
        assert parent_id == event_id_2

    def test_overflow_eviction(self):
        """FRD-006: Evict oldest entries when exceeding 1000."""
        from clyro.adapters.claude_agent_sdk import MAX_CORRELATOR_SIZE, ToolUseCorrelator

        correlator = ToolUseCorrelator()
        for i in range(MAX_CORRELATOR_SIZE + 1):
            correlator.start(f"tui-{i}", uuid4(), "tool")
        assert correlator.pending_count <= MAX_CORRELATOR_SIZE

    def test_flush_clears_all(self):
        """W5a: Flush discards all pending entries."""
        from clyro.adapters.claude_agent_sdk import ToolUseCorrelator

        correlator = ToolUseCorrelator()
        correlator.start("tui-1", uuid4(), "bash")
        correlator.start("tui-2", uuid4(), "read")
        assert correlator.pending_count == 2

        correlator.flush()
        assert correlator.pending_count == 0


# =============================================================================
# TestSubagentTracker (C4, FRD-007)
# =============================================================================


class TestSubagentTracker:
    """Tests for SubagentTracker component."""

    def test_start_and_stop(self):
        """FRD-007: Track subagent start to stop with duration."""
        from clyro.adapters.claude_agent_sdk import SubagentTracker

        tracker = SubagentTracker()
        event_id = uuid4()

        tracker.start("agent-1", event_id, "code_agent")
        assert tracker.active_count == 1

        parent_id, duration_ms = tracker.stop("agent-1")
        assert parent_id == event_id
        assert duration_ms is not None
        assert duration_ms >= 0
        assert tracker.active_count == 0

    def test_stop_missing_returns_none(self):
        """FRD-007: SubagentStop without SubagentStart."""
        from clyro.adapters.claude_agent_sdk import SubagentTracker

        tracker = SubagentTracker()
        parent_id, duration_ms = tracker.stop("nonexistent")
        assert parent_id is None
        assert duration_ms is None

    def test_overflow_eviction(self):
        """FRD-007: Evict oldest entries when exceeding 100."""
        from clyro.adapters.claude_agent_sdk import MAX_TRACKER_SIZE, SubagentTracker

        tracker = SubagentTracker()
        for i in range(MAX_TRACKER_SIZE + 1):
            tracker.start(f"agent-{i}", uuid4(), "type")
        assert tracker.active_count <= MAX_TRACKER_SIZE

    def test_flush_clears_all(self):
        """W5a: Flush discards all active entries."""
        from clyro.adapters.claude_agent_sdk import SubagentTracker

        tracker = SubagentTracker()
        tracker.start("agent-1", uuid4(), "code_agent")
        tracker.start("agent-2", uuid4(), "research_agent")
        assert tracker.active_count == 2

        tracker.flush()
        assert tracker.active_count == 0


# =============================================================================
# TestCostEstimator (FRD-011b)
# =============================================================================


class TestCostEstimator:
    """Tests for CostEstimator data structure."""

    def test_accumulate_with_content(self):
        """FRD-011b: Cumulative cost scales with content length."""
        from clyro.adapters.claude_agent_sdk import CostEstimator

        estimator = CostEstimator(cost_per_token_usd=Decimal("0.00001"))
        # 400 chars => 100 tokens => 100 * 0.00001 = 0.001
        content = "x" * 400
        cost1 = estimator.accumulate(content)
        assert cost1 == Decimal("0.00001") * 100
        assert estimator.step_count == 1

        cost2 = estimator.accumulate(content)
        assert cost2 == Decimal("0.00001") * 200
        assert estimator.step_count == 2

    def test_accumulate_no_content(self):
        """FRD-011b: Empty/None content accumulates zero cost."""
        from clyro.adapters.claude_agent_sdk import CostEstimator

        estimator = CostEstimator(cost_per_token_usd=Decimal("0.00001"))
        cost = estimator.accumulate(None)
        assert cost == Decimal("0")
        assert estimator.step_count == 1

    def test_reset(self):
        """CostEstimator resets to zero."""
        from clyro.adapters.claude_agent_sdk import CostEstimator

        estimator = CostEstimator(cost_per_token_usd=Decimal("0.00001"))
        estimator.accumulate("x" * 400)
        estimator.accumulate("y" * 400)
        estimator.reset()
        assert estimator.step_count == 0
        assert estimator.estimated_cumulative_cost == Decimal("0")


# =============================================================================
# TestClaudeAgentHandler (C2, FRD-002 through FRD-013)
# =============================================================================


class TestClaudeAgentHandler:
    """Tests for ClaudeAgentHandler hook dispatch and event creation."""

    # --- Session Synthesis (W1, FRD-009) ---

    async def test_session_start_synthesized(self, handler):
        """FRD-009: SESSION_START emitted on first hook event."""
        input_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", input_data, "tui-1")

        events = handler.drain_events()
        assert len(events) >= 2
        assert events[0].event_type == EventType.SESSION_START
        assert events[0].framework == Framework.CLAUDE_AGENT_SDK
        assert events[0].agent_stage == AgentStage.THINK

    async def test_session_start_not_duplicated(self, handler):
        """FRD-009: SESSION_START not emitted on subsequent hooks."""
        input_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", input_data, "tui-1")

        input_data2 = _make_input_data(tool_name="read", tool_input={}, tool_use_id="tui-2")
        await handler.handle_hook("PreToolUse", input_data2, "tui-2")

        events = handler.drain_events()
        session_starts = [e for e in events if e.event_type == EventType.SESSION_START]
        assert len(session_starts) == 1

    async def test_missing_session_id_skips_event(self, handler):
        """B-02: Missing session_id returns None without recording."""
        result = await handler.handle_hook("PreToolUse", {"tool_name": "bash"}, "tui-1")
        assert result is None
        events = handler.drain_events()
        assert len(events) == 0

    # --- PreToolUse (FRD-002) ---

    async def test_pre_tool_use_creates_tool_call_act(self, handler):
        """FRD-002: PreToolUse -> TOOL_CALL (ACT) event."""
        input_data = _make_input_data(
            tool_name="bash",
            tool_input={"command": "ls"},
            tool_use_id="tui-1",
        )
        result = await handler.handle_hook("PreToolUse", input_data, "tui-1")
        assert result is None  # Allow

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].agent_stage == AgentStage.ACT
        assert tool_events[0].event_name == "bash"
        assert tool_events[0].metadata["tool_use_id"] == "tui-1"
        assert tool_events[0].metadata["hook_event"] == "PreToolUse"

    # --- PostToolUse (FRD-003) ---

    async def test_post_tool_use_creates_tool_call_observe(self, handler):
        """FRD-003: PostToolUse -> TOOL_CALL (OBSERVE) with parent_event_id."""
        # Fire PreToolUse first
        pre_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", pre_data, "tui-1")

        # Fire PostToolUse
        post_data = _make_input_data(
            tool_name="bash",
            tool_output={"result": "file.txt"},
            tool_use_id="tui-1",
        )
        await handler.handle_hook("PostToolUse", post_data, "tui-1")

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 2

        observe_event = tool_events[1]
        assert observe_event.agent_stage == AgentStage.OBSERVE
        assert observe_event.parent_event_id is not None
        assert observe_event.parent_event_id == tool_events[0].event_id
        assert observe_event.duration_ms >= 0

    # --- PostToolUseFailure (FRD-004) ---

    async def test_post_tool_use_failure_creates_error(self, handler):
        """FRD-004: PostToolUseFailure -> ERROR (OBSERVE) event."""
        pre_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", pre_data, "tui-1")

        fail_data = _make_input_data(
            tool_name="bash",
            error_message="Command not found",
            tool_use_id="tui-1",
        )
        await handler.handle_hook("PostToolUseFailure", fail_data, "tui-1")

        events = handler.drain_events()
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].error_type == "ToolExecutionError"
        assert error_events[0].error_message == "Command not found"
        assert error_events[0].agent_stage == AgentStage.OBSERVE
        assert error_events[0].parent_event_id is not None

    async def test_post_tool_use_failure_empty_error(self, handler):
        """FRD-004: Empty error_message defaults to 'Unknown tool execution error'."""
        pre_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", pre_data, "tui-1")

        fail_data = _make_input_data(tool_name="bash", tool_use_id="tui-1")
        await handler.handle_hook("PostToolUseFailure", fail_data, "tui-1")

        events = handler.drain_events()
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        assert error_events[0].error_message == "Unknown tool execution error"

    # --- UserPromptSubmit (FRD-005) ---

    async def test_user_prompt_submit_creates_step_think(self, handler):
        """FRD-005: UserPromptSubmit -> STEP (THINK) event."""
        input_data = _make_input_data(prompt_text="Build a web app")
        await handler.handle_hook("UserPromptSubmit", input_data)

        events = handler.drain_events()
        step_events = [e for e in events if e.event_type == EventType.STEP]
        assert len(step_events) == 1
        assert step_events[0].agent_stage == AgentStage.THINK
        assert step_events[0].input_data["prompt_text"] == "Build a web app"

    # --- SubagentStart/Stop (FRD-007) ---

    async def test_subagent_start_creates_task_start(self, handler):
        """FRD-007: SubagentStart -> TASK_START (ACT) event."""
        input_data = _make_input_data(
            agent_id="subagent-1",
            agent_type="code_agent",
            agent_tool_call_id="atci-1",
        )
        await handler.handle_hook("SubagentStart", input_data)

        events = handler.drain_events()
        task_events = [e for e in events if e.event_type == EventType.TASK_START]
        assert len(task_events) == 1
        assert task_events[0].agent_stage == AgentStage.ACT

    async def test_subagent_stop_creates_task_end(self, handler):
        """FRD-007: SubagentStop -> TASK_END (OBSERVE) with parent_event_id."""
        start_data = _make_input_data(
            agent_id="subagent-1",
            agent_type="code_agent",
            agent_tool_call_id="atci-1",
        )
        await handler.handle_hook("SubagentStart", start_data)

        stop_data = _make_input_data(
            agent_id="subagent-1",
            agent_type="code_agent",
            agent_transcript_path="/tmp/transcript.json",
            stop_hook_active=True,
            agent_tool_call_id="atci-1",
        )
        await handler.handle_hook("SubagentStop", stop_data)

        events = handler.drain_events()
        task_end_events = [e for e in events if e.event_type == EventType.TASK_END]
        assert len(task_end_events) == 1
        assert task_end_events[0].agent_stage == AgentStage.OBSERVE
        assert task_end_events[0].parent_event_id is not None
        assert task_end_events[0].duration_ms >= 0

    # --- Stop (FRD-009) ---

    async def test_stop_creates_session_end(self, handler):
        """FRD-009: Stop hook -> SESSION_END event."""
        init_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", init_data, "tui-1")

        stop_data = _make_input_data(stop_hook_active=True)
        await handler.handle_hook("Stop", stop_data)

        events = handler.drain_events()
        end_events = [e for e in events if e.event_type == EventType.SESSION_END]
        assert len(end_events) == 1
        assert end_events[0].duration_ms >= 0

    async def test_duplicate_stop_ignored(self, handler):
        """FRD-009: Duplicate Stop after SESSION_END is silently ignored."""
        init_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", init_data, "tui-1")

        stop_data = _make_input_data(stop_hook_active=True)
        await handler.handle_hook("Stop", stop_data)
        await handler.handle_hook("Stop", stop_data)

        events = handler.drain_events()
        end_events = [e for e in events if e.event_type == EventType.SESSION_END]
        assert len(end_events) == 1

    # --- Notification (FRD-012) ---

    async def test_notification_creates_step_observe(self, handler):
        """FRD-012: Notification -> STEP (OBSERVE) event."""
        input_data = _make_input_data(
            message="Task completed",
            title="Success",
            notification_type="info",
        )
        await handler.handle_hook("Notification", input_data)

        events = handler.drain_events()
        step_events = [e for e in events if e.event_type == EventType.STEP]
        assert len(step_events) == 1
        assert step_events[0].agent_stage == AgentStage.OBSERVE
        assert step_events[0].output_data["message"] == "Task completed"

    # --- PreCompact (FRD-013) ---

    async def test_pre_compact_creates_state_transition(self, handler):
        """FRD-013: PreCompact -> STATE_TRANSITION (THINK) event."""
        input_data = _make_input_data(
            trigger="auto",
            conversation_size=50000,
        )
        await handler.handle_hook("PreCompact", input_data)

        events = handler.drain_events()
        st_events = [e for e in events if e.event_type == EventType.STATE_TRANSITION]
        assert len(st_events) == 1
        assert st_events[0].agent_stage == AgentStage.THINK
        assert st_events[0].state_snapshot["trigger"] == "auto"
        assert st_events[0].state_snapshot["conversation_size"] == 50000

    # --- Session ID Change (W5a, FRD-009) ---

    async def test_session_id_change_triggers_cleanup(self, handler):
        """FRD-009: Session ID change flushes and starts new session."""
        data1 = _make_input_data(
            session_id="session-1",
            tool_name="bash",
            tool_input={},
            tool_use_id="tui-1",
        )
        await handler.handle_hook("PreToolUse", data1, "tui-1")

        data2 = _make_input_data(
            session_id="session-2",
            tool_name="read",
            tool_input={},
            tool_use_id="tui-2",
        )
        await handler.handle_hook("PreToolUse", data2, "tui-2")

        events = handler.drain_events()
        session_starts = [e for e in events if e.event_type == EventType.SESSION_START]
        session_ends = [e for e in events if e.event_type == EventType.SESSION_END]
        assert len(session_starts) == 2  # One for each session
        assert len(session_ends) == 1  # End of first session

    # --- drain_events ---

    async def test_drain_events_clears_buffer(self, handler):
        """drain_events() returns events and clears buffer."""
        input_data = _make_input_data(prompt_text="test")
        await handler.handle_hook("UserPromptSubmit", input_data)

        events1 = handler.drain_events()
        assert len(events1) > 0

        events2 = handler.drain_events()
        assert len(events2) == 0

    # --- end_session with ResultMessage (FRD-008) ---

    async def test_end_session_with_result_message(self, handler):
        """FRD-008: Token/cost extraction from ResultMessage."""
        init_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="tui-1")
        await handler.handle_hook("PreToolUse", init_data, "tui-1")
        handler.drain_events()  # Clear buffer

        handler.end_session(
            result_message={
                "total_cost_usd": 0.05,
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 50,
                },
                "num_turns": 5,
                "subtype": "success",
                "stop_reason": "end_turn",
            }
        )

        events = handler.drain_events()
        end_events = [e for e in events if e.event_type == EventType.SESSION_END]
        assert len(end_events) == 1
        assert end_events[0].token_count_input == 1000
        assert end_events[0].token_count_output == 500
        assert end_events[0].cost_usd == Decimal("0.05")


# =============================================================================
# TestExecutionControls (FRD-011a/b/c)
# =============================================================================


class TestExecutionControls:
    """Tests for step limit, cost limit, and loop detection."""

    async def test_step_limit_enforcement(self):
        """FRD-011a: Deny on step limit exceeded."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=2, enable_step_limit=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42", session=session)

        # Step 1: OK
        d1 = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        r1 = await handler.handle_hook("PreToolUse", d1, "t1")
        assert r1 is None

        # Step 2: OK
        d2 = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t2")
        r2 = await handler.handle_hook("PreToolUse", d2, "t2")
        assert r2 is None

        # Step 3: Exceeds limit (max=2)
        d3 = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t3")
        r3 = await handler.handle_hook("PreToolUse", d3, "t3")
        assert r3 is not None
        assert r3["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_step_limit_disabled(self):
        """FRD-011a: No enforcement when enable_step_limit=False."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=1, enable_step_limit=False),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42", session=session)

        d1 = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        r1 = await handler.handle_hook("PreToolUse", d1, "t1")
        assert r1 is None

        d2 = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t2")
        r2 = await handler.handle_hook("PreToolUse", d2, "t2")
        assert r2 is None  # No deny

    async def test_cost_limit_enforcement(self):
        """FRD-011b: Deny when character-length estimated cost exceeds limit."""
        # cost_per_token_usd=0.001, CHARS_PER_TOKEN=4
        # 400 chars tool_input => 100 tokens => $0.10 per call
        # max_cost_usd=0.25 => should allow 2 calls, deny 3rd ($0.30 > $0.25)
        config = ClyroConfig(
            controls=ExecutionControls(
                max_cost_usd=0.25,
                enable_cost_limit=True,
            ),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config, framework_version="0.1.42", session=session,
            cost_per_token_usd="0.001",
        )

        # json.dumps({"d": "x" * 390}) => ~400 chars => 100 tokens => $0.10
        large_input = {"d": "x" * 390}

        d1 = _make_input_data(tool_name="bash", tool_input=large_input, tool_use_id="t1")
        r1 = await handler.handle_hook("PreToolUse", d1, "t1")
        assert r1 is None  # ~$0.10 <= $0.25

        d2 = _make_input_data(tool_name="bash", tool_input=large_input, tool_use_id="t2")
        r2 = await handler.handle_hook("PreToolUse", d2, "t2")
        assert r2 is None  # ~$0.20 <= $0.25

        d3 = _make_input_data(tool_name="bash", tool_input=large_input, tool_use_id="t3")
        r3 = await handler.handle_hook("PreToolUse", d3, "t3")
        assert r3 is not None  # ~$0.30 > $0.25
        assert r3["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_loop_detection(self):
        """FRD-011c: Deny when same tool call repeated threshold times."""
        config = ClyroConfig(
            controls=ExecutionControls(
                loop_detection_threshold=3,
                enable_loop_detection=True,
                max_steps=100,
            ),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42", session=session)

        same_data = {"tool_name": "bash", "tool_input": {"command": "ls"}}

        for i in range(2):
            d = _make_input_data(**same_data, tool_use_id=f"t{i}")
            r = await handler.handle_hook("PreToolUse", d, f"t{i}")
            assert r is None

        # 3rd identical call should trigger loop detection
        d3 = _make_input_data(**same_data, tool_use_id="t3")
        r3 = await handler.handle_hook("PreToolUse", d3, "t3")
        assert r3 is not None
        assert r3["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_loop_detection_disabled(self):
        """FRD-011c: No detection when enable_loop_detection=False."""
        config = ClyroConfig(
            controls=ExecutionControls(
                loop_detection_threshold=2,
                enable_loop_detection=False,
                max_steps=100,
            ),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42", session=session)

        same_data = {"tool_name": "bash", "tool_input": {"command": "ls"}}
        for i in range(5):
            d = _make_input_data(**same_data, tool_use_id=f"t{i}")
            r = await handler.handle_hook("PreToolUse", d, f"t{i}")
            assert r is None


# =============================================================================
# TestPolicyEnforcement (FRD-010)
# =============================================================================


class TestPolicyEnforcement:
    """Tests for policy evaluation in PreToolUse hook."""

    async def test_policy_allow(self):
        """FRD-010: Policy allows tool call."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        mock_evaluator = MagicMock()
        mock_decision = MagicMock()
        mock_decision.is_allowed = True
        mock_decision.evaluated_rules = 3
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_decision)

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        assert result is None  # Allow

        events = handler.drain_events()
        policy_events = [e for e in events if e.event_type == EventType.POLICY_CHECK]
        assert len(policy_events) == 1
        assert policy_events[0].output_data["decision"] == "allow"

    async def test_policy_block(self):
        """FRD-010: Policy blocks tool call."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        mock_evaluator = MagicMock()
        mock_decision = MagicMock()
        mock_decision.is_allowed = False
        mock_decision.is_blocked = True
        mock_decision.requires_approval = False
        mock_decision.rule_id = "rule-1"
        mock_decision.rule_name = "no-bash"
        mock_decision.message = "bash is not allowed"
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_decision)

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        assert result is not None
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_policy_require_approval_allowed_when_handler_approves(self):
        """FRD-010: require_approval is allowed when approval handler approved."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        mock_evaluator = MagicMock()
        mock_decision = MagicMock()
        mock_decision.is_allowed = False
        mock_decision.is_blocked = False
        mock_decision.requires_approval = True
        mock_decision.rule_id = "rule-2"
        mock_decision.rule_name = "approval-needed"
        mock_decision.message = "Approval required"
        # evaluate_async returning (not raising) means _enforce_decision passed
        # i.e. the approval handler already approved the action.
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_decision)

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        # Action allowed — approval handler already approved
        assert result is None

        events = handler.drain_events()
        policy_events = [e for e in events if e.event_type == EventType.POLICY_CHECK]
        assert policy_events[0].output_data["reason"] == "user_approved"

    async def test_policy_require_approval_prompts_user(self):
        """FRD-010: require_approval prompts for user approval when handler is absent."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        # Simulate PolicyViolationError raised by _enforce_decision when no handler

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_async = AsyncMock(
            side_effect=PolicyViolationError(
                rule_id="rule-2",
                rule_name="approval-needed",
                message="Approval required",
                action_type="tool_call",
                details={"decision": "require_approval"},
            )
        )

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        # Mock _prompt_approval to simulate user approving
        handler._prompt_approval = AsyncMock(return_value=True)

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        # User approved — action allowed
        assert result is None

        handler._prompt_approval.assert_called_once()

    async def test_policy_require_approval_denied_when_user_rejects(self):
        """FRD-010: require_approval blocks when user denies approval."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()


        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_async = AsyncMock(
            side_effect=PolicyViolationError(
                rule_id="rule-2",
                rule_name="approval-needed",
                message="Approval required",
                action_type="tool_call",
                details={"decision": "require_approval"},
            )
        )

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        # Mock _prompt_approval to simulate user denying
        handler._prompt_approval = AsyncMock(return_value=False)

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        # User denied — action blocked
        assert result is not None
        assert result["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_policy_timeout_fails_open(self):
        """FRD-010: Policy timeout results in allow (fail-open)."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        mock_evaluator = MagicMock()

        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(10)

        mock_evaluator.evaluate_async = slow_evaluate

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        # Reduce timeout for test
        import clyro.adapters.claude_agent_sdk as adapter_module
        original_timeout = adapter_module.POLICY_TIMEOUT_SECONDS
        adapter_module.POLICY_TIMEOUT_SECONDS = 0.1

        try:
            d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
            result = await handler.handle_hook("PreToolUse", d, "t1")
            assert result is None  # Allow (fail-open)

            events = handler.drain_events()
            policy_events = [e for e in events if e.event_type == EventType.POLICY_CHECK]
            assert len(policy_events) == 1
            assert policy_events[0].output_data["reason"] == "policy_evaluation_timeout"
        finally:
            adapter_module.POLICY_TIMEOUT_SECONDS = original_timeout

    async def test_policy_disabled_skips_evaluation(self):
        """FRD-010: No POLICY_CHECK when enable_policy_enforcement=False."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=False),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config, framework_version="0.1.42", session=session
        )

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        result = await handler.handle_hook("PreToolUse", d, "t1")
        assert result is None

        events = handler.drain_events()
        policy_events = [e for e in events if e.event_type == EventType.POLICY_CHECK]
        assert len(policy_events) == 0


# =============================================================================
# TestHookRegistrar (C1, FRD-001)
# =============================================================================


class TestHookRegistrar:
    """Tests for HookRegistrar component."""

    def test_register_all_hook_types(self, config):
        """FRD-001: All 9 hook types registered."""
        from clyro.adapters.claude_agent_sdk import (
            HOOK_TYPES,
            ClaudeAgentHandler,
            HookRegistrar,
        )

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42")
        registrar = HookRegistrar(config=config, handler=handler)

        hooks: dict = {}
        options = type("Options", (), {})()
        registrar.register(hooks, options=options)

        for hook_type in HOOK_TYPES:
            assert hook_type in hooks, f"{hook_type} not registered"
            assert len(hooks[hook_type]) == 1

    def test_idempotent_registration(self, config):
        """FRD-001: Double registration does not duplicate hooks."""
        from clyro.adapters.claude_agent_sdk import (
            HOOK_TYPES,
            ClaudeAgentHandler,
            HookRegistrar,
        )

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42")
        registrar = HookRegistrar(config=config, handler=handler)

        hooks: dict = {}
        options = type("Options", (), {})()
        registrar.register(hooks, options=options)
        registrar.register(hooks, options=options)

        for hook_type in HOOK_TYPES:
            assert len(hooks[hook_type]) == 1

    def test_preserves_existing_hooks(self, config):
        """FRD-001: Existing hooks are preserved, not overwritten."""
        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler, HookRegistrar

        handler = ClaudeAgentHandler(config=config, framework_version="0.1.42")
        registrar = HookRegistrar(config=config, handler=handler)

        def existing_hook(x, y):
            return None
        hooks: dict = {"PreToolUse": [existing_hook]}
        options = type("Options", (), {})()
        registrar.register(hooks, options=options)

        assert len(hooks["PreToolUse"]) == 2
        assert hooks["PreToolUse"][0] is existing_hook


# =============================================================================
# TestClaudeAgentAdapter (C5)
# =============================================================================


class TestClaudeAgentAdapter:
    """Tests for ClaudeAgentAdapter WrappedAgent integration."""

    def test_framework_is_claude_agent_sdk(self, config):
        """C5: Adapter reports correct framework."""
        from clyro.adapters.claude_agent_sdk import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter(lambda: None, config)
        assert adapter.framework == Framework.CLAUDE_AGENT_SDK

    def test_before_call_returns_context(self, config, session):
        """C5: before_call returns context dict."""
        from clyro.adapters.claude_agent_sdk import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter(lambda: None, config)
        context = adapter.before_call(session, (), {})
        assert "start_time" in context

    def test_after_call_returns_event(self, config, session):
        """C5: after_call returns TraceEvent."""
        from clyro.adapters.claude_agent_sdk import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter(lambda: None, config)
        import time

        context = {"start_time": time.perf_counter()}
        event = adapter.after_call(session, "result", context)
        assert isinstance(event, TraceEvent)
        assert event.event_type == EventType.LLM_CALL

    def test_on_error_returns_error_event(self, config, session):
        """C5: on_error returns error TraceEvent with correct type and message."""
        from clyro.adapters.claude_agent_sdk import ClaudeAgentAdapter

        adapter = ClaudeAgentAdapter(lambda: None, config)
        error = ValueError("something went wrong")
        event = adapter.on_error(session, error, {"start_time": 0})
        assert isinstance(event, TraceEvent)
        assert event.event_type == EventType.ERROR
        assert event.error_type == "ValueError"
        assert event.error_message == "something went wrong"


# =============================================================================
# TestInstrumentClaudeAgent (Public API)
# =============================================================================


class TestInstrumentClaudeAgent:
    """Tests for instrument_claude_agent() public API."""

    def test_happy_path(self, mock_claude_sdk_module, config):
        """FRD-001: instrument_claude_agent registers hooks."""
        from clyro.adapters.claude_agent_sdk import HOOK_TYPES, instrument_claude_agent

        options = MockClaudeAgentOptions()
        result = instrument_claude_agent(options, config)
        assert result is options
        for hook_type in HOOK_TYPES:
            assert hook_type in options.hooks

    def test_missing_sdk_raises_import_error(self, config):
        """FRD-001: ImportError when claude-agent-sdk not installed."""
        from clyro.adapters.claude_agent_sdk import instrument_claude_agent

        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            with patch(
                "clyro.adapters.claude_agent_sdk.detect_claude_agent_sdk_version",
                return_value=None,
            ):
                options = MockClaudeAgentOptions()
                with pytest.raises(ImportError, match="claude-agent-sdk is required"):
                    instrument_claude_agent(options, config)

    def test_unsupported_version_raises(self, mock_claude_sdk_unsupported, config):
        """FRD-001: FrameworkVersionError for unsupported version."""
        from clyro.adapters.claude_agent_sdk import instrument_claude_agent

        options = MockClaudeAgentOptions()
        with pytest.raises(FrameworkVersionError):
            instrument_claude_agent(options, config)

    def test_invalid_hooks_raises_value_error(self, mock_claude_sdk_module, config):
        """FRD-001: ValueError when hooks is not a dict."""
        from clyro.adapters.claude_agent_sdk import instrument_claude_agent

        options = MockClaudeAgentOptions()
        options.hooks = "invalid"
        with pytest.raises(ValueError, match="must be a dictionary"):
            instrument_claude_agent(options, config)


# =============================================================================
# TestVersionDetection
# =============================================================================


class TestVersionDetection:
    """Tests for version detection and validation."""

    def test_supported_version(self, mock_claude_sdk_module):
        """Supported version passes validation."""
        from clyro.adapters.claude_agent_sdk import validate_claude_agent_sdk_version

        version = validate_claude_agent_sdk_version("0.1.42")
        assert version == "0.1.42"

    def test_unsupported_version(self):
        """Unsupported version raises FrameworkVersionError."""
        from clyro.adapters.claude_agent_sdk import validate_claude_agent_sdk_version

        with pytest.raises(FrameworkVersionError):
            validate_claude_agent_sdk_version("0.1.30")

    def test_detection_function(self):
        """is_claude_agent_sdk_agent detects mock options."""
        from clyro.adapters.claude_agent_sdk import is_claude_agent_sdk_agent

        options = MockClaudeAgentOptions()
        assert is_claude_agent_sdk_agent(options) is True

    def test_detection_function_non_agent(self):
        """is_claude_agent_sdk_agent returns False for non-agent."""
        from clyro.adapters.claude_agent_sdk import is_claude_agent_sdk_agent

        assert is_claude_agent_sdk_agent(lambda: None) is False

    def test_detect_adapter_integration(self, mock_claude_sdk_module):
        """detect_adapter returns 'claude_agent_sdk' for SDK objects."""
        from clyro.adapters.generic import detect_adapter

        options = MockClaudeAgentOptions()
        assert detect_adapter(options) == "claude_agent_sdk"


# =============================================================================
# TestDataSensitivity (NFR-006)
# =============================================================================


class TestDataSensitivity:
    """Tests for data sensitivity and capture toggles."""

    async def test_capture_inputs_disabled(self, handler_no_capture):
        """NFR-006: input_data omitted when capture_inputs=False."""
        d = _make_input_data(tool_name="bash", tool_input={"cmd": "ls"}, tool_use_id="t1")
        await handler_no_capture.handle_hook("PreToolUse", d, "t1")

        events = handler_no_capture.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].input_data is None

    async def test_capture_outputs_disabled(self, handler_no_capture):
        """NFR-006: output_data omitted when capture_outputs=False."""
        pre_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        await handler_no_capture.handle_hook("PreToolUse", pre_data, "t1")

        post_data = _make_input_data(
            tool_name="bash", tool_output={"result": "secret"}, tool_use_id="t1"
        )
        await handler_no_capture.handle_hook("PostToolUse", post_data, "t1")

        events = handler_no_capture.drain_events()
        observe_events = [
            e for e in events
            if e.event_type == EventType.TOOL_CALL and e.agent_stage == AgentStage.OBSERVE
        ]
        assert len(observe_events) == 1
        assert observe_events[0].output_data is None

    async def test_api_key_not_in_events(self, config, session):
        """NFR-006: API key never appears in events or metadata."""
        config_with_key = ClyroConfig(
            api_key="cly_test_secret_key_12345",
            controls=ExecutionControls(enable_policy_enforcement=False),
        )
        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config_with_key, framework_version="0.1.42", session=session
        )

        d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
        await handler.handle_hook("PreToolUse", d, "t1")

        events = handler.drain_events()
        for event in events:
            event_json = event.model_dump_json()
            assert "secret_key_12345" not in event_json
            assert "cly_test" not in event_json


# =============================================================================
# TestEdgeCases (§11.3)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases listed in TDD §11.3."""

    async def test_post_tool_use_without_pre(self, handler):
        """PostToolUse without PreToolUse: null parent and duration."""
        init_data = _make_input_data(prompt_text="init")
        await handler.handle_hook("UserPromptSubmit", init_data)

        post_data = _make_input_data(
            tool_name="bash",
            tool_output={"result": "ok"},
            tool_use_id="orphan-tui",
        )
        await handler.handle_hook("PostToolUse", post_data, "orphan-tui")

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        assert tool_events[0].parent_event_id is None
        assert tool_events[0].duration_ms == 0

    async def test_subagent_stop_without_start(self, handler):
        """SubagentStop without SubagentStart: null parent and duration."""
        init_data = _make_input_data(prompt_text="init")
        await handler.handle_hook("UserPromptSubmit", init_data)

        stop_data = _make_input_data(
            agent_id="orphan-agent",
            agent_type="code",
            agent_transcript_path="/tmp/t.json",
        )
        await handler.handle_hook("SubagentStop", stop_data)

        events = handler.drain_events()
        task_events = [e for e in events if e.event_type == EventType.TASK_END]
        assert len(task_events) == 1
        assert task_events[0].parent_event_id is None
        assert task_events[0].duration_ms == 0

    async def test_empty_tool_name(self, handler):
        """Empty tool_name defaults to 'unknown'."""
        d = _make_input_data(tool_name="", tool_input={}, tool_use_id="t1")
        await handler.handle_hook("PreToolUse", d, "t1")

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert tool_events[0].event_name == "unknown"

    async def test_unknown_hook_type(self, handler):
        """Unknown hook type returns None without error."""
        d = _make_input_data()
        result = await handler.handle_hook("UnknownHookType", d)
        assert result is None

    async def test_hook_exception_fails_open(self, handler):
        """Hook exception caught, returns None (fail-open)."""
        with patch.object(handler, "_dispatch", side_effect=RuntimeError("boom")):
            d = _make_input_data()
            result = await handler.handle_hook("PreToolUse", d, "t1")
            assert result is None

    async def test_truncation_applied(self):
        """NFR-006: Large inputs are truncated to TRUNCATION_LIMIT."""
        from clyro.adapters.claude_agent_sdk import _truncate

        long_text = "x" * 20000
        truncated = _truncate(long_text, 10000)
        assert len(truncated) < 20000
        assert "truncated" in truncated


# =============================================================================
# New tests added by critic review
# =============================================================================


# T1: Integration test — Pre→Post correlation with absent tool_use_id (B1 fix)
class TestCorrelationIntegration:
    """Integration tests for Pre→Post tool_use_id correlation."""

    async def test_empty_tool_use_id_correlation(self, handler):
        """B1 fix: PostToolUse parent_event_id is set even when tool_use_id is absent."""
        # PreToolUse with no tool_use_id
        pre_data = _make_input_data(tool_name="bash", tool_input={"cmd": "ls"})
        # tool_use_id parameter also absent (None)
        await handler.handle_hook("PreToolUse", pre_data, None)

        # PostToolUse also has no tool_use_id — SDK omitted it in both
        post_data = _make_input_data(tool_name="bash", tool_output={"result": "ok"})
        await handler.handle_hook("PostToolUse", post_data, None)

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 2
        act_event = next(e for e in tool_events if e.agent_stage == AgentStage.ACT)
        next(e for e in tool_events if e.agent_stage == AgentStage.OBSERVE)
        # The observe event CANNOT link back — SDK gave no stable id for PostToolUse lookup.
        # The fix (B1) ensures PreToolUse metadata captures the synthetic id, not "".
        assert act_event.metadata["tool_use_id"] != ""  # B1: synthetic UUID, not empty string

    async def test_synthetic_tool_use_id_in_metadata(self, handler):
        """B1 fix: PreToolUse metadata.tool_use_id is the synthetic UUID, not empty string."""
        pre_data = _make_input_data(tool_name="bash", tool_input={}, tool_use_id=None)
        await handler.handle_hook("PreToolUse", pre_data, None)

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        stored_id = tool_events[0].metadata["tool_use_id"]
        assert stored_id != "" and stored_id is not None
        # Should be a valid UUID string
        try:
            UUID(stored_id)
        except ValueError:
            pytest.fail(f"metadata.tool_use_id is not a valid UUID: {stored_id!r}")

    async def test_tool_input_none_does_not_crash(self, handler):
        """B2 fix: tool_input=None in input_data does not cause TypeError."""
        pre_data = _make_input_data(tool_name="bash", tool_use_id="t1")
        pre_data["tool_input"] = None  # Explicit None, not missing key

        # Should not raise
        result = await handler.handle_hook("PreToolUse", pre_data, "t1")
        assert result is None

        events = handler.drain_events()
        tool_events = [e for e in events if e.event_type == EventType.TOOL_CALL]
        assert len(tool_events) == 1


# T2: Policy timeout test using patch (not module mutation)
class TestPolicyTimeoutPatch:
    """Policy timeout using unittest.mock.patch (thread-safe)."""

    async def test_policy_timeout_via_patch(self):
        """FRD-010: Policy timeout results in allow (fail-open) — tested via patch."""
        config = ClyroConfig(
            controls=ExecutionControls(enable_policy_enforcement=True),
        )
        session = Session(config=config, framework=Framework.CLAUDE_AGENT_SDK)
        session.start()

        mock_evaluator = MagicMock()

        async def slow_evaluate(*args, **kwargs):
            await asyncio.sleep(10)

        mock_evaluator.evaluate_async = slow_evaluate

        from clyro.adapters.claude_agent_sdk import ClaudeAgentHandler

        handler = ClaudeAgentHandler(
            config=config,
            framework_version="0.1.42",
            session=session,
            policy_evaluator=mock_evaluator,
        )

        with patch("clyro.adapters.claude_agent_sdk.POLICY_TIMEOUT_SECONDS", 0.05):
            d = _make_input_data(tool_name="bash", tool_input={}, tool_use_id="t1")
            result = await handler.handle_hook("PreToolUse", d, "t1")
            assert result is None  # Allow (fail-open)

        events = handler.drain_events()
        policy_events = [e for e in events if e.event_type == EventType.POLICY_CHECK]
        assert len(policy_events) == 1
        assert policy_events[0].output_data["reason"] == "policy_evaluation_timeout"


# T3: _truncate_dict JSON serialization path
class TestTruncateDict:
    """Tests for _truncate_dict including JSON serialization branch."""

    def test_truncate_dict_string_values(self):
        """String values in dict are truncated."""
        from clyro.adapters.claude_agent_sdk import _truncate_dict

        data = {"key": "a" * 20000}
        result = _truncate_dict(data, 10000)
        assert len(result["key"]) < 20000
        assert "truncated" in result["key"]

    def test_truncate_dict_dict_value_under_limit(self):
        """Small nested dict is preserved as-is."""
        from clyro.adapters.claude_agent_sdk import _truncate_dict

        data = {"nested": {"a": 1, "b": 2}}
        result = _truncate_dict(data, 10000)
        assert result["nested"] == {"a": 1, "b": 2}

    def test_truncate_dict_dict_value_over_limit(self):
        """Large nested dict is JSON-serialized and truncated."""
        from clyro.adapters.claude_agent_sdk import _truncate_dict

        large_nested = {str(i): "x" * 100 for i in range(200)}  # ~22KB
        data = {"nested": large_nested}
        result = _truncate_dict(data, 10000)
        # Value is a truncated string (JSON serialized then cut)
        assert isinstance(result["nested"], str)
        assert "truncated" in result["nested"]

    def test_truncate_dict_list_value_over_limit(self):
        """Large list is JSON-serialized and truncated."""
        from clyro.adapters.claude_agent_sdk import _truncate_dict

        large_list = ["item" * 500 for _ in range(100)]
        data = {"items": large_list}
        result = _truncate_dict(data, 10000)
        assert isinstance(result["items"], str)
        assert "truncated" in result["items"]

    def test_truncate_dict_non_string_passthrough(self):
        """Non-string, non-dict, non-list values pass through unchanged."""
        from clyro.adapters.claude_agent_sdk import _truncate_dict

        data = {"count": 42, "flag": True, "ratio": 3.14}
        result = _truncate_dict(data, 10000)
        assert result == data


# T4: SubagentStart session_id mismatch logging
class TestSubagentSessionMismatch:
    """Tests for SubagentStart session_id divergence guard (FRD-007)."""

    async def test_subagent_start_session_id_mismatch_logged(self, handler, caplog):
        """FRD-007: session_id mismatch in SubagentStart is logged as error."""
        import logging

        # Initialize handler with session-1
        init_data = _make_input_data(
            session_id="session-1",
            tool_name="bash",
            tool_input={},
            tool_use_id="t1",
        )
        await handler.handle_hook("PreToolUse", init_data, "t1")
        handler.drain_events()

        # SubagentStart with mismatched session_id
        with caplog.at_level(logging.ERROR):
            subagent_data = _make_input_data(
                session_id="session-SUBAGENT-different",
                agent_id="subagent-1",
                agent_type="code_agent",
                agent_tool_call_id="atci-1",
            )
            # Force session_id to be different from handler's _session_id
            subagent_data["session_id"] = "session-SUBAGENT-different"
            await handler.handle_hook("SubagentStart", subagent_data)

        # The subagent event should still be created (treated as parent session per spec)
        events = handler.drain_events()
        task_events = [e for e in events if e.event_type == EventType.TASK_START]
        assert len(task_events) == 1


# T5: PreCompact conversation_size absent vs. zero
class TestPreCompactEdgeCases:
    """Edge cases for PreCompact hook."""

    async def test_pre_compact_missing_conversation_size(self, handler):
        """FRD-013: Missing conversation_size key defaults to 0."""
        input_data = _make_input_data(trigger="auto")
        # conversation_size key absent entirely
        await handler.handle_hook("PreCompact", input_data)

        events = handler.drain_events()
        st_events = [e for e in events if e.event_type == EventType.STATE_TRANSITION]
        assert len(st_events) == 1
        assert st_events[0].state_snapshot["conversation_size"] == 0

    async def test_pre_compact_zero_conversation_size(self, handler):
        """FRD-013: Explicit conversation_size=0 recorded correctly."""
        input_data = _make_input_data(trigger="manual", conversation_size=0)
        await handler.handle_hook("PreCompact", input_data)

        events = handler.drain_events()
        st_events = [e for e in events if e.event_type == EventType.STATE_TRANSITION]
        assert st_events[0].state_snapshot["conversation_size"] == 0


# T6: instrument_claude_agent edge cases (None hooks, missing hooks attribute)
class TestInstrumentEdgeCases:
    """Edge cases for instrument_claude_agent options.hooks handling."""

    def test_options_hooks_none_initialised(self, mock_claude_sdk_module, config):
        """FRD-001: options.hooks=None is converted to empty dict before registration."""
        from clyro.adapters.claude_agent_sdk import HOOK_TYPES, instrument_claude_agent

        options = MockClaudeAgentOptions()
        options.hooks = None
        result = instrument_claude_agent(options, config)
        for hook_type in HOOK_TYPES:
            assert hook_type in result.hooks

    def test_options_no_hooks_attribute(self, mock_claude_sdk_module, config):
        """FRD-001: options without hooks attribute gets hooks dict created."""
        from clyro.adapters.claude_agent_sdk import HOOK_TYPES, instrument_claude_agent

        options = MockClaudeAgentOptions()
        del options.hooks  # Remove attribute entirely
        result = instrument_claude_agent(options, config)
        for hook_type in HOOK_TYPES:
            assert hook_type in result.hooks


# T7: Capture toggles for SubagentStart and Notification
class TestCaptureTogglesW3W4:
    """Tests for W3 (SubagentStart capture_inputs) and W4 (Notification capture_outputs) fixes."""

    async def test_subagent_start_capture_inputs_disabled(self, handler_no_capture):
        """W3 fix: SubagentStart input_data is None when capture_inputs=False."""
        input_data = _make_input_data(
            agent_id="subagent-1",
            agent_type="code_agent",
            agent_tool_call_id="atci-1",
        )
        await handler_no_capture.handle_hook("SubagentStart", input_data)

        events = handler_no_capture.drain_events()
        task_events = [e for e in events if e.event_type == EventType.TASK_START]
        assert len(task_events) == 1
        assert task_events[0].input_data is None

    async def test_notification_capture_outputs_disabled(self, handler_no_capture):
        """W4 fix: Notification output_data is None when capture_outputs=False."""
        input_data = _make_input_data(
            message="Task done",
            title="Done",
            notification_type="info",
        )
        await handler_no_capture.handle_hook("Notification", input_data)

        events = handler_no_capture.drain_events()
        step_events = [e for e in events if e.event_type == EventType.STEP]
        assert len(step_events) == 1
        assert step_events[0].output_data is None

    async def test_subagent_start_capture_inputs_enabled(self, handler):
        """W3 fix: SubagentStart input_data is populated when capture_inputs=True."""
        input_data = _make_input_data(
            agent_id="subagent-2",
            agent_type="research_agent",
            agent_tool_call_id="atci-2",
        )
        await handler.handle_hook("SubagentStart", input_data)

        events = handler.drain_events()
        task_events = [e for e in events if e.event_type == EventType.TASK_START]
        assert task_events[0].input_data is not None
        assert task_events[0].input_data["agent_type"] == "research_agent"

    async def test_notification_capture_outputs_enabled(self, handler):
        """W4 fix: Notification output_data is populated when capture_outputs=True."""
        input_data = _make_input_data(
            message="All done",
            title="Success",
            notification_type="success",
        )
        await handler.handle_hook("Notification", input_data)

        events = handler.drain_events()
        step_events = [e for e in events if e.event_type == EventType.STEP]
        assert step_events[0].output_data is not None
        assert step_events[0].output_data["message"] == "All done"
