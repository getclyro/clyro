"""Unit tests for backend integration module. FRD-HK-007, HK-008, HK-009."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from clyro.backend.circuit_breaker import (
    FAILURE_THRESHOLD as CIRCUIT_FAILURE_THRESHOLD,
    OPEN_TIMEOUT_SECONDS as CIRCUIT_OPEN_TIMEOUT_SECONDS,
    SUCCESS_THRESHOLD as CIRCUIT_SUCCESS_THRESHOLD,
)
from clyro.hooks.backend import (
    _agent_id_path,
    _event_queue_path,
    _load_persisted_agent_id,
    _memory_fallback,
    _persist_agent_id,
    _unconfirmed_path,
    circuit_can_execute,
    circuit_record_failure,
    circuit_record_success,
    clear_event_queue,
    compute_instance_id,
    create_trace_event,
    enqueue_event,
    estimate_tokens,
    flush_event_queue,
    load_queued_events,
    report_violation,
    resolve_agent_id,
    truncate_output,
)
from clyro.hooks.constants import MEMORY_FALLBACK_MAX_EVENTS, OUTPUT_TRUNCATE_BYTES
from clyro.hooks.models import CircuitBreakerSnapshot, SessionState


# ── Agent Registration ────────────────────────────────────────────────────


class TestComputeInstanceId:
    def test_deterministic(self):
        id1 = compute_instance_id("my-agent")
        id2 = compute_instance_id("my-agent")
        assert id1 == id2

    def test_different_names_produce_different_ids(self):
        id1 = compute_instance_id("agent-a")
        id2 = compute_instance_id("agent-b")
        assert id1 != id2

    def test_returns_12_char_hex(self):
        result = compute_instance_id("test")
        assert len(result) == 12
        int(result, 16)  # Should not raise


class TestAgentIdPersistence:
    def test_persist_and_load(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            test_id = "550e8400-e29b-41d4-a716-446655440000"
            instance_id = "abc123"
            _persist_agent_id(instance_id, test_id, confirmed=True)
            loaded = _load_persisted_agent_id(instance_id)
            assert loaded == test_id

    def test_persist_unconfirmed_creates_marker(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            test_id = "550e8400-e29b-41d4-a716-446655440000"
            instance_id = "abc123"
            _persist_agent_id(instance_id, test_id, confirmed=False)
            path = _agent_id_path(instance_id)
            marker = _unconfirmed_path(path)
            assert marker.exists()

    def test_persist_confirmed_removes_marker(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            test_id = "550e8400-e29b-41d4-a716-446655440000"
            instance_id = "abc123"
            _persist_agent_id(instance_id, test_id, confirmed=False)
            _persist_agent_id(instance_id, test_id, confirmed=True)
            path = _agent_id_path(instance_id)
            marker = _unconfirmed_path(path)
            assert not marker.exists()

    def test_load_nonexistent_returns_none(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            assert _load_persisted_agent_id("nonexistent") is None

    def test_load_invalid_uuid_returns_none(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            path = tmp_path / "agents" / "hook-agent-bad.id"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not-a-uuid")
            assert _load_persisted_agent_id("bad") is None


class TestResolveAgentId:
    def test_no_api_key_returns_none(self):
        config = MagicMock()
        config.api_key = None
        state = SessionState(session_id="s1")
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_agent_id(config, state)
        assert result is None
        assert state.agent_id is None

    def test_returns_existing_state_agent_id(self):
        config = MagicMock()
        config.api_key = "test-key"
        config.agent_name = "test-agent"
        state = SessionState(session_id="s1", agent_id="existing-id")
        result = resolve_agent_id(config, state)
        assert result == "existing-id"

    def test_loads_confirmed_persisted_id(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"):
            config = MagicMock()
            config.api_key = "test-key"
            config.agent_name = "test-agent"
            state = SessionState(session_id="s1")

            instance_id = compute_instance_id("test-agent")
            test_id = "550e8400-e29b-41d4-a716-446655440000"
            _persist_agent_id(instance_id, test_id, confirmed=True)

            result = resolve_agent_id(config, state)
            assert result == test_id
            assert state.agent_id == test_id

    def test_backend_registration_fallback_to_local_uuid(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"), \
             patch("clyro.hooks.backend.HttpSyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.register_agent = AsyncMock(side_effect=Exception("network error"))
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            config = MagicMock()
            config.api_key = "test-key"
            config.agent_name = "test-agent"
            config.api_url = "https://api.test.dev"
            state = SessionState(session_id="s1")

            result = resolve_agent_id(config, state)

            # Should get a local UUID
            assert result is not None
            UUID(result)  # Valid UUID format
            assert state.agent_id == result

    def test_env_var_api_key_fallback(self, tmp_path):
        with patch("clyro.hooks.backend.AGENT_ID_DIR", tmp_path / "agents"), \
             patch.dict("os.environ", {"CLYRO_API_KEY": "env-key"}), \
             patch("clyro.hooks.backend.HttpSyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.register_agent = AsyncMock(side_effect=Exception("fail"))
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            config = MagicMock()
            config.api_key = None
            config.agent_name = "test-agent"
            config.api_url = "https://api.test.dev"
            state = SessionState(session_id="s1")

            result = resolve_agent_id(config, state)
            assert result is not None  # Should use env key and fallback to local UUID


# ── Circuit Breaker ───────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_closed_allows_execution(self):
        snapshot = CircuitBreakerSnapshot(state="closed")
        assert circuit_can_execute(snapshot) is True

    def test_open_denies_execution(self):
        snapshot = CircuitBreakerSnapshot(state="open", opened_at=time.monotonic())
        assert circuit_can_execute(snapshot) is False

    def test_open_transitions_to_half_open_after_timeout(self):
        snapshot = CircuitBreakerSnapshot(
            state="open",
            opened_at=time.monotonic() - CIRCUIT_OPEN_TIMEOUT_SECONDS - 1,
        )
        assert circuit_can_execute(snapshot) is True
        assert snapshot.state == "half_open"

    def test_half_open_allows_probe(self):
        snapshot = CircuitBreakerSnapshot(state="half_open")
        assert circuit_can_execute(snapshot) is True

    def test_success_in_half_open_counts(self):
        snapshot = CircuitBreakerSnapshot(state="half_open", half_open_successes=0)
        circuit_record_success(snapshot)
        assert snapshot.half_open_successes == 1

    def test_enough_successes_close_circuit(self):
        snapshot = CircuitBreakerSnapshot(
            state="half_open",
            half_open_successes=CIRCUIT_SUCCESS_THRESHOLD - 1,
        )
        circuit_record_success(snapshot)
        assert snapshot.state == "closed"
        assert snapshot.failure_count == 0

    def test_failure_in_half_open_reopens(self):
        snapshot = CircuitBreakerSnapshot(state="half_open")
        circuit_record_failure(snapshot)
        assert snapshot.state == "open"
        assert snapshot.opened_at is not None

    def test_failures_trip_circuit(self):
        snapshot = CircuitBreakerSnapshot(state="closed", failure_count=0)
        for _ in range(CIRCUIT_FAILURE_THRESHOLD):
            circuit_record_failure(snapshot)
        assert snapshot.state == "open"
        assert snapshot.total_trips == 1

    def test_success_resets_failure_count(self):
        snapshot = CircuitBreakerSnapshot(state="closed", failure_count=3)
        circuit_record_success(snapshot)
        assert snapshot.failure_count == 0


# ── Event Queue ───────────────────────────────────────────────────────────


class TestEventQueue:
    def test_enqueue_and_load(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            event1 = {"event_type": "tool_call_observe", "tool_name": "Bash"}
            event2 = {"event_type": "session_end", "session_id": "s1"}
            enqueue_event("s1", event1)
            enqueue_event("s1", event2)

            events = load_queued_events("s1")
            assert len(events) == 2
            assert events[0]["event_type"] == "tool_call_observe"
            assert events[1]["event_type"] == "session_end"

    def test_load_empty_queue(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            events = load_queued_events("nonexistent")
            assert events == []

    def test_clear_queue(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            enqueue_event("s1", {"test": True})
            clear_event_queue("s1")
            events = load_queued_events("s1")
            assert events == []

    def test_sanitizes_session_id(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            path = _event_queue_path("../../etc/passwd")
            # Should not contain path traversal
            assert ".." not in path.name

    def test_empty_session_id_uses_unknown(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            path = _event_queue_path("")
            assert "unknown" in path.name


# ── Flush Event Queue ─────────────────────────────────────────────────────


class TestFlushEventQueue:
    def test_skips_when_circuit_open(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            enqueue_event("s1", {"test": True})
            circuit = CircuitBreakerSnapshot(state="open", opened_at=time.monotonic())
            flush_event_queue("s1", "key", "https://api.test.dev", circuit)
            # Events should still be in queue (not flushed)
            events = load_queued_events("s1")
            assert len(events) == 1

    def test_skips_when_no_events(self):
        circuit = CircuitBreakerSnapshot(state="closed")
        with patch("clyro.hooks.backend.load_queued_events", return_value=[]):
            # Should not raise
            flush_event_queue("s1", "key", "https://api.test.dev", circuit)

    def test_successful_flush_clears_queue(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"), \
             patch("clyro.hooks.backend.HttpSyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.send_batch = AsyncMock(return_value={"accepted": 2})
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            enqueue_event("s1", {"test": True})
            circuit = CircuitBreakerSnapshot(state="closed")
            flush_event_queue("s1", "key", "https://api.test.dev", circuit)
            events = load_queued_events("s1")
            assert events == []

    def test_failed_flush_records_failure(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"), \
             patch("clyro.hooks.backend.HttpSyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.send_batch = AsyncMock(side_effect=Exception("network"))
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            enqueue_event("s1", {"test": True})
            circuit = CircuitBreakerSnapshot(state="closed")
            flush_event_queue("s1", "key", "https://api.test.dev", circuit)
            assert circuit.failure_count == 1
            # Events remain in queue
            events = load_queued_events("s1")
            assert len(events) == 1


# ── Violation Reporting ───────────────────────────────────────────────────


class TestReportViolation:
    def test_skips_when_circuit_open(self):
        circuit = CircuitBreakerSnapshot(state="open", opened_at=time.monotonic())
        # Should not raise or make any calls
        report_violation(
            api_key="key", api_url="https://api.test.dev",
            agent_id="a1", session_id="s1", tool_name="Bash",
            reason="test", rule_results=None, circuit=circuit,
        )

    def test_successful_report_records_success(self):
        circuit = CircuitBreakerSnapshot(state="closed")
        with patch("clyro.hooks.backend.HttpSyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.report_violations = AsyncMock()
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            report_violation(
                api_key="key", api_url="https://api.test.dev",
                agent_id="a1", session_id="s1", tool_name="Bash",
                reason="test violation", rule_results=[{"rule": "r1"}],
                circuit=circuit,
            )
        assert circuit.failure_count == 0

    def test_failed_report_records_failure(self):
        circuit = CircuitBreakerSnapshot(state="closed")
        with patch("clyro.hooks.backend.HttpSyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.report_violations = AsyncMock(side_effect=Exception("fail"))
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            report_violation(
                api_key="key", api_url="https://api.test.dev",
                agent_id="a1", session_id="s1", tool_name="Bash",
                reason="test", rule_results=None, circuit=circuit,
            )
        assert circuit.failure_count == 1

    def test_rich_violation_payload(self):
        """FRD-HK-006: Violation report includes policy_id, operator, parameters_hash."""
        circuit = CircuitBreakerSnapshot(state="closed")
        violation_details = {
            "policy_id": "pol-123",
            "rule_name": "block_sudo",
            "operator": "contains",
            "expected": "sudo",
            "actual": "sudo rm -rf /",
        }
        tool_input = {"command": "sudo rm -rf /"}

        with patch("clyro.hooks.backend.HttpSyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.report_violations = AsyncMock()
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client
            report_violation(
                api_key="key", api_url="https://api.test.dev",
                agent_id="a1", session_id="s1", tool_name="Bash",
                reason="Policy violation: block_sudo",
                rule_results=[{"rule": "block_sudo"}],
                circuit=circuit,
                violation_details=violation_details,
                tool_input=tool_input,
                step_number=5,
            )

        # Verify the payload passed to report_violations
        mock_client.report_violations.assert_called_once()
        violations = mock_client.report_violations.call_args[0][0]
        v = violations[0]
        assert v["policy_id"] == "pol-123"
        assert v["rule_id"] == "block_sudo"
        assert v["operator"] == "contains"
        assert v["parameters_hash"]  # SHA-256 hash present
        assert v["step_number"] == 5
        assert v["decision"] == "block"


# ── Trace Event Factory ──────────────────────────────────────────────────


class TestCreateTraceEvent:
    def test_basic_event_structure(self):
        event = create_trace_event("policy_check", "s1", agent_id="a1")
        assert event["event_type"] == "policy_check"
        assert event["session_id"] == "s1"
        assert event["agent_id"] == "a1"
        # UUID event_id
        UUID(event["event_id"])
        assert event["framework"] == "claude_code_hooks"
        assert event["metadata"]["_source"] == "claude_code_hooks"
        assert event["metadata"]["cost_estimated"] is True

    def test_event_id_is_unique(self):
        e1 = create_trace_event("test", "s1")
        e2 = create_trace_event("test", "s1")
        assert e1["event_id"] != e2["event_id"]

    def test_parent_event_id_wiring(self):
        parent = create_trace_event("policy_check", "s1")
        child = create_trace_event(
            "error", "s1", parent_event_id=parent["event_id"]
        )
        assert child["parent_event_id"] == parent["event_id"]

    def test_output_truncation(self):
        large_output = {"data": "x" * (OUTPUT_TRUNCATE_BYTES + 1000)}
        event = create_trace_event("test", "s1", output_data=large_output)
        assert event["metadata"].get("output_truncated") is True
        assert "_truncated" in event["output_data"]

    def test_small_output_not_truncated(self):
        small_output = {"data": "hello"}
        event = create_trace_event("test", "s1", output_data=small_output)
        assert event["output_data"]["data"] == "hello"
        assert "output_truncated" not in event["metadata"]

    def test_token_counts_preserved(self):
        event = create_trace_event(
            "tool_call_observe", "s1",
            token_count_input=100, token_count_output=50,
        )
        assert event["token_count_input"] == 100
        assert event["token_count_output"] == 50

    def test_error_fields(self):
        event = create_trace_event(
            "error", "s1",
            error_type="policy_violation",
            error_message="Step limit exceeded",
        )
        assert event["error_type"] == "policy_violation"
        assert event["error_message"] == "Step limit exceeded"

    def test_metadata_merged(self):
        event = create_trace_event(
            "test", "s1",
            metadata={"decision": "block", "custom": True},
        )
        assert event["metadata"]["decision"] == "block"
        assert event["metadata"]["custom"] is True
        assert event["metadata"]["_source"] == "claude_code_hooks"

    def test_step_number_and_cost(self):
        event = create_trace_event(
            "test", "s1",
            step_number=10, accumulated_cost_usd=1.5,
        )
        assert event["step_number"] == 10
        assert event["cumulative_cost"] == 1.5


class TestTruncateOutput:
    def test_none_returns_none(self):
        assert truncate_output(None) is None

    def test_small_data_unchanged(self):
        data = {"key": "value"}
        assert truncate_output(data) == data

    def test_large_data_truncated(self):
        data = {"key": "x" * (OUTPUT_TRUNCATE_BYTES + 500)}
        result = truncate_output(data)
        assert "_truncated" in result


class TestEstimateTokens:
    def test_basic_estimation(self):
        assert estimate_tokens(400) == 100

    def test_zero(self):
        assert estimate_tokens(0) == 0

    def test_small_text(self):
        assert estimate_tokens(3) == 0  # Floor division


# ── Memory Fallback ──────────────────────────────────────────────────────


class TestMemoryFallback:
    def setup_method(self):
        _memory_fallback.clear()

    def test_fallback_on_file_io_failure(self, tmp_path):
        """Events fall back to memory when file I/O fails."""
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"), \
             patch("os.open", side_effect=OSError("mock disk full")):
            enqueue_event("s1", {"test": True})
            assert "s1" in _memory_fallback
            assert len(_memory_fallback["s1"]) == 1

    def test_memory_fallback_included_in_load(self, tmp_path):
        """load_queued_events includes memory fallback events."""
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            _memory_fallback["s1"] = [{"memory": True}]
            events = load_queued_events("s1")
            assert any(e.get("memory") for e in events)

    def test_clear_removes_memory_fallback(self, tmp_path):
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"):
            _memory_fallback["s1"] = [{"memory": True}]
            clear_event_queue("s1")
            assert "s1" not in _memory_fallback

    def test_memory_fallback_respects_limit(self, tmp_path):
        """Memory fallback caps at MEMORY_FALLBACK_MAX_EVENTS."""
        with patch("clyro.hooks.backend.EVENT_QUEUE_DIR", tmp_path / "pending"), \
             patch("os.open", side_effect=OSError("mock disk full")):
            for i in range(MEMORY_FALLBACK_MAX_EVENTS + 10):
                enqueue_event("s1", {"i": i})
            assert len(_memory_fallback["s1"]) == MEMORY_FALLBACK_MAX_EVENTS
