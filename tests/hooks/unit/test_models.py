"""Unit tests for data models."""

from datetime import datetime, timezone

from clyro.hooks.models import HookInput, HookOutput, PolicyCache, SessionState


class TestHookInput:
    def test_minimal_input(self):
        inp = HookInput(session_id="s1", tool_name="Bash")
        assert inp.session_id == "s1"
        assert inp.tool_name == "Bash"
        assert inp.tool_input == {}
        assert inp.tool_result is None

    def test_full_input(self):
        inp = HookInput(
            session_id="s1",
            tool_name="Bash",
            tool_input={"command": "ls"},
            tool_result={"stdout": "file.txt", "exitCode": 0},
        )
        assert inp.tool_input["command"] == "ls"
        assert inp.tool_result["exitCode"] == 0

    def test_extra_fields_ignored(self):
        inp = HookInput(session_id="s1", tool_name="Bash", unknown_field="ignored")
        assert not hasattr(inp, "unknown_field")

    def test_missing_tool_input_defaults_to_empty(self):
        inp = HookInput(session_id="s1")
        assert inp.tool_input == {}
        assert inp.tool_name == ""


class TestHookOutput:
    def test_block_output(self):
        out = HookOutput(decision="block", reason="Policy violation")
        assert out.decision == "block"
        assert out.reason == "Policy violation"

    def test_allow_output(self):
        out = HookOutput(decision="allow")
        assert out.decision == "allow"
        assert out.reason is None


class TestSessionState:
    def test_default_state(self):
        state = SessionState(session_id="s1")
        assert state.step_count == 0
        assert state.accumulated_cost_usd == 0.0
        assert state.loop_history == []
        assert state.cloud_disabled is False
        assert isinstance(state.started_at, datetime)

    def test_roundtrip_serialization(self):
        state = SessionState(session_id="s1", step_count=5, accumulated_cost_usd=0.01)
        data = state.model_dump(mode="json")
        restored = SessionState.model_validate(data)
        assert restored.session_id == "s1"
        assert restored.step_count == 5
        assert restored.accumulated_cost_usd == 0.01


class TestCircuitBreakerSnapshot:
    def test_default_state(self):
        from clyro.hooks.models import CircuitBreakerSnapshot
        cb = CircuitBreakerSnapshot()
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.half_open_successes == 0
        assert cb.opened_at is None
        assert cb.total_trips == 0

    def test_roundtrip_serialization(self):
        from clyro.hooks.models import CircuitBreakerSnapshot
        cb = CircuitBreakerSnapshot(state="open", failure_count=3, opened_at=12345.0, total_trips=1)
        data = cb.model_dump(mode="json")
        restored = CircuitBreakerSnapshot.model_validate(data)
        assert restored.state == "open"
        assert restored.failure_count == 3
        assert restored.total_trips == 1


class TestSessionStateAgentId:
    def test_agent_id_defaults_to_none(self):
        state = SessionState(session_id="s1")
        assert state.agent_id is None

    def test_agent_id_persists_in_serialization(self):
        state = SessionState(session_id="s1", agent_id="test-agent-123")
        data = state.model_dump(mode="json")
        restored = SessionState.model_validate(data)
        assert restored.agent_id == "test-agent-123"

    def test_circuit_breaker_in_state(self):
        from clyro.hooks.models import CircuitBreakerSnapshot
        state = SessionState(
            session_id="s1",
            circuit_breaker=CircuitBreakerSnapshot(state="open", failure_count=5),
        )
        data = state.model_dump(mode="json")
        restored = SessionState.model_validate(data)
        assert restored.circuit_breaker.state == "open"
        assert restored.circuit_breaker.failure_count == 5


class TestPolicyCache:
    def test_default_cache(self):
        cache = PolicyCache()
        assert cache.fetched_at is None
        assert cache.ttl_seconds == 300
        assert cache.merged_policies == []
