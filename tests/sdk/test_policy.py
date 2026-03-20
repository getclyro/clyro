# Tests for Clyro SDK Policy Enforcement
# Implements PRD-011, PRD-012

"""Unit tests for policy client, evaluator, and session integration."""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.exceptions import PolicyViolationError
from clyro.policy import ConsoleApprovalHandler, PolicyClient, PolicyDecision, PolicyEvaluator
from clyro.session import Session
from clyro.trace import EventType, Framework


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def policy_config(temp_dir):
    """Config with policy enforcement enabled."""
    return ClyroConfig(
        api_key="cly_test_key",
        endpoint="https://api.example.com",
        local_storage_path=str(temp_dir / "traces.db"),
        controls=ExecutionControls(enable_policy_enforcement=True),
    )


@pytest.fixture
def policy_disabled_config(temp_dir):
    """Config with policy enforcement disabled (default)."""
    return ClyroConfig(
        api_key="cly_test_key",
        endpoint="https://api.example.com",
        local_storage_path=str(temp_dir / "traces.db"),
    )


@pytest.fixture
def local_only_config(temp_dir):
    """Config without API key (local-only mode)."""
    return ClyroConfig(
        local_storage_path=str(temp_dir / "traces.db"),
        controls=ExecutionControls(enable_policy_enforcement=True),
    )


@pytest.fixture
def agent_id():
    return uuid4()


@pytest.fixture
def org_id():
    return uuid4()


@pytest.fixture
def session_id():
    return uuid4()


@pytest.fixture
def allow_response():
    """Backend response for an allowed action."""
    return {
        "decision": "allow",
        "evaluated_rules": 5,
        "evaluation_time_ms": 2.1,
    }


@pytest.fixture
def block_response():
    """Backend response for a blocked action."""
    return {
        "decision": "block",
        "rule_id": "rule-001",
        "rule_name": "max_refund",
        "message": "Refund amount $150 exceeds maximum of $100",
        "evaluated_rules": 3,
        "evaluation_time_ms": 1.8,
    }


@pytest.fixture
def approval_response():
    """Backend response for an action requiring approval."""
    return {
        "decision": "require_approval",
        "rule_id": "rule-002",
        "rule_name": "high_value_transaction",
        "message": "Transaction requires manager approval",
        "evaluated_rules": 4,
        "evaluation_time_ms": 2.5,
    }


# =============================================================================
# PolicyDecision Tests
# =============================================================================


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_allow_decision(self, allow_response):
        """Test creating an allow decision from response."""
        decision = PolicyDecision.from_response(allow_response)

        assert decision.decision == "allow"
        assert decision.is_allowed is True
        assert decision.is_blocked is False
        assert decision.requires_approval is False
        assert decision.rule_id is None
        assert decision.evaluated_rules == 5
        assert decision.evaluation_time_ms == 2.1

    def test_block_decision(self, block_response):
        """Test creating a block decision from response."""
        decision = PolicyDecision.from_response(block_response)

        assert decision.decision == "block"
        assert decision.is_allowed is False
        assert decision.is_blocked is True
        assert decision.requires_approval is False
        assert decision.rule_id == "rule-001"
        assert decision.rule_name == "max_refund"
        assert decision.message == "Refund amount $150 exceeds maximum of $100"

    def test_approval_decision(self, approval_response):
        """Test creating a require_approval decision from response."""
        decision = PolicyDecision.from_response(approval_response)

        assert decision.decision == "require_approval"
        assert decision.is_allowed is False
        assert decision.is_blocked is False
        assert decision.requires_approval is True
        assert decision.rule_id == "rule-002"

    def test_allow_factory(self):
        """Test the static allow() factory method."""
        decision = PolicyDecision.allow()

        assert decision.is_allowed is True
        assert decision.rule_id is None
        assert decision.evaluated_rules == 0

    def test_frozen_dataclass(self, allow_response):
        """Test that PolicyDecision is immutable."""
        decision = PolicyDecision.from_response(allow_response)
        with pytest.raises(AttributeError):
            decision.decision = "block"

    def test_from_response_defaults(self):
        """Test from_response with minimal response data."""
        decision = PolicyDecision.from_response({})

        assert decision.decision == "block"  # defaults to block (fail-closed)
        assert decision.rule_id is None
        assert decision.evaluated_rules == 0


# =============================================================================
# PolicyClient Tests
# =============================================================================


class TestPolicyClient:
    """Tests for PolicyClient HTTP client."""

    def test_build_payload_basic(self, policy_config, agent_id):
        """Test building a basic evaluation payload."""
        client = PolicyClient(policy_config)
        payload = client._build_payload(
            agent_id=agent_id,
            action_type="tool_call",
            parameters={"tool_name": "get_queues"},
        )

        assert payload["agent_id"] == str(agent_id)
        assert payload["action"]["type"] == "tool_call"
        assert payload["action"]["parameters"]["tool_name"] == "get_queues"
        assert "context" not in payload

    def test_build_payload_with_context(self, policy_config, agent_id, session_id):
        """Test building payload with session context."""
        client = PolicyClient(policy_config)
        payload = client._build_payload(
            agent_id=agent_id,
            action_type="tool_call",
            parameters={"tool_name": "get_queues"},
            session_id=session_id,
            step_number=3,
        )

        assert payload["context"]["session_id"] == str(session_id)
        assert payload["context"]["step_number"] == 3

    def test_headers_include_api_key(self, policy_config):
        """Test that headers include the API key."""
        client = PolicyClient(policy_config)
        headers = client._get_headers()

        assert headers["X-Clyro-API-Key"] == "cly_test_key"
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self, local_only_config):
        """Test headers when no API key is configured."""
        client = PolicyClient(local_only_config)
        headers = client._get_headers()

        assert "X-Clyro-API-Key" not in headers

    def test_evaluate_sync_success(self, policy_config, agent_id, allow_response):
        """Test successful synchronous evaluation."""
        client = PolicyClient(policy_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = allow_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.Client, "post", return_value=mock_response):
            decision = client.evaluate_sync(
                agent_id=agent_id,
                action_type="tool_call",
                parameters={"tool_name": "get_queues"},
            )

        assert decision.is_allowed is True
        assert decision.evaluated_rules == 5

    def test_evaluate_sync_network_error(self, policy_config, agent_id):
        """Test synchronous evaluation with network error."""
        client = PolicyClient(policy_config)

        with patch.object(
            httpx.Client, "post", side_effect=httpx.ConnectError("Connection refused")
        ):
            with pytest.raises(httpx.ConnectError):
                client.evaluate_sync(
                    agent_id=agent_id,
                    action_type="tool_call",
                    parameters={},
                )

    @pytest.mark.asyncio
    async def test_evaluate_async_success(self, policy_config, agent_id, allow_response):
        """Test successful asynchronous evaluation."""
        client = PolicyClient(policy_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = allow_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            decision = await client.evaluate_async(
                agent_id=agent_id,
                action_type="tool_call",
                parameters={"tool_name": "get_queues"},
            )

        assert decision.is_allowed is True

    @pytest.mark.asyncio
    async def test_evaluate_async_network_error(self, policy_config, agent_id):
        """Test asynchronous evaluation with network error."""
        client = PolicyClient(policy_config)

        with patch.object(
            httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Connection refused")
        ):
            with pytest.raises(httpx.ConnectError):
                await client.evaluate_async(
                    agent_id=agent_id,
                    action_type="tool_call",
                    parameters={},
                )


# =============================================================================
# PolicyEvaluator Tests
# =============================================================================


class TestPolicyEvaluator:
    """Tests for PolicyEvaluator enforcement logic."""

    def test_is_enabled_when_configured(self, policy_config, agent_id):
        """Test that evaluator is enabled when config flag is set."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        assert evaluator.is_enabled is True

    def test_is_disabled_by_default(self, policy_disabled_config, agent_id):
        """Test that evaluator is disabled by default."""
        evaluator = PolicyEvaluator(policy_disabled_config, agent_id)
        assert evaluator.is_enabled is False

    def test_is_disabled_without_api_key(self, local_only_config, agent_id):
        """Test that evaluator is disabled without API key."""
        evaluator = PolicyEvaluator(local_only_config, agent_id)
        assert evaluator.is_enabled is False

    def test_evaluate_sync_allow(self, policy_config, agent_id, allow_response):
        """Test that allowed actions pass through silently."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True

    def test_evaluate_sync_block_raises(self, policy_config, agent_id, block_response):
        """Test that blocked actions raise PolicyViolationError."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                evaluator.evaluate_sync("tool_call", {"tool_name": "process_refund"})

        assert exc_info.value.rule_id == "rule-001"
        assert exc_info.value.rule_name == "max_refund"
        assert "exceeds maximum" in exc_info.value.message
        assert exc_info.value.action_type == "tool_call"

    def test_evaluate_sync_require_approval_raises_without_handler(
        self, policy_config, agent_id, approval_response
    ):
        """Test that require_approval raises when no approval_handler is configured."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(approval_response),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                evaluator.evaluate_sync("tool_call", {"tool_name": "process_refund"})

        assert exc_info.value.rule_id == "rule-002"
        assert exc_info.value.details.get("decision") == "require_approval"

    def test_evaluate_sync_require_approval_calls_handler(
        self, policy_config, agent_id, approval_response
    ):
        """Test that require_approval calls approval_handler when configured."""
        handler = MagicMock(return_value=True)
        evaluator = PolicyEvaluator(policy_config, agent_id, approval_handler=handler)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(approval_response),
        ):
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "query_site"})

        handler.assert_called_once()
        call_args = handler.call_args
        assert call_args[0][0].decision == "require_approval"
        assert call_args[0][1] == "tool_call"
        assert decision.requires_approval is True  # decision is returned as-is

    def test_evaluate_sync_require_approval_handler_denies(
        self, policy_config, agent_id, approval_response
    ):
        """Test that denied approval raises PolicyViolationError."""
        handler = MagicMock(return_value=False)
        evaluator = PolicyEvaluator(policy_config, agent_id, approval_handler=handler)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(approval_response),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                evaluator.evaluate_sync("tool_call", {"tool_name": "query_site"})

        handler.assert_called_once()
        assert exc_info.value.rule_id == "rule-002"
        assert exc_info.value.details.get("decision") == "require_approval"

    def test_evaluate_sync_require_approval_handler_error_denies(
        self, policy_config, agent_id, approval_response
    ):
        """Test that handler errors are treated as denial (safe default)."""
        handler = MagicMock(side_effect=RuntimeError("handler crashed"))
        evaluator = PolicyEvaluator(policy_config, agent_id, approval_handler=handler)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(approval_response),
        ):
            with pytest.raises(PolicyViolationError):
                evaluator.evaluate_sync("tool_call", {"tool_name": "query_site"})

        handler.assert_called_once()

    def test_block_decision_ignores_approval_handler(
        self, policy_config, agent_id, block_response
    ):
        """Test that block decisions always raise, even with approval_handler."""
        handler = MagicMock(return_value=True)
        evaluator = PolicyEvaluator(policy_config, agent_id, approval_handler=handler)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError):
                evaluator.evaluate_sync("tool_call", {"tool_name": "process_refund"})

        # Handler should NOT be called for block decisions
        handler.assert_not_called()

    def test_evaluate_sync_skips_when_disabled(self, policy_disabled_config, agent_id):
        """Test that evaluation is skipped when disabled."""
        evaluator = PolicyEvaluator(policy_disabled_config, agent_id)

        # Should not call the client at all
        with patch.object(evaluator._client, "evaluate_sync") as mock_eval:
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "anything"})

        mock_eval.assert_not_called()
        assert decision.is_allowed is True

    def test_evaluate_sync_fail_open(self, policy_config, agent_id):
        """Test fail-open: network errors allow action when fail_open=True."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True

    def test_evaluate_sync_fail_closed(self, temp_dir, agent_id):
        """Test fail-closed: network errors block action when fail_open=False."""
        config = ClyroConfig(
            api_key="cly_test_key",
            endpoint="https://api.example.com",
            local_storage_path=str(temp_dir / "traces.db"),
            controls=ExecutionControls(enable_policy_enforcement=True),
            fail_open=False,
        )
        evaluator = PolicyEvaluator(config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert exc_info.value.rule_id == "system_error"
        assert "fail_open=False" in exc_info.value.message

    def test_evaluate_sync_403_disables_enforcement(self, policy_config, agent_id):
        """Test that HTTP 403 auto-disables enforcement and allows through."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        assert evaluator.is_enabled is True

        mock_response = MagicMock()
        mock_response.status_code = 403
        error = httpx.HTTPStatusError(
            "Forbidden", request=MagicMock(), response=mock_response,
        )

        with patch.object(
            evaluator._client, "evaluate_sync", side_effect=error,
        ):
            # First call: hits 403, disables enforcement, allows through
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True
        assert evaluator.is_enabled is False
        assert evaluator._disabled_reason == "HTTP 403"

        # Subsequent calls skip evaluation entirely (no more HTTP calls)
        decision2 = evaluator.evaluate_sync("tool_call", {"tool_name": "anything"})
        assert decision2.is_allowed is True

    def test_evaluate_sync_401_disables_enforcement(self, policy_config, agent_id):
        """Test that HTTP 401 auto-disables enforcement and allows through."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        mock_response = MagicMock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_response,
        )

        with patch.object(
            evaluator._client, "evaluate_sync", side_effect=error,
        ):
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True
        assert evaluator.is_enabled is False
        assert evaluator._disabled_reason == "HTTP 401"

    def test_evaluate_sync_500_respects_fail_open(self, policy_config, agent_id):
        """Test that HTTP 500 respects fail_open (not an auth error)."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response,
        )

        with patch.object(
            evaluator._client, "evaluate_sync", side_effect=error,
        ):
            decision = evaluator.evaluate_sync("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True
        # 500 does NOT disable enforcement — it's transient
        assert evaluator.is_enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_async_allow(self, policy_config, agent_id, allow_response):
        """Test async allowed actions pass through."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_async",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            decision = await evaluator.evaluate_async("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True

    @pytest.mark.asyncio
    async def test_evaluate_async_block_raises(
        self, policy_config, agent_id, block_response
    ):
        """Test async blocked actions raise PolicyViolationError."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_async",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                await evaluator.evaluate_async("tool_call", {"tool_name": "process_refund"})

        assert exc_info.value.rule_id == "rule-001"

    @pytest.mark.asyncio
    async def test_evaluate_async_fail_open(self, policy_config, agent_id):
        """Test async fail-open on network error."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_async",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            decision = await evaluator.evaluate_async("tool_call", {"tool_name": "get_queues"})

        assert decision.is_allowed is True


class TestPolicyEvaluatorEventCreation:
    """Tests for POLICY_CHECK event creation."""

    def test_create_policy_check_event_allow(self, policy_config, agent_id, session_id):
        """Test creating an audit event for allowed action."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        decision = PolicyDecision.allow()

        event = evaluator.create_policy_check_event(
            decision=decision,
            action_type="tool_call",
            parameters={"tool_name": "get_queues"},
            session_id=session_id,
            step_number=3,
        )

        assert event.event_type == EventType.POLICY_CHECK
        assert event.agent_id == agent_id
        assert event.session_id == session_id
        assert event.step_number == 3
        assert event.metadata["decision"] == "allow"
        assert event.metadata["action_type"] == "tool_call"
        assert event.metadata["parameters"]["tool_name"] == "get_queues"

    def test_create_policy_check_event_block(self, policy_config, agent_id, block_response):
        """Test creating an audit event for blocked action."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        decision = PolicyDecision.from_response(block_response)

        event = evaluator.create_policy_check_event(
            decision=decision,
            action_type="tool_call",
            parameters={"tool_name": "process_refund"},
        )

        assert event.metadata["decision"] == "block"
        assert event.metadata["rule_id"] == "rule-001"
        assert event.metadata["rule_name"] == "max_refund"

    def test_evaluate_emits_audit_event_on_allow(
        self, policy_config, agent_id, allow_response
    ):
        """Test that evaluate_sync buffers a POLICY_CHECK event for allowed actions."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            evaluator.evaluate_sync(
                "tool_call", {"tool_name": "get_queues"},
                session_id=uuid4(), step_number=1,
            )

        events = evaluator.drain_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.POLICY_CHECK
        assert events[0].metadata["decision"] == "allow"

    def test_evaluate_emits_audit_event_on_block(
        self, policy_config, agent_id, block_response
    ):
        """Test that evaluate_sync buffers a POLICY_CHECK event before raising."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError):
                evaluator.evaluate_sync(
                    "tool_call", {"tool_name": "process_refund"},
                    session_id=uuid4(), step_number=2,
                )

        # Event should still be buffered even though PolicyViolationError was raised
        events = evaluator.drain_events()
        assert len(events) == 1
        assert events[0].metadata["decision"] == "block"
        assert events[0].metadata["rule_id"] == "rule-001"

    def test_drain_events_clears_buffer(self, policy_config, agent_id, allow_response):
        """Test that drain_events returns and clears buffered events."""
        evaluator = PolicyEvaluator(policy_config, agent_id)

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            evaluator.evaluate_sync("tool_call", {"tool_name": "a"})
            evaluator.evaluate_sync("llm_call", {"model": "gpt-4"})

        events = evaluator.drain_events()
        assert len(events) == 2

        # Second drain should be empty
        assert len(evaluator.drain_events()) == 0


# =============================================================================
# Session Integration Tests
# =============================================================================


class TestSessionPolicyIntegration:
    """Tests for Session.check_policy integration."""

    def test_check_policy_with_evaluator(self, policy_config, agent_id, allow_response):
        """Test Session.check_policy delegates to evaluator."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            # Should not raise
            session.check_policy("tool_call", {"tool_name": "get_queues"})

    def test_check_policy_block_raises(self, policy_config, agent_id, block_response):
        """Test Session.check_policy raises on block."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError):
                session.check_policy("tool_call", {"tool_name": "process_refund"})

    def test_check_policy_noop_without_evaluator(self, policy_config, agent_id):
        """Test Session.check_policy is a no-op without evaluator."""
        session = Session(
            config=policy_config,
            agent_id=agent_id,
        )

        # Should not raise (no evaluator configured)
        session.check_policy("tool_call", {"tool_name": "anything"})

    def test_check_policy_noop_with_none_params(self, policy_config, agent_id, allow_response):
        """Test Session.check_policy handles None parameters."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            session.check_policy("agent_execution")

    def test_check_policy_records_audit_event(self, policy_config, agent_id, allow_response):
        """Test Session.check_policy records POLICY_CHECK event in session."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            session.check_policy("tool_call", {"tool_name": "get_queues"})

        # Session should have recorded the POLICY_CHECK event
        policy_events = [
            e for e in session.events
            if e.event_type == EventType.POLICY_CHECK
        ]
        assert len(policy_events) == 1
        assert policy_events[0].metadata["decision"] == "allow"

    def test_check_policy_records_audit_event_on_block(
        self, policy_config, agent_id, block_response
    ):
        """Test Session.check_policy records POLICY_CHECK even when block raises."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError):
                session.check_policy("tool_call", {"tool_name": "process_refund"})

        # Audit event should still be recorded despite the exception
        policy_events = [
            e for e in session.events
            if e.event_type == EventType.POLICY_CHECK
        ]
        assert len(policy_events) == 1
        assert policy_events[0].metadata["decision"] == "block"

    @pytest.mark.asyncio
    async def test_check_policy_async(self, policy_config, agent_id, allow_response):
        """Test Session.check_policy_async delegates to evaluator."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_async",
            return_value=PolicyDecision.from_response(allow_response),
        ):
            await session.check_policy_async("tool_call", {"tool_name": "get_queues"})

    @pytest.mark.asyncio
    async def test_check_policy_async_block_raises(
        self, policy_config, agent_id, block_response
    ):
        """Test Session.check_policy_async raises on block."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )

        with patch.object(
            evaluator._client, "evaluate_async",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError):
                await session.check_policy_async("tool_call", {"tool_name": "process_refund"})


# =============================================================================
# Config Tests
# =============================================================================


class TestPolicyConfig:
    """Tests for policy enforcement configuration."""

    def test_default_disabled(self):
        """Test that policy enforcement is disabled by default."""
        controls = ExecutionControls()
        assert controls.enable_policy_enforcement is False

    def test_enable_via_constructor(self):
        """Test enabling policy enforcement via constructor."""
        controls = ExecutionControls(enable_policy_enforcement=True)
        assert controls.enable_policy_enforcement is True

    def test_enable_via_env(self):
        """Test enabling policy enforcement via environment variable."""
        with patch.dict("os.environ", {"CLYRO_ENABLE_POLICIES": "true"}):
            config = ClyroConfig.from_env()
            assert config.controls.enable_policy_enforcement is True

    def test_disable_via_env(self):
        """Test explicit disable via environment variable."""
        with patch.dict("os.environ", {"CLYRO_ENABLE_POLICIES": "false"}):
            config = ClyroConfig.from_env()
            assert config.controls.enable_policy_enforcement is False

    def test_enable_via_env_numeric(self):
        """Test enabling with numeric '1'."""
        with patch.dict("os.environ", {"CLYRO_ENABLE_POLICIES": "1"}):
            config = ClyroConfig.from_env()
            assert config.controls.enable_policy_enforcement is True


# =============================================================================
# Integration Test: Full Flow
# =============================================================================


class TestPolicyEnforcementFlow:
    """Tests for the full enforcement flow (evaluator + session)."""

    def test_full_flow_allow(self, policy_config, agent_id, allow_response):
        """Test complete allow flow: config → evaluator → session → allow."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )
        session.start(input_data={"query": "test"})

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ) as mock_eval:
            session.check_policy("tool_call", {"tool_name": "get_queues"})

        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args
        assert call_kwargs.kwargs["action_type"] == "tool_call"
        assert call_kwargs.kwargs["parameters"]["tool_name"] == "get_queues"
        assert call_kwargs.kwargs["session_id"] == session.session_id

    def test_full_flow_block(self, policy_config, agent_id, block_response):
        """Test complete block flow: config → evaluator → session → raise."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )
        session.start(input_data={"query": "test"})

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(block_response),
        ):
            with pytest.raises(PolicyViolationError) as exc_info:
                session.check_policy("tool_call", {"tool_name": "process_refund"})

        error = exc_info.value
        assert error.rule_id == "rule-001"
        assert error.rule_name == "max_refund"
        assert error.action_type == "tool_call"

    def test_full_flow_disabled_noop(self, policy_disabled_config, agent_id):
        """Test that disabled enforcement is truly a no-op."""
        session = Session(config=policy_disabled_config, agent_id=agent_id)
        session.start()

        # No evaluator, should be instant no-op
        session.check_policy("tool_call", {"tool_name": "anything"})
        session.check_policy("llm_call", {"model": "gpt-4"})
        session.check_policy("agent_execution", {"input": "hello"})

    def test_multiple_checks_in_session(self, policy_config, agent_id, allow_response):
        """Test multiple policy checks within a single session."""
        evaluator = PolicyEvaluator(policy_config, agent_id)
        session = Session(
            config=policy_config,
            agent_id=agent_id,
            policy_evaluator=evaluator,
        )
        session.start()

        with patch.object(
            evaluator._client, "evaluate_sync",
            return_value=PolicyDecision.from_response(allow_response),
        ) as mock_eval:
            session.check_policy("agent_execution", {"input": "query"})
            session.check_policy("tool_call", {"tool_name": "tool_a"})
            session.check_policy("tool_call", {"tool_name": "tool_b"})
            session.check_policy("llm_call", {"model": "gpt-4o-mini"})

        assert mock_eval.call_count == 4


# =============================================================================
# ConsoleApprovalHandler Tests
# =============================================================================


class TestConsoleApprovalHandler:
    """Tests for ConsoleApprovalHandler."""

    def test_approve_with_y(self, approval_response):
        """Test that 'y' input approves the action."""
        handler = ConsoleApprovalHandler()
        decision = PolicyDecision.from_response(approval_response)

        with patch("builtins.input", return_value="y"):
            result = handler(decision, "tool_call")

        assert result is True

    def test_approve_with_yes(self, approval_response):
        """Test that 'yes' input approves the action."""
        handler = ConsoleApprovalHandler()
        decision = PolicyDecision.from_response(approval_response)

        with patch("builtins.input", return_value="yes"):
            result = handler(decision, "tool_call")

        assert result is True

    def test_deny_with_n(self, approval_response):
        """Test that 'n' input denies the action."""
        handler = ConsoleApprovalHandler()
        decision = PolicyDecision.from_response(approval_response)

        with patch("builtins.input", return_value="n"):
            result = handler(decision, "tool_call")

        assert result is False

    def test_deny_with_no(self, approval_response):
        """Test that 'no' input denies the action."""
        handler = ConsoleApprovalHandler()
        decision = PolicyDecision.from_response(approval_response)

        with patch("builtins.input", return_value="no"):
            result = handler(decision, "tool_call")

        assert result is False

    def test_retries_on_invalid_input(self, approval_response):
        """Test that invalid input is retried until valid."""
        handler = ConsoleApprovalHandler()
        decision = PolicyDecision.from_response(approval_response)

        with patch("builtins.input", side_effect=["maybe", "x", "y"]):
            result = handler(decision, "tool_call")

        assert result is True
