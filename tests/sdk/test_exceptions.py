# Tests for Clyro SDK Exception Hierarchy
# Implements PRD-001, PRD-002

"""Unit tests for SDK exceptions."""

from clyro.constants import DEFAULT_API_URL, ISSUE_TRACKER_URL
from clyro.exceptions import (
    ClyroConfigError,
    ClyroError,
    ClyroWrapError,
    CostLimitExceededError,
    ExecutionControlError,
    FrameworkVersionError,
    LoopDetectedError,
    PolicyViolationError,
    StepLimitExceededError,
    TraceError,
    TransportError,
)


class TestClyroError:
    """Tests for base ClyroError."""

    def test_basic_error(self):
        """Test creating a basic error with message."""
        error = ClyroError("Something went wrong")
        assert "Something went wrong" in str(error)
        # FRD-SOF-009: str() includes issue tracker URL
        assert ISSUE_TRACKER_URL in str(error)
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_error_with_details(self):
        """Test creating error with details."""
        error = ClyroError("Error", details={"key": "value"})
        assert "key" in error.details
        assert error.details["key"] == "value"
        assert "details:" in str(error)

    def test_error_inheritance(self):
        """Test that ClyroError is an Exception."""
        error = ClyroError("Test")
        assert isinstance(error, Exception)


class TestClyroConfigError:
    """Tests for configuration errors."""

    def test_basic_config_error(self):
        """Test basic configuration error."""
        error = ClyroConfigError("Invalid config")
        assert "Invalid config" in str(error)

    def test_config_error_with_field(self):
        """Test config error with field information."""
        error = ClyroConfigError(
            "Invalid value",
            field="api_key",
            value="bad_key",
        )
        assert error.field == "api_key"
        assert error.value == "bad_key"
        assert "api_key" in str(error.details)

    def test_config_error_inherits_clyro_error(self):
        """Test inheritance from ClyroError."""
        error = ClyroConfigError("Test")
        assert isinstance(error, ClyroError)


class TestClyroWrapError:
    """Tests for wrap errors."""

    def test_basic_wrap_error(self):
        """Test basic wrap error."""
        error = ClyroWrapError("Cannot wrap agent")
        assert "Cannot wrap agent" in str(error)

    def test_wrap_error_with_type(self):
        """Test wrap error with agent type."""
        error = ClyroWrapError(
            "Agent not callable",
            agent_type="str",
        )
        assert error.agent_type == "str"
        assert "str" in str(error.details)

    def test_wrap_error_inherits_clyro_error(self):
        """Test inheritance from ClyroError."""
        error = ClyroWrapError("Test")
        assert isinstance(error, ClyroError)


class TestFrameworkVersionError:
    """Tests for framework version errors."""

    def test_framework_version_error(self):
        """Test framework version error creation."""
        error = FrameworkVersionError(
            framework="langgraph",
            version="0.1.0",
            supported=">=0.2.0",
        )
        assert error.framework == "langgraph"
        assert error.version == "0.1.0"
        assert error.supported == ">=0.2.0"
        assert "langgraph 0.1.0 is not supported" in str(error)
        assert ">=0.2.0" in str(error)

    def test_framework_version_error_details(self):
        """Test framework version error has proper details."""
        error = FrameworkVersionError(
            framework="crewai",
            version="0.20.0",
            supported=">=0.30.0",
        )
        assert error.details["framework"] == "crewai"
        assert error.details["detected_version"] == "0.20.0"
        assert error.details["supported_versions"] == ">=0.30.0"


class TestExecutionControlErrors:
    """Tests for execution control errors."""

    def test_step_limit_exceeded(self):
        """Test step limit exceeded error."""
        error = StepLimitExceededError(
            limit=100,
            current_step=101,
            session_id="test-session",
        )
        assert error.limit == 100
        assert error.current_step == 101
        assert error.session_id == "test-session"
        assert "Step limit exceeded" in str(error)
        assert "101" in str(error)
        assert "limit: 100" in str(error)

    def test_cost_limit_exceeded(self):
        """Test cost limit exceeded error."""
        error = CostLimitExceededError(
            limit_usd=10.0,
            current_cost_usd=15.5,
            session_id="test-session",
            step_number=50,
        )
        assert error.limit_usd == 10.0
        assert error.current_cost_usd == 15.5
        assert error.session_id == "test-session"
        assert error.step_number == 50
        assert "Cost limit exceeded" in str(error)
        assert "$15.5" in str(error)

    def test_loop_detected(self):
        """Test loop detected error."""
        error = LoopDetectedError(
            iterations=3,
            state_hash="abc123def456",
            session_id="test-session",
            step_number=30,
        )
        assert error.iterations == 3
        assert error.state_hash == "abc123def456"
        assert "Loop detected" in str(error)
        assert "3 times" in str(error)
        assert "abc123de" in str(error)  # Truncated hash

    def test_execution_control_error_base(self):
        """Test base execution control error."""
        error = ExecutionControlError(
            message="Execution stopped",
            session_id="session-1",
            step_number=25,
        )
        assert isinstance(error, ClyroError)
        assert error.session_id == "session-1"
        assert error.step_number == 25


class TestPolicyViolationError:
    """Tests for policy violation errors."""

    def test_policy_violation_error(self):
        """Test policy violation error."""
        error = PolicyViolationError(
            rule_id="rule-001",
            rule_name="max_quantity",
            message="Order quantity exceeds limit",
            action_type="create_order",
        )
        assert error.rule_id == "rule-001"
        assert error.rule_name == "max_quantity"
        assert error.action_type == "create_order"
        assert "Order quantity exceeds limit" in str(error)

    def test_policy_violation_details(self):
        """Test policy violation error details."""
        error = PolicyViolationError(
            rule_id="rule-002",
            rule_name="max_refund",
            message="Refund exceeds maximum",
        )
        assert error.details["rule_id"] == "rule-002"
        assert error.details["rule_name"] == "max_refund"


class TestTraceError:
    """Tests for trace errors."""

    def test_trace_error(self):
        """Test trace error creation."""
        error = TraceError(
            message="Failed to capture trace",
            event_id="event-123",
        )
        assert error.event_id == "event-123"
        assert "Failed to capture trace" in str(error)


class TestTransportError:
    """Tests for transport errors."""

    def test_transport_error(self):
        """Test transport error creation."""
        error = TransportError(
            message="Connection failed",
            endpoint=DEFAULT_API_URL,
            status_code=500,
        )
        assert error.endpoint == DEFAULT_API_URL
        assert error.status_code == 500
        assert "Connection failed" in str(error)

    def test_transport_error_without_status(self):
        """Test transport error without status code."""
        error = TransportError(
            message="Network unreachable",
            endpoint=DEFAULT_API_URL,
        )
        assert error.status_code is None
        assert error.endpoint == DEFAULT_API_URL


class TestExceptionHierarchy:
    """Test the complete exception hierarchy."""

    def test_all_exceptions_inherit_from_clyro_error(self):
        """Verify all custom exceptions inherit from ClyroError."""
        exceptions = [
            ClyroConfigError("test"),
            ClyroWrapError("test"),
            FrameworkVersionError("fw", "1.0", ">=2.0"),
            ExecutionControlError("test"),
            StepLimitExceededError(100, 101),
            CostLimitExceededError(10.0, 11.0),
            LoopDetectedError(3, "hash"),
            PolicyViolationError("id", "name", "msg"),
            TraceError("test"),
            TransportError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ClyroError), (
                f"{type(exc).__name__} should inherit from ClyroError"
            )
            assert isinstance(exc, Exception), (
                f"{type(exc).__name__} should be an Exception"
            )

    def test_execution_control_errors_inherit_from_base(self):
        """Verify execution control errors inherit from ExecutionControlError."""
        exceptions = [
            StepLimitExceededError(100, 101),
            CostLimitExceededError(10.0, 11.0),
            LoopDetectedError(3, "hash"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ExecutionControlError), (
                f"{type(exc).__name__} should inherit from ExecutionControlError"
            )
