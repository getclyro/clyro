# Implements FRD-015, NFR-001: Verify all public symbols importable
"""
Integration tests verifying all public symbols from all subpackages
are importable after monorepo consolidation.
"""



class TestCoreImports:
    """Verify SDK core imports work."""

    def test_import_clyro(self):
        import clyro
        assert hasattr(clyro, "wrap")
        assert hasattr(clyro, "configure")
        assert hasattr(clyro, "__version__")

    def test_import_config(self):
        from clyro.config import ClyroConfig, ExecutionControls
        assert ClyroConfig is not None
        assert ExecutionControls is not None

    def test_import_cost(self):
        from clyro.cost import CostCalculator, CostTracker, HeuristicCostEstimator
        assert CostCalculator is not None
        assert CostTracker is not None
        assert HeuristicCostEstimator is not None

    def test_import_loop_detector(self):
        from clyro.loop_detector import LoopDetector, compute_call_signature
        assert LoopDetector is not None
        assert compute_call_signature is not None

    def test_import_policy(self):
        from clyro.policy import LocalPolicyEvaluator, PolicyClient
        assert PolicyClient is not None
        assert LocalPolicyEvaluator is not None

    def test_import_exceptions(self):
        from clyro.exceptions import (
            AuthenticationError,
            BackendUnavailableError,
            ClyroError,
            RateLimitExhaustedError,
        )
        assert issubclass(AuthenticationError, ClyroError)
        assert issubclass(RateLimitExhaustedError, ClyroError)
        assert issubclass(BackendUnavailableError, ClyroError)


class TestMcpImports:
    """Verify MCP subpackage imports work."""

    def test_import_mcp_subpackage(self):
        from clyro.mcp import MessageRouter, PreventionStack
        assert MessageRouter is not None
        assert PreventionStack is not None

    def test_import_mcp_cli(self):
        from clyro.mcp.cli import main
        assert callable(main)

    def test_import_mcp_session(self):
        from clyro.mcp.session import McpSession
        assert McpSession is not None


class TestHooksImports:
    """Verify hooks subpackage imports work."""

    def test_import_hooks_cli(self):
        from clyro.hooks.cli import main
        assert callable(main)

    def test_import_hooks_evaluator(self):
        from clyro.hooks.evaluator import evaluate
        assert callable(evaluate)

    def test_import_hooks_state(self):
        from clyro.hooks.state import load_state
        assert callable(load_state)


class TestBackendImports:
    """Verify backend subpackage imports work."""

    def test_import_http_client(self):
        from clyro.backend.http_client import HttpSyncClient
        assert HttpSyncClient is not None

    def test_import_cloud_policy(self):
        from clyro.backend.cloud_policy import CloudPolicyFetcher
        assert CloudPolicyFetcher is not None

    def test_import_exceptions_from_backend(self):
        """AuthenticationError importable from both locations."""
        from clyro.backend.http_client import AuthenticationError as exc2
        from clyro.exceptions import AuthenticationError as exc1
        assert exc1 is exc2


class TestMcpConfigConsolidation:
    """Verify MCP config models accessible from clyro.config."""

    def test_policy_rule(self):
        from clyro.config import PolicyRule
        rule = PolicyRule(parameter="amount", operator="max_value", value=100)
        assert rule.parameter == "amount"

    def test_wrapper_config(self):
        from clyro.config import WrapperConfig
        config = WrapperConfig()
        assert config.global_.max_steps == 50

    def test_loop_detection_config(self):
        from clyro.config import LoopDetectionConfig
        config = LoopDetectionConfig()
        assert config.threshold == 3
