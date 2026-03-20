# Tests for Clyro SDK Loop Detector
# Implements PRD-010

"""Unit tests for enhanced loop detection."""

import pytest

from clyro.exceptions import LoopDetectedError
from clyro.loop_detector import (
    DEFAULT_EXCLUDED_FIELDS,
    LoopDetector,
    LoopDetectorState,
    LoopSignal,
)


class TestLoopDetector:
    """Tests for LoopDetector class."""

    def test_init_default_threshold(self):
        """Test default threshold initialization."""
        detector = LoopDetector()
        assert detector.threshold == 3

    def test_init_custom_threshold(self):
        """Test custom threshold initialization."""
        detector = LoopDetector(threshold=5)
        assert detector.threshold == 5

    def test_init_invalid_threshold(self):
        """Test that threshold must be at least 2."""
        with pytest.raises(ValueError, match="threshold must be at least 2"):
            LoopDetector(threshold=1)

    def test_reset(self):
        """Test resetting detector state."""
        detector = LoopDetector()

        # Simulate some usage
        detector.check(state={"counter": 1})
        detector.check(state={"counter": 1})

        assert detector.step_count == 2

        # Reset
        detector.reset()

        assert detector.step_count == 0


class TestStateHashComputation:
    """Tests for state hash computation."""

    def test_compute_state_hash_deterministic(self):
        """Test that hash is deterministic for same state."""
        detector = LoopDetector()

        state = {"key": "value", "number": 42}
        hash1 = detector.compute_state_hash(state)
        hash2 = detector.compute_state_hash(state)

        assert hash1 == hash2

    def test_compute_state_hash_different_states(self):
        """Test that different states produce different hashes."""
        detector = LoopDetector()

        hash1 = detector.compute_state_hash({"counter": 1})
        hash2 = detector.compute_state_hash({"counter": 2})

        assert hash1 != hash2

    def test_compute_state_hash_key_order_independent(self):
        """Test that key order doesn't affect hash."""
        detector = LoopDetector()

        state1 = {"a": 1, "b": 2, "c": 3}
        state2 = {"c": 3, "a": 1, "b": 2}

        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

    def test_compute_state_hash_none(self):
        """Test that None state returns None."""
        detector = LoopDetector()
        assert detector.compute_state_hash(None) is None

    def test_compute_state_hash_excludes_timestamps(self):
        """Test that timestamp fields are excluded from hash."""
        detector = LoopDetector()

        state1 = {"data": "value", "timestamp": "2024-01-01T00:00:00Z"}
        state2 = {"data": "value", "timestamp": "2024-01-02T00:00:00Z"}

        # Should be equal because timestamp is excluded
        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

    def test_compute_state_hash_excludes_request_id(self):
        """Test that request_id is excluded from hash."""
        detector = LoopDetector()

        state1 = {"data": "value", "request_id": "req-123"}
        state2 = {"data": "value", "request_id": "req-456"}

        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

    def test_compute_state_hash_nested_excluded_fields(self):
        """Test that excluded fields in nested dicts are filtered."""
        detector = LoopDetector()

        state1 = {"data": {"value": 1, "timestamp": "2024-01-01"}}
        state2 = {"data": {"value": 1, "timestamp": "2024-02-01"}}

        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

    def test_compute_state_hash_non_serializable(self):
        """Test handling of non-serializable state."""
        detector = LoopDetector()

        class NonSerializable:
            pass

        state = NonSerializable()
        hash_result = detector.compute_state_hash({"obj": state})

        # compute_state_hash uses json.dumps with default=str, so it converts
        # non-serializable objects to strings and successfully hashes them
        assert hash_result is not None
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex digest length


class TestStateLoopDetection:
    """Tests for state-based loop detection."""

    def test_detects_repeated_state(self):
        """Test detection of repeated state."""
        detector = LoopDetector(threshold=3)

        state = {"counter": 42}

        # First two iterations should be fine
        detector.check(state=state, raise_on_loop=False)
        detector.check(state=state, raise_on_loop=False)

        # Third iteration should detect loop
        signal = detector.check(state=state, raise_on_loop=False)

        assert signal is not None
        assert signal.signal_type == "state_repeat"
        assert signal.iterations == 3

    def test_raises_on_loop_detection(self):
        """Test that exception is raised when loop detected."""
        detector = LoopDetector(threshold=2)

        state = {"stuck": True}

        detector.check(state=state)  # First

        with pytest.raises(LoopDetectedError) as exc_info:
            detector.check(state=state)  # Second - should trigger

        assert exc_info.value.iterations == 2

    def test_no_loop_with_different_states(self):
        """Test that different states don't trigger loop detection."""
        detector = LoopDetector(threshold=2)

        for i in range(100):
            signal = detector.check(state={"counter": i}, raise_on_loop=False)
            assert signal is None

    def test_loop_detection_with_session_id(self):
        """Test loop detection includes session_id in error."""
        detector = LoopDetector(threshold=2)

        state = {"value": "test"}
        detector.check(state=state, session_id="session-123")

        with pytest.raises(LoopDetectedError) as exc_info:
            detector.check(state=state, session_id="session-123")

        assert exc_info.value.session_id == "session-123"


class TestActionSequenceDetection:
    """Tests for action sequence loop detection."""

    def test_detects_single_action_repeat(self):
        """Test detection of single repeated action."""
        detector = LoopDetector(threshold=3, action_sequence_length=1)

        for _ in range(2):
            signal = detector.check(action="process", raise_on_loop=False)
            assert signal is None

        signal = detector.check(action="process", raise_on_loop=False)
        assert signal is not None
        assert signal.signal_type == "action_repeat"

    def test_detects_action_sequence_repeat(self):
        """Test detection of repeated action sequences."""
        detector = LoopDetector(threshold=3, action_sequence_length=2)

        # Repeat A -> B sequence 3 times
        actions = ["A", "B"] * 3

        signal = None
        for action in actions:
            signal = detector.check(action=action, raise_on_loop=False)

        assert signal is not None
        assert signal.signal_type == "action_repeat"
        assert signal.action_sequence == ("A", "B") or signal.action_sequence == ("B",)

    def test_no_action_loop_with_varied_actions(self):
        """Test that varied actions don't trigger loop."""
        detector = LoopDetector(threshold=3)

        actions = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for action in actions:
            signal = detector.check(action=action, raise_on_loop=False)
            assert signal is None


class TestCombinedDetection:
    """Tests for combined state + action detection."""

    def test_state_takes_precedence(self):
        """Test that state-based detection takes precedence."""
        detector = LoopDetector(threshold=2)

        # Same state but different actions
        signal = detector.check(state={"stuck": True}, action="A", raise_on_loop=False)
        signal = detector.check(state={"stuck": True}, action="B", raise_on_loop=False)

        assert signal is not None
        assert signal.signal_type == "state_repeat"

    def test_action_used_when_no_state(self):
        """Test that action detection works without state."""
        detector = LoopDetector(threshold=3, action_sequence_length=1)

        for _ in range(3):
            signal = detector.check(action="stuck_action", raise_on_loop=False)

        assert signal is not None
        assert signal.signal_type == "action_repeat"


class TestLoopSignal:
    """Tests for LoopSignal dataclass."""

    def test_state_repeat_signal(self):
        """Test creating state repeat signal."""
        signal = LoopSignal(
            signal_type="state_repeat",
            iterations=3,
            state_hash="abc123",
            confidence=1.0,
        )

        assert signal.signal_type == "state_repeat"
        assert signal.iterations == 3
        assert signal.state_hash == "abc123"
        assert signal.action_sequence is None

    def test_action_repeat_signal(self):
        """Test creating action repeat signal."""
        signal = LoopSignal(
            signal_type="action_repeat",
            iterations=4,
            action_sequence=("A", "B"),
            confidence=0.8,
        )

        assert signal.signal_type == "action_repeat"
        assert signal.action_sequence == ("A", "B")
        assert signal.confidence == 0.8


class TestLoopDetectorStatistics:
    """Tests for detector statistics."""

    def test_get_statistics(self):
        """Test getting detector statistics."""
        detector = LoopDetector(threshold=5)

        # Add some states
        detector.check(state={"a": 1}, raise_on_loop=False)
        detector.check(state={"b": 2}, raise_on_loop=False)
        detector.check(state={"a": 1}, raise_on_loop=False)

        stats = detector.get_statistics()

        assert stats["step_count"] == 3
        assert stats["unique_states"] == 2
        assert stats["max_state_repetitions"] == 2
        assert stats["threshold"] == 5

    def test_statistics_after_reset(self):
        """Test that statistics are reset properly."""
        detector = LoopDetector()

        detector.check(state={"test": True}, raise_on_loop=False)
        detector.reset()

        stats = detector.get_statistics()
        assert stats["step_count"] == 0
        assert stats["unique_states"] == 0


class TestExcludedFields:
    """Tests for field exclusion configuration."""

    def test_default_excluded_fields(self):
        """Test that default excluded fields are correct."""
        expected = {
            "timestamp",
            "created_at",
            "updated_at",
            "request_id",
            "trace_id",
            "span_id",
            "correlation_id",
            "execution_id",
            "run_id",
            "session_id",
            "message_id",
            "_id",
            "id",
            "uuid",
        }
        assert DEFAULT_EXCLUDED_FIELDS == expected

    def test_custom_excluded_fields(self):
        """Test custom excluded fields configuration."""
        custom_excluded = frozenset({"custom_field", "another_field"})
        detector = LoopDetector(excluded_fields=custom_excluded)

        state1 = {"data": "value", "custom_field": "123"}
        state2 = {"data": "value", "custom_field": "456"}

        # Should be equal because custom_field is excluded
        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

        # But timestamp should no longer be excluded
        state3 = {"data": "value", "timestamp": "2024-01-01"}
        state4 = {"data": "value", "timestamp": "2024-02-01"}

        assert detector.compute_state_hash(state3) != detector.compute_state_hash(state4)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_state(self):
        """Test handling of empty state dict."""
        detector = LoopDetector(threshold=2)

        # Empty states should all hash the same
        detector.check(state={}, raise_on_loop=False)

        with pytest.raises(LoopDetectedError):
            detector.check(state={})

    def test_deeply_nested_state(self):
        """Test handling of deeply nested state."""
        detector = LoopDetector()

        nested_state = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": 42,
                        }
                    }
                }
            }
        }

        hash_result = detector.compute_state_hash(nested_state)
        assert hash_result is not None
        assert len(hash_result) == 64

    def test_state_with_list(self):
        """Test handling of state with lists."""
        detector = LoopDetector()

        state1 = {"items": [1, 2, 3]}
        state2 = {"items": [1, 2, 3]}
        state3 = {"items": [3, 2, 1]}  # Different order

        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)
        assert detector.compute_state_hash(state1) != detector.compute_state_hash(state3)

    def test_state_with_none_values(self):
        """Test handling of state with None values."""
        detector = LoopDetector()

        state1 = {"a": None, "b": 1}
        state2 = {"a": None, "b": 1}

        assert detector.compute_state_hash(state1) == detector.compute_state_hash(state2)

    def test_high_threshold(self):
        """Test with high threshold value."""
        detector = LoopDetector(threshold=100)

        state = {"constant": True}

        # Should not trigger until 100th iteration
        for i in range(99):
            signal = detector.check(state=state, raise_on_loop=False)
            assert signal is None

        # 100th should trigger
        signal = detector.check(state=state, raise_on_loop=False)
        assert signal is not None
        assert signal.iterations == 100

    def test_mixed_types_in_state(self):
        """Test state with mixed types."""
        detector = LoopDetector()

        state = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        hash_result = detector.compute_state_hash(state)
        assert hash_result is not None
