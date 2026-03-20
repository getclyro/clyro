"""
Unit tests for LoopDetector — TDD §11.1 tests #1–#5.
"""

from __future__ import annotations

import pytest

from clyro.loop_detector import LoopDetector, compute_call_signature


class TestComputeCallSignature:
    """Deterministic call signature hashing."""

    def test_same_input_same_hash(self) -> None:
        """Identical tool+params produce the same signature."""
        sig1 = compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        sig2 = compute_call_signature("read_file", {"path": "/tmp/a.txt"})
        assert sig1 == sig2

    def test_different_params_different_hash(self) -> None:
        sig1 = compute_call_signature("read_file", {"path": "/a"})
        sig2 = compute_call_signature("read_file", {"path": "/b"})
        assert sig1 != sig2

    def test_canonical_json_key_order(self) -> None:
        """TDD §11.1 #5 — key order does not affect hash."""
        sig1 = compute_call_signature("t", {"b": 1, "a": 2})
        sig2 = compute_call_signature("t", {"a": 2, "b": 1})
        assert sig1 == sig2

    def test_none_params_handled(self) -> None:
        sig = compute_call_signature("tool", None)
        assert isinstance(sig, str) and len(sig) == 64  # SHA-256 hex


class TestLoopDetector:
    """LoopDetector sliding-window tests — TDD §11.1 #1–#4."""

    def test_allows_below_threshold(self) -> None:
        """TDD §11.1 #1 — N-1 identical calls allowed."""
        det = LoopDetector(threshold=3, window=10)
        for _ in range(2):
            is_loop, _ = det.check("tool", {"x": 1})
            assert not is_loop

    def test_blocks_at_threshold(self) -> None:
        """TDD §11.1 #2 — Nth identical call blocked."""
        det = LoopDetector(threshold=3, window=10)
        results = []
        for _ in range(3):
            is_loop, details = det.check("tool", {"x": 1})
            results.append(is_loop)
        assert results == [False, False, True]

    def test_different_params_not_loop(self) -> None:
        """TDD §11.1 #3 — same tool, different params not a loop."""
        det = LoopDetector(threshold=3, window=10)
        for i in range(5):
            is_loop, _ = det.check("tool", {"x": i})
            assert not is_loop

    def test_window_eviction(self) -> None:
        """TDD §11.1 #4 — old calls evicted from window reset counter."""
        det = LoopDetector(threshold=3, window=5)

        # Two identical calls
        for _ in range(2):
            det.check("tool", {"x": 1})

        # Fill window with different calls to evict old ones
        for i in range(5):
            det.check("other", {"i": i})

        # Now the original signature should be gone
        is_loop, _ = det.check("tool", {"x": 1})
        assert not is_loop  # only 1 in window, not 3

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            LoopDetector(threshold=0, window=10)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            LoopDetector(threshold=3, window=0)

    def test_loop_details_populated(self) -> None:
        """Block decision includes repetition_count, threshold, pattern_hash."""
        det = LoopDetector(threshold=2, window=10)
        det.check("t", {"a": 1})
        _, details = det.check("t", {"a": 1})
        assert details["repetition_count"] == 2
        assert details["threshold"] == 2
        assert "pattern_hash" in details

    def test_reset_clears_history(self) -> None:
        """After reset, previously seen calls are forgotten."""
        det = LoopDetector(threshold=2, window=10)
        det.check("t", {"a": 1})
        det.reset()
        is_loop, _ = det.check("t", {"a": 1})
        assert not is_loop  # only 1 after reset
