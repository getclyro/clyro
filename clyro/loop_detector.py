# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Loop Detector
# Implements PRD-010

"""
Enhanced loop detection for preventing infinite agent execution cycles.

This module provides sophisticated loop detection that tracks both state
hashes and recent action sequences to identify repetitive patterns that
may indicate an agent is stuck.

Detection Strategies:
1. State Hash Comparison: Detects identical state snapshots
2. Action Sequence Analysis: Detects repetitive action patterns
3. Combined Analysis: Correlates both for higher confidence detection
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import structlog

from clyro.exceptions import LoopDetectedError

logger = structlog.get_logger(__name__)


# Fields to exclude from state hashing (non-deterministic or timing-related)
DEFAULT_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
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
)


@dataclass(frozen=True)
class LoopSignal:
    """
    Represents a detected loop signal.

    Attributes:
        signal_type: Type of loop detected ("state_repeat" or "action_repeat")
        iterations: Number of repetitions observed
        state_hash: Hash of the repeated state (if state-based)
        action_sequence: Repeated action sequence (if action-based)
        confidence: Confidence level (0.0 to 1.0)
    """

    signal_type: str
    iterations: int
    state_hash: str | None = None
    action_sequence: tuple[str, ...] | None = None
    confidence: float = 1.0


@dataclass
class LoopDetectorState:
    """
    Internal state for the loop detector.

    Tracks state hashes and action history for analysis.
    """

    state_hash_counts: dict[str, int] = field(default_factory=dict)
    recent_actions: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    recent_states: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    step_count: int = 0


class LoopDetector:
    """
    Detects infinite loops in agent execution.

    The detector uses multiple strategies to identify loops:

    1. **State Hash Detection**: Tracks how many times each unique state
       has been seen. If a state repeats more than the threshold times,
       a loop is detected.

    2. **Action Sequence Detection**: Tracks recent actions and looks
       for repeating patterns (e.g., A → B → A → B → A → B).

    3. **Combined Detection**: Uses both signals for higher confidence.

    Example:
        ```python
        detector = LoopDetector(threshold=3)

        # Check state on each step
        for state in agent_states:
            detector.check(state=state, action="process")
        ```
    """

    def __init__(
        self,
        threshold: int = 3,
        action_sequence_length: int = 3,
        excluded_fields: frozenset[str] | None = None,
        window: int | None = None,
    ):
        """
        Initialize the loop detector.

        Args:
            threshold: Number of repetitions before triggering loop detection
            action_sequence_length: Length of action sequences to track
            excluded_fields: Fields to exclude from state hashing
            window: Sliding window size for the simple call-tracking API.
                    When set, enables the legacy ``check(tool, params)`` style.
        """
        if threshold < 2:
            raise ValueError("threshold must be at least 2")
        if window is not None and window < 1:
            raise ValueError("window must be at least 1")

        self.threshold = threshold
        self.action_sequence_length = action_sequence_length
        self.excluded_fields = excluded_fields or DEFAULT_EXCLUDED_FIELDS
        self._state = LoopDetectorState()
        # Sliding-window tracking for legacy check(tool, params) API
        self._window = window
        self._call_history: deque[str] = deque(maxlen=window) if window else deque()

    def reset(self) -> None:
        """Reset the detector state for a new session."""
        self._state = LoopDetectorState()
        self._call_history.clear()

    @property
    def step_count(self) -> int:
        """Number of steps processed by this detector."""
        return self._state.step_count

    def _filter_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Filter out non-deterministic fields from state.

        Args:
            state: Raw state dictionary

        Returns:
            Filtered state with excluded fields removed
        """
        if not isinstance(state, dict):
            return state

        filtered = {}
        for key, value in state.items():
            # Skip excluded fields
            key_lower = key.lower()
            if key_lower in self.excluded_fields or key in self.excluded_fields:
                continue

            # Recursively filter nested dicts
            if isinstance(value, dict):
                filtered[key] = self._filter_state(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_state(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                filtered[key] = value

        return filtered

    def compute_state_hash(self, state: dict[str, Any] | None) -> str | None:
        """
        Compute a deterministic hash of the state.

        Args:
            state: State dictionary to hash

        Returns:
            SHA-256 hash of the filtered state, or None if state is None
        """
        if state is None:
            return None

        try:
            filtered = self._filter_state(state)
            serialized = json.dumps(filtered, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()
        except (TypeError, ValueError) as e:
            logger.debug("state_hash_computation_failed", error=str(e))
            return None

    def _check_state_loop(self, state_hash: str) -> LoopSignal | None:
        """
        Check if a state hash indicates a loop.

        Args:
            state_hash: Hash of the current state

        Returns:
            LoopSignal if loop detected, None otherwise
        """
        self._state.state_hash_counts[state_hash] = (
            self._state.state_hash_counts.get(state_hash, 0) + 1
        )
        count = self._state.state_hash_counts[state_hash]

        self._state.recent_states.append(state_hash)

        if count >= self.threshold:
            return LoopSignal(
                signal_type="state_repeat",
                iterations=count,
                state_hash=state_hash,
                confidence=min(1.0, count / self.threshold),
            )

        return None

    def _check_action_sequence_loop(self) -> LoopSignal | None:
        """
        Check if recent actions form a repeating pattern.

        Looks for sequences like A → B → A → B → A → B

        Returns:
            LoopSignal if repeating pattern detected, None otherwise
        """
        actions = list(self._state.recent_actions)
        # Need at least threshold * sequence_length actions to detect a pattern
        if len(actions) < self.action_sequence_length * self.threshold:
            return None

        # Check for repeating sequences of various lengths (1 to action_sequence_length)
        # Start with length 1 (single action repeated) up to configured max length
        for seq_len in range(1, self.action_sequence_length + 1):
            # Skip if we don't have enough actions for this sequence length
            if len(actions) < seq_len * self.threshold:
                continue

            # Extract the most recent seq_len actions as the candidate pattern
            # e.g., if seq_len=2 and actions=['A','B','A','B','A','B'], pattern=('A','B')
            pattern = tuple(actions[-seq_len:])

            # Use a sliding window to count consecutive repetitions of this pattern
            # Walk backwards from the end of the action list in steps of seq_len
            repeat_count = 0
            for i in range(len(actions) - seq_len, -1, -seq_len):
                # Extract a window of size seq_len starting at position i
                window = tuple(actions[i : i + seq_len])
                if window == pattern:
                    repeat_count += 1
                else:
                    # Stop counting at first non-matching window (only consecutive repeats count)
                    break

            # If pattern repeated enough times, we've detected a loop
            if repeat_count >= self.threshold:
                return LoopSignal(
                    signal_type="action_repeat",
                    iterations=repeat_count,
                    action_sequence=pattern,
                    confidence=min(1.0, repeat_count / self.threshold),
                )

        return None

    def check(self, *args: Any, **kwargs: Any) -> LoopSignal | tuple[bool, dict[str, Any]] | None:
        """
        Check for loop conditions.

        Supports two calling conventions:

        **Legacy (MCP wrapper):**
            ``check(tool_name, params)`` → ``(is_loop, details_dict)``

        **Enhanced (SDK):**
            ``check(state=..., action=...)`` → ``LoopSignal | None``

        The legacy form is detected when two positional args are passed
        with the first being a ``str`` and the second a ``dict``.
        """
        # --- Legacy positional API: check("tool", {"x": 1}) -> (bool, dict) ---
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            return self._check_legacy(args[0], args[1])

        # --- Enhanced keyword API ---
        return self._check_enhanced(
            state=args[0] if args else kwargs.get("state"),
            state_hash=args[1] if len(args) > 1 else kwargs.get("state_hash"),
            action=kwargs.get("action"),
            raise_on_loop=kwargs.get("raise_on_loop", True),
            session_id=kwargs.get("session_id"),
        )

    def _check_legacy(self, tool_name: str, params: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Sliding-window check for MCP wrapper compatibility."""
        call_sig = self.compute_state_hash({"tool": tool_name, **params}) or ""
        self._call_history.append(call_sig)
        count = sum(1 for s in self._call_history if s == call_sig)
        is_loop = count >= self.threshold
        details: dict[str, Any] = {
            "repetition_count": count,
            "threshold": self.threshold,
            "pattern_hash": call_sig[:16],
        }
        return is_loop, details

    def _check_enhanced(
        self,
        state: dict[str, Any] | None = None,
        state_hash: str | None = None,
        action: str | None = None,
        raise_on_loop: bool = True,
        session_id: str | None = None,
    ) -> LoopSignal | None:
        """Enhanced check using state hashing and action sequence detection."""
        self._state.step_count += 1
        signal = None

        # Check state-based loop
        if state_hash is None and state is not None:
            state_hash = self.compute_state_hash(state)
        if state_hash:
            signal = self._check_state_loop(state_hash)

        # Check action-based loop
        if action is not None:
            self._state.recent_actions.append(action)
            action_signal = self._check_action_sequence_loop()
            # Prefer state-based signal, but use action signal if stronger
            if action_signal and (signal is None or action_signal.confidence > signal.confidence):
                signal = action_signal

        if signal and raise_on_loop:
            logger.warning(
                "loop_detected",
                signal_type=signal.signal_type,
                iterations=signal.iterations,
                state_hash=signal.state_hash[:16] if signal.state_hash else None,
                action_sequence=signal.action_sequence,
                step_count=self._state.step_count,
            )
            raise LoopDetectedError(
                iterations=signal.iterations,
                state_hash=signal.state_hash or "action_pattern",
                session_id=session_id,
                step_number=self._state.step_count,
            )

        return signal

    def get_statistics(self) -> dict[str, Any]:
        """
        Get detector statistics for debugging.

        Returns:
            Dictionary with detector state information
        """
        return {
            "step_count": self._state.step_count,
            "unique_states": len(self._state.state_hash_counts),
            "recent_actions_count": len(self._state.recent_actions),
            "recent_states_count": len(self._state.recent_states),
            "max_state_repetitions": (
                max(self._state.state_hash_counts.values()) if self._state.state_hash_counts else 0
            ),
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Consolidated from clyro_mcp.loop_detector — Implements FRD-004
# Simple signature-based loop detection for MCP/hooks contexts.
# ---------------------------------------------------------------------------


def compute_call_signature(tool_name: str, params: dict[str, Any] | None) -> str:
    """
    Compute a SHA-256 signature for a tool call.

    Uses canonical JSON (keys sorted, no whitespace) for determinism.
    Standalone utility preserved from the MCP wrapper for use by
    MCP/hooks evaluators.

    Args:
        tool_name: MCP or Claude Code tool name.
        params: Tool call arguments (may be ``None``).

    Returns:
        Hex-encoded SHA-256 hash.
    """
    import json as _json

    canonical = _json.dumps(
        {"tool": tool_name, "params": params or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
