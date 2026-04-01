# Performance Tests for Clyro SDK Execution Controls
# Implements PRD-NFR-001

"""
Performance tests to validate SDK overhead requirements.

Target: Execution controls should add <10ms p95 overhead to agent execution.
"""

import time
from decimal import Decimal

import pytest

from clyro.config import ClyroConfig, ExecutionControls
from clyro.cost import CostCalculator, OpenAITokenExtractor, TiktokenEstimator
from clyro.loop_detector import LoopDetector
from clyro.session import Session


def measure_time_us(func, *args, **kwargs):
    """Measure function execution time in microseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return (end - start) * 1_000_000, result  # Convert to microseconds


def calculate_percentile(timings: list[float], percentile: float) -> float:
    """Calculate the given percentile from timing data."""
    sorted_timings = sorted(timings)
    index = int(len(sorted_timings) * percentile / 100)
    return sorted_timings[min(index, len(sorted_timings) - 1)]


class TestCostCalculatorPerformance:
    """Performance tests for cost calculator."""

    def test_token_extraction_performance(self):
        """Test that token extraction is fast (<100μs p95)."""
        extractor = OpenAITokenExtractor()

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(extractor.extract, response)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Token extraction should be very fast
        assert p95 < 100, f"Token extraction p95 ({p95:.2f}μs) exceeds 100μs"

        print(f"\nToken extraction timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_cost_calculation_performance(self):
        """Test that cost calculation is fast (<500μs p95)."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(
                calculator.calculate, 1000, 500, "gpt-4-turbo"
            )
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Cost calculation should be fast
        assert p95 < 500, f"Cost calculation p95 ({p95:.2f}μs) exceeds 500μs"

        print(f"\nCost calculation timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_full_cost_from_response_performance(self):
        """Test that full cost calculation from response is fast (<1ms p95)."""
        config = ClyroConfig()
        calculator = CostCalculator(config)

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(
                calculator.calculate_from_response, response
            )
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Full calculation should be under 1ms
        assert p95 < 1000, f"Full cost calculation p95 ({p95:.2f}μs) exceeds 1ms"

        print(f"\nFull cost calculation timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")


class TestLoopDetectorPerformance:
    """Performance tests for loop detector."""

    def test_state_hash_performance(self):
        """Test that state hashing is fast (<500μs p95)."""
        detector = LoopDetector()

        state = {
            "counter": 42,
            "messages": ["hello", "world"],
            "nested": {"key": "value"},
        }

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(detector.compute_state_hash, state)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        assert p95 < 500, f"State hashing p95 ({p95:.2f}μs) exceeds 500μs"

        print(f"\nState hashing timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_loop_check_performance(self):
        """Test that loop check is fast (<1ms p95)."""
        detector = LoopDetector(threshold=1000)  # High threshold to avoid raises

        timings = []
        for i in range(1000):
            state = {"counter": i}  # Different state each time
            duration_us, _ = measure_time_us(
                detector.check, state=state, action=f"action_{i}", raise_on_loop=False
            )
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        assert p95 < 1000, f"Loop check p95 ({p95:.2f}μs) exceeds 1ms"

        print(f"\nLoop check timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_large_state_hash_performance(self):
        """Test hashing performance with larger state objects."""
        detector = LoopDetector()

        # Create a larger state object
        large_state = {
            f"key_{i}": {
                "value": i,
                "nested": {"data": f"value_{i}" * 10},
            }
            for i in range(100)
        }

        timings = []
        for _ in range(100):
            duration_us, _ = measure_time_us(detector.compute_state_hash, large_state)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Large state hashing should still be reasonable
        assert p95 < 5000, f"Large state hashing p95 ({p95:.2f}μs) exceeds 5ms"

        print(f"\nLarge state hashing timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")


class TestSessionPerformance:
    """Performance tests for session operations."""

    def test_record_step_performance(self):
        """Test that record_step is fast (<2ms p95)."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        timings = []
        for i in range(100):
            duration_us, _ = measure_time_us(
                session.record_step,
                event_name=f"step_{i}",
                input_data={"input": f"data_{i}"},
                output_data={"output": f"result_{i}"},
                state_snapshot={"counter": i},
                duration_ms=100,
                cost_usd=Decimal("0.001"),
            )
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Record step should be fast
        assert p95 < 2000, f"record_step p95 ({p95:.2f}μs) exceeds 2ms"

        print(f"\nrecord_step timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_record_llm_call_performance(self):
        """Test that record_llm_call is fast (<5ms p95)."""
        config = ClyroConfig()
        session = Session(config=config)
        session.start()

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "model": "gpt-4-turbo",
        }

        timings = []
        for i in range(100):
            duration_us, _ = measure_time_us(
                session.record_llm_call,
                model="gpt-4-turbo",
                input_data={"prompt": f"Test {i}"},
                output_data={"response": f"Result {i}"},
                llm_response=response,
                duration_ms=150,
            )
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # record_llm_call should be under 5ms including cost calculation
        assert p95 < 5000, f"record_llm_call p95 ({p95:.2f}μs) exceeds 5ms"

        print(f"\nrecord_llm_call timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")


class TestEndToEndExecutionControlPerformance:
    """End-to-end performance tests for execution controls."""

    def test_execution_controls_overhead(self):
        """Test total execution controls overhead is <10ms p95."""
        config = ClyroConfig(
            controls=ExecutionControls(
                max_steps=1000,
                max_cost_usd=100.0,
                loop_detection_threshold=100,
            )
        )

        response = {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "model": "gpt-4-turbo",
        }

        def full_step_with_controls():
            """Simulate a full step with all execution controls active."""
            session = Session(config=config)
            session.start()

            # Record an LLM call with cost tracking and loop detection
            session.record_llm_call(
                model="gpt-4-turbo",
                input_data={"prompt": "Test"},
                output_data={"response": "Result"},
                llm_response=response,
                duration_ms=150,
            )

            # Check step limit (happens in record_step/record_llm_call)
            # Check cost limit (happens in record_step/record_llm_call)
            # Loop detection (would happen with state_snapshot)

            session.end()

        timings = []
        for _ in range(100):
            duration_us, _ = measure_time_us(full_step_with_controls)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Total overhead should be under 10ms at p95
        # Note: This includes session creation, start, LLM call recording, and end
        assert p95 < 10000, f"Execution controls overhead p95 ({p95:.2f}μs) exceeds 10ms"

        print(f"\nExecution controls overhead (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")

    def test_step_limit_check_performance(self):
        """Test step limit check is very fast (<100μs)."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=10000)
        )
        session = Session(config=config)
        session.start()

        # Pre-populate with many steps
        for _ in range(5000):
            session._step_number += 1

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(session._check_step_limit)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)

        # Step limit check should be essentially free
        assert p95 < 100, f"Step limit check p95 ({p95:.2f}μs) exceeds 100μs"

        print(f"\nStep limit check timings (μs): p50={p50:.2f}, p95={p95:.2f}")

    def test_cost_limit_check_performance(self):
        """Test cost limit check is very fast (<100μs)."""
        config = ClyroConfig(
            controls=ExecutionControls(max_cost_usd=1000.0)
        )
        session = Session(config=config)
        session.start()
        session._cumulative_cost = Decimal("500.0")

        timings = []
        for _ in range(1000):
            duration_us, _ = measure_time_us(session._check_cost_limit)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)

        # Cost limit check should be very fast
        assert p95 < 100, f"Cost limit check p95 ({p95:.2f}μs) exceeds 100μs"

        print(f"\nCost limit check timings (μs): p50={p50:.2f}, p95={p95:.2f}")


class TestMemoryEfficiency:
    """Memory efficiency tests for execution controls."""

    def test_loop_detector_memory_bounded(self):
        """Test that loop detector memory usage is bounded."""
        detector = LoopDetector(threshold=100)

        # Run many checks - memory should be bounded by deque maxlen
        for i in range(10000):
            detector.check(
                state={"counter": i},
                action=f"action_{i}",
                raise_on_loop=False,
            )

        stats = detector.get_statistics()

        # Recent actions should be bounded (maxlen=50 by default)
        assert stats["recent_actions_count"] <= 50

        # Recent states should be bounded (maxlen=20 by default)
        assert stats["recent_states_count"] <= 20

        # Unique states grows but that's expected for unique inputs
        assert stats["unique_states"] == 10000

    def test_session_event_list_growth(self):
        """Test session event list growth is reasonable."""
        config = ClyroConfig(
            controls=ExecutionControls(max_steps=10000)
        )
        session = Session(config=config)
        session.start()

        # Record many steps
        for i in range(1000):
            session.record_step(event_name=f"step_{i}")

        # Events list should have expected size
        assert len(session.events) == 1001  # start + 1000 steps


@pytest.mark.skipif(
    not TiktokenEstimator.is_available(),
    reason="tiktoken not installed",
)
class TestTiktokenPerformance:
    """Performance tests for tiktoken-based estimation."""

    def test_tiktoken_count_tokens_performance(self):
        """Test tiktoken token counting performance."""
        text = "This is a sample text for token counting. " * 100

        timings = []
        for _ in range(100):
            duration_us, _ = measure_time_us(TiktokenEstimator.count_tokens, text)
            timings.append(duration_us)

        p50 = calculate_percentile(timings, 50)
        p95 = calculate_percentile(timings, 95)
        p99 = calculate_percentile(timings, 99)

        # Tiktoken should be reasonably fast (but slower than extraction)
        assert p95 < 10000, f"Tiktoken p95 ({p95:.2f}μs) exceeds 10ms"

        print(f"\nTiktoken count_tokens timings (μs): p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")
