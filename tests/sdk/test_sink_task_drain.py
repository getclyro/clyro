# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sink-task drain tests (A10) — a would_block emitted at teardown must ship.

Regression for the async-sink race: ``_buffer_event_sink`` can only *schedule*
buffering on the async transport (``create_task``), and ``_flush_buffer``
snapshots the buffer under an uncontended lock — which never yields. A marker
emitted just before cleanup therefore missed the batch and died with the
process.

That is exactly when the step limit trips (on the final ``record_step``), which
is why step markers vanished from the dashboard while cost/policy — emitted
mid-run, with later awaits to let their tasks run — arrived fine.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from clyro.trace import TraceEvent


def _event(event_type: str = "would_block", check: str = "step") -> TraceEvent:
    return TraceEvent(
        event_id=uuid4(),
        session_id=uuid4(),
        agent_id=uuid4(),
        event_type=event_type,
        event_name=f"{event_type}:{check}",
    )


class _FakeAsyncTransport:
    """Mimics Transport: buffer_event is a coroutine; flush snapshots without yielding."""

    def __init__(self) -> None:
        self._buffer: list[TraceEvent] = []
        self.sent: list[TraceEvent] = []
        self.storage = MagicMock()

    async def buffer_event(self, event: TraceEvent) -> None:
        # A real await point before the append — this is what loses the race.
        await asyncio.sleep(0)
        self._buffer.append(event)

    async def flush(self) -> None:
        # Snapshot-then-clear, exactly like Transport._flush_buffer under an
        # uncontended lock: no yield before the copy.
        events = self._buffer.copy()
        self._buffer.clear()
        await asyncio.sleep(0)
        self.sent.extend(events)


@pytest.mark.asyncio
async def test_drain_ships_event_scheduled_immediately_before_flush():
    """The bug: schedule a sink task, then flush at once — event must still ship."""
    from clyro.wrapper import WrappedAgent

    w = object.__new__(WrappedAgent)
    w._sink_tasks = set()
    transport = _FakeAsyncTransport()
    w._transport = transport

    loop = asyncio.get_running_loop()
    task = loop.create_task(transport.buffer_event(_event("would_block")))
    w._sink_tasks.add(task)
    task.add_done_callback(w._sink_tasks.discard)

    # Drain BEFORE flush — the fix.
    await w._drain_sink_tasks()
    await transport.flush()

    types = [e.event_type for e in transport.sent]
    assert "would_block" in types, (
        "would_block scheduled just before flush was lost — the drain did not await it"
    )


@pytest.mark.asyncio
async def test_without_drain_the_event_is_lost():
    """Characterise the bug: no drain → flush snapshots an empty buffer.

    Pins WHY the drain is required, so removing it fails loudly rather than
    silently reintroducing a vanished governance marker.
    """
    transport = _FakeAsyncTransport()
    loop = asyncio.get_running_loop()
    task = loop.create_task(transport.buffer_event(_event("would_block")))

    await transport.flush()  # no drain
    assert transport.sent == [], "expected the un-drained event to miss the batch"

    await task  # task lands in a buffer nobody flushes again
    assert len(transport._buffer) == 1


@pytest.mark.asyncio
async def test_drain_is_noop_when_no_tasks_pending():
    from clyro.wrapper import WrappedAgent

    w = object.__new__(WrappedAgent)
    w._sink_tasks = set()
    await w._drain_sink_tasks()  # must not raise


@pytest.mark.asyncio
async def test_drain_tolerates_failing_sink_task():
    """Fail-open: a sink task that raises must not break agent teardown."""
    from clyro.wrapper import WrappedAgent

    async def boom() -> None:
        raise RuntimeError("transport exploded")

    w = object.__new__(WrappedAgent)
    w._sink_tasks = set()
    loop = asyncio.get_running_loop()
    task = loop.create_task(boom())
    w._sink_tasks.add(task)

    await w._drain_sink_tasks()  # gather(return_exceptions=True) swallows it


@pytest.mark.asyncio
async def test_drain_awaits_every_pending_task():
    """All markers ship, not just the first."""
    from clyro.wrapper import WrappedAgent

    w = object.__new__(WrappedAgent)
    w._sink_tasks = set()
    transport = _FakeAsyncTransport()
    w._transport = transport

    loop = asyncio.get_running_loop()
    for check in ("step", "cost", "policy"):
        t = loop.create_task(transport.buffer_event(_event("would_block", check)))
        w._sink_tasks.add(t)
        t.add_done_callback(w._sink_tasks.discard)

    await w._drain_sink_tasks()
    await transport.flush()

    assert len(transport.sent) == 3, f"expected all 3 markers, shipped {len(transport.sent)}"
    assert {e.event_name for e in transport.sent} == {
        "would_block:step",
        "would_block:cost",
        "would_block:policy",
    }
