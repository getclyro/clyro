# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Async traced-client policy-event sink tests (A10) — FRD-008/017/020.

Regression: the async OpenAI/Anthropic traced clients set the session's
policy-event sink to ``_buffer_event_sync``, which only did::

    self._session._events.append(event)

Nothing ever reads ``session._events`` back — every reference in both adapters
is an ``.append``. So every policy event was silently dropped on the async
clients: ``policy_check`` in enforce mode and ``would_block`` in dry_run.

Enforce mode hid the bug — the raised PolicyViolationError made the block
visible regardless of whether its audit event shipped. In dry_run the marker IS
the only output, so cloud policies in dry_run produced nothing at all.

``Session.record_event`` already appends to ``_events`` before the sink runs, so
the old append was a duplicate on top of being a dead end.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from clyro.config import ClyroConfig
from clyro.trace import TraceEvent
from clyro.transport import Transport


def _event(event_type: str = "would_block") -> TraceEvent:
    return TraceEvent(
        event_id=uuid4(),
        session_id=uuid4(),
        agent_id=uuid4(),
        event_type=event_type,
        event_name=f"{event_type}:policy",
    )


def _transport(batch_size: int = 100) -> Transport:
    cfg = ClyroConfig(agent_name="t", api_key="cly_test_x", batch_size=batch_size)
    return Transport(cfg)


class TestBufferEventNowait:
    def test_appends_to_the_transport_buffer(self):
        t = _transport()
        t.buffer_event_nowait(_event())
        assert len(t._event_buffer) == 1

    def test_survives_a_flush_snapshot(self):
        """The point: a synchronously-buffered event is in the batch flush sees.

        The async race this replaces — create_task then flush — lost the event,
        because _flush_buffer snapshots under an uncontended lock without
        yielding, so the scheduled append had not run yet.
        """

        async def scenario():
            t = _transport()
            sent: list[list[TraceEvent]] = []

            async def _capture(evs):
                sent.append(list(evs))

            t.send_events = _capture

            t.buffer_event_nowait(_event("would_block"))
            await t.flush()
            return sent

        sent = asyncio.run(scenario())
        assert sent and [e.event_type for e in sent[0]] == ["would_block"]

    def test_overflow_drops_oldest_not_newest(self):
        """A governance marker must not be the one discarded on overflow."""
        t = _transport(batch_size=1)  # MAX_BUFFER_SIZE = 10
        for _ in range(10):
            t.buffer_event_nowait(_event("policy_check"))
        t.buffer_event_nowait(_event("would_block"))
        assert len(t._event_buffer) == 10
        assert t._event_buffer[-1].event_type == "would_block"

    def test_never_flushes(self):
        """Must not flush: it cannot await, so flushing here would be a no-op
        at best and a dropped batch at worst."""
        t = _transport(batch_size=2)
        t.send_events = MagicMock()
        for _ in range(5):
            t.buffer_event_nowait(_event())
        t.send_events.assert_not_called()
        assert len(t._event_buffer) == 5


class TestAsyncClientSinks:
    """Both async clients must route the sink to the transport, not a dead list."""

    @pytest.mark.parametrize(
        "module,cls",
        [
            ("clyro.adapters.openai", "AsyncOpenAITracedClient"),
            ("clyro.adapters.anthropic", "AsyncAnthropicTracedClient"),
        ],
    )
    def test_sink_reaches_the_transport(self, module, cls):
        import importlib

        klass = getattr(importlib.import_module(module), cls)
        c = klass.__new__(klass)
        c._config = ClyroConfig(agent_name="t", api_key="cly_test_x")
        c._transport = _transport()
        c._session = MagicMock()
        c._session._events = []

        c._buffer_event_sync(_event("would_block"))

        assert len(c._transport._event_buffer) == 1, (
            f"{cls}._buffer_event_sync did not reach the transport — "
            "the policy event is dropped"
        )
        assert c._transport._event_buffer[0].event_type == "would_block"

    @pytest.mark.parametrize(
        "module,cls",
        [
            ("clyro.adapters.openai", "AsyncOpenAITracedClient"),
            ("clyro.adapters.anthropic", "AsyncAnthropicTracedClient"),
        ],
    )
    def test_sink_is_fail_open(self, module, cls):
        """A broken sink must never break agent execution."""
        import importlib

        klass = getattr(importlib.import_module(module), cls)
        c = klass.__new__(klass)
        c._config = ClyroConfig(agent_name="t", api_key="cly_test_x", fail_open=True)
        c._transport = MagicMock()
        c._transport.buffer_event_nowait.side_effect = RuntimeError("boom")
        c._session = MagicMock()

        c._buffer_event_sync(_event())  # must not raise
