# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""Background workers for Clyro SDK."""


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency chains."""
    if name == "sync_trace_buffer":
        from clyro.workers.sync_trace_buffer import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["sync_trace_buffer"]
