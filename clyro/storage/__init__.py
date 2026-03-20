# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK Storage Module
# Implements PRD-005, PRD-006

"""
Storage backends and sync components for the Clyro SDK.

This module provides:
- LocalStorage: SQLite-based persistent storage for offline operation
- SyncWorker: Background sync orchestrator with retry and recovery
- Circuit breaker pattern for failure protection
- Connectivity detection and auto-recovery
"""

from clyro.storage.sqlite import (
    EventPriority,
    LocalStorage,
    StorageHealthStatus,
    StorageMetrics,
)
from clyro.workers.sync_worker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ConnectivityDetector,
    ConnectivityStatus,
    SyncMetrics,
    SyncWorker,
    SyncWorkerFactory,
)

# Backwards compatibility alias - SyncPriority is now EventPriority
SyncPriority = EventPriority

__all__ = [
    # SQLite Storage
    "LocalStorage",
    "StorageHealthStatus",
    "StorageMetrics",
    "EventPriority",
    # Sync Worker
    "SyncWorker",
    "SyncWorkerFactory",
    "SyncMetrics",
    "SyncPriority",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    # Connectivity
    "ConnectivityDetector",
    "ConnectivityStatus",
]
