# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Anonymous Telemetry Client
# Implements FRD-CT-008, FRD-CT-009, FRD-CT-010, FRD-CT-011

"""
Opt-in anonymous telemetry client for local-mode SDK usage tracking.

Sends aggregate-only data (SDK version, Python version, framework, adapter,
session/error counts) to POST /v1/telemetry at session end.

Privacy (FRD-CT-009):
- Zero PII — no user identity, no org ID, no trace data, no cost data
- Payload logged to stderr before sending (FRD-CT-010)
- Disabled by default — only enabled when CLYRO_TELEMETRY=true (FRD-CT-011)
- Strict case-sensitive "true" only — deliberately stricter than CLYRO_QUIET

Design (TDD §5.4, §6.2):
- Fire-and-forget: all exceptions swallowed, never interrupts user code
- 5-second timeout on HTTP request
- Uses httpx (already an SDK dependency)
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from clyro.constants import DEFAULT_API_URL

if TYPE_CHECKING:
    from clyro.config import ClyroConfig
    from clyro.session import Session

logger = structlog.get_logger(__name__)

# Telemetry endpoint path
_TELEMETRY_PATH = "/v1/telemetry"

# HTTP timeout for telemetry submission (TDD §5.4 step 6)
_TELEMETRY_TIMEOUT_SECONDS = 5.0


def _is_telemetry_enabled() -> bool:
    """
    Check if telemetry is opted in.  Implements FRD-CT-011.

    Only exact string "true" (case-sensitive) enables telemetry.
    This is deliberately stricter than CLYRO_QUIET which accepts
    "true", "1", "yes" case-insensitively.
    """
    return os.environ.get("CLYRO_TELEMETRY") == "true"


def _collect_telemetry_payload(
    config: ClyroConfig,
    session: Session | None = None,
    session_count: int = 0,
    error_count: int = 0,
) -> dict[str, Any]:
    """
    Collect telemetry payload from environment and session state.

    Implements FRD-CT-008 step 1: collect all fields.
    Implements FRD-CT-009: only specified fields, no PII.

    Framework/adapter are sourced from the session (where WrappedAgent sets them),
    not from ClyroConfig (which doesn't have these fields).
    """
    import clyro

    # Get framework from session (set by WrappedAgent), fall back to "unknown"
    framework = "unknown"
    if session is not None:
        framework = getattr(session, "framework", None)
        if framework is not None:
            # Framework is an enum — get its string value
            framework = framework.value if hasattr(framework, "value") else str(framework)
        else:
            framework = "unknown"

    return {
        "sdk_version": getattr(clyro, "__version__", "0.0.0"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "framework": framework,
        "adapter": framework,  # adapter matches framework in current SDK
        "os": platform.system(),
        "session_count": session_count,
        "error_count": error_count,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _get_counts_from_sqlite(config: ClyroConfig) -> tuple[int, int]:
    """
    Read session_count and error_count from local SQLite.

    Implements TDD §5.4 step 3:
    - session_count: SELECT COUNT(DISTINCT session_id) FROM sync_status
    - error_count: SELECT COUNT(*) FROM trace_buffer WHERE event_type LIKE '%error%'

    Returns (session_count, error_count). Falls back to (0, 0) on any error.
    """
    try:
        from clyro.storage.sqlite import LocalStorage

        storage = LocalStorage(config)
        with storage._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(DISTINCT session_id) FROM sync_status")
            session_count = cursor.fetchone()[0] or 0

            cursor = conn.execute(
                "SELECT COUNT(*) FROM trace_buffer WHERE event_type LIKE '%error%'"
            )
            error_count = cursor.fetchone()[0] or 0

        return session_count, error_count
    except Exception:
        logger.debug("telemetry_sqlite_read_failed", exc_info=True)
        return 0, 0


def submit_telemetry(config: ClyroConfig, session: Session | None = None) -> None:
    """
    Submit anonymous telemetry event at session end.

    Implements FRD-CT-008: send telemetry if CLYRO_TELEMETRY=true.
    Implements FRD-CT-010: log payload to stderr before sending.
    Implements FRD-CT-011: strict opt-in gating.

    This function swallows ALL exceptions — telemetry must never
    interrupt user code (TDD §6.2 error boundary).
    """
    try:
        # Step 1: Check opt-in (FRD-CT-011)
        if not _is_telemetry_enabled():
            return

        # Step 2: Collect data
        session_count, error_count = _get_counts_from_sqlite(config)
        payload = _collect_telemetry_payload(
            config,
            session=session,
            session_count=session_count,
            error_count=error_count,
        )

        # Step 3: Log to stderr before sending (FRD-CT-010)
        # Telemetry audit log is exempt from CLYRO_QUIET
        payload_json = json.dumps(payload, default=str)
        try:
            print(
                f"[clyro] Telemetry (opt-in): sending {payload_json}",
                file=sys.stderr,
            )
        except Exception:
            pass  # stderr closed — continue with submission

        # Step 4: Send via httpx (TDD §9.2)
        endpoint = os.environ.get("CLYRO_ENDPOINT", DEFAULT_API_URL)
        url = f"{endpoint.rstrip('/')}{_TELEMETRY_PATH}"

        import httpx

        with httpx.Client(timeout=_TELEMETRY_TIMEOUT_SECONDS) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        logger.debug("telemetry_submitted", status=response.status_code)

    except Exception:
        # Swallow all exceptions — telemetry must never crash user code
        logger.debug("telemetry_submission_failed", exc_info=True)
