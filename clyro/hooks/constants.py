# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Claude Code Hooks — Constants
# Implements FRD-HK-002, FRD-HK-011, FRD-HK-012

"""Default paths, exit codes, and configuration constants."""

from pathlib import Path

from clyro.config import GlobalConfig as _GlobalConfig
from clyro.config import LoopDetectionConfig as _LoopDetectionConfig
from clyro.redaction import DEFAULT_REDACT_PATTERNS as DEFAULT_REDACT_PARAMETERS  # noqa: F401

# ── Exit codes ──────────────────────────────────────────────────────────────
# Exit 0: decision rendered (PreToolUse) or trace completed (PostToolUse/Stop)
# Exit 1: internal error — fail-open (Claude Code allows the tool call)
# Exit 2: internal error — fail-closed (Claude Code blocks the tool call)
#          Used when the governance/prevention stack itself fails, so we deny
#          by default rather than silently allowing an unevaluated call.
EXIT_OK = 0
EXIT_FAIL_OPEN = 1
EXIT_FAIL_CLOSED = 2

# ── Default paths ───────────────────────────────────────────────────────────
HOOKS_DIR = Path.home() / ".clyro" / "hooks"
SESSIONS_DIR = HOOKS_DIR / "sessions"
DEFAULT_CONFIG_PATH = HOOKS_DIR / "claude-code-policy.yaml"
DEFAULT_AUDIT_PATH = HOOKS_DIR / "audit.jsonl"

# ── File permissions ────────────────────────────────────────────────────────
DIR_PERMISSIONS = 0o700
FILE_PERMISSIONS = 0o600

# ── Timeouts ────────────────────────────────────────────────────────────────
STATE_LOCK_TIMEOUT_SECONDS = 5
CLOUD_POLICY_TIMEOUT_SECONDS = 2.0

# ── Defaults (derived from clyro.config canonical models) ──────────────────
# GlobalConfig and LoopDetectionConfig Pydantic field defaults are canonical.
_gc = _GlobalConfig()
_lc = _LoopDetectionConfig()

DEFAULT_MAX_STEPS: int = _gc.max_steps
DEFAULT_MAX_COST_USD: float = _gc.max_cost_usd
DEFAULT_COST_PER_TOKEN_USD: float = _gc.cost_per_token_usd
DEFAULT_LOOP_THRESHOLD: int = _lc.threshold
DEFAULT_LOOP_WINDOW: int = _lc.window
DEFAULT_POLICY_CACHE_TTL_SECONDS: int = 300

del _gc, _lc

# ── Stale session cleanup ──────────────────────────────────────────────────
STALE_SESSION_AGE_HOURS = 24

# ── Enriched parameter prefix ──────────────────────────────────────────────
CLYRO_PARAM_PREFIX = "_clyro_"

# ── Backend integration ────────────────────────────────────────────────────
DEFAULT_AGENT_NAME = "claude-code-session"
AGENT_FRAMEWORK = "claude_code_hooks"
AGENT_ID_DIR = HOOKS_DIR / "agents"
EVENT_QUEUE_DIR = HOOKS_DIR / "pending"
EVENT_QUEUE_MAX_MB = 10
BACKEND_TIMEOUT_SECONDS = 5.0

# ── Trace event constants ─────────────────────────────────────────────────
OUTPUT_TRUNCATE_BYTES = 10 * 1024  # 10KB max output_data per event (FRD-HK-008)
MEMORY_FALLBACK_MAX_EVENTS = 1000  # Max events in memory when file I/O fails
