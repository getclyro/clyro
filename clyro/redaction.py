# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro SDK — Parameter Redaction
# Shared by MCP wrapper (clyro.mcp.audit) and hooks (clyro.hooks.audit)

"""
Recursive parameter redaction using fnmatch glob patterns.

Used by audit loggers across MCP and hooks to strip sensitive values
(passwords, tokens, secrets, API keys) before writing to JSONL logs.
"""

from __future__ import annotations

import copy
import fnmatch
from typing import Any

REDACTED = "[REDACTED]"
REDACTION_ERROR = "[REDACTION_ERROR]"

# Default patterns matching common sensitive parameter names (fnmatch syntax).
DEFAULT_REDACT_PATTERNS: list[str] = [
    "*password*",
    "*token*",
    "*secret*",
    "*api_key*",
]


def redact_value(value: Any, patterns: list[str], key: str = "") -> Any:
    """Recursively redact values whose keys match any glob pattern.

    Args:
        value: The value to inspect (may be dict, list, or scalar).
        patterns: fnmatch-style glob patterns to match against keys.
        key: Current key name (empty for root-level calls).

    Returns:
        A copy of *value* with matching keys replaced by ``[REDACTED]``.
    """
    if key and any(fnmatch.fnmatch(key.lower(), p.lower()) for p in patterns):
        return REDACTED

    if isinstance(value, dict):
        return {k: redact_value(v, patterns, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(item, patterns) for item in value]
    return value


def redact_params(
    params: dict[str, Any] | None,
    patterns: list[str],
) -> dict[str, Any]:
    """Redact sensitive parameters from a dict. Fail-safe per key.

    If redaction fails on a specific value, that key is replaced with
    ``[REDACTION_ERROR]`` rather than propagating the exception.

    Args:
        params: Parameter dict to redact (``None`` returns empty dict).
        patterns: fnmatch-style glob patterns.

    Returns:
        A new dict with sensitive values replaced.
    """
    if not params:
        return {}
    result: dict[str, Any] = {}
    for key, value in params.items():
        try:
            result[key] = redact_value(value, patterns, key)
        except Exception:
            result[key] = REDACTION_ERROR
    return result


def redact_dict_deepcopy(
    params: dict[str, Any] | None,
    patterns: list[str],
) -> dict[str, Any]:
    """Redact via deep copy + in-place key mutation.

    This variant is used by the MCP audit logger which needs deep-copy
    semantics to avoid mutating the original message dict.

    Args:
        params: Parameter dict to redact (``None`` returns empty dict).
        patterns: fnmatch-style glob patterns.

    Returns:
        A deep-copied dict with sensitive values replaced.
    """
    if not params or not patterns:
        return params or {}
    redacted = copy.deepcopy(params)
    for pattern in patterns:
        key_pattern = pattern.split(".")[-1] if "." in pattern else pattern
        _redact_keys_inplace(redacted, key_pattern)
    return redacted


def _redact_keys_inplace(obj: dict[str, Any], pattern: str) -> None:
    """Recursively redact dict keys matching *pattern* in place."""
    for key in list(obj.keys()):
        if fnmatch.fnmatch(key, pattern):
            obj[key] = REDACTED
        elif isinstance(obj[key], dict):
            _redact_keys_inplace(obj[key], pattern)
