# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro Policy Recommender — LLM transport layer
# Implements policy-recommender FRD-PR-011, 012, 013, 014, 015

"""Transport-agnostic LLM invocation.

Three paths (FRD-PR-011): Claude Code CLI subprocess, Anthropic API key, and
rule-based (no LLM). Selection follows ``llm_transport`` (FRD-PR-012):
``auto`` tries claude-code → anthropic-api → rule-based; explicit modes fail loud
(FRD-PR-015). Cloud forces ``anthropic-api`` (FRD-PR-013).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Protocol

logger = logging.getLogger("clyro.recommender.transport")

VALID_TRANSPORTS = ("auto", "claude-code", "anthropic-api", "rule-based")

# CLI exit codes (TDD §6 / FRD-PR-015).
EXIT_CONFIG_ERROR = 2
EXIT_TRANSPORT_UNAVAILABLE = 3
EXIT_TRANSPORT_ERROR = 4
EXIT_UNEXPECTED = 5  # catastrophic / unexpected failure (FRD-PR exit-code table)


class RecommenderConfigError(Exception):
    """Invalid ``llm_transport`` value (exit 2)."""


class TransportUnavailable(Exception):
    """An explicitly-requested transport is not available (exit 3)."""

    def __init__(self, reason: str, remediation: str = ""):
        self.reason = reason
        self.remediation = remediation
        super().__init__(reason)


class TransportError(Exception):
    """A requested transport failed at runtime (exit 4)."""

    def __init__(self, path: str, cause: str):
        self.path = path
        self.cause = cause
        super().__init__(f"{path}: {cause}")


class Transport(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def invoke(self, prompt: str) -> str:
        """Return the LLM's raw text response (expected to be JSON)."""
        ...


class ClaudeCodeTransport:
    """Shell out to the Claude Code CLI (FRD-PR-011 path 1)."""

    name = "claude-code"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    def invoke(self, prompt: str) -> str:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "none"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise TransportUnavailable("claude_code_missing") from exc
        except subprocess.TimeoutExpired as exc:
            raise TransportError("claude-code", "timeout") from exc
        except OSError as exc:
            # e.g. E2BIG (argument list too long) on restrictive containers.
            raise TransportError("claude-code", f"{type(exc).__name__}: {exc}") from exc
        if proc.returncode != 0:
            raise TransportError(
                "claude-code", f"exit_code={proc.returncode}: {proc.stderr[-500:]}"
            )
        return _extract_claude_code_text(proc.stdout)


class AnthropicApiTransport:
    """Direct Anthropic SDK call with the customer's API key (FRD-PR-011 path 2)."""

    name = "anthropic-api"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def invoke(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise TransportUnavailable("anthropic_sdk_missing") from exc
        if not self._api_key:
            raise TransportUnavailable("no_anthropic_key")
        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            resp = client.messages.create(
                model=self._model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception as exc:  # network / auth / rate limit
            raise TransportError("anthropic-api", str(exc)) from exc


def _extract_claude_code_text(stdout: str) -> str:
    """Pull the assistant text out of Claude Code's ``--output-format json``."""
    try:
        doc = json.loads(stdout)
    except ValueError:
        return stdout  # already plain text
    if isinstance(doc, dict):
        for key in ("result", "text", "content"):
            if isinstance(doc.get(key), str):
                return doc[key]
    return stdout


def resolve_transport(
    requested: str,
    *,
    deployment_mode: str = "self-hosted",
    api_key: str | None = None,
) -> Transport | None:
    """Resolve the transport to use. ``None`` means rule-based (no LLM).

    Raises ``RecommenderConfigError`` on an invalid value, or ``TransportUnavailable``
    when an explicit mode's transport is missing (FRD-PR-012/014/015).
    """
    if requested not in VALID_TRANSPORTS:
        raise RecommenderConfigError(
            f"Invalid llm_transport '{requested}'. Valid: {', '.join(VALID_TRANSPORTS)}"
        )

    # Cloud forces anthropic-api server-side regardless of request (FRD-PR-013).
    if deployment_mode == "cloud" and requested != "anthropic-api":
        logger.warning(
            "clyro.recommender.transport_override_cloud requested=%s forced=anthropic-api",
            requested,
        )
        requested = "anthropic-api"

    claude_code = ClaudeCodeTransport()
    anthropic_api = AnthropicApiTransport(api_key=api_key)

    if requested == "rule-based":
        return None
    if requested == "claude-code":
        if not claude_code.is_available():
            raise TransportUnavailable(
                "claude_code_missing",
                "Install Claude Code from https://claude.com/code, or use --llm-transport auto.",
            )
        return claude_code
    if requested == "anthropic-api":
        if not anthropic_api.is_available():
            raise TransportUnavailable(
                "no_anthropic_key",
                "export ANTHROPIC_API_KEY=sk-..., or use --llm-transport auto.",
            )
        return anthropic_api

    # auto: first available wins, else rule-based (FRD-PR-014).
    if claude_code.is_available():
        return claude_code
    if anthropic_api.is_available():
        return anthropic_api
    return None
