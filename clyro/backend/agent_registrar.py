# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Agent Registrar
# Implements FRD-016

"""
Auto-register MCP agent in the Clyro backend and persist agent_id
locally for cross-session reuse.

Agent ID file: ``~/.clyro/mcp-wrapper/mcp-agent-{instance_id}.id``
(plain text containing a single UUID).

Registration flow (TDD §2.18):
1. Try loading persisted agent_id from file.
2. If the persisted ID was a local fallback (not yet confirmed with backend),
   attempt backend registration and adopt the backend-assigned ID.
3. If absent or corrupted, register via POST /v1/agents/register.
4. If registration fails, fall back to a local-only UUID and persist it.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from clyro.backend.http_client import HttpSyncClient
from clyro.mcp.log import get_logger

logger = get_logger(__name__)

# Marker suffix appended to the ID file when the agent_id is a local
# fallback that has not been confirmed with the backend.
_UNCONFIRMED_SUFFIX = ".unconfirmed"


def _extract_org_id_from_api_key(api_key: str) -> UUID | None:
    """Extract org_id from JWT-style API key WITHOUT signature verification.

    Mirrors the SDK's _extract_org_id_from_jwt_api_key() so we can generate
    the same deterministic agent_id locally.
    """
    try:
        import base64 as _b64
        import json

        parts = api_key.split("_", 2)
        if len(parts) != 3:
            return None
        jwt_part = parts[2]
        jwt_components = jwt_part.rsplit(".", 1)
        if len(jwt_components) != 2:
            return None
        payload_b64, _ = jwt_components
        padding = "=" * (4 - len(payload_b64) % 4)
        payload_bytes = _b64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_bytes.decode("utf-8"))
        org_id_str = payload.get("org_id")
        if not org_id_str:
            return None
        return UUID(org_id_str)
    except (ValueError, Exception):
        return None


def _sanitize_agent_name(name: str) -> str:
    """Canonical agent name sanitization — MUST match the API's sanitize_agent_name()."""
    import re

    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    name = name[:255]
    return name.lower()


def _generate_deterministic_agent_id(agent_name: str, org_id: UUID) -> UUID:
    """Generate stable UUID5 agent_id — matches SDK and backend generate_agent_id()."""
    return uuid5(org_id, _sanitize_agent_name(agent_name))


class AgentRegistrar:
    """
    Manage agent identity for backend integration (FRD-016).

    Persists agent_id to ``~/.clyro/mcp-wrapper/mcp-agent-{instance_id}.id``
    so the same MCP server wrapper always uses the same identity.

    A companion ``.unconfirmed`` marker file tracks whether the persisted
    ID was generated locally (backend was offline).  When the marker exists,
    the next startup will attempt backend registration and replace the
    local ID with the backend-assigned one.

    Args:
        instance_id: Unique identifier derived from agent name.
        http_client: HTTP client for backend API calls.
    """

    def __init__(self, instance_id: str, http_client: HttpSyncClient, *, api_key: str = "") -> None:
        self._id_path = Path.home() / ".clyro" / "mcp-wrapper" / f"mcp-agent-{instance_id}.id"
        self._unconfirmed_path = self._id_path.with_suffix(
            self._id_path.suffix + _UNCONFIRMED_SUFFIX
        )
        self._http_client = http_client
        self._api_key = api_key

    async def get_or_register(self, agent_name: str) -> UUID:
        """
        Load persisted agent_id or register a new one (FRD-016).

        Returns:
            Agent UUID (from backend registration or local fallback).
        """
        # Try loading persisted ID
        persisted_id = self._load_persisted()

        if persisted_id is not None:
            # If the ID was confirmed with the backend, use it directly.
            if not self._unconfirmed_path.exists():
                return persisted_id

            # ID exists but was never confirmed — try registering now.
            try:
                agent_id_str = await self._http_client.register_agent(agent_name)
                agent_id = UUID(agent_id_str)
                self._persist(agent_id, confirmed=True)
                logger.info(
                    "agent_registered_after_recovery",
                    old_local_id=str(persisted_id),
                    new_backend_id=str(agent_id),
                )
                return agent_id
            except Exception as e:
                # Backend still unreachable — keep using the local ID.
                logger.debug("agent_re_registration_failed", error=str(e), fail_open=True)
                return persisted_id

        # No persisted ID — try registering with backend
        try:
            agent_id_str = await self._http_client.register_agent(agent_name)
            agent_id = UUID(agent_id_str)
            self._persist(agent_id, confirmed=True)
            return agent_id
        except Exception:
            # Fallback: deterministic UUID5 (matches SDK & backend) so identity
            # stays stable even if backend is down — no drift on recovery.
            org_id = _extract_org_id_from_api_key(self._api_key) if self._api_key else None
            if org_id:
                local_id = _generate_deterministic_agent_id(agent_name, org_id)
            else:
                local_id = uuid4()
            self._persist(local_id, confirmed=False)
            logger.warning("agent_registration_failed", local_id=str(local_id))
            return local_id

    def _load_persisted(self) -> UUID | None:
        """Load agent_id from the persisted file, or None if absent/corrupt."""
        if not self._id_path.exists():
            return None
        try:
            raw = self._id_path.read_text(encoding="utf-8").strip()
            return UUID(raw)
        except (ValueError, OSError):
            return None

    def _persist(self, agent_id: UUID, *, confirmed: bool) -> None:
        """Persist agent_id to file with restricted permissions (FRD-016)."""
        try:
            self._id_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(str(self._id_path.parent), 0o700)
            except OSError:
                pass
            self._id_path.write_text(str(agent_id), encoding="utf-8")

            if confirmed:
                # Remove the unconfirmed marker if it exists
                self._unconfirmed_path.unlink(missing_ok=True)
            else:
                # Write the unconfirmed marker
                self._unconfirmed_path.write_text("", encoding="utf-8")
        except OSError:
            logger.warning("agent_id_persist_failed")

    @property
    def id_path(self) -> Path:
        """Path to the agent ID file."""
        return self._id_path
