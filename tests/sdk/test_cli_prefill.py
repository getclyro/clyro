# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Tests for `clyro suggest --prefill` — POSTing the recommendation to the backend
# and building the ?prefill=<token> wizard link (policy-recommender FRD-PR-FE-002).

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import clyro.wrapper as wrapper
from clyro.recommender import cli


class _Resp:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def _cfg(api_key="cly_live_x", endpoint="https://api.clyro.dev"):
    c = MagicMock()
    c.api_key = api_key
    c.endpoint = endpoint
    return c


def _rec_cfg(base="https://app.clyro.dev"):
    r = MagicMock()
    r.dashboard_base_url = base
    return r


class TestCreatePrefillToken:
    def test_success_posts_and_returns_token(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG-1")
        captured = {}

        def _fake_urlopen(req, timeout=15):
            captured["url"] = req.full_url
            captured["key"] = req.headers.get("X-clyro-api-key") or req.headers.get(
                "X-Clyro-API-Key"
            )
            captured["method"] = req.get_method()
            return _Resp({"prefill_token": "TOK123", "expires_at": "2026Z"})

        with patch("urllib.request.urlopen", _fake_urlopen):
            tok = cli._create_prefill_token({"agent_fingerprint": "a" * 64}, _cfg())
        assert tok == "TOK123"
        assert captured["url"].endswith("/v1/organizations/ORG-1/agent-setup/prefill")
        assert captured["method"] == "POST"

    def test_no_api_key_raises_prefill_error(self):
        with pytest.raises(cli._PrefillError):
            cli._create_prefill_token({}, _cfg(api_key=None))

    def test_no_agent_name_or_id_sends_no_agent_id(self, monkeypatch):
        # New-agent prefill: neither --agent-name nor --agent-id → no agent_id.
        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG-1")
        captured = {}

        def _fake_urlopen(req, timeout=15):
            captured["body"] = json.loads(req.data.decode())
            return _Resp({"prefill_token": "TOK"})

        with patch("urllib.request.urlopen", _fake_urlopen):
            cli._create_prefill_token({"agent_fingerprint": "a" * 64}, _cfg())
        assert "agent_id" not in captured["body"]

    def test_agent_name_injects_stable_agent_id(self, monkeypatch):
        # Re-recommend via --agent-name: agent_id derived as uuid5(org, name).
        from uuid import UUID

        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG-1")
        monkeypatch.setattr(wrapper, "_generate_agent_id_from_name", lambda name, org: UUID(int=42))
        captured = {}

        def _fake_urlopen(req, timeout=15):
            captured["body"] = json.loads(req.data.decode())
            return _Resp({"prefill_token": "TOK"})

        with patch("urllib.request.urlopen", _fake_urlopen):
            cli._create_prefill_token(
                {"agent_fingerprint": "a" * 64}, _cfg(), agent_name="mypkg:agent"
            )
        assert captured["body"]["agent_id"] == str(UUID(int=42))

    def test_explicit_agent_id_used_directly_over_name(self, monkeypatch):
        # Re-recommend via --agent-id: used verbatim, name-derivation NOT consulted.
        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG-1")

        def _boom(name, org):  # must not be called when agent_id is explicit
            raise AssertionError("name derivation should be skipped")

        monkeypatch.setattr(wrapper, "_generate_agent_id_from_name", _boom)
        captured = {}

        def _fake_urlopen(req, timeout=15):
            captured["body"] = json.loads(req.data.decode())
            return _Resp({"prefill_token": "TOK"})

        with patch("urllib.request.urlopen", _fake_urlopen):
            cli._create_prefill_token(
                {"agent_fingerprint": "a" * 64},
                _cfg(),
                agent_name="mypkg:agent",
                agent_id="11111111-1111-1111-1111-111111111111",
            )
        assert captured["body"]["agent_id"] == "11111111-1111-1111-1111-111111111111"

    def test_existing_agent_id_in_payload_is_not_overwritten(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG-1")
        captured = {}

        def _fake_urlopen(req, timeout=15):
            captured["body"] = json.loads(req.data.decode())
            return _Resp({"prefill_token": "TOK"})

        with patch("urllib.request.urlopen", _fake_urlopen):
            cli._create_prefill_token(
                {"agent_fingerprint": "a" * 64, "agent_id": "preset"},
                _cfg(),
                agent_name="mypkg:agent",
            )
        assert captured["body"]["agent_id"] == "preset"

    def test_unresolvable_org_raises(self, monkeypatch):
        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: None)
        with pytest.raises(cli._PrefillError):
            cli._create_prefill_token({}, _cfg())

    def test_http_error_raises_prefill_error(self, monkeypatch):
        import urllib.error

        monkeypatch.setattr(wrapper, "_extract_org_id_from_jwt_api_key", lambda k: "ORG")

        def _boom(req, timeout=15):
            raise urllib.error.HTTPError(req.full_url, 422, "Unprocessable", {}, None)

        with patch("urllib.request.urlopen", _boom), pytest.raises(cli._PrefillError):
            cli._create_prefill_token({}, _cfg())


class TestWizardLink:
    def test_no_prefill_returns_bare_link(self):
        link, prefilled = cli._wizard_link({}, _cfg(), _rec_cfg(), want_prefill=False)
        assert link == "https://app.clyro.dev/agents/new" and prefilled is False

    def test_prefill_success_appends_token(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_create_prefill_token",
            lambda payload, config, agent_name=None, agent_id=None: "TOK9",
        )
        link, prefilled = cli._wizard_link({}, _cfg(), _rec_cfg(), want_prefill=True)
        assert link == "https://app.clyro.dev/agents/new?prefill=TOK9" and prefilled is True

    def test_prefill_failure_degrades_to_bare_link(self, monkeypatch, capsys):
        def _raise(payload, config, agent_name=None, agent_id=None):
            raise cli._PrefillError("no api_key configured")

        monkeypatch.setattr(cli, "_create_prefill_token", _raise)
        link, prefilled = cli._wizard_link({}, _cfg(api_key=None), _rec_cfg(), want_prefill=True)
        assert link == "https://app.clyro.dev/agents/new" and prefilled is False
        assert "prefill skipped" in capsys.readouterr().err
