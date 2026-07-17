# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for outbound auth (CredentialProvider) —
FRD-033, FRD-051, FRD-054, FRD-055.
"""

from __future__ import annotations

from clyro.mcp.auth import (
    AttachOutcome,
    AuthOutcome,
    AuthProvider,
    CredentialProvider,
)

ORIGIN = "https://srv.internal"
TOKEN = "Bearer secret-123"


def _attach(cp, *, target=ORIGIN, request=ORIGIN, encrypted=True):
    headers: dict[str, str] = {}
    outcome = cp.attach(
        headers, target_origin=target, request_origin=request, encrypted=encrypted
    )
    return outcome, headers


def test_conforms_to_authprovider_protocol() -> None:
    assert isinstance(CredentialProvider(TOKEN), AuthProvider)


class TestAttach:
    def test_attaches_over_https_same_origin(self) -> None:
        # FRD-033: present the credential.
        outcome, headers = _attach(CredentialProvider(TOKEN))
        assert outcome is AttachOutcome.ATTACHED
        assert headers["Authorization"] == TOKEN

    def test_no_credential_configured_is_omitted(self) -> None:
        outcome, headers = _attach(CredentialProvider(None))
        assert outcome is AttachOutcome.OMITTED
        assert headers == {}

    def test_withheld_on_cross_origin_redirect(self) -> None:
        # FRD-051: token must not follow a redirect to a different origin.
        outcome, headers = _attach(
            CredentialProvider(TOKEN), target="https://evil.example", request=ORIGIN
        )
        assert outcome is AttachOutcome.OMITTED
        assert "Authorization" not in headers

    def test_refused_over_plaintext_non_loopback(self) -> None:
        # FRD-055: never send a credential over an unencrypted non-loopback wire.
        outcome, headers = _attach(
            CredentialProvider(TOKEN),
            target="http://10.0.0.5",
            request="http://10.0.0.5",
            encrypted=False,
        )
        assert outcome is AttachOutcome.REFUSED
        assert "Authorization" not in headers

    def test_allowed_over_plaintext_loopback(self) -> None:
        # AC-8.7: plaintext loopback may carry the credential (nothing on a wire).
        outcome, headers = _attach(
            CredentialProvider(TOKEN),
            target="http://127.0.0.1",
            request="http://127.0.0.1",
            encrypted=False,
        )
        assert outcome is AttachOutcome.ATTACHED

    def test_custom_header_name(self) -> None:
        cp = CredentialProvider("k-abc", header_name="X-API-Key")
        _, headers = _attach(cp)
        assert headers == {"X-API-Key": "k-abc"}


class TestMidSessionAuth:
    """FRD-054: detect a mid-session credential rejection."""

    def test_401_is_auth_failed(self) -> None:
        assert CredentialProvider(TOKEN).classify_response(401) is AuthOutcome.AUTH_FAILED

    def test_403_is_auth_failed(self) -> None:
        assert CredentialProvider(TOKEN).classify_response(403) is AuthOutcome.AUTH_FAILED

    def test_200_is_ok(self) -> None:
        assert CredentialProvider(TOKEN).classify_response(200) is AuthOutcome.OK


def test_credential_value_exposed_for_redaction() -> None:
    # FRD-034 depends on masking the transmitted representation.
    assert CredentialProvider(TOKEN).credential_value == TOKEN
    assert CredentialProvider(None).credential_value is None
