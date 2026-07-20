# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for TlsPolicy — FRD-040, FRD-046, FRD-047.
"""

from __future__ import annotations

import inspect

import pytest

from clyro.mcp.server_transport import TransportError
from clyro.mcp.tls import TlsPolicy


def test_default_uses_system_store() -> None:
    # FRD-040: no CA bundle -> system trust store (verify=True).
    assert TlsPolicy().verify_value() is True


def test_ca_bundle_used_when_provided(tmp_path) -> None:
    # FRD-046: a real bundle path is passed through as the verify value.
    bundle = tmp_path / "ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n")
    assert TlsPolicy(bundle).verify_value() == str(bundle)


def test_missing_ca_bundle_fails(tmp_path) -> None:
    # FRD-046 failure: a supplied-but-missing bundle must not fall back silently.
    with pytest.raises(TransportError):
        TlsPolicy(tmp_path / "does-not-exist.pem")


class TestNoBypass:
    """FRD-047: no control disables verification, at any default."""

    def test_no_insecure_parameter_exists(self) -> None:
        params = set(inspect.signature(TlsPolicy.__init__).parameters)
        for forbidden in ("insecure", "verify", "verify_disable", "no_verify", "skip_verify"):
            assert forbidden not in params

    def test_verify_value_is_never_false(self, tmp_path) -> None:
        assert TlsPolicy().verify_value() is not False
        bundle = tmp_path / "ca.pem"
        bundle.write_text("x")
        assert TlsPolicy(bundle).verify_value() is not False
