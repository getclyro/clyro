# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the outbound SafetyFloor —
FRD-035, FRD-036, FRD-037, FRD-038, FRD-039, FRD-052, FRD-057.

Hermetic: a fake resolver is injected so no real DNS is performed.
"""

from __future__ import annotations

import pytest

from clyro.mcp.safety import (
    REDIRECT_MAX_HOPS,
    FloorReason,
    SafetyFloor,
)


def _fixed_resolver(mapping: dict[str, list[str]]):
    """Return a resolver that maps host -> IPs from *mapping* (else raises)."""

    def resolve(host: str) -> list[str]:
        if host not in mapping:
            raise OSError(f"no such host: {host}")
        return mapping[host]

    return resolve


class TestRefusedTargets:
    """Metadata and link-local are refused after resolution (FRD-035/036)."""

    def test_cloud_metadata_ip_refused(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"evil": ["169.254.169.254"]}))
        v = floor.validate("https://evil/mcp")
        assert not v.allowed
        assert v.reason is FloorReason.METADATA  # FRD-035

    def test_metadata_via_hostname_resolution_refused(self) -> None:
        # A benign-looking hostname that resolves to the metadata IP is caught
        # because the floor resolves locally and validates the resolved IP.
        floor = SafetyFloor(resolver=_fixed_resolver({"internal.example": ["169.254.169.254"]}))
        assert floor.validate("https://internal.example/").reason is FloorReason.METADATA

    def test_link_local_refused(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["169.254.10.5"]}))
        assert floor.validate("https://h/").reason is FloorReason.LINK_LOCAL  # FRD-036

    def test_ipv6_aws_metadata_ula_refused(self) -> None:
        # R-6: the AWS IPv6 IMDS address is a ULA that would otherwise be allowed
        # as a private range — it must be refused as metadata (checked first).
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["fd00:ec2::254"]}))
        assert floor.validate("https://h/").reason is FloorReason.METADATA  # FRD-035

    def test_multi_record_with_one_dangerous_is_refused(self) -> None:
        # Anti-rebinding: a good + a metadata record still refuses the whole target.
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["10.0.0.5", "169.254.169.254"]}))
        assert floor.validate("https://h/").reason is FloorReason.METADATA


class TestAllowedTargets:
    """Private and public non-metadata targets are allowed (FRD-038)."""

    @pytest.mark.parametrize("ip", ["10.0.0.5", "172.16.3.4", "192.168.1.10"])
    def test_private_ranges_allowed(self, ip: str) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"srv": [ip]}))
        v = floor.validate("https://srv/mcp")
        assert v.allowed  # FRD-038
        assert v.resolved_ip == ip  # pinned to the validated IP (anti-rebinding)

    def test_public_https_allowed(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"pub": ["1.1.1.1"]}))
        assert floor.validate("https://pub/").allowed

    def test_ip_literal_target_needs_no_dns(self) -> None:
        # No resolver entry; an IP literal is used as-is.
        floor = SafetyFloor(resolver=_fixed_resolver({}))
        v = floor.validate("https://10.1.2.3/mcp")
        assert v.allowed and v.resolved_ip == "10.1.2.3"


class TestPlaintextRelaxation:
    """Plaintext and loopback require the single relaxation (FRD-039)."""

    def test_plaintext_refused_without_relaxation(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["10.0.0.1"]}))
        assert floor.validate("http://h/").reason is FloorReason.PLAINTEXT_DISALLOWED

    def test_plaintext_allowed_with_relaxation(self) -> None:
        floor = SafetyFloor(
            allow_plaintext=True, resolver=_fixed_resolver({"h": ["10.0.0.1"]})
        )
        assert floor.validate("http://h/").allowed

    def test_loopback_refused_without_relaxation(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"lo": ["127.0.0.1"]}))
        assert floor.validate("https://lo/").reason is FloorReason.LOOPBACK_DISALLOWED

    def test_loopback_allowed_with_relaxation(self) -> None:
        floor = SafetyFloor(
            allow_plaintext=True, resolver=_fixed_resolver({"lo": ["127.0.0.1"]})
        )
        assert floor.validate("https://lo/").allowed

    def test_metadata_still_refused_even_with_relaxation(self) -> None:
        # The relaxation must NOT weaken the metadata/link-local floor (AC-10.3).
        floor = SafetyFloor(
            allow_plaintext=True, resolver=_fixed_resolver({"h": ["169.254.169.254"]})
        )
        assert floor.validate("http://h/").reason is FloorReason.METADATA


class TestMalformedAndUnresolvable:
    def test_non_http_scheme_refused(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({}))
        assert floor.validate("ftp://h/").reason is FloorReason.BAD_URL

    def test_missing_host_refused(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({}))
        assert floor.validate("https:///path").reason is FloorReason.BAD_URL

    def test_unresolvable_host_refused(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({}))
        assert floor.validate("https://nope/").reason is FloorReason.UNRESOLVABLE


class TestRedirectContract:
    """Redirect bound (FRD-037) and per-hop re-validation (FRD-052)."""

    def test_redirect_bound_is_five(self) -> None:
        assert REDIRECT_MAX_HOPS == 5  # FRD-037 (D12)

    def test_redirect_hop_to_metadata_refused(self) -> None:
        # A redirect destination is a target: same floor applies (FRD-052).
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["169.254.169.254"]}))
        v = floor.validate("https://h/", after_redirect=True)
        assert v.reason is FloorReason.METADATA

    def test_redirect_hop_to_private_allowed(self) -> None:
        floor = SafetyFloor(resolver=_fixed_resolver({"h": ["10.9.9.9"]}))
        assert floor.validate("https://h/", after_redirect=True).allowed
