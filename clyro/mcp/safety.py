# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — Outbound Safety Floor (native HTTP transport)
# Implements FRD-035, FRD-036, FRD-037, FRD-038, FRD-039, FRD-052, FRD-057

"""
Outbound safety floor for the native HTTP transport.

Before the wrapper connects to any remote MCP target — and again before it
follows any redirect hop — the target passes through here. The floor:

- resolves the hostname to an IP **locally** (the wrapper controls resolution,
  which is what lets FRD-057 ignore any proxy and still enforce the floor);
- refuses the cloud-metadata address (FRD-035) and link-local ranges (FRD-036),
  after resolution, so a hostname that resolves to ``169.254.169.254`` is caught;
- permits private / RFC1918 ranges — internal servers are the target use case
  (FRD-038);
- permits plaintext and loopback targets only under the single operator
  relaxation (FRD-039, D6);
- is re-run on every redirect hop, bounded at :data:`REDIRECT_MAX_HOPS`
  (FRD-037/052).

The verdict carries the resolved IP so the caller connects to *that* IP (with
SNI/Host preserved), closing the DNS-rebinding gap (TDD T-2). If any resolved
address is dangerous the whole target is refused, defeating multi-record tricks.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from clyro.mcp.log import get_logger

logger = get_logger(__name__)

# FRD-037 (D12): a redirect chain is bounded at 5 hops. Fixed, not configurable.
REDIRECT_MAX_HOPS = 5

# FRD-035: cloud metadata endpoints, refused explicitly with their own reason.
# The IPv4 address is a subset of link-local; the AWS IPv6 address is a ULA
# (fc00::/7) that would otherwise be permitted as a private range, so it must
# be matched here (before the private-range allow) or it slips through (R-6).
_CLOUD_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS / GCP / Azure IMDS (IPv4)
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS (IPv6)
        ipaddress.ip_address("192.0.0.192"),  # Oracle Cloud IMDS
        ipaddress.ip_address("100.100.100.200"),  # Alibaba Cloud IMDS
    }
)

# FRD-035: NAT64 (RFC 6052) embeds an IPv4 address in an IPv6 one, so
# 64:ff9b::a9fe:a9fe reaches 169.254.169.254 in a NAT64 environment. Neither the
# metadata set nor is_link_local catches that form, so the prefix is matched and
# its embedded IPv4 re-checked below.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


def _embedded_ipv4(ip: ipaddress._BaseAddress) -> ipaddress.IPv4Address | None:
    """Return the IPv4 address embedded in a NAT64 address, if any (RFC 6052)."""
    if isinstance(ip, ipaddress.IPv6Address) and ip in _NAT64_PREFIX:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


# A resolver maps a hostname to a list of IP strings. Injectable for hermetic tests.
Resolver = Callable[[str], list[str]]


class FloorOutcome(Enum):
    ALLOW = "allow"
    REFUSE = "refuse"


class FloorReason(Enum):
    OK = "ok"
    METADATA = "metadata"
    LINK_LOCAL = "link_local"
    PLAINTEXT_DISALLOWED = "plaintext_disallowed"
    LOOPBACK_DISALLOWED = "loopback_disallowed"
    UNRESOLVABLE = "unresolvable"
    BAD_URL = "bad_url"


@dataclass(frozen=True)
class SafetyVerdict:
    """Result of validating one target (or redirect hop) against the floor."""

    outcome: FloorOutcome
    reason: FloorReason
    resolved_ip: str | None = None
    host: str | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome is FloorOutcome.ALLOW


def _is_ip_literal(host: str) -> bool:
    """True if *host* is already an IP address (no DNS needed)."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _default_resolver(host: str) -> list[str]:
    """Resolve *host* to a de-duplicated list of IP strings via ``getaddrinfo``.

    IP literals are handled in :meth:`SafetyFloor.validate` before the resolver
    is consulted, so this only ever sees real hostnames.
    """
    infos = socket.getaddrinfo(host, None)
    seen: list[str] = []
    for info in infos:
        ip = info[4][0]
        if ip not in seen:
            seen.append(ip)
    return seen


class SafetyFloor:
    """Validate outbound targets and redirect hops. Implements FRD-035/036/038/039/052.

    The redirect *bound* (FRD-037) is exposed as :data:`REDIRECT_MAX_HOPS`; the
    follow-loop that enforces it lives in the transport (it re-calls
    :meth:`validate` per hop with ``after_redirect=True``).
    """

    def __init__(self, *, allow_plaintext: bool = False, resolver: Resolver | None = None) -> None:
        self._allow_plaintext = allow_plaintext
        self._resolver = resolver or _default_resolver

    def validate(self, url: str, *, after_redirect: bool = False) -> SafetyVerdict:
        """Resolve and classify *url*; return a :class:`SafetyVerdict`.

        Same checks apply to the initial target and to every redirect hop
        (FRD-052); ``after_redirect`` only annotates the log line.
        """
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        host = parts.hostname

        if not host or scheme not in ("http", "https"):
            return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.BAD_URL, host=host)

        # FRD-039: a plaintext (http) target is refused unless the operator has
        # enabled the single relaxation. Loopback is handled after resolution.
        if scheme == "http" and not self._allow_plaintext:
            logger.warning("floor_refuse", reason="plaintext_disallowed", host=host)
            return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.PLAINTEXT_DISALLOWED, host=host)

        # An IP-literal target is handled by the floor itself, independent of the
        # resolver — the floor must always know how to classify a literal address
        # and must never delegate that to an injected resolver.
        if _is_ip_literal(host):
            ips = [host]
        else:
            try:
                ips = self._resolver(host)
            except OSError:
                return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.UNRESOLVABLE, host=host)
            if not ips:
                return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.UNRESOLVABLE, host=host)

        # Validate EVERY resolved address; a single dangerous record refuses the
        # whole target (defeats a multi-A DNS-rebinding trick).
        for ip_str in ips:
            ip = ipaddress.ip_address(ip_str)
            # FRD-035: a NAT64 address carries an IPv4 target inside it — check
            # the embedded address too, or 64:ff9b::a9fe:a9fe reaches the
            # metadata endpoint while matching none of the tests below.
            embedded = _embedded_ipv4(ip)
            if embedded is not None and (embedded in _CLOUD_METADATA_IPS or embedded.is_link_local):
                logger.warning("floor_refuse", reason="metadata", host=host, ip=ip_str)
                return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.METADATA, ip_str, host)
            if ip in _CLOUD_METADATA_IPS:  # FRD-035
                logger.warning("floor_refuse", reason="metadata", host=host, ip=ip_str)
                return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.METADATA, ip_str, host)
            if ip.is_link_local:  # FRD-036
                logger.warning("floor_refuse", reason="link_local", host=host, ip=ip_str)
                return SafetyVerdict(FloorOutcome.REFUSE, FloorReason.LINK_LOCAL, ip_str, host)
            # FRD-039 (loopback arm). ``is_unspecified`` (0.0.0.0, ::) is NOT
            # is_loopback, yet 0.0.0.0 routes to localhost on Linux — so without
            # this it slips the gate the relaxation is meant to control.
            if (ip.is_loopback or ip.is_unspecified) and not self._allow_plaintext:
                return SafetyVerdict(
                    FloorOutcome.REFUSE, FloorReason.LOOPBACK_DISALLOWED, ip_str, host
                )

        # FRD-038: private (and public non-metadata) targets are allowed. Pin the
        # first resolved IP — it was validated above — so the caller connects to
        # exactly what the floor cleared.
        if after_redirect:
            logger.debug("floor_allow_redirect", host=host, ip=ips[0])
        return SafetyVerdict(FloorOutcome.ALLOW, FloorReason.OK, ips[0], host)
