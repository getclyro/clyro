# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — TLS trust policy (native HTTP transport)
# Implements FRD-040, FRD-046, FRD-047

"""
TLS trust policy for the native HTTP transport.

- Verify the server certificate against the **system trust store** by default
  (FRD-040).
- Verify against an operator-supplied **CA bundle** when one is provided
  (FRD-046) — for privately-signed internal servers.
- Expose **no** control that disables verification (FRD-047): this class can
  only ever yield a trust source, never ``False``. Its very design forbids a
  bypass, which is why there is no ``insecure`` / ``verify=False`` parameter
  anywhere in its surface.

The value returned by :meth:`verify_value` is passed straight to the HTTP
client's ``verify=`` argument (``True`` = system store, or a CA-bundle path).
"""

from __future__ import annotations

from pathlib import Path

from clyro.mcp.log import get_logger
from clyro.mcp.server_transport import TransportError

logger = get_logger(__name__)


class TlsPolicy:
    """Resolve the TLS trust source for outbound connections (FRD-040/046/047)."""

    def __init__(self, ca_bundle_path: str | Path | None = None) -> None:
        if ca_bundle_path is None:
            self._ca_bundle: Path | None = None
            return
        bundle = Path(ca_bundle_path)
        if not bundle.is_file():
            # FRD-046 failure: a supplied bundle that does not exist must fail,
            # not silently fall back to the system store (which could trust a
            # cert the operator did not intend).
            raise TransportError(f"TLS CA bundle not found: {bundle}")
        self._ca_bundle = bundle

    def verify_value(self) -> bool | str:
        """Return the HTTP client's ``verify=`` value.

        ``True`` = system trust store (FRD-040); a path = the CA bundle
        (FRD-046). Never ``False`` — verification cannot be disabled (FRD-047).
        """
        if self._ca_bundle is not None:
            return str(self._ca_bundle)
        return True
