# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for credential masking in audit records — FRD-034.

The known credential value must never appear in an emitted record, in a key or
a value position, at any depth. Value-based masking of one known secret (which
the key-based redaction module cannot do).
"""

from __future__ import annotations

import json

from clyro.config import AuditConfig
from clyro.mcp.audit import AuditLogger
from clyro.mcp.session import McpSession

TOKEN = "Bearer super-secret-abc123"


def _logger(tmp_path) -> AuditLogger:
    return AuditLogger(AuditConfig(log_path=str(tmp_path / "audit.jsonl")), McpSession().session_id)


def test_credential_masked_in_value_position(tmp_path) -> None:
    # FRD-034: a token embedded inside a value (e.g. an error message) is masked.
    log = _logger(tmp_path)
    log.set_credential_mask(TOKEN)
    log.log_tool_call(
        tool_name="t",
        parameters={"note": f"request failed with {TOKEN} attached"},
        decision="allowed",
        step_number=1,
        accumulated_cost_usd=0.0,
    )
    text = (tmp_path / "audit.jsonl").read_text()
    assert TOKEN not in text  # the secret never hit disk
    assert "[REDACTED]" in text


def test_credential_masked_at_depth_and_in_keys(tmp_path) -> None:
    log = _logger(tmp_path)
    log.set_credential_mask(TOKEN)
    log.log_tool_call(
        tool_name="t",
        parameters={"nested": {"list": [TOKEN, "safe"], TOKEN: "as-a-key"}},
        decision="allowed",
        step_number=1,
        accumulated_cost_usd=0.0,
    )
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert TOKEN not in json.dumps(rec)


def test_no_mask_registered_is_noop(tmp_path) -> None:
    # Without a registered credential, behaviour is unchanged (stdio default).
    log = _logger(tmp_path)
    log.log_tool_call(
        tool_name="t", parameters={"x": 1}, decision="allowed",
        step_number=1, accumulated_cost_usd=0.0,
    )
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert rec["tool_name"] == "t"  # writes normally


def test_none_credential_disables_mask(tmp_path) -> None:
    log = _logger(tmp_path)
    log.set_credential_mask(None)
    assert log._mask_credential({"a": "b"}) == {"a": "b"}
