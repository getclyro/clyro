# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

"""A transport failure at startup exits cleanly (non-zero), never a traceback."""

from __future__ import annotations

import pytest

from clyro.mcp.cli import _async_main


@pytest.mark.asyncio
async def test_floor_refusal_exits_cleanly(tmp_path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f'default_action: allow\naudit:\n  log_path: "{tmp_path}/a.jsonl"\n')
    # metadata target -> SafetyFloor refuses in open(); CLI must return 1, not raise.
    code = await _async_main(
        [], str(cfg), transport="http", url="http://169.254.169.254/", allow_plaintext=True
    )
    assert code == 1
