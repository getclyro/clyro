# Copyright 2026 Clyro Inc.
# SPDX-License-Identifier: Apache-2.0

# Clyro MCP Wrapper — python -m entrypoint
# Implements FRD-009

"""Allow running the wrapper with ``python -m clyro.mcp``."""

from clyro.mcp.cli import main

if __name__ == "__main__":
    main()
