# Implements NFR-001: Verify subpackage isolation
"""
Verify that importing one subpackage does not trigger loading of another.
"""

import subprocess
import sys


class TestSubpackageIsolation:
    """Ensure clyro.mcp and clyro.hooks don't cross-load."""

    def test_mcp_import_does_not_load_hooks(self):
        """Importing clyro.mcp should NOT load clyro.hooks."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import clyro.mcp; import sys; "
                "assert 'clyro.hooks' not in sys.modules, "
                "'clyro.hooks was loaded when only clyro.mcp was imported'"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

    def test_hooks_import_does_not_load_mcp(self):
        """Importing clyro.hooks should NOT load clyro.mcp."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import clyro.hooks; import sys; "
                "assert 'clyro.mcp' not in sys.modules, "
                "'clyro.mcp was loaded when only clyro.hooks was imported'"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
