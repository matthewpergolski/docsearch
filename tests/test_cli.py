import contextlib
import io
import re
import unittest
from pathlib import Path

from docsearch.cli import main

ROOT = Path(__file__).resolve().parents[1]


class TestCliFlags(unittest.TestCase):
    def test_flags_used_by_run_scripts_exist(self):
        """Every `docsearch <cmd> --flag` in run_*.sh must be accepted (guards against detached-run crashes)."""
        for script in ROOT.glob("run_*.sh"):
            text = script.read_text().replace("\\\n", " ")
            for cmd, rest in re.findall(r"docsearch (\w[\w-]*)([^\n]*)", text):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                    main([cmd, "--help"])
                for flag in re.findall(r"(--[a-z][\w-]*)", rest):
                    self.assertIn(flag, buf.getvalue(), f"{script.name}: `{cmd} {flag}` not accepted")


if __name__ == "__main__":
    unittest.main()
