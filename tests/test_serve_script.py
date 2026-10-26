"""scripts/serve.sh keeps the torch extra the user synced with (cpu or cuda), so `uv run` does not re-sync to the
default torch. Fake uv and bun on PATH record their arguments; nothing is built or served."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "scripts" / "serve.sh"


class TestServeScript(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = Path(self.td.name)
        self.log = self.bin / "uv.args"
        (self.bin / "uv").write_text(f'#!/bin/sh\necho "$@" > "{self.log}"\n')
        (self.bin / "bun").write_text("#!/bin/sh\nexit 0\n")
        for f in ("uv", "bun"):
            (self.bin / f).chmod(0o755)

    def tearDown(self):
        self.td.cleanup()

    def serve(self, *args: str, torch: str | None = None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if k != "DOCSEARCH_TORCH"}
        env["PATH"] = f"{self.bin}:/usr/bin:/bin"
        if torch is not None:
            env["DOCSEARCH_TORCH"] = torch
        return subprocess.run(["bash", str(SERVE), *args], env=env, capture_output=True, text=True, timeout=30)

    def test_default_adds_only_the_ui_extra(self):
        r = self.serve("--port", "9000")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.log.read_text().split(), ["run", "--extra", "ui", "docsearch", "serve", "--port", "9000"])

    def test_torch_extra_is_kept(self):
        for extra in ("cpu", "cuda"):
            r = self.serve(torch=extra)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(
                self.log.read_text().split(), ["run", "--extra", "ui", "--extra", extra, "docsearch", "serve"]
            )

    def test_unknown_torch_extra_stops(self):
        r = self.serve(torch="rocm")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DOCSEARCH_TORCH must be cpu or cuda", r.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
