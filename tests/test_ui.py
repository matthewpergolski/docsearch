import contextlib
import io
import logging
import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path

from docsearch import ui


class _Reset(unittest.TestCase):
    def setUp(self):
        self._saved = dict(ui._state)
        self._root = list(logging.getLogger().handlers)
        os.environ["DOCSEARCH_PLAIN"] = "1"
        ui._state.update(tty=None, path=None, warnings=0)

    def tearDown(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in self._root:
            root.addHandler(h)
        logging.captureWarnings(False)
        ui._state.clear()
        ui._state.update(self._saved)
        os.environ.pop("DOCSEARCH_PLAIN", None)


class TestRetention(_Reset):
    def test_prune_keeps_newest_run_logs_and_never_touches_others(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for i in range(5):
                p = d / f"run-2026010{i}-000000.log"
                p.write_text("x")
                os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
            (d / "phase1_run.log").write_text("keep me")
            removed = ui.prune(d, 2)
            self.assertEqual(len(removed), 3)
            left = sorted(p.name for p in d.iterdir())
            self.assertEqual(left, ["phase1_run.log", "run-20260103-000000.log", "run-20260104-000000.log"])


class TestWarningsCaptured(_Reset):
    def test_warnings_go_to_run_log_are_counted_and_stay_off_screen(self):
        with tempfile.TemporaryDirectory() as t:
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                path = ui.setup(Path(t), verbose=False, keep=20)
                for _ in range(3):  # repeated identical warnings are all kept
                    warnings.warn(
                        "Provenance bbox coordinate l on page 1 is outside page bounds", UserWarning, stacklevel=1
                    )
                logging.getLogger("docling").warning("library warning")
                ui.finish()
            for h in logging.getLogger().handlers:
                h.flush()
            text = path.read_text()
            self.assertEqual(text.count("outside page bounds"), 3)
            self.assertIn("library warning", text)
            self.assertEqual(err.getvalue(), "")
            self.assertIn(f"4 warnings logged → {path}", out.getvalue())

    def test_verbose_shows_warnings_live(self):
        with tempfile.TemporaryDirectory() as t:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                ui.setup(Path(t), verbose=True, keep=20)
            levels = sorted(h.level for h in logging.getLogger().handlers if type(h) is logging.StreamHandler)
            self.assertEqual(levels, [logging.WARNING])


class TestPlainProgress(_Reset):
    def test_plain_lines_when_not_a_tty(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), ui.stage(2, 6, "index hybrid"), ui.progress(4, "embed", "chunks ") as p:
            for _ in range(4):
                p.advance()
        s = out.getvalue()
        self.assertIn("[2/6] index hybrid start", s)
        self.assertIn("4/4 chunks", s)
        self.assertIn("ETA", s)
        self.assertIn("[2/6] index hybrid done in", s)


if __name__ == "__main__":
    unittest.main()


class TestLoading(_Reset):
    def test_plain_loading_lines(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), ui.loading("reranker", "org/rr"):
            pass
        self.assertEqual(out.getvalue().splitlines(), ["loading reranker org/rr…", "reranker ready (0s)"])

    def test_tty_spinner_then_check(self):
        from rich.console import Console

        buf = io.StringIO()
        ui._state.update(tty=True, console=Console(file=buf, force_terminal=True, color_system=None, width=80))
        with ui.loading("embedder", "org/emb"):
            pass
        self.assertIn("✓ embedder ready (0s)", buf.getvalue())
