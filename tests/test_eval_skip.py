import contextlib
import io
import os
import unittest

from docsearch import ui
from docsearch.eval import DEFAULT_PLAN, skip_missing


class TestSkipMissingIndex(unittest.TestCase):
    def test_runs_without_a_table_are_skipped_with_a_message(self):
        os.environ["DOCSEARCH_PLAIN"] = "1"
        ui._state["tty"] = None
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kept = skip_missing(DEFAULT_PLAN, {"chunks_hybrid"})
        self.assertEqual([r[1] for r in kept], ["hybrid"] * 4)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("skipped control_recursive_rerank: no recursive index (run chunk --variant recursive", lines[0])
        self.assertEqual(skip_missing(DEFAULT_PLAN, {"chunks_hybrid", "chunks_recursive"}), DEFAULT_PLAN)


class TestOverall(unittest.TestCase):
    names = ["chunk hybrid", "index hybrid", "render pages", "index pages (pplx-v2-late)", "eval", "report"]

    def test_weighted_by_mac_timings(self):
        o = ui.Overall(self.names)
        self.assertEqual(o.w, [150, 1512, 501, 1578, 518, 2])
        o.at(1, 0.5)  # halfway through index text
        self.assertEqual(o.line(), "overall 2/6 · 21% · ~56 min left")
        o.at(6)
        self.assertEqual(o.line(), "overall 6/6 · 100% · ~0s left")

    def test_stored_timings_override_but_not_noop_resumes(self):
        o = ui.Overall(self.names, {"index text": 600, "render": 3})
        self.assertEqual(o.w[1], 600)  # measured last run
        self.assertEqual(o.w[2], 501)  # 3 s was a resumed no-op; keep the default


if __name__ == "__main__":
    unittest.main()
