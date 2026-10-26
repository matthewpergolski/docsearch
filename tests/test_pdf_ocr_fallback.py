import unittest
from pathlib import Path

from docsearch.config import Config
from docsearch.ingest import pdf_ocr_on, pdf_options


class TestPdfOcr(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(pdf_ocr_on(Config(raw={}, root=Path("."))))
        self.assertFalse(pdf_options().do_ocr)

    def test_setting_turns_it_on(self):
        self.assertTrue(pdf_ocr_on(Config(raw={"ingest": {"pdf_ocr": True}}, root=Path("."))))

    def test_on_is_region_by_region_not_the_full_page(self):
        o = pdf_options(pdf_ocr=True)
        self.assertTrue(o.do_ocr)
        self.assertFalse(o.ocr_options.force_full_page_ocr)  # typed text keeps its text layer

    def test_on_allows_time_for_a_long_scan(self):
        self.assertGreaterEqual(pdf_options(600, pdf_ocr=True).document_timeout, 3600)


class TestCrashGuard(unittest.TestCase):
    """parse_pdf_isolated with a stand-in for subprocess.run: no model and no child process is started."""

    def runner(self, failing):
        import json
        from types import SimpleNamespace

        calls = []

        def run(cmd, capture_output, timeout):
            out, mode = cmd[-2], cmd[-1]
            calls.append(mode)
            if mode in failing:
                return SimpleNamespace(returncode=11)  # the child died with a segfault
            Path(out).write_text(json.dumps({"status": "success", "errors": [], "doc": {"mode": mode}}))
            return SimpleNamespace(returncode=0)

        return run, calls

    def parse(self, failing):
        from docsearch.ingest import parse_pdf_isolated

        run, calls = self.runner(failing)
        payload, note = parse_pdf_isolated(Path("x.pdf"), run=run)
        return payload, note, calls

    def test_region_mode_is_used_when_it_works(self):
        payload, note, calls = self.parse(set())
        self.assertEqual((payload["doc"]["mode"], note, calls), ("region", "", ["region"]))

    def test_a_crash_in_region_mode_falls_back_to_full_page_ocr(self):
        payload, note, calls = self.parse({"region"})
        self.assertEqual((payload["doc"]["mode"], calls), ("full", ["region", "full"]))
        self.assertIn("region failed (exit 11)", note)

    def test_no_ocr_is_the_last_resort(self):
        payload, _, calls = self.parse({"region", "full"})
        self.assertEqual((payload["doc"]["mode"], calls), ("off", ["region", "full", "off"]))

    def test_all_modes_failing_returns_no_document(self):
        payload, note, _ = self.parse({"region", "full", "off"})
        self.assertIsNone(payload)
        self.assertEqual(note.count("failed"), 3)

    def test_full_page_option(self):
        self.assertTrue(pdf_options(pdf_ocr=True, full_page=True).ocr_options.force_full_page_ocr)


if __name__ == "__main__":
    unittest.main()
