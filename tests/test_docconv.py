"""Old Word files (.doc): converter choice, conversion through a fake program, the cache, and the parse path."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from docsearch import docconv
from docsearch.config import load_config
from docsearch.docconv import Converter, DocConvertError, pick, to_docx
from docsearch.manifest import Doc

OLE = bytes.fromhex("d0cf11e0a1b11ae1")  # the first 8 bytes of a Word 97-2003 file


def which_of(*found):
    """A fake shutil.which that finds only these program names, at /bin/<name>."""
    return lambda name: f"/bin/{name}" if name in found else None


class FakeRun:
    """A fake subprocess.run: writes the .docx the real program would write, or fails as told."""

    def __init__(self, code=0, stderr="", timeout=False, write=True):
        self.code, self.stderr, self.timeout, self.write = code, stderr, timeout, write
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        if self.write and self.code == 0:
            if "-output" in cmd:
                out = Path(cmd[cmd.index("-output") + 1])
            else:
                out = Path(cmd[cmd.index("--outdir") + 1]) / (Path(cmd[-1]).stem + ".docx")
            out.write_bytes(b"PK converted from " + Path(cmd[-1]).name.encode())
        return SimpleNamespace(returncode=self.code, stdout="", stderr=self.stderr)


class TestPick(unittest.TestCase):
    def test_auto_prefers_libreoffice_then_textutil(self):
        self.assertEqual(pick("auto", which_of("soffice", "textutil")), (Converter("soffice", "/bin/soffice"), ""))
        self.assertEqual(pick("auto", which_of("libreoffice", "textutil")),
                         (Converter("soffice", "/bin/libreoffice"), ""))  # fmt: skip
        self.assertEqual(pick("auto", which_of("textutil")), (Converter("textutil", "/bin/textutil"), ""))

    def test_a_named_converter_is_the_only_one_tried(self):
        self.assertEqual(pick("textutil", which_of("soffice", "textutil")),
                         (Converter("textutil", "/bin/textutil"), ""))  # fmt: skip
        self.assertEqual(pick("soffice", which_of("textutil")),
                         (None, "soffice not found (install LibreOffice or set ingest.doc_converter)"))  # fmt: skip

    def test_off_and_none_found_give_the_skip_reason(self):
        self.assertEqual(pick("off", which_of("soffice")), (None, "ingest.doc_converter is off"))
        self.assertEqual(pick("auto", which_of()),
                         (None, "no converter (install LibreOffice or set ingest.doc_converter)"))  # fmt: skip

    def test_config_rejects_an_unknown_converter(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "docsearch.yaml"
            p.write_text("ingest: {doc_converter: word}\n")
            with self.assertRaisesRegex(ValueError, "ingest.doc_converter must be one of auto, soffice, textutil, off"):
                load_config(p, env={})
            p.write_text("ingest: {doc_converter: off}\n")  # YAML reads a bare off as false
            self.assertEqual(load_config(p, env={})["ingest"]["doc_converter"], "off")


class TestToDocx(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.src = self.dir / "in" / "report.doc"
        self.src.parent.mkdir()
        self.src.write_bytes(OLE + b" old word")
        self.cache = self.dir / "cache" / "doc_docx"

    def tearDown(self):
        self.td.cleanup()

    def test_textutil_writes_the_cached_docx(self):
        run = FakeRun()
        out = to_docx(self.src, "abc", self.cache, Converter("textutil", "/usr/bin/textutil"), run)
        self.assertEqual(out, self.cache / "abc-textutil.docx")
        self.assertEqual(out.read_bytes(), b"PK converted from report.doc")
        cmd = run.cmds[0]
        self.assertEqual(cmd[:4] + cmd[5:], ["/usr/bin/textutil", "-convert", "docx", "-output", str(self.src)])
        self.assertEqual(Path(cmd[4]).name, "report.docx")
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()), ["abc-textutil.docx"])  # temp folder gone

    def test_soffice_runs_headless_with_its_own_profile(self):
        run = FakeRun()
        out = to_docx(self.src, "abc", self.cache, Converter("soffice", "/bin/soffice"), run)
        self.assertEqual(out.read_bytes(), b"PK converted from report.doc")
        cmd = run.cmds[0]
        self.assertEqual([cmd[0], cmd[1], cmd[2], cmd[4], cmd[5], cmd[6], cmd[8]],
                         ["/bin/soffice", "--headless", "--norestore", "--convert-to", "docx", "--outdir",
                          str(self.src)])  # fmt: skip
        self.assertTrue(cmd[3].startswith("-env:UserInstallation=file:///"))

    def test_same_sha_is_a_cache_hit_new_sha_converts_again(self):
        run = FakeRun()
        conv = Converter("soffice", "/bin/soffice")
        first = to_docx(self.src, "abc", self.cache, conv, run)
        self.assertEqual(to_docx(self.src, "abc", self.cache, conv, run), first)
        self.assertEqual(len(run.cmds), 1)
        self.assertEqual(to_docx(self.src, "def", self.cache, conv, run), self.cache / "def-soffice.docx")
        self.assertEqual(len(run.cmds), 2)

    def test_a_failed_conversion_names_the_program_and_its_last_error_line(self):
        run = FakeRun(code=1, stderr="Fontconfig warning\nError: source file could not be loaded\n")
        with self.assertRaises(DocConvertError) as e:
            to_docx(self.src, "abc", self.cache, Converter("soffice", "/bin/soffice"), run)
        self.assertEqual(str(e.exception),
                         "soffice could not convert report.doc (exit 1): Error: source file could not be loaded")  # fmt: skip
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_exit_zero_without_output_is_a_failure(self):
        with self.assertRaises(DocConvertError) as e:
            to_docx(self.src, "abc", self.cache, Converter("soffice", "/bin/soffice"), FakeRun(write=False))
        self.assertEqual(str(e.exception), "soffice could not convert report.doc (exit 0): no .docx written")

    def test_bytes_that_are_not_word_rtf_or_html_never_reach_the_converter(self):
        run = FakeRun()
        self.src.write_bytes(b"\x8f\x02random bytes")
        with self.assertRaises(DocConvertError) as e:
            to_docx(self.src, "abc", self.cache, Converter("soffice", "/bin/soffice"), run)
        self.assertEqual(str(e.exception), "report.doc is not a Word file (no Word 97-2003, RTF or HTML header)")
        self.assertEqual(run.cmds, [])
        for head in (b"{\\rtf1\\ansi hello}", b"\xef\xbb\xbf  <html><body>hi</body></html>"):
            self.src.write_bytes(head)
            to_docx(self.src, head[:4].hex(), self.cache, Converter("soffice", "/bin/soffice"), run)
        self.assertEqual(len(run.cmds), 2)

    def test_ooxml_and_utf16_html_saved_as_doc_are_word_files(self):
        # CR2: some exporters write a .docx zip or a UTF-16 web page under a .doc name.
        run = FakeRun()
        self.src.write_bytes(b"PK\x03\x04 a docx zip")
        self.assertEqual(to_docx(self.src, "zip", self.cache, Converter("soffice", "/bin/soffice"), run).read_bytes(),
                         b"PK\x03\x04 a docx zip")  # fmt: skip
        self.assertEqual(run.cmds, [])
        self.src.write_bytes("\ufeff<html><body>hi</body></html>".encode("utf-16-le"))
        to_docx(self.src, "u16", self.cache, Converter("soffice", "/bin/soffice"), run)
        self.assertEqual(len(run.cmds), 1)

    def test_soffice_keeps_one_profile_per_process_in_the_cache(self):
        # CR2: a fresh profile per file pays LibreOffice's first start every time.
        run = FakeRun()
        conv = Converter("soffice", "/bin/soffice")
        to_docx(self.src, "a", self.cache, conv, run)
        to_docx(self.src, "b", self.cache, conv, run)
        profiles = {c[3] for c in run.cmds}
        self.assertEqual(len(profiles), 1)
        self.assertTrue(profiles.pop().endswith(f"/soffice-profile-{os.getpid()}"))

    def test_render_and_parse_share_one_soffice_command(self):
        # CR2: render used bare `soffice` and the user's profile; a running LibreOffice window took the job.
        from docsearch import visual

        run = FakeRun()
        with mock.patch("docsearch.docconv.shutil.which", which_of("libreoffice")), \
             mock.patch("docsearch.visual.subprocess.run", run):  # fmt: skip
            visual._office_pdf(self.src, self.dir / "pdf", self.cache)
        cmd = run.cmds[0]
        self.assertEqual(cmd[:3], ["/bin/libreoffice", "--headless", "--norestore"])
        self.assertTrue(cmd[3].endswith(f"/soffice-profile-{os.getpid()}"))
        self.assertEqual(cmd[4:6], ["--convert-to", "pdf"])

    def test_a_timeout_is_a_clear_failure(self):
        with self.assertRaises(DocConvertError) as e:
            to_docx(self.src, "abc", self.cache, Converter("textutil", "/usr/bin/textutil"), FakeRun(timeout=True),
                    timeout=120)  # fmt: skip
        self.assertEqual(str(e.exception), "textutil timed out after 120 s on report.doc")


class FakeDocling:
    """Stands in for Docling's DocumentConverter: records the file it was given and returns a one-line document."""

    def __init__(self):
        self.paths: list[Path] = []

    def convert(self, path, raises_on_error=False):
        from docling_core.types.doc import DocItemLabel, DoclingDocument

        self.paths.append(Path(path))
        doc = DoclingDocument(name=Path(path).stem)
        doc.add_text(label=DocItemLabel.TEXT, text="Quarterly staff leave policy.")
        return SimpleNamespace(status=SimpleNamespace(value="success"), errors=[], document=doc)


class TestParseDoc(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n")
        self.cfg = load_config(root / "docsearch.yaml", env={})
        src = self.cfg.path("corpus_dir") / "inbox" / "policy.doc"
        src.parent.mkdir(parents=True)
        src.write_bytes(OLE + b" old word")
        self.doc = Doc("inbox/policy.doc", "inbox/policy.doc", "docx", "inbox", "")

    def tearDown(self):
        self.td.cleanup()

    def parse(self, which=None, run=None):
        from docsearch.ingest import _parse_one, file_sha256, parsed_path

        conv = FakeDocling()
        with mock.patch.object(docconv.shutil, "which", which or which_of("soffice")), \
             mock.patch.object(docconv.subprocess, "run", run or FakeRun()):  # fmt: skip
            row = _parse_one(self.cfg, conv, self.doc, parsed_path(self.cfg, self.doc.doc_id))
        return row, conv, file_sha256(self.cfg.path("corpus_dir") / self.doc.path)

    def test_parses_the_converted_docx_and_keeps_the_doc_identity(self):
        from docsearch.ingest import parsed_path

        row, conv, sha = self.parse()
        self.assertEqual(conv.paths, [self.cfg.path("cache_dir") / "doc_docx" / f"{sha}-soffice.docx"])
        self.assertEqual((row["doc_id"], row["format"], row["status"], row["sha256"]),
                         ("inbox/policy.doc", "docx", "ok", sha))  # fmt: skip
        saved = json.loads(parsed_path(self.cfg, "inbox/policy.doc").read_text())
        self.assertEqual(saved["texts"][0]["text"], "Quarterly staff leave policy.")

    def test_no_converter_or_a_failed_one_fails_the_parse_with_the_reason(self):
        row, conv, _ = self.parse(which=which_of())
        self.assertEqual((row["status"], row["error"]),
                         ("failed", "DocConvertError: policy.doc: no converter (install LibreOffice or set "
                                    "ingest.doc_converter)"))  # fmt: skip
        self.assertEqual(conv.paths, [])
        row, _, _ = self.parse(run=FakeRun(code=77, stderr="Error: boom"))
        self.assertEqual(row["error"], "DocConvertError: soffice could not convert policy.doc (exit 77): Error: boom")


if __name__ == "__main__":
    unittest.main()
