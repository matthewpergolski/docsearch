"""M15 table-aware PDF chunking: tables whole (split by rows, header repeated), everything else verbatim."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from docsearch.chunk import Chunk, merge_pdf_tables, pdf_table_chunks, table_pieces
from docsearch.manifest import Doc


def count(s: str) -> int:
    return len(s.split())


def make_doc(n_rows: int = 3):
    from docling_core.types.doc import (
        BoundingBox,
        DocItemLabel,
        DoclingDocument,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )

    doc = DoclingDocument(name="report")
    doc.add_page(page_no=4, size=Size(width=600, height=800))
    pv = ProvenanceItem(page_no=4, bbox=BoundingBox(l=0, t=0, r=10, b=10), charspan=(0, 1))
    doc.add_heading(text="Annex 2 Employment", prov=pv)
    doc.add_text(label=DocItemLabel.TEXT, text="Some body text about employment trends in Europe.", prov=pv)
    cells = []
    rows = [["Country", "2022", "2023"]] + [[f"C{i}", f"{i}.1", f"{i}.2"] for i in range(n_rows)]
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cells.append(TableCell(text=v, start_row_offset_idx=r, end_row_offset_idx=r + 1, start_col_offset_idx=c,
                                   end_col_offset_idx=c + 1, column_header=r == 0))  # fmt: skip
    cap = doc.add_text(label=DocItemLabel.CAPTION, text="Employment rate by country", prov=pv)
    t = doc.add_table(data=TableData(num_rows=len(rows), num_cols=3, table_cells=cells), prov=pv)
    t.captions.append(cap.get_ref())
    return doc


class FakeMeta:
    def __init__(self, items):
        self.doc_items, self.headings = items, []


class FakeChunk:
    def __init__(self, items):
        self.meta = FakeMeta(items)


class FakeChunker:
    """Mimics HybridChunker.chunk: chunk 0 = body text, chunk 1 = the table only."""

    def chunk(self, dl_doc):
        texts = [x for x in dl_doc.texts if x.label != "caption" and x.label != "section_header"]
        return [FakeChunk(texts), FakeChunk(list(dl_doc.tables))]


D = Doc(doc_id="pdfs/a", source="vidore", format="pdf", path="/x/a.pdf", license="x")


class TestPdfTables(unittest.TestCase):
    def test_pieces_repeat_header(self):
        p = table_pieces("T\n", "| h |\n| --- |\n", [f"| r{i} |" for i in range(6)], count, 8)
        self.assertGreater(len(p), 1)
        for x in p:
            self.assertTrue(x.startswith("T\n| h |\n| --- |\n"))
        self.assertEqual(sum(x.count("| r") for x in p), 6)

    def test_table_chunk_whole_with_prefix(self):
        drop, new = pdf_table_chunks(make_doc(), D, FakeChunker(), count, 400)
        self.assertEqual(drop, {"hybrid:pdfs/a:1"})
        self.assertEqual(len(new), 1)
        t = new[0].text
        self.assertTrue(t.startswith("File: a.pdf\nSection: Annex 2 Employment\nTable 1 (page 4): Employment rate"))
        self.assertIn("| Country | 2022 | 2023 |", t)
        self.assertIn("| C2 | 2.1 | 2.2 |", t)
        self.assertEqual(new[0].pages, [4])

    def test_long_table_split_by_rows(self):
        _, new = pdf_table_chunks(make_doc(40), D, FakeChunker(), count, 60)
        self.assertGreater(len(new), 2)
        for c in new:
            self.assertIn("| Country | 2022 | 2023 |", c.text)
            self.assertIn("Employment rate by country", c.text)

    def test_non_pdf_and_non_table_chunks_byte_identical(self):
        mk = lambda cid, fmt, txt: Chunk(cid, cid.split(":")[1], fmt, "s", [1], txt, 3, {"headings": ["h"]})  # noqa: E731
        base = [mk("hybrid:pdfs/a:0", "pdf", "body"), mk("hybrid:pdfs/a:1", "pdf", "| tbl |"),
                mk("hybrid:x.xlsx:0", "xlsx", "| a |"), mk("hybrid:d.docx:0", "docx", "doc"),
                mk("fallback:s.png:fb0", "slide_image", "ocr")]  # fmt: skip
        drop, new = pdf_table_chunks(make_doc(), D, FakeChunker(), count, 400)
        out = merge_pdf_tables(base, drop, new)
        dump = lambda cs: [json.dumps(c.__dict__, sort_keys=True) for c in cs]  # noqa: E731
        kept = [c for c in base if c.chunk_id != "hybrid:pdfs/a:1"]
        self.assertEqual(dump([c for c in out if not c.chunk_id.startswith("hybrid_pdftable:")]), dump(kept))
        self.assertTrue(all(c.format == "pdf" for c in out if c not in kept))

    def test_cli_flag_selects_table_and_base_reference(self):
        from unittest import mock

        from docsearch import cli

        seen = {}

        def fake_eval(cfg, plan, *a, **k):
            seen["plan"] = plan
            return []

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        y = Path(td.name) / "docsearch.yaml"
        y.write_text("models: {embedders: {pplx-v2-late: {id: org/vis, kind: visual}}}\n")
        with mock.patch("docsearch.eval.run_eval", fake_eval), mock.patch("docsearch.report.build_report"), mock.patch(
            "docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), mock.patch(
            "docsearch.eval.cached_reference", lambda cfg, full, variant, vid: seen.setdefault("ref", variant) and
            {"name": "m3", "run_id": "r", "ndcg@10": "0.7", "n_queries": "1"}), contextlib.suppress(SystemExit):  # fmt: skip
            cli.main(["--config", str(y), "eval", "--visual", "pplx-v2-late", "--fusion", "format_aware", "--pdf-tables",
                      "--name", "m15_pdf_tables"])  # fmt: skip
        self.assertEqual(seen.get("ref"), "hybrid")
        self.assertEqual(seen["plan"][0][:2], ("m15_pdf_tables", "hybrid_pdftable"))


if __name__ == "__main__":
    unittest.main()
