import tempfile
import unittest
from pathlib import Path

from docsearch.manifest import (
    Doc,
    Question,
    chunk_units,
    file_unit,
    page_unit,
    ranked_units,
    read_manifest,
    read_questions,
    unit_doc,
    validate,
    write_manifest,
    write_questions,
)


class TestUnits(unittest.TestCase):
    def test_page_and_file_units(self):
        self.assertEqual(page_unit("vidore_v3_hr/doc", 3), "vidore_v3_hr/doc#p3")
        self.assertEqual(file_unit("ocb/a.xlsx"), "ocb/a.xlsx")
        self.assertEqual(unit_doc("vidore_v3_hr/doc#p3"), "vidore_v3_hr/doc")
        self.assertEqual(unit_doc("ocb/a.xlsx"), "ocb/a.xlsx")

    def test_chunk_units(self):
        self.assertEqual(chunk_units("d", [4, 3, 4], "page"), ["d#p3", "d#p4"])
        self.assertEqual(chunk_units("d", [], "page"), ["d#p1"])  # single-page images
        self.assertEqual(chunk_units("d", [7], "file"), ["d"])

    def test_ranked_units_dedup_keeps_first_rank(self):
        hits = [
            {"doc_id": "a", "pages": [2]},
            {"doc_id": "b", "pages": [1]},
            {"doc_id": "a", "pages": [2, 3]},
        ]
        self.assertEqual(ranked_units(hits, "page"), ["a#p2", "b#p1", "a#p3"])
        self.assertEqual(ranked_units(hits, "file"), ["a", "b"])


class TestManifestIO(unittest.TestCase):
    def test_roundtrip_and_validate(self):
        docs = [
            Doc("vidore_v3_hr/r", "pdf/vidore_v3_hr/r.pdf", "pdf", "vidore_v3_hr", "CC-BY-4.0", pages=5),
            Doc("ocb/x.xlsx", "office/ocb/x.xlsx", "xlsx", "ocb", "CDLA-Permissive-2.0"),
        ]
        qs = [
            Question("q1", "what?", "vidore_v3_hr", "pdf", "page", {"vidore_v3_hr/r#p2": 2}),
            Question("q2", "which?", "ocb", "xlsx", "file", {"ocb/x.xlsx": 1}),
        ]
        with tempfile.TemporaryDirectory() as td:
            mp, qp = Path(td) / "m.csv", Path(td) / "q.jsonl"
            write_manifest(mp, docs)
            write_questions(qp, qs)
            docs2, qs2 = read_manifest(mp), read_questions(qp)
        self.assertEqual({d.doc_id for d in docs2}, {d.doc_id for d in docs})
        self.assertEqual(next(d for d in docs2 if d.format == "pdf").pages, 5)
        self.assertEqual(qs2[0].gold, {"vidore_v3_hr/r#p2": 2})
        self.assertEqual(validate(docs2, qs2), [])

    def test_validate_catches_bad_gold(self):
        docs = [Doc("a", "a.pdf", "pdf", "s", "l", pages=2)]
        qs = [
            Question("q1", "?", "s", "pdf", "page", {"a#p9": 1}),
            Question("q2", "?", "s", "pdf", "page", {"zzz#p1": 1}),
            Question("q3", "?", "s", "pdf", "page", {}),
        ]
        probs = validate([Doc("a", "a2.pdf", "pdf", "s", "l")] + docs, qs)
        self.assertTrue(any("beyond page count" in p for p in probs))
        self.assertTrue(any("not in manifest" in p for p in probs))
        self.assertTrue(any("no gold" in p for p in probs))
        self.assertTrue(any("duplicate" in p for p in probs))


if __name__ == "__main__":
    unittest.main()
