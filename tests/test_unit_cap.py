import unittest

from docsearch.search import Retriever, SearchConfig


def row(i, fmt, doc, page=None):
    return {
        "chunk_id": str(i),
        "doc_id": doc,
        "format": fmt,
        "source": "s",
        "pages": [page] if page else [],
        "text": "t",
    }


def retriever(rows, cap):
    """A Retriever whose text search returns `rows` in order and whose visual model saw nothing."""
    r = Retriever.__new__(Retriever)
    r.sc = SearchConfig(bm25_k=100, vector_k=100, rrf_k=60, rerank_candidates=30, top_k=10, rerank_unit_chunks=cap)
    r.bm25 = lambda query, k: rows
    r._visual_hits = lambda vq: []
    return r


class TestRerankUnitChunks(unittest.TestCase):
    # one spreadsheet with four retrieved chunks ahead of a pdf page and a docx file
    rows = [row(0, "xlsx", "big"), row(1, "xlsx", "big"), row(2, "xlsx", "big"), row(3, "xlsx", "big"),
            row(4, "pdf", "p", 7), row(5, "docx", "w")]  # fmt: skip

    def ids(self, cap):
        return [c["chunk_id"] for c in retriever(self.rows, cap).candidates("q", None, "hybrid_visual")]

    def test_off_keeps_every_chunk_of_a_file_in_a_row(self):
        self.assertEqual(self.ids(0), ["0", "1", "2", "3", "4", "5"])

    def test_cap_moves_a_files_extra_chunks_behind_the_other_units(self):
        self.assertEqual(self.ids(2), ["0", "1", "4", "5", "2", "3"])

    def test_no_chunk_is_lost(self):
        self.assertEqual(sorted(self.ids(1)), ["0", "1", "2", "3", "4", "5"])


if __name__ == "__main__":
    unittest.main()
