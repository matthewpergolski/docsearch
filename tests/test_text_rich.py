import unittest
from types import SimpleNamespace

from docsearch.search import Retriever, SearchConfig


def retriever(chars):
    r = Retriever.__new__(Retriever)
    r.sc = SearchConfig(bm25_k=100, vector_k=100, rrf_k=60, rerank_candidates=30, top_k=10)
    long_text = "File: a.pdf\n" + "word " * 100
    r._by_unit = {
        "a#p1": [{"format": "pdf", "text": long_text}],  # a typed page
        "a#p2": [{"format": "pdf", "text": "File: a.pdf\n"}],  # a scanned page: no text of its own
        "s#p1": [{"format": "slide_image", "text": long_text}],  # not a format the rule covers
    }
    r._text_rich_chars, r._text_rich_formats = chars, {"pdf", "docx", "pptx"}
    r.visual = SimpleNamespace(meta={}, docs={"a", "s", "x"})
    return r


class TestTextRich(unittest.TestCase):
    def test_only_typed_pages_of_the_listed_formats_are_text_rich(self):
        self.assertEqual(retriever(200).text_rich, {"a#p1"})

    def test_off_by_default(self):
        self.assertEqual(retriever(0).text_rich, set())

    def fuse(self, chars):
        # text: x first, the typed page second; pictures: the typed page first, the scanned page second
        return [u for u, _ in retriever(chars)._fuse_format_aware(["x#p1", "a#p1"], ["a#p1", "a#p2", "x#p1"])]

    def test_a_typed_page_is_not_lifted_by_its_picture(self):
        self.assertEqual(self.fuse(0)[0], "a#p1")
        self.assertEqual(self.fuse(200)[0], "x#p1")

    def test_a_scanned_page_is_still_found_by_its_picture(self):
        self.assertIn("a#p2", self.fuse(200))


if __name__ == "__main__":
    unittest.main()
