import unittest

from docsearch.chunk import pages_for_span, recursive_spans
from docsearch.search import rerank, rrf


def words(s: str) -> int:
    return len(s.split())


class TestRecursive(unittest.TestCase):
    def test_spans_are_contiguous_and_capped(self):
        text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(30)) for p in range(5))
        spans = recursive_spans(text, words, 50)
        self.assertTrue(all(words(text[s:e]) <= 50 for s, e in spans))
        self.assertEqual("".join(text[s:e] for s, e in spans).split(), text.split())

    def test_long_paragraph_splits_on_words(self):
        text = " ".join(f"t{i}" for i in range(120))
        spans = recursive_spans(text, words, 40)
        self.assertGreaterEqual(len(spans), 3)
        self.assertTrue(all(words(text[s:e]) <= 40 for s, e in spans))

    def test_pages_for_span(self):
        bounds = [(0, 10, 1), (10, 20, 2), (20, 30, 3)]
        self.assertEqual(pages_for_span(bounds, 5, 15), [1, 2])
        self.assertEqual(pages_for_span(bounds, 20, 25), [3])
        self.assertEqual(pages_for_span([(0, 10, 0)], 0, 5), [])


class TestFusion(unittest.TestCase):
    def test_rrf(self):
        fused = [c for c, _ in rrf([["a", "b", "c"], ["b", "d"]], k=60)]
        self.assertEqual(fused[0], "b")
        self.assertEqual(set(fused), {"a", "b", "c", "d"})

    def test_rerank_reorders_head_only(self):
        class Fake:
            def score(self, q, passages):
                return [len(p) for p in passages]

        cands = [{"text": "a"}, {"text": "ccc"}, {"text": "bb"}, {"text": "zzzzzz"}]
        out = rerank("q", cands, Fake(), 3)
        self.assertEqual([c["text"] for c in out], ["ccc", "bb", "a", "zzzzzz"])


if __name__ == "__main__":
    unittest.main()
