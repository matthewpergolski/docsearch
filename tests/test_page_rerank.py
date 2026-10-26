import tempfile
import unittest
from pathlib import Path

from docsearch.config import load_config
from docsearch.search import SearchConfig, page_rerank


class FakeScorer:
    """Scores a page unit from a table; units not in the table have no image (None)."""

    def __init__(self, scores):
        self.scores, self.seen = scores, []

    def score(self, query, units):
        self.seen += units
        return [self.scores.get(u) for u in units]


def c(i, fmt, doc, page=1):
    return {"chunk_id": str(i), "format": fmt, "doc_id": doc, "pages": [page], "text": "t"}


def ids(out):
    return [x["chunk_id"] for x in out]


class TestPageRerank(unittest.TestCase):
    def test_pages_swap_within_their_slots_and_other_formats_stay(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "docx", "w"), c(2, "pdf", "a", 2), c(3, "xlsx", "x"), c(4, "pdf", "a", 3)]
        sc = FakeScorer({"a#p1": 0.1, "a#p2": 0.9, "a#p3": 0.5})
        out = page_rerank("q", ranked, sc, n=5, pages=10, formats=("pdf",))
        self.assertEqual(ids(out), ["2", "1", "4", "3", "0"])  # docx and xlsx keep slots 1 and 3
        self.assertEqual(out[0]["page_score"], 0.9)

    def test_only_the_top_pages_are_scored(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "pdf", "a", 2), c(2, "pdf", "a", 3)]
        sc = FakeScorer({"a#p1": 0.1, "a#p2": 0.9, "a#p3": 5.0})
        out = page_rerank("q", ranked, sc, n=3, pages=2, formats=("pdf",))
        self.assertEqual(ids(out), ["1", "0", "2"])  # page 3 has the best score but is outside the top 2 pages
        self.assertEqual(sc.seen, ["a#p1", "a#p2"])

    def test_a_page_keeps_its_chunks_together_and_is_scored_once(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "pdf", "a", 2), c(2, "pdf", "a", 1)]
        sc = FakeScorer({"a#p1": 0.1, "a#p2": 0.9})
        out = page_rerank("q", ranked, sc, n=3, pages=10, formats=("pdf",))
        self.assertEqual(ids(out), ["1", "0", "2"])
        self.assertEqual(sc.seen, ["a#p1", "a#p2"])

    def test_a_page_with_no_image_keeps_its_place(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "pdf", "a", 2), c(2, "pdf", "a", 3)]
        sc = FakeScorer({"a#p1": 0.1, "a#p3": 0.9})
        out = page_rerank("q", ranked, sc, n=3, pages=10, formats=("pdf",))
        self.assertEqual(ids(out), ["2", "1", "0"])

    def test_candidates_below_n_are_untouched(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "pdf", "a", 2), c(2, "pdf", "a", 3)]
        sc = FakeScorer({"a#p1": 0.1, "a#p2": 0.2, "a#p3": 0.9})
        out = page_rerank("q", ranked, sc, n=2, pages=10, formats=("pdf",))
        self.assertEqual(ids(out), ["1", "0", "2"])

    def test_off_without_a_scorer(self):
        ranked = [c(0, "pdf", "a", 1), c(1, "pdf", "a", 2)]
        self.assertIs(page_rerank("q", ranked, None, n=2, pages=10, formats=("pdf",)), ranked)

    def test_config_default_is_off(self):
        with tempfile.TemporaryDirectory() as t:
            sc = SearchConfig.from_cfg(load_config(Path(t) / "none.yaml", env={}))
            self.assertEqual((sc.page_reranker, sc.page_rerank_pages, sc.page_rerank_formats),
                             ("", 10, ("pdf", "slide_image")))  # fmt: skip


if __name__ == "__main__":
    unittest.main()


class TestProgress(unittest.TestCase):
    """The search page shows the text-ranked list before the slow page rerank; progress sees it first."""

    def test_progress_gets_the_text_ranked_list_before_the_page_rerank(self):
        from docsearch.search import rerank_pipeline

        class Reranker:
            def score(self, query, texts):
                return [float(len(texts) - i) for i in range(len(texts))]

        sc = SearchConfig.from_cfg(load_config(env={}))
        sc = SearchConfig(**{**sc.__dict__, "page_rerank_formats": ("slide_image",), "rerank_skip_formats": ()})
        ranked = [c(0, "pdf", "a"), c(1, "slide_image", "s1"), c(2, "slide_image", "s2")]
        events = []
        scorer = FakeScorer({"s1#p1": 0.1, "s2#p1": 0.9})
        scorer.score = lambda q, units, f=scorer.score: events.append("page rerank") or f(q, units)
        out = rerank_pipeline(
            "q", ranked, Reranker(), sc, scorer, progress=lambda kind, hits: events.append((kind, ids(hits)))
        )
        self.assertEqual(events, [("ranked", ["0", "1", "2"]), "page rerank"])
        self.assertEqual(ids(out), ["0", "2", "1"])


class TestEarlySafe(unittest.TestCase):
    """The search page shows the text-ranked list before the page rerank only when no later step can move it."""

    def test_early_results_only_when_every_step_after_the_page_rerank_is_off(self):
        from docsearch.search import early_safe

        sc = SearchConfig.from_cfg(load_config(env={}))
        self.assertTrue(early_safe(sc))  # the locked default
        for change in ({"distinct_boost": 0.5}, {"number_boost": 2.0}, {"late_visual_formats": ("pdf",)},
                       {"table_lookup": True}):  # fmt: skip
            self.assertFalse(early_safe(SearchConfig(**{**sc.__dict__, **change})), change)


class TestStopPoints(unittest.TestCase):
    """A newer search (docsearch serve) stops a text search after retrieval and after the text rerank, not only
    between page-reranker pages."""

    def test_rerank_pipeline_stops_before_the_page_rerank_when_a_newer_search_arrived(self):
        from docsearch import models
        from docsearch.search import rerank_pipeline

        class Reranker:
            def score(self, query, texts):
                models.stop_when(lambda: True)  # a newer search arrives during the text rerank
                return [1.0] * len(texts)

        self.addCleanup(models.stop_when, None)
        sc = SearchConfig.from_cfg(load_config(env={}))
        scorer = FakeScorer({})
        with self.assertRaises(models.Superseded):
            long = [{**c(0, "pdf", "a"), "text": "word " * 40}, {**c(1, "slide_image", "s"), "text": "word " * 40}]
            rerank_pipeline("q", long, Reranker(), sc, scorer)
        self.assertEqual(scorer.seen, [])
