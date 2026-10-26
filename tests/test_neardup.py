import unittest

from docsearch.config import DEFAULTS
from docsearch.search import SearchConfig, context_text, distinct_tiebreak, query_terms, rerank, rerank_pipeline


def page(i, text, fmt="pdf", doc="vidore/annex.pdf"):
    return {"chunk_id": f"c{i}", "doc_id": doc, "format": fmt, "pages": [i], "text": text}


TWINS = [page(1, "Young (15-24) 2022 = 30.1 2023 = 31.0 unemployment annex"),
         page(2, "Young (15-24) 2022 = 28.0 2023 = 29.4 unemployment annex"),
         page(3, "Young (15-24) 2022 = 32.7 2023 = 33.4 unemployment annex"),  # target
         page(4, "Young (15-24) 2022 = 25.5 2023 = 26.0 unemployment annex")]  # fmt: skip
Q = "What 2023 rate does the annex give for 15-24 year olds whose 2022 rate was 32.7?"


class Order:
    calls = 0

    def score(self, q, texts):
        self.texts = texts
        return [-i for i in range(len(texts))]  # keeps the given order


class TestDistinct(unittest.TestCase):
    def test_terms_keep_decimals_drop_stopwords(self):
        t = query_terms(Q)
        self.assertIn("32.7", t)
        self.assertIn("annex", t)
        self.assertNotIn("what", t)

    def test_rare_cue_moves_target_up(self):
        out = distinct_tiebreak(Q, TWINS, n=4, boost=3)
        self.assertEqual(out[0]["chunk_id"], "c3")
        self.assertEqual(out[0]["distinct_hits"], 1)

    def test_shared_terms_do_not_reorder_and_off_is_noop(self):
        self.assertEqual([c["chunk_id"] for c in distinct_tiebreak("unemployment annex", TWINS, 4, 3)],
                         ["c1", "c2", "c3", "c4"])  # fmt: skip
        self.assertIs(distinct_tiebreak(Q, TWINS, 4, 0), TWINS)

    def test_boost_is_bounded(self):
        out = distinct_tiebreak(Q, TWINS, n=4, boost=1.5)  # moves past one twin, not to the top
        self.assertEqual([c["chunk_id"] for c in out][:3], ["c1", "c3", "c2"])


class TestContext(unittest.TestCase):
    def test_context_text_has_doc_and_page(self):
        c = page(20, "Operating EPS bridge", "slide_image", "finslides/ibm-4q13-earnings-charts_page_20")
        self.assertTrue(context_text(c).startswith("Document: ibm 4q13 earnings charts page 20 · page 20\n"))

    def test_rerank_context_feeds_reranker_and_reranks_slides(self):
        cands = [page(1, "a", "slide_image"), page(2, "b")]
        rr = Order()
        sc = SearchConfig(10, 10, 60, 30, 10, rerank_skip_formats=("slide_image",), rerank_context=True)
        rerank_pipeline("q", cands, rr, sc)
        self.assertEqual(len(rr.texts), 2)  # slide not skipped
        self.assertTrue(all(t.startswith("Document: annex") for t in rr.texts))
        rr2 = Order()
        rerank("q", cands, rr2, 30, skip_formats=("slide_image",))
        self.assertEqual(rr2.texts, ["b"])  # default M3 behaviour unchanged

    def test_defaults_off(self):
        self.assertEqual(DEFAULTS["search"]["distinct_boost"], 0.0)
        self.assertFalse(DEFAULTS["search"]["rerank_context"])

    def test_cli_flags_parse(self):
        from docsearch.cli import main

        with self.assertRaises(SystemExit) as e:
            main(["eval", "--help"])
        self.assertEqual(e.exception.code, 0)


class TestPdfChallengers(unittest.TestCase):
    def test_query_numbers_skip_years_and_single_digits(self):
        from docsearch.search import query_numbers

        self.assertEqual(query_numbers("In 2023, 15 to 24 year olds at 32.7 or 7%?"), {"15", "24", "32.7"})

    def test_number_boost_only_for_listed_formats(self):
        from docsearch.search import number_boost

        out = number_boost(Q, TWINS, 4, 3)
        self.assertEqual(out[0]["chunk_id"], "c3")
        slides = [{**c, "format": "slide_image"} for c in TWINS]
        self.assertEqual([c["chunk_id"] for c in number_boost(Q, slides, 4, 3)], ["c1", "c2", "c3", "c4"])
        self.assertIs(number_boost(Q, TWINS, 4, 0), TWINS)

    def test_late_visual_lifts_visual_top_pdf_only(self):
        from docsearch.search import late_visual

        c = [{**TWINS[0], "vis_rank": 9}, TWINS[1], {**TWINS[2], "vis_rank": 1},
             {**TWINS[3], "format": "docx", "vis_rank": 1}]  # fmt: skip
        out = [x["chunk_id"] for x in late_visual(c, 4, ("pdf",), k=10)]
        self.assertEqual(out[0], "c3")  # visual #1 pdf page rises over reranker #1
        self.assertEqual(out[-1], "c4")  # docx not fused
        self.assertIs(late_visual(c, 4, ()), c)

    def test_vis_rank_flows_through_pipeline_defaults_off(self):
        rr = Order()
        sc = SearchConfig(10, 10, 60, 30, 10)
        c = [{**TWINS[0]}, {**TWINS[2], "vis_rank": 1}]
        self.assertEqual([x["chunk_id"] for x in rerank_pipeline(Q, c, rr, sc)], ["c1", "c3"])
        sc2 = SearchConfig(10, 10, 60, 30, 10, late_visual_formats=("pdf",))
        self.assertEqual(rerank_pipeline(Q, c, rr, sc2)[0]["chunk_id"], "c3")


def _mk(cid, fmt, page, text="x", vis=None):
    d = {"chunk_id": cid, "doc_id": cid.split("_")[0], "format": fmt, "pages": [page], "text": text}
    return {**d, "vis_rank": vis} if vis else d


class TestPdfChallengersB(unittest.TestCase):
    def test_late_visual_keeps_non_pdf_slots(self):
        from docsearch.search import late_visual

        c = [_mk("a_1", "docx", 1, vis=1), _mk("p_1", "pdf", 1), _mk("x_1", "xlsx", 1),
             _mk("p_2", "pdf", 2, vis=1), _mk("i_1", "image", 1, vis=2)]  # fmt: skip
        out = late_visual(c, 5, ("pdf",), k=10)
        self.assertEqual([x["chunk_id"] for x in out], ["a_1", "p_2", "x_1", "p_1", "i_1"])

    def test_number_boost_page_level_and_distinctive(self):
        from docsearch.search import number_boost

        # 2023 is on every page (not distinctive); 32.7 only on p_3; each page has several chunks
        c = [_mk("p_1", "pdf", 1, "2023 rates 10"), _mk("p_1b", "pdf", 1, "2023 more"),
             _mk("d_1", "docx", 1, "2023"), _mk("p_2", "pdf", 2, "2023 rate 11"),
             _mk("p_2b", "pdf", 2, "2023"), _mk("p_3", "pdf", 3, "2023 rate 32.7")]  # fmt: skip
        out = [x["chunk_id"] for x in number_boost("rate 32.7 in 2023", c, 6, 2)]
        self.assertEqual(out[2], "d_1")  # non-pdf slot unchanged
        self.assertEqual(out[0], "p_3")  # page 3 jumps two pages
        self.assertEqual(number_boost("rate in 2023", c, 6, 2), c)  # only non-distinctive number

    def test_number_boost_stats(self):
        from docsearch import search

        search.NUMBER_BOOST_STATS.clear()
        c = [_mk("p_1", "pdf", 1, "1"), _mk("p_3", "pdf", 3, "32.7")]
        search.number_boost("32.7", c, 2, 2)
        self.assertEqual(search.NUMBER_BOOST_STATS["touched"], 1)
