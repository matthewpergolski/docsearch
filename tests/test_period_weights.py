import unittest
from types import SimpleNamespace

from docsearch.search import Retriever, SearchConfig, period_demote, periods


class TestPeriods(unittest.TestCase):
    def test_file_name_quarter(self):
        self.assertEqual(periods("ibm-3q13-earnings-charts_page_18"), ({(3, 2013)}, {2013}))

    def test_query_forms(self):
        self.assertEqual(periods("revenue in the third quarter of 2013?")[0], {(3, 2013)})
        self.assertEqual(periods("What changed in Q4 2021 and 1Q20?")[0], {(4, 2021), (1, 2020)})
        self.assertEqual(periods("change from 2020 to 2021"), (set(), {2020, 2021}))

    def test_page_numbers_and_ids_are_not_periods(self):
        self.assertEqual(periods("report-KE0125029ENN_page_18"), (set(), set()))


class TestPeriodDemote(unittest.TestCase):
    fused = [("s/ibm-4q13-earnings_page_2#p1", 0.5), ("s/ibm-3q13-earnings_page_9#p1", 0.4),
             ("reports/annual_2019.pdf#p3", 0.3), ("s/ibm-3q14-earnings_page_9#p1", 0.2)]  # fmt: skip

    def units(self, query):
        return [u.split("/")[1][:8] for u, _ in period_demote(query, self.fused)]

    def test_quarter_query_moves_other_quarters_back(self):
        self.assertEqual(
            self.units("growth in the third quarter of 2013"), ["ibm-3q13", "annual_2", "ibm-4q13", "ibm-3q14"]
        )

    def test_year_query_moves_quarters_of_other_years_back(self):
        self.assertEqual(self.units("growth during 2014"), ["annual_2", "ibm-3q14", "ibm-4q13", "ibm-3q13"])

    def test_query_without_a_period_changes_nothing(self):
        self.assertEqual(period_demote("what is the growth?", self.fused), self.fused)


class TestVisualWeights(unittest.TestCase):
    def fuse(self, weights):
        r = Retriever.__new__(Retriever)
        r.sc = SearchConfig(bm25_k=100, vector_k=100, rrf_k=60, rerank_candidates=30, top_k=10, visual_weights=weights)
        meta = {"a#p1": {"doc_id": "a", "format": "pdf"}, "b#p1": {"doc_id": "b", "format": "slide_image"}}
        r.visual = SimpleNamespace(meta=meta, docs={"a", "b"})
        # text likes the pdf page first; the visual model likes the slide first
        return [u for u, _ in r._fuse_format_aware(["a#p1", "x#p1", "y#p1", "b#p1"], ["b#p1", "z#p1", "w#p1", "a#p1"])]

    def test_default_weight_keeps_the_tie_in_text_order(self):
        self.assertEqual(self.fuse(())[0], "a#p1")

    def test_heavier_visual_rank_lifts_the_slide(self):
        self.assertEqual(self.fuse((("slide_image", 2.0),))[0], "b#p1")


class TestOfficeOwnSlots(unittest.TestCase):
    def test_office_files_reorder_among_their_own_slots_only(self):
        from docsearch.search import office_own_slots

        base = [("sheet.xlsx", 0.9), ("a.docx", 0.8), ("p#p3", 0.7), ("b.pptx", 0.6)]
        # with every page rendered, b.pptx and a new deck c.pptx beat everything in the full fusion
        full = [("c.pptx", 1.0), ("b.pptx", 0.95), ("sheet.xlsx", 0.9), ("a.docx", 0.8), ("p#p3", 0.7)]
        out = [u for u, _ in office_own_slots(base, full, {"a.docx", "b.pptx", "c.pptx"})]
        self.assertEqual(out, ["sheet.xlsx", "c.pptx", "p#p3", "b.pptx", "a.docx"])


if __name__ == "__main__":
    unittest.main()
