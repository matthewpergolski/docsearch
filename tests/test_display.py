import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.config import load_config
from docsearch.meta import DocMeta, update_meta

QUERY = "total revenue in the third quarter of 2018"
LONG = ("Section 4 explains the budget process in detail and lists every step the board takes before the annual "
        "vote. The total revenue in the third quarter of 2018 rose to 4.2 million after the new contracts, which "
        "the board noted with approval. Costs stayed flat across all units and the reserve was kept.")  # fmt: skip


def hit(doc: str, page, text: str, fmt: str, n: int = 0, **scores) -> dict:
    return {"chunk_id": f"h:{doc}:{n}", "doc_id": doc, "format": fmt, "pages": [page] if page else None,
            "text": f"File: {doc.rsplit('/', 1)[-1]}\n{text}", **scores}  # fmt: skip


HITS = [
    hit("fin/acme-3q18_page_3", 1, "Overview\n3Q18 $18.8B Revenue, up 2% in the third quarter", "slide_image",
        score=0.0267, page_score=0.43),
    hit("ocb/sales.xlsx", 2, "Table 1\nYear | Year | Revenue\n2018 | 2018 | 197056", "xlsx", score=0.0294,
        rerank_score=0.829),
    hit("fin/acme-3q18_page_17", 1, "Segment Revenue\nRevenue growth rates", "slide_image", score=0.0268,
        page_score=0.24),
    hit("ocb/plan.docx", None, "Budget\n" + LONG.replace("the annual vote.", "the annual\nvote."), "docx", score=0.02,
        rerank_score=0.7),
    hit("fin/acme-3q18_page_26", 1, "Reconciliation\nRevenue growth 3Q 2018", "slide_image", score=0.0317,
        page_score=0.16),
    hit("ocb/sales.xlsx", 2, "Table 1\n2019 | 432290", "xlsx", 1, score=0.0291, rerank_score=0.81),
    hit("fin/acme-3q18_page_8", 1, "Cloud\nCloud revenue", "slide_image", score=0.025, page_score=0.1),
]  # fmt: skip

TEXT = """\
 1. acme-3q18 | Q3 2018
    fin/acme-3q18
    p. 3   Overview · 3Q18 $18.8B Revenue, up 2% in the third quarter
    p. 17  Segment Revenue · Revenue growth rates
    p. 26  Reconciliation · Revenue growth 3Q 2018
    also p. 8
 2. sales.xlsx
    ocb/sales.xlsx
    p. 2   Table 1 · Year | Revenue / 2018 | 197056
 3. Revenue plan | 2024-02-15
    ocb/plan.docx
    …before the annual vote. The total revenue in the third quarter of 2018 rose to 4.2 million after the new \
contracts, which the board noted with…"""

TEXT_SCORES = """\
 1. acme-3q18 | Q3 2018
    fin/acme-3q18
    p. 3   page 0.430  Overview · 3Q18 $18.8B Revenue, up 2% in the third quarter
    p. 17  page 0.240  Segment Revenue · Revenue growth rates
    p. 26  page 0.160  Reconciliation · Revenue growth 3Q 2018
    also p. 8
 2. sales.xlsx
    ocb/sales.xlsx
    p. 2   rerank 0.829  Table 1 · Year | Revenue / 2018 | 197056
 3. Revenue plan | 2024-02-15
    ocb/plan.docx
    rerank 0.700  …before the annual vote. The total revenue in the third quarter of 2018 rose to 4.2 million \
after the new contracts, which the board noted with…"""

SNIPPET = ("…before the annual vote. The total revenue in the third quarter of 2018 rose to 4.2 million after the new "
           "contracts, which the board noted with…")  # fmt: skip
JSON = {
    "query": QUERY,
    "confident": None,
    "answer": None,
    "groups": [
        {"rank": 1, "file": "fin/acme-3q18", "format": "slide_image", "name": "acme-3q18", "date": "",
         "period": "Q3 2018", "pages": [
            {"page": 3, "rank": 1, "title": "Overview", "snippet": "3Q18 $18.8B Revenue, up 2% in the third quarter",
             "score": 0.43, "scorer": "page"},
            {"page": 17, "rank": 3, "title": "Segment Revenue", "snippet": "Revenue growth rates", "score": 0.24,
             "scorer": "page"},
            {"page": 26, "rank": 5, "title": "Reconciliation", "snippet": "Revenue growth 3Q 2018", "score": 0.16,
             "scorer": "page"},
            {"page": 8, "rank": 7, "title": "Cloud", "snippet": "Cloud revenue", "score": 0.1, "scorer": "page"}]},
        {"rank": 2, "file": "ocb/sales.xlsx", "format": "xlsx", "name": "sales.xlsx", "date": "", "period": "",
         "pages": [{"page": 2, "rank": 2, "title": "Table 1", "snippet": "Year | Revenue / 2018 | 197056",
                    "score": 0.829, "scorer": "rerank"}]},
        {"rank": 3, "file": "ocb/plan.docx", "format": "docx", "name": "Revenue plan", "date": "2024-02-15",
         "period": "", "pages": [{"page": None, "rank": 4, "title": "", "snippet": SNIPPET, "score": 0.7,
                                  "scorer": "rerank"}]},
    ],
    "chunk_ids": ["h:fin/acme-3q18_page_3:0", "h:ocb/sales.xlsx:0", "h:fin/acme-3q18_page_17:0", "h:ocb/plan.docx:0",
                  "h:fin/acme-3q18_page_26:0", "h:ocb/sales.xlsx:1", "h:fin/acme-3q18_page_8:0"],
}  # fmt: skip

LOOKALIKE = [
    hit("s/a.pdf", 3, "Croatia 12.1 2022", "pdf", score=0.03, rerank_score=0.95),
    hit("s/b.pdf", 1, "other report", "pdf", score=0.03, rerank_score=0.9),
    hit("s/a.pdf", 4, "Croatia 14.9 2023", "pdf", score=0.02, rerank_score=0.88),
]  # fmt: skip
LOW = "No confident match; the nearest results follow."


class FakeClient:
    s = None  # the LLM settings check_available reads (patched out here)

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


class SearchCli(unittest.TestCase):
    yaml = "subset: {enabled: false}\nvisual: {default: null}\nsearch: {abstain_threshold: 0.868}\n"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.config = root / "docsearch.yaml"
        self.config.write_text(self.yaml)
        cfg = load_config(self.config, env={})
        update_meta(cfg, [DocMeta("ocb/plan.docx", "plan.docx", "docx", "Revenue plan", modified="2024-02-15",
                                  file_name="plan.docx")])  # fmt: skip

    def tearDown(self):
        self.td.cleanup()

    def run_search(self, hits, *args, client=None):
        """stdout and stderr of `docsearch search QUERY ARGS` when search returns `hits`; the hits must come back
        unchanged."""
        given = copy.deepcopy(hits)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch("docsearch.search.search", return_value=given))
            st.enter_context(mock.patch("docsearch.search.page_rows", return_value=copy.deepcopy(hits)))
            st.enter_context(mock.patch("docsearch.llm.get_client", return_value=client))
            st.enter_context(mock.patch("docsearch.llm.check_available"))
            st.enter_context(contextlib.redirect_stdout(out))
            st.enter_context(contextlib.redirect_stderr(err))
            from docsearch.cli import main

            self.assertEqual(main(["--config", str(self.config), "search", QUERY, *args]), 0)
        self.assertEqual(given, hits)
        return out.getvalue(), err.getvalue()

    def results(self, stdout: str) -> str:
        """The lines after the status lines (device, model cache) and before the warnings line."""
        lines = stdout.rstrip("\n").split("\n")
        start = next(i for i, x in enumerate(lines) if x.startswith((" 1.", "Answer:", LOW)))
        end = next(i for i in range(len(lines) - 1, -1, -1) if "logged" in lines[i])
        return "\n".join(lines[start:end])


class TestGroupedText(SearchCli):
    def test_files_grouped_deck_as_one_file_snippets_dates_and_ranks(self):
        out, _ = self.run_search(HITS)
        self.assertEqual(self.results(out), TEXT)

    def test_scores_and_their_scorer_only_with_scores(self):
        out, _ = self.run_search(HITS, "--scores")
        self.assertEqual(self.results(out), TEXT_SCORES)

    def test_no_hits_says_so(self):
        out, _ = self.run_search([])
        self.assertIn("\nno hits\n", out)


class TestJson(SearchCli):
    def test_stdout_is_only_the_grouped_json_and_the_hit_list_is_searchs(self):
        out, err = self.run_search(HITS, "--json")
        self.assertEqual(json.loads(out), JSON)
        self.assertIn("device: ", err)

    def test_the_hit_list_is_the_same_with_text_and_json_display(self):
        text, _ = self.run_search(HITS)
        js, _ = self.run_search(HITS, "--json")
        self.assertEqual(json.loads(js)["chunk_ids"], [h["chunk_id"] for h in HITS])
        self.assertEqual(self.results(text), TEXT)


class TestConfidence(SearchCli):
    def test_text_reranker_top_below_the_threshold_says_no_confident_match_and_lists_the_rest(self):
        hits = [HITS[1], HITS[0], HITS[3]]
        out, _ = self.run_search(hits)
        self.assertEqual(self.results(out).split("\n")[:4], [LOW, "", " 1. sales.xlsx", "    ocb/sales.xlsx"])
        js, _ = self.run_search(hits, "--json")
        self.assertIs(json.loads(js)["confident"], False)

    def test_text_reranker_top_at_or_above_the_threshold_shows_no_warning(self):
        out, _ = self.run_search(LOOKALIKE[:2])
        self.assertEqual(self.results(out).split("\n")[0], " 1. a.pdf")
        js, _ = self.run_search(LOOKALIKE[:2], "--json")
        self.assertIs(json.loads(js)["confident"], True)

    def test_a_top_from_another_scorer_gets_no_verdict_even_with_a_tiny_score(self):
        unranked = [hit("s/u.pdf", 1, "words", "pdf", score=0.01), *LOOKALIKE]
        for hits in (HITS, unranked):
            out, _ = self.run_search(hits)
            self.assertNotIn(LOW, out)
            js, _ = self.run_search(hits, "--json")
            self.assertIsNone(json.loads(js)["confident"])


class TestSyncedFromOutsideCorpus(SearchCli):
    def test_search_output_names_no_absolute_path(self):
        from docsearch.manifest import Doc
        from docsearch.meta import extract

        outside = Path(self.td.name) / "home" / "files" / "sub"
        outside.mkdir(parents=True)
        (outside / "list.csv").write_text("item,price\nrye,4\n")
        cfg = load_config(self.config, env={})
        m = extract(cfg.path("corpus_dir"), Doc("files/sub/list.csv", str(outside / "list.csv"), "csv", "files", ""))
        update_meta(cfg, [m])
        hits = [hit("files/sub/list.csv", None, "item | price\nrye | 4", "csv", score=0.03, rerank_score=0.95)]
        for args in ((), ("--json",), ("--folder", "files/sub")):
            out, _ = self.run_search(hits, *args)
            shown = out if "--json" in args else self.results(out)  # the log-file line names the project's logs/
            self.assertIn("files/sub/list.csv", shown)
            for root in (self.td.name, str(Path(self.td.name).resolve()), str(Path.home())):
                self.assertNotIn(root, shown)


class TestNoThreshold(SearchCli):
    yaml = "subset: {enabled: false}\nvisual: {default: null}\n"

    def test_without_a_threshold_there_is_no_verdict(self):
        out, _ = self.run_search([HITS[1]])
        self.assertNotIn(LOW, out)
        self.assertEqual(self.results(out).split("\n")[0], " 1. sales.xlsx")


class TestAgent(SearchCli):
    def test_answer_and_its_page_first_then_the_plain_list(self):
        client = FakeClient('{"candidate": "C", "value": "14.9%"}')
        out, _ = self.run_search(LOOKALIKE, "--agent", client=client)
        self.assertEqual(self.results(out).split("\n")[:5], [
            "Answer: 14.9% (a.pdf, p. 4)", "", " 1. a.pdf", "    s/a.pdf", "    p. 3   Croatia 12.1 2022"])  # fmt: skip
        self.assertIn("[C] a.pdf, page 4\nCroatia 14.9 2023", client.prompts[0])

    def test_json_carries_the_answer_and_keeps_searchs_order(self):
        client = FakeClient('{"candidate": "C", "value": "14.9%"}')
        out, _ = self.run_search(LOOKALIKE, "--agent", "--json", client=client)
        got = json.loads(out)
        self.assertEqual(got["answer"], {"value": "14.9%", "file": "s/a.pdf", "page": 4})
        self.assertEqual(got["chunk_ids"], ["h:s/a.pdf:0", "h:s/b.pdf:0", "h:s/a.pdf:0"])

    def test_no_page_prints_nothing_extra(self):
        out, _ = self.run_search(LOOKALIKE, "--agent", client=FakeClient('{"candidate": null, "value": null}'))
        self.assertEqual(self.results(out).split("\n")[:2], [" 1. a.pdf", "    s/a.pdf"])

    def test_gate_closed_makes_no_call_and_prints_nothing_extra(self):
        client = FakeClient()
        out, _ = self.run_search(HITS, "--agent", client=client)
        self.assertEqual((client.prompts, self.results(out)), ([], TEXT))


class TestAgentUnavailable(SearchCli):
    def test_a_usage_limit_prints_the_plain_results_and_a_one_line_note(self):
        from docsearch.llm import LLMUnavailable

        class Limited(FakeClient):
            def complete(self, prompt: str) -> str:
                raise LLMUnavailable("`claude` hit a usage limit: resets 9pm")

        out, err = self.run_search(LOOKALIKE, "--agent", client=Limited())
        self.assertEqual(self.results(out).split("\n")[:2], [" 1. a.pdf", "    s/a.pdf"])
        self.assertIn("search --agent: the agent is not available (`claude` hit a usage limit: resets 9pm); "
                      "plain results only\n", err)  # fmt: skip


class TestAgentModeOn(SearchCli):
    yaml = "subset: {enabled: false}\nvisual: {default: null}\nagent: {mode: 'on'}\n"

    def test_agent_mode_on_runs_the_step_without_the_flag(self):
        out, _ = self.run_search(LOOKALIKE, client=FakeClient('{"candidate": "A", "value": "12.1"}'))
        self.assertEqual(self.results(out).split("\n")[0], "Answer: 12.1 (a.pdf, p. 3)")


class TestFileOf(unittest.TestCase):
    def test_a_slide_image_belongs_to_its_deck_and_other_units_to_their_doc(self):
        from docsearch.manifest import file_of, file_page

        self.assertEqual([file_of("fin/ibm-3q18_page_17#p1", "slide_image"), file_of("ocb/a.pdf#p3", "pdf"),
                          file_of("ocb/a.docx", "docx")],
                         ["fin/ibm-3q18", "ocb/a.pdf", "ocb/a.docx"])  # fmt: skip
        self.assertEqual([file_page("fin/ibm-3q18_page_17", [1], "slide_image"), file_page("ocb/a.pdf", [5, 4], "pdf"),
                          file_page("ocb/a.docx", None, "docx")],
                         [("fin/ibm-3q18", 17), ("ocb/a.pdf", 4), ("ocb/a.docx", None)])  # fmt: skip

    def test_a_pdf_named_like_a_slide_stays_its_own_file_with_its_real_page(self):
        from docsearch.manifest import file_of, file_page

        self.assertEqual(file_of("inv/invoice_page_1#p3", "pdf"), "inv/invoice_page_1")
        self.assertEqual(file_page("inv/invoice_page_1", [3], "pdf"), ("inv/invoice_page_1", 3))
        self.assertEqual(file_page("inv/invoice_page_1", None, "image"), ("inv/invoice_page_1", None))


class TestPdfNamedLikeASlide(SearchCli):
    def test_two_pdfs_named_page_n_are_two_files(self):
        hits = [hit("inv/invoice_page_1", 3, "Total 120", "pdf", score=0.03, rerank_score=0.95),
                hit("inv/invoice_page_2", 1, "Total 80", "pdf", score=0.02, rerank_score=0.9)]  # fmt: skip
        out, _ = self.run_search(hits)
        self.assertEqual(self.results(out).split("\n"), [
            " 1. invoice_page_1", "    inv/invoice_page_1", "    p. 3   Total 120",
            " 2. invoice_page_2", "    inv/invoice_page_2", "    p. 1   Total 80"])  # fmt: skip


if __name__ == "__main__":
    unittest.main()


class TestCleanNames(unittest.TestCase):
    """Snippets show no Markdown bold marks; a file named by an opaque id shows its first line as its name."""

    UUID = "ocb/69deff4d-60ce-545d-40e6-af3e257499f2.docx"

    def test_snippets_have_no_bold_marks(self):
        from docsearch.display import group

        g = group(
            "supplier", [hit("ocb/terms.docx", None, "12. **Indemnification:** Supplier shall indemnify", "docx")], {}
        )
        self.assertEqual(g[0].pages[0].snippet, "12. Indemnification: Supplier shall indemnify")

    def test_an_opaque_id_file_is_named_by_the_first_line_of_its_first_chunk(self):
        from docsearch.display import first_lines, group

        rows = [{"chunk_id": f"hybrid:{self.UUID}:3", "doc_id": self.UUID, "text": "File: x.docx\n6. **Labeling:** Supplier"},
                {"chunk_id": f"hybrid:{self.UUID}:0", "doc_id": self.UUID,
                 "text": "File: x.docx\n**General Terms and Conditions (APAC)**\n“\nmore"},
                {"chunk_id": "hybrid:ocb/plan.docx:0", "doc_id": "ocb/plan.docx", "text": "File: plan.docx\nBudget"}]  # fmt: skip
        with mock.patch("docsearch.search.page_rows", return_value=rows) as rows_of:
            names = first_lines(load_config(env={}), "hybrid", [self.UUID, "ocb/plan.docx"])
        self.assertEqual(rows_of.call_args.args[2], {self.UUID})  # only opaque names are looked up
        self.assertEqual(names, {self.UUID: "General Terms and Conditions (APAC)"})
        g = group("supplier", [hit(self.UUID, None, "x", "docx"), hit("ocb/plan.docx", None, "y", "docx")], {}, names)
        self.assertEqual([x.name for x in g], ["General Terms and Conditions (APAC)", "plan.docx"])

    def test_a_template_title_falls_back_to_the_first_line_then_the_file_name(self):
        from docsearch.display import group

        metas = {d: DocMeta(d, d, "pptx", t) for d, t in (("s/deck.pptx", "PowerPoint Presentation"),
                 ("s/talk.pptx", "Slide 1"), ("s/plan.docx", "Microsoft®"), ("s/memo.docx", "01531831"),
                 ("s/case.docx", "Full Business\nCase"))}  # fmt: skip
        hits = [hit(d, None, "x", "pptx") for d in metas]
        g = group("x", hits, metas, {"s/talk.pptx": "Research methods"})
        self.assertEqual([x.name for x in g], ["deck.pptx", "Research methods", "plan.docx", "memo.docx",
                                               "Full Business Case"])  # fmt: skip

    def test_the_first_line_skips_marks_and_default_table_names(self):
        from docsearch.display import first_lines

        docs = {"ocb/01e02bba-2310-dea2-09d0-0448b06c05b2.docx": "```\nSTUDY ON BANK REPORTS\nM.A Nguyen",
                "ocb/e7a447e5-640d-8c9c-d429-8bffd64c5a2e.docx": "*Seller to provide the information*\n**Buyer**",
                "ocb/3e149c6d-eb16-e7e8-1546-9d41171ab25e.xlsx": "Table 1\n" + "word " * 40,
                "ocb/5a0e8b1c-1d2e-4f3a-9b8c-7d6e5f4a3b2c.xlsx": "| Feature | Explanation |\n| Cost | Low |"}  # fmt: skip
        rows = [{"chunk_id": f"hybrid:{d}:0", "doc_id": d, "text": f"File: x\n{t}"} for d, t in docs.items()]
        with mock.patch("docsearch.search.page_rows", return_value=rows):
            names = first_lines(load_config(env={}), "hybrid", list(docs))
        self.assertEqual(list(names.values()), ["STUDY ON BANK REPORTS", "Seller to provide the information"])
