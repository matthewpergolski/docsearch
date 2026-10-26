import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.config import load_config
from docsearch.llm import LLMError
from docsearch.manifest import ranked_units


def chunk(doc: str, page: int | None, text: str, n: int = 0, fmt: str = "pdf") -> dict:
    return {"chunk_id": f"{doc}:{page}:{n}", "doc_id": doc, "format": fmt, "pages": [page] if page else None,
            "text": f"File: {doc.split('/')[-1]}\n{text}"}  # fmt: skip


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


LOOKALIKE = [chunk("s/a.pdf", 3, "Croatia 12.1 2022"), chunk("s/b.pdf", 1, "other report"),
             chunk("s/a.pdf", 4, "Croatia 14.9 2023"), chunk("s/a.pdf", 3, "more of page 3", 1),
             chunk("s/c.pdf", 2, "unrelated")]  # fmt: skip


class TestGate(unittest.TestCase):
    def test_two_pages_of_one_file_in_the_top_k_pass(self):
        from docsearch.agent import gate

        self.assertTrue(gate(LOOKALIKE, 5))

    def test_one_page_per_file_does_not_pass(self):
        from docsearch.agent import gate

        hits = [chunk(f"s/{d}.pdf", 1, "x") for d in "abcde"] + [chunk("s/a.pdf", 2, "x")]
        self.assertFalse(gate(hits, 5))
        self.assertTrue(gate(hits, 6))

    def test_chunks_of_the_same_page_count_once(self):
        from docsearch.agent import gate

        hits = [chunk("s/a.pdf", 1, "x", 0), chunk("s/a.pdf", 1, "y", 1), chunk("s/b.pdf", 1, "z")]
        self.assertFalse(gate(hits, 5))

    def test_sheets_of_one_workbook_pass_and_a_file_without_pages_is_one_page(self):
        from docsearch.agent import gate

        self.assertTrue(gate([chunk("s/w.xlsx", 1, "a", fmt="xlsx"), chunk("s/w.xlsx", 2, "b", fmt="xlsx")], 5))
        self.assertFalse(gate([chunk("s/d.docx", None, "a", 0, "docx"), chunk("s/d.docx", None, "b", 1, "docx")], 5))


def slide(deck: str, n: int, text: str = "x") -> dict:
    return chunk(f"f/{deck}_page_{n}", None, text, fmt="slide_image")


# Slides of 4 decks: two slides of deck q1 in the top 5 pages, then 20 more slides, one per deck.
DECK = [slide("q1", 3, "revenue 2021"), slide("q2", 3), slide("q1", 7, "margin 2021"), slide("q3", 3)] + [
    slide(f"d{i}", 1, f"slide {i}") for i in range(20)
]


class TestDeckGate(unittest.TestCase):
    def test_two_slides_of_one_deck_pass_only_when_a_deck_counts_as_one_file(self):
        from docsearch.agent import gate

        self.assertFalse(gate(DECK, 5))
        self.assertTrue(gate(DECK, 5, decks=True))
        self.assertFalse(gate(DECK, 2, decks=True))

    def test_other_formats_are_unchanged(self):
        from docsearch.agent import gate

        pdfs = [chunk("f/r_page_1", 1, "x"), chunk("f/r_page_2", 1, "y"), chunk("f/s.pdf", 1, "z")]
        self.assertEqual([gate(h, 5, decks=True) for h in (pdfs, LOOKALIKE)], [False, True])


class TestParseReply(unittest.TestCase):
    def test_replies(self):
        from docsearch.agent import parse_reply

        cases = {
            '{"candidate": "B", "value": "14.9%"}': (2, "14.9%"),
            'Here: {"candidate": "[c]", "value": 14.9}': (3, "14.9"),
            '{"candidate": null, "value": null}': (None, None),
            '{"candidate": 2, "value": "1"}': "malformed",
            '{"candidate": "F", "value": "1"}': "malformed",
            '{"candidate": "two"}': "malformed",
            '{"page": 2, "value": "1"}': "malformed",
            "no json at all": "malformed",
        }
        for reply, want in cases.items():
            got = parse_reply(reply, 5)
            self.assertEqual("malformed" if got is None else got, want, reply)


class AgentCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cache = Path(self.td.name) / "agent"
        rows = LOOKALIKE + [chunk("s/a.pdf", 4, "Croatia rest of page 4", 7)]
        from docsearch.agent import PageTexts

        self.texts = PageTexts(rows)

    def tearDown(self):
        self.td.cleanup()

    def agent(self, client, **kw):
        from docsearch.agent import Agent

        return Agent(client, "openai:gpt-6-astra", self.texts, self.cache, **kw)


class TestAgentStep(AgentCase):
    def test_chosen_page_goes_first_then_the_plain_order(self):
        client = FakeClient('{"candidate": "C", "value": "14.9"}')
        hits, step = self.agent(client)("Croatia 2023 rate?", LOOKALIKE)
        self.assertEqual([c["chunk_id"] for c in hits],
                         ["s/a.pdf:4:0", "s/a.pdf:3:0", "s/b.pdf:1:0", "s/a.pdf:3:1", "s/c.pdf:2:0"])  # fmt: skip
        self.assertEqual((step.chosen, step.value, step.calls, step.cached), ("s/a.pdf#p4", "14.9", 1, False))
        self.assertEqual(step.candidates, ["s/a.pdf#p3", "s/b.pdf#p1", "s/a.pdf#p4", "s/c.pdf#p2"])

    def test_reorder_keeps_every_chunk_and_unit_exactly_once(self):
        for page in range(1, 5):
            hits, _ = self.agent(FakeClient(f'{{"candidate": "{"ABCD"[page - 1]}", "value": "1"}}'))(
                f"q{page}", LOOKALIKE
            )
            self.assertEqual(sorted(c["chunk_id"] for c in hits), sorted(c["chunk_id"] for c in LOOKALIKE))
            units = ranked_units(hits, "page")
            self.assertEqual(sorted(units), sorted(ranked_units(LOOKALIKE, "page")))
            self.assertEqual(len(units), len(set(units)))
            self.assertEqual(units[0], ["s/a.pdf#p3", "s/b.pdf#p1", "s/a.pdf#p4", "s/c.pdf#p2"][page - 1])

    def test_no_page_and_unusable_replies_keep_the_plain_order(self):
        for replies, calls, error in (
            (['{"candidate": null, "value": null}'], 1, None),
            (["sorry", '{"candidate": "G"}'], 2, "malformed reply"),
            ([LLMError("codex exited 1")], 1, "codex exited 1"),
        ):
            hits, step = self.agent(FakeClient(*replies))("q " + str(replies), LOOKALIKE)
            self.assertEqual(hits, LOOKALIKE)
            self.assertEqual((step.chosen, step.calls, step.error), (None, calls, error))

    def test_one_retry_after_a_malformed_reply(self):
        client = FakeClient("not json", '{"candidate": "A", "value": "12.1"}')
        hits, step = self.agent(client)("q", LOOKALIKE)
        self.assertEqual((step.chosen, step.calls, hits[0]["chunk_id"]), ("s/a.pdf#p3", 2, "s/a.pdf:3:0"))
        self.assertEqual(client.prompts[0], client.prompts[1])

    def test_a_query_that_fails_the_gate_makes_no_call(self):
        hits = [chunk(f"s/{d}.pdf", 1, "x") for d in "abc"]
        client = FakeClient()
        out, step = self.agent(client)("q", hits)
        self.assertEqual((out, step.gate, step.calls, client.prompts), (hits, False, 0, []))

    def test_cache_hit_makes_no_call(self):
        first, _ = self.agent(FakeClient('{"candidate": "C", "value": "14.9"}'))("Croatia 2023 rate?", LOOKALIKE)
        client = FakeClient()
        again, step = self.agent(client)("Croatia 2023 rate?", LOOKALIKE)
        self.assertEqual((again, step.chosen, step.cached, client.prompts), (first, "s/a.pdf#p4", True, []))

    def test_errors_are_not_cached(self):
        self.agent(FakeClient(LLMError("boom")))("q", LOOKALIKE)
        client = FakeClient('{"candidate": "B", "value": "x"}')
        _, step = self.agent(client)("q", LOOKALIKE)
        self.assertEqual((step.chosen, len(client.prompts)), ("s/b.pdf#p1", 1))

    def test_prompt_shows_retrieved_chunks_first_and_caps_each_page(self):
        client = FakeClient('{"candidate": null, "value": null}')
        self.agent(client, page_chars=38)("Croatia 2023 rate?", LOOKALIKE)
        p = client.prompts[0]
        self.assertIn("Question: Croatia 2023 rate?", p)
        self.assertIn("[A] a.pdf, page 3\nCroatia 12.1 2022\nmore of page 3\n", p)
        self.assertIn("[C] a.pdf, page 4\nCroatia 14.9 2023\nCroatia rest of page\n", p)
        self.assertNotIn("File:", p)

    def test_row_records_the_step_without_page_text(self):
        _, step = self.agent(FakeClient('{"candidate": "A", "value": "12.1"}'))("q", LOOKALIKE)
        row = step.row()
        self.assertEqual({k: row[k] for k in ("gate", "candidates", "chosen", "value", "model", "calls", "cached")},
                         {"gate": True, "candidates": ["s/a.pdf#p3", "s/b.pdf#p1", "s/a.pdf#p4", "s/c.pdf#p2"],
                          "chosen": "s/a.pdf#p3", "value": "12.1", "model": "openai:gpt-6-astra", "calls": 1,
                          "cached": False})  # fmt: skip
        cached = json.loads((self.cache / f"{row['prompt_hash']}.json").read_text())
        self.assertEqual(cached["reply"], '{"candidate": "A", "value": "12.1"}')
        self.assertIn("Croatia 12.1 2022", cached["prompt"])

    def test_an_unusable_llm_stops_the_run_and_a_rerun_resumes_from_the_cache(self):
        from docsearch.llm import LLMUnavailable

        queries, lists = ["q1", "q2", "q3"], [LOOKALIKE] * 3
        client = FakeClient(
            '{"candidate": "B", "value": "a"}', LLMUnavailable("`codex` hit a usage limit: try at 9:35 PM")
        )
        with self.assertRaises(LLMUnavailable):
            self.agent(client).run_all(queries, lists, workers=1)
        self.assertEqual(len(client.prompts), 2)
        client = FakeClient('{"candidate": "C", "value": "b"}', '{"candidate": "D", "value": "c"}')
        out = self.agent(client).run_all(queries, lists, workers=1)
        self.assertEqual([(s.chosen, s.cached) for _, s in out],
                         [("s/b.pdf#p1", True), ("s/a.pdf#p4", False), ("s/c.pdf#p2", False)])  # fmt: skip

    def test_page_text_quoting_429_in_a_failed_call_is_one_error_and_the_run_goes_on(self):
        from types import SimpleNamespace

        from docsearch.agent import summarize
        from docsearch.llm import LLMClient, LLMSettings

        page = "Errors by day: HTTP 429 (rate limit) on 12 days"
        procs = [SimpleNamespace(returncode=1, stdout=f"The page says: {page}", stderr="error: stream disconnected"),
                 SimpleNamespace(returncode=0, stdout='{"candidate": "C", "value": "14.9"}', stderr="")]  # fmt: skip
        client = LLMClient(LLMSettings("cli", "claude", "haiku", 5), runner=lambda cmd, **kw: procs.pop(0),
                           which=lambda name: f"/bin/{name}")  # fmt: skip
        out = self.agent(client).run_all(["q1", "q2"], [LOOKALIKE, LOOKALIKE], workers=1)
        self.assertEqual([s.chosen for _, s in out], [None, "s/a.pdf#p4"])
        self.assertTrue(out[0][1].error.startswith("`claude` exited 1: "))
        self.assertEqual(summarize([s for _, s in out])["errors"], 1)

    def test_run_all_keeps_the_query_order_and_counts(self):
        from docsearch.agent import summarize

        client = FakeClient('{"candidate": "A", "value": "12.1"}', '{"candidate": null, "value": null}')
        lone = [chunk("s/z.pdf", 1, "x")]
        out = self.agent(client).run_all(["q1", "q2", "q3"], [LOOKALIKE, lone, LOOKALIKE], workers=1)
        self.assertEqual([s.chosen for _, s in out], ["s/a.pdf#p3", None, None])
        stats = summarize([s for _, s in out])
        self.assertEqual({k: stats[k] for k in ("queries", "gate_passed", "llm_calls", "chosen", "moved", "no_page")},
                         {"queries": 3, "gate_passed": 2, "llm_calls": 2, "chosen": 1, "moved": 0, "no_page": 1})  # fmt: skip


class TestDeckStep(AgentCase):
    def test_off_by_default_slides_of_one_deck_make_no_call(self):
        client = FakeClient()
        out, step = self.agent(client)("q1 margin 2021?", DECK)
        self.assertEqual((out, step.gate, client.prompts), (DECK, False, []))

    def test_a_deck_gate_reads_the_top_20_pages_lettered_a_to_t(self):
        client = FakeClient('{"candidate": null, "value": null}')
        _, step = self.agent(client, deck_gate=True)("q1 margin 2021?", DECK)
        p = client.prompts[0]
        self.assertTrue(p.startswith("You help a document search engine. A user asked the question below. The search "
                                     "found the 20 pages that\nfollow;"))  # fmt: skip
        self.assertIn("[C] q1_page_7\nmargin 2021\n", p)
        self.assertIn("[T] d15_page_1\nslide 15\n", p)
        self.assertNotIn("[U]", p)
        self.assertEqual((step.gate, step.deck, len(step.candidates), step.candidates[19]),
                         (True, True, 20, "f/d15_page_1#p1"))  # fmt: skip

    def test_a_deck_gate_never_reads_fewer_pages_than_the_plain_gate(self):
        # CR2: with agent.pages above 20 the deck reading must not shrink.
        hits = DECK + [chunk(f"f/x{i}_page_1", 1, f"x {i}", fmt="slide_image") for i in range(10)]
        client = FakeClient('{"candidate": null, "value": null}')
        _, step = self.agent(client, deck_gate=True, pages=24)("q1 margin 2021?", hits)
        self.assertEqual(len(step.candidates), min(24, len({c["doc_id"] for c in hits})))

    def test_a_page_from_ranks_11_to_20_goes_first_then_the_plain_order(self):
        client = FakeClient('{"candidate": "P", "value": "slide 11"}')
        hits, step = self.agent(client, deck_gate=True)("q?", DECK)
        self.assertEqual([c["doc_id"] for c in hits[:3]], ["f/d11_page_1", "f/q1_page_3", "f/q2_page_3"])
        self.assertEqual(sorted(c["chunk_id"] for c in hits), sorted(c["chunk_id"] for c in DECK))
        self.assertEqual((step.chosen, step.moved, step.value), ("f/d11_page_1#p1", True, "slide 11"))

    def test_a_letter_past_the_shown_pages_is_malformed(self):
        hits, step = self.agent(FakeClient('{"candidate": "U"}', '{"candidate": "U"}'), deck_gate=True)("q?", DECK)
        self.assertEqual((hits, step.error, step.calls), (DECK, "malformed reply", 2))

    def test_a_query_the_m34h_gate_passes_gets_the_same_10_page_prompt_and_its_cached_reply(self):
        hits = LOOKALIKE + [chunk(f"s/x{i}.pdf", 1, "x") for i in range(8)]
        first, off = self.agent(FakeClient('{"candidate": "C", "value": "14.9"}'))("Croatia 2023?", hits)
        client = FakeClient()
        again, on = self.agent(client, deck_gate=True)("Croatia 2023?", hits)
        self.assertEqual((again, on.prompt_hash, on.cached, on.deck, client.prompts), (first, off.prompt_hash, True,
                                                                                       False, []))  # fmt: skip
        self.assertEqual(len(on.candidates), 10)


class FakeSearch:
    """search_many for agent.research: one fixed hit list per rewrite text."""

    def __init__(self, by_text: dict[str, list[dict]]):
        self.by_text, self.calls = by_text, []

    def __call__(self, texts: list[str]) -> list[list[dict]]:
        self.calls.append(list(texts))
        return [self.by_text[t] for t in texts]


NO_PAGE = '{"candidate": null, "value": null}'
REWRITES = '{"rewrites": ["Croatia employment 2023", "Croatian rate 2023"]}'
NEW = chunk("s/d.pdf", 9, "Croatia 2023 rate 14.9")
SEARCH = FakeSearch({"Croatia employment 2023": [NEW, LOOKALIKE[2]],
                     "Croatian rate 2023": [NEW, LOOKALIKE[1]]})  # fmt: skip


class TestParseRewrites(unittest.TestCase):
    def test_replies(self):
        from docsearch.agent import parse_rewrites

        cases = {
            REWRITES: ["Croatia employment 2023", "Croatian rate 2023"],
            'ok {"rewrites": ["a", " b ", "c"]}': ["a", "b"],
            '{"rewrites": ["a", "a", ""]}': ["a"],
            '{"rewrites": []}': "malformed",
            '{"rewrites": "a"}': "malformed",
            '{"queries": ["a"]}': "malformed",
            "no json": "malformed",
        }
        for reply, want in cases.items():
            got = parse_rewrites(reply)
            self.assertEqual("malformed" if got is None else got, want, reply)


class TestResearch(AgentCase):
    """D1: a gated query the reader answers with no page gets two rewrites, plain search for each, an RRF merge of
    the page lists with the original, and one more reading of the merged top pages."""

    def run_research(self, client, search=SEARCH, hit_lists=None, queries=None):
        ag = self.agent(client, rrf_k=60)
        queries = queries or ["Croatia 2023 rate?"]
        hit_lists = hit_lists or [LOOKALIKE] * len(queries)
        out = ag.run_all(queries, hit_lists, workers=1)
        return ag.research_all(queries, hit_lists, out, search, workers=1)

    def test_named_page_from_the_merged_list_goes_first_then_the_plain_order(self):
        client = FakeClient(NO_PAGE, REWRITES, '{"candidate": "A", "value": "14.9"}')
        [(hits, step)] = self.run_research(client)
        self.assertEqual([c["chunk_id"] for c in hits], ["s/d.pdf:9:0", *[c["chunk_id"] for c in LOOKALIKE]])
        self.assertIn("Question: Croatia 2023 rate?", client.prompts[1])
        self.assertIn('{"rewrites": ["<query 1>", "<query 2>"]}', client.prompts[1])
        r = step.research
        self.assertEqual((step.chosen, r.rewrites, r.chosen, r.value, r.moved, r.new_pages),
                         (None, ["Croatia employment 2023", "Croatian rate 2023"], "s/d.pdf#p9", "14.9", True, 1))  # fmt: skip
        self.assertEqual(r.candidates, ["s/d.pdf#p9", "s/b.pdf#p1", "s/a.pdf#p4", "s/a.pdf#p3", "s/c.pdf#p2"])
        self.assertIn("[A] d.pdf, page 9\nCroatia 2023 rate 14.9\n", client.prompts[2])
        self.assertEqual((r.rewrite.calls, r.read.calls, len(client.prompts)), (1, 1, 3))

    def test_a_page_that_was_already_listed_moves_up_as_in_m34h(self):
        client = FakeClient(NO_PAGE, REWRITES, '{"candidate": "C", "value": "14.9"}')
        [(hits, step)] = self.run_research(client)
        self.assertEqual([c["chunk_id"] for c in hits],
                         ["s/a.pdf:4:0", "s/a.pdf:3:0", "s/b.pdf:1:0", "s/a.pdf:3:1", "s/c.pdf:2:0"])  # fmt: skip
        self.assertEqual((step.research.chosen, step.research.moved), ("s/a.pdf#p4", True))

    def test_no_page_again_or_unusable_replies_keep_the_plain_order(self):
        for replies, error, searched in (
            ([NO_PAGE, REWRITES, NO_PAGE], None, True),
            ([NO_PAGE, "no", '{"rewrites": []}'], "malformed rewrites", False),
            ([NO_PAGE, LLMError("claude exited 1")], "claude exited 1", False),
            ([NO_PAGE, REWRITES, "x", "y"], "malformed reply", True),
        ):
            search = FakeSearch(SEARCH.by_text)
            [(hits, step)] = self.run_research(FakeClient(*replies), search, queries=[f"q {replies}"])
            self.assertEqual(hits, LOOKALIKE)
            self.assertEqual((step.research.chosen, step.research.error, bool(search.calls)), (None, error, searched))

    def test_only_gated_queries_with_no_page_are_researched(self):
        lone = [chunk("s/z.pdf", 1, "x")]
        client = FakeClient('{"candidate": "A", "value": "1"}', NO_PAGE, REWRITES, NO_PAGE)
        out = self.run_research(
            client, queries=["picked", "nothing", "ungated"], hit_lists=[LOOKALIKE, LOOKALIKE, lone]
        )
        self.assertEqual([s.research is not None for _, s in out], [False, True, False])
        self.assertEqual(len(client.prompts), 4)

    def test_a_rerun_makes_no_call_and_gives_the_same_lists(self):
        first = self.run_research(FakeClient(NO_PAGE, REWRITES, '{"candidate": "A", "value": "14.9"}'))
        client = FakeClient()
        again = self.run_research(client)
        self.assertEqual(([h for h, _ in again], client.prompts), ([h for h, _ in first], []))
        self.assertEqual((again[0][1].research.rewrite.cached, again[0][1].research.read.cached), (True, True))

    def test_summary_counts_the_research_calls(self):
        from docsearch.agent import summarize

        out = self.run_research(FakeClient(NO_PAGE, REWRITES, '{"candidate": "A", "value": "14.9"}'))
        st = summarize([s for _, s in out])
        self.assertEqual({k: st[k] for k in ("llm_calls", "no_page")}, {"llm_calls": 1, "no_page": 1})
        self.assertEqual({k: st["research"][k] for k in ("researched", "llm_calls", "chosen", "moved", "new_pages")},
                         {"researched": 1, "llm_calls": 2, "chosen": 1, "moved": 1, "new_pages": 1})  # fmt: skip

    def test_without_research_the_step_row_records_none(self):
        _, step = self.agent(FakeClient(NO_PAGE))("q", LOOKALIKE)
        self.assertIsNone(step.row()["research"])


def write_yaml(d: str, text: str) -> Path:
    p = Path(d) / "docsearch.yaml"
    p.write_text(text)
    return p


class TestAgentRunSettings(unittest.TestCase):
    YAML = "models: {embedders: {v: {id: org/v}}}\nagent: {provider: openai, model: gpt-6-astra}\n"

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cfg = load_config(write_yaml(self.td.name, self.YAML), env={})

    def tearDown(self):
        self.td.cleanup()

    def write_runs(self, rows: list[tuple[str, dict]]) -> None:
        from docsearch.eval import RUN_FIELDS

        out = Path(self.td.name) / "results"
        (out / "runs").mkdir(parents=True)
        with (out / "runs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
            w.writeheader()
            for run_id, blob in rows:
                w.writerow({"run_id": run_id, "name": run_id, "chunking": "hybrid", "retrieval": "hybrid_visual_fa",
                            "rerank": "True", "embedder": "q + org/v", "reranker": self.cfg["models"]["reranker"],
                            "rerank_candidates": self.cfg["search"]["rerank_candidates"],
                            "rerank_max_length": self.cfg["models"]["rerank_max_length"],
                            "notes": "full; " + json.dumps(blob)})  # fmt: skip
                (out / "runs" / f"{run_id}.jsonl").write_text("{}\n")

    def test_agent_run_never_counts_as_a_plain_reference(self):
        from docsearch.eval import cached_reference, run_settings

        plain, agent = run_settings(self.cfg)["blob"], run_settings(self.cfg, "agent")["blob"]
        self.assertEqual(agent, {**plain, "mode": "agent", "agent": {"backend": "cli", "provider": "openai",
                                 "model": "gpt-6-astra", "gate_k": 5, "pages": 10, "page_chars": 4000,
                                 "prompt": agent["agent"]["prompt"]}})  # fmt: skip
        self.write_runs([("r1-plain", plain), ("r2-agent", {**agent, "agent_stats": {"llm_calls": 3}})])
        self.assertEqual(cached_reference(self.cfg, True, "hybrid", "org/v")["run_id"], "r1-plain")
        self.assertEqual(cached_reference(self.cfg, True, "hybrid", "org/v", "agent")["run_id"], "r2-agent")

    def test_a_research_run_records_it_and_is_not_an_m34h_reference(self):
        from docsearch.agent import RESEARCH_VERSION
        from docsearch.eval import cached_reference, run_settings

        agent = run_settings(self.cfg, "agent")["blob"]
        self.cfg["agent"]["research"] = "on"
        research = run_settings(self.cfg, "agent")["blob"]
        self.assertEqual(research["agent"], {**agent["agent"], "research": {"rewrites": 2, "prompt": RESEARCH_VERSION}})
        self.write_runs([("r1-research", research)])
        self.cfg["agent"]["research"] = "off"
        self.assertIsNone(cached_reference(self.cfg, True, "hybrid", "org/v", "agent"))

    def test_a_deck_run_records_it_and_is_not_an_m34h_reference(self):
        from docsearch.eval import cached_reference, run_settings

        agent = run_settings(self.cfg, "agent")["blob"]
        self.cfg["agent"]["deck_gate"] = "on"
        deck = run_settings(self.cfg, "agent")["blob"]
        self.assertEqual(deck["agent"], {**agent["agent"], "deck_gate": {"pages": 20}})
        self.write_runs([("r1-deck", deck)])
        self.cfg["agent"]["deck_gate"] = "off"
        self.assertIsNone(cached_reference(self.cfg, True, "hybrid", "org/v", "agent"))

    def test_a_grok_agent_run_is_not_a_codex_agent_reference(self):
        from docsearch.eval import cached_reference, run_settings

        agent = run_settings(self.cfg, "agent")["blob"]
        self.write_runs([("r1-grok", {**agent, "agent": {**agent["agent"], "provider": "grok", "model": ""}})])
        self.assertIsNone(cached_reference(self.cfg, True, "hybrid", "org/v", "agent"))


class TestAgentEvalCli(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.yaml = write_yaml(self.td.name, "models: {embedders: {v: {id: org/v}}}\n")
        self.calls = []

    def tearDown(self):
        self.td.cleanup()

    def eval_cli(self, *args):
        from docsearch import cli

        def fake_eval(cfg, plan, *a, **k):
            self.calls.append({"plan": plan, "mode": k.get("eval_mode"), "agent.mode": cfg["agent"]["mode"],
                               "provider": cfg["agent"]["provider"], "model": cfg["agent"]["model"],
                               **({"research": "on"} if cfg["agent"]["research"] == "on" else {}),
                               **({"deck": "on"} if cfg["agent"]["deck_gate"] == "on" else {})})  # fmt: skip
            return []

        ref = {"name": "ref", "run_id": "r", "ndcg@10": "0.8", "n_queries": 1}
        with mock.patch("docsearch.eval.run_eval", fake_eval), mock.patch("docsearch.report.build_report"), \
                mock.patch("docsearch.cli._device_line", lambda c: ""), mock.patch("docsearch.cli._quiet_hub"), \
                mock.patch("docsearch.eval.cached_reference", return_value=ref), \
                mock.patch("docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            return cli.main(["--config", str(self.yaml), "eval", "--full", *args])

    def test_agent_mode_is_one_named_challenger_run(self):
        self.eval_cli("--visual", "v", "--fusion", "format_aware", "--mode", "agent", "--name", "m34")
        self.eval_cli("--visual", "v", "--fusion", "format_aware", "--mode", "agent", "--agent-provider", "grok",
                      "--name", "m34_grok")  # fmt: skip
        self.assertEqual(self.calls, [
            {"plan": [("m34", "hybrid", "hybrid_visual_fa", True)], "mode": "agent", "agent.mode": "on",
             "provider": "claude", "model": "haiku"},
            {"plan": [("m34_grok", "hybrid", "hybrid_visual_fa", True)], "mode": "agent", "agent.mode": "on",
             "provider": "grok", "model": ""}])  # fmt: skip

    def test_agent_research_is_its_own_named_run_and_needs_agent_mode(self):
        self.eval_cli("--visual", "v", "--fusion", "format_aware", "--mode", "agent", "--agent-research")
        self.assertEqual(self.calls, [
            {"plan": [("hybrid_visual_fa_v_rerank_agent_research", "hybrid", "hybrid_visual_fa", True)],
             "mode": "agent", "agent.mode": "on", "provider": "claude", "model": "haiku", "research": "on"}])  # fmt: skip
        with self.assertRaises(SystemExit) as cm:
            self.eval_cli("--visual", "v", "--fusion", "format_aware", "--agent-research")
        self.assertEqual(str(cm.exception), "--agent-research needs --mode agent; nothing was run")
        self.assertEqual(len(self.calls), 1)

    def test_agent_deck_is_its_own_named_run_and_needs_agent_mode(self):
        self.eval_cli("--visual", "v", "--fusion", "format_aware", "--mode", "agent", "--agent-deck")
        self.assertEqual(self.calls, [
            {"plan": [("hybrid_visual_fa_v_rerank_agent_deck", "hybrid", "hybrid_visual_fa", True)],
             "mode": "agent", "agent.mode": "on", "provider": "claude", "model": "haiku", "deck": "on"}])  # fmt: skip
        with self.assertRaises(SystemExit) as cm:
            self.eval_cli("--visual", "v", "--fusion", "format_aware", "--agent-deck")
        self.assertEqual(str(cm.exception), "--agent-deck needs --mode agent; nothing was run")
        self.assertEqual(len(self.calls), 1)

    def test_agent_mode_needs_the_default_pipeline(self):
        with self.assertRaises(SystemExit) as cm:
            self.eval_cli("--mode", "agent")
        self.assertEqual(str(cm.exception), "eval --mode agent runs one challenger on the default pipeline: "
                                            "give --visual ALIAS --fusion format_aware (and --name)")  # fmt: skip
        self.assertEqual(self.calls, [])


class TestNotesSettings(unittest.TestCase):
    def test_the_blob_that_ends_the_notes_with_a_nested_agent_blob(self):
        from docsearch.eval import _notes_settings

        blob = {"dpi": 96, "sheet_cap": "shared", "mode": "agent",
                "agent": {"provider": "claude", "research": {"rewrites": 2}}, "agent_stats": {"llm_calls": 3}}  # fmt: skip
        notes = ('full; questions=hard(reviewed); reviewed_by={"reviewer": 87}; {"by_type": {"x": {"n": 1}}}; '
                 + json.dumps(blob))  # fmt: skip
        self.assertEqual(_notes_settings(notes), blob)

    def test_an_old_flat_blob_and_notes_without_one(self):
        from docsearch.eval import _notes_settings

        self.assertEqual(_notes_settings('full; {"dpi": 96, "visual_k": 100, "sheet_cap": "shared"}'),
                         {"dpi": 96, "visual_k": 100, "sheet_cap": "shared"})  # fmt: skip
        self.assertEqual(_notes_settings("full; box subset"), {})
        self.assertEqual(_notes_settings('full; {"a": 1}; trailing words'), {})


if __name__ == "__main__":
    unittest.main()
