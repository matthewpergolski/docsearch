import ast
import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.chunk import Chunk
from docsearch.enrich import Enricher, apply, outline


def ck(doc: str, n: int, body: str, fmt: str = "pdf") -> Chunk:
    return Chunk(f"hybrid:{doc}:{n}", doc, fmt, "src", [n + 1], f"File: {doc.split('/')[-1]}\n{body}")


class FakeClient:
    """Replies with one line per chunk number it finds in the prompt, or with the given replies in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.replies:
            return self.replies.pop(0)
        n = prompt.count("\n[")
        return json.dumps({str(i): f"line {i}" for i in range(1, n + 1)})


DOC = [ck("src/report", 0, "Annual report 2024\nIntro text"), ck("src/report", 1, "Table 3 wages by region\n1 2 3"),
       ck("src/report", 2, "Annex B\nmore")]  # fmt: skip


class TestEnrich(unittest.TestCase):
    def test_enriched_variant_reuses_hybrid(self):
        from docsearch.chunk import base_variant

        self.assertEqual(base_variant("hybrid_ctx"), "hybrid")
        self.assertEqual(base_variant("hybrid_tbl"), "hybrid")
        self.assertIsNone(base_variant("hybrid"))

    def test_eval_of_the_enriched_variant_is_a_named_one_change_run(self):
        # H1: --variant hybrid_ctx ran the generic rerank/norerank pair and dropped --name.
        import contextlib

        from docsearch import cli

        seen = {}

        def fake_eval(cfg, plan, *a, **k):
            seen["plan"] = plan
            return []

        with tempfile.TemporaryDirectory() as td:
            y = Path(td) / "docsearch.yaml"
            y.write_text("models: {embedders: {pplx-v2-late: {id: org/vis, kind: visual}}}\n")
            ref = lambda cfg, full, variant, vid: seen.setdefault("ref", variant) and {  # noqa: E731
                "name": "m3", "run_id": "r", "ndcg@10": "0.7", "n_queries": "1"}  # fmt: skip
            with mock.patch("docsearch.eval.run_eval", fake_eval), mock.patch("docsearch.report.build_report"), \
                 mock.patch("docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), \
                 mock.patch("docsearch.eval.cached_reference", ref), contextlib.suppress(SystemExit):  # fmt: skip
                cli.main(["--config", str(y), "eval", "--visual", "pplx-v2-late", "--fusion", "format_aware",
                          "--variant", "hybrid_ctx", "--name", "h1_ctx"])  # fmt: skip
        self.assertEqual(seen.get("ref"), "hybrid")
        self.assertEqual([p[:2] for p in seen["plan"]], [("h1_ctx", "hybrid_ctx")])

    def test_lines_for_every_chunk_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            c = FakeClient()
            e = Enricher(c, "claude:haiku", Path(d), batch=2)
            lines = e.lines(DOC)
            self.assertEqual(lines, {DOC[0].chunk_id: "line 1", DOC[1].chunk_id: "line 2", DOC[2].chunk_id: "line 1"})
            self.assertEqual(len(c.prompts), 2)
            self.assertEqual(e.stats["calls"], 2)
            again = Enricher(FakeClient(), "claude:haiku", Path(d), batch=2)
            self.assertEqual(again.lines(DOC), lines)
            self.assertEqual(again.stats["calls"], 0)
            self.assertEqual(again.stats["cached_batches"], 2)

    def test_changed_chunk_calls_again_only_for_its_batch(self):
        with tempfile.TemporaryDirectory() as d:
            Enricher(FakeClient(), "claude:haiku", Path(d), batch=2).lines(DOC)
            changed = [DOC[0], DOC[1], ck("src/report", 2, "Annex B\nnew text")]
            e = Enricher(FakeClient(), "claude:haiku", Path(d), batch=2)
            e.lines(changed)
            self.assertEqual(e.stats["calls"], 1)

    def test_unusable_reply_retried_once_then_chunk_left_without_line(self):
        with tempfile.TemporaryDirectory() as d:
            e = Enricher(FakeClient("no json", "still none"), "claude:haiku", Path(d), batch=5)
            self.assertEqual(e.lines(DOC[:1]), {})
            self.assertEqual(e.stats["calls"], 2)
            self.assertEqual(e.stats["missing"], 1)

    def test_a_reply_that_skips_chunks_is_asked_again_and_the_fuller_one_kept(self):
        # H1: Haiku sometimes answered a batch of 8 with 2 lines; the other 6 chunks never got one.
        with tempfile.TemporaryDirectory() as d:
            e = Enricher(FakeClient('{"1": "a"}', '{"1": "a", "2": "b", "3": "c"}'), "claude:haiku", Path(d), batch=3)
            self.assertEqual(sorted(e.lines(DOC).values()), ["a", "b", "c"])
            self.assertEqual((e.stats["calls"], e.stats["missing"]), (2, 0))
            e = Enricher(FakeClient('{"1": "x", "2": "y"}', '{"1": "x"}'), "claude:haiku", Path(d) / "2", batch=3)
            self.assertEqual(sorted(e.lines(DOC).values()), ["x", "y"])
            self.assertEqual(e.stats["missing"], 1)
            again = Enricher(FakeClient(), "claude:haiku", Path(d) / "2", batch=3)
            again.lines(DOC)
            self.assertEqual(again.stats["calls"], 0)  # asked twice already: the cache holds the best reply

    def test_a_cached_partial_reply_from_one_call_gets_its_second_try(self):
        with tempfile.TemporaryDirectory() as d:
            Enricher(FakeClient('{"1": "a"}', "no json"), "claude:haiku", Path(d), batch=3).lines(DOC)
            path = next(Path(d).glob("*.json"))
            entry = json.loads(path.read_text())
            entry["calls"] = 1  # as written before the retry rule
            path.write_text(json.dumps(entry))
            e = Enricher(FakeClient(), "claude:haiku", Path(d), batch=3)
            self.assertEqual(len(e.lines(DOC)), 3)
            self.assertEqual(e.stats["calls"], 1)

    def test_prompt_holds_outline_and_chunks_only(self):
        with tempfile.TemporaryDirectory() as d:
            c = FakeClient()
            Enricher(c, "claude:haiku", Path(d), batch=3).lines(DOC)
            self.assertIn("report", c.prompts[0])
            self.assertIn("Annual report 2024", c.prompts[0])
            self.assertIn("[3] Annex B", c.prompts[0])

    def test_outline_lists_headings_once(self):
        o = outline(DOC)
        self.assertTrue(o.startswith("File name: report"))
        self.assertEqual(o.count("Annual report 2024"), 1)
        self.assertIn("Table 3 wages by region", o)

    def test_apply_puts_line_after_file_line(self):
        out = apply(DOC, {DOC[1].chunk_id: "Wages table, 2024 annual report"})
        self.assertEqual(
            out[1].text, "File: report\nContext: Wages table, 2024 annual report\nTable 3 wages by region\n1 2 3"
        )
        self.assertEqual(out[0].text, DOC[0].text)
        self.assertEqual(out[1].chunk_id, DOC[1].chunk_id)

    def test_never_opens_a_question_file(self):
        real_open = builtins.open

        def guarded(path, *a, **k):
            if "question" in str(path).lower():
                raise AssertionError(f"enrich opened {path}")
            return real_open(path, *a, **k)

        read = mock.patch.object(Path, "read_text", lambda p, *a, **k: guarded(p).read())
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.open", guarded), read:
            Enricher(FakeClient(), "claude:haiku", Path(d), batch=2).lines(DOC)

    def test_module_imports_no_question_or_eval_code(self):
        src = Path(__file__).resolve().parents[1] / "src" / "docsearch" / "enrich.py"
        names = set()
        for node in ast.walk(ast.parse(src.read_text())):
            if isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
        self.assertFalse({"eval", "questions", "hard", "draft_questions", "judge"} & names, names)
        self.assertNotIn("question", src.read_text().lower())


if __name__ == "__main__":
    unittest.main()
