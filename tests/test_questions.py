import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from docsearch import questions as Q
from docsearch.config import load_config
from docsearch.eval import abstention, format_custom
from docsearch.llm import LLMClient, LLMSettings, cli_command
from docsearch.metrics import score_query


def have(_):
    return "/usr/bin/x"


TEXT = "Revenue grew 12% to $4.2B in 2019 while margins fell 3 points to 41% across segments; " * 4


def chunks():
    out = []
    for src, fmt in (("ocb", "docx"), ("finslides", "slide_image")):
        for d in range(3):
            out.append({"chunk_id": f"hybrid:{src}/d{d}:0", "doc_id": f"{src}/d{d}", "format": fmt, "source": src,
                        "pages": [1], "text": f"{src} doc {d} " + TEXT})  # fmt: skip
    return out


class FakeCLI:
    """Plays the grok CLI: answers drafting prompts with valid JSON citing the shown chunk ids;
    closed-book prompts with UNKNOWN (or an answer for questions containing 'capital')."""

    def __init__(self, bad_citations=False):
        self.calls, self.bad = [], bad_citations

    def __call__(self, cmd, **kw):
        prompt = Path(cmd[cmd.index("--prompt-file") + 1]).read_text()
        self.calls.append(cmd)
        ids = re.findall(r"\[chunk_id: ([^\]]+)\]", prompt)
        if "general knowledge only" in prompt:
            out = "Paris" if "capital" in prompt else "UNKNOWN"
        elif "do NOT answer" in prompt:
            out = json.dumps({"question": "What was 2031 revenue?", "why_unanswerable": "no 2031 data"})
        else:
            cites = ["hybrid:nope:0"] if self.bad else ids[:2]
            out = "Sure:\n" + json.dumps({"question": f"How did revenue change ({len(self.calls)})?",
                                          "answer": "+12%", "citations": cites})  # fmt: skip
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def cfg_in(td):
    p = Path(td) / "docsearch.yaml"
    p.write_text("paths: {data_dir: data, work_dir: work}\n")
    return load_config(p, env={})


class TestGrokCLI(unittest.TestCase):
    def test_headless_flags_from_help(self):
        cmd, stdin = cli_command(LLMSettings("cli", "grok", "grok-4", 5), Path("/t/p.txt"), Path("/t/o"))
        self.assertEqual(cmd[:3], ["grok", "--prompt-file", "/t/p.txt"])
        for f in ("--output-format", "--disable-web-search", "--no-subagents", "-m"):
            self.assertIn(f, cmd)
        self.assertFalse(stdin)

    def test_grok_is_text_only_others_take_images(self):
        g = LLMClient(LLMSettings("cli", "grok", "", 5), runner=FakeCLI(), which=have)
        self.assertFalse(g.supports_images())
        cmd, _ = cli_command(LLMSettings("cli", "claude", "sonnet", 5), Path("p"), Path("o"), ["/x/pages/a.png"])
        self.assertIn("--add-dir", cmd)
        self.assertEqual(cmd[cmd.index("--allowedTools") + 1], "Read")
        cmd, _ = cli_command(LLMSettings("cli", "openai", "", 5), Path("p"), Path("o"), ["/x/a.png"])
        self.assertEqual(cmd[cmd.index("-i") + 1], "/x/a.png")
        self.assertEqual(cmd[-1], "-")

    def test_default_backend_is_grok_cli(self):
        from docsearch.cli import QUESTION_DEFAULTS

        self.assertEqual(QUESTION_DEFAULTS, {"backend": "cli", "provider": "grok"})


class TestDraft(unittest.TestCase):
    def run_draft(self, td, runner, n=10, **kw):
        cfg = cfg_in(td)
        client = LLMClient(LLMSettings("cli", "grok", "", 5), runner=runner, which=have)
        out = Path(td) / "data" / "questions_custom.jsonl"
        with redirect_stdout(io.StringIO()):
            Q.draft(cfg, client, n, chunks=chunks(), images={}, out=out, **kw)
        return Q.read_rows(out)

    def test_types_gold_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            fake = FakeCLI()
            rows = self.run_draft(td, fake)
            by = {t: [r for r in rows if r["meta"]["type"] == t] for t in Q.TYPES}
            self.assertEqual({t: len(v) for t, v in by.items()}, {"multi_doc": 5, "no_answer": 2, "image": 3})
            for r in by["multi_doc"]:
                self.assertGreaterEqual(len({u.split("#")[0] for u in r["gold"]}), 2)  # 2+ docs, same source
                self.assertEqual(len({r["source"]}), 1)
            self.assertTrue(all(r["gold"] == {} for r in by["no_answer"]))
            self.assertTrue(all(r["format"] == "slide_image" for r in by["image"]))
            self.assertTrue(all(r["meta"]["status"] == "unreviewed" for r in rows))
            n_calls = len(fake.calls)
            again = self.run_draft(td, fake)  # resumable: nothing new to do
            self.assertEqual(len(again), 10)
            self.assertEqual(len(fake.calls), n_calls)

    def test_invalid_citations_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            rows = self.run_draft(td, FakeCLI(bad_citations=True), mix={"multi_doc": 1, "no_answer": 0, "image": 0})
            self.assertEqual(rows, [])

    def test_closed_book_filter(self):
        class Leaky(FakeCLI):
            def __call__(self, cmd, **kw):
                r = super().__call__(cmd, **kw)
                return SimpleNamespace(returncode=0, stdout=r.stdout.replace("How did revenue", "What capital"),
                                       stderr="")  # fmt: skip

        with tempfile.TemporaryDirectory() as td:
            mix = {"multi_doc": 1, "no_answer": 0, "image": 0}
            self.assertEqual(self.run_draft(td, Leaky(), n=3, mix=mix), [])
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(len(self.run_draft(td, Leaky(), n=3, mix=mix, closed_book=False)), 3)

    def test_mix_counts(self):
        self.assertEqual(sum(Q.plan_counts(200, Q.DEFAULT_MIX).values()), 200)
        self.assertEqual(Q.parse_mix("multi_doc=1,image=1"), {"multi_doc": 1.0, "image": 1.0})


class TestReview(unittest.TestCase):
    def test_keep_drop_edit_quit(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            path = Path(td) / "q.jsonl"
            rows = [{"qid": f"custom:image:{i}", "query": f"q{i}", "source": "s", "format": "pdf", "level": "page",
                     "gold": {"s/d0#p1": 1}, "meta": {"type": "image", "status": "unreviewed",
                                                      "citations": ["hybrid:ocb/d0:0"], "answer": "a"}}
                    for i in range(4)]  # fmt: skip
            Q.write_rows(path, rows)
            answers = iter(["k", "x", "d", "e", "better q?", "", "q"])
            out = []
            from rich.console import Console

            con = Console(file=io.StringIO(), width=100, color_system=None)
            Q.review(cfg, path, ask=lambda _: next(answers), say=out.append, chunks=chunks(), console=con)
            out.append(con.file.getvalue())
            got = Q.read_rows(path)
            self.assertEqual([r["meta"]["status"] for r in got], ["keep", "drop", "edited", "unreviewed"])
            self.assertEqual(got[2]["query"], "better q?")
            self.assertEqual(got[2]["meta"]["original_query"], "q2")
            self.assertTrue(any("ocb doc 0" in line for line in out))  # snippet shown
            self.assertEqual([r["qid"] for r in Q.load_for_eval(path)], ["custom:image:0", "custom:image:2",
                                                                         "custom:image:3"])  # fmt: skip
            self.assertEqual(len(Q.load_for_eval(path, reviewed_only=True)), 2)


class TestCustomScoring(unittest.TestCase):
    def test_all_gold_metric(self):
        self.assertEqual(score_query(["a", "x", "b"], {"a": 1, "b": 1})["all@10"], 1.0)
        self.assertEqual(score_query(["a", "x"], {"a": 1, "b": 1})["all@10"], 0.0)

    def test_abstention(self):
        pq = [{"gold": ["a"], "top_score": s} for s in (0.5, 0.6, 0.7, 0.8, 0.9)]
        pq += [{"gold": [], "top_score": 0.1}, {"gold": [], "top_score": 0.95}, {"gold": [], "top_score": None}]
        ab = abstention(pq, threshold=0.55)
        self.assertEqual((ab["n"], ab["abstain_rate"], ab["false_abstain_rate"]), (3, round(2 / 3, 4), 0.2))
        self.assertIn("p10", abstention(pq)["threshold_rule"])
        txt = format_custom("r", {"by_type": {"multi_doc": {"n": 2, "recall@10": 0.5, "all@10": 0.0,
                                                            "ndcg@10": 0.4}}, "abstention": ab})  # fmt: skip
        self.assertIn("abstain 0.667", txt)


if __name__ == "__main__":
    unittest.main()


class TestRealConfig(unittest.TestCase):
    """Uses the repo's real docsearch.yaml (catches made-up config keys); no LLM is called."""

    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / "docsearch.yaml", env={})

    def test_page_images_uses_render_output(self):
        from docsearch.visual import pages_dir

        imgs = Q.page_images(self.cfg)
        self.assertIsInstance(imgs, dict)
        if (pages_dir(self.cfg) / "pages.jsonl").exists():
            self.assertTrue(imgs and all("#p" in u for u in list(imgs)[:5]))

    def test_draft_setup_with_mocked_client(self):
        from docsearch.chunk import chunks_path

        real = chunks_path(self.cfg, "hybrid").exists()
        client = LLMClient(LLMSettings("cli", "grok", "", 5), runner=FakeCLI(), which=have)
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            out = Path(td) / "q.jsonl"
            Q.draft(self.cfg, client, 1, {"multi_doc": 1}, out, chunks=None if real else chunks())
            self.assertEqual(len(Q.read_rows(out)), 1)


class TestDraftProgress(unittest.TestCase):
    def setUp(self):
        from docsearch import ui

        self.ui = ui
        self._tty = ui._state["tty"]
        ui._state["tty"] = False  # plain lines

    def tearDown(self):
        self.ui._state["tty"] = self._tty

    def _client(self, runner):
        return LLMClient(LLMSettings("cli", "grok", "", 5), runner=runner, which=have)

    def test_plain_progress_shows_counts_drops_eta(self):
        class Mixed(FakeCLI):
            def __call__(self, cmd, **kw):
                r = super().__call__(cmd, **kw)
                if len(self.calls) == 1:
                    return SimpleNamespace(returncode=0, stdout="no json here", stderr="warning: slow")
                return r

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf), self.assertLogs("docsearch", level="INFO") as logs:
                Q.draft(cfg_in(td), self._client(Mixed()), 4, out=Path(td) / "q.jsonl", chunks=chunks(), images={})
            text = buf.getvalue()
            self.assertIn("questions: 0/4 kept", text)
            self.assertIn("multi-doc 0/2", text)
            self.assertIn("dropped (parse error)", text)
            self.assertRegex(text, r"kept multi-doc 1/2: .*\[\d/4, ETA ")
            joined = "\n".join(logs.output)
            self.assertIn("grok stderr: warning: slow", joined)
            self.assertIn("unparseable LLM reply", joined)
            self.assertRegex(joined, r"call multi-doc draft done in \d")

    def test_ctrl_c_saves_and_resumes(self):
        class Interrupt(FakeCLI):
            def __call__(self, cmd, **kw):
                if len(self.calls) == 4:  # mid-run, after two kept questions
                    raise KeyboardInterrupt
                return super().__call__(cmd, **kw)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "q.jsonl"
            with redirect_stdout(io.StringIO()) as buf, self.assertRaises(KeyboardInterrupt):
                Q.draft(cfg_in(td), self._client(Interrupt()), 6, out=out, chunks=chunks(), images={})
            self.assertIn("re-run the same command to resume", buf.getvalue())
            first = Q.read_rows(out)
            self.assertEqual(len(first), 2)
            with out.open("a") as f:
                f.write('{"qid": "torn')  # torn last line is tolerated
            with redirect_stdout(io.StringIO()):
                Q.draft(cfg_in(td), self._client(FakeCLI()), 6, out=out, chunks=chunks(), images={})
            rows = Q.read_rows(out)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({r["qid"] for r in rows}), 6)
            self.assertEqual(rows[:2], first)

    def test_call_errors_retry_next_run(self):
        from docsearch.llm import LLMError

        class Flaky(FakeCLI):
            def __call__(self, cmd, **kw):
                if len(self.calls) < 1:
                    self.calls.append(cmd)
                    return SimpleNamespace(returncode=1, stdout="", stderr="upstream timeout")
                return super().__call__(cmd, **kw)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "q.jsonl"
            mix = {"multi_doc": 1, "no_answer": 0, "image": 0}
            with redirect_stdout(io.StringIO()) as buf:
                Q.draft(cfg_in(td), self._client(Flaky()), 2, mix, out, chunks=chunks(), images={})
            self.assertIn("call error", buf.getvalue())
            fp = out.with_suffix(".failed.json")
            self.assertNotIn("multi_doc:0", fp.read_text() if fp.exists() else "")  # errored slot is retried later
            with redirect_stdout(io.StringIO()), self.assertRaises(LLMError):
                Q.draft(cfg_in(td), self._client(lambda c, **k: SimpleNamespace(returncode=1, stdout="", stderr="x")),
                        5, mix, Path(td) / "q2.jsonl", chunks=chunks(), images={}, max_errors=2)  # fmt: skip


class TestStitchedAndTimeouts(unittest.TestCase):
    STITCHED = ("By 2021, was the rate back above 2019, and was the same true for medium-skilled people? In 2023, "
                "how many points above was each group? Among domestic workers, what share in Portugal were born "
                "outside the EU, and how does Portugal compare with Poland?")  # fmt: skip

    def test_stitched_rules(self):
        self.assertEqual(Q.stitched("How did Croatia's employment growth in 2023 compare with Slovenia's?"), [])
        why = Q.stitched(self.STITCHED)
        self.assertTrue(any("question marks" in w for w in why))
        self.assertTrue(Q.stitched("Did A rise and B fall and C hold and D move?"))  # 3 'and'

    def test_lint_counts(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "q.jsonl"
            mk = lambda i, q, st="unreviewed": {"qid": f"custom:multi_doc:{i}", "query": q, "gold": {},  # noqa: E731
                                                "meta": {"type": "multi_doc", "status": st}}  # fmt: skip
            Q.write_rows(p, [mk(0, self.STITCHED), mk(1, "How did X compare with Y?"),
                             mk(2, "What was A? And B?"), mk(3, self.STITCHED, "drop")])  # fmt: skip
            out = []
            self.assertEqual(Q.lint(p, say=out.append), {"multi_doc": 1, "multi_doc_warn": 1})
            self.assertIn("custom:multi_doc:0", "\n".join(out))

    def test_draft_drops_stitched(self):
        class Glue(FakeCLI):
            def __call__(self, cmd, **kw):
                r = super().__call__(cmd, **kw)
                return SimpleNamespace(returncode=0, stdout=r.stdout.replace(
                    "How did revenue change", "A? B? C? D and E and F"), stderr="")  # fmt: skip

        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            out = Path(td) / "q.jsonl"
            Q.draft(cfg_in(td), LLMClient(LLMSettings("cli", "grok", "", 5), runner=Glue(), which=have), 2,
                    {"multi_doc": 1}, out, chunks=chunks(), images={})  # fmt: skip
            self.assertEqual(Q.read_rows(out), [])

    def test_timeout_skips_sample_without_stopping(self):
        import subprocess

        class Slow(FakeCLI):
            def __call__(self, cmd, **kw):
                if len(self.calls) < 6:  # more timeouts than max_errors
                    self.calls.append(cmd)
                    raise subprocess.TimeoutExpired(cmd, 5)
                return super().__call__(cmd, **kw)

        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
            out = Path(td) / "q.jsonl"
            Q.draft(cfg_in(td), LLMClient(LLMSettings("cli", "grok", "", 5), runner=Slow(), which=have), 2,
                    {"image": 1}, out, chunks=chunks(), images={}, max_errors=2)  # fmt: skip
            self.assertEqual(len(Q.read_rows(out)), 2)
            self.assertIn("timed out, sample skipped", buf.getvalue())
            self.assertIn("image:0", out.with_suffix(".failed.json").read_text())

    def test_grok_timeout_override(self):
        from docsearch.llm import settings

        cfg = load_config(Path(__file__).resolve().parents[1] / "docsearch.yaml", env={})
        self.assertEqual(settings(cfg, "questions", backend="cli", provider="grok").timeout_s, 300)
        self.assertEqual(settings(cfg, "questions", backend="cli", provider="claude").timeout_s, 180)


class TestRefillDropped(unittest.TestCase):
    def test_dropped_in_review_are_refilled(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            out = Path(td) / "q.jsonl"
            mk = lambda: LLMClient(LLMSettings("cli", "grok", "", 5), runner=FakeCLI(), which=have)  # noqa: E731
            Q.draft(cfg_in(td), mk(), 2, {"multi_doc": 1}, out, chunks=chunks(), images={})
            rows = Q.read_rows(out)
            rows[0]["meta"]["status"] = "drop"
            Q.write_rows(out, rows)
            Q.draft(cfg_in(td), mk(), 2, {"multi_doc": 1}, out, chunks=chunks(), images={})
            self.assertEqual(len(Q.load_for_eval(out)), 2)


class TestReviewPanel(unittest.TestCase):
    def render(self, row, width=100):
        from rich.console import Console

        con = Console(file=io.StringIO(), width=width, color_system=None)
        con.print(Q.render_question(row, 3, 80, {c["chunk_id"]: c for c in chunks()}))
        return con.file.getvalue()

    def row(self, q="How did revenue compare?", **meta):
        return {"qid": "custom:multi_doc:0", "query": q, "format": "mixed", "gold": {"ocb/d0#p1": 1},
                "meta": {"type": "multi_doc", "status": "unreviewed", "answer": "It grew 12% to $4.2B",
                         "citations": ["hybrid:ocb/d0:0", "hybrid:finslides/d1:0"], **meta}}  # fmt: skip

    def test_panel_layout(self):
        out = self.render(self.row())
        self.assertIn("3/80", out)
        self.assertIn("multi-doc", out)
        self.assertIn("unreviewed", out)
        self.assertIn("How did revenue compare?", out)
        self.assertIn("Answer: It grew 12% to $4.2B", out)
        self.assertIn("ocb · d0 · p.1", out)  # short doc name + page, not the chunk id
        self.assertNotIn("hybrid:ocb/d0:0", out)
        self.assertIn("[k]eep  [d]rop  [e]dit  [s]kip  [q]uit", out)
        self.assertNotIn("stitched", out)

    def test_lint_badges(self):
        self.assertIn("stitched", self.render(self.row("A? B? C? and x and y and z")))
        self.assertIn("borderline", self.render(self.row("What was A? What was B?")))

    def test_snippet_highlights_answer_numbers(self):
        from rich.text import Text

        snip, nums = Q.snippet("File: x.pdf " + "intro words " * 30 + "revenue rose 12% to 4.2B in 2019", "12% and 4.2")
        self.assertIn("12%", snip)
        self.assertLessEqual(len(snip), 202)
        self.assertTrue(snip.startswith("…"))
        t = Text(snip)
        self.assertGreaterEqual(t.highlight_words(nums, style="bold"), 2)

    def test_short_doc(self):
        self.assertEqual(Q.short_doc("vidore_v3_hr/joint_employment_report_2025-KE012501"),
                         "vidore · joint employment report 2025")  # fmt: skip
        self.assertEqual(Q.short_doc("ocb/716b3a07-16b6-1bda-f4f7-be2d62e03ce6.pptx"), "ocb · 716b3a07.pptx")


class TestLintDrop(unittest.TestCase):
    def test_drop_and_undo(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "questions_custom.jsonl"
            rows = [{"qid": f"custom:multi_doc:{i}", "query": q, "gold": {},
                     "meta": {"type": "multi_doc", "status": st}}
                    for i, (q, st) in enumerate([("A? B? C?", "unreviewed"), ("Fine question?", "unreviewed"),
                                                 ("X? Y? Z?", "keep"), ("P? Q? R?", "drop")])]  # fmt: skip
            Q.write_rows(p, rows)
            out = []
            self.assertEqual(Q.lint_drop(p, say=out.append), 2)
            self.assertIn("dropped 2", out[0])
            got = Q.read_rows(p)
            self.assertEqual([r["meta"]["status"] for r in got], ["dropped", "unreviewed", "dropped", "drop"])
            self.assertEqual(got[0]["meta"]["reason"], "lint")
            self.assertEqual(len(list(Path(td).glob("questions_custom.bak-*.jsonl"))), 1)
            self.assertEqual(len(Q.load_for_eval(p)), 1)
            self.assertEqual(Q.lint_drop(p, say=out.append), 0)  # idempotent
            self.assertEqual(Q.lint_drop(p, undo=True, say=out.append), 2)
            back = Q.read_rows(p)
            self.assertEqual([r["meta"]["status"] for r in back], ["unreviewed", "unreviewed", "keep", "drop"])
            self.assertTrue(all("reason" not in r["meta"] and "lint_prev" not in r["meta"] for r in back))


class TestSuggestions(unittest.TestCase):
    def setup_files(self, td):
        p, sp = Path(td) / "questions_custom.jsonl", Path(td) / "questions_custom.suggestions.jsonl"
        mk = lambda i, t, st="unreviewed": {"qid": f"custom:{t}:{i}", "query": f"q{i}?", "source": "s",  # noqa: E731
                                            "format": "pdf", "level": "page", "gold": {"s/d0#p1": 1} if t != "no_answer" else {},
                                            "meta": {"type": t, "status": st, "answer": "a",
                                                     "citations": ["hybrid:ocb/d0:0"]}}  # fmt: skip
        Q.write_rows(p, [mk(0, "image"), mk(1, "image"), mk(2, "multi_doc"), mk(3, "no_answer", "keep"),
                         mk(4, "multi_doc", "dropped")])  # fmt: skip
        sugg = [{"id": "custom:image:0", "suggestion": "keep", "reason": "grounded"},
                {"id": "custom:image:1", "suggestion": "edit", "reason": "add scope", "proposed_question": "Q1 in EU?",
                 "proposed_answer": "b"},
                {"id": "custom:multi_doc:2", "suggestion": "drop", "reason": "contrived"},
                {"id": "custom:no_answer:3", "suggestion": "drop", "reason": "answerable"},
                {"id": "custom:multi_doc:4", "suggestion": "keep", "reason": "x"}]  # fmt: skip
        Q.write_rows(sp, sugg)
        return p, sp

    def test_apply_backup_provenance_undo(self):
        with tempfile.TemporaryDirectory() as td:
            p, sp = self.setup_files(td)
            before = p.read_text()
            out = []
            c = Q.apply_suggestions(p, sp, say=out.append)
            self.assertEqual(c, {"keep": 1, "edit": 1, "drop": 1, "skipped (already reviewed)": 1})
            self.assertEqual(len(list(Path(td).glob("questions_custom.bak-*.jsonl"))), 1)
            rows = {r["qid"]: r for r in Q.read_rows(p)}
            self.assertEqual(rows["custom:image:0"]["meta"]["status"], "keep")
            self.assertEqual(rows["custom:image:1"]["query"], "Q1 in EU?")
            self.assertEqual(rows["custom:image:1"]["meta"]["answer"], "b")
            self.assertEqual(rows["custom:image:1"]["meta"]["status"], "edited")
            self.assertEqual(rows["custom:multi_doc:2"]["meta"]["status"], "drop")
            self.assertEqual(rows["custom:no_answer:3"]["meta"]["status"], "keep")  # human decision untouched
            self.assertEqual(rows["custom:multi_doc:4"]["meta"]["status"], "dropped")  # lint-dropped untouched
            for q in ("custom:image:0", "custom:image:1", "custom:multi_doc:2"):
                self.assertEqual(rows[q]["meta"]["reviewer"], "reviewer")
                self.assertTrue(rows[q]["meta"]["review_reason"])
            ev = Q.load_for_eval(p, reviewed_only=True)
            self.assertEqual(sorted(r["qid"] for r in ev), ["custom:image:0", "custom:image:1", "custom:no_answer:3"])
            self.assertEqual(Q.provenance(ev), {"reviewer": 2, "human": 1})
            Q.apply_suggestions(p, sp, undo=True, say=out.append)
            self.assertEqual(Q.read_rows(p), [json.loads(line) for line in before.splitlines()])

    def test_only_filter(self):
        with tempfile.TemporaryDirectory() as td:
            p, sp = self.setup_files(td)
            self.assertEqual(Q.apply_suggestions(p, sp, {"drop"}, say=lambda _: None)["drop"], 1)
            st = {r["qid"]: r["meta"]["status"] for r in Q.read_rows(p)}
            self.assertEqual(st["custom:image:0"], "unreviewed")

    def test_review_enter_accepts_or_override(self):
        from rich.console import Console

        with tempfile.TemporaryDirectory() as td:
            p, sp = self.setup_files(td)
            con = Console(file=io.StringIO(), width=110, color_system=None)
            keys = iter(["", "k", "", "q"])  # accept keep, override edit with keep, accept drop
            Q.review(cfg_in(td), p, ask=lambda _: next(keys), say=lambda _: None, chunks=chunks(), console=con,
                     suggestions=sp, reviewer="reviewer1")  # fmt: skip
            shown = con.file.getvalue()
            self.assertIn("suggested: keep", shown)
            self.assertIn("grounded", shown)
            self.assertIn("proposed Q: Q1 in EU?", shown)
            self.assertIn("[Enter] accept suggestion", shown)
            rows = {r["qid"]: r["meta"] for r in Q.read_rows(p)}
            self.assertEqual(
                (rows["custom:image:0"]["status"], rows["custom:image:0"]["reviewer"]), ("keep", "reviewer1")
            )
            self.assertTrue(rows["custom:image:0"]["accepted_suggestion"])
            self.assertEqual(rows["custom:image:1"]["status"], "keep")
            self.assertEqual(rows["custom:image:1"]["overrode_suggestion"], "edit")
            self.assertEqual(rows["custom:multi_doc:2"]["status"], "drop")
