import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from docsearch import questions as Q
from docsearch.config import load_config
from docsearch.judge import Judge, judge_abstention
from docsearch.llm import LLMClient, LLMSettings


def have(_):
    return "/usr/bin/x"


def cfg_in(td):
    p = Path(td) / "docsearch.yaml"
    p.write_text("paths: {data_dir: data, cache_dir: cache}\n")
    return load_config(p, env={})


def runner(reply):
    calls = []

    def run(cmd, **kw):
        prompt = Path(cmd[cmd.index("--prompt-file") + 1]).read_text()
        calls.append(prompt)
        out = reply(prompt, len(calls))
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    run.calls = calls
    return run


HITS = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "pages": [1], "text": f"text {i}"} for i in range(7)]


class TestJudge(unittest.TestCase):
    def test_verdict_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            r = runner(lambda p, n: '{"verdict": "ANSWERABLE", "chunk_ids": ["c1", "zzz"]}')
            j = Judge(cfg, LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have))
            out = j("What is X?", HITS)
            self.assertEqual((out["answerable"], out["chunk_ids"], out["cached"]), (True, ["c1"], False))
            prompt = r.calls[0]
            self.assertIn("[chunk_id: c4]", prompt)
            self.assertNotIn("[chunk_id: c5]", prompt)  # top 5 only
            self.assertTrue(j("What is X?", HITS)["cached"])
            j2 = Judge(cfg, LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have))  # new run: cache
            self.assertTrue(j2("What is X?", HITS)["cached"])
            self.assertEqual(len(r.calls), 1)
            self.assertTrue((Path(td) / "cache" / "judge" / "grok.jsonl").exists())

    def test_bad_reply_and_error_not_cached(self):
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            r = runner(lambda p, n: "hmm" if n == 1 else subprocess.TimeoutExpired("grok", 5))
            j = Judge(cfg_in(td), LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have))
            self.assertIsNone(j("Q?", HITS)["answerable"])
            self.assertIsNone(j("Q?", HITS)["answerable"])
            self.assertEqual(j.errors, 1)
            self.assertEqual(j.cache, {})

    def test_judge_abstention_rates(self):
        pq = [{"gold": [], "judge": False}, {"gold": [], "judge": True}, {"gold": [], "judge": None},
              {"gold": ["a#p1"], "judge": True, "judge_units": ["a#p1"]},
              {"gold": ["b#p1"], "judge": True, "judge_units": ["x#p1"]},
              {"gold": ["c#p1"], "judge": False}]  # fmt: skip
        ab = judge_abstention(pq)
        self.assertEqual((ab["n_no_answer"], ab["abstain_rate"], ab["failed_calls"]), (2, 0.5, 1))
        self.assertEqual((ab["false_abstain_rate"], ab["answerable_cites_gold"]), (round(1 / 3, 4), 0.5))

    def test_side_by_side_report(self):
        from docsearch.eval import format_custom

        txt = format_custom("r", {"by_type": {}, "abstention": {"n": 3, "threshold": 0.8, "threshold_rule": "p10",
                                  "abstain_rate": 0.4, "false_abstain_rate": 0.1},
                                  "abstention_judge": {"model": "grok:default", "abstain_rate": 0.7,
                                                       "false_abstain_rate": 0.05, "failed_calls": 0,
                                                       "answerable_cites_gold": 0.9}})  # fmt: skip
        self.assertIn("threshold < 0.8", txt)
        self.assertIn("judge grok:default", txt)
        self.assertRegex(txt, r"0\.700\s+0\.050")


class TestParaphrase(unittest.TestCase):
    SRC = "Fossil fuels accounted for 80.6% of total world energy consumption in 2010 and 78.4% in 2013."

    def test_overlap_metric(self):
        copied = Q.ngram_overlap(
            "What share of total world energy consumption did fossil fuels have in 2010?", [self.SRC]
        )
        self.assertGreaterEqual(copied["longest"], 4)
        fresh = Q.ngram_overlap("How much of global energy use came from coal, oil and gas in 2010?", [self.SRC])
        self.assertLess(fresh["longest"], 3)
        self.assertLess(fresh["share3"], copied["share3"])

    def setup(self, td):
        cfg = cfg_in(td)
        (Path(td) / "data").mkdir()
        src = Path(td) / "data" / "questions_custom.jsonl"
        mk = lambda i, st: {"qid": f"custom:multi_doc:{i}", "query": "What share of total world energy consumption "  # noqa: E731
                            "did fossil fuels have in 2010?", "source": "s", "format": "png", "level": "page",
                            "gold": {"w#p1": 1}, "meta": {"type": "multi_doc", "status": st, "answer": "80.6%",
                                                         "citations": ["c0"]}}  # fmt: skip
        Q.write_rows(src, [mk(0, "keep"), mk(1, "drop"), mk(2, "unreviewed"), mk(3, "edited")])
        chunks = [{"chunk_id": "c0", "doc_id": "w", "text": self.SRC}]
        return cfg, src, chunks

    def test_rewrite_retry_and_resume(self):
        def reply(p, n):
            if n == 1:  # too close: copies "total world energy consumption"
                return '{"question": "What share of total world energy consumption came from fossil fuels in 2010?"}'
            return '{"question": "How much of global energy use came from coal, oil and gas in 2010?"}'

        with tempfile.TemporaryDirectory() as td:
            cfg, src, chunks = self.setup(td)
            before = src.read_text()
            r = runner(reply)
            client = LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have)
            with redirect_stdout(io.StringIO()) as buf:
                out = Q.paraphrase(cfg, client, chunks=chunks)
            self.assertEqual(src.read_text(), before)  # original untouched
            rows = Q.read_rows(out)
            self.assertEqual([x["qid"] for x in rows], ["custom:multi_doc:0", "custom:multi_doc:3"])  # reviewed only
            x = rows[0]
            self.assertEqual((x["gold"], x["meta"]["answer"]), ({"w#p1": 1}, "80.6%"))
            self.assertIn("coal, oil and gas", x["query"])
            self.assertGreater(x["meta"]["overlap_before"]["longest"], x["meta"]["overlap_after"]["longest"])
            self.assertIn("previous rewrite copied", r.calls[1])
            self.assertIn("too close, retried", buf.getvalue())
            n = len(r.calls)
            with redirect_stdout(io.StringIO()):
                Q.paraphrase(cfg, client, chunks=chunks)
            self.assertEqual(len(r.calls), n)  # resumable

    def test_skips_when_always_too_close(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, src, chunks = self.setup(td)
            r = runner(
                lambda p, n: '{"question": "What share of total world energy consumption did fossil fuels have?"}'
            )
            with redirect_stdout(io.StringIO()) as buf:
                out = Q.paraphrase(cfg, LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have),
                                   attempts=2, chunks=chunks)  # fmt: skip
            self.assertEqual(Q.read_rows(out), [])
            self.assertIn("'skipped': 2", buf.getvalue())

    def test_eval_loads_para_set(self):
        from docsearch.eval import load_custom

        with tempfile.TemporaryDirectory() as td:
            cfg, src, chunks = self.setup(td)
            row = json.loads(src.read_text().splitlines()[0])
            row["query"] = "rewritten?"
            Q.write_rows(Q.para_path(cfg), [row])
            qs = load_custom(cfg, True, "custom_para")
            self.assertEqual([(q.qid, q.query) for q in qs], [("custom:multi_doc:0", "rewritten?")])
            self.assertEqual(len(load_custom(cfg, True, "custom")), 2)


if __name__ == "__main__":
    unittest.main()


class TestGated(unittest.TestCase):
    PQ = [
        {"gold": [], "top_score": s, "judge": j} for s, j in [(0.95, False), (0.7, False), (0.6, False), (None, False)]
    ] + [
        {"gold": ["g"], "top_score": s, "judge": j} for s, j in [(0.99, False), (0.97, True), (0.8, True), (0.5, False)]
    ]

    def test_gated_rates(self):
        from docsearch.judge import gated

        g = gated(self.PQ, 0.9)  # >=0.9 answered without the judge
        self.assertEqual((g["abstain_rate"], g["false_abstain_rate"], g["judge_calls"]), (0.75, 0.25, 5))
        j = gated(self.PQ, float("inf"))  # judge only
        self.assertEqual((j["abstain_rate"], j["false_abstain_rate"]), (1.0, 0.5))

    def test_missing_verdicts_reported_not_guessed(self):
        from docsearch.judge import gated

        pq = self.PQ + [{"gold": [], "top_score": 0.1}]
        g = gated(pq, 0.9)
        self.assertEqual(g["missing_verdicts"], 1)
        self.assertEqual(g["n_no_answer"], 4)

    def test_sweep_best_respects_ceiling(self):
        from docsearch.judge import sweep

        rows, best = sweep(self.PQ, [0.5, 0.9, 0.96, float("inf")], max_false=0.25)
        self.assertEqual((best["gate"], best["abstain_rate"]), (0.96, 1.0))
        self.assertEqual(len(rows), 4)
        _, none = sweep(self.PQ, [float("inf")], max_false=0.1)
        self.assertIsNone(none)

    def test_cache_only_never_calls(self):
        with tempfile.TemporaryDirectory() as td:
            r = runner(lambda p, n: '{"verdict": "NOT", "chunk_ids": []}')
            j = Judge(cfg_in(td), LLMClient(LLMSettings("cli", "grok", "", 5), runner=r, which=have), cache_only=True)
            out = j("Q?", HITS)
            self.assertTrue(out["missing"])
            self.assertEqual((r.calls, j.missing), ([], 1))

    def test_sweep_script_offline(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "results" / "runs"
            runs.mkdir(parents=True)
            with (runs / "20261011-000000-x_judge.jsonl").open("w") as f:
                for r in self.PQ:
                    f.write(json.dumps({"qid": str(id(r)), **r}) + "\n")
            script = Path(__file__).resolve().parents[1] / "scripts" / "abstain_sweep.py"
            out = subprocess.run([sys.executable, str(script), "x_judge", "--max-false", "0.25"], cwd=td,
                                 capture_output=True, text=True, check=True).stdout  # fmt: skip
            self.assertIn("judge only", out)
            self.assertIn("<- best", out)
            self.assertIn("best: gate", out)
