import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from docsearch import questions as Q
from docsearch.config import load_config
from docsearch.hard import HARD_TYPES, HardDrafter, hard_path, jaccard, missed_pages
from docsearch.llm import LLMClient, LLMSettings
from docsearch.manifest import Question, write_questions

SLIDE = ("IBM quarterly results cloud software consulting infrastructure revenue segment margin growth outlook "
         "hybrid platform client count signings backlog quarter {q} revenue {v} billion")  # fmt: skip
TABLE = "Sheet Budget | region | year | amount " + " ".join(f"| row{i} | {2000 + i} | {i * 37}" for i in range(20))


def hard_chunks():
    out = [{"chunk_id": f"hybrid:finslides/ibm:{p}", "doc_id": "finslides/ibm", "format": "slide_image",
            "source": "finslides", "pages": [p], "text": SLIDE.format(q=p, v=10 + p) * 2} for p in (1, 2, 3)]  # fmt: skip
    out += [{"chunk_id": f"hybrid:ocb/b{d}.xlsx:0", "doc_id": f"ocb/b{d}.xlsx", "format": "xlsx", "source": "ocb",
             "pages": [1], "text": TABLE + f" doc{d}"} for d in range(2)]  # fmt: skip
    out += [{"chunk_id": "hybrid:ocb/memo.docx:0", "doc_id": "ocb/memo.docx", "format": "docx", "source": "ocb",
             "pages": [2], "text": "Memo on budget amount per region and year for the regional offices " * 4}]  # fmt: skip
    return out


class HardFake:
    """Plays grok: cites the TARGET only (or both docs for pairs), writes the question in its own words."""

    def __init__(self, copy=False, cite_distractor=False):
        self.calls, self.copy, self.cite_distractor = [], copy, cite_distractor

    def __call__(self, cmd, **kw):
        prompt = Path(cmd[cmd.index("--prompt-file") + 1]).read_text()
        self.calls.append(prompt)
        ids = re.findall(r"\[chunk_id: ([^\]]+)\]", prompt)
        if "general knowledge only" in prompt:
            out = "UNKNOWN"
        else:
            q = ("IBM quarterly results cloud software consulting infrastructure revenue?" if self.copy
                 else f"Which figure did that deck report for its period {len(self.calls)}?")  # fmt: skip
            cites = ids[:2] if "EACH document" in prompt else ids[:1] + (ids[1:2] if self.cite_distractor else [])
            out = json.dumps({"question": q, "answer": "x", "citations": cites})
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def cfg_in(td):
    p = Path(td) / "docsearch.yaml"
    p.write_text("paths: {data_dir: data, work_dir: work, results_dir: results}\n")
    return load_config(p, env={})


def write_missed(td):
    (Path(td) / "data").mkdir(exist_ok=True)
    write_questions(Path(td) / "data" / "questions.jsonl", [
        Question("q1", "?", "ocb", "docx", "page", {"ocb/memo.docx#p2": 1}),
        Question("q2", "?", "ocb", "xlsx", "page", {"ocb/b0.xlsx#p1": 1})])  # fmt: skip
    runs = Path(td) / "results" / "runs"
    runs.mkdir(parents=True)
    with (runs / "20261009-000000-m3_skip_slide_rerank.jsonl").open("w") as f:
        f.write(
            json.dumps({"qid": "q1", "top": ["ocb/b0.xlsx#p1", "ocb/b1.xlsx#p1", "x#p1", "ocb/memo.docx#p2"]}) + "\n"
        )
        f.write(json.dumps({"qid": "q2", "top": ["ocb/b0.xlsx#p1"]}) + "\n")  # rank 1: not hard


class TestHard(unittest.TestCase):
    def client(self, fake):
        return LLMClient(LLMSettings("cli", "grok", "", 5), runner=fake, which=lambda _: "/usr/bin/grok")

    def test_missed_pages_from_per_question_results(self):
        with tempfile.TemporaryDirectory() as td:
            write_missed(td)
            m = missed_pages(cfg_in(td))
            self.assertEqual(m, [{"qid": "q1", "gold": ["ocb/memo.docx#p2"],
                                  "wrong": ["ocb/b0.xlsx#p1", "ocb/b1.xlsx#p1", "x#p1"]}])  # fmt: skip

    def test_draft_hard_all_types_gold_is_the_target_page(self):
        with tempfile.TemporaryDirectory() as td:
            write_missed(td)
            cfg = cfg_in(td)
            fake = HardFake()
            Q.draft(cfg, self.client(fake), 8, chunks=hard_chunks(), images={}, hard=True, say=lambda s: None)
            rows = Q.read_rows(hard_path(cfg))
            kinds = {r["meta"]["type"] for r in rows}
            self.assertEqual(kinds, set(HARD_TYPES))
            for r in rows:
                self.assertTrue(r["qid"].startswith("hard:"))
                if r["meta"]["type"] == "hard_multi":
                    self.assertEqual(len(r["gold"]), 2)
                else:
                    self.assertEqual(len(r["gold"]), 1)  # exactly the target page
                    self.assertTrue(r["meta"]["distractors"])
            nd = next(r for r in rows if r["meta"]["type"] == "hard_neardup")
            self.assertIn("finslides/ibm#p", next(iter(nd["gold"])))
            ms = next(r for r in rows if r["meta"]["type"] == "hard_missed")
            self.assertEqual(ms["gold"], {"ocb/memo.docx#p2": 1})
            self.assertEqual(ms["meta"]["from_qid"], "q1")
            self.assertFalse((Path(td) / "data" / "questions_custom.jsonl").exists())
            n = len(fake.calls)
            Q.draft(cfg, self.client(fake), 8, chunks=hard_chunks(), images={}, hard=True, say=lambda s: None)
            self.assertEqual(len(fake.calls), n)  # resumable

    def test_copied_wording_and_distractor_citations_dropped(self):
        for fake in (HardFake(copy=True), HardFake(cite_distractor=True)):
            with tempfile.TemporaryDirectory() as td:
                cfg = cfg_in(td)
                d = HardDrafter(cfg, self.client(fake), hard_chunks(), {}, 11, True)
                self.assertIsNone(d.neardup(0))
                self.assertTrue(d.dropped["copied"] or d.dropped["citations"])

    def test_eval_loads_hard_set(self):
        from docsearch.eval import load_custom

        with tempfile.TemporaryDirectory() as td:
            cfg = cfg_in(td)
            Q.draft(cfg, self.client(HardFake()), 4, {"hard_table": 1}, chunks=hard_chunks(), images={}, hard=True,
                    say=lambda s: None)  # fmt: skip
            qs = load_custom(cfg, False, "hard")
            self.assertTrue(qs)
            self.assertTrue(all(q.meta["type"] == "hard_table" for q in qs))

    def test_cli_flags(self):
        from docsearch.cli import main

        with self.assertRaises(SystemExit) as e:
            main(["questions", "draft", "--hard", "--help"])
        self.assertEqual(e.exception.code, 0)
        self.assertEqual(jaccard({"a"}, {"a", "b"}), 0.5)
