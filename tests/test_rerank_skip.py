import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from docsearch.config import load_config
from docsearch.eval import RUN_FIELDS
from docsearch.search import SearchConfig, rerank


class FakeReranker:
    def __init__(self):
        self.seen = []

    def score(self, query, passages):
        self.seen += passages
        return [float(len(p)) for p in passages]  # longer text scores higher


def c(i, fmt, text):
    return {"chunk_id": str(i), "format": fmt, "text": text}


class TestRerankSkipFormats(unittest.TestCase):
    def test_skipped_formats_keep_fused_slot_and_are_not_scored(self):
        cands = [c(0, "pdf", "a"), c(1, "slide_image", "a long slide text here"), c(2, "pdf", "much longer pdf")]
        rr = FakeReranker()
        out = rerank("q", cands, rr, n=3, skip_formats=("slide_image",))
        self.assertEqual([x["chunk_id"] for x in out], ["2", "1", "0"])  # slide stays at position 1
        self.assertNotIn("a long slide text here", rr.seen)

    def test_default_empty_is_plain_rerank(self):
        cands = [c(0, "pdf", "a"), c(1, "slide_image", "a long slide text here"), c(2, "pdf", "bb")]
        out = rerank("q", cands, FakeReranker(), n=3)
        self.assertEqual([x["chunk_id"] for x in out], ["1", "2", "0"])

    def test_config_default_is_empty(self):
        with tempfile.TemporaryDirectory() as t:
            sc = SearchConfig.from_cfg(load_config(Path(t) / "none.yaml", env={}))
            self.assertEqual(sc.rerank_skip_formats, ())


class TestOneChangePlan(unittest.TestCase):
    """Reference reuse: any cached format-aware rerank run with the current default settings, any name."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        y = Path(self.td.name) / "docsearch.yaml"
        y.write_text(
            "search: {rerank_candidates: 30, rerank_min_chars: 40, rerank_skip_formats: [slide_image]}\n"
            "models: {rerank_max_length: 384, reranker: org/gte, embedders: {v: {id: org/vis, kind: visual}}}\n"
        )
        self.cfg = load_config(y, env={})
        from docsearch.eval import run_settings

        self.blob = run_settings(self.cfg)["blob"]  # what a default run records in its notes
        self.M3 = "eval --full; " + json.dumps(self.blob)

    def tearDown(self):
        self.td.cleanup()

    def args(self, **kw):
        base = dict(visual="v", fusion="format_aware", name="m4", variant="hybrid", full=True,
                    rerank_skip_formats=None, visual_blind_formats=["pptx,csv"])  # fmt: skip
        return SimpleNamespace(**{**base, **kw})

    def write_ref(self, notes, n=30, name="m3_skip_slide_rerank", **kw):
        rd = self.cfg.path("results_dir")
        (rd / "runs").mkdir(parents=True, exist_ok=True)
        row = {"run_id": "r1", "name": name, "chunking": "hybrid", "retrieval": "hybrid_visual_fa", "rerank": "True",
               "embedder": "org/emb + org/vis", "reranker": "org/gte", "ndcg@10": 0.709, "n_queries": 734,
               "notes": notes, "rerank_candidates": n, "rerank_max_length": 384, **kw}  # fmt: skip
        with (rd / "runs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
            w.writeheader()
            w.writerow(row)
        (rd / "runs" / "r1.jsonl").write_text("{}\n")

    def test_reuses_cached_m3_and_runs_only_m4(self):
        from docsearch.cli import _one_change_plan

        self.write_ref("eval --full; " + json.dumps({**self.blob, "page_pairs_scored": 5}))  # a count, not a setting
        plan, ref, over = _one_change_plan(self.cfg, self.args())
        self.assertEqual(plan, [("m4", "hybrid", "hybrid_visual_fa", True)])
        self.assertIsNone(ref)
        self.assertEqual(over, {("visual", "blind_formats"): ["pptx", "csv"]})

    def test_reference_rerun_when_settings_or_scope_differ(self):
        from docsearch.cli import _one_change_plan

        no_skip = {k: v for k, v in self.blob.items() if k != "rerank_skip_formats"}
        pre_m28 = {k: v for k, v in self.blob.items() if k != "sheet_cap"}
        cases = [
            ("docsearch all --full; " + json.dumps(no_skip), {}),  # V2b: no slide skip -> not the M3 default
            (self.M3 + "; box subset", {}),  # other scope
            (self.M3, {"n": 50}),  # other rerank settings
            (self.M3, {"reranker": "org/other"}),
            ("x; " + json.dumps({**self.blob, "visual_blind_formats": ["pptx"]}), {}),  # an earlier challenger
            ("x; " + json.dumps({**self.blob, "page_rerank_pages": 30, "page_reranker": "org/vl"}), {}),  # M19
            ("x; " + json.dumps({**self.blob, "text_rich_chars": 200, "text_rich_formats": []}), {}),  # M25
            ("eval --full; " + json.dumps(pre_m28), {}),  # chunked before the shared spreadsheet cap
        ]
        for notes, kw in cases:
            self.write_ref(notes, **kw)
            _, ref, _ = _one_change_plan(self.cfg, self.args())
            self.assertIsNotNone(ref, notes)

    def test_skip_formats_still_supported(self):
        from docsearch.cli import _one_change_plan

        self.write_ref(self.M3)
        _, _, over = _one_change_plan(self.cfg, self.args(visual_blind_formats=None, rerank_skip_formats=["xlsx"]))
        self.assertEqual(over, {("search", "rerank_skip_formats"): ["xlsx"]})

    def test_page_rerank_pages_alone_is_a_change(self):
        from docsearch.cli import _one_change_plan

        self.write_ref(self.M3)
        plan, _, over = _one_change_plan(self.cfg, self.args(visual_blind_formats=None, page_rerank_pages=30,
                                                              name="m19b"))  # fmt: skip
        self.assertEqual(over, {("search", "page_rerank_pages"): 30})
        self.assertEqual(plan, [("m19b", "hybrid", "hybrid_visual_fa", True)])


class TestRerankerChallenger(TestOneChangePlan):
    def test_reranker_swap_reuses_m3_reference(self):
        from docsearch.cli import _one_change_plan

        self.write_ref(self.M3)
        plan, ref, over = _one_change_plan(self.cfg, self.args(visual_blind_formats=None, reranker="qwen3-4b",
                                                                name="m5_qwen3_4b"))  # fmt: skip
        self.assertIsNone(ref)  # reference = default gte run, found before the reranker is swapped
        self.assertEqual(plan, [("m5_qwen3_4b", "hybrid", "hybrid_visual_fa", True)])
        self.assertEqual(over, {("models", "reranker"): "qwen3-4b"})

    def test_per_reranker_batch_size(self):
        from docsearch.config import resolve_reranker
        from docsearch.models import rerank_batch_size

        self.cfg["models"]["rerankers"] = {"qwen3-4b": "Qwen/Qwen3-Reranker-4B"}
        self.cfg["models"]["rerank_batch_sizes"] = {"qwen3-4b": 4}
        self.assertEqual(rerank_batch_size(self.cfg), 16)  # gte: global default
        resolve_reranker(self.cfg, "qwen3-4b")
        self.assertEqual(rerank_batch_size(self.cfg), 4)


class TestBlindFormatsFusion(unittest.TestCase):
    def make(self, blind):
        from docsearch.search import Retriever

        r = Retriever.__new__(Retriever)
        r.sc = SimpleNamespace(rrf_k=60)
        meta = {"deck.pptx#p1": {"doc_id": "deck.pptx", "format": "pptx"},
                "a.pdf#p1": {"doc_id": "a.pdf", "format": "pdf"}}  # fmt: skip
        r.visual = SimpleNamespace(docs={"deck.pptx", "a.pdf"}, meta=meta)
        r.visual_seen = {m["doc_id"] for m in meta.values() if m["format"] not in blind}
        return r

    def test_blind_format_keeps_text_rank(self):
        t = ["deck.pptx", "a.pdf#p1", "x.docx"]  # text ranking (pptx is file-level)
        v = ["a.pdf#p1"]  # visual ranking: the pdf page only (the pptx render did not match)
        seen = [u for u, _ in self.make(set())._fuse_format_aware(t, v)]
        blind = [u for u, _ in self.make({"pptx"})._fuse_format_aware(t, v)]
        self.assertEqual(seen[0], "a.pdf#p1")  # default: visual vote lifts the pdf above the pptx
        self.assertEqual(blind[:2], ["deck.pptx", "a.pdf#p1"])  # blind: pptx keeps its text rank 1

    def test_default_is_empty(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(load_config(Path(t) / "none.yaml", env={})["visual"]["blind_formats"], [])


if __name__ == "__main__":
    unittest.main()
