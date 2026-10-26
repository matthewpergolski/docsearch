import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from docsearch.config import load_config
from docsearch.search import rrf
from docsearch.visual import fusion_unit


class TestVisualFusion(unittest.TestCase):
    def test_fusion_units(self):
        self.assertEqual(fusion_unit("vidore/a", "pdf", 7), "vidore/a#p7")
        self.assertEqual(fusion_unit("finslides/s", "slide_image", None), "finslides/s#p1")
        self.assertEqual(fusion_unit("ocb/x", "docx", 3), "ocb/x")

    def test_rrf_unit_fusion_rewards_agreement(self):
        fused = [u for u, _ in rrf([["a", "b", "c"], ["c", "a", "d"]], 60)]
        self.assertEqual(fused[0], "a")
        self.assertEqual(fused[1], "c")
        self.assertEqual(set(fused), {"a", "b", "c", "d"})


class _R:
    def __init__(self):
        self.seen = []

    def score(self, q, texts):
        self.seen.append(list(texts))
        return [float(len(t)) for t in texts]


class TestFormatAwareAndSkip(unittest.TestCase):
    def test_unseen_format_keeps_text_rank(self):
        from types import SimpleNamespace

        from docsearch.search import Retriever

        r = Retriever.__new__(Retriever)
        r.sc = SimpleNamespace(rrf_k=60)
        r.visual = SimpleNamespace(docs={"p", "q"})
        # text: xlsx file first, then pages; visual sees only the pages
        fused = [u for u, _ in r._fuse_format_aware(["ocb/x.xlsx", "p#p1", "q#p1"], ["q#p1", "p#p1"])]
        self.assertEqual(fused[0], "ocb/x.xlsx")
        # plain RRF would bury it below both pages
        self.assertEqual([u for u, _ in rrf([["ocb/x.xlsx", "p#p1", "q#p1"], ["q#p1", "p#p1"]], 60)][-1], "ocb/x.xlsx")

    def test_rerank_skips_short_text(self):
        from docsearch.search import rerank

        c = [{"text": "File: a\n" + t} for t in ["bb", "", "ccccccccccccc", "dddddddddddddddddddd"]]
        out = rerank("q", c, _R(), 4, min_chars=5)
        # "bb" and "" are short: they keep positions 0 and 1; the long ones are reranked (longest first)
        self.assertEqual([o["text"][8:] for o in out], ["bb", "", "d" * 20, "c" * 13])


class _FakeEmbedder:
    device, dim, name = "cpu", 4, "Qwen/Qwen3-Embedding-0.6B"

    def docs(self, texts, on_batch=None):
        return np.ones((len(texts), 4), np.float32)

    def queries(self, texts):
        return np.ones((len(texts), 4), np.float32)


class TestEmptyPageStore(unittest.TestCase):
    """Word files without LibreOffice give no page images: sync leaves an empty page store behind."""

    def setUp(self):
        from docsearch.index import build_index
        from docsearch.visual import build_visual_store, pages_dir

        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: pplx-v2-late}\n")
        p = root / "work" / "chunks" / "hybrid.jsonl"
        p.parent.mkdir(parents=True)
        rows = [{"chunk_id": f"hybrid:{d}:0", "doc_id": d, "format": "docx", "source": "s", "pages": [1],
                 "text": t, "n_tokens": 1, "meta": {}}
                for d, t in (("a", "rye sourdough loaves"), ("b", "paid leave days for a new baker"))]  # fmt: skip
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.cfg = load_config(root / "docsearch.yaml", env={})
        pages_dir(self.cfg).mkdir(parents=True)
        (pages_dir(self.cfg) / "pages.jsonl").write_text("")
        with mock.patch("docsearch.models.Embedder", return_value=_FakeEmbedder()), \
                mock.patch("docsearch.altembed.load_encoder", return_value=_FakeEmbedder()):  # fmt: skip
            build_index(self.cfg, "hybrid")
            build_visual_store(self.cfg, "pplx-v2-late")

    def tearDown(self):
        self.td.cleanup()

    def search(self):
        from docsearch.search import search

        err = io.StringIO()
        with mock.patch("docsearch.models.Embedder", return_value=_FakeEmbedder()), \
                contextlib.redirect_stderr(err), self.assertLogs("docsearch.search", "WARNING") as logs:  # fmt: skip
            hits = search(self.cfg, "paid leave for a baker", use_rerank=False, k=2)
        return [h["doc_id"] for h in hits], err.getvalue(), logs.output

    def test_search_runs_text_only_and_says_so(self):
        ids, err, logs = self.search()
        note = "page store pplx-v2-late is empty: page pictures were not used for this search (text only)"
        self.assertEqual(ids[0], "b")
        self.assertEqual(err.strip(), note)
        self.assertEqual(len(logs), 1)
        self.assertIn(note, logs[0])

    def test_search_reads_the_live_snapshot(self):
        from docsearch.sync import Ops, promote

        promote(self.cfg, Ops(self.cfg, full=True).candidate(), None, "test")
        ids, err, _ = self.search()
        self.assertEqual(ids[0], "b")
        self.assertIn("page pictures were not used", err)

    def test_eval_with_a_visual_still_stops(self):
        from docsearch.search import Retriever

        with self.assertRaises(SystemExit) as e:
            Retriever(self.cfg, "hybrid", None, "pplx-v2-late")
        self.assertIn("run `docsearch render` and `docsearch index --visual pplx-v2-late`", str(e.exception))


if __name__ == "__main__":
    unittest.main()
