import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from docsearch.altembed import ContextualEncoder, VectorStore, _done_ids, _save_shard, store_dir
from docsearch.chunk import Chunk
from docsearch.config import load_config


def _cfg(tmp: Path, kind: str):
    cfg = load_config(Path("/nonexistent.yaml"), env={})
    cfg.root = tmp
    cfg["paths"]["index_dir"] = str(tmp / "index")
    cfg["models"]["embedders"] = {"x": {"id": "fake/x", "kind": kind}}
    return cfg


class TestVectorStore(unittest.TestCase):
    def test_maxsim_ranks_by_sum_of_best_token_matches(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = _cfg(Path(t), "multivector")
            d = store_dir(cfg, "x", "hybrid")
            d.mkdir(parents=True)
            e = np.eye(4, dtype=np.float16)
            # chunk a matches query tokens 0 and 1; chunk b matches only token 0; chunk c matches nothing
            _save_shard(d, ["a", "b"], [np.stack([e[0], e[1]]), np.stack([e[0], e[2], e[2]])])
            _save_shard(d, ["c"], [np.stack([e[3]])])
            self.assertEqual(_done_ids(d), {"a", "b", "c"})
            hits = VectorStore(cfg, "x").search(np.stack([e[0], e[1]]).astype(np.float32), 3)
            self.assertEqual([h[0] for h in hits], ["a", "b", "c"])
            self.assertAlmostEqual(hits[0][1], 2.0)
            self.assertAlmostEqual(hits[1][1], 1.0)

    def test_dense_rows_cosine(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = _cfg(Path(t), "contextual")
            d = store_dir(cfg, "x", "hybrid")
            d.mkdir(parents=True)
            _save_shard(d, ["a", "b"], [np.array([1.0, 0.0]), np.array([0.6, 0.8])])
            meta = json.loads((d / "shard_00000.json").read_text())
            self.assertEqual(meta["lens"], [1, 1])
            hits = VectorStore(cfg, "x").search(np.array([0.0, 1.0], dtype=np.float32), 2)
            self.assertEqual([h[0] for h in hits], ["b", "a"])


class TestWindows(unittest.TestCase):
    def test_windows_split_on_doc_and_budget(self):
        enc = ContextualEncoder.__new__(ContextualEncoder)
        enc.window_tokens, enc.chunk_tokens = 100, 60

        def c(i, doc, n):
            return Chunk(f"c{i}", doc, "pdf", "s", [1], "t", n)

        chunks = [c(0, "d1", 40), c(1, "d1", 40), c(2, "d1", 40), c(3, "d2", 10), c(4, "d2", 500)]
        w = [[x.chunk_id for x in win] for win in enc.windows(chunks)]
        self.assertEqual(w, [["c0", "c1"], ["c2"], ["c3", "c4"]])


if __name__ == "__main__":
    unittest.main()
