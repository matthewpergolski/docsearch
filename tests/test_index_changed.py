import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from docsearch.config import Config
from docsearch.index import build_index, connect


class FakeEmbedder:
    """4-dim vectors from the text length, so a changed text gives a changed vector."""

    dim, name, device = 4, "fake", "cpu"
    texts: list[str] = []

    def __init__(self, cfg):
        pass

    def docs(self, texts, on_batch=None):
        FakeEmbedder.texts += texts
        return [np.full(4, len(t), dtype=np.float32) for t in texts]


def write_chunks(root: Path, texts: dict[str, str]) -> None:
    p = root / "work" / "chunks" / "hybrid.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"chunk_id": cid, "doc_id": "d", "format": "xlsx", "source": "s", "pages": [], "text": t,
             "n_tokens": 1, "meta": {}} for cid, t in texts.items()]  # fmt: skip
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestIndexChanged(unittest.TestCase):
    def test_changed_text_under_same_id_is_embedded_again(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("docsearch.models.Embedder", FakeEmbedder):
            root = Path(d)
            cfg = Config(raw={"paths": {"index_dir": "work/index"}}, root=root)
            write_chunks(root, {"hybrid:d:0": "table 1 rows", "hybrid:d:1": "table 1 more"})
            build_index(cfg, "hybrid")
            FakeEmbedder.texts = []
            write_chunks(root, {"hybrid:d:0": "table 1 rows", "hybrid:d:1": "table 2 header and rows"})
            stats = build_index(cfg, "hybrid")
            self.assertEqual(FakeEmbedder.texts, ["table 2 header and rows"])
            self.assertEqual((stats["rows"], stats["embedded_now"], stats["deleted_stale"]), (2, 1, 1))
            rows = {r["chunk_id"]: r for r in connect(cfg).open_table("chunks_hybrid").to_arrow().to_pylist()}
            self.assertEqual(rows["hybrid:d:1"]["text"], "table 2 header and rows")
            self.assertEqual(rows["hybrid:d:1"]["vector"][0], len("table 2 header and rows"))


if __name__ == "__main__":
    unittest.main()
