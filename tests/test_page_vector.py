import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from docsearch.visual import PageVectorEncoder


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, items, **kw):
        self.calls.append((items, kw))
        return np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(items), 1))


class TestPageVectorEncoder(unittest.TestCase):
    def setUp(self):
        self.enc = object.__new__(PageVectorEncoder)  # no model load
        self.enc.model, self.enc.max_side = FakeModel(), 64

    def test_a_page_is_one_row_and_gets_no_lead_in(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "page.png"
            Image.new("RGB", (200, 100), "white").save(p)
            seen = []
            vecs = self.enc.docs([str(p)], on_batch=seen.append)
        self.assertEqual([v.shape for v in vecs], [(1, 2)])
        self.assertEqual(seen, [1])
        items, kw = self.enc.model.calls[0]
        self.assertEqual(items[0].size, (64, 32))  # held to max_side
        self.assertNotIn("prompt_name", kw)

    def test_a_question_is_one_row_with_the_query_lead_in(self):
        vecs = self.enc.queries(["a", "b"])
        self.assertEqual([v.shape for v in vecs], [(1, 2), (1, 2)])
        self.assertEqual(self.enc.model.calls[0][1]["prompt_name"], "query")

    def test_maxsim_over_one_row_is_the_cosine(self):
        q, page = self.enc.queries(["a"])[0], np.array([[0.6, 0.8]], dtype=np.float32)
        sims = page @ q.T  # as VectorStore.search: (rows, q rows)
        self.assertAlmostEqual(float(np.maximum.reduceat(sims, [0], axis=0).sum(axis=1)[0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
