import math
import unittest

from docsearch.metrics import hit_at_k, mean_metrics, mrr_at_k, ndcg_at_k, recall_at_k, score_query


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        gold = {"a": 1, "b": 1, "c": 1, "d": 1}
        self.assertEqual(recall_at_k(["a", "x", "b"], gold, 10), 0.5)
        self.assertEqual(recall_at_k(["x"] * 10 + ["a"], gold, 10), 0.0)
        self.assertEqual(recall_at_k(["a"], {}, 10), 0.0)

    def test_recall_cutoff(self):
        ranked = [f"x{i}" for i in range(10)] + ["a"]
        self.assertEqual(recall_at_k(ranked, {"a": 1}, 10), 0.0)
        self.assertEqual(recall_at_k(ranked, {"a": 1}, 11), 1.0)

    def test_ndcg_perfect_and_graded(self):
        gold = {"a": 2, "b": 1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], gold), 1.0)
        # swapped order: dcg = 1/log2(2) + 3/log2(3); idcg = 3 + 1/log2(3)
        exp = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["b", "a"], gold), exp)
        self.assertEqual(ndcg_at_k(["x"], gold), 0.0)
        self.assertEqual(ndcg_at_k(["x"], {}), 0.0)

    def test_mrr_and_hit(self):
        gold = {"c": 1}
        self.assertAlmostEqual(mrr_at_k(["a", "b", "c"], gold), 1 / 3)
        self.assertEqual(mrr_at_k(["a"] * 10 + ["c"], gold), 0.0)
        self.assertEqual(hit_at_k(["c"], gold), 1.0)
        self.assertEqual(hit_at_k(["a"], gold), 0.0)

    def test_score_and_mean(self):
        s = score_query(["a"], {"a": 1})
        self.assertEqual(set(s), {"recall@10", "ndcg@10", "mrr@10", "hit@10", "all@10"})
        m = mean_metrics([{"x": 1.0}, {"x": 0.0}])
        self.assertEqual(m["x"], 0.5)
        self.assertEqual(mean_metrics([]), {})


if __name__ == "__main__":
    unittest.main()
