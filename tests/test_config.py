import tempfile
import unittest
from pathlib import Path

from docsearch.config import load_config, pick_device


class TestConfig(unittest.TestCase):
    def test_relative_paths_resolve_to_config_dir(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "docsearch.yaml"
            p.write_text("paths: {index_dir: idx}\n")
            cfg = load_config(p, env={})
            self.assertEqual(cfg.path("index_dir"), Path(td).resolve() / "idx")
            self.assertEqual(cfg.path("corpus_dir"), Path(td).resolve() / "corpus")

    def test_env_overrides_with_types(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "docsearch.yaml"
            p.write_text("search: {rerank_candidates: 50}\n")
            env = {
                "DOCSEARCH_SEARCH_RERANK_CANDIDATES": "30",
                "DOCSEARCH_PATHS_DATA_DIR": "/abs/data",
                "DOCSEARCH_SAMPLING_VIDORE_CORPORA": "a,b",
            }
            cfg = load_config(p, env=env)
            self.assertEqual(cfg["search"]["rerank_candidates"], 30)
            self.assertEqual(cfg.path("data_dir"), Path("/abs/data"))
            self.assertEqual(cfg["sampling"]["vidore_corpora"], ["a", "b"])

    def test_resolve_reranker_alias_and_raw_id(self):
        from docsearch.config import resolve_reranker

        cfg = load_config(Path("/nonexistent.yaml"), env={})
        cfg["models"]["rerankers"] = {"zr": "org/zr-model"}
        self.assertEqual(resolve_reranker(cfg, "zr"), "zr")
        self.assertEqual(cfg["models"]["reranker"], "org/zr-model")
        self.assertEqual(resolve_reranker(cfg, "Org/Other-Reranker"), "Other-Reranker")
        self.assertEqual(cfg["models"]["reranker"], "Org/Other-Reranker")

    def test_device_pin(self):
        self.assertEqual(pick_device("cpu"), "cpu")
        self.assertIn(pick_device("auto"), {"cpu", "cuda", "mps"})


if __name__ == "__main__":
    unittest.main()
