import tempfile
import unittest
from pathlib import Path

from docsearch.config import Config
from docsearch.subset import load_subset


class TestLoadSubsetWithoutBenchmarkFiles(unittest.TestCase):
    def test_no_corpus_manifest_means_no_subset(self):
        """A fresh clone has no benchmark corpus; `sync --dir` on your own files must not need one."""
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(
                raw={"subset": {"enabled": True, "seed": 1}, "paths": {"data_dir": "data", "corpus_dir": "corpus"}},
                root=Path(d),
            )
            self.assertIsNone(load_subset(cfg, full=False))

    def test_manifest_without_questions_means_no_subset(self):
        """`sync --dir` writes corpus/manifest.csv on its own; the next sync must not need data/questions.jsonl."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "corpus").mkdir()
            (Path(d) / "corpus" / "manifest.csv").write_text("doc_id,path\n")
            cfg = Config(
                raw={"subset": {"enabled": True, "seed": 1}, "paths": {"data_dir": "data", "corpus_dir": "corpus"}},
                root=Path(d),
            )
            self.assertIsNone(load_subset(cfg, full=False))


if __name__ == "__main__":
    unittest.main()
