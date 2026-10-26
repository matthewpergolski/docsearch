"""run_eval end to end on a tiny LanceDB table with a fake embedder: no model, no data/."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from docsearch import ui
from docsearch.config import load_config

QVEC = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestRunNotes(unittest.TestCase):
    def setUp(self):
        os.environ["DOCSEARCH_PLAIN"] = "1"
        ui._state["tty"] = None
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n")
        self.cfg = load_config(root / "docsearch.yaml", env={})
        from docsearch.chunk import chunks_path
        from docsearch.index import connect

        rows = [{"chunk_id": f"c{i}", "doc_id": f"s/d{i}", "format": "pdf", "source": "s", "pages": [1],
                 "text": f"rye bread {i}", "vector": np.array([1.0, i / 10, 0.0, 0.0], np.float32)} for i in range(4)]  # fmt: skip
        connect(self.cfg).create_table("chunks_hybrid", rows).create_fts_index("text", use_tantivy=False)
        cp = chunks_path(self.cfg, "hybrid")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("".join(json.dumps({"doc_id": r["doc_id"]}) + "\n" for r in rows))
        data = self.cfg.path("data_dir")
        data.mkdir(parents=True, exist_ok=True)
        who = ["bot-a", "bot-a", "bot-b", None]
        qs = [{"qid": f"hard:{i}", "query": f"rye bread {i}", "source": "s", "format": "pdf", "level": "page",
               "gold": {f"s/d{i}#p1": 1}, "meta": {"type": "fact", "status": "keep",
                                                   **({"reviewer": w} if w else {})}} for i, w in enumerate(who)]  # fmt: skip
        (data / "questions_hard.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))

    def tearDown(self):
        self.td.cleanup()

    def test_notes_count_the_reviewed_questions_without_naming_the_reviewer(self):
        from docsearch.eval import run_eval

        fake = SimpleNamespace(queries=lambda texts: np.stack([QVEC for _ in texts]))
        with mock.patch("docsearch.models.Embedder", return_value=fake):
            run_eval(self.cfg, [("t", "hybrid", "hybrid", False)], questions="hard", reviewed_only=True)
        with (self.cfg.path("results_dir") / "runs.csv").open() as f:
            notes = next(csv.DictReader(f))["notes"]
        self.assertIn('reviewed_by={"human": 1, "reviewer": 3}', notes)
        self.assertNotIn("bot-", notes)

    def test_a_local_model_path_is_written_as_its_repo_id(self):
        from docsearch.eval import run_eval

        snap = Path.home() / ".cache/huggingface/hub/models--acme--Tiny-Embed/snapshots/e24ca1e346bcc53572bd"
        self.cfg["models"]["embedder"] = str(snap)
        fake = SimpleNamespace(queries=lambda texts: np.stack([QVEC for _ in texts]))
        with mock.patch("docsearch.models.Embedder", return_value=fake):
            run_eval(self.cfg, [("t", "hybrid", "hybrid", False)], questions="hard")
        with (self.cfg.path("results_dir") / "runs.csv").open() as f:
            embedder = next(csv.DictReader(f))["embedder"]
        self.assertEqual(embedder, "acme/Tiny-Embed@e24ca1e346bc")


class TestModelLabel(unittest.TestCase):
    """runs.csv never records a local path: the fake paths are built from parts, so no home path is in this file."""

    def test_a_windows_cache_path_is_written_as_its_repo_id(self):
        from docsearch.eval import _model_label

        parts = ["C:", "Users", "alice", ".cache", "huggingface", "hub", "models--acme--Tiny-Embed", "snapshots"]
        self.assertEqual(_model_label("\\".join([*parts, "e24ca1e346bcc53572bd"])), "acme/Tiny-Embed@e24ca1e346bc")

    def test_a_local_model_folder_outside_the_cache_is_written_as_its_name(self):
        from docsearch.eval import _model_label

        for path in (str(Path.home() / "models" / "Tiny-Embed"), "\\".join(["D:", "models", "Tiny-Embed", ""])):
            with self.subTest(path=path):
                self.assertEqual(_model_label(path), "Tiny-Embed")

    def test_a_hub_id_is_kept(self):
        from docsearch.eval import _model_label

        self.assertEqual(_model_label("Qwen/Qwen3-Embedding-0.6B"), "Qwen/Qwen3-Embedding-0.6B")


if __name__ == "__main__":
    unittest.main()
