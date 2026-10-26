import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.models import fixed_modules_dir


def snapshot(root: Path, head_path: str, head_dir: str) -> Path:
    snap = root / "snap" / "abc123"
    (snap / head_dir).mkdir(parents=True)
    (snap / head_dir / "config.json").write_text('{"true_token_id": 1, "false_token_id": 2}')
    (snap / "config.json").write_text("{}")
    mods = [{"idx": 0, "name": "0", "path": "", "type": "T"}, {"idx": 1, "name": "1", "path": head_path, "type": "L"}]
    (snap / "modules.json").write_text(json.dumps(mods))
    return snap


class TestFixedModules(unittest.TestCase):
    def test_missing_head_folder_is_renamed_in_a_linked_copy(self):
        with tempfile.TemporaryDirectory() as d:
            snap = snapshot(Path(d), "1_CausalScoreHead", "1_LogitScore")
            with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=str(snap / "modules.json")):
                out = Path(fixed_modules_dir("Qwen/X-8B", Path(d) / "fixed"))
            self.assertEqual(out, Path(d) / "fixed" / "Qwen__X-8B" / "abc123")
            self.assertEqual(json.loads((out / "modules.json").read_text())[1]["path"], "1_LogitScore")
            self.assertTrue((out / "1_LogitScore" / "config.json").exists())
            self.assertTrue((out / "config.json").is_symlink())
            self.assertEqual(json.loads((snap / "modules.json").read_text())[1]["path"], "1_CausalScoreHead")

    def test_consistent_snapshot_keeps_the_repo_id(self):
        with tempfile.TemporaryDirectory() as d:
            snap = snapshot(Path(d), "1_LogitScore", "1_LogitScore")
            with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=str(snap / "modules.json")):
                self.assertEqual(fixed_modules_dir("Qwen/X-2B", Path(d) / "fixed"), "Qwen/X-2B")

    def test_not_cached_keeps_the_repo_id(self):
        with mock.patch("huggingface_hub.try_to_load_from_cache", return_value=None):
            self.assertEqual(fixed_modules_dir("Qwen/X-8B", Path("/nonexistent")), "Qwen/X-8B")


if __name__ == "__main__":
    unittest.main()
