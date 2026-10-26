import gc
import json
import tempfile
import unittest
import warnings
from pathlib import Path

from docsearch.config import load_config


def unclosed(fn) -> list[str]:
    """ResourceWarnings raised while fn runs and its garbage is collected. A warning raised when a file object is
    freed never reaches the caller as an exception (the error filter would only print it), so record them."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ResourceWarning)
        fn()
        gc.collect()
    return [str(w.message) for w in seen if issubclass(w.category, ResourceWarning)]


class TestCacheFilesAreClosed(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        (root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n")
        self.cfg = load_config(root / "docsearch.yaml", env={})
        self.cache = self.cfg.path("cache_dir")
        self.cache.mkdir(parents=True)

    def tearDown(self):
        self.td.cleanup()

    def test_reranker_reads_its_score_cache_and_closes_it(self):
        from docsearch.models import Reranker

        probe = Reranker(self.cfg)
        probe.cache_path.write_text(json.dumps(["k", 0.5]) + "\n")
        box = []
        self.assertEqual(unclosed(lambda: box.append(Reranker(self.cfg))), [])
        self.assertEqual(box[0].cache, {"k": 0.5})

    def test_page_reranker_and_read_pages_close_their_files(self):
        from docsearch.models import PageReranker
        from docsearch.visual import pages_dir

        pages_dir(self.cfg).mkdir(parents=True)
        (pages_dir(self.cfg) / "pages.jsonl").write_text(json.dumps({"unit": "d#p1", "path": "d_1.png"}) + "\n")
        probe = PageReranker(self.cfg, "some/model")
        probe.cache_path.write_text(json.dumps(["k", 0.25]) + "\n")
        box = []
        self.assertEqual(unclosed(lambda: box.append(PageReranker(self.cfg, "some/model"))), [])
        self.assertEqual((box[0].paths, box[0].cache), ({"d#p1": "d_1.png"}, {"k": 0.25}))


if __name__ == "__main__":
    unittest.main()
