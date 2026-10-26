import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from docsearch.config import load_config
from docsearch.hfcache import ModelRef, missing, models_for

SHA_MAIN = "a" * 40
SHA_PIN = "b" * 40


def fake_model(hub: Path, repo: str, sha: str, ref_main: bool = True, weights: bool = True) -> None:
    d = hub / ("models--" + repo.replace("/", "--"))
    snap = d / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}")
    if weights:
        (snap / "model.safetensors").write_text("x")
    if ref_main:
        (d / "refs").mkdir(exist_ok=True)
        (d / "refs" / "main").write_text(sha)


class TestHFCache(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.home = Path(self.td.name)
        self.hub = self.home / "hub"
        self.env = mock.patch.dict(os.environ, {"HF_HOME": str(self.home)}, clear=False)
        self.env.start()
        os.environ.pop("HF_HUB_CACHE", None)
        y = self.home / "docsearch.yaml"
        y.write_text(
            "models:\n  embedder: org/emb\n  reranker: org/rr\n  embedders:\n"
            f"    vis: {{id: org/vis, revision: {SHA_PIN}, kind: visual}}\n"
            "  rerankers: {small: org/rr-small}\nvisual: {default: vis}\n"
        )
        self.cfg = load_config(y, env={})

    def tearDown(self):
        self.env.stop()
        self.td.cleanup()

    def refs(self, cmd="all", **kw):
        return models_for(self.cfg, cmd, SimpleNamespace(**kw))

    def test_all_cached_no_warning_pinned_revision_without_main_ref(self):
        fake_model(self.hub, "org/emb", SHA_MAIN)
        fake_model(self.hub, "org/rr", SHA_MAIN)
        fake_model(self.hub, "org/vis", SHA_PIN, ref_main=False)  # only the pinned commit was downloaded
        self.assertEqual([str(r) for r in self.refs()], ["org/emb", f"org/vis@{SHA_PIN[:12]}", "org/rr"])
        self.assertEqual(missing(self.refs()), [])

    def test_one_missing_is_named(self):
        fake_model(self.hub, "org/emb", SHA_MAIN)
        fake_model(self.hub, "org/vis", SHA_PIN)
        self.assertEqual(missing(self.refs()), [ModelRef("reranker", "org/rr")])

    def test_pinned_revision_must_match(self):
        fake_model(self.hub, "org/emb", SHA_MAIN)
        fake_model(self.hub, "org/rr", SHA_MAIN)
        fake_model(self.hub, "org/vis", SHA_MAIN)  # main cached, but the pinned commit is not
        self.assertEqual([r.repo for r in missing(self.refs())], ["org/vis"])

    def test_config_without_weights_is_not_cached(self):
        fake_model(self.hub, "org/emb", SHA_MAIN, weights=False)
        self.assertEqual([r.repo for r in missing(self.refs("chunk"))], ["org/emb"])

    def test_aliases_and_flags_pick_the_models_loaded(self):
        self.assertEqual([r.repo for r in self.refs("eval", reranker="small", no_rerank=False, visual=None,
                                                    embedder=None)], ["org/emb", "org/rr-small"])  # fmt: skip
        self.assertEqual([r.repo for r in self.refs("eval", no_rerank=True, visual="vis", embedder=None)],
                         ["org/emb", "org/vis"])  # fmt: skip
        self.assertEqual(self.refs("render"), [])


if __name__ == "__main__":
    unittest.main()
