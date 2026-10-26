"""A long-lived process (the web server) keeps models loaded between searches; a CLI run builds them as before."""

import unittest
from unittest import mock

from docsearch import models
from docsearch.config import load_config


class TestKeepLoaded(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(env={})
        self.addCleanup(models.keep_loaded, False)
        self.built = []

    def fake(self, kind):
        return lambda *a: self.built.append(kind) or object()

    def test_off_by_default_each_call_builds_a_new_model(self):
        with (
            mock.patch.object(models, "Embedder", self.fake("e")),
            mock.patch.object(models, "Reranker", self.fake("r")),
        ):
            self.assertIsNot(models.load_embedder(self.cfg), models.load_embedder(self.cfg))
            models.load_reranker(self.cfg)
            models.load_reranker(self.cfg)
        self.assertEqual(self.built, ["e", "e", "r", "r"])

    def test_kept_loaded_each_model_is_built_once_per_setting(self):
        models.keep_loaded(True)
        with mock.patch.object(models, "Embedder", self.fake("e")), mock.patch.object(models, "Reranker", self.fake("r")), \
             mock.patch.object(models, "PageReranker", self.fake("p")):  # fmt: skip
            self.assertIs(models.load_embedder(self.cfg), models.load_embedder(self.cfg))
            self.assertIs(models.load_reranker(self.cfg), models.load_reranker(self.cfg))
            self.assertIs(models.load_page_reranker(self.cfg, "a"), models.load_page_reranker(self.cfg, "a"))
            models.load_page_reranker(self.cfg, "b")
            self.cfg["models"]["embedder"] = "other/model"
            models.load_embedder(self.cfg)
        self.assertEqual(self.built, ["e", "r", "p", "p", "e"])

    def test_kept_loaded_the_page_picture_query_encoder_is_built_once(self):
        # The server loaded the visual query encoder again on each new query (about 6 s a search).
        import tempfile

        import numpy as np

        from docsearch import altembed

        class Enc:
            def queries(self, texts):
                return [np.zeros(2, dtype=np.float32) for _ in texts]

        models.keep_loaded(True)
        with tempfile.TemporaryDirectory() as d:
            self.cfg.raw["paths"]["cache_dir"] = d
            with mock.patch.object(altembed, "load_encoder", lambda cfg, alias: self.built.append(alias) or Enc()):
                altembed.query_embeddings(self.cfg, "vis", ["first query"])
                altembed.query_embeddings(self.cfg, "vis", ["second query"])
        self.assertEqual(self.built, ["vis"])


if __name__ == "__main__":
    unittest.main()


class TestKeptRetriever(unittest.TestCase):
    """The server rebuilt the index reader (page vectors, every table row) on each search: 0.4 to 0.8 s of 1.3 s."""

    def setUp(self):
        from docsearch import search

        self.search = search
        self.cfg = load_config(env={})
        self.addCleanup(models.keep_loaded, False)
        self.addCleanup(setattr, search, "_RETRIEVER", None)
        self.built = []
        self.stamp = "v1"
        fake = mock.patch.object(
            search, "Retriever", lambda cfg, variant, emb, vis, live: self.built.append(vis) or object()
        )
        stamp = mock.patch.object(search, "index_stamp", lambda cfg, variant, vis, live: self.stamp)
        fake.start(), stamp.start()
        self.addCleanup(fake.stop), self.addCleanup(stamp.stop)

    def test_a_cli_run_builds_a_reader_each_time(self):
        self.search.kept_retriever(self.cfg, "hybrid", "vis", None)
        self.search.kept_retriever(self.cfg, "hybrid", "vis", None)
        self.assertEqual(self.built, ["vis", "vis"])

    def test_kept_loaded_one_reader_until_the_index_or_the_setting_changes(self):
        models.keep_loaded(True)
        first = self.search.kept_retriever(self.cfg, "hybrid", "vis", None)
        self.assertIs(self.search.kept_retriever(self.cfg, "hybrid", "vis", None), first)
        self.stamp = "v2"  # a sync or an index run wrote a new table version
        self.assertIsNot(self.search.kept_retriever(self.cfg, "hybrid", "vis", None), first)
        self.search.kept_retriever(self.cfg, "hybrid", None, None)  # the page store went empty: text only
        self.assertEqual(self.built, ["vis", "vis", None])
