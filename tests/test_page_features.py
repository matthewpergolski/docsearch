"""The page reranker keeps the vision features of recent pages: a page seen again skips the vision tower. On the
real model the scores are the same to the last digit (checked once by hand); here a fake model counts the calls."""

import unittest
from types import SimpleNamespace

from docsearch.models import FeatureCache


class Inner:
    def __init__(self):
        self.calls = []

    def get_image_features(self, pixel_values, image_grid_thw=None, **kw):
        self.calls.append(pixel_values)
        return SimpleNamespace(pooler_output=(f"emb {pixel_values}",), deepstack_features=[(f"deep {pixel_values}",)],
                               last_hidden_state="big")  # fmt: skip


class TestFeatureCache(unittest.TestCase):
    def test_a_page_seen_again_reuses_its_features_and_the_oldest_page_leaves_first(self):
        inner = Inner()
        cache = FeatureCache(inner, size=2)
        for unit in ("a", "b", "a", "c", "b"):
            with cache.page(unit):
                out = inner.get_image_features(f"pixels {unit}")
        self.assertEqual(inner.calls, ["pixels a", "pixels b", "pixels c", "pixels b"])
        self.assertEqual(out.pooler_output, ("emb pixels b",))
        self.assertFalse(hasattr(out, "last_hidden_state") and out.last_hidden_state == "big")

    def test_outside_a_page_the_vision_tower_always_runs(self):
        inner = Inner()
        FeatureCache(inner, size=2)
        inner.get_image_features("x")
        inner.get_image_features("x")
        self.assertEqual(inner.calls, ["x", "x"])


class TestStop(unittest.TestCase):
    """A newer search stops the page reranker between pages; the pages it scored stay in the score cache."""

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        from docsearch import models
        from docsearch.config import load_config

        self.models = models
        d = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, d)
        (d / "pages").mkdir()
        rows = [{"unit": f"deck#p{i}", "doc_id": "deck", "format": "pdf", "page": i, "path": str(d / f"{i}.png")}
                for i in (1, 2, 3)]  # fmt: skip
        (d / "pages" / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = load_config(env={})
        cfg.raw["paths"]["index_dir"] = str(d / "index")
        cfg.raw["paths"]["cache_dir"] = str(d / "cache")
        self.pr = models.PageReranker(cfg, "fake")
        self.scored = []
        self.stop_after_first = False

        def predict(pairs, **kw):
            self.scored.append(pairs[0][1])
            if self.stop_after_first and len(self.scored) == 1:
                models.stop_when(lambda: True)  # a newer search arrives while the first page is scored
            return [0.5]

        patches = [mock.patch.object(models.PageReranker, "model", mock.PropertyMock(return_value=SimpleNamespace(predict=predict))),
                   mock.patch.object(models.PageReranker, "_load", lambda self, unit: unit)]  # fmt: skip
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(models.stop_when, None)
        self.addCleanup(models.on_page, None)

    def test_the_page_hook_hears_which_pages_will_be_read_and_each_one_done(self):
        heard = []
        self.models.on_page(lambda units, done: heard.append((list(units), list(done))))
        self.pr.cache[self.models.Reranker.key("q", "deck#p2")] = 0.7  # scored before: no new reading
        self.pr.score("q", ["deck#p1", "deck#p2", "deck#p3", "deck#p9"])  # p9 has no rendered page
        units = ["deck#p1", "deck#p2", "deck#p3"]
        self.assertEqual(
            heard, [(units, ["deck#p2"]), (units, ["deck#p2", "deck#p1"]), (units, ["deck#p2", "deck#p1", "deck#p3"])]
        )

    def test_the_page_hook_is_quiet_when_every_page_was_scored_before(self):
        heard = []
        self.models.on_page(lambda units, done: heard.append(units))
        self.pr.cache[self.models.Reranker.key("q", "deck#p1")] = 0.7
        self.pr.score("q", ["deck#p1"])
        self.assertEqual(heard, [])

    def test_a_newer_search_stops_it_between_pages_and_keeps_what_it_scored(self):
        self.stop_after_first = True
        with self.assertRaises(self.models.Superseded):
            self.pr.score("q", ["deck#p1", "deck#p2", "deck#p3"])
        self.assertEqual(self.scored, ["deck#p1"])
        self.models.stop_when(None)
        self.assertEqual(self.pr.score("q", ["deck#p1"]), [0.5])
        self.assertEqual(self.scored, ["deck#p1"])
