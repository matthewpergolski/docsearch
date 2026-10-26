"""+enrich in sync (H2 wired in): a context line on each slide image, the hybrid_ctx_slides variant, the cheap-model
guard, and the variant that search and the search page read. A fake LLM; no model, no data/."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch import sync as S
from docsearch.chunk import Chunk, chunks_path, read_chunks, replace_doc_chunks
from docsearch.config import load_config

HAVE_DJANGO = importlib.util.find_spec("django") is not None and importlib.util.find_spec("ninja") is not None


class FakeLLM:
    s = None  # the settings check_available reads (patched out)

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        n = prompt.count("\n[")
        return json.dumps({str(i): f"slide line {i}" for i in range(1, n + 1)})


class ChunkOps(S.Ops):
    """The real Ops, with parse, chunk and render replaced: a .png file is a slide image, others are a pdf."""

    def parse(self, d):
        return {"status": "ok"}

    def chunk(self, d):
        src = self.cfg.path("corpus_dir") / d.path
        fmt = "slide_image" if d.doc_id.endswith(".png") else "pdf"
        c = Chunk(f"hybrid:{d.doc_id}:0", d.doc_id, fmt, "inbox", [1], f"File: {src.name}\n{src.read_text()}")
        replace_doc_chunks(chunks_path(self.cfg, "hybrid"), {d.doc_id}, [c])
        return 1

    def render(self, d):
        return 0

    def meta(self, d):
        pass

    def flush_meta(self):
        pass


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.inbox = self.root / "inbox"
        self.inbox.mkdir()
        (self.inbox / "deck_page_3.png").write_text("Overview\n3Q18 revenue up 2%")
        (self.inbox / "report.pdf").write_text("Annual report\nwages by region")
        self.indexed: list[str] = []

    def tearDown(self):
        self.td.cleanup()

    def cfg(self, yaml: str = "", env=None):
        (self.root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n" + yaml)
        return load_config(self.root / "docsearch.yaml", env=env or {})

    def build_index(self, cfg, variant, rebuild=False):
        self.indexed.append(variant)
        return {"embedded_now": 0}

    def sync(self, cfg, llm):
        with mock.patch("docsearch.index.build_index", self.build_index), \
             mock.patch("docsearch.llm.get_client", lambda *a, **k: llm), \
             mock.patch("docsearch.llm.check_available", lambda *a, **k: None):  # fmt: skip
            return S.sync(cfg, self.inbox, ops=ChunkOps(cfg))

    def chunks(self, cfg, variant):
        return {c.doc_id: c.text for c in read_chunks(chunks_path(cfg, variant))}


class TestSyncEnrich(Base):
    def test_enrich_on_builds_the_slide_variant_and_only_the_slide_gets_a_line(self):
        cfg = self.cfg("enrich: {llm: local}\n")
        llm = FakeLLM()
        self.sync(cfg, llm)
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("3Q18 revenue", llm.prompts[0])
        plain, ctx = self.chunks(cfg, "hybrid"), self.chunks(cfg, "hybrid_ctx_slides")
        self.assertEqual(
            ctx["inbox/deck_page_3.png"], "File: deck_page_3.png\nContext: slide line 1\nOverview\n3Q18 revenue up 2%"
        )
        self.assertEqual(ctx["inbox/report.pdf"], plain["inbox/report.pdf"])
        self.assertEqual(self.indexed, ["hybrid", "hybrid_ctx_slides"])

    def test_a_second_sync_asks_the_llm_nothing_for_an_unchanged_slide(self):
        cfg = self.cfg("enrich: {llm: local}\n")
        self.sync(cfg, FakeLLM())
        (self.inbox / "report.pdf").write_text("Annual report\nwages by region, revised")
        llm = FakeLLM()
        self.sync(cfg, llm)
        self.assertEqual(llm.prompts, [])
        self.assertIn("Context: slide line 1", self.chunks(cfg, "hybrid_ctx_slides")["inbox/deck_page_3.png"])
        self.assertIn("revised", self.chunks(cfg, "hybrid_ctx_slides")["inbox/report.pdf"])
        cached = [json.loads(p.read_text()) for p in (cfg.path("cache_dir") / "enrich").glob("*.json")]
        self.assertEqual(
            [c["model"] for c in cached], ["claude:haiku"]
        )  # the H2 trial's cache key: its lines are reused

    def test_turning_enrich_on_for_an_indexed_folder_builds_the_slide_variant(self):
        self.sync(self.cfg(), FakeLLM())
        cfg, llm = self.cfg("enrich: {llm: local}\n"), FakeLLM()
        self.indexed.clear()
        self.sync(cfg, llm)
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("inbox/deck_page_3.png", self.chunks(cfg, "hybrid_ctx_slides"))

    def test_enrich_off_makes_no_llm_call_and_no_enriched_variant(self):
        cfg = self.cfg()
        llm = mock.Mock(side_effect=AssertionError("no LLM with enrich.llm off"))
        self.sync(cfg, llm)
        self.assertFalse(chunks_path(cfg, "hybrid_ctx_slides").exists())
        self.assertEqual(self.indexed, ["hybrid"])
        self.assertEqual(ChunkOps(cfg).tables, {"hybrid": "chunks_hybrid"})


class TestCheapModelGuard(Base):
    def test_defaults_are_claude_haiku_and_off(self):
        e = self.cfg()["enrich"]
        self.assertEqual((e["llm"], e["backend"], e["provider"], e["model"]), ("off", "cli", "claude", "haiku"))

    def test_a_model_that_is_not_a_small_one_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            self.cfg("enrich: {llm: local, model: sonnet}\n")
        self.assertIn("enrich.model must be a small, cheap model", str(cm.exception))
        with self.assertRaises(ValueError):
            self.cfg(env={"DOCSEARCH_ENRICH_MODEL": "anthropic/claude-opus-5"})

    def test_a_large_model_whose_name_holds_a_small_word_is_refused(self):
        for model in ("gemini/gemini-2.5-pro", "minimax-m2", "nanoGPT-xl"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.cfg(f"enrich: {{llm: api, backend: api, provider: openai, model: {model}}}\n")

    def test_small_models_pass(self):
        self.cfg("enrich: {llm: api, backend: api, model: anthropic/claude-haiku-5-5}\n")
        self.cfg("enrich: {llm: api, backend: api, provider: openai, model: openai/gpt-5.4-mini}\n")
        self.cfg("enrich: {llm: api, backend: api, model: gemini/gemini-2.5-flash-lite}\n")
        self.cfg("enrich: {llm: api, backend: api, provider: ollama, model: ollama/qwen3:8b}\n")  # local: no token cost


class TestSearchVariant(Base):
    def run_search(self, cfg):
        from docsearch import display

        seen = []

        def fake_search(cfg, query, variant, *a, **k):
            seen.append(variant)
            return []

        with mock.patch("docsearch.search.search", fake_search), mock.patch("docsearch.meta.read_meta", lambda c: {}):
            display.run(cfg, "revenue", agent=False)
        return seen

    def test_search_reads_the_slide_variant_only_with_enrich_on(self):
        self.assertEqual(self.run_search(self.cfg()), ["hybrid"])
        self.assertEqual(self.run_search(self.cfg("enrich: {llm: local}\n")), ["hybrid_ctx_slides"])

    def test_the_context_line_is_not_shown_as_a_title(self):
        from docsearch import display

        plain = "File: deck_page_3.png\nOverview\n3Q18 revenue"
        ctx = "File: deck_page_3.png\nContext: IBM 3Q18 earnings overview\nOverview\n3Q18 revenue"
        self.assertEqual(display.title_and_body(ctx, "slide_image"), display.title_and_body(plain, "slide_image"))

    def test_cli_search_follows_enrich_unless_a_variant_is_named(self):
        from docsearch import cli

        seen = []
        run = lambda cfg, q, variant=None, *a, **k: (seen.append(variant), (mock.MagicMock(groups=()), []))[1]  # noqa: E731
        with mock.patch("docsearch.display.run", run), mock.patch("docsearch.display.as_json", lambda r: {}), \
             mock.patch("docsearch.cli.load_config", lambda *a, **k: self.cfg("enrich: {llm: local}\n")):  # fmt: skip
            cli.main(["search", "revenue", "--json"])
        self.assertIn(seen[0], (None, "hybrid_ctx_slides"))


@unittest.skipUnless(HAVE_DJANGO, "the web extra is not installed (uv sync --extra ui)")
class TestPageVariant(Base):
    def test_the_page_preview_reads_the_slide_variant_with_enrich_on(self):
        from docsearch import web

        web.setup()
        from ninja.testing import TestClient

        from docsearch.web import api

        seen = []
        row = {"chunk_id": "hybrid:s/deck_page_3:0", "doc_id": "s/deck_page_3", "format": "slide_image", "pages": [],
               "text": "File: deck_page_3.png\nContext: a line\nOverview\n3Q18 revenue"}  # fmt: skip

        def rows(cfg, variant, docs):
            seen.append(variant)
            return [row]

        for yaml, want in (("", "hybrid"), ("enrich: {llm: local}\n", "hybrid_ctx_slides")):
            api.CFG = self.cfg(yaml)
            with mock.patch("docsearch.search.page_rows", rows):
                parts = TestClient(api.api).get("/text?file=s/deck&page=3").json()["parts"]
            self.assertEqual(seen.pop(), want)
            self.assertEqual(parts, [{"title": "Overview", "body": "3Q18 revenue"}])


if __name__ == "__main__":
    unittest.main()
