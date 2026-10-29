"""Corpus-driven empty-page example searches (not the static web.examples yaml list)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from docsearch.config import Config


def _cfg(root: Path, examples: list[str] | None = None) -> Config:
    raw = {
        "paths": {
            "data_dir": "data",
            "corpus_dir": "corpus",
            "index_dir": "work/index",
        },
        "enrich": {"llm": "off"},
        "web": {"examples": examples or ["yaml only fallback"]},
    }
    return Config(raw=raw, root=root)


def _write_chunks(root: Path, texts: list[str], name: str = "hybrid.jsonl") -> Path:
    p = root / "work" / "chunks" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(texts):
        rows.append(
            {
                "chunk_id": f"hybrid:doc{i}:{i}",
                "doc_id": f"src/doc{i}",
                "format": "pdf",
                "source": "src",
                "pages": [1],
                "text": t,
                "n_tokens": 10,
                "meta": {},
            }
        )
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


class FakeDraftClient:
    """Minimal LLM stand-in: returns JSON citing shown chunk ids (Drafter path)."""

    def __init__(self):
        self.s = mock.Mock(backend="api", provider="claude", model="fake")
        self.n = 0

    def complete(self, prompt: str, images=None) -> str:
        self.n += 1
        import re

        ids = re.findall(r"\[chunk_id: ([^\]]+)\]", prompt)
        if "do NOT answer" in prompt or "do NOT answer" in prompt.lower() or "passages do NOT answer" in prompt:
            return json.dumps({"question": f"Missing year detail {self.n}?", "why_unanswerable": "absent"})
        cites = ids[:2] if len(ids) >= 2 else ids[:1]
        return json.dumps({"question": f"How did the figure change ({self.n})?", "answer": "12%", "citations": cites})


class TestWebExamples(unittest.TestCase):
    def test_draft_from_chunks_uses_questions_drafter(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root)
            _write_chunks(
                root,
                [
                    "Revenue grew 12% to $4.2B in 2019 while margins fell 3 points across segments. " * 4,
                    "Supplier keeps financial records for seven years under the master agreement. " * 4,
                    "Barista annual salary assumptions list base pay at fifty thousand pounds. " * 4,
                ],
            )
            from docsearch.questions import load_chunks

            qs = W.draft_from_chunks(cfg, n=2, client=FakeDraftClient(), chunks=load_chunks(cfg))
            self.assertEqual(len(qs), 2)
            self.assertTrue(all(q.endswith("?") for q in qs))

    def test_no_chunks_returns_yaml_fallback(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root, ["static A", "static B"])
            self.assertEqual(W.examples(cfg), ["static A", "static B"])

    def test_chunks_drive_examples_not_yaml(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root, ["yaml only fallback"])
            _write_chunks(
                root,
                [
                    "Revenue from Mobile in Q3 2018 was 1.2 billion dollars according to the filing. " * 3,
                    "Supplier must keep financial records for seven years under the contract. " * 3,
                ],
            )
            drafted = ["corpus Q about mobile revenue?", "corpus Q about record retention?"]
            got = W.examples(cfg, draft=lambda cfg, n=6: drafted[:n])
            self.assertEqual(got, drafted)
            self.assertNotIn("yaml only fallback", got)

    def test_cache_reused_until_corpus_changes(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root)
            _write_chunks(root, ["Chunk text with enough characters to look like a real passage. " * 5])
            calls = []

            def draft(cfg, n=6):
                calls.append(n)
                return [f"drafted-{len(calls)}"]

            first = W.examples(cfg, draft=draft)
            second = W.examples(cfg, draft=draft)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)

            # corpus change (rewrite chunks) must regenerate
            import time

            time.sleep(0.01)
            _write_chunks(root, ["A different corpus passage that is long enough for sampling. " * 5])
            third = W.examples(cfg, draft=draft)
            self.assertEqual(third, ["drafted-2"])
            self.assertEqual(len(calls), 2)

    def test_draft_failure_falls_back_to_yaml(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root, ["yaml fallback"])
            _write_chunks(root, ["Enough text for a chunk so the corpus key is present. " * 5])

            def boom(cfg, n=6):
                raise RuntimeError("llm down")

            self.assertEqual(W.examples(cfg, draft=boom), ["yaml fallback"])

    def test_empty_draft_falls_back_to_yaml(self):
        from docsearch import web_examples as W

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root, ["yaml fallback"])
            _write_chunks(root, ["Enough text for a chunk so the corpus key is present. " * 5])
            self.assertEqual(W.examples(cfg, draft=lambda cfg, n=6: []), ["yaml fallback"])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("django") is not None
    and __import__("importlib").util.find_spec("ninja") is not None,
    "the web extra is not installed (uv sync --extra ui)",
)
class TestConfigServesCorpusExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from docsearch import web

        web.setup()

    def test_config_examples_come_from_resolver(self):
        from ninja.testing import TestClient

        from docsearch.config import load_config
        from docsearch.web import api

        api.CFG = load_config(env={})
        client = TestClient(api.api)
        with mock.patch("docsearch.web_examples.examples", return_value=["from corpus"]):
            body = client.get("/config").json()
        self.assertEqual(body["examples"], ["from corpus"])


if __name__ == "__main__":
    unittest.main()
