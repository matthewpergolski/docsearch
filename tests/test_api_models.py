"""models.backend: api (Phase G). A fake OpenAI-compatible server on 127.0.0.1 stands in for the org's model server;
no real model and no data/ are used."""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

from docsearch.config import Config, load_config

ST_CONFIG = {"prompts": {"query": "Q: ", "document": ""}, "default_prompt_name": None}


def fake_vector(text: str) -> list[float]:
    """Deterministic and not unit length, so the client's normalisation shows."""
    return [float(len(text)), float(text.count("a")), float(text.count("e")), 1.0]


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeServer:
    """POST /v1/embeddings and /rerank. Knobs: `fail` (statuses to answer first, e.g. [503, 503]), `sleep_first`
    (seconds to stall the first request), `reverse` (answer out of order), `rerank_shape` ("results" or "tei"),
    `dims` (per-item vector length override)."""

    def __init__(self):
        self.requests: list[tuple[str, dict, dict]] = []
        self.fail: list[int] = []
        self.sleep_first = 0.0
        self.reverse = False
        self.rerank_shape = "results"
        self.dims: dict[int, int] = {}
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                srv.requests.append((self.path, body, dict(self.headers)))
                if srv.sleep_first and len(srv.requests) == 1:
                    time.sleep(srv.sleep_first)
                if srv.fail:
                    code = srv.fail.pop(0)
                    self.send_response(code)
                    self.end_headers()
                    self.wfile.write(b'{"error": "nope"}')
                    return
                if self.path == "/v1/embeddings":
                    data = [{"object": "embedding", "index": i,
                             "embedding": fake_vector(t)[: srv.dims.get(i, 4)] + [0.0] * max(0, srv.dims.get(i, 4) - 4)}
                            for i, t in enumerate(body["input"])]  # fmt: skip
                    out = {"object": "list", "model": body["model"], "data": data[::-1] if srv.reverse else data}
                elif self.path == "/rerank":
                    q = set(body["query"].lower().split())
                    docs = body.get("documents") or body.get("texts")
                    res = [(i, float(len(q & set(d.lower().split())))) for i, d in enumerate(docs)]
                    res = res[::-1] if srv.reverse else res
                    if srv.rerank_shape == "tei":
                        out = [{"index": i, "score": s} for i, s in res]
                    else:
                        out = {"results": [{"index": i, "relevance_score": s} for i, s in res]}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def paths(self) -> list[str]:
        return [p for p, _, _ in self.requests]


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "docsearch.yaml").write_text(
            "subset: {enabled: false}\n"
            "models:\n"
            "  backend: api\n"
            f"  embed_api_url: {self.srv.url}/v1\n"
            f"  rerank_api_url: {self.srv.url}\n"
            "  embed_api_key_env: TEST_EMBED_KEY\n"
            "  rerank_api_key_env: TEST_RERANK_KEY\n"
            "  api_batch_size: 2\n"
            "  api_retries: 3\n"
            "  api_timeout_s: 5\n"
        )
        self.cfg = load_config(self.root / "docsearch.yaml", env={})
        p = mock.patch("docsearch.models.st_config", return_value=ST_CONFIG)
        p.start()
        self.addCleanup(p.stop)
        w = mock.patch("docsearch.apimodels.RETRY_WAIT_S", 0.0)
        w.start()
        self.addCleanup(w.stop)

    def tearDown(self):
        self.srv.close()
        self.td.cleanup()


class TestEmbeddings(ApiCase):
    def test_batches_in_order_by_index_and_normalised(self):
        from docsearch.models import load_embedder

        self.srv.reverse = True
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        got = load_embedder(self.cfg).docs(texts)
        self.assertEqual([len(b["input"]) for _, b, _ in self.srv.requests], [2, 2, 1])
        np.testing.assert_allclose(got, np.stack([unit(fake_vector(t)) for t in texts]), rtol=1e-6)

    def test_client_adds_the_query_lead_in_and_sends_documents_bare(self):
        from docsearch.models import load_embedder

        emb = load_embedder(self.cfg)
        emb.queries(["what is rye"])
        emb.docs(["rye bread"])
        sent = [b for _, b, _ in self.srv.requests]
        self.assertEqual(sent[0], {"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["Q: what is rye"]})
        self.assertEqual(sent[1]["input"], ["rye bread"])

    def test_no_model_config_on_the_hub_embeds_with_no_lead_in_and_names_the_key(self):
        from docsearch.models import load_embedder

        self.cfg["models"]["embedder"] = "text-embedding-3-small"
        for err in (OSError("offline and not cached"), ValueError("Repository Not Found")):
            self.srv.requests.clear()
            with self.subTest(err=err), mock.patch("docsearch.models.st_config", side_effect=err), \
                    self.assertLogs("docsearch.api", "WARNING") as logs:  # fmt: skip
                load_embedder(self.cfg).queries(["what is rye"])
            self.assertEqual(self.srv.requests[0][1]["input"], ["what is rye"])
            self.assertIn("models.embed_api_query_prefix", logs.output[0])
            self.assertIn("text-embedding-3-small", logs.output[0])

    def test_query_prefix_key_sets_the_lead_in_without_the_hub(self):
        from docsearch.models import load_embedder

        self.cfg["models"]["embed_api_query_prefix"] = "query: "
        with mock.patch("docsearch.models.st_config", side_effect=AssertionError("hub read")):
            emb = load_embedder(self.cfg)
            emb.queries(["what is rye"])
            emb.docs(["rye bread"])
        self.assertEqual([b["input"] for _, b, _ in self.srv.requests], [["query: what is rye"], ["rye bread"]])

    def test_chunking_with_a_tokenizer_it_cannot_load_stops_with_the_key_to_fix(self):
        from docsearch.chunk import get_tokenizer

        self.cfg["models"]["embedder"] = "text-embedding-3-small"
        with mock.patch("transformers.AutoTokenizer.from_pretrained",
                        side_effect=OSError("not a valid model identifier")), \
                self.assertRaises(SystemExit) as e:  # fmt: skip
            get_tokenizer(self.cfg)
        msg = str(e.exception.code)
        self.assertIn("models.embedder", msg)
        self.assertIn("text-embedding-3-small", msg)
        self.assertIn("models.embed_api_model", msg)

    def test_model_name_sent_can_differ_from_the_model_identity(self):
        from docsearch.models import load_embedder

        self.cfg["models"]["embed_api_model"] = "qwen3-embed"
        emb = load_embedder(self.cfg)
        emb.queries(["x"])
        self.assertEqual(self.srv.requests[0][1]["model"], "qwen3-embed")
        self.assertEqual(emb.name, "Qwen/Qwen3-Embedding-0.6B")

    def test_dim_comes_from_the_server(self):
        from docsearch.models import load_embedder

        self.assertEqual(load_embedder(self.cfg).dim, 4)

    def test_retries_5xx_and_logs_each_attempt(self):
        from docsearch.models import load_embedder

        self.srv.fail = [503, 502]
        with self.assertLogs("docsearch.api", "WARNING") as logs:
            got = load_embedder(self.cfg).queries(["rye"])
        self.assertEqual(len(self.srv.requests), 3)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("attempt 1/4", logs.output[0])
        self.assertIn("HTTP 503", logs.output[0])
        self.assertIn("attempt 2/4", logs.output[1])
        np.testing.assert_allclose(got[0], unit(fake_vector("Q: rye")), rtol=1e-6)

    def test_retries_a_timeout(self):
        from docsearch.models import load_embedder

        self.cfg["models"]["api_timeout_s"] = 0.3
        self.srv.sleep_first = 1.0
        with self.assertLogs("docsearch.api", "WARNING") as logs:
            load_embedder(self.cfg).queries(["rye"])
        self.assertEqual(len(self.srv.requests), 2)
        self.assertIn("timed out", logs.output[0])

    def test_gives_up_after_the_last_retry(self):
        from docsearch.apimodels import ApiError
        from docsearch.models import load_embedder

        self.cfg["models"]["api_retries"] = 1
        self.srv.fail = [503, 503, 503]
        with self.assertLogs("docsearch.api", "WARNING"), self.assertRaises(ApiError) as e:
            load_embedder(self.cfg).queries(["rye"])
        self.assertEqual(len(self.srv.requests), 2)
        self.assertIn("2 attempts", str(e.exception))
        self.assertIn(f"{self.srv.url}/v1/embeddings", str(e.exception))

    def test_a_4xx_is_not_retried(self):
        from docsearch.apimodels import ApiError
        from docsearch.models import load_embedder

        self.srv.fail = [401]
        with self.assertRaises(ApiError) as e:
            load_embedder(self.cfg).queries(["rye"])
        self.assertEqual(len(self.srv.requests), 1)
        self.assertIn("HTTP 401", str(e.exception))

    def test_vectors_of_unequal_length_stop(self):
        from docsearch.apimodels import ApiError
        from docsearch.models import load_embedder

        self.srv.dims = {1: 6}
        with self.assertRaises(ApiError) as e:
            load_embedder(self.cfg).docs(["a", "b"])
        self.assertIn("dimension", str(e.exception))

    def test_key_comes_from_the_named_env_var_only(self):
        from docsearch.models import load_embedder

        with mock.patch.dict(os.environ, {"TEST_EMBED_KEY": "s3cret"}):
            load_embedder(self.cfg).queries(["rye"])
        self.assertEqual(self.srv.requests[0][2].get("Authorization"), "Bearer s3cret")
        load_embedder(self.cfg).queries(["rye"])
        self.assertNotIn("Authorization", self.srv.requests[1][2])

    def test_an_empty_url_stops_with_the_key_name(self):
        from docsearch.models import load_embedder

        self.cfg["models"]["embed_api_url"] = ""
        with self.assertRaises(SystemExit) as e:
            load_embedder(self.cfg)
        self.assertIn("models.embed_api_url", str(e.exception))


class TestRerank(ApiCase):
    def test_results_shape_in_input_order(self):
        from docsearch.models import load_reranker

        self.srv.reverse = True
        rr = load_reranker(self.cfg)
        got = rr.score("rye bread", ["white bread", "rye bread loaf", "cake"])
        self.assertEqual(got, [1.0, 2.0, 0.0])
        body = self.srv.requests[0][1]
        self.assertEqual(body, {"model": "Qwen/Qwen3-Reranker-0.6B", "query": "rye bread",
                                "documents": ["white bread", "rye bread loaf"], "top_n": 2})  # fmt: skip
        self.assertEqual(rr.calls, 3)

    def test_tei_bare_list_shape_and_texts_field(self):
        from docsearch.models import load_reranker

        self.srv.rerank_shape = "tei"
        self.cfg["models"]["rerank_api_field"] = "texts"
        got = load_reranker(self.cfg).score("rye", ["rye", "wheat"])
        self.assertEqual(got, [1.0, 0.0])
        self.assertIn("texts", self.srv.requests[0][1])

    def test_scores_are_cached_apart_from_local_scores(self):
        from docsearch.models import Reranker, load_reranker

        rr = load_reranker(self.cfg)
        local = Reranker(Config(raw={**self.cfg.raw, "models": {**self.cfg["models"], "backend": "local"}},
                                root=self.root))  # fmt: skip
        self.assertNotEqual(rr.cache_path, local.cache_path)
        rr.score("rye", ["rye"])
        rr2 = load_reranker(self.cfg)
        rr2.score("rye", ["rye"])
        self.assertEqual(len(self.srv.requests), 1)

    def test_a_missing_index_stops(self):
        from docsearch.apimodels import ApiError, parse_rerank

        with self.assertRaises(ApiError):
            parse_rerank({"results": [{"index": 0, "relevance_score": 1.0}]}, 2)


class TestQueryVectorCache(ApiCase):
    def test_api_run_does_not_reuse_local_query_vectors(self):
        from docsearch.eval import query_vectors
        from docsearch.manifest import Question

        qs = [Question("q1", "rye", "s", "pdf", "page", {"d#p1": 1})]
        local = Config(raw={**self.cfg.raw, "models": {**self.cfg["models"], "backend": "local"}}, root=self.root)
        fake = mock.Mock(queries=lambda t: np.full((len(t), 4), 7.0, np.float32))
        with mock.patch("docsearch.models.Embedder", return_value=fake):
            np.testing.assert_array_equal(query_vectors(local, qs), np.full((1, 4), 7.0))
        got = query_vectors(self.cfg, qs)
        np.testing.assert_allclose(got[0], unit(fake_vector("Q: rye")), rtol=1e-6)


def write_chunks(root: Path, texts: dict[str, str]) -> None:
    p = root / "work" / "chunks" / "hybrid.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"chunk_id": cid, "doc_id": cid.split(":")[1], "format": "pdf", "source": "s", "pages": [1], "text": t,
             "n_tokens": 1, "meta": {}} for cid, t in texts.items()]  # fmt: skip
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


NO_LOCAL = AssertionError("a local model was loaded")


class TestNoLocalModel(ApiCase):
    """Goal 1: index (what sync runs), search and eval in api mode build no SentenceTransformer or CrossEncoder."""

    def test_index_search_and_eval_use_only_the_server(self):
        from docsearch.eval import run_eval
        from docsearch.index import build_index
        from docsearch.search import search

        write_chunks(self.root, {"hybrid:a:0": "rye sourdough loaves sold in july",
                                 "hybrid:b:0": "paid leave days for a new baker",
                                 "hybrid:c:0": "cafe visitors asked for weekend hours"})  # fmt: skip
        q = {"qid": "q1", "query": "paid leave for a baker", "source": "s", "format": "pdf", "level": "page",
             "gold": {"b#p1": 1}, "meta": {}}  # fmt: skip
        (self.root / "data").mkdir()
        (self.root / "data" / "questions.jsonl").write_text(json.dumps(q) + "\n")
        with (
            mock.patch("sentence_transformers.SentenceTransformer", side_effect=NO_LOCAL),
            mock.patch("sentence_transformers.CrossEncoder", side_effect=NO_LOCAL),
        ):
            stats = build_index(self.cfg, "hybrid")
            hits = search(self.cfg, "paid leave for a baker", k=3)
            res = run_eval(self.cfg, [("api_text", "hybrid", "hybrid", True)], full=True)
        self.assertEqual((stats["rows"], stats["embedder"], stats["device"]), (3, "Qwen/Qwen3-Embedding-0.6B", "api"))
        self.assertEqual(hits[0]["doc_id"], "b")
        self.assertEqual((res[0]["device"], res[0]["recall@10"]), ("api", 1.0))
        self.assertEqual(set(self.srv.paths()), {"/v1/embeddings", "/rerank"})

    def test_api_mode_is_text_only(self):
        env = {"DOCSEARCH_VISUAL_DEFAULT": "pplx-v2-late"}
        local = load_config(self.root / "docsearch.yaml", env={**env, "DOCSEARCH_MODELS_BACKEND": "local"})
        api = load_config(self.root / "docsearch.yaml", env=env)
        self.assertEqual((local["visual"]["default"], api["visual"]["default"]), ("pplx-v2-late", None))

    def test_unknown_backend_stops(self):
        with self.assertRaises(ValueError) as e:
            load_config(self.root / "docsearch.yaml", env={"DOCSEARCH_MODELS_BACKEND": "remote"})
        self.assertIn("models.backend", str(e.exception))


class FakeEmbedder:
    device = "cpu"

    def __init__(self, name, dim):
        self.name, self.dim = name, dim

    def docs(self, texts, on_batch=None):
        return np.ones((len(texts), self.dim), np.float32)


class TestIndexModelGuard(unittest.TestCase):
    """Goal 3: the index records its embedder (stats.json) and dimension (the vector column); another model stops."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        (self.root / "docsearch.yaml").write_text("subset: {enabled: false}\nvisual: {default: null}\n")
        write_chunks(self.root, {"hybrid:a:0": "rye bread", "hybrid:b:0": "wheat bread"})

    def tearDown(self):
        self.td.cleanup()

    def build(self, name, dim):
        from docsearch.index import build_index

        cfg = load_config(self.root / "docsearch.yaml", env={"DOCSEARCH_MODELS_EMBEDDER": name})
        with mock.patch("docsearch.models.Embedder", return_value=FakeEmbedder(name, dim)):
            return build_index(cfg, "hybrid")

    def test_search_with_another_embedder_names_both(self):
        from docsearch.search import Retriever

        self.build("org/model-a", 4)
        cfg = load_config(self.root / "docsearch.yaml", env={"DOCSEARCH_MODELS_EMBEDDER": "org/model-b"})
        with self.assertRaises(SystemExit) as e:
            Retriever(cfg, "hybrid").candidates("rye", np.ones(4, np.float32))
        self.assertIn("org/model-a", str(e.exception))
        self.assertIn("org/model-b", str(e.exception))

    def test_query_vector_of_another_dimension_names_both(self):
        from docsearch.search import Retriever

        self.build("org/model-a", 4)
        cfg = load_config(self.root / "docsearch.yaml", env={"DOCSEARCH_MODELS_EMBEDDER": "org/model-a"})
        with self.assertRaises(SystemExit) as e:
            Retriever(cfg, "hybrid").candidates("rye", np.ones(8, np.float32))
        self.assertIn("4", str(e.exception))
        self.assertIn("8", str(e.exception))

    def test_bm25_alone_needs_no_embedder_match(self):
        from docsearch.search import Retriever

        self.build("org/model-a", 4)
        cfg = load_config(self.root / "docsearch.yaml", env={"DOCSEARCH_MODELS_EMBEDDER": "org/model-b"})
        self.assertTrue(Retriever(cfg, "hybrid").candidates("rye", None, "bm25"))

    def test_adding_to_an_index_with_another_embedder_stops(self):
        self.build("org/model-a", 4)
        write_chunks(self.root, {"hybrid:a:0": "rye bread", "hybrid:c:0": "spelt"})
        with self.assertRaises(SystemExit) as e:
            self.build("org/model-b", 4)
        self.assertIn("org/model-a", str(e.exception))
        self.assertIn("org/model-b", str(e.exception))


if __name__ == "__main__":
    unittest.main()


class TestCliApi(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.yaml = Path(self.td.name) / "docsearch.yaml"
        self.yaml.write_text("models:\n  embedders: {vis: {id: org/vis, kind: visual}}\nvisual: {default: vis}\n")
        self.plans = []

    def tearDown(self):
        self.td.cleanup()

    def eval_cli(self, *args, env=None):
        import contextlib
        import io

        from docsearch import cli

        def fake_eval(cfg, plan, *a, **k):
            self.plans.append(plan)
            return []

        with mock.patch.dict(os.environ, env or {}), mock.patch("docsearch.eval.run_eval", fake_eval), \
                mock.patch("docsearch.report.build_report"), mock.patch("docsearch.cli._quiet_hub"), \
                mock.patch("docsearch.cli.timed", lambda *a, **k: contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()):  # fmt: skip
            return cli.main(["--config", str(self.yaml), "eval", "--full", *args])

    def test_api_hard_eval_is_text_only(self):
        api = {"DOCSEARCH_MODELS_BACKEND": "api", "DOCSEARCH_MODELS_EMBED_API_URL": "http://127.0.0.1:9/v1"}
        self.eval_cli("--questions", "hard", "--reviewed-only", "--name", "g_api_text_hard", env=api)
        self.eval_cli("--questions", "hard", "--reviewed-only", "--name", "local_hard")
        self.assertEqual(self.plans, [[("g_api_text_hard", "hybrid", "hybrid", True)],
                                      [("local_hard", "hybrid", "hybrid_visual_fa", True)]])  # fmt: skip

    def test_name_applies_to_a_single_runs_run(self):
        self.eval_cli("--runs", "baseline_hybrid_rerank", "--name", "g_local_text")
        self.assertEqual(self.plans, [[("g_local_text", "hybrid", "hybrid", True)]])

    def test_visual_with_api_stops(self):
        with self.assertRaises(SystemExit) as e:
            self.eval_cli("--visual", "vis", env={"DOCSEARCH_MODELS_BACKEND": "api"})
        self.assertIn("text-only", str(e.exception))
        self.assertEqual(self.plans, [])

    def test_api_needs_no_cached_embedder_or_reranker(self):
        from argparse import Namespace

        from docsearch.hfcache import models_for

        local = load_config(self.yaml, env={})
        api = load_config(self.yaml, env={"DOCSEARCH_MODELS_BACKEND": "api"})
        self.assertEqual([r.label for r in models_for(local, "search")], ["embedder", "visual vis", "reranker"])
        for cmd in ("search", "sync", "eval", "index"):
            self.assertEqual(models_for(api, cmd, Namespace(gate=True)), [], cmd)
