"""A stdlib model server for models.backend: api (Phase G): POST /v1/embeddings (OpenAI shape) and POST /rerank
(Cohere/Jina/vLLM shape, also reads TEI's "texts").

  uv run python scripts/model_server.py [--port 8088] [--port-file F] [--log F.jsonl]
      serves models.embedder and models.reranker from the local HF cache, on the device config.pick_device picks,
      with the local settings (dtype, max lengths, batch sizes). It embeds each text exactly as given, as a real
      server does: the docsearch client adds the query lead-in.
  uv run python scripts/model_server.py --fake ...
      deterministic stand-ins (hashed bag of words, word overlap); no model, no torch. Used by scripts/verify.sh.

It answers one request at a time (one model copy, no locking). A request that names another model gets 404, as vLLM
does. Each request is logged as one JSON line (--log) with its route, item count and seconds.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

FAKE_DIM = 64


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class FakeModels:
    """Deterministic: a text's vector counts its words hashed into FAKE_DIM buckets; a passage's score is the share of
    the query's words it holds."""

    embedder = reranker = None  # any model name is accepted

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * FAKE_DIM
            for w in words(t):
                v[int(hashlib.sha1(w.encode()).hexdigest(), 16) % FAKE_DIM] += 1.0
            v[0] += 1e-3  # an empty text still gets a vector that can be normalised
            out.append(v)
        return out

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        q = set(words(query))
        return [len(q & set(words(d))) / (len(q) or 1) for d in docs]


class LocalModels:
    """The docsearch local models with the local settings (models.Embedder / models.Reranker), loaded once."""

    def __init__(self):
        from docsearch.config import load_config
        from docsearch.models import Embedder, Reranker

        cfg = load_config()
        self.emb = Embedder(cfg)
        self.emb.model.default_prompt_name = None  # embed the text as given; the client adds any lead-in
        self.rr = Reranker(cfg)
        self.rr.model  # noqa: B018 - load now, not on the first request
        self.embedder, self.reranker = self.emb.name, self.rr.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        v = self.emb.model.encode(texts, batch_size=self.emb.batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)  # fmt: skip
        return v.astype("float32").tolist()

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.rr._predict(query, docs)]  # the cross-encoder itself; no score cache


def make_handler(models, log_path: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code: int, obj) -> None:
            raw = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            t0 = time.time()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            route = self.path.rstrip("/")
            want = {"/v1/embeddings": models.embedder, "/rerank": models.reranker, "/v1/rerank": models.reranker}
            if route not in want:
                return self.reply(404, {"error": f"no route {self.path}"})
            if want[route] and body.get("model") != want[route]:
                return self.reply(404, {"error": f"model {body.get('model')!r} is not served here ({want[route]})"})
            if route == "/v1/embeddings":
                texts = [body["input"]] if isinstance(body["input"], str) else body["input"]
                vecs = models.embed(texts)
                out = {"object": "list", "model": body.get("model"),
                       "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)]}  # fmt: skip
                n = len(texts)
            else:
                docs = body.get("documents") or body.get("texts") or []
                scores = models.rerank(body["query"], docs)
                ranked = sorted(range(len(docs)), key=lambda i: -scores[i])[: body.get("top_n") or len(docs)]
                out = {
                    "model": body.get("model"),
                    "results": [{"index": i, "relevance_score": scores[i]} for i in ranked],
                }
                n = len(docs)
            self.reply(200, out)
            if log_path:
                with log_path.open("a") as f:
                    f.write(json.dumps({"route": route, "n": n, "seconds": round(time.time() - t0, 4)}) + "\n")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8088, help="0 = any free port (see --port-file)")
    ap.add_argument("--port-file", type=Path, help="write the bound port here once the server listens")
    ap.add_argument("--log", type=Path, help="append one JSON line per request")
    ap.add_argument("--fake", action="store_true", help="deterministic stand-ins; no model, no torch")
    a = ap.parse_args()
    models = FakeModels() if a.fake else LocalModels()
    httpd = HTTPServer(("127.0.0.1", a.port), make_handler(models, a.log))
    port = httpd.server_address[1]
    if a.port_file:
        a.port_file.write_text(str(port))
    print(f"model_server: {'fake' if a.fake else f'{models.embedder} + {models.reranker}'} on "
          f"http://127.0.0.1:{port} (/v1/embeddings, /rerank)", file=sys.stderr, flush=True)  # fmt: skip
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()


if __name__ == "__main__":
    main()
