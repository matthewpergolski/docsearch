"""models.backend: api (Phase G). The embedder and the text reranker run on an HTTP server instead of in this process.

Wire shapes (the parsers below accept exactly these):
  POST {embed_api_url}/embeddings   {"model", "input": [text, ...]}
       -> {"data": [{"index": i, "embedding": [float, ...]}, ...]}            OpenAI, vLLM, TEI, Infinity, Ollama
  POST {rerank_api_url}/rerank      {"model", "query", "documents": [text, ...], "top_n": n}
       -> {"results": [{"index": i, "relevance_score": float}, ...]}          Cohere, Jina, vLLM, Infinity
       -> [{"index": i, "score": float}, ...]                                 TEI (passages field "texts")

The client adds the model's query lead-in (models.lead_in, or models.embed_api_query_prefix) and normalises the
vectors, as the local path does. The key
is read from the env var the config names; it is never in the config file and never logged.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import models
from .config import Config
from .models import Reranker, lead_in

log = logging.getLogger("docsearch.api")
RETRY_WAIT_S = 1.0  # first wait between attempts; doubles each time

# part -> (models.<identity key>, route under the part's base URL)
ROUTES = {"embed": ("embedder", "/embeddings"), "rerank": ("reranker", "/rerank")}


class ApiError(RuntimeError):
    """The model server failed, or answered in a shape the client does not accept."""


@dataclass(frozen=True)
class Endpoint:
    part: str  # "embed" | "rerank"
    url: str  # the full route URL
    model: str  # the name sent to the server
    key_env: str  # env var that holds the bearer key; "" = no key
    timeout_s: float
    retries: int
    batch_size: int


def endpoint(cfg: Config, part: str) -> Endpoint:
    m = cfg["models"]
    ident, route = ROUTES[part]
    base = (m.get(f"{part}_api_url") or "").rstrip("/")
    if not base:
        raise SystemExit(f"models.backend is api but models.{part}_api_url is empty: set it to the model server's "
                         "base URL (e.g. https://models.example.internal/v1)")  # fmt: skip
    return Endpoint(part, base + route, m.get(f"{part}_api_model") or m[ident], m.get(f"{part}_api_key_env") or "",
                    float(m["api_timeout_s"]), int(m["api_retries"]), int(m["api_batch_size"]))  # fmt: skip


def post_json(ep: Endpoint, payload: dict) -> Any:
    """POST with retries on 5xx, 429, timeouts and connection errors; each failed attempt is logged."""
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(ep.key_env) if ep.key_env else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    attempts = ep.retries + 1
    why = ""
    for i in range(1, attempts + 1):
        req = urllib.request.Request(ep.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=ep.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            why = f"HTTP {e.code}: {e.read(300).decode(errors='replace')}"
            if e.code < 500 and e.code != 429:
                raise ApiError(f"{ep.part} POST {ep.url}: {why}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            timed_out = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
            why = f"timed out after {ep.timeout_s:g}s" if timed_out else f"{type(e).__name__}: {e}"
        if i < attempts:
            wait = RETRY_WAIT_S * 2 ** (i - 1)
            log.warning("%s POST %s attempt %d/%d failed: %s; retry in %.1fs", ep.part, ep.url, i, attempts, why, wait)
            time.sleep(wait)
    log.error("%s POST %s failed after %d attempts: %s", ep.part, ep.url, attempts, why)
    raise ApiError(f"{ep.part} POST {ep.url} failed after {attempts} attempts: {why}")


def parse_embeddings(body: Any, n: int) -> np.ndarray:
    """(n, dim) float32 in input order, by each item's "index"."""
    try:
        by = {int(d["index"]): d["embedding"] for d in body["data"]}
    except (KeyError, TypeError, ValueError):
        raise ApiError(f"embeddings: not an OpenAI embeddings response: {str(body)[:200]}") from None
    if sorted(by) != list(range(n)):
        raise ApiError(f"embeddings: asked for {n} vectors, got indexes {sorted(by)[:10]}")
    dims = {len(v) for v in by.values()}
    if len(dims) != 1:
        raise ApiError(f"embeddings: vectors of unequal dimension {sorted(dims)} in one response")
    return np.asarray([by[i] for i in range(n)], dtype=np.float32)


def parse_rerank(body: Any, n: int) -> list[float]:
    """One score per passage in input order, from {"results": [{index, relevance_score}]} or [{index, score}]."""
    try:
        items = body["results"] if isinstance(body, dict) else body
        by = {int(r["index"]): float(r["relevance_score"] if "relevance_score" in r else r["score"]) for r in items}
    except (KeyError, TypeError, ValueError):
        raise ApiError(f"rerank: not a rerank response: {str(body)[:200]}") from None
    if sorted(by) != list(range(n)):
        raise ApiError(f"rerank: asked for {n} scores, got indexes {sorted(by)[:10]}")
    return [by[i] for i in range(n)]


class ApiEmbedder:
    """The local Embedder's interface (name, dim, device, docs, queries) over {embed_api_url}/embeddings."""

    device = "api"

    def __init__(self, cfg: Config):
        self.name = cfg["models"]["embedder"]  # the model identity the index records; the server may call it else
        self.ep = endpoint(cfg, "embed")
        self.lead = self._lead_ins(cfg["models"].get("embed_api_query_prefix"))
        self._dim: int | None = None
        self.requests = 0

    def _lead_ins(self, query_prefix: str | None) -> dict[str, str]:
        """role -> lead-in. models.embed_api_query_prefix set: it is the query lead-in and documents get none.
        Else the model's sentence-transformers config on the HF hub; a model the hub does not have, or no network
        and no cache, gives no lead-in and a warning (query vectors may then be worse, not wrong in shape)."""
        if query_prefix is not None:
            return {"query": query_prefix, "document": ""}
        try:
            st = models.st_config(self.name)
        except Exception as e:  # noqa: BLE001 - any hub failure: not found, offline, a name that is not a repo id
            log.warning("embed: cannot read the lead-ins of %s from the Hugging Face hub (%s: %s); embedding with "
                        "no lead-in. Set models.embed_api_query_prefix to the model's query lead-in (\"\" for none).",
                        self.name, type(e).__name__, e)  # fmt: skip
            st = {}
        return {role: lead_in(st, role) for role in ("query", "document")}

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._embed(["dimension probe"])
        return self._dim

    def _embed(self, texts: list[str], on_batch=None) -> np.ndarray:
        out, t0, bs = [], time.time(), self.ep.batch_size
        for i in range(0, len(texts), bs):
            part = texts[i : i + bs]
            v = parse_embeddings(post_json(self.ep, {"model": self.ep.model, "input": part}), len(part))
            self.requests += 1
            if self._dim is None:
                self._dim = v.shape[1]
            elif v.shape[1] != self._dim:
                raise ApiError(f"embeddings: dimension changed from {self._dim} to {v.shape[1]} between requests")
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            out.append(v / np.where(norm == 0, 1, norm))
            if on_batch:
                on_batch(len(part))
        log.info("embed: %d texts in %d requests of up to %d, %.1fs (%s)", len(texts), -(-len(texts) // bs), bs,
                 time.time() - t0, self.ep.url)  # fmt: skip
        return np.concatenate(out) if out else np.zeros((0, self._dim or 0), np.float32)

    def docs(self, texts: list[str], on_batch=None) -> np.ndarray:
        return self._embed([self.lead["document"] + t for t in texts], on_batch)

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.lead["query"] + t for t in texts])


class ApiReranker(Reranker):
    """The local Reranker (score cache, call count) with its scores from {rerank_api_url}/rerank."""

    def __init__(self, cfg: Config):
        self.ep = endpoint(cfg, "rerank")
        self.field = cfg["models"]["rerank_api_field"]
        super().__init__(cfg)

    def cache_name(self) -> str:
        # The server's truncation and dtype are not the local ones, so its scores never share the local cache file.
        return f"api_{self.ep.model.replace('/', '__')}_{hashlib.sha1(self.ep.url.encode()).hexdigest()[:8]}"

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        out, bs = [], self.ep.batch_size
        for i in range(0, len(passages), bs):
            part = passages[i : i + bs]
            body = {"model": self.ep.model, "query": query, self.field: part, "top_n": len(part)}
            out += parse_rerank(post_json(self.ep, body), len(part))
        return out
