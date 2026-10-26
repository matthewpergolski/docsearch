"""Embedding + reranking model wrappers (Sentence Transformers), device/dtype picked from config."""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import ui
from .config import Config, pick_device, pick_dtype


def set_threads(device: str) -> None:
    """Use every core for CPU inference (torch sometimes defaults lower inside containers)."""
    if device == "cpu":
        import torch

        torch.set_num_threads(os.cpu_count() or 1)


def seq_limit(want: int, model_max: int | None) -> int:
    """The length to embed at: `want`, held to the model's own limit (a BERT embedder fails past its 512)."""
    return min(want, model_max) if model_max else want


def doc_prompt(prompts: dict) -> dict:
    """The model's own lead-in for a document, when it declares one that is not empty (EmbeddingGemma does;
    Qwen3-Embedding and mxbai declare none, so their documents go in bare, as before)."""
    return {"prompt_name": "document"} if prompts.get("document") else {}


def prompt_kw(prompts: dict, role: str) -> dict:
    """The prompt_name the local path passes to sentence-transformers for a "query" or a "document"."""
    if role == "query":
        return {"prompt_name": "query"} if "query" in prompts else {}
    return doc_prompt(prompts)


def lead_in(st: dict, role: str) -> str:
    """The text sentence-transformers puts before a `role` input on the local path (prompt_kw, else the model's
    default prompt). A model server embeds text as given, so the api client adds this itself.
    st: the model's config_sentence_transformers.json."""
    prompts = st.get("prompts") or {}
    name = prompt_kw(prompts, role).get("prompt_name") or st.get("default_prompt_name")
    return prompts.get(name, "") if name else ""


def st_config(repo: str) -> dict:
    """The model's config_sentence_transformers.json (its declared prompts) from the HF cache or hub, without its
    weights; {} when the repo has no such file. Any other failure (offline and not cached, not a repo) raises."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(repo, "config_sentence_transformers.json")).read_text())
    except RemoteEntryNotFoundError:
        return {}


def api_backend(cfg: Config) -> bool:
    return (cfg.raw.get("models") or {}).get("backend", "local") == "api"


_KEPT: dict[str, object] | None = None  # None: build each model on each call (a CLI run searches once)


def keep_loaded(on: bool) -> None:
    """A long-lived process (`docsearch serve`) keeps each model it built, keyed by kind and the models settings,
    so a search after the first loads nothing."""
    global _KEPT
    _KEPT = {} if on else None


class Superseded(Exception):
    """A newer search arrived; this one stopped (docsearch serve)."""


_STOP = None  # docsearch serve: says whether a newer search has arrived; None in a CLI run


def stop_when(check) -> None:
    """Set (or clear, with None) the check a long step runs between its parts to stop for a newer search."""
    global _STOP
    _STOP = check


def stop_point() -> None:
    """Between the parts of a search: raise Superseded when a newer search has arrived (docsearch serve)."""
    if _STOP is not None and _STOP():
        raise Superseded


_ON_PAGE = None  # docsearch serve: hears the pages the page reranker will read, and each one it has read


def on_page(hook) -> None:
    """Set (or clear) hook(pages, done): called when the page reranker starts reading pages and after each one."""
    global _ON_PAGE
    _ON_PAGE = hook


def keeping() -> bool:
    return _KEPT is not None


def _kept(kind: str, cfg: Config, build):
    if _KEPT is None:
        return build()
    key = f"{kind}:{json.dumps(cfg['models'], sort_keys=True, default=str)}"
    if key not in _KEPT:
        _KEPT[key] = build()
    return _KEPT[key]


def load_embedder(cfg: Config):
    """The one place that picks the embedder: local (sentence-transformers) or api (models.backend)."""
    if api_backend(cfg):
        from .apimodels import ApiEmbedder

        return _kept("embed-api", cfg, lambda: ApiEmbedder(cfg))
    return _kept("embed", cfg, lambda: Embedder(cfg))


def load_reranker(cfg: Config):
    """The one place that picks the text reranker: local (cross-encoder) or api (models.backend)."""
    if api_backend(cfg):
        from .apimodels import ApiReranker

        return _kept("rerank-api", cfg, lambda: ApiReranker(cfg))
    return _kept("rerank", cfg, lambda: Reranker(cfg))


def load_page_reranker(cfg: Config, name: str):
    return _kept(f"page:{name}", cfg, lambda: PageReranker(cfg, name))


def embed_cache_tag(cfg: Config) -> str:
    """What a cached query vector depends on besides the text: the model, and for api also the server and the
    name sent, so an api run never reads local vectors or the other way round."""
    m = cfg["models"]
    if not api_backend(cfg):
        return m["embedder"]
    from .apimodels import endpoint

    ep = endpoint(cfg, "embed")
    return f"{m['embedder']}|api|{ep.url}|{ep.model}"


class Embedder:
    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        m = cfg["models"]
        self.device = pick_device(m["device"])
        set_threads(self.device)
        dtype = pick_dtype(self.device, m["embed_dtype"], "embed")
        self.name = m["embedder"]
        with ui.loading("embedder", self.name):
            self.model = SentenceTransformer(self.name, device=self.device, model_kwargs={"torch_dtype": dtype})
        # Chunks are capped at chunking.max_tokens; the extra margin only covers the file/heading prefix.
        self.model.max_seq_length = seq_limit(int(cfg["chunking"]["max_tokens"]) + 32, self.model.max_seq_length)
        self.batch_size = int(m["embed_batch_size"])
        self.dim = self.model.get_embedding_dimension()

    def docs(self, texts: list[str], on_batch=None) -> np.ndarray:
        """Embed in batch_size slices; on_batch(n) is called after each slice (drives our progress bar,
        sentence-transformers' own tqdm bar is off)."""
        out = []
        kw = doc_prompt(getattr(self.model, "prompts", {}) or {})
        for i in range(0, len(texts), self.batch_size):
            part = texts[i : i + self.batch_size]
            out.append(
                self.model.encode(
                    part,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    **kw,
                ).astype(np.float32)
            )
            if on_batch:
                on_batch(len(part))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        kw = prompt_kw(getattr(self.model, "prompts", {}) or {}, "query")
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kw,
        ).astype(np.float32)


def rerank_batch_size(cfg: Config) -> int:
    """models.rerank_batch_sizes[alias or HF id of models.reranker], else models.rerank_batch_size."""
    m = cfg["models"]
    per = m.get("rerank_batch_sizes") or {}
    ids = {alias: rid for alias, rid in (m.get("rerankers") or {}).items()}
    for key, n in per.items():
        if key == m["reranker"] or ids.get(key) == m["reranker"]:
            return int(n)
    return int(m["rerank_batch_size"])


class Reranker:
    """Cross-encoder with an on-disk score cache keyed by (model, max_length, query, passage)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        m = cfg["models"]
        self.name = m["reranker"]
        self.max_length = int(m["rerank_max_length"])
        self.batch_size = rerank_batch_size(cfg)
        self._model = None
        self.cache_path = cfg.path("cache_dir") / f"rerank_{self.cache_name()}.jsonl"
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    k, v = json.loads(line)
                    self.cache[k] = v
        self.calls = 0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = pick_device(self.cfg["models"]["device"])
            set_threads(device)
            dtype = pick_dtype(device, self.cfg["models"]["rerank_dtype"], "rerank")
            with ui.loading("reranker", self.name):
                self._model = CrossEncoder(
                    self.name, device=device, max_length=self.max_length, model_kwargs={"torch_dtype": dtype}
                )
        return self._model

    def cache_name(self) -> str:
        return f"{self.name.replace('/', '__')}_{self.max_length}"

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        return self.model.predict([(query, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False)

    @staticmethod
    def key(query: str, passage: str) -> str:
        return hashlib.sha1(f"{query}\x00{passage}".encode()).hexdigest()

    def score(self, query: str, passages: list[str]) -> list[float]:
        keys = [self.key(query, p) for p in passages]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            scores = self._predict(query, [passages[i] for i in todo])
            self.calls += len(todo)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with Path(self.cache_path).open("a") as f:
                for i, s in zip(todo, scores, strict=True):
                    self.cache[keys[i]] = float(s)
                    f.write(json.dumps([keys[i], float(s)]) + "\n")
        return [self.cache[k] for k in keys]


def named_chat_template(model, repo: str) -> None:
    """With no network, transformers lists no `additional_chat_templates` when the cached snapshot lacks any file
    of the repo, and the template name the model asks for (Qwen3-VL-Reranker: "reranker") is then rendered as the
    template itself, so the pair never reaches the prompt. Read the named template from the cache in that case."""
    from huggingface_hub import try_to_load_from_cache

    first = model[0]
    want = ((getattr(first, "processing_kwargs", None) or {}).get("chat_template") or {}).get("chat_template")
    have = first.processor.chat_template
    if not want or (isinstance(have, dict) and want in have):
        return
    path = try_to_load_from_cache(repo, f"additional_chat_templates/{want}.jinja")
    if isinstance(path, str):
        base = have if isinstance(have, dict) else {"default": have}
        first.processor.chat_template = {**base, want: Path(path).read_text()}


def fixed_modules_dir(repo: str, root: Path) -> str:
    """Qwen3-VL-Reranker-8B (revision b212dc8c) names its score head `1_CausalScoreHead` in modules.json but ships
    the folder as `1_LogitScore`, so sentence-transformers builds the head with no config and fails. When a module
    folder is missing and exactly one folder with the same index prefix exists, return a copy of the cached snapshot
    (links) under root whose modules.json names that folder. Otherwise return the repo id unchanged."""
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo, "modules.json")
    if not isinstance(p, str):
        return repo
    snap = Path(p).parent
    mods = json.loads(Path(p).read_text())
    changed = False
    for m in mods:
        if m.get("path") and not (snap / m["path"]).is_dir():
            alt = [d.name for d in snap.iterdir() if d.is_dir() and d.name.startswith(f"{m['idx']}_")]
            if len(alt) != 1:
                return repo
            m["path"], changed = alt[0], True
    if not changed:
        return repo
    out = root / repo.replace("/", "__") / snap.name
    out.mkdir(parents=True, exist_ok=True)
    for e in snap.iterdir():
        if e.name != "modules.json" and not (out / e.name).exists():
            (out / e.name).symlink_to(e)
    (out / "modules.json").write_text(json.dumps(mods, indent=2))
    return str(out)


class FeatureCache:
    """The vision features of the last `size` pages a vision reranker read. Scoring a page again (a new query over
    the same top pages, as when a question is refined) then skips the vision tower, about half the time of a page.
    The language model gets the same tensors, so the scores do not change (Qwen3-VL-Reranker-2B on MPS: 6 of 6
    scores equal to the last digit, 1.0 s a page without, 0.45 s with). About 14 MB a page in float16."""

    def __init__(self, inner, size: int):
        self._run = inner.get_image_features
        self.size = size
        self.kept: OrderedDict[str, object] = OrderedDict()
        self.unit: str | None = None
        inner.get_image_features = self._features

    @contextmanager
    def page(self, unit: str):
        self.unit = unit
        try:
            yield
        finally:
            self.unit = None

    def _features(self, pixel_values, image_grid_thw=None, **kw):
        if self.unit is None:
            return self._run(pixel_values, image_grid_thw, **kw)
        if self.unit in self.kept:
            self.kept.move_to_end(self.unit)
            return self.kept[self.unit]
        out = self._run(pixel_values, image_grid_thw, **kw)
        # the model reads only these two; the full vision hidden states would cost several times the memory
        slim = SimpleNamespace(pooler_output=out.pooler_output, deepstack_features=out.deepstack_features)
        self.kept[self.unit] = slim
        if len(self.kept) > self.size:
            self.kept.popitem(last=False)
        return slim


class PageReranker:
    """M16: a vision-language cross-encoder that scores (query, rendered page image) pairs, with an on-disk
    score cache keyed by (model, max_side, query, page unit). Pages come from the render step's pages.jsonl."""

    PROMPT = "Retrieve images or text relevant to the user's query."  # the model card's multimodal instruction
    FEATURES_KEPT = 64  # pages whose vision features stay in memory (FeatureCache), about 0.9 GB

    def __init__(self, cfg: Config, name: str):
        from .visual import read_pages

        self.cfg = cfg
        self.name = (cfg["models"].get("page_rerankers") or {}).get(name, name)
        self.max_side = int(cfg["visual"]["max_side"])
        self.paths = {p["unit"]: p["path"] for p in read_pages(cfg)}
        self._model = None
        self.features: FeatureCache | None = None
        safe = self.name.replace("/", "__")
        self.cache_path = cfg.path("cache_dir") / f"pagererank_{safe}_{self.max_side}.jsonl"
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    k, v = json.loads(line)
                    self.cache[k] = v
        self.calls = 0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = pick_device(self.cfg["models"]["device"])
            dtype = pick_dtype(device, self.cfg["models"]["rerank_dtype"], "rerank")
            with ui.loading("page reranker", self.name):
                src = fixed_modules_dir(self.name, self.cfg.path("cache_dir") / "fixed_snapshots")
                self._model = CrossEncoder(src, device=device, model_kwargs={"torch_dtype": dtype})
            named_chat_template(self._model, self.name)
            inner = [
                mod for mod in self._model.modules() if hasattr(mod, "visual") and hasattr(mod, "get_image_features")
            ]
            if inner and hasattr(inner[-1], "language_model"):  # the model whose forward calls get_image_features
                self.features = FeatureCache(inner[-1], self.FEATURES_KEPT)
        return self._model

    def _load(self, unit: str):
        from PIL import Image

        im = Image.open(self.paths[unit]).convert("RGB")
        s = self.max_side / max(im.size)
        return im.resize((round(im.width * s), round(im.height * s))) if s < 1 else im

    def score(self, query: str, units: list[str]) -> list[float | None]:
        """One score per unit; None for a unit with no rendered page (it keeps its place)."""
        keys = [Reranker.key(query, u) for u in units]
        todo = [i for i, (k, u) in enumerate(zip(keys, units, strict=True)) if k not in self.cache and u in self.paths]
        reads = [u for u in units if u in self.paths]
        done = [u for k, u in zip(keys, units, strict=True) if k in self.cache and u in self.paths]
        if todo and _ON_PAGE is not None:
            _ON_PAGE(reads, done)
        for i in todo:  # one page per call: bounded memory on MPS
            stop_point()
            model = self.model
            with self.features.page(units[i]) if self.features else nullcontext():
                s = float(
                    model.predict([(query, self._load(units[i]))], prompt=self.PROMPT, show_progress_bar=False)[0]
                )
            self.calls += 1
            self.cache[keys[i]] = s
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with Path(self.cache_path).open("a") as f:
                f.write(json.dumps([keys[i], s]) + "\n")
            done.append(units[i])
            if _ON_PAGE is not None:
                _ON_PAGE(reads, done)
        return [self.cache.get(k) for k in keys]
