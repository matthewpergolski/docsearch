"""Which Hugging Face models a command will load, and whether each is fully in the local HF cache.

Deterministic: the answer depends only on the config, the command's flags and the files in the cache. A model
counts as cached when its config.json and a weights file exist in the snapshot for the pinned revision (or
the commit `refs/main` points to when no revision is pinned).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import Config

WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")


@dataclass(frozen=True)
class ModelRef:
    label: str  # e.g. "embedder", "reranker", "visual pplx-v2-late"
    repo: str
    revision: str | None = None

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision[:12]}" if self.revision else self.repo


def hub_cache_dir() -> Path:
    """Resolved at call time (not import time) so HF_HUB_CACHE / HF_HOME changes are honoured."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface")
    return Path(home).expanduser() / "hub"


def is_cached(ref: ModelRef, cache_dir: Path | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache

    cd = str(cache_dir or hub_cache_dir())

    def has(name: str) -> bool:
        return isinstance(try_to_load_from_cache(ref.repo, name, cache_dir=cd, revision=ref.revision), str)

    return has("config.json") and any(has(w) for w in WEIGHTS)


def _spec(cfg: Config, alias: str) -> ModelRef:
    s = (cfg["models"].get("embedders") or {})[alias]
    return ModelRef(f"{s.get('kind', 'embedder')} {alias}", s["id"], s.get("revision"))


def _reranker(cfg: Config, name: str | None) -> ModelRef:
    m = cfg["models"]
    name = name or m["reranker"]
    return ModelRef("reranker", (m.get("rerankers") or {}).get(name, name))


def models_for(cfg: Config, cmd: str, a=None) -> list[ModelRef]:
    """The models `cmd` (with argparse namespace `a`) will actually load."""
    m = cfg["models"]
    g = (lambda k, d=None: getattr(a, k, d)) if a is not None else (lambda k, d=None: d)
    emb = ModelRef("embedder", m["embedder"], m.get("embedder_revision"))
    vis_default = cfg["visual"].get("default")
    vis = [_spec(cfg, vis_default)] if vis_default and vis_default in (m.get("embedders") or {}) else []
    # The vision page reranker (M16b) runs after the text reranker wherever page pictures are searched.
    pr_name = g("page_reranker") or cfg["search"].get("page_reranker") or ""
    page_rr = [ModelRef("page reranker", (m.get("page_rerankers") or {}).get(pr_name, pr_name))] if pr_name else []
    out: list[ModelRef] = []
    if cmd == "chunk":
        out = [emb]  # tokenizer of the embedder
    elif cmd == "index":
        if g("visual"):
            out = [_spec(cfg, g("visual"))]
        elif g("embedder"):
            out = [_spec(cfg, g("embedder"))]
        else:
            out = [emb]
    elif cmd == "eval":
        out = [_spec(cfg, g("embedder"))] if g("embedder") else [emb]
        if g("visual"):
            out.append(_spec(cfg, g("visual")))
        if not g("no_rerank", False):
            out.append(_reranker(cfg, g("reranker")))
            if g("visual"):
                out += page_rr
    elif cmd in ("search", "answer"):
        out = [emb, *vis]
        if not g("no_rerank", False):
            out.append(_reranker(cfg, g("reranker")))
            if vis:
                out += page_rr
    elif cmd == "sync":
        if not g("dry_run", False):
            out = [emb, *vis]
            if g("gate", False):
                out.append(_reranker(cfg, None))
        else:
            out = [emb, *vis, _reranker(cfg, None)]  # report what a real sync --gate would need
    elif cmd == "all":
        vis_a = [_spec(cfg, g("visual"))] if g("visual") else vis
        out = [emb, *vis_a, _reranker(cfg, None), *(page_rr if vis_a else [])]
    if m.get("backend") == "api":  # the embedder and text reranker run on the model server (Phase G)
        out = [r for r in out if r.label not in ("embedder", "reranker")]
    seen, uniq = set(), []
    for r in out:
        if (r.repo, r.revision) not in seen:
            seen.add((r.repo, r.revision))
            uniq.append(r)
    return uniq


def missing(refs: list[ModelRef], cache_dir: Path | None = None) -> list[ModelRef]:
    return [r for r in refs if not is_cached(r, cache_dir)]
