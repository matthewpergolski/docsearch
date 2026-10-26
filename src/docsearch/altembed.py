"""Challenger embedders kept outside LanceDB: contextual (pplx-embed-context) and multi-vector (LateOn).

The BM25 leg still comes from the LanceDB table, so only the vector leg changes. Vectors are stored as
resumable numpy shards under work/index/emb_<alias>_<variant>/ and searched exactly (brute force), which
matches the flat (no ANN index) LanceDB search used for the baseline embedder.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import ui
from .chunk import Chunk, chunks_path, read_chunks
from .config import Config, pick_device
from .models import set_threads


def embedder_spec(cfg: Config, alias: str) -> dict:
    specs = cfg["models"].get("embedders") or {}
    if alias not in specs:
        raise SystemExit(f"unknown embedder alias {alias!r}; known: {', '.join(specs) or 'none'}")
    return {"alias": alias, **specs[alias]}


def store_dir(cfg: Config, alias: str, variant: str) -> Path:
    return cfg.path("index_dir") / f"emb_{alias}_{variant}"


# ------------------------------------------------------------------ encoders
class ContextualEncoder:
    """pplx-embed-context: chunks of one document are encoded together (late chunking) in windows."""

    kind = "contextual"

    def __init__(self, spec: dict, cfg: Config):
        import torch
        from transformers import AutoModel

        self.device = pick_device(cfg["models"]["device"])
        set_threads(self.device)
        self.model = AutoModel.from_pretrained(spec["id"], revision=spec.get("revision"), trust_remote_code=True)
        self.model.to(self.device).eval()
        self.tok = self.model.tokenizer
        self.window_tokens = int(spec.get("window_tokens", 4096))
        self.chunk_tokens = int(cfg["chunking"]["max_tokens"]) + 32
        self._torch = torch

    def _clip(self, text: str) -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= self.chunk_tokens else self.tok.decode(ids[: self.chunk_tokens])

    def windows(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Consecutive chunks of the same doc, packed up to window_tokens."""
        out, cur, n, doc = [], [], 0, None
        for c in chunks:
            t = min(c.n_tokens or len(c.text) // 3, self.chunk_tokens) + 1
            if cur and (c.doc_id != doc or n + t > self.window_tokens):
                out.append(cur)
                cur, n = [], 0
            cur.append(c)
            n += t
            doc = c.doc_id
        if cur:
            out.append(cur)
        return out

    def encode_window(self, texts: list[str]) -> np.ndarray:
        with self._torch.inference_mode():
            e = self.model.encode([[self._clip(t) for t in texts]], batch_size=1, show_progress_bar=False)[0]
        e = np.asarray(e, dtype=np.float32)
        assert e.shape[0] == len(texts), f"late-chunking split mismatch: {e.shape[0]} != {len(texts)}"
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

    def queries(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode_window([t])[0] for t in texts]


class MultiVectorEncoder:
    """LateOn (PyLate ColBERT) re-implemented on plain transformers: [Q]/[D] prefix token after [CLS],
    three residual Dense projections to 128-d, punctuation skiplist on documents, L2-normalised tokens."""

    kind = "multivector"

    def __init__(self, spec: dict, cfg: Config):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = pick_device(cfg["models"]["device"])
        set_threads(self.device)
        path = Path(snapshot_download(spec["id"], revision=spec.get("revision")))
        st = json.loads((path / "config_sentence_transformers.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.denses = []
        for m in json.loads((path / "modules.json").read_text())[1:]:
            conf = json.loads((path / m["path"] / "config.json").read_text())
            w = load_file(str(path / m["path"] / "model.safetensors"))
            self.denses.append((w["linear.weight"].to(self.device), w.get("residual.weight"), conf["use_residual"]))
        self.q_id = self.tok.convert_tokens_to_ids(st["query_prefix"])
        self.d_id = self.tok.convert_tokens_to_ids(st["document_prefix"])
        self.q_len = int(st["query_length"])
        self.d_len = int(spec.get("document_length", int(cfg["chunking"]["max_tokens"]) + 32))
        self.skip = {self.tok.convert_tokens_to_ids(w) for w in st["skiplist_words"]}
        self.batch = int(spec.get("batch_size", 16))

    def _encode(self, texts: list[str], query: bool) -> list[np.ndarray]:
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(
                texts[i : i + self.batch],
                max_length=(self.q_len if query else self.d_len) - 1,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            ids, att = enc["input_ids"], enc["attention_mask"]
            pre = torch.full((ids.size(0), 1), self.q_id if query else self.d_id, dtype=ids.dtype)
            ids = torch.cat([ids[:, :1], pre, ids[:, 1:]], 1)
            att = torch.cat([att[:, :1], torch.ones_like(pre), att[:, 1:]], 1)
            with torch.inference_mode():
                h = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).last_hidden_state
                for w, r, use_res in self.denses:
                    p = h @ w.T
                    if use_res:
                        p = p + (h @ r.to(self.device).T if r is not None else h)
                    h = p
                h = torch.nn.functional.normalize(h, p=2, dim=-1).float().cpu()
            mask = att.bool()
            if not query:
                mask &= ~torch.isin(ids, torch.tensor(sorted(self.skip)))
            out += [h[j][mask[j]].numpy().astype(np.float16) for j in range(h.size(0))]
        return out

    def docs(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(texts, query=False)

    def queries(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(texts, query=True)


def load_encoder(cfg: Config, alias: str):
    spec = embedder_spec(cfg, alias)
    kind = spec.get("kind", "dense")
    if kind == "contextual":
        return ContextualEncoder(spec, cfg)
    if kind == "multivector":
        return MultiVectorEncoder(spec, cfg)
    if kind == "visual":
        from .visual import PageVectorEncoder, VisualEncoder

        return (PageVectorEncoder if spec.get("vectors") == "single" else VisualEncoder)(spec, cfg)
    raise SystemExit(f"embedder {alias!r} has kind {kind!r}; dense embedders use the LanceDB index (models.embedder)")


# ------------------------------------------------------------------ store (resumable shards)
def _done_ids(d: Path) -> set[str]:
    done: set[str] = set()
    for p in sorted(d.glob("shard_*.json")):
        done.update(json.loads(p.read_text())["chunk_ids"])
    return done


def has_vectors(d: Path) -> bool:
    """The store holds at least one row (what VectorStore needs); stats.json alone is written for 0 rows too."""
    return any(json.loads(p.read_text())["chunk_ids"] for p in d.glob("shard_*.json"))


def _save_shard(d: Path, ids: list[str], vecs: list[np.ndarray]) -> None:
    # Next free number (not a count: shards can be removed by drop_ids). Files are written to a temp name and
    # renamed, never rewritten in place, so a hard-linked live snapshot of the store is never modified.
    nums = [int(p.stem.split("_")[1]) for p in d.glob("shard_*.json")]
    stem = d / f"shard_{(max(nums) + 1) if nums else 0:05d}"
    lens = np.array([v.shape[0] if v.ndim == 2 else 1 for v in vecs], dtype=np.int64)
    with open(f"{stem}.npy.tmp", "wb") as f:
        np.save(f, np.concatenate([v.reshape(-1, v.shape[-1]) for v in vecs]))
    Path(f"{stem}.npy.tmp").replace(f"{stem}.npy")
    tmp = Path(f"{stem}.json.tmp")
    tmp.write_text(json.dumps({"chunk_ids": ids, "lens": lens.tolist()}))
    tmp.rename(f"{stem}.json")  # the json is the commit marker; an orphan .npy is overwritten on resume


def drop_ids(d: Path, drop: set[str]) -> int:
    """Remove ids from a shard store (deleted or changed documents). Crash-safe: the old shard's json (its commit
    marker) goes first, then the kept rows are written as a new shard; an interrupted drop only loses rows,
    which the next build re-embeds as missing."""
    removed = 0
    for p in sorted(d.glob("shard_*.json")):
        meta = json.loads(p.read_text())
        hit = [i for i, x in enumerate(meta["chunk_ids"]) if x in drop]
        if not hit:
            continue
        mat = np.load(p.with_suffix(".npy"))
        starts = np.concatenate([[0], np.cumsum(meta["lens"])])
        keep = [i for i in range(len(meta["chunk_ids"])) if i not in set(hit)]
        rows = [mat[starts[i] : starts[i + 1]] for i in keep]
        p.unlink()  # uncommit first
        p.with_suffix(".npy").unlink(missing_ok=True)  # before saving: the new shard may reuse this number
        if keep:
            _save_shard(d, [meta["chunk_ids"][i] for i in keep], rows)
        removed += len(hit)
    return removed


def build_store(cfg: Config, alias: str, variant: str = "hybrid", rebuild: bool = False) -> dict:
    d = store_dir(cfg, alias, variant)
    if rebuild and d.exists():
        for p in d.iterdir():
            p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    chunks = read_chunks(chunks_path(cfg, variant))
    done = _done_ids(d)
    stale = done - {c.chunk_id for c in chunks}
    if stale:
        raise SystemExit(f"{len(stale)} stored chunk ids no longer exist; re-run with --rebuild")
    enc = load_encoder(cfg, alias)
    t0, n_done = time.time(), 0
    if enc.kind == "contextual":
        # A window is (re)encoded whole so every chunk keeps the same context on resume.
        units = [w for w in enc.windows(chunks) if not all(c.chunk_id in done for c in w)]
        total = sum(len(w) for w in units)
        buf_ids, buf_vecs = [], []
        pb = ui.progress(total, f"embed {alias}", "chunks ").start()
        for w in units:
            buf_ids += [c.chunk_id for c in w]
            buf_vecs += list(enc.encode_window([c.text for c in w]))
            n_done += len(w)
            pb.advance(len(w))
            if len(buf_ids) >= 64 or w is units[-1]:
                _save_shard(d, buf_ids, buf_vecs)
                buf_ids, buf_vecs = [], []
                el = time.time() - t0
                ui.info(
                    f"[index] {alias}/{variant}: {n_done}/{total} chunks, {el:.0f}s ({n_done / el:.2f}/s)", flush=True
                )
        pb.close()
    else:
        todo = sorted((c for c in chunks if c.chunk_id not in done), key=lambda c: -len(c.text))
        total = len(todo)
        pb = ui.progress(total, f"embed {alias}", "chunks ").start()
        for i in range(0, total, 64):
            block = todo[i : i + 64]
            _save_shard(d, [c.chunk_id for c in block], enc.docs([c.text for c in block]))
            n_done += len(block)
            pb.advance(len(block))
            el = time.time() - t0
            ui.info(f"[index] {alias}/{variant}: {n_done}/{total} chunks, {el:.0f}s ({n_done / el:.2f}/s)", flush=True)
        pb.close()
    stats = {
        "embedder": alias,
        "model": embedder_spec(cfg, alias)["id"],
        "kind": enc.kind,
        "variant": variant,
        "rows": len(_done_ids(d)),
        "embedded_now": n_done,
        "embed_seconds": round(time.time() - t0, 1),
        "device": enc.device,
    }
    (d / "stats.json").write_text(json.dumps(stats, indent=2))
    ui.info(f"[index] {json.dumps(stats)}", flush=True)
    return stats


class VectorStore:
    """Exact search over a stored challenger embedding (cosine for dense rows, MaxSim for multi-vector)."""

    def __init__(self, cfg: Config, alias: str, variant: str = "hybrid", path: Path | None = None):
        d = path or store_dir(cfg, alias, variant)
        ids, lens, mats = [], [], []
        for p in sorted(d.glob("shard_*.json")):
            meta = json.loads(p.read_text())
            ids += meta["chunk_ids"]
            lens += meta["lens"]
            mats.append(np.load(p.with_suffix(".npy")))
        if not ids:
            fix = f"`docsearch render` and `docsearch index --visual {alias}`" if variant == "pages" else \
                f"`docsearch index --embedder {alias}`"  # fmt: skip
            raise SystemExit(f"no stored vectors in {d}; run {fix} first")
        self.ids = ids
        self.kind = embedder_spec(cfg, alias).get("kind")
        self.mat = np.concatenate(mats).astype(np.float32)
        self.lens = np.array(lens)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        sims = self.mat @ (q.T if q.ndim == 2 else q)  # (tokens, q_tokens) or (rows,)
        if self.kind in ("multivector", "visual"):
            sims = np.maximum.reduceat(sims, self.starts, axis=0).sum(axis=1)  # MaxSim per chunk
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]


def query_encoder(cfg: Config, alias: str):
    """The encoder for query vectors; a long-lived process keeps it loaded (models.keep_loaded)."""
    from .models import _kept

    return _kept(f"encoder:{alias}", cfg, lambda: load_encoder(cfg, alias))


def query_embeddings(cfg: Config, alias: str, texts: list[str]) -> list[np.ndarray]:
    h = hashlib.sha1((alias + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    p = cfg.path("cache_dir") / f"qemb_{alias}_{h}.npz"
    if p.exists():
        z = np.load(p)
        return [z[f"q{i}"] for i in range(len(texts))]
    vecs = query_encoder(cfg, alias).queries(texts)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{f"q{i}": v for i, v in enumerate(vecs)})
    return vecs
