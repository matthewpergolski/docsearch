"""Visual page retrieval: render pages/slides/images, embed them with a multi-vector visual retriever,
and fuse the visual page ranking with the hybrid text ranking (RRF at page/file level).

Units: page units (doc#pN) for pdf/slide_image/image, file units for docx/pptx (their text chunks have
no reliable page numbers, and OCB gold is file-level). xlsx/csv are not rendered (text only).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import ui
from .config import Config
from .manifest import page_unit, read_manifest

FILE_LEVEL = {"docx", "pptx", "xlsx", "csv"}
OFFICE_RENDER = {"docx", "pptx"}


def pages_dir(cfg: Config) -> Path:
    return cfg.path("index_dir").parent / "pages"


def page_snapshot(cfg: Config, alias: str, live: dict | None) -> Path | None:
    """The page-store snapshot the live pointer pins for `alias`; None means the working store."""
    snap = (live or {}).get("visual_snapshot") if alias == (live or {}).get("visual") else None
    return cfg.path("index_dir") / snap if snap else None


def fusion_unit(doc_id: str, fmt: str, page: int | None) -> str:
    return doc_id if fmt in FILE_LEVEL else page_unit(doc_id, page or 1)


def _render_pdf(
    pdf: Path, pages: list[int] | None, out: Path, dpi: int, max_pages: int | None
) -> list[tuple[int, Path]]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        n = len(doc)
        want = [p for p in (pages or range(1, n + 1)) if 1 <= p <= n]
        if max_pages:
            want = want[:max_pages]
        res = []
        for p in want:
            dst = out / f"p{p}.png"
            if not dst.exists():
                tmp = dst.with_suffix(".tmp.png")
                doc[p - 1].render(scale=dpi / 72).to_pil().convert("RGB").save(tmp)
                tmp.rename(dst)
            res.append((p, dst))
        return res
    finally:
        doc.close()


def page_store_ready(cfg: Config, alias: str, live: dict | None) -> bool:
    """The page store `alias` serves (the live snapshot, else the working store) holds at least one page. A store
    with 0 pages still writes its settings file, so its existence says nothing (FIX1)."""
    from .altembed import has_vectors, store_dir

    return has_vectors(page_snapshot(cfg, alias, live) or store_dir(cfg, alias, "pages"))


def _office_pdf(src: Path, work: Path, cache: Path, timeout: int = 180) -> Path | None:
    from .docconv import find_soffice, soffice_command

    dst = work / (src.stem + ".pdf")
    if dst.exists():
        return dst
    exe = find_soffice()
    if exe is None:
        ui.info(f"[render] office->pdf skipped for {src.name}: no soffice or libreoffice on PATH", flush=True)
        return None
    work.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            soffice_command(exe, cache, "--convert-to", "pdf", "--outdir", str(work), str(src)),
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        ui.info(f"[render] office->pdf failed for {src.name}: {type(e).__name__}", flush=True)
        return None
    return dst if dst.exists() else None


def render(cfg: Config, full: bool = False) -> dict:
    """Write page images and work/pages/pages.jsonl. Idempotent: existing images are reused."""
    from .subset import load_subset

    sub = load_subset(cfg, full)
    docs = read_manifest(cfg.path("corpus_dir") / "manifest.csv")
    if sub:
        keep = set(sub["docs"])
        docs = [d for d in docs if d.doc_id in keep]
    out = pages_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    rows, counts, failed = [], {}, []
    t0 = time.time()
    pb = ui.progress(len(docs), "render", "docs ").start()
    for d in docs:
        pb.advance()
        rows_d = render_doc(cfg, d, (sub or {}).get("pages", {}).get(d.doc_id))
        if rows_d is None:
            failed.append(d.doc_id)
            continue
        rows += rows_d
        if rows_d:
            counts[d.format] = counts.get(d.format, 0) + len(rows_d)
    pb.close()
    tmp = out / "pages.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.rename(out / "pages.jsonl")
    stats = {"pages": len(rows), "by_format": counts, "office_failed": failed, "seconds": round(time.time() - t0, 1)}
    (out / "render.stats.json").write_text(json.dumps(stats, indent=2))
    ui.info(f"[render] {json.dumps(stats)}", flush=True)
    return stats


def render_doc(cfg: Config, d, want: list[int] | None = None) -> list[dict] | None:
    """Render one document's pages (reusing existing images). Returns pages.jsonl rows, [] for formats the visual
    model does not see (xlsx/csv), or None when the Office->PDF conversion failed."""
    v = cfg["visual"]
    out = pages_dir(cfg)
    src = cfg.path("corpus_dir") / d.path
    ddir = out / d.doc_id
    if d.format == "pdf":
        ddir.mkdir(parents=True, exist_ok=True)
        pages = _render_pdf(src, want, ddir, int(v["dpi"]), None)
    elif d.format in ("slide_image", "image"):
        pages = [(1, src)]
    elif d.format in OFFICE_RENDER:
        pdf = _office_pdf(src, office_pdf_dir(cfg, d.doc_id), cfg.path("cache_dir") / "doc_docx")
        if pdf is None:
            return None
        ddir.mkdir(parents=True, exist_ok=True)
        pages = _render_pdf(pdf, None, ddir, int(v["dpi"]), int(v["office_max_pages"]))
    else:
        return []
    return [{"unit": page_unit(d.doc_id, p), "doc_id": d.doc_id, "format": d.format, "page": p, "path": str(path)}
            for p, path in pages]  # fmt: skip


def office_pdf_dir(cfg: Config, doc_id: str) -> Path:
    return pages_dir(cfg) / "_office_pdf" / doc_id.replace("/", "__")


def read_pages(cfg: Config, path: Path | None = None) -> list[dict]:
    p = path or pages_dir(cfg) / "pages.jsonl"
    if not p.exists():
        raise SystemExit("no rendered pages; run `docsearch render` first")
    with p.open() as f:
        return [json.loads(line) for line in f]


class VisualEncoder:
    """Sentence Transformers MultiVectorEncoder (e.g. pplx-embed-v2-late): images -> token vectors."""

    kind = "visual"

    def __init__(self, spec: dict, cfg: Config):
        from sentence_transformers import MultiVectorEncoder

        from .config import pick_device
        from .models import set_threads

        self.device = pick_device(cfg["models"]["device"])
        set_threads(self.device)
        with ui.loading("visual model", spec["id"]):
            self.model = MultiVectorEncoder(spec["id"], device=self.device, revision=spec.get("revision"))
        self.max_side = int(cfg["visual"]["max_side"])

    def _load(self, path: str):
        from PIL import Image

        im = Image.open(path).convert("RGB")
        s = self.max_side / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        return im

    def docs(self, paths: list[str], on_batch=None) -> list[np.ndarray]:
        out = []
        for p in paths:  # one page per call: bounded memory, and drives our progress bar per page
            e = self.model.encode_document([self._load(p)], batch_size=1, show_progress_bar=False)
            out += [np.asarray(x.float().cpu() if hasattr(x, "float") else x, dtype=np.float16) for x in e]
            if on_batch:
                on_batch(1)
        return out

    def queries(self, texts: list[str]) -> list[np.ndarray]:
        e = self.model.encode_query(texts, batch_size=16, show_progress_bar=False)
        return [np.asarray(x.float().cpu() if hasattr(x, "float") else x, dtype=np.float32) for x in e]


class PageVectorEncoder(VisualEncoder):
    """One vector per page image (`vectors: single`, e.g. EmbeddingGemma 2). A page is stored as a matrix of
    one row, so the store's MaxSim is the cosine."""

    def __init__(self, spec: dict, cfg: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        from .config import pick_device
        from .models import set_threads

        self.device = pick_device(cfg["models"]["device"])
        set_threads(self.device)
        with ui.loading("visual model", spec["id"]):
            # float32: the EmbeddingGemma 2 card warns that float16 can give NaN or worse vectors with no error.
            self.model = SentenceTransformer(
                spec["id"],
                device=self.device,
                revision=spec.get("revision"),
                model_kwargs={"torch_dtype": torch.float32},
            )
        if spec.get("image_tokens"):  # picture detail; the card allows 70 to 1120 (default 280)
            ip = self.model[0].processor.image_processor
            ip.max_soft_tokens = ip.image_seq_length = self.model[0].processor.image_seq_length = int(
                spec["image_tokens"]
            )
        self.max_side = int(cfg["visual"]["max_side"])

    def docs(self, paths: list[str], on_batch=None) -> list[np.ndarray]:
        out = []
        for p in paths:  # the model card: an image gets no lead-in
            v = self.model.encode(
                [self._load(p)], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
            out.append(v.astype(np.float16))
            if on_batch:
                on_batch(1)
        return out

    def queries(self, texts: list[str]) -> list[np.ndarray]:
        v = self.model.encode(
            texts,
            prompt_name="query",
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [x[None, :].astype(np.float32) for x in v]


def build_visual_store(cfg: Config, alias: str, rebuild: bool = False) -> dict:
    from .altembed import _done_ids, _save_shard, load_encoder, store_dir

    d = store_dir(cfg, alias, "pages")
    if rebuild and d.exists():
        for p in d.iterdir():
            p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    pages = read_pages(cfg)
    done = _done_ids(d)
    todo = [p for p in pages if p["unit"] not in done]
    enc = load_encoder(cfg, alias)
    t0 = time.time()
    block = int(cfg["visual"]["shard_pages"])
    pb = ui.progress(len(todo), f"embed pages ({alias})", "pages ").start()
    for i in range(0, len(todo), block):
        b = todo[i : i + block]
        _save_shard(d, [p["unit"] for p in b], enc.docs([p["path"] for p in b], on_batch=pb.advance))
        n, el = i + len(b), time.time() - t0
        eta = el / n * (len(todo) - n)
        ui.info(
            f"[visual] {alias}: {n}/{len(todo)} pages, {el:.0f}s ({el / n:.1f}s/page, ~{eta / 60:.0f} min left)",
            flush=True,
        )
    pb.close()
    stats = {"embedder": alias, "pages": len(_done_ids(d)), "embedded_now": len(todo),
             "embed_seconds": round(time.time() - t0, 1), "device": enc.device}  # fmt: skip
    (d / "stats.json").write_text(json.dumps(stats, indent=2))
    ui.info(f"[visual] {json.dumps(stats)}", flush=True)
    return stats


class VisualRanker:
    """Ranks fusion units (page or file) by MaxSim over rendered page images."""

    def __init__(self, cfg: Config, alias: str, snapshot: Path | None = None):
        """snapshot: a live snapshot dir (see sync.promote); None reads the working store."""
        from .altembed import VectorStore

        with ui.loading("page vectors", alias):
            self.store = VectorStore(cfg, alias, "pages", snapshot)
        meta = {p["unit"]: p for p in read_pages(cfg, snapshot / "pages.jsonl" if snapshot else None)}
        self.meta = meta
        stored = set(self.store.ids)
        self.docs = {m["doc_id"] for u, m in meta.items() if u in stored}  # docs the visual model has seen

    def rank(self, qvec: np.ndarray, k: int, office_pages: int = 0, capped=OFFICE_RENDER,
             allow: frozenset[str] | None = None) -> list[tuple[str, dict]]:  # fmt: skip
        """office_pages > 0 (M24): only the first N rendered pages of a file of a `capped` format count.
        allow: the doc ids a search filter lets through (None = all), skipped before the top k is cut."""
        out, seen = [], set()
        for unit, _s in self.store.search(qvec, len(self.store.ids)):
            m = self.meta.get(unit)
            if m is None or (office_pages and m["format"] in capped and m["page"] > office_pages):
                continue
            if allow is not None and m["doc_id"] not in allow:
                continue
            fu = fusion_unit(m["doc_id"], m["format"], m["page"])
            if fu in seen:
                continue
            seen.add(fu)
            out.append((fu, m))
            if len(out) >= k:
                break
        return out
