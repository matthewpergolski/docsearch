"""Chunking strategies. Every chunk keeps doc_id + page list so eval can score at page/file level.

- hybrid:    Docling HybridChunker (structure-aware, token-capped), with "File: <name>" and the
             heading path prefixed to the indexed text.
- recursive: plain control. Whole-document text split recursively (paragraph > line > sentence >
             word) into <= max_tokens pieces. No file name, no headings.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ui
from .config import Config
from .manifest import Doc, read_manifest

VARIANTS = ("hybrid", "recursive")
# Derived variants: the base variant's chunks verbatim, with only some formats re-chunked (one change).
DERIVED = {"hybrid_tbl": ("hybrid", {"xlsx", "csv"}), "hybrid_pdf": ("hybrid", {"pdf"}),
           # M28: a spreadsheet over the chunk cap shares the cap over its tables (other files: same chunks).
           # Adopted 2026-10-13: `hybrid` now does the same, so this variant gives the same chunks.
           "hybrid_sheets": ("hybrid", {"xlsx", "csv"}),
           "hybrid_pdftable": ("hybrid", {"pdf"}),
           # M22 test variants: the PDFs re-chunked from a parse that scripts/scan_pdfs.py wrote to work/scan.
           "hybrid_scan_noocr": ("hybrid", {"pdf"}), "hybrid_scan": ("hybrid", {"pdf"}),
           "hybrid_mix_noocr": ("hybrid", {"pdf"}), "hybrid_mix": ("hybrid", {"pdf"}),
           "hybrid_pdfocr": ("hybrid", {"pdf"})}  # fmt: skip
# Phase H: the base variant's chunks with an LLM context line in some of them (scripts/enrich_trial.py writes them).
ENRICHED = {"hybrid_ctx": "hybrid", "hybrid_ctx_slides": "hybrid"}
SLIDE_CTX = "hybrid_ctx_slides"  # H2: `sync` writes it with enrich.llm on (a context line on each slide image)


def search_variant(cfg: Config) -> str:
    """The chunk variant that search and the search page read: the slide variant with enrich.llm on, else hybrid."""
    return SLIDE_CTX if cfg["enrich"]["llm"] != "off" else "hybrid"


def base_variant(variant: str) -> str | None:
    """The variant whose vectors a derived or enriched variant reuses for unchanged chunks, and its eval reference."""
    return DERIVED[variant][0] if variant in DERIVED else ENRICHED.get(variant)


# variant -> work/scan/<dir>/parsed. hybrid_scan_noocr has no parse: a scan read without OCR gives no text.
SCAN_PARSED = {"hybrid_scan": "full_on", "hybrid_mix_noocr": "mixed_off", "hybrid_mix": "mixed_on",
               "hybrid_pdfocr": "orig_on"}  # fmt: skip
SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    format: str
    source: str
    pages: list[int]
    text: str  # what gets embedded + keyword-indexed
    n_tokens: int = 0
    meta: dict = field(default_factory=dict)


def chunks_path(cfg: Config, variant: str) -> Path:
    return cfg.path("index_dir").parent / "chunks" / f"{variant}.jsonl"


# ------------------------------------------------------------------ recursive control
def recursive_spans(
    text: str, count: Callable[[str], int], max_tokens: int, seps: list[str] = SEPARATORS
) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) spans of <= max_tokens (best effort at word level)."""

    def atoms(s: int, e: int, level: int) -> Iterator[tuple[int, int, int]]:
        n = count(text[s:e])
        if n <= max_tokens or level >= len(seps):
            yield s, e, n
            return
        sep = seps[level]
        pos = s
        while pos < e:
            j = text.find(sep, pos, e)
            nxt = e if j == -1 else j + len(sep)
            if nxt > pos:
                yield from atoms(pos, nxt, level + 1)
            pos = nxt

    spans: list[tuple[int, int]] = []
    cur_s = cur_e = None
    cur_n = 0
    for s, e, n in atoms(0, len(text), 0):
        if cur_s is not None and cur_n + n <= max_tokens:
            cur_e, cur_n = e, cur_n + n
        else:
            if cur_s is not None:
                spans.append((cur_s, cur_e))
            cur_s, cur_e, cur_n = s, e, n
    if cur_s is not None:
        spans.append((cur_s, cur_e))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def pages_for_span(bounds: list[tuple[int, int, int]], s: int, e: int) -> list[int]:
    return sorted({p for bs, be, p in bounds if bs < e and be > s and p > 0})


def doc_text_by_page(doc, max_chars: int = 0) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate per-page markdown; return text and (start, end, page_no) bounds.
    Stops adding pages once max_chars is reached (0 = no limit)."""
    parts: list[str] = []
    bounds: list[tuple[int, int, int]] = []
    pos = 0
    page_nos = sorted(doc.pages) if doc.pages else []
    if not page_nos:
        t = doc.export_to_markdown(image_placeholder="")
        if max_chars:
            t = t[:max_chars]
        return t, [(0, len(t), 0)]
    for p in page_nos:
        if max_chars and pos >= max_chars:
            break
        t = doc.export_to_markdown(page_no=p, image_placeholder="").strip()
        if not t:
            continue
        t += "\n\n"
        parts.append(t)
        bounds.append((pos, pos + len(t), p))
        pos += len(t)
    return "".join(parts), bounds


def chunk_recursive(doc, d: Doc, count: Callable[[str], int], max_tokens: int, max_chunks: int = 0) -> list[Chunk]:
    # ~8 chars/token upper bound keeps giant spreadsheets from being exported/split in full.
    text, bounds = doc_text_by_page(doc, max_chars=max_chunks * max_tokens * 8 if max_chunks else 0)
    out = []
    for i, (s, e) in enumerate(recursive_spans(text, count, max_tokens)):
        if max_chunks and i >= max_chunks:
            break
        body = text[s:e].strip()
        out.append(
            Chunk(
                chunk_id=f"recursive:{d.doc_id}:{i}",
                doc_id=d.doc_id,
                format=d.format,
                source=d.source,
                pages=pages_for_span(bounds, s, e),
                text=body,
                n_tokens=count(body),
            )
        )
    return out


# ------------------------------------------------------------------ Docling hybrid
def chunk_hybrid(doc, d: Doc, chunker, count: Callable[[str], int], max_chunks: int = 0) -> list[Chunk]:
    out = []
    fname = Path(d.path).name
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        if max_chunks and i >= max_chunks:
            break
        ctx = chunker.contextualize(chunk=ch)  # heading path + text
        text = f"File: {fname}\n{ctx}"
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in (it.prov or [])})
        out.append(
            Chunk(
                chunk_id=f"hybrid:{d.doc_id}:{i}",
                doc_id=d.doc_id,
                format=d.format,
                source=d.source,
                pages=pages,
                text=text,
                n_tokens=count(text),
                meta={"headings": list(ch.meta.headings or [])},
            )
        )
    return out


TABULAR = {"xlsx", "csv"}
IMAGE_FORMATS = {"image", "slide_image"}


def share_cap(sizes: list[int], cap: int) -> list[int]:
    """How many chunks each table keeps when a file is over the cap: one more per table in turn, in table
    order, until the cap is used. With the plain cap the first table can take every slot (M28)."""
    keep, left = [0] * len(sizes), cap
    while left > 0 and any(k < n for k, n in zip(keep, sizes, strict=True)):
        for i, n in enumerate(sizes):
            if left > 0 and keep[i] < n:
                keep[i] += 1
                left -= 1
    return keep


def chunk_table_rows(
    doc, d: Doc, count: Callable[[str], int], max_tokens: int, max_chunks: int = 0, spread: bool = False
) -> list[Chunk]:
    """Spreadsheet chunker for the hybrid variant: each table's header row is repeated at the top of
    every chunk, followed by as many whole rows as fit. Docling's HybridChunker re-tokenizes very
    wide/long tables quadratically, which took >10 min per CSV on the box.
    spread: a file over max_chunks shares the cap over its tables (share_cap; M28, the hybrid default)."""
    out: list[Chunk] = []
    fname = Path(d.path).name
    tables: list[tuple[int, list[int], str, int, list[tuple[list[str], int]]]] = []
    for ti, table in enumerate(doc.tables):
        try:
            df = table.export_to_dataframe(doc=doc)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        pages = sorted({p.page_no for p in (table.prov or [])})
        header = " | ".join(str(c) for c in df.columns)
        prefix = f"File: {fname}\nTable {ti + 1}\n{header}\n"
        budget = max_tokens - count(prefix)
        done = 0 if spread else sum(len(t[4]) for t in tables)
        groups: list[tuple[list[str], int]] = []
        rows, used = [], 0
        for row in df.itertuples(index=False):
            line = " | ".join("" if v is None else str(v) for v in row)
            n = count(line) + 1
            if rows and used + n > budget:
                groups.append((rows, used))
                rows, used = [], 0
                if max_chunks and done + len(groups) >= max_chunks:
                    break
            rows.append(line[: budget * 6])  # a single over-long row is truncated, not split
            used += min(n, budget)
        if rows and not (max_chunks and done + len(groups) >= max_chunks):
            groups.append((rows, used))
        tables.append((ti, pages, prefix, budget, groups))
        if max_chunks and not spread and done + len(groups) >= max_chunks:
            break
    keep = [len(t[4]) for t in tables]
    if spread and max_chunks:
        keep = share_cap(keep, max_chunks)
    for (ti, pages, prefix, budget, groups), k in zip(tables, keep, strict=True):
        for g_rows, g_used in groups[:k]:
            out.append(
                Chunk(
                    chunk_id=f"hybrid:{d.doc_id}:{len(out)}",
                    doc_id=d.doc_id,
                    format=d.format,
                    source=d.source,
                    pages=pages,
                    text=prefix + "\n".join(g_rows),
                    n_tokens=g_used + max_tokens - budget,
                    meta={"table": ti + 1},
                )
            )
    # Non-table text in the workbook (titles, notes) via the normal chunker is skipped on purpose:
    # sheets are almost entirely tables, and the header row carries the column semantics.
    return out[:max_chunks] if max_chunks else out


def table_summary(sheet: str, ti: int, n_tables: int, df, max_cols: int = 12, n_samples: int = 3) -> str:
    """Deterministic table summary: sheet, columns, row count, a few sample values per column."""
    cols = [str(c) for c in df.columns]
    shown = ", ".join(cols[:max_cols]) + (f", ... (+{len(cols) - max_cols})" if len(cols) > max_cols else "")
    samples = []
    for j, c in enumerate(df.columns[: min(6, len(cols))]):
        vals = []
        for v in df.iloc[:, j].tolist():  # positional: column names can repeat
            sv = "" if v is None else str(v).strip()
            if sv and sv.lower() != "nan" and sv not in vals:
                vals.append(sv[:24])
            if len(vals) >= n_samples:
                break
        if vals:
            samples.append(f"{str(c)[:24]}: {'; '.join(vals)}")
    return (
        f"Sheet: {sheet} (table {ti + 1} of {n_tables})\n"
        f"Columns ({len(cols)}): {shown}\n"
        f"Rows: {len(df)}\n"
        f"Sample values: {' | '.join(samples)}\n"
    )


def chunk_table_struct(doc, d: Doc, count: Callable[[str], int], max_tokens: int, max_chunks: int = 0) -> list[Chunk]:
    """C2 structure-aware spreadsheet chunker: every chunk carries file, sheet name, a short table
    summary and the header row, then whole rows. No LLM; output depends only on the parsed table."""
    out: list[Chunk] = []
    fname = Path(d.path).name
    n_tables = len(doc.tables)
    for ti, table in enumerate(doc.tables):
        try:
            df = table.export_to_dataframe(doc=doc)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        par = table.parent.resolve(doc) if table.parent else None
        sheet = getattr(par, "name", None) if getattr(par, "label", None) == "sheet" else None
        sheet = sheet or Path(fname).stem
        pages = sorted({p.page_no for p in (table.prov or [])})
        summary = table_summary(sheet, ti, n_tables, df)
        if count(summary) > max_tokens // 3:  # keep room for rows on very wide tables
            summary = table_summary(sheet, ti, n_tables, df, max_cols=6, n_samples=1)
        header = "Header: " + " | ".join(str(c) for c in df.columns)
        prefix = f"File: {fname}\n{summary}{header}\n"
        budget = max(max_tokens - count(prefix), 32)
        groups, rows, used = [], [], 0
        for row in df.itertuples(index=False):
            line = " | ".join("" if v is None else str(v) for v in row)
            n = count(line) + 1
            if rows and used + n > budget:
                groups.append((rows, used))
                rows, used = [], 0
                if max_chunks and len(out) + len(groups) >= max_chunks:
                    break
            rows.append(line[: budget * 6])
            used += min(n, budget)
        if rows and not (max_chunks and len(out) + len(groups) >= max_chunks):
            groups.append((rows, used))
        for g_rows, g_used in groups:
            out.append(Chunk(chunk_id=f"hybrid_tbl:{d.doc_id}:{len(out)}", doc_id=d.doc_id, format=d.format,
                             source=d.source, pages=pages, text=prefix + "\n".join(g_rows),
                             n_tokens=g_used + max_tokens - budget, meta={"table": ti + 1, "sheet": sheet}))  # fmt: skip
        if max_chunks and len(out) >= max_chunks:
            return out[:max_chunks]
    return out


def chunk_pdf_overlap(cfg: Config, doc, d: Doc, count: Callable[[str], int]) -> list[Chunk]:
    """C3 PDF tweak: smaller HybridChunker chunks (chunking.pdf_max_tokens) and each chunk prefixed
    with the last chunking.pdf_overlap_tokens tokens of the previous chunk's text."""
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    tok = get_tokenizer(cfg)
    size = int(cfg["chunking"]["pdf_max_tokens"])
    ov = int(cfg["chunking"]["pdf_overlap_tokens"])
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tok, max_tokens=size - 24 - ov), merge_peers=True)
    fname = Path(d.path).name
    out, prev_ids, prev_pages = [], [], []
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        ctx = chunker.contextualize(chunk=ch)
        pages = sorted({p.page_no for it in ch.meta.doc_items for p in (it.prov or [])})
        lead = tok.decode(prev_ids[-ov:]) + "\n" if ov and prev_ids else ""
        text = f"File: {fname}\n{lead}{ctx}"
        out.append(Chunk(chunk_id=f"hybrid_pdf:{d.doc_id}:{i}", doc_id=d.doc_id, format=d.format, source=d.source,
                         pages=pages, text=text, n_tokens=count(text), meta={"overlap": bool(lead)}))  # fmt: skip
        prev_ids = tok.encode(ch.text, add_special_tokens=False)
        prev_pages = pages
    del prev_pages
    return out


def _md_row(cells) -> str:
    return "| " + " | ".join(" ".join(str(x).split()) for x in cells) + " |"


def table_pieces(prefix: str, header: str, rows: list[str], count: Callable[[str], int], max_tokens: int) -> list[str]:
    """Split a table by rows into pieces of <= max_tokens, each starting with prefix + header (header row and
    its separator). A single row longer than the cap gets its own piece."""
    head = prefix + header
    out, cur = [], []
    for r in rows:
        if cur and count(head + "\n".join(cur + [r])) > max_tokens:
            out.append(head + "\n".join(cur))
            cur = []
        cur.append(r)
    if cur or not out:
        out.append(head + "\n".join(cur))
    return out


def pdf_table_chunks(
    doc, d: Doc, chunker, count: Callable[[str], int], max_tokens: int
) -> tuple[set[str], list[Chunk]]:
    """M15: table-aware PDF chunking. Returns (ids of the base hybrid chunks that hold only table items, to drop;
    new table chunks). Each Docling table becomes whole chunks, split by rows when too long, every piece
    prefixed with file name, section title, page and caption, and the header row repeated. Base chunk ids follow
    chunk_hybrid's numbering, so all other chunks of the document stay byte-identical and keep their vectors."""
    from docling_core.types.doc import SectionHeaderItem, TableItem, TitleItem

    drop = set()
    for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
        items = list(ch.meta.doc_items or [])
        if items and all(isinstance(it, TableItem) for it in items):
            drop.add(f"hybrid:{d.doc_id}:{i}")
    fname = Path(d.path).name
    new: list[Chunk] = []
    title = ""
    ti = 0
    for item, _lvl in doc.iterate_items():
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            title = " ".join(str(item.text).split())
            continue
        if not isinstance(item, TableItem):
            continue
        ti += 1
        pages = sorted({p.page_no for p in (item.prov or [])})
        try:
            cap = " ".join(str(item.caption_text(doc) or "").split())
        except Exception:  # noqa: BLE001
            cap = ""
        try:
            df = item.export_to_dataframe(doc=doc)
        except TypeError:
            df = item.export_to_dataframe()
        if df is None or df.empty:
            md = item.export_to_markdown(doc=doc).splitlines()
            header, rows = ("\n".join(md[:2]) + "\n", md[2:]) if len(md) > 2 else ("", md)
        else:
            header = _md_row(df.columns) + "\n" + _md_row(["---"] * len(df.columns)) + "\n"
            rows = [_md_row(r) for r in df.itertuples(index=False)]
        pg = ", ".join(map(str, pages)) or "?"
        prefix = (f"File: {fname}\n" + (f"Section: {title}\n" if title else "")
                  + f"Table {ti} (page {pg})" + (f": {cap}" if cap else "") + "\n")  # fmt: skip
        for k, text in enumerate(table_pieces(prefix, header, rows, count, max_tokens)):
            new.append(Chunk(chunk_id=f"hybrid_pdftable:{d.doc_id}:t{ti}_{k}", doc_id=d.doc_id, format=d.format,
                             source=d.source, pages=pages, text=text, n_tokens=count(text),
                             meta={"table": ti, "piece": k, "caption": cap}))  # fmt: skip
    return drop, new


def merge_pdf_tables(base: list[Chunk], drop: set[str], new: list[Chunk]) -> list[Chunk]:
    """Base chunks minus the dropped table-only ones, verbatim, then the new table chunks."""
    return [c for c in base if c.chunk_id not in drop] + new


def build_derived(cfg: Config, variant: str, full: bool = False) -> dict:
    """Copy the base variant's chunks and re-chunk only the variant's target formats."""
    from .ingest import load_parsed

    base, formats = DERIVED[variant]
    base_chunks = read_chunks(chunks_path(cfg, base))
    keep = [c for c in base_chunks if c.format not in formats]
    target_docs = list(dict.fromkeys(c.doc_id for c in base_chunks if c.format in formats))
    docs = {d.doc_id: d for d in read_manifest(cfg.path("corpus_dir") / "manifest.csv")}
    tok = get_tokenizer(cfg)

    def count(x: str) -> int:
        return len(tok.encode(x, add_special_tokens=False))

    max_tokens = int(cfg["chunking"]["max_tokens"])
    from .subset import load_subset

    sub = load_subset(cfg, full)
    t0 = time.time()
    new: list[Chunk] = []
    drop: set[str] = set()
    hy = DocChunker(cfg, "hybrid").chunker if variant == "hybrid_pdftable" or variant in SCAN_PARSED else None
    pb = ui.progress(len(target_docs), f"chunk {variant}", "docs ").start()
    for did in target_docs:
        pb.advance()
        d = docs[did]
        if variant == "hybrid_scan_noocr":
            continue
        if variant in SCAN_PARSED:
            from docling_core.types.doc import DoclingDocument

            scan = cfg.path("parsed_dir").parent / "scan" / SCAN_PARSED[variant] / "parsed" / f"{did}.json"
            doc = DoclingDocument.model_validate_json(scan.read_text())
            new += chunk_hybrid(doc, d, hy, count) or fallback_chunk(doc, d, "hybrid", count, max_tokens)
            continue
        doc = load_parsed(cfg, did)
        if variant == "hybrid_tbl":
            cap = int(cfg["subset"]["office_max_chunks"]) if sub else int(cfg["chunking"].get("max_chunks_per_doc", 0))
            new += chunk_table_struct(doc, d, count, max_tokens, cap)
        elif variant == "hybrid_sheets":
            cap = int(cfg["subset"]["office_max_chunks"]) if sub else int(cfg["chunking"].get("max_chunks_per_doc", 0))
            new += chunk_table_rows(doc, d, count, max_tokens, cap, spread=True) or fallback_chunk(
                doc, d, "hybrid", count, max_tokens
            )
        elif variant == "hybrid_pdftable":
            dr, nw = pdf_table_chunks(doc, d, hy, count, max_tokens)
            drop |= dr
            new += nw
        else:  # hybrid_pdf: recursive-style token windows with overlap inside Docling's structure
            new += chunk_pdf_overlap(cfg, doc, d, count)
    pb.close()
    if variant in ("hybrid_pdf", "hybrid_pdftable") and sub:
        new = [c for c in new if c.doc_id not in sub["pages"] or set(c.pages) & set(sub["pages"][c.doc_id])]
    # hybrid_pdftable: non-table PDF chunks stay verbatim (same ids, so their vectors are reused)
    out = merge_pdf_tables(base_chunks, drop, new) if variant == "hybrid_pdftable" else keep + new
    path = chunks_path(cfg, variant)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for c in out:
            f.write(json.dumps(c.__dict__) + "\n")
    tmp.rename(path)
    by_fmt: dict = {}
    for c in out:
        by_fmt.setdefault(c.format, {"chunks": 0, "docs": set()})
        by_fmt[c.format]["chunks"] += 1
        by_fmt[c.format]["docs"].add(c.doc_id)
    base_stats = json.loads(chunks_path(cfg, base).with_suffix(".stats.json").read_text())
    stats = {"variant": variant, "base": base, "rechunked_formats": sorted(formats), "chunks": len(out),
             "rechunked_docs": len(target_docs), "new_chunks": len(new), "dropped_table_chunks": len(drop), "seconds": round(time.time() - t0, 1),
             "by_format": {k: {"chunks": v["chunks"], "docs": len(v["docs"])} for k, v in by_fmt.items()},
             "excluded_oversize": base_stats.get("excluded_oversize", [])}  # fmt: skip
    path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    ui.info(f"[chunk] {variant}: {json.dumps(stats)}", flush=True)
    return stats


def all_text(doc) -> str:
    """Every text item in reading order, including OCR text nested inside pictures (charts, slides),
    which structure-based chunkers skip."""
    parts = []
    for item, _lvl in doc.iterate_items(with_groups=False, traverse_pictures=True):
        t = (getattr(item, "text", "") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def fallback_chunk(doc, d: Doc, variant: str, count: Callable[[str], int], max_tokens: int) -> list[Chunk]:
    """If a parsed doc produced no chunks but has any text (typically OCR'd charts/slides), index the
    text anyway so every parsed document is searchable."""
    text = all_text(doc)
    if not text:
        return []
    prefix = f"File: {Path(d.path).name}\n" if variant == "hybrid" else ""
    out = []
    for i, (s, e) in enumerate(recursive_spans(text, count, max_tokens - (16 if prefix else 0))):
        body = prefix + text[s:e].strip()
        pages = sorted(doc.pages) if doc.pages and len(doc.pages) == 1 else []
        out.append(Chunk(chunk_id=f"{variant}:{d.doc_id}:fb{i}", doc_id=d.doc_id, format=d.format,
                         source=d.source, pages=pages, text=body, n_tokens=count(body), meta={"fallback": True}))  # fmt: skip
    return out


def get_tokenizer(cfg: Config):
    """The embedder's tokenizer: chunk sizes are counted in its tokens, also with models.backend api."""
    from transformers import AutoTokenizer

    name = cfg["models"]["embedder"]
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as e:
        raise SystemExit(f"chunking counts tokens with the tokenizer of models.embedder ({name}) and cannot load it: "
                         f"{e}. Set models.embedder to the Hugging Face id of the model (with models.backend api, "
                         "put the server's own name in models.embed_api_model), or connect once so the tokenizer "
                         "is cached.") from None  # fmt: skip


class DocChunker:
    """Chunks one parsed document for a variant. Shared by `build_chunks` (whole corpus) and `sync` (one file)."""

    def __init__(self, cfg: Config, variant: str, sub: dict | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; choose from {VARIANTS}")
        self.cfg, self.variant, self.sub = cfg, variant, sub
        self.max_tokens = int(cfg["chunking"]["max_tokens"])
        self.max_chunks = int(cfg["chunking"].get("max_chunks_per_doc", 0) or 0)
        self.max_parsed_bytes = int(float(cfg["chunking"].get("max_parsed_mb", 0) or 0) * 1e6)
        self.office_cap = int(cfg["subset"]["office_max_chunks"]) if sub else 0
        tok = get_tokenizer(cfg)
        self.count = lambda s: len(tok.encode(s, add_special_tokens=False))
        self.chunker = None
        if variant == "hybrid":
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

            # Leave room for the "File: <name>" prefix we add on top of Docling's contextualization.
            self.chunker = HybridChunker(
                tokenizer=HuggingFaceTokenizer(tokenizer=tok, max_tokens=self.max_tokens - 24),
                merge_peers=True,
            )

    def __call__(self, d: Doc) -> tuple[list[Chunk], str, int]:
        """Returns (chunks, status, parsed_bytes); status is ok | missing_parse | oversize."""
        from .ingest import load_parsed, parsed_path

        pp = parsed_path(self.cfg, d.doc_id)
        if not pp.exists():
            return [], "missing_parse", 0
        size = pp.stat().st_size
        if self.max_parsed_bytes and size > self.max_parsed_bytes:
            # Multi-million-cell spreadsheets: Docling JSON this large needs >10 GB RAM to load.
            return [], "oversize", size
        try:
            doc = load_parsed(self.cfg, d.doc_id)
        except Exception as e:  # noqa: BLE001 - e.g. a file still being written by ingest
            ui.info(f"[chunk] skip {d.doc_id}: {type(e).__name__}", flush=True)
            return [], "missing_parse", size
        count, max_tokens, variant = self.count, self.max_tokens, self.variant
        # The per-doc cap only applies to spreadsheets; text documents are always chunked fully.
        cap = self.max_chunks if d.format in TABULAR else 0
        if self.office_cap and d.format in TABULAR | {"docx", "pptx"}:
            cap = min(cap, self.office_cap) if cap else self.office_cap
        if d.format in IMAGE_FORMATS:
            # OCR text of charts/slides is nested inside picture items, which both the
            # HybridChunker and markdown export skip; index all OCR text directly.
            chunks = fallback_chunk(doc, d, variant, count, max_tokens)
        elif variant == "hybrid" and d.format in TABULAR:
            chunks = chunk_table_rows(doc, d, count, max_tokens, cap, spread=True)
        elif variant == "hybrid":
            chunks = chunk_hybrid(doc, d, self.chunker, count, cap)
        else:
            chunks = chunk_recursive(doc, d, count, max_tokens, cap)
        if not chunks:
            chunks = fallback_chunk(doc, d, variant, count, max_tokens)
        if self.sub and d.doc_id in self.sub["pages"]:  # page-sampled PDFs: keep chunks touching kept pages
            kp = set(self.sub["pages"][d.doc_id])
            chunks = [c for c in chunks if set(c.pages) & kp]
        del doc
        if size > 20e6:
            gc.collect()
        return chunks, ("capped" if cap and len(chunks) >= cap else "ok"), size


def build_chunks(cfg: Config, variant: str, full: bool = False) -> dict:
    from .subset import load_subset

    sub = load_subset(cfg, full)
    chunk_doc = DocChunker(cfg, variant, sub)
    docs = read_manifest(cfg.path("corpus_dir") / "manifest.csv")
    if sub:
        keep = set(sub["docs"])
        docs = [d for d in docs if d.doc_id in keep]
    out_path = chunks_path(cfg, variant)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats: dict = {
        "variant": variant,
        "subset": bool(sub),
        "max_chunks_per_doc": chunk_doc.max_chunks,
        "docs": 0,
        "missing_parse": 0,
        "chunks": 0,
        "capped_docs": [],
        "excluded_oversize": [],
        "by_format": {},
    }
    pb = ui.progress(len(docs), f"chunk {variant}", "docs ").start()
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for d in docs:
            pb.advance()
            chunks, status, size = chunk_doc(d)
            if status == "missing_parse":
                stats["missing_parse"] += 1
                continue
            if status == "oversize":
                stats["excluded_oversize"].append(d.doc_id)
                ui.info(f"[chunk] exclude {d.doc_id}: parsed JSON {size / 1e6:.0f} MB", flush=True)
                continue
            if status == "capped":
                stats["capped_docs"].append(d.doc_id)
            stats["docs"] += 1
            stats["chunks"] += len(chunks)
            bf = stats["by_format"].setdefault(d.format, {"docs": 0, "chunks": 0, "tokens": 0, "capped": 0})
            bf["capped"] += int(status == "capped")
            bf["zero_chunk_docs"] = bf.get("zero_chunk_docs", 0) + int(not chunks)
            bf["docs"] += 1
            bf["chunks"] += len(chunks)
            bf["tokens"] += sum(c.n_tokens for c in chunks)
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    stats["seconds"] = round(time.time() - t0, 1)
    (out_path.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2))
    pb.close()
    ui.info(f"[chunk] {variant}: {json.dumps(stats)}", flush=True)
    return stats


def replace_doc_chunks(path: Path, doc_ids: set[str], new: list[Chunk]) -> int:
    """Rewrite a chunks file without the chunks of `doc_ids`, plus `new`. Atomic; returns rows removed."""
    kept, removed = [], 0
    if path.exists():
        for line in path.open():
            if json.loads(line)["doc_id"] in doc_ids:
                removed += 1
            else:
                kept.append(line if line.endswith("\n") else line + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        f.writelines(kept)
        for c in new:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return removed


def read_chunks(path: Path) -> list[Chunk]:
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
