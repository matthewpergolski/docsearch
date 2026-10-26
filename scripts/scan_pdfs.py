"""M22 test data: copies of every PDF in the corpus as scans, and their parses with and without OCR.

Usage: uv run python scripts/scan_pdfs.py --kind full|mixed|orig --ocr on|off [--dpi 200] [--limit N]
  full   every page is an image (a scanned document)
  mixed  every odd page is an image, every even page keeps its text layer (a typed PDF with scanned pages)
  orig   the corpus PDF itself (shows what OCR changes in a born-digital file)
Writes work/scan/<kind>/corpus/<path> (the copies) and work/scan/<kind>_<ocr>/parsed/<doc_id>.json (Docling
JSON). Resumable: a document whose JSON exists is skipped. The real corpus and work/parsed are not touched.
Then `docsearch chunk --variant <v> --full` and `docsearch index --variant <v>`; chunk.SCAN_PARSED maps v to a parse.
"""

import argparse
import copy
from pathlib import Path

from docsearch.config import Config, load_config
from docsearch.ingest import _parse_one, build_converter
from docsearch.manifest import read_manifest


def text_chars(path: Path) -> list[int]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    return [pdf[i].get_textpage().count_chars() for i in range(len(pdf))]


def full_copy(src: Path, dst: Path, dpi: int) -> None:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(src))
    pages = [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    dst.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(dst, "PDF", resolution=dpi, save_all=True, append_images=pages[1:], quality=80)


def mixed_copy(src: Path, scan: Path, dst: Path) -> None:
    """Pages 1, 3, 5, ... from the scanned copy; pages 2, 4, 6, ... from the original."""
    import pypdfium2 as pdfium

    a, b, out = pdfium.PdfDocument(str(src)), pdfium.PdfDocument(str(scan)), pdfium.PdfDocument.new()
    for i in range(len(a)):
        out.import_pages(b if i % 2 == 0 else a, [i])
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["full", "mixed", "orig"], required=True)
    ap.add_argument("--ocr", choices=["on", "off"], required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config()
    root = cfg.path("parsed_dir").parent / "scan"
    corpus = cfg.path("corpus_dir") if a.kind == "orig" else root / a.kind / "corpus"
    raw = copy.deepcopy(cfg.raw)
    raw["paths"]["corpus_dir"] = str(corpus)
    raw.setdefault("ingest", {})["pdf_ocr"] = a.ocr == "on"  # on: each PDF is parsed in a child process (crash guard)
    scan_cfg = Config(raw=raw, root=cfg.root)
    docs = [d for d in read_manifest(cfg.path("corpus_dir") / "manifest.csv") if d.format == "pdf"]
    docs.sort(key=lambda d: int(d.pages or 0))
    conv = build_converter(pdf_ocr=a.ocr == "on")
    for d in docs[: a.limit]:
        out = root / f"{a.kind}_{a.ocr}" / "parsed" / f"{d.doc_id}.json"
        if out.exists():
            continue
        src, dst = cfg.path("corpus_dir") / d.path, corpus / d.path
        if a.kind != "orig" and not dst.exists():
            scan = root / "full" / "corpus" / d.path
            if not scan.exists():
                full_copy(src, scan, a.dpi)
            if a.kind == "mixed":
                mixed_copy(src, scan, dst)
        chars = text_chars(dst)
        row = _parse_one(scan_cfg, conv, d, out)
        print(f"{a.kind}/{a.ocr} {d.doc_id.split('/')[-1][:40]}: {len(chars)} pages, {sum(c < 50 for c in chars)} with "
              f"no text layer, parse {row['seconds']}s, status {row['status']}, chars {row.get('chars')}, "
              f"pages_with_text {row.get('pages_with_text')} {row['error'][:100]}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
