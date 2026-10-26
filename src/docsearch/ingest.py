"""Parse every manifest document with Docling and store DoclingDocument JSON + a parse log."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

from . import ui
from .config import Config
from .manifest import Doc, read_manifest

PARSE_LOG_FIELDS = [
    "doc_id",
    "format",
    "source",
    "status",
    "seconds",
    "pages_expected",
    "pages_parsed",
    "pages_with_text",
    "items",
    "tables",
    "chars",
    "error",
    "sha256",  # of the source file when parsed; a changed file with the same name is re-parsed
    "pdf_ocr",  # "on" when a PDF was parsed with ingest.pdf_ocr; turning the flag on re-parses the others
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_current(src: Path, out: Path, row: dict | None) -> bool:
    """Parsed output is up to date with the source file (same sha256 as when it was parsed).

    Rows from logs written before the sha256 column existed are trusted once and backfilled with the current
    hash, so copying a parsed corpus to another machine (new mtimes) never triggers a full re-parse; from then
    on any content change is caught."""
    if not row or row.get("status") != "ok" or not out.exists() or not src.exists():
        return False
    sha = file_sha256(src)
    if not row.get("sha256"):
        row["sha256"] = sha
        return True
    return row["sha256"] == sha


def parsed_path(cfg: Config, doc_id: str) -> Path:
    return cfg.path("parsed_dir") / f"{doc_id}.json"


def pdf_ocr_on(cfg: Config) -> bool:
    """M22: ingest.pdf_ocr. Off: a PDF gives only its embedded text layer, so a scanned page gives no text."""
    return bool((cfg.raw.get("ingest") or {}).get("pdf_ocr", False))


def accelerator(device: str | None = None):
    """Docling layout/table/OCR models run on the shared device choice (config.pick_device)."""
    from docling.datamodel.pipeline_options import AcceleratorOptions

    from .config import pick_device

    return AcceleratorOptions(device=pick_device(device or "auto"))


def pdf_options(timeout_s: float = 600, pdf_ocr: bool = False, full_page: bool = False, device: str | None = None):
    """Docling options for PDFs. With pdf_ocr, Docling runs OCR region by region, only where a page region has
    no text layer (a scanned page inside a typed PDF, a pasted picture); typed text keeps its text layer.
    full_page: OCR every page whole instead (the fallback when the region mode kills the parser)."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

    pdf = PdfPipelineOptions()
    pdf.accelerator_options = accelerator(device)
    pdf.do_ocr = pdf_ocr
    if pdf_ocr:
        # default mode: PDF-aware layout regions, not the full page
        pdf.ocr_options = RapidOcrOptions(force_full_page_ocr=True) if full_page else RapidOcrOptions()
        timeout_s = max(timeout_s, 3600)  # OCR of a long scan takes minutes
    pdf.do_table_structure = True
    pdf.document_timeout = timeout_s
    return pdf


OCR_MODES = ("region", "full", "off")  # tried in this order; each next one is the fallback for a crash


def quiet_crash() -> None:
    """In a parse worker: a crash inside a native library (seen: a segfault in RapidOCR on one scanned page)
    ends the process with the signal number as its exit code. The parent reads the code and falls back; on
    macOS no "Python quit unexpectedly" dialog opens."""
    import ctypes
    import signal

    libc = ctypes.CDLL(None)
    for sig in (signal.SIGSEGV, signal.SIGBUS, signal.SIGABRT, signal.SIGILL):
        libc.signal(int(sig), ctypes.cast(libc._exit, ctypes.c_void_p))


def parse_worker(src: str, out: str, mode: str, device: str | None = None) -> None:
    """Child process: parse one PDF with the OCR mode and write {status, errors, doc} as JSON to `out`."""
    quiet_crash()
    conv = build_converter(pdf_ocr=mode != "off", pdf_full_page=mode == "full", device=device)
    res = conv.convert(Path(src), raises_on_error=False)
    Path(out).write_text(json.dumps({
        "status": str(getattr(res.status, "value", res.status)).lower(),
        "errors": [str(e.error_message)[:200] for e in (res.errors or [])],
        "doc": res.document.export_to_dict(),
    }))  # fmt: skip


def parse_pdf_isolated(
    src: Path, run=None, timeout_s: float = 7200, device: str | None = None
) -> tuple[dict | None, str]:
    """M22 crash guard: parse a PDF with OCR in a child process, so that a crash in the OCR library fails one
    attempt and not the whole ingest run. Modes in OCR_MODES order: region OCR; full-page OCR; no OCR.
    Returns (the worker's payload or None, a note that names each mode that failed)."""
    import subprocess
    import sys
    import tempfile

    from . import config

    run = run or subprocess.run
    dev = config.pick_device(device or "auto")  # resolved here and passed to the child on its command line
    notes = []
    for mode in OCR_MODES:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "doc.json"
            cmd = [
                sys.executable,
                "-m",
                "docsearch.ingest",
                "--parse-worker",
                "--device",
                dev,
                str(src),
                str(out),
                mode,
            ]
            try:
                code = run(cmd, capture_output=True, timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            if code == 0 and out.exists():
                return json.loads(out.read_text()), "; ".join(notes)
            notes.append(f"ocr mode {mode} failed (exit {code})")
    return None, "; ".join(notes)


def build_converter(
    ocr_lang: list[str] | None = None,
    timeout_s: float = 600,
    pdf_ocr: bool = False,
    pdf_full_page: bool = False,
    device: str | None = None,
):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

    # Born-digital PDFs: use the embedded text layer, no OCR (fast). Table structure on. M22: pdf_ocr.
    pdf = pdf_options(timeout_s, pdf_ocr, pdf_full_page, device)
    # Images (slides, charts, scanned forms): full-page OCR.
    img = PdfPipelineOptions()
    img.accelerator_options = pdf.accelerator_options
    img.do_ocr = True
    # Pinned to RapidOCR (onnxruntime) so a CPU-only machine and an Apple-silicon laptop produce the same text.
    img.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    img.do_table_structure = True
    img.document_timeout = timeout_s
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.CSV,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=img),
        },
    )


def doc_stats(doc) -> dict:
    """Coverage stats for a DoclingDocument."""
    pages_with_text: set[int] = set()
    items = chars = 0
    for item, _lvl in doc.iterate_items(traverse_pictures=True):
        items += 1
        text = getattr(item, "text", "") or ""
        if hasattr(item, "export_to_markdown") and not text and item.__class__.__name__ == "TableItem":
            text = item.export_to_markdown(doc=doc)
        chars += len(text)
        if text.strip():
            for prov in getattr(item, "prov", []) or []:
                pages_with_text.add(prov.page_no)
    return {
        "pages_parsed": len(doc.pages),
        "pages_with_text": len(pages_with_text),
        "items": items,
        "tables": len(doc.tables),
        "chars": chars,
    }


def ingest(cfg: Config, only_source: str | None = None, force: bool = False, limit: int | None = None) -> Path:
    docs = read_manifest(cfg.path("corpus_dir") / "manifest.csv")
    if only_source:
        docs = [d for d in docs if d.source == only_source]
    if limit:
        docs = docs[:limit]
    log_path = cfg.path("parsed_dir") / "parse_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev: dict[str, dict] = {}
    if log_path.exists():
        with log_path.open() as f:
            prev = {r["doc_id"]: r for r in csv.DictReader(f)}
    conv = None
    mine: dict[str, dict] = {}
    t_all = time.time()
    pb = ui.progress(len(docs), "parse", "docs ").start()
    for i, d in enumerate(docs):
        pb.advance()
        out = parsed_path(cfg, d.doc_id)
        row = prev.get(d.doc_id)
        had_sha = bool(row and row.get("sha256"))
        needs_ocr = d.format == "pdf" and pdf_ocr_on(cfg) and (row or {}).get("pdf_ocr") != "on"
        if not force and not needs_ocr and is_current(cfg.path("corpus_dir") / d.path, out, row):
            if not had_sha:
                mine[d.doc_id] = row  # backfilled sha256 gets written to the log
            continue
        if conv is None:
            conv = build_converter(pdf_ocr=pdf_ocr_on(cfg), device=cfg["models"]["device"])
        prev[d.doc_id] = mine[d.doc_id] = _parse_one(cfg, conv, d, out)
        r = prev[d.doc_id]
        ui.info(
            f"[ingest] {i + 1}/{len(docs)} {d.doc_id} {r['status']} {r['seconds']}s "
            f"pages={r['pages_parsed']} chars={r['chars']} {r['error'][:120]}",
            flush=True,
        )
        if (i + 1) % 10 == 0:
            _write_log(log_path, mine)
    pb.close()
    _write_log(log_path, mine)
    ui.info(f"[ingest] done in {time.time() - t_all:.0f}s", flush=True)
    return log_path


def _parse_one(cfg: Config, conv, d: Doc, out: Path) -> dict:
    src = cfg.path("corpus_dir") / d.path
    row = {k: "" for k in PARSE_LOG_FIELDS}
    row.update(doc_id=d.doc_id, format=d.format, source=d.source, pages_expected=d.pages or "")
    t0 = time.time()
    if src.exists():
        row["sha256"] = file_sha256(src)
    try:
        note = ""
        if src.suffix.lower() == ".doc":  # DOC1: Docling reads the converted .docx; the doc keeps its id and path
            from .docconv import docx_for

            src = docx_for(cfg, src, row["sha256"])
        if d.format == "pdf" and pdf_ocr_on(cfg):  # M22: OCR runs in a child process (crash guard)
            from docling_core.types.doc import DoclingDocument

            payload, note = parse_pdf_isolated(src, device=cfg["models"]["device"])
            row["pdf_ocr"] = "on"
            if payload is None:
                raise RuntimeError(note)
            status, errors = payload["status"], payload["errors"]
            doc = DoclingDocument.model_validate(payload["doc"])
        else:
            res = conv.convert(src, raises_on_error=False)
            ui.quiet_libraries()  # OCR engines attach console handlers when first loaded; route them to the run log
            status = str(getattr(res.status, "value", res.status)).lower()
            errors = [str(e.error_message)[:200] for e in (res.errors or [])]
            doc = res.document
        stats = doc_stats(doc)
        if stats["chars"] == 0:
            status = "empty"
        elif status not in ("success",):
            status = f"partial:{status}"
        else:
            status = "ok"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc.export_to_dict()))
        tmp.replace(out)
        row.update(status=status, error="; ".join([*errors, *([note] if note else [])]), **stats)
    except Exception as e:  # noqa: BLE001
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:300])
    row["seconds"] = round(time.time() - t0, 2)
    return row


def _write_log(path: Path, rows: dict[str, dict]) -> None:
    # Merge with whatever is on disk so concurrent ingest runs (different sources) don't clobber
    # each other's rows; this run's rows win for its own documents.
    if path.exists():
        with path.open() as f:
            disk = {r["doc_id"]: r for r in csv.DictReader(f)}
        rows = {**disk, **rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARSE_LOG_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({f: rows[k].get(f, "") for f in PARSE_LOG_FIELDS})
    tmp.replace(path)


def restat(cfg: Config, max_mb: float = 100) -> None:
    """Recompute coverage stats in parse_log.csv from the stored JSON (e.g. after a stats fix)."""
    log_path = cfg.path("parsed_dir") / "parse_log.csv"
    with log_path.open() as f:
        rows = {r["doc_id"]: r for r in csv.DictReader(f)}
    for did, r in rows.items():
        p = parsed_path(cfg, did)
        if not p.exists() or p.stat().st_size > max_mb * 1e6:
            continue
        st = doc_stats(load_parsed(cfg, did))
        r.update(st)
        if r["status"] == "empty" and st["chars"] > 0:
            r["status"] = "ok"
    with log_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARSE_LOG_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({fld: rows[k].get(fld, "") for fld in PARSE_LOG_FIELDS})


def load_parsed(cfg: Config, doc_id: str):
    from docling_core.types.doc import DoclingDocument

    return DoclingDocument.model_validate_json(parsed_path(cfg, doc_id).read_text())


if (
    __name__ == "__main__"
):  # python -m docsearch.ingest --parse-worker --device DEV SRC OUT MODE (see parse_pdf_isolated)
    import sys

    if len(sys.argv) == 7 and sys.argv[1] == "--parse-worker" and sys.argv[2] == "--device":
        parse_worker(*sys.argv[4:], device=sys.argv[3])
