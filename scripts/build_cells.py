"""Phase C1: build work/cells.sqlite, one long table of table cells, from the corpus. Rebuildable; touches no other
file. Spreadsheets (xlsx, csv) come from the original files (openpyxl, data_only: formulas give their cached
values; hidden sheets skipped and logged). PDF, Word and PowerPoint tables come from Docling (work/parsed),
spans expanded. No stitching across pages.

  uv run python scripts/build_cells.py            # all documents in corpus/manifest.csv
  uv run python scripts/build_cells.py --limit 20 # a quick look

Prints coverage per format: tables, cells, share numeric, share fused, mean quality (filled share of the grid).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from openpyxl.utils import get_column_letter

from docsearch.cells import SCHEMA, cells_path, csv_grid, docling_grid, grid_cells, table_quality, xlsx_sheets
from docsearch.config import load_config
from docsearch.manifest import read_manifest

FROM_DOCLING = {"pdf", "docx", "pptx"}


def tables_of(doc, corpus: Path, parsed: Path, log) -> list[dict]:
    """One dict per table: table_id, page, sheet, caption, grid, a1 (True when the grid starts at sheet cell A1)."""
    if doc.format == "xlsx":
        sheets, hidden = xlsx_sheets(corpus / doc.path)
        for name in hidden:
            log(f"{doc.doc_id}: hidden sheet skipped: {name}")
        return [{"table_id": f"{doc.doc_id}#s{page}", "page": page, "sheet": name, "caption": None, "grid": grid,
                 "a1": True} for page, name, grid in sheets]  # fmt: skip
    if doc.format == "csv":
        text = (corpus / doc.path).read_bytes()
        try:
            s = text.decode("utf-8-sig")
        except UnicodeDecodeError:
            s = text.decode("latin-1")
        return [{"table_id": f"{doc.doc_id}#csv", "page": 1, "sheet": None, "caption": None, "grid": csv_grid(s),
                 "a1": True}]  # fmt: skip
    path = parsed / f"{doc.doc_id}.json"
    if not path.exists():
        log(f"{doc.doc_id}: no Docling parse")
        return []
    d = json.loads(path.read_text())
    texts = d.get("texts") or []

    def ref_text(ref: str) -> str:
        kind, _, i = ref.lstrip("#/").partition("/")
        return texts[int(i)].get("text", "") if kind == "texts" and i.isdigit() and int(i) < len(texts) else ""

    out = []
    for i, t in enumerate(d.get("tables") or []):
        prov = t.get("prov") or []
        caption = " ".join(ref_text(c.get("$ref", "")) for c in t.get("captions") or []).strip() or None
        out.append({"table_id": f"{doc.doc_id}#t{i}", "page": prov[0].get("page_no", 1) if prov else 1, "sheet": None,
                    "caption": caption, "grid": docling_grid(t["data"]), "a1": False})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="first N documents of the manifest (a look; still writes the file)")
    a = ap.parse_args()
    cfg = load_config()
    corpus, parsed = cfg.path("corpus_dir"), cfg.path("parsed_dir")
    out = cells_path(cfg)
    tmp = out.with_suffix(".sqlite.tmp")
    tmp.unlink(missing_ok=True)
    docs = [d for d in read_manifest(corpus / "manifest.csv") if d.format in FROM_DOCLING | {"xlsx", "csv"}]
    docs = docs[: a.limit] if a.limit else docs
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg)

    stats: dict[str, dict] = defaultdict(lambda: {"docs": 0, "tables": 0, "cells": 0, "num": 0, "fused": 0, "q": 0.0})
    t0 = time.time()
    for doc in docs:
        try:
            tables = tables_of(doc, corpus, parsed, log)
        except Exception as e:  # noqa: BLE001 - one broken file must not stop the build; it is logged
            log(f"{doc.doc_id}: skipped: {type(e).__name__}: {e}")
            continue
        s = stats[doc.format]
        s["docs"] += 1
        for t in tables:
            n_rows, n_cols, q = table_quality(t["grid"])
            if not n_rows or not n_cols:
                continue
            cells = grid_cells(t["grid"])
            db.execute("INSERT INTO tables VALUES (?,?,?,?,?,?,?,?,?)",
                       (t["table_id"], doc.doc_id, doc.format, t["page"], t["sheet"], n_rows, n_cols, q, t["caption"]))  # fmt: skip
            db.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           [(t["table_id"], doc.doc_id, doc.format, t["page"], t["sheet"], c["row"], c["col"],
                             f"{get_column_letter(c['col'])}{c['row']}" if t["a1"] else None, c["text"], c["num"],
                             c["row_label"], c["col_header"], c["section"], int(c["fused"])) for c in cells])  # fmt: skip
            s["tables"] += 1
            s["cells"] += len(cells)
            s["num"] += sum(c["num"] is not None for c in cells)
            s["fused"] += sum(c["fused"] for c in cells)
            s["q"] += q
    db.commit()
    db.close()
    os.replace(tmp, out)
    log_path = cfg.root / cfg["logging"]["dir"] / "build_cells.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(f"{x}\n" for x in log_lines))
    print(f"\n{out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s); log: {log_path}")
    print(f"{'format':7s} {'docs':>5s} {'tables':>7s} {'cells':>9s} {'numeric':>8s} {'fused':>7s} {'quality':>8s}")
    for fmt, s in sorted(stats.items()):
        n, c = s["tables"], s["cells"]
        print(
            f"{fmt:7s} {s['docs']:5d} {n:7d} {c:9d} {s['num'] / c if c else 0:8.1%} {s['fused'] / c if c else 0:7.1%} "
            f"{s['q'] / n if n else 0:8.2f}"
        )


if __name__ == "__main__":
    main()
