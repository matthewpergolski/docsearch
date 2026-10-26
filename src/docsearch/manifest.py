"""Corpus manifest + question set: IDs and the mapping from retrieved chunks to gold units.

Gold is defined at page level ("<doc_id>#p<page>") or file level ("<doc_id>"), never at chunk
level, so different chunking strategies are scored on the same targets.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_FIELDS = [
    "doc_id",
    "path",
    "format",
    "source",
    "license",
    "origin_url",
    "pages",
    "bytes",
    "note",
]

# Map file extensions to the format tag used in reports.
EXT_FORMAT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",  # parsed as the .docx a converter writes (docconv); needs LibreOffice or textutil
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "csv",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
}


@dataclass
class Doc:
    doc_id: str
    path: str  # relative to corpus_dir
    format: str
    source: str
    license: str
    origin_url: str = ""
    pages: int = 0
    bytes: int = 0
    note: str = ""


@dataclass
class Question:
    qid: str
    query: str
    source: str
    format: str  # format of the gold document(s)
    level: str  # "page" or "file"
    gold: dict[str, int]  # unit -> graded relevance (>=1)
    meta: dict = field(default_factory=dict)


def page_unit(doc_id: str, page: int) -> str:
    return f"{doc_id}#p{int(page)}"


def file_unit(doc_id: str) -> str:
    return doc_id


def unit_doc(unit: str) -> str:
    return unit.split("#p", 1)[0]


_SLIDE = re.compile(r"_page_(\d+)$")


def file_of(unit: str, fmt: str) -> str:
    """The file a unit or doc_id of format `fmt` belongs to: its doc_id, except a slide image (one file per slide),
    whose file is its deck, the name before "_page_N". Any other format keeps a name that ends in "_page_N"."""
    doc = unit.split("#")[0]
    return _SLIDE.sub("", doc) if fmt == "slide_image" else doc


def file_page(doc_id: str, pages: list[int] | None, fmt: str) -> tuple[str, int | None]:
    """(file, page) a chunk of format `fmt` shows as: a slide image is page N of its deck; other chunks are their
    first page, None for a file without pages (docx, csv)."""
    slide = _SLIDE.search(doc_id) if fmt == "slide_image" else None
    if slide:
        return file_of(doc_id, fmt), int(slide.group(1))
    return doc_id, (min(pages) if pages else None)


def chunk_units(doc_id: str, pages: list[int] | None, level: str) -> list[str]:
    """Units a retrieved chunk counts toward. A chunk spanning pages counts for each page."""
    if level == "file":
        return [file_unit(doc_id)]
    if not pages:
        # Single-page items (images, slides) or formats without pagination: page 1.
        return [page_unit(doc_id, 1)]
    return [page_unit(doc_id, p) for p in sorted(set(pages))]


def ranked_units(hits: list[dict], level: str) -> list[str]:
    """Collapse a ranked chunk list to a ranked, de-duplicated unit list (first occurrence wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        for u in chunk_units(h["doc_id"], h.get("pages"), level):
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def write_manifest(path: Path, docs: list[Doc]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for d in sorted(docs, key=lambda d: d.doc_id):
            w.writerow(asdict(d))


def read_manifest(path: Path) -> list[Doc]:
    with path.open(newline="") as f:
        return [Doc(**{**r, "pages": int(r["pages"] or 0), "bytes": int(r["bytes"] or 0)}) for r in csv.DictReader(f)]


def write_questions(path: Path, qs: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in qs:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def read_questions(path: Path) -> list[Question]:
    with path.open() as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def validate(docs: list[Doc], qs: list[Question]) -> list[str]:
    """Return problems: duplicate doc ids, gold pointing at unknown docs or out-of-range pages."""
    problems: list[str] = []
    by_id: dict[str, Doc] = {}
    for d in docs:
        if d.doc_id in by_id:
            problems.append(f"duplicate doc_id {d.doc_id}")
        by_id[d.doc_id] = d
    for q in qs:
        if not q.gold:
            problems.append(f"{q.qid}: no gold")
        for u in q.gold:
            d = by_id.get(unit_doc(u))
            if d is None:
                problems.append(f"{q.qid}: gold {u} not in manifest")
            elif "#p" in u and d.pages and int(u.rsplit("#p", 1)[1]) > d.pages:
                problems.append(f"{q.qid}: gold {u} beyond page count {d.pages}")
    return problems
