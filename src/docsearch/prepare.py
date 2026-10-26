"""Download the source datasets and build corpus/ + manifest.csv + questions.jsonl.

Each source is optional; a failed source is logged in data/prepare_log.json and skipped.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from .config import Config
from .manifest import Doc, Question, file_unit, page_unit, validate, write_manifest, write_questions

UA = "Mozilla/5.0 (docsearch research test; +https://huggingface.co/datasets)"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
OCB_REPO = "microsoft/OfficeComprehensionBenchmark"
FINSLIDES_REPO = "ibm-research/REAL-MM-RAG_FinSlides"

# OCB "File Fidelity" questions are written for a single, already-open file. Many only make sense
# with the file in hand (fonts, colours, "slide 4", "cell B12"). For pooled retrieval we keep only
# content questions that could plausibly identify their file. Rules are recorded in the README.
OCB_DROP_FEATURES = {
    "Font", "Highlight", "Header", "Footer", "Header/Footer/SlideNum/DateTime", "Strikeout",
    "Comment", "Metadata", "Language", "Layout", "Position/Association/Grouping", "Animations",
    "Ink", "Video", "Icons", "Slide Properties - Hidden", "Tables / Cell color coding",
    "Text Highlight", "Hyperlink", "Image", "Slide Note", "Formula",
}  # fmt: skip
OCB_DROP_PATTERN = re.compile(
    r"\b(this|the) (document|presentation|deck|workbook|file|spreadsheet|sheet)\b"
    r"|\bslide \d+|\bpage \d+|\bsection \d+|\bitem #?\s?\d+|\bfootnote #?\s?\d+|\bfont\b"
    r"|\bcolou?r\b|\bhighlight|\bhow many (equations|tables|images|slides|sheets|comments|charts|pages)"
    r"|text ?box|\bcell [A-Z]+\d|\b[A-Z]{1,2}\d{1,4}\b|\bfirst (header|footer|slide|table)",
    re.I,
)
OCB_SUFFIX = "Do not use web search"


def _log(log: dict, source: str, **kw) -> None:
    log.setdefault(source, {}).update(kw)
    print(f"[prepare] {source}: {kw}", flush=True)


def _fetch(url: str, dest: Path, timeout: int = 30, deadline_s: int = 120, max_bytes: int = 0) -> int:
    """Download with a socket timeout, an overall deadline, and an optional size cap."""
    ua = UA if "wikimedia" in url else BROWSER_UA
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 16):
            n += len(chunk)
            f.write(chunk)
            if max_bytes and n > max_bytes:
                raise ValueError(f"larger than {max_bytes} bytes")
            if time.time() - t0 > deadline_s:
                raise TimeoutError(f"exceeded {deadline_s}s")
    return n


# ---------------------------------------------------------------- ViDoRe V3 (PDF, page gold)
def prep_vidore(cfg: Config, log: dict) -> tuple[list[Doc], list[Question]]:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    s = cfg["sampling"]
    rng = random.Random(s["seed"])
    docs: list[Doc] = []
    qs: list[Question] = []
    for corpus in s["vidore_corpora"]:
        raw = cfg.path("data_dir") / "raw" / corpus
        snapshot_download(f"vidore/{corpus}", repo_type="dataset", local_dir=raw)
        meta = pq.read_table(raw / "documents_metadata/test-00000-of-00001.parquet").to_pandas()
        corp = pq.read_table(
            raw / "corpus/test-00000-of-00001.parquet",
            columns=["corpus_id", "doc_id", "page_number_in_doc"],
        ).to_pandas()
        queries = pq.read_table(raw / "queries/test-00000-of-00001.parquet").to_pandas()
        qrels = pq.read_table(
            raw / "qrels/test-00000-of-00001.parquet", columns=["query_id", "corpus_id", "score"]
        ).to_pandas()
        out = cfg.path("corpus_dir") / "pdf" / corpus
        out.mkdir(parents=True, exist_ok=True)
        for m in meta.itertuples():
            src = raw / "pdfs" / m.file_name
            if not src.exists():
                _log(log, corpus, missing_pdf=m.file_name)
                continue
            shutil.copy2(src, out / m.file_name)
            docs.append(
                Doc(
                    doc_id=f"{corpus}/{m.doc_id}",
                    path=f"pdf/{corpus}/{m.file_name}",
                    format="pdf",
                    source=corpus,
                    license=str(m.license).upper(),
                    origin_url=m.url,
                    pages=int(m.page_number),
                    bytes=src.stat().st_size,
                )
            )
        # page_number_in_doc is 0-based in the corpus parquet; Docling pages are 1-based.
        base = 0 if corp.page_number_in_doc.min() == 0 else 1
        cmap = {
            r.corpus_id: page_unit(f"{corpus}/{r.doc_id}", r.page_number_in_doc - base + 1) for r in corp.itertuples()
        }
        gold: dict[int, dict[str, int]] = defaultdict(dict)
        for r in qrels.itertuples():
            gold[r.query_id][cmap[r.corpus_id]] = int(r.score)
        en = queries[queries.language == "english"].sort_values("query_id")
        n_all = len(en)
        ids = list(en.query_id)
        rng.shuffle(ids)
        keep = set(ids[: s["vidore_queries_per_corpus"]])
        for r in en.itertuples():
            if r.query_id not in keep:
                continue
            qs.append(
                Question(
                    qid=f"{corpus}:{r.query_id}",
                    query=r.query,
                    source=corpus,
                    format="pdf",
                    level="page",
                    gold=gold[r.query_id],
                    meta={
                        "query_types": list(r.query_types),
                        "query_format": r.query_format,
                        "answer": r.answer,
                    },
                )
            )
        _log(
            log,
            corpus,
            ok=True,
            docs=len(meta),
            pages=int(meta.page_number.sum()),
            english_queries=n_all,
            sampled_queries=len(keep),
            page_base=base,
        )
    return docs, qs


# ------------------------------------------------- REAL-MM-RAG FinSlides (slide images, page gold)
def prep_finslides(cfg: Config, log: dict) -> tuple[list[Doc], list[Question]]:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    s = cfg["sampling"]
    raw = cfg.path("data_dir") / "raw" / "finslides"
    snapshot_download(FINSLIDES_REPO, repo_type="dataset", local_dir=raw)
    t = pq.read_table(raw / "data/test-00000-of-00001.parquet").to_pandas()
    t["deck"] = t.image_filename.str.replace(r"_page_\d+\.png$", "", regex=True)
    decks = sorted(t.deck.unique())
    rng = random.Random(s["seed"])
    chosen = sorted(rng.sample(decks, min(s["finslides_decks"], len(decks))))
    t = t[t.deck.isin(chosen)]
    out = cfg.path("corpus_dir") / "slides" / "finslides"
    out.mkdir(parents=True, exist_ok=True)
    docs: list[Doc] = []
    for row in t.drop_duplicates("image_filename").itertuples():
        p = out / row.image_filename
        if not p.exists():
            p.write_bytes(row.image["bytes"])
        docs.append(
            Doc(
                doc_id=f"finslides/{Path(row.image_filename).stem}",
                path=f"slides/finslides/{row.image_filename}",
                format="slide_image",
                source="finslides",
                license="CDLA-Permissive-2.0",
                origin_url=f"https://huggingface.co/datasets/{FINSLIDES_REPO}",
                pages=1,
                bytes=p.stat().st_size,
                note=f"deck={row.deck}",
            )
        )
    gold: dict[str, dict[str, int]] = defaultdict(dict)
    for row in t[t["query"].notna()].itertuples():
        gold[row.query][page_unit(f"finslides/{Path(row.image_filename).stem}", 1)] = 1
    qs = [
        Question(
            qid=f"finslides:{i}",
            query=q,
            source="finslides",
            format="slide_image",
            level="page",
            gold=g,
        )
        for i, (q, g) in enumerate(sorted(gold.items()))
    ]
    _log(
        log,
        "finslides",
        ok=True,
        decks_total=len(decks),
        decks_chosen=chosen,
        slides=len(docs),
        queries=len(qs),
    )
    return docs, qs


# ------------------------------------------ Office Comprehension Bench (docx/xlsx/pptx, file gold)
def ocb_filter(df):
    """Keep single-file File Fidelity questions about content (see OCB_* rules above)."""
    df = df[(df.track != "Domain") & (df.reference_files.map(len) == 1)].copy()
    df["file"] = df.reference_files.map(lambda x: list(x)[0])
    df["q"] = df.question.str.split(OCB_SUFFIX).str[0].str.strip()
    df = df[~df.feature.isin(OCB_DROP_FEATURES)]
    df = df[~df.q.str.contains(OCB_DROP_PATTERN)]
    return df[df.q.str.split().map(len) >= 6]


def prep_ocb(cfg: Config, log: dict) -> tuple[list[Doc], list[Question]]:
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    raw = cfg.path("data_dir") / "raw" / "ocb"
    for f in ["README.md", "NOTICES.md", "data/ocb_qna_data.parquet", "data/ocb_source_urls.parquet"]:
        hf_hub_download(OCB_REPO, f, repo_type="dataset", local_dir=raw)
    qna = pd.read_parquet(raw / "data/ocb_qna_data.parquet")
    urls = pd.read_parquet(raw / "data/ocb_source_urls.parquet")
    # Size caps: tabular files explode into thousands of row chunks, so they get a tight cap;
    # docx/pptx size is mostly embedded images, so their cap is loose.
    max_bytes = int(cfg["sampling"]["ocb_max_file_mb"] * 1e6)
    max_doc_bytes = int(cfg["sampling"].get("ocb_max_office_doc_mb", 40) * 1e6)
    hosted = {
        s.rfilename.split("/", 1)[1]: s.size
        for s in HfApi().dataset_info(OCB_REPO, files_metadata=True).siblings
        if s.rfilename.startswith("reference_files/")
    }
    out = cfg.path("corpus_dir") / "office" / "ocb"
    out.mkdir(parents=True, exist_ok=True)
    docs: list[Doc] = []
    skipped: dict[str, list[str]] = defaultdict(list)

    def add(name: str, path: Path, lic: str, origin: str) -> None:
        ext = path.suffix.lower()
        fmt = {".xlsm": "xlsx"}.get(ext, ext.lstrip("."))
        docs.append(
            Doc(
                doc_id=f"ocb/{name}",
                path=f"office/ocb/{name}",
                format=fmt,
                source="ocb",
                license=lic,
                origin_url=origin,
                bytes=path.stat().st_size,
            )
        )

    for name, size in sorted(hosted.items()):
        if size and size > max_bytes:
            skipped["too_large"].append(name)
            continue
        dest = out / name
        if not dest.exists():
            p = hf_hub_download(OCB_REPO, f"reference_files/{name}", repo_type="dataset", local_dir=raw)
            shutil.copy2(p, dest)
        add(name, dest, "CDLA-Permissive-2.0 (OCB-hosted)", f"https://huggingface.co/datasets/{OCB_REPO}")

    # URL-referenced native Office files (no conversion needed). Fetched from the original
    # publisher under the publisher's terms; used locally for evaluation, never redistributed.
    native = urls[(urls.convert == "no") & urls.original_format.isin(["docx", "pptx", "xlsx"])]
    for r in native.itertuples():
        dest = out / r.filename
        if r.filename in hosted:
            continue
        if not dest.exists():
            try:
                print(f"[prepare] ocb fetch {r.filename}", flush=True)
                cap = max_doc_bytes if r.original_format in ("docx", "pptx") else max_bytes
                _fetch(r.url, dest, max_bytes=cap)
                if not _looks_like_zip(dest):
                    dest.unlink()
                    skipped["not_office_file"].append(r.filename)
                    continue
            except Exception as e:  # noqa: BLE001
                dest.unlink(missing_ok=True)
                kind = "too_large" if "larger than" in str(e) else "download_failed"
                skipped[kind].append(f"{r.filename}: {type(e).__name__}: {str(e)[:80]}")
                continue
            time.sleep(0.3)
        add(r.filename, dest, "Publisher terms (URL-referenced by OCB; local eval only)", r.url)
    for r in urls[urls.convert == "yes"].itertuples():
        skipped["needs_pdf_conversion"].append(r.filename)

    have = {d.doc_id for d in docs}
    fdf = ocb_filter(qna)
    fmt_of = {d.doc_id: d.format for d in docs}
    qs = []
    for i, r in enumerate(fdf.itertuples()):
        did = f"ocb/{r.file}"
        if did not in have:
            continue
        qs.append(
            Question(
                qid=f"ocb:{i}",
                query=r.q,
                source="ocb",
                format=fmt_of[did],
                level="file",
                gold={file_unit(did): 1},
                meta={"feature": r.feature, "track": r.track},
            )
        )
    _log(
        log,
        "ocb",
        ok=True,
        questions_total=len(qna),
        questions_after_filter=len(fdf),
        questions_kept=len(qs),
        files=len(docs),
        skipped={k: len(v) for k, v in skipped.items()},
        skipped_files=dict(skipped),
    )
    return docs, qs


def _looks_like_zip(p: Path) -> bool:
    with p.open("rb") as f:
        return f.read(2) == b"PK"


# ------------------------------------------------------ Public-domain images (labeled by us)
def prep_images(cfg: Config, log: dict) -> tuple[list[Doc], list[Question]]:
    """Images listed in sources/images.json, each with our own hand-written questions."""
    spec_path = Path(__file__).resolve().parents[2] / "sources" / "images.json"
    if not spec_path.exists():
        _log(log, "pd_images", ok=False, error="sources/images.json missing")
        return [], []
    spec = json.loads(spec_path.read_text())
    out = cfg.path("corpus_dir") / "images" / "pd"
    out.mkdir(parents=True, exist_ok=True)
    docs, qs, failed = [], [], []
    for item in spec:
        dest = out / item["file"]
        if not dest.exists():
            try:
                if item.get("render_pdf"):
                    _render_pdf_page(item["url"], item.get("page", 1), dest)
                else:
                    _fetch(item["url"], dest)
                    _normalise_image(dest)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{item['file']}: {type(e).__name__}: {e}")
                dest.unlink(missing_ok=True)
                continue
        did = f"pd_images/{Path(item['file']).stem}"
        docs.append(
            Doc(
                doc_id=did,
                path=f"images/pd/{item['file']}",
                format="image",
                source="pd_images",
                license=item["license"],
                origin_url=item.get("page_url", item["url"]),
                pages=1,
                bytes=dest.stat().st_size,
                note=item.get("kind", ""),
            )
        )
        for j, q in enumerate(item["questions"]):
            qs.append(
                Question(
                    qid=f"pd_images:{Path(item['file']).stem}:{j}",
                    query=q,
                    source="pd_images",
                    format="image",
                    level="page",
                    gold={page_unit(did, 1): 1},
                    meta={"kind": item.get("kind", "")},
                )
            )
    _log(log, "pd_images", ok=not failed or bool(docs), images=len(docs), questions=len(qs), failed=failed)
    return docs, qs


def _normalise_image(p: Path) -> None:
    """Re-save GIF/other inputs as real PNG/JPEG so every OCR backend can read them."""
    from PIL import Image

    want = "PNG" if p.suffix.lower() == ".png" else "JPEG"
    with Image.open(p) as im:
        if im.format == want:
            return
        img = im.convert("RGB")
    img.save(p, format=want)


def _render_pdf_page(url: str, page: int, dest: Path) -> None:
    """Rasterise one PDF page at 150 dpi (simulates a scanned form)."""
    import tempfile

    import pypdfium2 as pdfium

    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "src.pdf"
        _fetch(url, pdf)
        doc = pdfium.PdfDocument(str(pdf))
        img = doc[page - 1].render(scale=150 / 72).to_pil().convert("L")
        img.save(dest)


SOURCES = {
    "vidore": prep_vidore,
    "finslides": prep_finslides,
    "ocb": prep_ocb,
    "images": prep_images,
}


def prepare(cfg: Config, only: list[str] | None = None) -> dict:
    data = cfg.path("data_dir")
    data.mkdir(parents=True, exist_ok=True)
    log_path = data / "prepare_log.json"
    log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
    parts_dir = data / "parts"
    parts_dir.mkdir(exist_ok=True)
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            docs, qs = fn(cfg, log)
            (parts_dir / f"{name}.json").write_text(
                json.dumps({"docs": [d.__dict__ for d in docs], "questions": [q.__dict__ for q in qs]})
            )
        except Exception as e:  # noqa: BLE001
            _log(log, name, ok=False, error=f"{type(e).__name__}: {e}")
        log.setdefault(name, {})["seconds"] = round(time.time() - t0, 1)
        log_path.write_text(json.dumps(log, indent=2, default=str))
    # Merge all parts into the manifest + question set.
    docs, qs = [], []
    for p in sorted(parts_dir.glob("*.json")):
        part = json.loads(p.read_text())
        docs += [Doc(**d) for d in part["docs"]]
        qs += [Question(**q) for q in part["questions"]]
    problems = validate(docs, qs)
    write_manifest(cfg.path("corpus_dir") / "manifest.csv", docs)
    write_questions(data / "questions.jsonl", qs)
    log["_merged"] = {"docs": len(docs), "questions": len(qs), "problems": problems[:50]}
    log_path.write_text(json.dumps(log, indent=2, default=str))
    return log
