"""GET /api/config and GET /api/search: the search `docsearch search --json` runs (display.run), same ranking.
/api/search also gives the query's terms (search.query_terms), the words the snippets were cut around.
GET /api/file: a result's file for the browser to show; only files in corpus/manifest.csv.
GET /api/search/stream: the same search as lines of JSON, with the results that are not slides first.
GET /api/text: the text of a page of a result, for a preview where there is no picture.
GET /api/page: the picture `docsearch render` made of a result's page; only pages in work/pages/pages.jsonl.

One search runs at a time: the models share one GPU, and they stay loaded between searches (models.keep_loaded).
warm() loads them when the server starts, so the first search does not wait for them. A newer search from the same
browser tab (header X-Docsearch-Tab) stops that tab's search still running, which then answers 409: the page searches
as the user types. The page reranker checks between pages. Searches from other tabs wait their turn.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import queue
import threading
from datetime import date
from pathlib import Path

from django.http import FileResponse, StreamingHttpResponse
from ninja import NinjaAPI, Query
from ninja.errors import HttpError

from .. import display
from ..chunk import search_variant
from ..meta import Filter

CFG = None  # set by serve(); the config found from the working folder otherwise
_ONE_AT_A_TIME = threading.Lock()
_SEARCHES = [0]  # searches that arrived
_NEWEST: dict[str, int] = {}  # per browser tab, the number of its last search; its older searches stop
_COUNT = threading.Lock()
READY = threading.Event()  # set when warm() has loaded the models (or failed: a search then loads them itself)
log = logging.getLogger("docsearch.web")
api = NinjaAPI(title="docsearch", version="1", urls_namespace="docsearch-api")
_REPEATED = Query(None)  # ?formats=pdf&formats=docx


def _cfg():
    global CFG
    if CFG is None:
        from ..config import load_config

        CFG = load_config()
    return CFG


def known_formats() -> list[str]:
    from ..manifest import EXT_FORMAT

    return sorted({*EXT_FORMAT.values(), "slide_image"})


def _day(name: str, s: str | None) -> str | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        raise HttpError(400, f"{name}: expected a date as YYYY-MM-DD, got {s!r}") from None


def warm(cfg) -> None:
    """Load every model a search uses (the embedder, the text reranker, the page-picture query encoder and the
    page reranker, which loads its weights on first use), then run the first example search once."""
    from .. import altembed, models

    try:
        with _ONE_AT_A_TIME:
            models.load_embedder(cfg)
            models.load_reranker(cfg)
            if vis := cfg["visual"].get("default"):
                altembed.query_encoder(cfg, vis)
                if page := cfg["search"].get("page_reranker"):
                    models.load_page_reranker(cfg, page).model  # noqa: B018 - the property loads the weights
            # one real search: the first run of each model pays one-time GPU costs (first search 8 s, then 1 s)
            display.run(cfg, str((cfg["web"]["examples"] or ["warm up"])[0]), agent=False)
    except Exception:  # noqa: BLE001 - the server still runs; the first search loads what it needs and says why
        log.exception("loading the models at start failed")
    finally:
        READY.set()


@api.get("/config")
def config(request):
    from ..meta import read_meta

    cfg = _cfg()
    return {"agent": cfg["agent"]["mode"] == "on", "formats": known_formats(), "ready": READY.is_set(),
            "examples": [str(q) for q in cfg["web"]["examples"]],
            "filters": bool(read_meta(cfg))}  # fmt: skip  # filters need file metadata (docsearch meta)


def _ask(request, q: str, k: int, formats, after, before, folder, undated: bool, agent: bool):
    """Check a search's input (400 with the reason), and number it for its browser tab: a newer search from the
    same tab makes replaced() true."""
    cfg = _cfg()
    if not q.strip():
        raise HttpError(400, "the query is empty")
    if not 1 <= k <= 100:
        raise HttpError(400, "k must be 1 to 100")
    if bad := sorted(set(formats or ()) - set(known_formats())):
        raise HttpError(400, f"unknown format: {', '.join(bad)} (known: {', '.join(known_formats())})")
    if agent and cfg["agent"]["mode"] != "on":
        raise HttpError(400, "the agent step is off (agent.mode in docsearch.yaml)")
    flt = Filter(tuple(formats or ()), _day("after", after), _day("before", before), folder or None, undated)
    tab = request.headers.get("X-Docsearch-Tab")
    with _COUNT:
        _SEARCHES[0] += 1
        me = _SEARCHES[0]
        if tab:
            _NEWEST[tab] = me

    def replaced() -> bool:
        return bool(tab) and _NEWEST.get(tab) != me

    return cfg, flt, replaced


def _run(cfg, q: str, k: int, flt, agent: bool, replaced, progress=None, on_page=None, started=None) -> dict:
    """One search, after the one running; the answer /api/search gives. started() runs when its turn comes."""
    from .. import models
    from ..search import query_terms

    with _ONE_AT_A_TIME:
        if started is not None:
            started()
        try:
            if replaced():
                raise models.Superseded
            models.stop_when(replaced)
            models.on_page(on_page)
            res, notes = display.run(cfg, q.strip(), k=k, flt=flt, agent=agent, progress=progress)
        except display.SearchRefused as e:
            raise HttpError(400, str(e)) from None
        except models.Superseded:
            raise HttpError(409, "a newer search replaced this one") from None
        finally:
            models.stop_when(None)
            models.on_page(None)
    return {"results": display.as_json(res), "notes": notes, "terms": sorted(query_terms(q))}


@api.get("/search")
def search(request, q: str, k: int = 10, formats: list[str] = _REPEATED, after: str | None = None,
           before: str | None = None, folder: str | None = None, undated: bool = False, agent: bool = False):  # fmt: skip
    cfg, flt, replaced = _ask(request, q, k, formats, after, before, folder, undated, agent)
    return _run(cfg, q, k, flt, agent, replaced)


@api.get("/search/stream")
def search_stream(request, q: str, k: int = 10, formats: list[str] = _REPEATED, after: str | None = None,
                  before: str | None = None, folder: str | None = None, undated: bool = False,
                  agent: bool = False):  # fmt: skip
    """The search of /api/search as lines of JSON, for the page to show results as they come. When the page
    reranker has slides to read (the slow step), first {"event": "early"}: the results that are not slides, which
    keep their order (the page reranker only swaps slides among the slots they hold; sent only when no later step
    can move them, search.early_safe); then {"event": "pages"}
    after each slide read; last {"event": "done"} with the answer of /api/search, or {"event": "error"}."""
    cfg, flt, replaced = _ask(request, q, k, formats, after, before, folder, undated, agent)
    from ..manifest import file_page
    from ..meta import read_meta
    from ..search import SearchConfig, early_safe, query_terms

    sc = SearchConfig.from_cfg(cfg)
    slow, show_early = set(sc.page_rerank_formats), early_safe(sc)
    events: queue.Queue = queue.Queue()
    early: list = []

    def progress(kind, hits):
        if kind == "candidates":
            files: dict[str, set] = {}
            for h in hits[: sc.rerank_candidates]:
                fmt = h.get("format", "")
                files.setdefault(fmt, set()).add(file_page(h["doc_id"], h.get("pages"), fmt)[0])
            events.put({"event": "stage", "stage": "ranking", "files": {f: len(v) for f, v in files.items()}})
        elif kind == "ranked" and show_early:
            early[:] = [[h for h in hits[:k] if h.get("format") not in slow]]

    def on_page(units, done):
        if early:
            hits = early.pop()
            names = display.first_lines(cfg, search_variant(cfg), {h["doc_id"] for h in hits})
            res = display.build(q.strip(), hits, read_meta(cfg), None, names=names)
            events.put({"event": "early", "results": display.as_json(res), "terms": sorted(query_terms(q))})
        pages = [{"unit": u, "file": u.rsplit("#p", 1)[0], "page": int(u.rsplit("#p", 1)[1])} for u in units]
        events.put({"event": "pages", "pages": pages, "done": list(done)})

    def work():
        if _ONE_AT_A_TIME.locked():
            events.put({"event": "waiting", "why": "another search" if READY.is_set() else "loading"})
        try:
            started = lambda: events.put({"event": "stage", "stage": "looking"})  # noqa: E731
            events.put({"event": "done", **_run(cfg, q, k, flt, agent, replaced, progress, on_page, started)})
        except HttpError as e:
            events.put({"event": "error", "status": e.status_code, "detail": str(e)})
        except Exception:  # noqa: BLE001 - the stream ends with the reason; the log has the traceback
            log.exception("search failed")
            events.put({"event": "error", "status": 500, "detail": "the search failed; see the server log"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True, name="search").start()
    lines = (json.dumps(e) + "\n" for e in iter(events.get, None))
    return StreamingHttpResponse(lines, content_type="application/x-ndjson")


PAGE_TEXT_PARTS = 60  # chunks a page-text preview shows at most


@api.get("/text")
def page_text(request, file: str, page: int | None = None):
    """The text the index holds for one page of a result (a slide of a deck: that slide), as parts (title, body)
    in reading order, for a preview where there is no page picture. Spreadsheet bodies keep their rows (" / ")
    and cells (" | ")."""
    from ..search import page_rows

    cfg = _cfg()
    docs = {file} | ({f"{file}_page_{page}"} if page is not None else set())
    rows = [r for r in page_rows(cfg, search_variant(cfg), docs) if r["doc_id"] in docs]
    deck = [r for r in rows if r["doc_id"] != file]
    rows = deck or [r for r in rows if page is None or not r.get("pages") or page in r["pages"]]
    if not rows:
        raise HttpError(404, f"no indexed text for {file}" + (f" page {page}" if page is not None else ""))

    def order(r):
        n = r["chunk_id"].rsplit(":", 1)[-1]
        return int(n) if n.isdigit() else 1 << 30

    fmt = rows[0].get("format", "")
    parts = []
    for r in sorted(rows, key=order)[:PAGE_TEXT_PARTS]:
        title, body = display.title_and_body(r.get("text") or "", fmt)
        parts.append({"title": title, "body": body})
    return {"format": fmt, "parts": parts}


# Formats a browser shows as they are; docx and pptx show as the PDF copy the page render made, if there is one.
_INLINE = {"pdf", "image", "slide_image", "csv"}


def _file_path(cfg, file: str, page: int | None):
    from ..manifest import read_manifest
    from ..visual import OFFICE_RENDER, office_pdf_dir

    man = cfg.path("corpus_dir") / "manifest.csv"
    docs = {d.doc_id: d for d in read_manifest(man)} if man.exists() else {}
    d = docs.get(file) or (docs.get(f"{file}_page_{page}") if page is not None else None)  # a slide of a deck
    if d is None:
        return None, False
    if d.format in OFFICE_RENDER:
        copy = office_pdf_dir(cfg, d.doc_id) / (Path(d.path).stem + ".pdf")
        if copy.is_file():
            return copy, True
    return cfg.path("corpus_dir") / d.path, d.format in _INLINE


@api.get("/file")
def file(request, file: str, page: int | None = None):
    path, inline = _file_path(_cfg(), file, page)
    if path is None or not path.is_file():
        raise HttpError(404, f"not an indexed file: {file}")
    kind = "text/plain; charset=utf-8" if path.suffix.lower() == ".csv" else mimetypes.guess_type(path.name)[0]
    r = FileResponse(path.open("rb"), as_attachment=not inline, filename=path.name,
                     content_type=kind or "application/octet-stream")  # fmt: skip
    # Every indexed type (pdf, png, jpg, tif, csv as text, office as a download) is inert; nosniff keeps it so. No
    # CSP sandbox: Chrome then refuses to show a PDF.
    r["X-Content-Type-Options"] = "nosniff"
    return r


@api.get("/page")
def page(request, file: str, page: int = 1):
    from ..visual import pages_dir

    store = pages_dir(_cfg()) / "pages.jsonl"
    want = {f"{file}#p{page}", f"{file}_page_{page}#p1"}  # a page of a file, or a slide of a deck
    found = None
    if store.exists():
        with store.open() as f:
            found = next((Path(r["path"]) for r in map(json.loads, f) if r["unit"] in want), None)
    if found is None or not found.is_file():
        raise HttpError(404, f"no picture of {file} page {page}")
    r = FileResponse(found.open("rb"), content_type=mimetypes.guess_type(found.name)[0] or "image/png")
    r["X-Content-Type-Options"] = "nosniff"
    r["Cache-Control"] = "private, max-age=3600"
    return r
