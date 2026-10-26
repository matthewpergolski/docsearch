"""Incremental sync: bring parsed docs, chunks, text vectors and page vectors in line with a folder.

State lives in a small SQLite manifest (paths.sync_manifest, default work/manifest.sqlite), one row per source
file: path, size, mtime, sha256, parsed_at, doc_id, status. Files are compared by sha256, never by name or mtime.

Per-file status moves pending -> parsed -> chunked -> ok (or failed). Each step is idempotent and the status is
written after the step, so a crashed or interrupted sync resumes where it stopped. Deletions are marked
`deleting` first, then removed from LanceDB, the page store, the parsed/render artifacts and meta.jsonl, then
dropped. Each parsed file gets a meta.jsonl row (title, author, dates from inside the file); the rows are written
once, after the file loop, also when the loop stops early.

A 0-byte file is not parsed (an indexed file that became empty leaves the index). A .doc file is parsed through a
converted .docx (docconv); with no converter it is skipped and keeps no state row, so a later sync picks it up.
The sync summary names every file it does not parse.

The quality gate (`--gate`) evaluates the working index and promotes it to live only if nDCG@10 does not drop by
more than sync.gate_tolerance versus the live version. "Live" is a pointer file (index_dir/live.json) holding the
LanceDB table version(s) and a hard-linked snapshot of the page-vector store; `search`/`answer` read live.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import ui
from .config import Config
from .docconv import drop_cached, for_want_of_converter
from .ingest import file_sha256
from .manifest import EXT_FORMAT, Doc, read_manifest, unit_doc, write_manifest

STATUSES = ("pending", "parsed", "chunked", "ok", "failed", "deleting")
SCHEMA = """CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, root TEXT, size INTEGER, mtime REAL, sha256 TEXT, parsed_at TEXT,
    doc_id TEXT, format TEXT, status TEXT, error TEXT, updated_at TEXT)"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ state
class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(SCHEMA)
        self.db.commit()

    def rows(self, root: str | None = None) -> dict[str, dict]:
        q, a = ("SELECT * FROM files WHERE root = ?", (root,)) if root else ("SELECT * FROM files", ())
        return {r["path"]: dict(r) for r in self.db.execute(q, a)}

    def upsert(self, row: dict) -> None:
        row = {**row, "updated_at": _now()}
        cols = ", ".join(row)
        self.db.execute(
            f"INSERT INTO files ({cols}) VALUES ({', '.join('?' * len(row))}) "
            f"ON CONFLICT(path) DO UPDATE SET {', '.join(f'{k}=excluded.{k}' for k in row if k != 'path')}",
            list(row.values()),
        )
        self.db.commit()

    def set(self, path: str, **kw) -> None:
        kw["updated_at"] = _now()
        self.db.execute(f"UPDATE files SET {', '.join(f'{k}=?' for k in kw)} WHERE path=?", [*kw.values(), path])
        self.db.commit()

    def delete(self, path: str) -> None:
        self.db.execute("DELETE FROM files WHERE path=?", (path,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


# ------------------------------------------------------------------ plan
@dataclass
class FileInfo:
    path: str  # absolute
    size: int
    mtime: float
    sha256: str
    doc: Doc
    adopt_status: str = "ok"


@dataclass
class Plan:
    new: list[FileInfo] = field(default_factory=list)
    changed: list[FileInfo] = field(default_factory=list)
    resume: list[FileInfo] = field(default_factory=list)  # same content, an earlier sync stopped mid-way
    unchanged: list[FileInfo] = field(default_factory=list)
    adopted: list[FileInfo] = field(default_factory=list)  # already parsed before the manifest existed
    deleted: list[dict] = field(default_factory=list)  # state rows
    ignored: list[str] = field(default_factory=list)  # unsupported extensions
    empty: list[str] = field(default_factory=list)  # 0-byte files of a supported type: nothing to parse
    skipped: list[str] = field(default_factory=list)  # .doc files with work to do but no converter (skip_reason)
    skip_reason: str = ""
    root: str = ""

    def counts(self) -> dict[str, int]:
        return {k: len(getattr(self, k)) for k in ("new", "changed", "resume", "deleted", "unchanged", "adopted",
                                                   "ignored", "empty", "skipped")}  # fmt: skip

    @property
    def work(self) -> list[FileInfo]:
        return self.new + self.changed + self.resume


def _rel_manifest_path(cfg: Config, p: Path) -> str:
    corpus = cfg.path("corpus_dir").resolve()
    try:
        return p.relative_to(corpus).as_posix()
    except ValueError:
        return str(p)  # outside corpus_dir: stored absolute (corpus_dir / absolute == absolute)


def scan(cfg: Config, root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Supported files under root as (absolute path, format); plus ignored paths."""
    files, ignored = [], []
    skip_dirs = {".git", "__pycache__"}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in skip_dirs and not d.startswith("."))
        for fn in sorted(fns):
            if fn.startswith(".") or fn == "manifest.csv":
                continue
            p = Path(dp) / fn
            fmt = EXT_FORMAT.get(p.suffix.lower())
            (files.append((p.resolve(), fmt)) if fmt else ignored.append(str(p)))
    return files, ignored


def make_plan(cfg: Config, state: State, root: Path) -> Plan:
    from .docconv import pick
    from .ingest import parsed_path

    root = root.resolve()
    corpus = {(cfg.path("corpus_dir") / d.path).resolve(): d for d in _read_corpus(cfg)}
    parse_log = _read_parse_log(cfg)
    rows = state.rows(str(root))
    files, ignored = scan(cfg, root)
    plan = Plan(ignored=ignored, root=str(root))
    source = cfg["sync"].get("source") or root.name
    converter, plan.skip_reason = pick(cfg["ingest"]["doc_converter"])
    seen = set()
    for p, fmt in files:
        st = p.stat()
        if st.st_size == 0:  # a cloud placeholder or a file mid-rewrite: an indexed version stays (CR2)
            plan.empty.append(str(p))
            if str(p) in rows:
                seen.add(str(p))
            continue
        sha = file_sha256(p)
        d = corpus.get(p)
        if d is None:
            rel = p.relative_to(root).as_posix()
            d = Doc(doc_id=f"{source}/{rel}", path=_rel_manifest_path(cfg, p), format=fmt, source=source,
                    license="", bytes=st.st_size, note="added by sync")  # fmt: skip
        fi = FileInfo(str(p), st.st_size, st.st_mtime, sha, d)
        seen.add(str(p))
        row = rows.get(str(p))
        is_doc = p.suffix.lower() == ".doc"
        if row is None:
            log = parse_log.get(d.doc_id)
            retry = is_doc and converter is not None and for_want_of_converter(log.get("error", "") if log else "")
            # Parsed before the manifest existed (ok, partial, empty or failed): adopt as-is rather than re-parse.
            if log and not retry and (log.get("sha256") or sha) == sha and (parsed_path(cfg, d.doc_id).exists() or
                                                                            log.get("status") == "failed"):  # fmt: skip
                fi.adopt_status = "ok" if parsed_path(cfg, d.doc_id).exists() else "failed"
                plan.adopted.append(fi)
                continue
            todo = plan.new
        elif row["sha256"] != sha:
            todo = plan.changed
        elif row["status"] in ("ok", "failed") and not (is_doc and _doc_again(cfg, row, converter)):
            plan.unchanged.append(fi)
            continue
        elif row["status"] in ("ok", "failed"):
            todo = plan.changed
        elif row["status"] == "deleting":
            todo = plan.changed  # came back before its delete finished: treat as changed
        else:
            todo = plan.resume
        # Seen: an indexed version stays until it converts. A .doc already parsed needs no converter to finish.
        if converter is None and is_doc and (todo is not plan.resume or row["status"] == "pending"):
            plan.skipped.append(str(p))
            continue
        todo.append(fi)
    plan.deleted = [r for p, r in rows.items() if p not in seen]
    return plan


def _doc_again(cfg: Config, row: dict, converter) -> bool:
    """Parse an unchanged .doc again: it failed only for want of a converter and one is here now, or textutil
    read it (tables lost) and LibreOffice is here now (CR2)."""
    from .docconv import converted_with

    if converter is None:
        return False
    if row["status"] == "failed":
        return for_want_of_converter(row.get("error") or "")
    have = converted_with(cfg.path("cache_dir") / "doc_docx", row["sha256"])
    return converter.name == "soffice" and "textutil" in have and "soffice" not in have


def notes(plan: Plan) -> list[str]:
    """The files a sync does not parse, by name: ignored ones by extension, empty ones, skipped .doc files."""
    root = Path(plan.root)

    def names(tag: str, paths: list[str]) -> list[str]:
        rel = [Path(p).relative_to(root).as_posix() if Path(p).is_relative_to(root) else p for p in paths]
        more = [f"  ... and {len(rel) - 20} more"] if len(rel) > 20 else []
        return [f"  {tag:8s} {r}" for r in rel[:20]] + more

    out = []
    if plan.ignored:
        by_ext = Counter(Path(p).suffix.lower() or "(no extension)" for p in plan.ignored)
        top = sorted(by_ext.items(), key=lambda kv: (-kv[1], kv[0]))
        out.append(f"ignored {len(plan.ignored)}: " + ", ".join(f"{ext} {n}" for ext, n in top))
    if plan.empty:
        out += [f"empty (0 bytes), skipped: {len(plan.empty)}", *names("empty", plan.empty)]
    if plan.skipped:
        n = len(plan.skipped)
        out += [f"{n} .doc file{'' if n == 1 else 's'} skipped: {plan.skip_reason}", *names("skipped", plan.skipped)]
    return out


def _read_corpus(cfg: Config) -> list[Doc]:
    p = cfg.path("corpus_dir") / "manifest.csv"
    return read_manifest(p) if p.exists() else []


def _read_parse_log(cfg: Config) -> dict[str, dict]:
    import csv

    p = cfg.path("parsed_dir") / "parse_log.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["doc_id"]: r for r in csv.DictReader(f)}


# ------------------------------------------------------------------ operations
class Ops:
    """The real side effects. Tests substitute a fake with the same methods."""

    def __init__(self, cfg: Config, full: bool = False):
        from .index import connect, table_name
        from .subset import load_subset, subset_path

        self.cfg, self.full = cfg, full
        from .chunk import SLIDE_CTX

        self.variants = [v for v in cfg["sync"]["variants"]]
        self.db = connect(cfg)
        # +enrich (H2): the slide variant, kept from the hybrid chunks after each change; None with enrich.llm off
        self.enrich = SLIDE_CTX if cfg["enrich"]["llm"] != "off" and "hybrid" in self.variants else None
        self.tables = {v: table_name(v) for v in self.variants + ([self.enrich] if self.enrich else [])}
        self.sub = load_subset(cfg, full)
        self.subset_path = subset_path(cfg)
        self.visual = cfg["visual"].get("default")
        self._conv = None
        self._chunkers: dict = {}
        self._meta: list = []  # rows of parsed files, written once by flush_meta

    # corpus manifest (+ box subset) registration, so full rebuilds see synced files too
    def register(self, docs: list[Doc]) -> None:
        cur = {d.doc_id: d for d in _read_corpus(self.cfg)}
        if all(cur.get(d.doc_id) == d for d in docs):
            return
        cur.update({d.doc_id: d for d in docs})
        write_manifest(self.cfg.path("corpus_dir") / "manifest.csv", list(cur.values()))
        if self.sub is not None:
            add = [d.doc_id for d in docs if d.doc_id not in set(self.sub["docs"])]
            if add:
                self.sub["docs"] = sorted(set(self.sub["docs"]) | set(add))
                _write_json(self.subset_path, self.sub)

    def unregister(self, doc_ids: set[str]) -> None:
        cur = _read_corpus(self.cfg)
        if any(d.doc_id in doc_ids for d in cur):
            write_manifest(self.cfg.path("corpus_dir") / "manifest.csv", [d for d in cur if d.doc_id not in doc_ids])
        if self.sub is not None and doc_ids & set(self.sub["docs"]):
            self.sub["docs"] = [x for x in self.sub["docs"] if x not in doc_ids]
            _write_json(self.subset_path, self.sub)

    def parse(self, d: Doc) -> dict:
        from .ingest import _parse_one, _write_log, build_converter, parsed_path, pdf_ocr_on

        if self._conv is None:
            self._conv = build_converter(pdf_ocr=pdf_ocr_on(self.cfg), device=self.cfg["models"]["device"])
        row = _parse_one(self.cfg, self._conv, d, parsed_path(self.cfg, d.doc_id))
        _write_log(self.cfg.path("parsed_dir") / "parse_log.csv", {d.doc_id: row})
        return row

    def chunk(self, d: Doc) -> int:
        from .chunk import DocChunker, chunks_path, replace_doc_chunks

        n = 0
        for v in self.variants:
            if v not in self._chunkers:
                self._chunkers[v] = DocChunker(self.cfg, v, self.sub)
            chunks, _status, _ = self._chunkers[v](d)
            replace_doc_chunks(chunks_path(self.cfg, v), {d.doc_id}, chunks)
            n += len(chunks)
        return n

    def unindex_text(self, doc_ids: set[str]) -> None:
        from .chunk import chunks_path, replace_doc_chunks

        names = set(self.db.table_names())
        for v, name in self.tables.items():
            replace_doc_chunks(chunks_path(self.cfg, v), doc_ids, [])
            if name in names:
                ids = sorted(doc_ids)
                for i in range(0, len(ids), 200):
                    q = ", ".join("'" + x.replace("'", "''") + "'" for x in ids[i : i + 200])
                    self.db.open_table(name).delete(f"doc_id IN ({q})")

    def unindex_pages(self, doc_ids: set[str]) -> None:
        from .altembed import _done_ids, drop_ids, store_dir
        from .visual import pages_dir

        pj = pages_dir(self.cfg) / "pages.jsonl"
        if pj.exists():
            rows = [json.loads(x) for x in pj.open()]
            _write_jsonl(pj, [r for r in rows if r["doc_id"] not in doc_ids])
        if self.visual:
            d = store_dir(self.cfg, self.visual, "pages")
            if d.exists():
                drop_ids(d, {u for u in _done_ids(d) if unit_doc(u) in doc_ids})

    def render(self, d: Doc) -> int:
        from .visual import office_pdf_dir, pages_dir, render_doc

        if not self.visual:
            return 0
        for p in (pages_dir(self.cfg) / d.doc_id, office_pdf_dir(self.cfg, d.doc_id)):
            shutil.rmtree(p, ignore_errors=True)  # changed file: never reuse the old page images
        want = (self.sub or {}).get("pages", {}).get(d.doc_id)
        rows = render_doc(self.cfg, d, want) or []
        pj = pages_dir(self.cfg) / "pages.jsonl"
        old = [json.loads(x) for x in pj.open()] if pj.exists() else []
        _write_jsonl(pj, [r for r in old if r["doc_id"] != d.doc_id] + rows)
        return len(rows)

    def meta(self, d: Doc) -> None:
        from .meta import extract

        self._meta.append(extract(self.cfg.path("corpus_dir"), d))

    def flush_meta(self) -> None:
        from .meta import update_meta

        if self._meta:
            update_meta(self.cfg, self._meta)
            self._meta = []

    def remove_artifacts(self, doc_ids: set[str]) -> None:
        from .ingest import parsed_path
        from .meta import update_meta
        from .visual import office_pdf_dir, pages_dir

        for did in doc_ids:
            parsed_path(self.cfg, did).unlink(missing_ok=True)
            for p in (pages_dir(self.cfg) / did, office_pdf_dir(self.cfg, did)):
                shutil.rmtree(p, ignore_errors=True)
        log = self.cfg.path("parsed_dir") / "parse_log.csv"
        rows = _read_parse_log(self.cfg)
        if doc_ids & set(rows):
            import csv

            from .ingest import PARSE_LOG_FIELDS

            tmp = log.with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PARSE_LOG_FIELDS)
                w.writeheader()
                for k in sorted(set(rows) - doc_ids):
                    w.writerow({c: rows[k].get(c, "") for c in PARSE_LOG_FIELDS})
            tmp.replace(log)
        update_meta(self.cfg, [], drop=doc_ids)
        self.unregister(doc_ids)

    def index(self) -> dict:
        """Embed whatever is missing: text chunks per variant, then page images. Both resume by id."""
        from .index import build_index
        from .visual import build_visual_store, pages_dir

        out = {}
        for v in self.variants:
            out[v] = build_index(self.cfg, v)["embedded_now"]
        if self.enrich:
            out["enrich_calls"] = self.enrich_slides()["calls"]
            out[self.enrich] = build_index(self.cfg, self.enrich)["embedded_now"]  # unchanged chunks reuse hybrid
        if self.visual and (pages_dir(self.cfg) / "pages.jsonl").exists():
            out["pages"] = build_visual_store(self.cfg, self.visual)["embedded_now"]
        return out

    def enrich_stale(self) -> bool:
        """With enrich.llm on: the slide variant is missing or older than the hybrid chunks."""
        from .chunk import chunks_path

        if not self.enrich:
            return False
        hy, ctx = chunks_path(self.cfg, "hybrid"), chunks_path(self.cfg, self.enrich)
        return hy.exists() and (not ctx.exists() or hy.stat().st_mtime_ns > ctx.stat().st_mtime_ns)

    def enrich_slides(self) -> dict:
        """The hybrid chunks with an LLM context line on each slide image -> the slide variant. Replies are cached
        in cache_dir/enrich (the H2 trial's cache), so an unchanged slide costs no call. Returns Enricher.stats."""
        from .chunk import chunks_path, read_chunks
        from .enrich import Enricher, apply
        from .llm import check_available, get_client

        e = self.cfg["enrich"]
        client = get_client(self.cfg, "enrich", backend=e["backend"], provider=e["provider"], model=e["model"])
        check_available(client.s)
        enricher = Enricher(client, f"{e['provider']}:{e['model']}", self.cfg.path("cache_dir") / "enrich")
        chunks = read_chunks(chunks_path(self.cfg, "hybrid"))
        slides = [c for c in chunks if c.format == "slide_image"]
        lines = enricher.lines(slides)
        _write_jsonl(chunks_path(self.cfg, self.enrich), [c.__dict__ for c in apply(chunks, lines)])
        s = enricher.stats
        ui.say(f"[sync] enrich: {len(lines)} of {len(slides)} slide images have a line; {s['calls']} LLM calls "
               f"({e['provider']}:{e['model']}), {s['cached_batches']} cached, {s['seconds']:.0f}s")  # fmt: skip
        return s

    def candidate(self) -> dict:
        names = set(self.db.table_names())
        return {"tables": {n: self.db.open_table(n).version for n in self.tables.values() if n in names},
                "visual": self.visual}  # fmt: skip


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(p)


# ------------------------------------------------------------------ apply
def apply(cfg: Config, state: State, plan: Plan, ops) -> dict:
    """Execute a plan. Safe to re-run after a crash: each row's status says which steps are left."""
    root_rows = state.rows()
    out = {"parsed": 0, "failed": 0, "failed_files": [], "chunks": 0, "pages": 0, "deleted": 0, "embedded": {}}
    for fi in plan.adopted:
        state.upsert({"path": fi.path, "root": plan.root, "size": fi.size, "mtime": fi.mtime,
                      "sha256": fi.sha256, "parsed_at": "", "doc_id": fi.doc.doc_id, "format": fi.doc.format,
                      "status": fi.adopt_status, "error": ""})  # fmt: skip
    if plan.deleted:
        ids = {r["doc_id"] for r in plan.deleted}
        for r in plan.deleted:
            state.set(r["path"], status="deleting")
            drop_cached(cfg.path("cache_dir") / "doc_docx", r["sha256"])
        ui.info(f"[sync] removing {len(ids)} deleted docs")
        ops.unindex_text(ids)
        ops.unindex_pages(ids)
        ops.remove_artifacts(ids)
        for r in plan.deleted:
            state.delete(r["path"])
        out["deleted"] = len(plan.deleted)
    work = plan.work
    if work:
        ops.register([fi.doc for fi in work])
    pb = ui.progress(len(work), "sync files", "files ").start()
    for fi in work:
        old = root_rows.get(fi.path)
        if old and old["sha256"] != fi.sha256:
            drop_cached(cfg.path("cache_dir") / "doc_docx", old["sha256"])
    try:
        _apply_work(state, plan, ops, work, root_rows, out, pb)
    finally:
        ops.flush_meta()  # one meta.jsonl write per sync; a sync that stops keeps the rows of files it finished
    pb.close()
    pending = [p for p, r in state.rows().items() if r["status"] == "chunked"]
    if pending or plan.deleted or getattr(ops, "enrich_stale", lambda: False)():
        out["embedded"] = ops.index()
        for p in pending:
            state.set(p, status="ok")
    return out


def _apply_work(state: State, plan: Plan, ops, work: list[FileInfo], root_rows: dict, out: dict, pb) -> None:
    for fi in work:
        row = root_rows.get(fi.path)
        if row is None or row["sha256"] != fi.sha256 or row["status"] in ("deleting", "ok", "failed"):
            state.upsert({"path": fi.path, "root": plan.root, "size": fi.size, "mtime": fi.mtime,
                          "sha256": fi.sha256, "parsed_at": "", "doc_id": fi.doc.doc_id, "format": fi.doc.format,
                          "status": "pending", "error": ""})  # fmt: skip
            status = "pending"
        else:
            status = row["status"]
        did = fi.doc.doc_id
        if status == "pending":
            r = ops.parse(fi.doc)
            if r.get("status") == "failed":
                ops.unindex_text({did})
                ops.unindex_pages({did})
                state.set(fi.path, status="failed", error=str(r.get("error", ""))[:300])
                out["failed"] += 1
                out["failed_files"].append((did, str(r.get("error", ""))[:200]))
                pb.advance()
                continue
            state.set(fi.path, status="parsed", parsed_at=_now(), error="")
            out["parsed"] += 1
            status = "parsed"
        if status == "parsed":
            ops.meta(fi.doc)
            ops.unindex_text({did})  # old rows of a changed file; chunk ids can repeat with new text
            out["chunks"] += ops.chunk(fi.doc)
            ops.unindex_pages({did})
            out["pages"] += ops.render(fi.doc)
            state.set(fi.path, status="chunked")
        pb.advance()


# ------------------------------------------------------------------ live pointer + gate
def live_path(cfg: Config) -> Path:
    return cfg.path("index_dir") / "live.json"


def read_live(cfg: Config) -> dict | None:
    p = live_path(cfg)
    return json.loads(p.read_text()) if p.exists() else None


def promote(cfg: Config, cand: dict, ndcg: float | None, reason: str) -> dict:
    """Point live at `cand`: LanceDB table versions plus a hard-linked snapshot of the page store."""
    from .altembed import store_dir
    from .visual import pages_dir

    live = {"tables": cand["tables"], "visual": cand.get("visual"), "visual_snapshot": None, "ndcg": ndcg,
            "promoted_at": _now(), "reason": reason}  # fmt: skip
    alias = cand.get("visual")
    src = store_dir(cfg, alias, "pages") if alias else None
    if src and src.exists():
        snaps = cfg.path("index_dir") / "live"
        dst = snaps / f"{alias}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        dst.mkdir(parents=True)
        for f in src.glob("shard_*"):
            if f.suffix in (".json", ".npy"):
                os.link(f, dst / f.name)  # store files are never rewritten in place, so links are safe
        pj = pages_dir(cfg) / "pages.jsonl"
        if pj.exists():
            shutil.copy2(pj, dst / "pages.jsonl")
        live["visual_snapshot"] = str(dst.relative_to(cfg.path("index_dir")))
    old = read_live(cfg)
    _write_json(live_path(cfg), live)
    keep = {live["visual_snapshot"], (old or {}).get("visual_snapshot")}
    snaps = cfg.path("index_dir") / "live"
    if snaps.exists():
        for d in snaps.iterdir():
            if str(d.relative_to(cfg.path("index_dir"))) not in keep:
                shutil.rmtree(d, ignore_errors=True)
    return live


def follow_working(cfg: Config) -> bool:
    """After a direct `index`/`all` rebuild: if a live pointer exists, move it to the working index (ungated),
    so search never keeps serving a stale version just because the rebuild did not go through `sync`."""
    live = read_live(cfg)
    if live is None:
        return False
    cand = Ops(cfg, full=True).candidate()
    if cand["tables"] == live["tables"]:
        return False
    promote(cfg, cand, None, "index rebuilt outside sync")
    ui.say(f"live pointer -> {cand['tables']} (index rebuilt; ungated)")
    return True


def gate(cfg: Config, cand: dict, eval_fn, tolerance: float) -> tuple[bool, dict]:
    """eval_fn(live_or_None) -> nDCG@10 (None = the working index). Returns (promoted, info)."""
    live = read_live(cfg)
    if live and live["tables"] == cand["tables"] and live.get("ndcg") is not None:
        return True, {"note": "working index is already live", "live_ndcg": live["ndcg"]}
    new = eval_fn(None)
    if live is None:
        promote(cfg, cand, new, "first gated version")
        return True, {"new_ndcg": new, "live_ndcg": None, "note": "no live version yet; promoted"}
    base = live.get("ndcg")
    if base is None:
        base = eval_fn(live)
    info = {"new_ndcg": new, "live_ndcg": base, "delta": round(new - base, 4), "tolerance": tolerance}
    if new >= base - tolerance:
        promote(cfg, cand, new, f"gate pass: {new:.4f} vs {base:.4f}")
        return True, info
    return False, info


def eval_ndcg(cfg: Config, live: dict | None, full: bool) -> float:
    from .eval import run_eval
    from .visual import page_store_ready

    alias = cfg["visual"].get("default")
    if alias and not page_store_ready(cfg, alias, live):  # CR2: as in search, text only on an empty page store
        ui.info(f"[sync] page store {alias} is empty: the gate eval runs text only", flush=True)
        alias = None
    mode = "hybrid_visual_fa" if alias else "hybrid"
    name = "sync_gate_live" if live else "sync_gate_candidate"
    res = run_eval(cfg, [(name, "hybrid", mode, True)], notes="sync --gate", full=full,
                   visual=alias, live=live)  # fmt: skip
    return float(res[0]["ndcg@10"])


# ------------------------------------------------------------------ entry points
def sync(cfg: Config, root: Path | None = None, dry_run: bool = False, use_gate: bool = False, full: bool = False,
         ops=None, eval_fn=None) -> dict:  # fmt: skip
    root = (root or cfg.path("corpus_dir")).resolve()
    if not root.is_dir():
        raise SystemExit(f"sync: {root} is not a directory")
    state = State(cfg.path("sync_manifest"))
    try:
        t0 = time.time()
        plan = make_plan(cfg, state, root)
        c = plan.counts()
        ui.say(f"sync {root}: " + ", ".join(f"{v} {k}" for k, v in c.items() if v or k in ("new", "changed",
               "deleted", "unchanged")))  # fmt: skip
        for line in notes(plan):
            ui.say(line)
        result: dict = {"counts": c, "dry_run": dry_run}
        if dry_run:
            for k in ("new", "changed", "resume"):
                for fi in getattr(plan, k)[:20]:
                    ui.say(f"  {k:8s} {fi.doc.doc_id}")
            for r in plan.deleted[:20]:
                ui.say(f"  deleted  {r['doc_id']}")
            return result
        ops = ops or Ops(cfg, full)
        result.update(apply(cfg, state, plan, ops))
        cand = ops.candidate()
        live = read_live(cfg)
        if use_gate:
            fn = eval_fn or (lambda lv: eval_ndcg(cfg, lv, full))
            ok, info = gate(cfg, cand, fn, float(cfg["sync"]["gate_tolerance"]))
            result["gate"] = {"promoted": ok, **info}
            if ok:
                ui.say(f"gate: live -> {cand['tables']} ({info})")
            else:
                ui.say(f"gate: HELD, live stays {live['tables']}: nDCG {info['new_ndcg']:.4f} vs live "
                       f"{info['live_ndcg']:.4f} (tolerance {info['tolerance']})")  # fmt: skip
        elif live is not None and live["tables"] != cand["tables"]:
            promote(cfg, cand, None, "ungated sync")  # live already in use: follow the working index
            result["promoted"] = True
        result["seconds"] = round(time.time() - t0, 1)
        ui.say(f"sync done in {result['seconds']}s: parsed {result['parsed']}, failed {result['failed']}, "
               f"deleted {result['deleted']}, embedded {result['embedded']}")  # fmt: skip
        for did, err in result["failed_files"]:
            ui.say(f"  failed   {did}: {err}")
        return result
    finally:
        state.close()


def watch(cfg: Config, root: Path | None, interval: int, use_gate: bool, full: bool) -> int:
    ui.say(f"watching {root or cfg.path('corpus_dir')} every {interval}s (Ctrl-C to stop)")
    try:
        while True:
            try:
                sync(cfg, root, use_gate=use_gate, full=full)
            except Exception as e:  # noqa: BLE001 - keep watching; the error is in the run log
                ui.logger.exception("sync failed")
                ui.say(f"sync failed: {type(e).__name__}: {e} (retrying next interval)")
            time.sleep(interval)
    except KeyboardInterrupt:
        ui.say("watch stopped")
        return 0
