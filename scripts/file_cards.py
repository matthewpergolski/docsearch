"""Phase B0: one card per file (file name, in-file title, headings, sheet names, column headers, the start of the
text), files ranked per question by BM25 and by the default embedder on the cards, and the gold file's card rank
compared with its file rank in a reference run (the first gold-file unit in the stored top 10). No LLM. Read-only
on data/, corpus/ and the index; cards, their vectors and the card BM25 table go to work/cache/file_cards/.

  uv run python scripts/file_cards.py m28_default m28_default_hard [--out logs/b0]

A file is a doc_id, except a slide image: its file is the deck (miss_buckets.unit_file), so the wrong-file count
matches miss_buckets. Fixed = cards put the gold file first and the run does not. Lost = the run puts the gold file
first and cards do not. --out writes per-question ranks (<out>_questions.jsonl) and the cards of the wrong-file
misses' gold files (<out>_miss_cards.jsonl).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from miss_buckets import QUESTION_FILES, latest_run, read_jsonl, unit_file

from docsearch.config import load_config
from docsearch.manifest import read_manifest
from docsearch.meta import read_meta

START_CHARS = 1500
HEADINGS_CHARS = 1000
COLUMNS_CHARS = 600
TABLE_FORMATS = ("xlsx", "csv")
RRF_K = 60


@dataclass(frozen=True)
class CardField:
    name: str  # file_name | title | sheets | columns | headings | start
    source: str  # manifest | file properties | docling groups | chunks
    text: str


@dataclass(frozen=True)
class FileCard:
    file: str
    format: str
    doc_ids: tuple[str, ...]
    fields: tuple[CardField, ...]

    @property
    def text(self) -> str:
        return "\n".join(f.text for f in self.fields if f.text)


def _body(chunk_text: str) -> str:
    """Chunk text without the 'File: name' line the chunker puts first."""
    return chunk_text.split("\n", 1)[1] if chunk_text.startswith("File: ") and "\n" in chunk_text else chunk_text


def _uniq_join(items, limit: int, sep: str = " | ") -> str:
    seen: dict[str, None] = {}
    for x in items:
        x = " ".join(str(x).split())
        if x:
            seen.setdefault(x, None)
    return sep.join(seen)[:limit]


def _columns(chunk_text: str) -> str:
    """The header row of a table chunk: the line after 'Table N'."""
    lines = _body(chunk_text).split("\n")
    return lines[1] if len(lines) > 1 and lines[0].startswith("Table ") else ""


def _sheets(parsed: Path) -> list[str]:
    if not parsed.exists():
        return []
    d = json.loads(parsed.read_text())
    return [g.get("name", "") for g in d.get("groups", []) if g.get("label") == "sheet"]


def build_cards(cfg, chunks_path: Path) -> dict[str, FileCard]:
    corpus = cfg.path("corpus_dir")
    docs = {d.doc_id: d for d in read_manifest(corpus / "manifest.csv")}
    metas = {k: asdict(m) for k, m in read_meta(cfg).items()}
    if missing := docs.keys() - metas.keys():
        raise SystemExit(f"{len(missing)} of {len(docs)} files have no metadata: run `uv run docsearch meta`")
    chunks: dict[str, list[dict]] = defaultdict(list)
    for line in chunks_path.open():
        c = json.loads(line)
        chunks[c["doc_id"]].append(c)
    formats = {k: d.format for k, d in docs.items()}
    by_file: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(docs, key=lambda k: (unit_file(k, formats), docs[k].pages, _page_no(k))):
        by_file[unit_file(doc_id, formats)].append(doc_id)
    cards = {}
    for f, ids in by_file.items():
        fmt = docs[ids[0]].format
        cs = [c for i in ids for c in chunks.get(i, [])]
        name = f.rsplit("/", 1)[-1] if fmt == "slide_image" else metas[ids[0]]["file_name"]
        fields = [CardField("file_name", "manifest", name), CardField("title", "file properties",
                  _uniq_join((metas[i]["title"] for i in ids), 300))]  # fmt: skip
        if fmt in TABLE_FORMATS:
            parsed = cfg.path("parsed_dir") / f"{ids[0]}.json"
            fields.append(CardField("sheets", "docling groups", _uniq_join(_sheets(parsed), 300)))
            fields.append(CardField("columns", "chunks", _uniq_join((_columns(c["text"]) for c in cs), COLUMNS_CHARS)))
        else:
            heads = (h for c in cs for h in (c.get("meta") or {}).get("headings") or [])
            fields.append(CardField("headings", "chunks", _uniq_join(heads, HEADINGS_CHARS)))
        fields.append(CardField("start", "chunks", "\n".join(_body(c["text"]) for c in cs)[:START_CHARS]))
        cards[f] = FileCard(f, fmt, tuple(ids), tuple(fields))
    return cards


def _page_no(doc_id: str) -> int:
    tail = doc_id.rsplit("_page_", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0


def _digest(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def embed_cached(path: Path, texts: list[str], fn) -> np.ndarray:
    if path.exists():
        return np.load(path)["v"]
    v = fn(texts)
    np.savez(path, v=v)
    return v


def bm25_table(cache: Path, cards: dict[str, FileCard]):
    import lancedb
    import pyarrow as pa

    tag = _digest(*(c.file + c.text for c in cards.values()))
    db = lancedb.connect(str(cache / "lance"))
    name = f"cards_{tag}"
    if name in db.list_tables().tables:
        return db.open_table(name)
    rows = [{"file": c.file, "format": c.format, "text": c.text} for c in cards.values()]
    tbl = db.create_table(name, pa.Table.from_pylist(rows))
    tbl.create_fts_index("text", use_tantivy=False, replace=True)
    return tbl


def bm25_rank(tbl, query: str, n: int) -> list[str]:
    from lancedb.query import MatchQuery

    try:
        q = tbl.search(MatchQuery(query, "text"), query_type="fts")
        return [r["file"] for r in q.select(["file", "_score"]).limit(n).to_list()]
    except Exception:  # noqa: BLE001 - odd query syntax: no BM25 vote, as in Retriever.bm25
        return []


def rrf(lists: list[list[str]], k: int = RRF_K) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for lst in lists:
        for r, f in enumerate(lst):
            score[f] += 1 / (k + r + 1)
    return sorted(score, key=lambda f: -score[f])


def rank_of(order: list[str], gold: set[str]) -> int | None:
    """1-based rank of the first gold file; None when absent."""
    return next((i + 1 for i, f in enumerate(order) if f in gold), None)


def run_file_order(top: list[str], formats: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(unit_file(u, formats) for u in top))


def prior_order(run_order: list[str], cards: list[str], w: float, k: int = RRF_K) -> list[str]:
    """B1 form 1 at file level: the run's file order (top 10 only) plus a card prior of weight w."""
    score: dict[str, float] = defaultdict(float)
    for r, f in enumerate(run_order):
        score[f] += 1 / (k + r + 1)
    for r, f in enumerate(cards):
        score[f] += w / (k + r + 1)
    return sorted(score, key=lambda f: -score[f])


def limit_first(run_order: list[str], cards: list[str], n: int, gold: set[str]) -> bool:
    """B1 form 2 at file level, optimistic: the page search keeps only the card top-n files. Gold first when the
    first kept run file is gold, or when no run file is kept and a gold file is in the card top n (the stored top
    10 cannot say what comes next, so that case counts as fixed)."""
    keep = set(cards[:n])
    kept = [f for f in run_order if f in keep]
    return kept[0] in gold if kept else bool(gold & keep)


LOOK_WEIGHTS = (0.1, 0.25, 0.5, 1.0)
LOOK_TOPN = (3, 5, 10, 20)


RANKERS = ("bm25", "vector", "rrf", "fused_look")
LOOKS = tuple(f"prior_w{w}" for w in LOOK_WEIGHTS) + tuple(f"limit_top{n}" for n in LOOK_TOPN)


def compare(rows: list[dict], rankers: tuple[str, ...] = RANKERS) -> dict:
    """Per format and ranker: misses (run not first), fixed, hits (run first), lost."""
    out: dict = defaultdict(lambda: Counter())
    for r in rows:
        for fmt in (r["format"], "ALL"):
            key = out[fmt]
            run_first = r["run_rank"] == 1
            key["hits" if run_first else "misses"] += 1
            for name in rankers:
                card_first = r[name] == 1
                if card_first and not run_first:
                    key[f"{name}_fixed"] += 1
                if run_first and not card_first:
                    key[f"{name}_lost"] += 1
    return out


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    head = f"{'format':<12}{'misses':>7}{'hits':>6}" + "".join(f"{n + ' fix/lost':>20}" for n in RANKERS)
    print(head)
    for fmt in sorted(table, key=lambda f: (f == "ALL", f)):
        t = table[fmt]
        cells = "".join(f"{t[n + '_fixed']:>11}/{t[n + '_lost']:<8}" for n in RANKERS)
        print(f"{fmt:<12}{t['misses']:>7}{t['hits']:>6}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--chunks", default="work/chunks/hybrid.jsonl")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = load_config()
    cache = cfg.path("cache_dir") / "file_cards"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cards = build_cards(cfg, Path(a.chunks))
    files = list(cards)
    formats = {i: c.format for c in cards.values() for i in c.doc_ids}
    print(f"cards: {len(cards)} files {dict(Counter(c.format for c in cards.values()))}, {time.time() - t0:.1f} s")
    questions = {q["qid"]: q for f in QUESTION_FILES if Path(f).exists() for q in read_jsonl(Path(f))}
    runs = {name: read_jsonl(latest_run(name)) for name in a.runs}
    qtexts = sorted({questions[r["qid"]]["query"] for rows in runs.values() for r in rows})

    from docsearch.models import load_embedder

    model = cfg["models"]["embedder"]
    emb = None

    def lazy():
        nonlocal emb
        emb = emb or load_embedder(cfg)
        return emb

    t1 = time.time()
    cvec = embed_cached(cache / f"cards_{_digest(model, *(cards[f].text for f in files))}.npz",
                        [cards[f].text for f in files], lambda t: lazy().docs(t))  # fmt: skip
    t2 = time.time()
    qvec = embed_cached(cache / f"queries_{_digest(model, *qtexts)}.npz", qtexts, lambda t: lazy().queries(t))
    t3 = time.time()
    qrow = {q: i for i, q in enumerate(qtexts)}
    tbl = bm25_table(cache, cards)
    print(f"embed cards {t2 - t1:.1f} s, queries {t3 - t2:.1f} s ({len(qtexts)}; 0 s = cache)")

    bm25_ms: list[float] = []
    for name, rows in runs.items():
        out_rows = []
        for r in rows:
            q = questions[r["qid"]]
            gold = {unit_file(u, formats) for u in q["gold"]}
            s = time.perf_counter()
            b = bm25_rank(tbl, q["query"], len(files))
            bm25_ms.append((time.perf_counter() - s) * 1000)
            v = [files[i] for i in np.argsort(-(cvec @ qvec[qrow[q["query"]]]), kind="stable")]
            fused = rrf([b, v])
            run_order = run_file_order(r["top"], formats)
            out_rows.append({"qid": r["qid"], "format": r["format"], "query": q["query"], "gold_files": sorted(gold),
                             "run_rank": rank_of(run_order, gold), "bm25": rank_of(b, gold),
                             "vector": rank_of(v, gold), "rrf": rank_of(fused, gold),
                             "fused_look": rank_of(rrf([run_order, fused]), gold),
                             **{f"prior_w{w}": rank_of(prior_order(run_order, fused, w), gold) for w in LOOK_WEIGHTS},
                             **{f"limit_top{n}": 1 if limit_first(run_order, fused, n, gold) else None
                                for n in LOOK_TOPN},
                             "run_first_file": run_order[0] if run_order else "",
                             "rrf_first_file": fused[0]})  # fmt: skip
        print_table(f"{name} ({len(out_rows)} questions): fixed / lost per ranker", compare(out_rows))
        looks = compare(out_rows, LOOKS)["ALL"]
        print(
            "  looks (card rrf as a prior on the run's file order), all formats, fixed/lost: "
            + ", ".join(f"{n} {looks[n + '_fixed']}/{looks[n + '_lost']}" for n in LOOKS)
        )
        if a.out:
            Path(f"{a.out}_{name}_questions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out_rows))
            misses = [x for x in out_rows if x["run_rank"] != 1]
            Path(f"{a.out}_{name}_miss_cards.jsonl").write_text("".join(
                json.dumps({**x, "cards": {g: asdict(cards[g]) for g in x["gold_files"] if g in cards}}) + "\n"
                for x in misses))  # fmt: skip
    ms = np.array(bm25_ms)
    print(f"\ncard BM25 per query: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms (n={len(ms)})")


if __name__ == "__main__":
    main()
