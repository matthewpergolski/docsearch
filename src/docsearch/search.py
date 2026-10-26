"""Hybrid retrieval: BM25 (LanceDB native FTS) + vector search, reciprocal rank fusion, optional rerank."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import Config
from .index import connect, table_name

log = logging.getLogger("docsearch.search")

COLS = ["chunk_id", "doc_id", "format", "source", "pages", "text"]


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Ties broken by first appearance order."""
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            order.setdefault(cid, len(order))
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


@dataclass
class SearchConfig:
    bm25_k: int
    vector_k: int
    rrf_k: int
    rerank_candidates: int
    top_k: int
    rerank_min_chars: int = 0
    rerank_skip_formats: tuple[str, ...] = ()
    distinct_boost: float = 0.0  # M11: rank positions gained per distinguishing query term (0 = off)
    rerank_context: bool = False  # M12: prepend doc/page context to reranker input and rerank slides too
    late_visual_formats: tuple[str, ...] = ()  # M13: after rerank, re-fuse these formats' order with visual rank
    late_visual_k: int = 10  # M13: RRF k for the late fusion (small = strong)
    number_boost: float = 0.0  # M14: rank positions per exact query number found on the candidate
    number_boost_formats: tuple[str, ...] = ("pdf",)  # M14: formats the number boost applies to
    rerank_unit_chunks: int = 0  # M17: most chunks one page/file puts in a row in the fused list (0 = no cap)
    page_reranker: str = ""  # M16: page-image reranker alias (models.page_rerankers) or HF id ("" = off)
    page_rerank_pages: int = 10  # M16: top pages per query the page reranker scores
    page_rerank_formats: tuple[str, ...] = ("pdf", "slide_image")  # M16: formats it reorders
    visual_weights: tuple[tuple[str, float], ...] = ()  # M20: (format, weight of its visual rank in the fusion)
    period_demote: bool = False  # M21: files named for another quarter than the query's go behind the rest
    table_lookup: bool = False  # M33 (C2): a file's page whose table row and column the query names goes first

    @classmethod
    def from_cfg(cls, cfg: Config) -> SearchConfig:
        s = cfg["search"]
        special = ("rerank_skip_formats", "distinct_boost", "rerank_context", "late_visual_formats",
                   "number_boost", "number_boost_formats", "page_reranker", "page_rerank_formats",
                   "visual_weights", "period_demote", "table_lookup")  # fmt: skip
        ints = {k: int(s[k]) for k in cls.__dataclass_fields__ if k not in special}
        return cls(**ints, rerank_skip_formats=tuple(s.get("rerank_skip_formats") or ()),
                   distinct_boost=float(s.get("distinct_boost") or 0.0),
                   rerank_context=bool(s.get("rerank_context", False)),
                   late_visual_formats=tuple(s.get("late_visual_formats") or ()),
                   number_boost=float(s.get("number_boost") or 0.0),
                   number_boost_formats=tuple(s.get("number_boost_formats") or ("pdf",)),
                   page_reranker=str(s.get("page_reranker") or ""),
                   page_rerank_formats=tuple(s.get("page_rerank_formats") or ("pdf", "slide_image")),
                   visual_weights=tuple((f, float(w)) for f, w in (s.get("visual_weights") or {}).items()),
                   period_demote=bool(s.get("period_demote", False)),
                   table_lookup=bool(s.get("table_lookup", False)))  # fmt: skip


class Retriever:
    def __init__(self, cfg: Config, variant: str = "hybrid", embedder: str | None = None, visual: str | None = None,
                 live: dict | None = None):  # fmt: skip
        """embedder: a challenger alias (models.embedders) whose vectors live in an altembed store;
        None uses the LanceDB vector column (models.embedder). BM25 always comes from LanceDB.
        live: the sync pointer (index_dir/live.json); pins the table version and page-store snapshot.
        None reads the working (latest) index."""
        self.cfg = cfg
        self.variant = variant
        self.sc = SearchConfig.from_cfg(cfg)
        self.tbl = connect(cfg).open_table(table_name(variant))
        version = (live or {}).get("tables", {}).get(table_name(variant))
        if version is not None:
            self.tbl.checkout(version)
        self.store = None
        self.visual = None
        if visual:
            from .visual import VisualRanker, page_snapshot

            self.visual = VisualRanker(cfg, visual, page_snapshot(cfg, visual, live))
            self.visual_k = int(cfg["visual"]["visual_k"])
            self.office_shared_pages = int(cfg["visual"].get("office_shared_pages") or 0)  # M24
            self.office_own_formats = tuple(cfg["visual"].get("office_own_formats") or ("docx", "pptx"))  # M24b
            self._text_rich_chars = int(cfg["visual"].get("text_rich_chars") or 0)  # M25
            self._text_rich_formats = set(cfg["visual"].get("text_rich_formats") or ())
            blind = set(cfg["visual"].get("blind_formats") or ())
            # docs the visual model may vote on in format-aware fusion: rendered and not of a blind format
            self.visual_seen = {m["doc_id"] for m in self.visual.meta.values()
                                if m["doc_id"] in self.visual.docs and m["format"] not in blind}  # fmt: skip
        if embedder or visual:
            n = self.tbl.count_rows()
            self._rows = {r["chunk_id"]: r for r in self.tbl.search().select(COLS).limit(n).to_list()}
        if visual:
            self._by_unit: dict[str, list[dict]] = {}
            for r in self._rows.values():
                for u in self._units(r):
                    self._by_unit.setdefault(u, []).append(r)
        if embedder:
            from .altembed import VectorStore

            self.store = VectorStore(cfg, embedder, variant)

    def bm25(self, query: str, k: int, allow: frozenset[str] | None = None) -> list[dict]:
        """allow: doc ids a filter lets through; LanceDB applies it before the top k is cut."""
        from lancedb.query import MatchQuery

        try:
            q = self.tbl.search(MatchQuery(query, "text"), query_type="fts")
            if allow is not None:
                q = q.where(_doc_in(allow), prefilter=True)
            return q.select([*COLS, "_score"]).limit(k).to_list()
        except Exception as e:  # noqa: BLE001 - queries with odd syntax should not crash retrieval
            log.warning("bm25 failed (%s: %s); this search uses the vector list only", type(e).__name__, e)
            return []

    def vector(self, qvec: np.ndarray, k: int, allow: frozenset[str] | None = None) -> list[dict]:
        if self.store is not None:
            hits = self.store.search(qvec, k if allow is None else len(self.store.ids))
            rows = [{**self._rows[cid], "_distance": 1 - s} for cid, s in hits if cid in self._rows]
            return rows if allow is None else [r for r in rows if r["doc_id"] in allow][:k]
        if not getattr(self, "_model_checked", False):
            from .index import check_model

            check_model(self.cfg, self.tbl, self.variant, self.cfg["models"]["embedder"], len(qvec))
            self._model_checked = True
        q = self.tbl.search(qvec, vector_column_name="vector").distance_type("cosine")
        if allow is not None:
            q = q.where(_doc_in(allow), prefilter=True)
        return q.select([*COLS, "_distance"]).limit(k).to_list()

    @staticmethod
    def _units(r: dict) -> list[str]:
        from .visual import fusion_unit

        if r["format"] in ("docx", "pptx", "xlsx", "csv"):
            return [fusion_unit(r["doc_id"], r["format"], None)]
        return [fusion_unit(r["doc_id"], r["format"], p) for p in (r.get("pages") or [1])]

    @property
    def text_rich(self) -> set[str]:
        """M25: the page/file units of visual.text_rich_formats whose chunks hold at least visual.text_rich_chars
        characters (the 'File: name' line not counted). In the fusion such a unit keeps its text rank, as a
        spreadsheet does: a page picture votes only where a page has little or no text (a scan, a chart)."""
        if getattr(self, "_rich", None) is None:
            n, fmts = getattr(self, "_text_rich_chars", 0), getattr(self, "_text_rich_formats", set())
            self._rich = set()
            if n > 0:
                for u, rows in self._by_unit.items():
                    if rows[0]["format"] in fmts and sum(len(r["text"].split("\n", 1)[-1].strip()) for r in rows) >= n:
                        self._rich.add(u)
        return self._rich

    def _fuse_format_aware(self, t_units: list[str], v_units: list[str]) -> list[tuple[str, float]]:
        """RRF where units of docs the visual model never saw (xlsx/csv, failed renders) or whose format is in
        visual.blind_formats are scored as if the visual list had placed them at their text rank, so the fusion
        neither pushes them down nor lifts them; blind-format units are also dropped from the visual list."""
        from .manifest import unit_doc

        k = self.sc.rrf_k
        seen_docs = getattr(self, "visual_seen", None)
        if seen_docs is None:
            seen_docs = self.visual.docs
        rich = getattr(self, "text_rich", ())  # M25: units with enough text of their own; their picture does not vote
        v_units = [u for u in v_units if unit_doc(u) in seen_docs and u not in rich]
        scores: dict[str, float] = {}
        for r, u in enumerate(t_units):
            seen = unit_doc(u) in seen_docs and u not in rich
            scores[u] = (1.0 if seen else 2.0) / (k + r + 1)
        w = dict(
            getattr(self.sc, "visual_weights", ())
        )  # M20: a format's visual rank can count more (or less) than its text rank
        fmt = {m["doc_id"]: m["format"] for m in self.visual.meta.values()} if w else {}
        for r, u in enumerate(v_units):
            scores[u] = scores.get(u, 0.0) + w.get(fmt.get(unit_doc(u)), 1.0) / (k + r + 1)
        order = {u: i for i, u in enumerate(t_units + [u for u in v_units if u not in set(t_units)])}
        return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))

    def _visual_hits(self, vq: np.ndarray, office_pages: int = 0, capped=None,
                     allow: frozenset[str] | None = None) -> list[tuple[str, list[dict], dict]]:  # fmt: skip
        """Visual ranking as (unit, chunks) pairs; pages without text get an empty placeholder chunk."""
        out = []
        args = (
            (office_pages, capped) if office_pages and capped is not None else (office_pages,) if office_pages else ()
        )
        kw = {} if allow is None else {"allow": allow}
        for unit, m in self.visual.rank(vq, self.visual_k, *args, **kw):
            rows = self._by_unit.get(unit) or [
                {
                    "chunk_id": f"visual:{unit}",
                    "doc_id": m["doc_id"],
                    "format": m["format"],
                    "source": "",
                    "pages": [m["page"]],
                    "text": "",
                }  # fmt: skip
            ]
            out.append((unit, rows, m))
        return out

    def candidates(
        self,
        query: str,
        qvec: np.ndarray | None,
        mode: str = "hybrid",
        vq: np.ndarray | None = None,
        allow: frozenset[str] | None = None,
    ) -> list[dict]:
        """mode: hybrid | bm25 | vector | visual | hybrid_visual. Returns chunk dicts in rank order.
        allow: the doc ids a search filter lets through (None = no filter); each list keeps its k.

        visual: page/file units by visual MaxSim, each expanded to its text chunks (for unit scoring).
        hybrid_visual: RRF of the hybrid unit ranking and the visual unit ranking, expanded to chunks:
        a unit's chunks come in hybrid order, then any of its chunks hybrid did not retrieve (max 2).
        """
        kw = {} if allow is None else {"allow": allow}  # no filter: the same calls as before filters existed
        if mode == "visual":
            # One pseudo-hit per unit carrying exactly the rendered page, so scoring sees the visual ranking.
            return [
                {**rows[0], "chunk_id": f"visual:{u}", "pages": [m["page"]], "score": 0.0}
                for u, rows, m in self._visual_hits(vq, **kw)
            ]
        if mode in ("hybrid_visual", "hybrid_visual_fa"):
            text = self.candidates(query, qvec, "hybrid", **kw)
            t_units: list[str] = []
            by_unit: dict[str, list[dict]] = {}
            for c in text:
                for u in self._units(c):
                    if u not in by_unit:
                        t_units.append(u)
                    by_unit.setdefault(u, []).append(c)
                    break  # a chunk belongs to its first page's unit for expansion
            shared = getattr(self, "office_shared_pages", 0)
            own = set(getattr(self, "office_own_formats", ("docx", "pptx"))) if shared else set()
            # M24b: an Office format outside `own` keeps only its first pages everywhere
            vis = (
                self._visual_hits(vq, shared, {"docx", "pptx"} - own, **kw)
                if shared and own != {"docx", "pptx"}
                else self._visual_hits(vq, **kw)
            )
            vrank = {u: i + 1 for i, (u, _, _) in enumerate(vis)}
            if mode == "hybrid_visual_fa":
                fused = self._fuse_format_aware(t_units, [u for u, _, _ in vis])
                if shared:  # M24: pages past the first N order the `own` Office files among themselves only
                    base = self._fuse_format_aware(t_units, [u for u, _, _ in self._visual_hits(vq, shared, **kw)])
                    office = {r["doc_id"] for r in self._rows.values() if r["format"] in own}
                    fused = office_own_slots(base, fused, office)
            else:
                fused = rrf([t_units, [u for u, _, _ in vis]], self.sc.rrf_k)
            if self.sc.period_demote:
                fused = period_demote(query, fused)
            extra = {u: rows for u, rows, _ in vis}
            cap = self.sc.rerank_unit_chunks  # M17: a file with many chunks must not fill the rerank window alone
            out, over, seen = [], [], set()
            for u, s in fused:
                rows = by_unit.get(u, [])
                rest = [r for r in extra.get(u, []) if r["chunk_id"] not in {x["chunk_id"] for x in rows}]
                for i, c in enumerate(rows + rest[:2]):
                    if c["chunk_id"] not in seen:
                        seen.add(c["chunk_id"])
                        (over if cap and i >= cap else out).append(
                            {**c, "score": s, **({"vis_rank": vrank[u]} if u in vrank else {})}
                        )
            return out + over
        lists, rows = [], {}
        if mode in ("hybrid", "bm25"):
            b = self.bm25(query, self.sc.bm25_k, **kw)
            lists.append([r["chunk_id"] for r in b])
            rows.update({r["chunk_id"]: r for r in b})
        if mode in ("hybrid", "vector") and qvec is not None:
            v = self.vector(qvec, self.sc.vector_k, **kw)
            lists.append([r["chunk_id"] for r in v])
            rows.update({r["chunk_id"]: r for r in v})
        fused = rrf(lists, self.sc.rrf_k)
        return [{**rows[cid], "score": s} for cid, s in fused]


def _doc_in(allow: frozenset[str]) -> str:
    return "doc_id IN (" + ", ".join("'" + x.replace("'", "''") + "'" for x in sorted(allow)) + ")"


_QTR = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
_P_QY = re.compile(r"(?<![a-z0-9])(?:q([1-4])|([1-4])q)[\s_\-']{0,2}((?:19|20)?\d{2})(?![0-9])")
_P_WORD = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-]quarter(?:\s+of)?\s+((?:19|20)\d{2})\b")
_P_YEAR = re.compile(r"(?<![0-9])((?:19|20)\d{2})(?![0-9])")


def periods(text: str) -> tuple[set[tuple[int, int]], set[int]]:
    """M21: the (quarter, year) pairs and the years named in a query or a file name
    ('3q13', 'Q3 2013', 'third quarter of 2013', 'from 2020 to 2021')."""
    t = text.lower()

    def year(s: str) -> int:
        y = int(s)
        return y if y > 99 else (2000 + y if y < 70 else 1900 + y)

    pairs = {(int(m.group(1) or m.group(2)), year(m.group(3))) for m in _P_QY.finditer(t)}
    pairs |= {(_QTR[m.group(1)], int(m.group(2))) for m in _P_WORD.finditer(t)}
    return pairs, {int(y) for y in _P_YEAR.findall(t)} | {y for _, y in pairs}


def office_own_slots(base: list[tuple[str, float]], full: list[tuple[str, float]], office: set[str]):
    """M24: `base` is the fusion with the first pages of each Office file only, `full` the fusion with every
    rendered page. Office files (units in `office`) take, in `full` order, the slots Office files hold in `base`;
    every other unit keeps its `base` slot. So the extra pages help the right Word or PowerPoint file against
    other Office files, and they never push a spreadsheet, a PDF page or a slide down."""
    slots = [i for i, (u, _) in enumerate(base) if u in office]
    ranked = [x for x in full if x[0] in office]
    out = list(base)
    for i, x in zip(slots, ranked, strict=False):
        out[i] = x
    return out + ranked[len(slots) :]


def period_demote(query: str, fused: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """M21: a unit whose file name names a quarter (e.g. 'ibm-3q13-earnings') that the query's periods exclude
    goes behind all the other units; order is otherwise kept. A query that names quarters excludes every other
    quarter; a query that names only years excludes quarters of other years. Files named for a year only
    (annual reports discuss many years) and files with no period in their name are never moved."""
    from .manifest import unit_doc

    q_pairs, q_years = periods(query)
    if not q_years:
        return fused
    keep, late = [], []
    for u, s in fused:
        d_pairs, _ = periods(unit_doc(u).rsplit("/", 1)[-1])
        clash = bool(d_pairs) and not (d_pairs & q_pairs if q_pairs else {y for _, y in d_pairs} & q_years)
        (late if clash else keep).append((u, s))
    return keep + late


_QTOK = re.compile(r"\d+(?:[.,]\d+)?%?|[a-z][a-z\-]{3,}")
_STOP = {"what", "which", "when", "where", "whose", "with", "from", "that", "this", "were", "does", "into", "than",
         "many", "much", "rate", "year", "years", "percent", "percentage", "points", "change", "during", "between",
         "their", "there", "they", "have", "been", "given", "show", "shown", "report", "reported", "about"}  # fmt: skip


def query_terms(q: str) -> set[str]:
    """Content terms of a query: numbers (with decimals, e.g. 32.7) and words of 4+ letters minus stopwords."""
    return {t.rstrip("%").replace(",", ".") if t[0].isdigit() else t for t in _QTOK.findall(q.lower())} - _STOP


def _norm_terms(text: str) -> set[str]:
    return {t.rstrip("%").replace(",", ".") if t[0].isdigit() else t for t in _QTOK.findall(text.lower())}


def distinct_tiebreak(query: str, ranked: list[dict], n: int, boost: float, share: float = 0.5) -> list[dict]:
    """M11: among the top n, move a candidate up by `boost` positions per query term it contains that at most
    `share` of the top n contain (terms that tell near-duplicate pages apart, e.g. the cue value '32.7' that is
    only on the target table). Terms every twin has (the topic words) do not count. Order is otherwise kept."""
    if boost <= 0 or not ranked:
        return ranked
    head, tail = ranked[:n], ranked[n:]
    terms = query_terms(query)
    if not terms:
        return ranked
    have = [_norm_terms(c.get("text", "")) & terms for c in head]
    df = {t: sum(t in h for h in have) for t in terms}
    rare = {t for t, d in df.items() if 0 < d <= max(1, int(share * len(head)))}
    keyed = [(i - boost * len(h & rare), i, c) for i, (h, c) in enumerate(zip(have, head, strict=True))]
    out = []
    for _, _, c in sorted(keyed, key=lambda x: (x[0], x[1])):
        out.append({**c, "distinct_hits": len(_norm_terms(c.get("text", "")) & rare)})
    return out + tail


_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def query_numbers(q: str) -> set[str]:
    """Exact numbers in a query worth matching: decimals, or integers of 2+ digits that are not years."""
    out = set()
    for t in _NUM.findall(q):
        t = t.replace(",", ".")
        if "." in t or (len(t) >= 2 and not re.fullmatch(r"(19|20)\d\d", t)):
            out.add(t)
    return out


NUMBER_BOOST_STATS: Counter = Counter()  # M14b: queries seen / with numbers / reordered (reported by eval)


def _nums(text: str) -> set[str]:
    return {t.replace(",", ".") for t in _NUM.findall(text)}


def _slot_reorder(head: list[dict], formats: tuple[str, ...], order_key) -> list[dict]:
    """Reorder only the items of `formats` among the slots they already hold; every other item keeps its index.
    order_key(groups) gets the format items grouped by page (first-appearance order) and returns them reordered."""
    slots = [i for i, c in enumerate(head) if c.get("format") in formats]
    groups: dict[tuple, list[dict]] = {}
    for i in slots:
        c = head[i]
        groups.setdefault((c.get("doc_id"), (c.get("pages") or [None])[0]), []).append(c)
    new = [c for g in order_key(list(groups.values())) for c in g]
    out = list(head)
    for i, c in zip(slots, new, strict=True):
        out[i] = c
    return out


def number_boost(query: str, ranked: list[dict], n: int, boost: float, formats: tuple[str, ...] = ("pdf",)):
    """M14b: page-level exact-number boost. Among the top n, pages of `formats` move up `boost` page positions per
    distinctive query number on the page (a number on at most a third of those pages, so a year on every page
    counts for nothing). Only those pages swap among their own slots; other formats keep their ranks.
    (M14 moved single chunks 2 slots; pages hold ~3.6 chunks each, so it changed 13/734 top-10s and no metric.)"""
    if boost <= 0:
        return ranked
    NUMBER_BOOST_STATS["queries"] += 1
    nums = _nums(query)
    if not nums or not ranked:
        return ranked
    NUMBER_BOOST_STATS["with_numbers"] += 1
    head, tail = ranked[:n], ranked[n:]

    def order(groups):
        have = [_nums(" ".join(c.get("text", "") for c in g)) for g in groups]
        cap = max(1, len(groups) // 3)
        dist = {x for x in nums if 0 < sum(x in h for h in have) <= cap}
        keyed = [(i - boost * len(dist & h), -len(dist & h), i) for i, h in enumerate(have)]
        return [groups[i] for *_, i in sorted(keyed)]

    out = _slot_reorder(head, formats, order)
    if [c["chunk_id"] for c in out] != [c["chunk_id"] for c in head]:
        NUMBER_BOOST_STATS["touched"] += 1
    return out + tail


def late_visual(ranked: list[dict], n: int, formats: tuple[str, ...], k: int = 10) -> list[dict]:
    """M13b: after rerank, re-fuse the reranked order with the visual page rank, but only among items of
    `formats`, inside the slots they already hold: score = 1/(k+pos) + 1/(k+vis_rank). Other formats keep their
    exact positions. (M13 sorted the whole head, so the visual bonus lifted PDFs over non-PDF golds.)"""
    if not formats or not ranked:
        return ranked
    head, tail = ranked[:n], ranked[n:]
    pos = {id(c): i for i, c in enumerate(head)}

    def order(groups):
        def sc(g):
            c = g[0]
            return 1.0 / (k + pos[id(c)] + 1) + (1.0 / (k + c["vis_rank"]) if c.get("vis_rank") else 0.0)

        return sorted(groups, key=lambda g: (-sc(g), pos[id(g[0])]))

    return _slot_reorder(head, formats, order) + tail


def page_rerank(query: str, ranked: list[dict], scorer, n: int, pages: int, formats: tuple[str, ...]) -> list[dict]:
    """M16: among the top n, the first `pages` pages of `formats` are reordered by a page-image reranker's score
    (scorer.score(query, units) -> one score per page unit, None when the page has no image). Like M13b, only
    those pages swap among the slots they already hold; other formats, later pages and unscored pages keep
    their places."""
    if scorer is None or pages <= 0 or not ranked:
        return ranked
    from .manifest import page_unit

    head, tail = ranked[:n], ranked[n:]

    def order(groups):
        top = groups[:pages]
        units = [page_unit(g[0]["doc_id"], (g[0].get("pages") or [1])[0]) for g in top]
        scores = scorer.score(query, units)
        scored = [i for i, s in enumerate(scores) if s is not None]
        new = list(top)
        for slot, j in zip(scored, sorted(scored, key=lambda i: (-scores[i], i)), strict=True):
            new[slot] = [{**c, "page_score": scores[j]} for c in top[j]]
        return new + groups[pages:]

    return _slot_reorder(head, formats, order) + tail


TABLE_LOOKUP_STATS: Counter = Counter()  # M33: queries seen / files looked up / files reordered (reported by eval)


def table_lookup(query: str, ranked: list[dict], cells) -> list[dict]:
    """M33 (C2): for each file with two or more pages in the ranked list, cells.scores(query, doc, pages) scores
    its pages by the table rows and columns the query names (cells.CellLookup). The best page, when it scores
    above the file's current first page, takes the file's first slots; the file's other pages follow in their
    order, in the file's remaining slots. Other files keep every slot."""
    if cells is None or not ranked:
        return ranked
    TABLE_LOOKUP_STATS["queries"] += 1

    def first(c: dict):
        return (c.get("pages") or [None])[0]

    pages: dict[str, list] = {}
    for c in ranked:
        p = first(c)
        if p is not None and p not in pages.setdefault(c["doc_id"], []):
            pages[c["doc_id"]].append(p)
    out = ranked
    for doc, ps in pages.items():
        if len(ps) < 2:
            continue
        TABLE_LOOKUP_STATS["files"] += 1
        sc = cells.scores(query, doc, ps)
        best = max(ps, key=lambda p: sc.get(p, 0))
        if sc.get(best, 0) <= sc.get(ps[0], 0):
            continue
        TABLE_LOOKUP_STATS["moved"] += 1
        slots = [i for i, c in enumerate(out) if c["doc_id"] == doc]
        items = [out[i] for i in slots]
        out = list(out)
        for i, c in zip(slots, [c for c in items if first(c) == best] + [c for c in items if first(c) != best],
                        strict=True):  # fmt: skip
            out[i] = c
    return out


def context_text(c: dict) -> str:
    """M12: reranker input with the document name and page number in front (slide filenames carry the quarter,
    e.g. 'ibm 4q13 earnings charts page 20'), so twins from different decks/pages are told apart."""
    doc = c.get("doc_id", "").split("/")[-1]
    doc = re.sub(r"\.(pdf|pptx?|docx?|xlsx?|csv|png|jpe?g)$", "", doc, flags=re.I)
    doc = re.sub(r"[_\-.]+", " ", doc).strip()
    pages = c.get("pages") or []
    pg = f" · page {pages[0]}" if pages else ""
    return f"Document: {doc}{pg}\n{c.get('text', '')}"


def rerank(query: str, cands: list[dict], reranker, n: int, min_chars: int = 0,
           skip_formats: tuple[str, ...] | list[str] = (), context: bool = False) -> list[dict]:  # fmt: skip
    """Rerank the first n candidates; the rest keep fused order below them.

    Candidates that are not reranked keep their fused position and the others are reranked into the remaining
    slots: those with fewer than min_chars non-space characters of text (min_chars > 0; e.g. slides with
    little OCR text), and those whose format is in skip_formats (e.g. slide_image)."""
    head, tail = cands[:n], cands[n:]
    if not head:
        return cands
    if min_chars > 0 or skip_formats:
        short = [i for i, c in enumerate(head)
                 if c.get("format") in skip_formats
                 or (min_chars > 0 and len("".join(c["text"].split()[2:])) < min_chars)]  # fmt: skip
        if short:
            fixed = {i: head[i] for i in short}
            rest = rerank(query, [c for i, c in enumerate(head) if i not in fixed], reranker, n, context=context)
            it = iter(rest)
            return [fixed[i] if i in fixed else next(it) for i in range(len(head))] + tail
    scores = reranker.score(query, [context_text(c) if context else c["text"] for c in head])
    head = [{**c, "rerank_score": s} for s, c in sorted(zip(scores, head, strict=True), key=lambda x: -x[0])]
    return head + tail


def early_safe(sc: SearchConfig) -> bool:
    """True when no step after the page rerank can move a result: the search page then shows the text-ranked
    results that are not of sc.page_rerank_formats before the page rerank ends. A new step after page_rerank in
    rerank_pipeline must be added here."""
    return not (sc.table_lookup or sc.late_visual_formats or sc.number_boost > 0 or sc.distinct_boost > 0)


def rerank_pipeline(query: str, cands: list[dict], reranker, sc: SearchConfig, page_reranker=None,
                    cells=None, progress=None) -> list[dict]:  # fmt: skip
    """Rerank with the configured options. M12 (rerank_context) also reranks the formats in rerank_skip_formats,
    now that they get document/page context; M11 (distinct_boost) then applies the near-duplicate tie-break.
    M16: page_reranker (a models.PageReranker) reorders the top pages by their images after the text rerank.
    M33: with sc.table_lookup, cells (a cells.CellLookup) puts each file's best table page first.
    progress(kind, hits): hears "ranked" with the text-reranked list, before the (slow) page rerank."""
    skip = () if sc.rerank_context else sc.rerank_skip_formats
    out = rerank(query, cands, reranker, sc.rerank_candidates, sc.rerank_min_chars, skip, sc.rerank_context)
    if progress is not None:
        progress("ranked", out)
    from .models import stop_point

    stop_point()
    out = page_rerank(query, out, page_reranker, sc.rerank_candidates, sc.page_rerank_pages, sc.page_rerank_formats)
    if sc.table_lookup:
        out = table_lookup(query, out, cells)
    out = late_visual(out, sc.rerank_candidates, sc.late_visual_formats, sc.late_visual_k)
    out = number_boost(query, out, sc.rerank_candidates, sc.number_boost, sc.number_boost_formats)
    return distinct_tiebreak(query, out, sc.rerank_candidates, sc.distinct_boost)


def indexed_docs(cfg: Config, variant: str) -> set[str]:
    """The doc ids in the table version search serves; empty when there is no table yet."""
    from .sync import read_live

    name = table_name(variant)
    db = connect(cfg)
    if name not in db.table_names():
        return set()
    tbl = db.open_table(name)
    version = (read_live(cfg) or {}).get("tables", {}).get(name)
    if version is not None:
        tbl.checkout(version)
    n = tbl.count_rows()
    return set(tbl.search().select(["doc_id"]).limit(n).to_arrow().column("doc_id").to_pylist()) if n else set()


def page_rows(cfg: Config, variant: str, docs: set[str]) -> list[dict]:
    """Every chunk of these docs, in index order, from the table version search serves (the live one when sync set
    it). The +agent step reads whole pages, not only the retrieved chunks."""
    from .sync import read_live

    if not docs:
        return []
    tbl = connect(cfg).open_table(table_name(variant))
    version = (read_live(cfg) or {}).get("tables", {}).get(table_name(variant))
    if version is not None:
        tbl.checkout(version)
    return tbl.search().where(_doc_in(frozenset(docs))).select(COLS).limit(tbl.count_rows()).to_list()


_RETRIEVER: tuple[str, Retriever] | None = None  # docsearch serve: the last reader built, and what it read


def index_stamp(cfg: Config, variant: str, vis: str | None, live: dict | None) -> str:
    """What a Retriever read: the live pointer, else the table version and the page store's files."""
    if live:
        return json.dumps(live, sort_keys=True)
    from .altembed import store_dir
    from .visual import pages_dir

    stamp = [connect(cfg).open_table(table_name(variant)).version]
    if vis:
        stamp += [
            p.stat().st_mtime_ns for p in (pages_dir(cfg) / "pages.jsonl", store_dir(cfg, vis, "pages")) if p.exists()
        ]
    return json.dumps(stamp)


def kept_retriever(cfg: Config, variant: str, vis: str | None, live: dict | None) -> Retriever:
    """A Retriever for search(). A long-lived process (models.keep_loaded) keeps the last one while the index and
    the settings it read are unchanged; a CLI run builds one."""
    global _RETRIEVER
    from .models import keeping

    if not keeping():
        return Retriever(cfg, variant, None, vis, live)
    key = json.dumps([variant, vis, index_stamp(cfg, variant, vis, live),
                      *(cfg[s] for s in ("search", "visual", "models", "paths"))], sort_keys=True, default=str)  # fmt: skip
    if _RETRIEVER is None or _RETRIEVER[0] != key:
        _RETRIEVER = (key, Retriever(cfg, variant, None, vis, live))
    return _RETRIEVER[1]


def search(cfg: Config, query: str, variant: str = "hybrid", use_rerank: bool = True, k: int | None = None,
           flt=None, progress=None):  # fmt: skip
    """Interactive search used by the CLI (and by `answer`, which passes no filter).
    flt: a meta.Filter; only the docs it allows are retrieved, before any list is cut to its k.
    progress(kind, hits): hears "candidates" with the fused list, then rerank_pipeline's "ranked" (the search page
    shows what runs, and the text-ranked list early)."""
    from .models import load_embedder, load_reranker

    allow = None
    if flt is not None and flt.active:
        from .meta import read_meta

        allow = frozenset(d for d, m in read_meta(cfg).items() if flt.allows(m))
        if not allow:
            return []

    from .sync import read_live

    live = read_live(cfg)
    vis = cfg["visual"].get("default")  # adopted V2: format-aware visual fusion when page vectors exist
    if vis:
        from .visual import page_store_ready

        if not page_store_ready(cfg, vis, live):
            note = f"page store {vis} is empty: page pictures were not used for this search (text only)"
            log.warning(note)
            print(note, file=sys.stderr)
            vis = None
    r = kept_retriever(cfg, variant, vis, live)  # serve the live version when sync set one
    qvec = load_embedder(cfg).queries([query])[0]
    if vis:
        from .altembed import query_embeddings

        cands = r.candidates(query, qvec, "hybrid_visual_fa", query_embeddings(cfg, vis, [query])[0], allow)
    else:
        cands = r.candidates(query, qvec, allow=allow)
    if progress is not None:
        progress("candidates", cands)
    from .models import stop_point

    stop_point()
    if use_rerank:
        page_rr = None
        if vis and r.sc.page_reranker:  # adopted M16b: the vision reranker needs the rendered pages
            from .models import load_page_reranker

            page_rr = load_page_reranker(cfg, r.sc.page_reranker)
        cells = None
        if r.sc.table_lookup:
            from .cells import CellLookup, cells_path

            cells = CellLookup(cells_path(cfg))
        cands = rerank_pipeline(query, cands, load_reranker(cfg), r.sc, page_rr, cells, progress)
    return cands[: (k or r.sc.top_k)]
