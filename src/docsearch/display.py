"""What `docsearch search` shows: the ranked hit list grouped by file, for people (text) and scripts (--json).

The hit list stays exactly as search returned it (eval scores that list); grouping happens here only. A file is a
doc_id, except a slide image, whose file is its deck (manifest.file_of). Files come in the order of their best
page, each with its matching pages in rank order.

Scores come from different scorers on different scales, so a page names its scorer: "rerank" (the text
reranker), "page" (the page-image reranker) or "fused" (fusion only, not reranked). "No confident match" needs a
threshold measured for the top hit's scorer; only the text reranker has one (search.abstain_threshold).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from .manifest import file_page
from .search import periods, query_terms

PAGE_LINES = 3  # pages per file shown with title and snippet; the rest are listed by number
SNIPPET_CHARS = 150
TITLE_CHARS = 80
LEAD_CHARS = 30  # context kept before the first query word in a snippet
# Formats whose lines are units (a slide's title and bullets, a table's label and rows): the first short line is
# the title and lines stay apart (" / "). Other formats break lines inside sentences, so their lines join as prose.
LINE_FORMATS = frozenset({"slide_image", "pptx", "xlsx", "csv"})


@dataclass(frozen=True)
class PageHit:
    page: int | None  # None: a file without pages (docx, csv)
    rank: int  # 1-based position of the page's best chunk in the hit list
    title: str  # the page's first line when it is short (slide or table title); "" otherwise
    snippet: str
    score: float | None
    scorer: str  # "rerank" | "page" | "fused"


@dataclass(frozen=True)
class ResultGroup:
    rank: int  # 1-based file rank, by the file's best page
    file: str
    format: str
    name: str  # title from the file's own properties, else the file name
    date: str  # last change, else creation, from inside the file; "" = unknown
    period: str  # the one quarter the file name names ("Q3 2018"); "" = none
    pages: tuple[PageHit, ...]


@dataclass(frozen=True)
class Answer:
    value: str
    file: str
    page: int | None


@dataclass(frozen=True)
class Results:
    query: str
    confident: bool | None  # None: no threshold, or the top hit's scorer has no measured threshold
    answer: Answer | None
    groups: tuple[ResultGroup, ...]
    chunk_ids: tuple[str, ...]  # the hit list as search returned it


def scored(h: dict) -> tuple[str, float | None]:
    """(scorer, score) of a hit: the last scorer that ordered it."""
    if h.get("page_score") is not None:
        return "page", float(h["page_score"])
    if h.get("rerank_score") is not None:
        return "rerank", float(h["rerank_score"])
    return "fused", None if h.get("score") is None else float(h["score"])


def confident(hits: list[dict], threshold: float | None) -> bool | None:
    if not hits or threshold is None:
        return None
    scorer, score = scored(hits[0])
    return None if scorer != "rerank" else score >= threshold


def _lines(text: str) -> list[str]:
    """The chunk's lines without the 'File: <name>' line, each table row with repeated cells (merged cells) once."""
    rows = text.split("\n")
    if rows and rows[0].startswith("File: "):
        rows = rows[1:]
    if rows and rows[0].startswith("Context: "):  # the +enrich line (enrich.apply) is for search, not a title
        rows = rows[1:]
    out = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        kept = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
        line = " | ".join(kept).replace("**", "").strip()  # Docling marks bold text with ** in its Markdown
        if line:
            out.append(re.sub(r"\s+", " ", line))
    return out


def title_and_body(text: str, fmt: str) -> tuple[str, str]:
    lines = _lines(text)
    if fmt not in LINE_FORMATS:
        return "", " ".join(lines)
    if len(lines) > 1 and len(lines[0]) <= TITLE_CHARS:
        return lines[0], " / ".join(lines[1:])
    return "", " / ".join(lines)


def snippet(body: str, terms: set[str], width: int = SNIPPET_CHARS) -> str:
    """A window of the body that holds the most distinct query terms (the earliest such window), cut at words."""
    low = body.lower()
    found = sorted((m.start(), t) for t in terms for m in re.finditer(r"(?<![a-z0-9])" + re.escape(t), low))
    start = best = 0
    for pos, _ in found:
        s = max(0, pos - LEAD_CHARS)
        n = len({t for p, t in found if s <= p and p + len(t) <= s + width})
        if n > best:
            start, best = s, n
    if start and body[start - 1] != " ":
        start = body.find(" ", start) + 1 or start
    end = min(len(body), start + width)
    if end < len(body):
        cut = body.rfind(" ", start, end)
        end = cut if cut > start else end
    return ("…" if start else "") + body[start:end].strip() + ("…" if end < len(body) else "")


def period_of(name: str) -> str:
    pairs, _ = periods(name)
    if len(pairs) != 1:
        return ""
    ((q, y),) = pairs
    return f"Q{q} {y}"


_OPAQUE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.I)  # a uuid file name
NAME_CHARS = 120
NAME_LINES = 3  # lines of the first chunk tried for a name
# Titles that templates and converters fill in, which name nothing ("PowerPoint Presentation", Docling's "Table 1")
_DEFAULT_TITLE = re.compile(r"^(power ?point[ -]presentation|slide \d+|table \d+|sheet ?\d*|enter title here|"
                            r"untitled|presentation|document ?\d*|microsoft®?)$", re.I)  # fmt: skip


def _name(text: str) -> str:
    """text as a file's name without Markdown marks and line breaks; "" when it names nothing: a template's default
    title, no words (a number, a code fence), a table row, or too long to be a title."""
    s = " ".join(re.sub(r"[*`]+", " ", text).split())
    if not 3 <= len(s) <= NAME_CHARS or sum(c.isalpha() for c in s) < 2 or "|" in s or _DEFAULT_TITLE.match(s):
        return ""
    return s


def first_lines(cfg, variant: str, docs) -> dict[str, str]:
    """doc_id -> the first line of its first chunk that can be a name (_name), for files named by an opaque id (a
    uuid). Such files mostly have no title in their properties either."""
    from .search import page_rows

    opaque = {d for d in docs if _OPAQUE.match(PurePosixPath(d).stem)}
    first: dict[str, dict] = {}
    for r in page_rows(cfg, variant, opaque) if opaque else []:
        if r["doc_id"] not in opaque:
            continue
        n = int(r["chunk_id"].rsplit(":", 1)[-1]) if r["chunk_id"].rsplit(":", 1)[-1].isdigit() else 1 << 30
        if r["doc_id"] not in first or n < first[r["doc_id"]][0]:
            first[r["doc_id"]] = (n, r)
    out = {}
    for doc, (_, r) in first.items():
        if name := next(filter(None, map(_name, _lines(r.get("text") or "")[:NAME_LINES])), ""):
            out[doc] = name
    return out


def group(query: str, hits: list[dict], metas: dict, names: dict | None = None) -> tuple[ResultGroup, ...]:
    """metas: doc_id -> meta.DocMeta (may be empty: no metadata built). names: doc_id -> a name for a file with no
    title in its metadata (first_lines)."""
    terms = query_terms(query)
    files: dict[str, dict] = {}
    for rank, h in enumerate(hits, 1):
        file, page = file_page(h["doc_id"], h.get("pages"), h.get("format", ""))
        g = files.setdefault(file, {"first": h, "pages": {}})
        if page not in g["pages"]:
            title, body = title_and_body(h.get("text") or "", h.get("format", ""))
            scorer, score = scored(h)
            g["pages"][page] = PageHit(page, rank, title, snippet(body, terms), score, scorer)
    out = []
    for i, (file, g) in enumerate(files.items(), 1):
        first, m = g["first"], metas.get(g["first"]["doc_id"])
        base = PurePosixPath(file).name
        name = _name(m.title if m else "") or (names or {}).get(first["doc_id"]) or base
        out.append(ResultGroup(i, file, first.get("format", ""), name,
                               m.date if m else "", period_of(base), tuple(g["pages"].values())))  # fmt: skip
    return tuple(out)


def build(query: str, hits: list[dict], metas: dict, threshold: float | None, answer: Answer | None = None,
          names: dict | None = None):  # fmt: skip
    return Results(query, confident(hits, threshold), answer, group(query, hits, metas, names),
                   tuple(h["chunk_id"] for h in hits))  # fmt: skip


def _cite(file: str, page: int | None) -> str:
    name = PurePosixPath(file).name
    return name if page is None else f"{name}, p. {page}"


def render(r: Results, scores: bool = False) -> str:
    out = []
    if r.answer:
        out.append(f"Answer: {r.answer.value or '(see the page)'} ({_cite(r.answer.file, r.answer.page)})")
    if r.confident is False:
        out.append("No confident match; the nearest results follow.")
    if out:
        out.append("")
    for g in r.groups:
        when = ", ".join(x for x in (g.period, g.date) if x)
        out.append(f"{g.rank:2d}. {g.name}" + (f" | {when}" if when else ""))
        out.append(f"    {g.file}")
        for p in g.pages[:PAGE_LINES]:
            where = "" if p.page is None else f"p. {p.page:<4d}"
            score = f"{p.scorer} {p.score:.3f}  " if scores and p.score is not None else ""
            text = " · ".join(x for x in (p.title, p.snippet) if x)
            out.append(f"    {where}{score}{text}".rstrip())
        rest = [str(p.page) for p in g.pages[PAGE_LINES:] if p.page is not None]
        if rest:
            out.append("    also p. " + ", ".join(rest))
    return "\n".join(out)


def as_json(r: Results) -> dict:
    return asdict(r)


class SearchRefused(ValueError):
    """A search that cannot run as asked: filters on an index whose files have no metadata."""


def run(cfg, query: str, variant: str | None = None, rerank: bool = True, k: int | None = None, flt=None,
        agent: bool | None = None, reranker: str | None = None, progress=None) -> tuple[Results, list[str]]:  # fmt: skip
    """One search as `docsearch search` and the web API run it: the grouped results, and notes for the user (files
    a date filter left out, no hits). The +agent step runs when `agent` is True; None follows agent.mode. variant
    None follows enrich.llm (chunk.search_variant)."""
    from .chunk import search_variant
    from .meta import Filter, read_meta
    from .search import indexed_docs, search

    variant = variant or search_variant(cfg)
    flt = flt or Filter()
    metas = read_meta(cfg)
    if flt.active and not metas:
        raise SearchRefused("search filters need file metadata: run `uv run docsearch meta` "
                            "(or `uv run docsearch sync`) first")  # fmt: skip
    if flt.active:
        docs = indexed_docs(cfg, variant)
        if missing := docs - metas.keys():
            raise SearchRefused(f"search filters need file metadata: {len(missing)} of {len(docs)} indexed files "
                                "have no metadata (indexed before metadata existed); run `uv run docsearch meta`")  # fmt: skip
    if reranker:
        from .config import resolve_reranker

        resolve_reranker(cfg, reranker)
    hits = search(cfg, query, variant, rerank, k, flt, progress)
    notes = [n for n in (flt.note(metas.values()),) if n]
    if not hits:
        notes.append("no hits: no indexed file matches the filter" if flt.active else "no hits")
    if agent is None:
        agent = cfg["agent"]["mode"] == "on"
    answer = agent_answer(cfg, query, hits, variant) if agent else None
    names = first_lines(cfg, variant, {h["doc_id"] for h in hits})
    return build(query, hits, metas, cfg["search"].get("abstain_threshold"), answer, names), notes


def agent_answer(cfg, query: str, hits: list[dict], variant: str) -> Answer | None:
    """The +agent step (M34h) on the hits: the page the LLM names and the value it reads there. The gate, no page,
    an unusable reply, a failed call or an LLM that cannot be used now (a usage limit) give None; the hit list
    itself is never changed."""
    import sys

    from . import agent, ui
    from .llm import LLMUnavailable
    from .search import page_rows

    if not hits:
        return None
    ag = agent.from_cfg(cfg, page_rows(cfg, variant, {h["doc_id"] for h in hits}), who="search --agent")
    try:
        picked, step = ag(query, hits)
    except LLMUnavailable as e:
        ui.logger.warning("search --agent: %s; showing plain results only", e)
        print(f"search --agent: the agent is not available ({e}); plain results only", file=sys.stderr)
        return None
    if step.error:
        ui.logger.warning("search --agent: %s; showing plain results only", step.error)
    if step.chosen is None:
        return None
    file, page = file_page(picked[0]["doc_id"], picked[0].get("pages"), picked[0].get("format", ""))
    return Answer(step.value or "", file, page)
