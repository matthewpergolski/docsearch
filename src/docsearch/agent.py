"""+agent, first layer (Phase C4): after plain search, an LLM reads the text of the top pages and names the page
that holds the answer, with the value. The ranked list becomes that page's chunks, then the plain order. Plain
search is the floor: no call (the gate), no page, an unusable reply or a failed call keeps the plain order.

A page is (doc_id, first page of the chunk); a file without pages (docx, csv) is one page. The LLM sees only page
text from the index, capped per page, and gets no tools (llm.py runs codex read-only). Replies are cached by prompt
hash, one file each, so a rerun makes no call.

D1 (agent.research, off by default): when the reader names no page, the LLM writes two rewrites of the question,
plain search runs for each, the page lists merge with the original (RRF), and the reader reads the merged top pages
once. A page it names goes first, then the plain order; anything else keeps the plain order.

D2 (agent.deck_gate, off by default): the gate also counts a slide deck as one file (manifest.file_of), not each
slide; when only that makes the gate pass, the reader reads the top DECK_PAGES pages instead of `pages`.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .manifest import file_of

PROMPT = """You help a document search engine. A user asked the question below. The search found the {n} pages that
follow; each starts with a letter in brackets, then its file name and its page. Find the one page whose text holds
the answer, and give the answer as written on that page. Use only the text shown.
Reply with one JSON object and nothing else:
{{"candidate": "<the letter in brackets of that page, or null if no page holds the answer>", "value": "<the answer \
as written on the page, or null>"}}

Question: {query}

{pages}"""
PROMPT_VERSION = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
REWRITES = 2
REWRITE_PROMPT = """You help a document search engine. The search found no page that answers the question below.
Write two other search queries for the same question, each in other words a document might use for it (names, terms,
units, periods). Do not answer the question.
Reply with one JSON object and nothing else:
{{"rewrites": ["<query 1>", "<query 2>"]}}

Question: {query}"""
RESEARCH_VERSION = hashlib.sha256(REWRITE_PROMPT.encode()).hexdigest()[:12]
DECK_PAGES = 20

Page = tuple[str, int | None]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def page_of(c: dict) -> Page:
    pages = c.get("pages")
    return c["doc_id"], (min(pages) if pages else None)


def unit_of(p: Page) -> str:
    return f"{p[0]}#p{p[1] or 1}"


def pages_in_order(hits: list[dict]) -> list[Page]:
    return list(dict.fromkeys(page_of(c) for c in hits))


def gate(hits: list[dict], k: int, decks: bool = False) -> bool:
    """Look-alike candidates: the top k pages hold two or more pages of one file. A file is a doc_id; with decks,
    slides of one deck are one file (D2)."""
    first: dict[Page, dict] = {}
    for c in hits:
        first.setdefault(page_of(c), c)
    files = Counter(
        file_of(c["doc_id"], c.get("format", "")) if decks else c["doc_id"] for c in list(first.values())[:k]
    )
    return bool(files) and max(files.values()) >= 2


def parse_reply(text: str, n: int) -> tuple[int | None, str | None] | None:
    """(candidate number 1..n from its letter, or None for no page; value) from a reply; None when it is unusable.
    Letters, not numbers, name the candidates: a model can mistake a candidate number for the page number."""
    start, end = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[start : end + 1]) if start >= 0 else None
    except ValueError:
        return None
    if not isinstance(d, dict) or "candidate" not in d:
        return None
    pick, value = d["candidate"], d.get("value")
    value = None if value is None else str(value)
    if pick is None:
        return None, value
    letter = pick.strip("[] ").upper() if isinstance(pick, str) else ""
    if len(letter) != 1 or letter not in LETTERS[:n]:
        return None
    return LETTERS.index(letter) + 1, value


def parse_rewrites(text: str) -> list[str] | None:
    """The first REWRITES distinct, non-empty rewrites from a reply; None when it is unusable."""
    start, end = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[start : end + 1]) if start >= 0 else None
    except ValueError:
        return None
    rw = d.get("rewrites") if isinstance(d, dict) else None
    if not isinstance(rw, list):
        return None
    out = list(dict.fromkeys(str(x).strip() for x in rw if str(x).strip()))[:REWRITES]
    return out or None


def reorder(hits: list[dict], p: Page) -> list[dict]:
    return [c for c in hits if page_of(c) == p] + [c for c in hits if page_of(c) != p]


class PageTexts:
    """Chunk texts of every page in the index, in index order (the 'File:' line dropped)."""

    def __init__(self, rows: Iterable[dict]):
        self.by_page: dict[Page, list[tuple[str, str]]] = {}
        for r in rows:
            self.by_page.setdefault(page_of(r), []).append((r["chunk_id"], _body(r["text"])))

    def excerpt(self, p: Page, hits: list[dict], cap: int) -> str:
        """The page's retrieved chunks in rank order, then its other chunks, cut at cap characters."""
        own = dict(self.by_page.get(p, []))
        retrieved = [c for c in hits if page_of(c) == p]
        texts = [own.get(c["chunk_id"]) or _body(c["text"]) for c in retrieved]
        texts += [t for cid, t in own.items() if cid not in {c["chunk_id"] for c in retrieved}]
        return "\n".join(t for t in dict.fromkeys(texts) if t)[:cap]


def _body(text: str) -> str:
    return text.split("\n", 1)[1].strip() if text.startswith("File: ") and "\n" in text else text.strip()


@dataclass
class Ask:
    """One cached LLM question: its prompt hash, reply, calls (one retry on an unusable reply) and time."""

    prompt_hash: str = ""
    reply: str | None = None
    seconds: float = 0.0
    calls: int = 0
    cached: bool = False
    error: str | None = None


@dataclass
class Research:
    """D1 for one query: the rewrites, the merged top pages and the second reading."""

    rewrite: Ask
    rewrites: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    new_pages: int = 0  # merged top pages that were not in the plain top pages
    read: Ask | None = None
    chosen: str | None = None
    value: str | None = None
    moved: bool = False
    error: str | None = None


@dataclass
class AgentStep:
    gate: bool
    deck: bool = False  # D2: the gate passed only because a deck counts as one file
    candidates: list[str] = field(default_factory=list)
    prompt_hash: str = ""
    model: str = ""
    reply: str | None = None
    chosen: str | None = None
    value: str | None = None
    seconds: float = 0.0
    calls: int = 0
    cached: bool = False
    moved: bool = False
    error: str | None = None
    research: Research | None = None

    def row(self) -> dict:
        return asdict(self)


class Agent:
    """client: anything with complete(prompt) -> str. model: 'provider:model', part of the cache key."""

    def __init__(self, client, model: str, texts: PageTexts, cache_dir: Path, gate_k: int = 5, pages: int = 10,
                 page_chars: int = 4000, rrf_k: int = 60, deck_gate: bool = False):  # fmt: skip
        self.client, self.model, self.texts, self.cache_dir = client, model, texts, Path(cache_dir)
        self.gate_k, self.pages, self.page_chars, self.rrf_k = gate_k, pages, page_chars, rrf_k
        self.deck_gate = deck_gate

    def prompt(self, query: str, hits: list[dict], shown: list[Page]) -> str:
        blocks = []
        for i, p in enumerate(shown, 1):
            name = p[0].split("/")[-1]
            head = f"[{LETTERS[i - 1]}] {name}" + (f", page {p[1]}" if p[1] is not None else "")
            blocks.append(f"{head}\n{self.texts.excerpt(p, hits, self.page_chars)}\n")
        return PROMPT.format(n=len(shown), query=query, pages="\n".join(blocks))

    def ask(self, prompt: str, parse) -> tuple[object | None, Ask]:
        """parse(reply) -> a value or None (unusable); one retry. Replies are cached by prompt hash; errors are not."""
        from .llm import LLMError

        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        a = Ask(prompt_hash=key)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            hit = json.loads(path.read_text())
            a.reply, a.seconds, a.calls, a.cached = hit["reply"], hit["seconds"], hit["calls"], True
            return parse(a.reply), a
        parsed, t0 = None, time.time()
        try:
            while a.calls < 2 and parsed is None:
                a.calls += 1
                a.reply = self.client.complete(prompt)
                parsed = parse(a.reply)
        except LLMError as e:
            a.seconds, a.error = round(time.time() - t0, 2), str(e)
            return None, a
        a.seconds = round(time.time() - t0, 2)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"model": self.model, "prompt": prompt, "reply": a.reply,
                                   "seconds": a.seconds, "calls": a.calls}))  # fmt: skip
        tmp.replace(path)
        return parsed, a

    def __call__(self, query: str, hits: list[dict]) -> tuple[list[dict], AgentStep]:
        deck = False
        if not gate(hits, self.gate_k):
            deck = self.deck_gate and gate(hits, self.gate_k, decks=True)
            if not deck:
                return hits, AgentStep(gate=False, model=self.model)
        shown = pages_in_order(hits)[: min(max(self.pages, DECK_PAGES) if deck else self.pages, len(LETTERS))]
        parsed, a = self.ask(self.prompt(query, hits, shown), lambda t: parse_reply(t, len(shown)))
        step = AgentStep(gate=True, deck=deck, candidates=[unit_of(p) for p in shown], prompt_hash=a.prompt_hash, model=self.model,
                         reply=a.reply, seconds=a.seconds, calls=a.calls, cached=a.cached, error=a.error)  # fmt: skip
        if parsed is None:
            step.error = step.error or "malformed reply"
            return hits, step
        n, step.value = parsed
        if n is None:
            return hits, step
        step.chosen, step.moved = unit_of(shown[n - 1]), n > 1
        return reorder(hits, shown[n - 1]), step

    def merge(self, hits: list[dict], lists: list[list[dict]]) -> list[dict]:
        """RRF of the page lists (the plain list first), each page's chunks in the order the lists bring them."""
        from .search import rrf

        order = rrf([[unit_of(p) for p in pages_in_order(h)] for h in [hits, *lists]], self.rrf_k)
        by_unit: dict[str, dict[str, dict]] = {}
        for c in [*hits, *(c for h in lists for c in h)]:
            by_unit.setdefault(unit_of(page_of(c)), {}).setdefault(c["chunk_id"], c)
        return [c for u, _ in order for c in by_unit[u].values()]

    def reread(self, query: str, hits: list[dict], merged: list[dict], r: Research) -> list[dict]:
        shown = pages_in_order(merged)[: min(self.pages, len(LETTERS))]
        plain = pages_in_order(hits)
        r.candidates = [unit_of(p) for p in shown]
        r.new_pages = len(set(shown) - set(plain[: self.pages]))
        parsed, r.read = self.ask(self.prompt(query, merged, shown), lambda t: parse_reply(t, len(shown)))
        if parsed is None:
            r.error = r.read.error or "malformed reply"
            return hits
        n, r.value = parsed
        if n is None:
            return hits
        p = shown[n - 1]
        r.chosen, r.moved = unit_of(p), p != plain[0]
        return [c for c in merged if page_of(c) == p] + [c for c in hits if page_of(c) != p]

    def research_all(self, queries: list[str], hit_lists: list[list[dict]], out: list, search_many, workers: int = 4,
                     log=None) -> list:  # fmt: skip
        """D1 after run_all: out is its (hits, step) list. search_many(texts) -> one plain hit list per text, in one
        batch (the models run on one thread); the LLM calls run on `workers` threads."""
        todo = [i for i, (_, s) in enumerate(out) if s.gate and s.chosen is None and s.error is None]
        asked = self.pool(lambda i: self.ask(REWRITE_PROMPT.format(query=queries[i]), parse_rewrites), todo, workers,
                          log, "rewrites")  # fmt: skip
        for i, (rw, a) in zip(todo, asked, strict=True):
            out[i][1].research = Research(
                rewrite=a, rewrites=rw or [], error=None if rw else a.error or "malformed rewrites"
            )
        todo = [i for i in todo if out[i][1].research.rewrites]
        lists = iter(search_many([t for i in todo for t in out[i][1].research.rewrites]) if todo else [])
        merged = {i: self.merge(hit_lists[i], [next(lists) for _ in out[i][1].research.rewrites]) for i in todo}
        new = self.pool(lambda i: self.reread(queries[i], hit_lists[i], merged[i], out[i][1].research), todo, workers,
                        log, "re-reads")  # fmt: skip
        out = list(out)
        for i, hits in zip(todo, new, strict=True):
            out[i] = (hits, out[i][1])
        return out

    def run_all(self, queries: list[str], hit_lists: list[list[dict]], workers: int = 4, log=None):
        """The agent step for every query, in query order; LLM calls run on `workers` threads."""
        return self.pool(lambda i: self(queries[i], hit_lists[i]), range(len(queries)), workers, log, "queries")

    @staticmethod
    def pool(fn, items, workers: int, log=None, what: str = "queries") -> list:
        """fn(item) for every item, in order, on `workers` threads. A call that raises (the LLM cannot be used: a
        usage limit, no sign-in) stops the run; finished replies stay cached."""
        items = list(items)
        done, stop = Counter(), threading.Event()

        def one(i):
            if stop.is_set():
                return None
            try:
                out = fn(i)
            except Exception:
                stop.set()
                raise
            done["n"] += 1
            if log and done["n"] % 25 == 0:
                log(f"[agent] {done['n']}/{len(items)} {what}")
            return out

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(one, i) for i in items]
            return [f.result() for f in futures]


def summarize(steps: list[AgentStep]) -> dict:
    """Counts for the run notes: how many queries passed the gate, calls made, pages chosen and moved."""
    passed = [s for s in steps if s.gate]
    fresh = [s for s in passed if not s.cached and s.calls]
    return {"queries": len(steps), "gate_passed": len(passed), "llm_calls": sum(s.calls for s in fresh),
            "cache_hits": sum(s.cached for s in passed), "chosen": sum(s.chosen is not None for s in passed),
            "moved": sum(s.moved for s in passed),
            "no_page": sum(s.chosen is None and s.error is None for s in passed),
            "malformed": sum(s.error == "malformed reply" for s in passed),
            "errors": sum(s.error not in (None, "malformed reply") for s in passed),
            "llm_seconds": round(sum(s.seconds for s in fresh), 1),
            **({"deck_gate_passed": sum(s.deck for s in passed)} if any(s.deck for s in passed) else {}),
            **({"research": _research_stats([s.research for s in passed if s.research])}
               if any(s.research for s in passed) else {})}  # fmt: skip


def _research_stats(rs: list[Research]) -> dict:
    asks = [a for r in rs for a in (r.rewrite, r.read) if a is not None]
    fresh = [a for a in asks if not a.cached]
    return {"researched": len(rs), "llm_calls": sum(a.calls for a in fresh), "cache_hits": len(asks) - len(fresh),
            "chosen": sum(r.chosen is not None for r in rs), "moved": sum(r.moved for r in rs),
            "new_pages": sum(r.new_pages for r in rs),
            "no_page": sum(r.read is not None and r.chosen is None and r.error is None for r in rs),
            "errors": sum(r.error is not None for r in rs), "llm_seconds": round(sum(a.seconds for a in fresh), 1)}  # fmt: skip


def settings_of(cfg: Config) -> dict:
    """The agent settings a run records; a run with other settings is another system."""
    a = cfg["agent"]
    return {"backend": a["backend"], "provider": a["provider"], "model": a["model"], "gate_k": int(a["gate_k"]),
            "pages": int(a["pages"]), "page_chars": int(a["page_chars"]), "prompt": PROMPT_VERSION,
            **({"research": {"rewrites": REWRITES, "prompt": RESEARCH_VERSION}} if a["research"] == "on" else {}),
            **({"deck_gate": {"pages": DECK_PAGES}} if a["deck_gate"] == "on" else {})}  # fmt: skip


def from_cfg(cfg: Config, rows: Iterable[dict], who: str = "eval --mode agent") -> Agent:
    """The configured agent; stops with a clear message (naming `who` asked for it) when its LLM cannot run here."""
    from .llm import LLMUnavailable, check_available, get_client

    a = cfg["agent"]
    client = get_client(cfg, "agent", backend=a["backend"], provider=a["provider"], model=a["model"])
    try:
        check_available(client.s)
    except LLMUnavailable as e:
        raise SystemExit(f"{who}: {e}") from None
    return Agent(client, f"{a['provider']}:{a['model'] or 'default'}", PageTexts(rows),
                 cfg.path("cache_dir") / "agent", int(a["gate_k"]), int(a["pages"]), int(a["page_chars"]),
                 int(cfg["search"]["rrf_k"]), a["deck_gate"] == "on")  # fmt: skip
