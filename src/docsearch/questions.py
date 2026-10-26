"""Harder custom eval questions, drafted by an LLM from the indexed chunks, then hand-reviewed.

Types (meta.type):
  multi_doc  - needs 2+ docs/pages from the same source; gold = every cited page (scored with all@10).
  no_answer  - plausible but unanswerable from the corpus; gold = {} (scored by the abstention rate).
  image      - chart values / trends on a slide, image or PDF page; the LLM sees OCR text and, when the
               backend supports it, the page render (claude CLI: Read tool on the PNG; codex: -i; api: inline).
Grounding: the LLM sees the real chunk text with chunk_ids and must cite them; citations are validated, and
a closed-book pass drops questions the model can answer without the documents. Output is resumable: rows
are appended to data/questions_custom.jsonl and slots already written are skipped on a re-run.
No LLM is called on import; search/eval never need this module.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from . import ui
from .config import Config
from .draft_questions import CLOSED_BOOK, extract_json
from .llm import LLMClient, LLMError, LLMTimeout, check_available
from .manifest import chunk_units

TYPES = ("multi_doc", "no_answer", "image")
DEFAULT_MIX = {"multi_doc": 0.4, "no_answer": 0.25, "image": 0.35}
IMAGE_FORMATS = {"slide_image", "image", "pdf", "pptx"}

RULES = """Rules: phrase it as a real user would; never mention "excerpt", "chunk", ids or file names.
Return JSON only, no prose."""

MULTI = """You write HARD evaluation questions for a document search engine.
Below are passages from {n} different documents of the same collection, each with a chunk_id.
Write ONE natural question, a single sentence of at most 30 words with exactly one question mark, whose
answer genuinely needs facts from two of these documents: a comparison or a combination of one fact from
each. Do NOT join several sub-questions with "and"; do not ask about a third topic. If no natural question
links two passages, pick the two most related ones. It must be answerable only from these passages.
{rules}
{{"question": "...", "answer": "...", "citations": ["<chunk_id>", "<chunk_id>"]}}

{passages}
"""

NO_ANSWER = """You write evaluation questions for a document search engine.
Below are passages from one collection. Write ONE realistic question on the same topic that the
collection would plausibly be asked but these passages do NOT answer (a missing year, metric, segment
or entity). It must not be answerable from general knowledge either.
{rules}
{{"question": "...", "why_unanswerable": "..."}}

{passages}
"""

IMAGE = """You write HARD evaluation questions for a document search engine about charts and figures.
Below is the OCR text of a page/slide (chunk_id given){img_note}. Write ONE question (one sentence, at most
30 words) about a specific value, comparison or trend shown in its chart or table (e.g. "by how much did X
change from A to B"). The answer must be a number or a short fact taken from this text. Work only from the
text below: do not search, open files or run tools. Reply right away.
{rules}
{{"question": "...", "answer": "...", "citations": ["<chunk_id>"]}}

{passages}
"""


def load_chunks(cfg: Config, variant: str = "hybrid") -> list[dict]:
    from .chunk import chunks_path

    with chunks_path(cfg, variant).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def page_images(cfg: Config) -> dict[str, str]:
    """page unit -> rendered PNG path, from the render step's pages.jsonl (visual.pages_dir), if renders exist."""
    from .visual import pages_dir, read_pages

    if not (pages_dir(cfg) / "pages.jsonl").exists():
        return {}
    return {r["unit"]: r["path"] for r in read_pages(cfg)}


_WORD = re.compile(r"[a-z][a-z0-9]{3,}")


def _words(t: str) -> set[str]:
    return set(_WORD.findall(t.lower()))


def _numbers(t: str) -> int:
    return len(re.findall(r"\d[\d.,%$]*", t))


MAX_QMARKS, MAX_ANDS, MAX_WORDS = 2, 3, 45  # drop: more than 2 "?", 3+ "and", or over 45 words
_AND = re.compile(r"\band\b", re.I)


def stitched(q: str, max_qmarks: int = MAX_QMARKS, max_ands: int = MAX_ANDS, max_words: int = MAX_WORDS) -> list[str]:
    """Reasons a question looks like several sub-questions glued together ([] = fine)."""
    why = []
    if q.count("?") > max_qmarks:
        why.append(f"{q.count('?')} question marks")
    if len(_AND.findall(q)) >= max_ands:
        why.append(f"{len(_AND.findall(q))} 'and'")
    if len(q.split()) > max_words:
        why.append(f"{len(q.split())} words")
    return why


def lint(path: Path, say: Callable[[str], None] = print, show: int = 5) -> dict[str, int]:
    """Read-only: count (and show a few) stitched questions per type. Also reports the stricter "warn" tier
    (2 question marks or over 30 words), which the drafting prompt now forbids."""
    rows = read_rows(path)
    out: dict[str, int] = {}
    for kind in TYPES:
        rs = [r for r in rows if r["meta"]["type"] == kind and not is_dropped(r)]
        if not rs:
            continue
        bad = [(r, stitched(r["query"])) for r in rs if stitched(r["query"])]
        warn = [r for r in rs if not stitched(r["query"]) and stitched(r["query"], 1, 3, 30)]
        out[kind], out[f"{kind}_warn"] = len(bad), len(warn)
        say(f"{SHORT[kind]:9s} {len(rs):4d} questions: {len(bad)} flagged stitched, {len(warn)} borderline "
            "(2 '?' or >30 words)")  # fmt: skip
        for r, why in bad[:show]:
            say(f"  {r['qid']} [{', '.join(why)}] {r['query'][:140]}")
    return out


def _fmt(chs: list[dict], max_chars: int) -> str:
    return "\n\n".join(f"[chunk_id: {c['chunk_id']}]\n{c['text'][:max_chars]}" for c in chs)


class Drafter:
    def __init__(self, cfg: Config, client: LLMClient, chunks: list[dict], images: dict[str, str] | None = None,
                 seed: int = 11, closed_book: bool = True, max_chars: int = 1500, use_images: bool = True):  # fmt: skip
        self.cfg, self.client, self.seed = cfg, client, seed
        self.closed_book, self.max_chars = closed_book, max_chars
        self.images = images or {}
        self.use_images = use_images
        self.chunks = [c for c in chunks if len(c.get("text", "")) >= 200]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for c in self.chunks:
            self.by_source[c["source"]][c["doc_id"]].append(c)
        self.dropped: dict[str, int] = defaultdict(int)
        self.view: DraftView | None = None
        self.image_chars = 1000  # one dense OCR table is enough context; long ones slowed grok to timeouts

    # -- helpers
    def _call(self, prompt: str, images: list[str] | None = None, what: str = "draft") -> str:
        if self.view:
            self.view.call(what)
        try:
            return self.client.complete(prompt, images=images or None)
        finally:
            if self.view:
                self.view.call(None)

    def _ask(self, prompt: str, images: list[str] | None = None) -> dict | None:
        text = self._call(prompt, images)
        res = extract_json(text)
        if not res:
            ui.logger.warning("unparseable LLM reply (first 300 chars): %s", text[:300])
        return res

    def _closed_book_ok(self, q: str) -> bool:
        if not self.closed_book:
            return True
        ok = "UNKNOWN" in self._call(CLOSED_BOOK.format(q=q), what="closed-book check").upper()
        if not ok:
            self.dropped["closed_book"] += 1
        return ok

    def _gold(self, cited: list[dict]) -> dict[str, int]:
        gold: dict[str, int] = {}
        for c in cited:
            for u in chunk_units(c["doc_id"], c.get("pages"), "page"):
                gold[u] = 1
        return gold

    def _valid_citations(self, res: dict, shown: list[dict], min_docs: int) -> list[dict] | None:
        ids = {c["chunk_id"] for c in shown}
        cited = [self.by_id[c] for c in dict.fromkeys(res.get("citations") or []) if c in ids]
        if len({c["doc_id"] for c in cited}) < min_docs:
            self.dropped["citations"] += 1
            return None
        return cited

    def _row(self, kind: str, i: int, q: str, gold: dict, fmt: str, source: str, meta: dict) -> dict:
        return {"qid": f"custom:{kind}:{i}", "query": q.strip(), "source": source, "format": fmt, "level": "page",
                "gold": gold, "meta": {"type": kind, "status": "unreviewed", "backend": self.client.s.backend,
                                       "provider": self.client.s.provider, "model": self.client.s.model, **meta}}  # fmt: skip

    # -- generators (one slot each; deterministic per (type, i))
    def multi_doc(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:multi:{i}")
        sources = sorted(s for s, docs in self.by_source.items() if len(docs) >= 2)
        if not sources:
            return None
        src = rng.choice(sources)
        docs = self.by_source[src]
        a = rng.choice(sorted(docs))
        ca = rng.choice(docs[a])
        wa = _words(ca["text"])
        others = [c for d, cs in docs.items() if d != a for c in cs]
        pool = rng.sample(others, min(80, len(others)))  # related = most shared vocabulary in a sample
        pool.sort(key=lambda c: (-len(wa & _words(c["text"])), c["chunk_id"]))
        shown, seen = [ca], {a}
        for c in pool:
            if c["doc_id"] not in seen:
                shown.append(c)
                seen.add(c["doc_id"])
            if len(shown) == 3:
                break
        res = self._ask(MULTI.format(n=len(shown), rules=RULES, passages=_fmt(shown, self.max_chars)))
        if not res or not res.get("question"):
            self.dropped["bad_json"] += 1
            return None
        cited = self._valid_citations(res, shown, 2)
        if cited is not None and stitched(res["question"]):
            ui.logger.info("stitched multi-doc question dropped: %s", stitched(res["question"]))
            self.dropped["stitched"] += 1
            return None
        if cited is None or not self._closed_book_ok(res["question"]):
            return None
        fmts = sorted({c["format"] for c in cited})
        return self._row("multi_doc", i, res["question"], self._gold(cited), fmts[0] if len(fmts) == 1 else "mixed",
                         src, {"answer": res.get("answer", ""), "citations": [c["chunk_id"] for c in cited],
                               "formats": fmts})  # fmt: skip

    def no_answer(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:none:{i}")
        src = rng.choice(sorted(self.by_source))
        docs = self.by_source[src]
        picks = rng.sample(sorted(docs), min(2, len(docs)))
        shown = [rng.choice(docs[d]) for d in picks]
        res = self._ask(NO_ANSWER.format(rules=RULES, passages=_fmt(shown, self.max_chars)))
        if not res or not res.get("question"):
            self.dropped["bad_json"] += 1
            return None
        return self._row("no_answer", i, res["question"], {}, shown[0]["format"], src,
                         {"why": res.get("why_unanswerable", ""), "near": [c["chunk_id"] for c in shown]})  # fmt: skip

    def image(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:image:{i}")
        pool = [c for c in self.chunks if c["format"] in IMAGE_FORMATS and _numbers(c["text"]) >= 6]
        if not pool:
            return None
        c = rng.choice(sorted(pool, key=lambda c: c["chunk_id"]))
        units = chunk_units(c["doc_id"], c.get("pages"), "page")
        imgs = [self.images[u] for u in units[:1] if u in self.images and Path(self.images[u]).exists()]
        imgs = imgs if self.use_images else []
        note = " and the page image is attached" if imgs else ""
        res = self._ask(IMAGE.format(rules=RULES, img_note=note, passages=_fmt([c], self.image_chars)), imgs)
        if not res or not res.get("question"):
            self.dropped["bad_json"] += 1
            return None
        if self._valid_citations(res, [c], 1) is None:
            return None
        if stitched(res["question"]):
            self.dropped["stitched"] += 1
            return None
        if not self._closed_book_ok(res["question"]):
            return None
        return self._row("image", i, res["question"], self._gold([c]), c["format"], c["source"],
                         {"answer": res.get("answer", ""), "citations": [c["chunk_id"]], "saw_image": bool(imgs)})  # fmt: skip


def plan_counts(n: int, mix: dict[str, float]) -> dict[str, int]:
    tot = sum(mix.values())
    counts = {t: int(n * w / tot) for t, w in mix.items()}
    for t in sorted(mix, key=lambda t: -mix[t])[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def parse_mix(s: str | None) -> dict[str, float]:
    if not s:
        return dict(DEFAULT_MIX)
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        from .hard import HARD_TYPES

        if k.strip() not in TYPES + HARD_TYPES:
            raise ValueError(f"unknown question type {k!r}; use {TYPES + HARD_TYPES}")
        out[k.strip()] = float(v)
    return out


def custom_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "questions_custom.jsonl"


def read_rows(path: Path) -> list[dict]:
    """Tolerates a torn last line (e.g. power loss mid-append): it is skipped and re-drafted on resume."""
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                ui.logger.warning("skipping unreadable line in %s", path)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


DROP_LABELS = {"citations": "bad citation", "closed_book": "closed-book", "bad_json": "parse error",
               "llm_error": "call error", "timeout": "timeout (skipped)", "stitched": "stitched"}  # fmt: skip
SHORT = {"multi_doc": "multi-doc", "no_answer": "no-answer", "image": "image", "hard_neardup": "near-dup",
         "hard_table": "table", "hard_missed": "missed", "hard_multi": "pair"}  # fmt: skip


class DraftView:
    """Kept-questions bar with ETA, per-type counts, drops by reason, and the running call's elapsed time.
    Rich on a TTY; plain lines (one per kept/dropped question, plus a heartbeat) otherwise."""

    def __init__(self, total: int, want: dict[str, int], have: dict[str, int], dropped: dict[str, int],
                 heartbeat: float = 60.0):  # fmt: skip
        self.total, self.want, self.have, self.dropped = total, want, have, dropped
        self.kept0 = sum(have.values())
        self.t0, self.heartbeat, self._beat = time.time(), heartbeat, time.time()
        self.current: str | None = None
        self.call_t0 = 0.0
        self._rich = None

    def kept(self) -> int:
        return sum(self.have.values())

    def stats(self) -> str:
        types = "  ".join(f"{SHORT[t]} {self.have.get(t, 0)}/{w}" for t, w in self.want.items())
        drops = ", ".join(f"{DROP_LABELS.get(k, k)} {v}" for k, v in sorted(self.dropped.items()) if v) or "none"
        return f"{types}  | dropped: {drops}"

    def eta(self) -> str:
        new, el = self.kept() - self.kept0, time.time() - self.t0
        if new <= 0:
            return "?"
        return ui._fmt((self.total - self.kept()) / (new / el))

    def start(self) -> DraftView:
        if ui.is_tty():
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

            self._rich = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                                  TextColumn("{task.fields[eta]}"), TimeElapsedColumn(),
                                  console=ui.console(), transient=False)  # fmt: skip
            self._rich.start()
            self._main = self._rich.add_task("questions kept", total=self.total, completed=self.kept(), eta="ETA ?")
            self._stats = self._rich.add_task("  " + self.stats(), total=None, eta="")
            self._callt = self._rich.add_task("  waiting", total=None, eta="")
        else:
            print(f"questions: {self.kept()}/{self.total} kept  {self.stats()}", flush=True)
        return self

    def call(self, what: str | None, kind: str = "") -> None:
        """Mark the start (what=label) or end (None) of one LLM call."""
        if what is None:
            ui.logger.info("call %s done in %.1fs", self.current, time.time() - self.call_t0)
            self.current = None
            return
        self.current, self.call_t0 = f"{self._kind} {what}", time.time()
        if self._rich:
            self._rich.reset(self._callt, total=None, description=f"  call: {self.current}")
        elif time.time() - self._beat >= self.heartbeat:
            self._beat = time.time()
            print(f"questions: {self.kept()}/{self.total} kept, ETA {self.eta()}  {self.stats()}", flush=True)

    _kind = ""

    def slot(self, kind: str) -> None:
        self._kind = SHORT[kind]

    def update(self, msg: str | None = None) -> None:
        ui.logger.info(msg or self.stats())
        if self._rich:
            self._rich.update(self._main, completed=self.kept(), eta=f"ETA {self.eta()}")
            self._rich.update(self._stats, description="  " + self.stats())
        elif msg:
            print(f"{msg}  [{self.kept()}/{self.total}, ETA {self.eta()}]", flush=True)

    def stop(self) -> None:
        if self._rich:
            self._rich.stop()
            self._rich = None


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def draft(cfg: Config, client: LLMClient, n: int = 200, mix: dict[str, float] | None = None,
          out: Path | None = None, closed_book: bool = True, use_images: bool = True, seed: int = 11,
          chunks: list[dict] | None = None, images: dict[str, str] | None = None,
          say: Callable[[str], None] = print, max_attempts: float = 2.0, max_errors: int = 5,
          hard: bool = False, hard_from: str | None = None) -> Path:  # fmt: skip
    """Fill each type to its target count. Resumable: each kept question is appended (one line) as soon as it
    passes, slots that failed validation are remembered in <out>.failed.json, and slots whose call errored are
    retried on the next run. Ctrl-C stops between writes; re-run the same command to continue."""
    check_available(client.s, client.env, client.which)
    if hard:
        from .hard import HARD_MIX, REFERENCE_RUN, HardDrafter, hard_path, missed_pages

        out = out or hard_path(cfg)
    out = out or custom_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = read_rows(out)
    if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
        with out.open("a") as f:  # terminate a torn last line so the next append starts clean
            f.write("\n")
    done = {r["qid"] for r in have}
    chunks = chunks if chunks is not None else load_chunks(cfg)
    if hard:
        d = HardDrafter(cfg, client, chunks, {}, seed, closed_book, use_images=False,
                        missed=missed_pages(cfg, hard_from or REFERENCE_RUN))  # fmt: skip
        gen, prefix = d.generators(), "hard"
    else:
        d = Drafter(cfg, client, chunks, images if images is not None else page_images(cfg), seed, closed_book,
                    use_images=use_images)  # fmt: skip
        gen, prefix = {"multi_doc": d.multi_doc, "no_answer": d.no_answer, "image": d.image}, "custom"
    counts = plan_counts(n, mix or (HARD_MIX if hard else DEFAULT_MIX))
    fails_path = out.with_suffix(".failed.json")
    failed = set(json.loads(fails_path.read_text())) if fails_path.exists() else set()
    # questions dropped in review do not count, so a re-run refills them with new slots
    got = {k: sum(1 for r in have if r["meta"]["type"] == k and not is_dropped(r)) for k in counts}
    view = DraftView(n, counts, got, d.dropped).start()
    d.view = view
    errors = 0
    try:
        for kind, want in counts.items():
            view.slot(kind)
            i = 0
            while got[kind] < want and i < int(want * max_attempts) + 5:
                slot = f"{kind}:{i}"
                i += 1
                if f"{prefix}:{slot}" in done or slot in failed:
                    continue
                before = dict(d.dropped)
                try:
                    row = gen[kind](i - 1)
                except LLMTimeout as e:  # a slow sample: skip it for good, do not count toward the stop
                    d.dropped["timeout"] += 1
                    failed.add(slot)
                    _write_json_atomic(fails_path, sorted(failed))
                    ui.logger.warning("%s: %s; sample skipped", slot, e)
                    view.update(f"[questions] {slot}: timed out, sample skipped")
                    continue
                except LLMError as e:
                    errors += 1
                    d.dropped["llm_error"] += 1
                    ui.logger.error("%s: %s", slot, e)
                    view.update(f"[questions] {slot}: call error ({e}); will retry on the next run")
                    if errors >= max_errors:
                        raise
                    continue
                errors = 0
                if row is None:
                    failed.add(slot)
                    _write_json_atomic(fails_path, sorted(failed))
                    why = [DROP_LABELS.get(k, k) for k in d.dropped if d.dropped[k] > before.get(k, 0)]
                    view.update(f"[questions] {slot}: dropped ({', '.join(why) or 'no candidates'})")
                    continue
                row["meta"]["slot"] = slot
                with out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                have.append(row)
                got[kind] += 1
                view.update(f"[questions] kept {SHORT[kind]} {got[kind]}/{want}: {row['query'][:80]}")
            if got[kind] < want:
                view.update(f"[questions] {SHORT[kind]}: only {got[kind]}/{want} after {i} attempts")
    except KeyboardInterrupt:
        view.stop()
        say(f"[questions] interrupted: {len(have)} questions saved in {out}; re-run the same command to resume")
        raise
    finally:
        view.stop()
    say(f"[questions] {len(have)} questions in {out}; dropped this run: "
        f"{ {DROP_LABELS.get(k, k): v for k, v in d.dropped.items() if v} or 'none'}")  # fmt: skip
    return out


DROPPED = ("drop", "dropped")  # "drop" = review, "dropped" = lint --drop-stitched (meta.reason = "lint")


def is_dropped(r: dict) -> bool:
    return r["meta"].get("status") in DROPPED


def lint_flag(q: str) -> str:
    """'stitched' (dropped by the draft filter), 'borderline' (2 '?' or >30 words), or ''."""
    if stitched(q):
        return "stitched"
    return "borderline" if stitched(q, 1, 3, 30) else ""


_NUM = re.compile(r"\d[\d.,]*%?")


def short_doc(doc_id: str, n: int = 38) -> str:
    """'vidore_v3_hr/joint_employment_report_2025-KE012501' -> 'joint employment report 2025'; uuid names -> 'ocb · 716b3a07.pptx'."""
    src, _, name = doc_id.rpartition("/")
    base = re.sub(r"-[A-Z0-9]{6,}$", "", name)
    if re.match(r"^[0-9a-f]{8}-[0-9a-f-]{20,}", base):
        base = base.split("-")[0] + Path(base).suffix
    base = base.replace("_", " ")
    base = base if len(base) <= n else base[: n - 1] + "…"
    return f"{src.split('_')[0]} · {base}" if src else base


def snippet(text: str, answer: str = "", n: int = 200) -> tuple[str, list[str]]:
    """About n chars of one-line text, centred on the first number from the answer that appears in it."""
    t = " ".join(text.split())
    t = re.sub(r"^File: \S+\s*", "", t)
    nums = sorted({x.rstrip(".,") for x in _NUM.findall(answer or "") if len(x.rstrip(".,")) >= 1}, key=len,
                  reverse=True)  # fmt: skip
    pos = min((t.find(x) for x in nums if x in t), default=0)
    a = max(0, pos - n // 3)
    cut = t[a : a + n]
    return ("…" if a else "") + cut + ("…" if a + n < len(t) else ""), nums


def render_question(r: dict, n: int, total: int, by_id: dict[str, dict], sug: dict | None = None):
    """One rich Panel: header (n/total, type, format, status, lint badge), bold question, answer, a sub-panel
    per source (short doc name + page, ~200-char snippet with the answer's numbers highlighted), key footer."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    m = r["meta"]
    head = Text.assemble((f"{n}/{total}", "bold cyan"), "  ", (SHORT.get(m["type"], m["type"]), "magenta"), "  ",
                         (r["format"], "blue"), "  ", (m.get("status", ""), "yellow"))  # fmt: skip
    flag = lint_flag(r["query"])
    if flag:
        head.append("  ")
        head.append(f" {flag} ", style="bold white on red" if flag == "stitched" else "black on yellow")
    if m.get("reviewer"):
        head.append(f"  by {m['reviewer']}", style="dim")
    parts = []
    if sug:
        col = {"keep": "bold white on green", "drop": "bold white on red", "edit": "bold black on yellow"}
        parts.append(Text.assemble((f" suggested: {sug['suggestion']} ", col.get(sug["suggestion"], "bold")),
                                   "  ", (sug.get("reason", ""), "italic")))  # fmt: skip
        if sug.get("proposed_question"):
            parts.append(Text.assemble(("  proposed Q: ", "yellow"), sug["proposed_question"]))
        if sug.get("proposed_answer"):
            parts.append(Text.assemble(("  proposed A: ", "yellow"), sug["proposed_answer"]))
    parts.append(Text(r["query"], style="bold"))
    if m.get("answer"):
        parts.append(Text.assemble(("Answer: ", "green"), m["answer"]))
    if m.get("why"):
        parts.append(Text.assemble(("Why unanswerable: ", "green"), m["why"]))
    if not r["gold"]:
        parts.append(Text("gold: none (no-answer question)", style="dim"))
    for cid in m.get("citations") or m.get("near") or []:
        c = by_id.get(cid)
        if not c:
            parts.append(Panel(Text("(chunk not in the current index)", style="dim"), title=cid, title_align="left"))
            continue
        pages = ",".join(map(str, c.get("pages") or [])) or "1"
        snip, nums = snippet(c["text"], m.get("answer", ""))
        body = Text(snip)
        body.highlight_words(nums, style="bold black on bright_yellow")
        parts.append(Panel(body, title=f"{short_doc(c['doc_id'])} · p.{pages}", title_align="left",
                           border_style="dim", padding=(0, 1)))  # fmt: skip
    keys = Text.assemble(*((("[Enter]", "bold cyan"), " accept suggestion  ") if sug else ()), ("[k]", "bold green"), "eep  ", ("[d]", "bold red"), "rop  ", ("[e]", "bold"), "dit  ",
                         ("[s]", "bold"), "kip  ", ("[q]", "bold"), "uit")  # fmt: skip
    return Panel(Group(*parts), title=head, title_align="left", subtitle=keys, subtitle_align="left",
                 border_style="cyan", padding=(0, 1))  # fmt: skip


def review(cfg: Config, path: Path | None = None, include_reviewed: bool = False,
           ask: Callable[[str], str] = input, say: Callable[[str], None] = print,
           chunks: list[dict] | None = None, console=None, suggestions: Path | None = None,
           reviewer: str = "human") -> dict[str, int]:  # fmt: skip
    """Terminal loop: [k]eep, [d]rop, [e]dit question, [s]kip, [q]uit. Saves after every decision.
    With suggestions, each panel shows the suggested decision + reason; Enter accepts it."""
    path = path or custom_path(cfg)
    rows = read_rows(path)
    if not rows:
        say(f"no questions in {path}; run `docsearch questions draft` first")
        return {}
    con = console or ui.console()
    by_id = {c["chunk_id"]: c for c in (chunks if chunks is not None else load_chunks(cfg))}
    sugg = read_suggestions(suggestions) if suggestions else {}
    todo = [r for r in rows if include_reviewed or r["meta"].get("status") == "unreviewed"]
    stats: dict[str, int] = defaultdict(int)
    for n, r in enumerate(todo, 1):
        m = r["meta"]
        if con.is_terminal:
            con.clear()
        sug = sugg.get(r["qid"])
        con.print(render_question(r, n, len(todo), by_id, sug))
        while True:
            a = ask("> ").strip().lower()
            if a == "" and sug:
                _apply_one(r, sug, reviewer, accepted=True)
                stats[m["status"]] += 1
                write_rows(path, rows)
                break
            if a in ("k", "d", "e", "s", "q"):
                break
        if a == "" and sug:
            continue
        if a == "q":
            break
        if a == "s":
            stats["skipped"] += 1
            continue
        if a == "e":
            new = ask("new question (empty = keep text): ").strip()
            if new:
                m.setdefault("original_query", r["query"])
                r["query"] = new
            ans = ask("new answer (empty = keep): ").strip() if m["type"] != "no_answer" else ""
            if ans:
                m["answer"] = ans
            m["status"] = "edited"
        else:
            m["status"] = "keep" if a == "k" else "drop"
        m["reviewer"] = reviewer
        if sug:
            m["overrode_suggestion"] = sug["suggestion"]
        stats[m["status"]] += 1
        write_rows(path, rows)
    say(f"[review] {dict(stats)}; {sum(1 for r in rows if r['meta'].get('status') == 'unreviewed')} still unreviewed")
    return dict(stats)


def suggestions_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "questions_custom.suggestions.jsonl"


def read_suggestions(path: Path) -> dict[str, dict]:
    """{qid: {suggestion: keep|drop|edit, reason, proposed_question?, proposed_answer?}}"""
    out = {}
    for r in read_rows(path):
        if r.get("suggestion") not in ("keep", "drop", "edit"):
            raise ValueError(f"{path}: bad suggestion {r.get('suggestion')!r} for {r.get('id')}")
        out[r["id"]] = r
    return out


def _apply_one(r: dict, sug: dict, reviewer: str, accepted: bool = False) -> None:
    m = r["meta"]
    m["review_prev"] = {"status": m.get("status", "unreviewed"), "query": r["query"], "answer": m.get("answer")}
    if sug["suggestion"] == "edit":
        if sug.get("proposed_question"):
            m.setdefault("original_query", r["query"])
            r["query"] = sug["proposed_question"]
        if sug.get("proposed_answer"):
            m["answer"] = sug["proposed_answer"]
        m["status"] = "edited"
    else:
        m["status"] = sug["suggestion"]
    m["reviewer"] = reviewer
    m["review_reason"] = sug.get("reason", "")
    if accepted:
        m["accepted_suggestion"] = True


def apply_suggestions(path: Path, sugg_path: Path, only: set[str] | None = None, reviewer: str = "reviewer",
                      undo: bool = False, force: bool = False, say: Callable[[str], None] = print) -> dict[str, int]:  # fmt: skip
    """Write suggested decisions into the questions file (status keep/edited/drop, meta.reviewer, meta.review_reason,
    previous state in meta.review_prev), after a timestamped backup. Questions already reviewed by someone else
    are left alone unless force. undo restores every row this reviewer applied."""
    rows = read_rows(path)
    counts: dict[str, int] = defaultdict(int)
    if undo:
        for r in rows:
            m = r["meta"]
            if m.get("reviewer") == reviewer and "review_prev" in m:
                prev = m.pop("review_prev")
                r["query"] = prev["query"]
                if prev.get("answer") is None:
                    m.pop("answer", None)
                else:
                    m["answer"] = prev["answer"]
                m["status"] = prev["status"]
                for k in ("reviewer", "review_reason", "accepted_suggestion"):
                    m.pop(k, None)
                if m.get("original_query") == r["query"]:
                    m.pop("original_query")
                counts["restored"] += 1
    else:
        sugg = read_suggestions(sugg_path)
        for r in rows:
            s = sugg.get(r["qid"])
            if not s or (only and s["suggestion"] not in only) or is_dropped(r):
                continue
            if r["meta"].get("status") != "unreviewed" and not force:
                counts["skipped (already reviewed)"] += 1
                continue
            _apply_one(r, s, reviewer)
            counts[s["suggestion"]] += 1
    changed = sum(v for k, v in counts.items() if not k.startswith("skipped"))
    if changed:
        import shutil
        from datetime import datetime

        bak = path.with_name(f"{path.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
        shutil.copy2(path, bak)
        write_rows(path, rows)
        say(f"[apply-suggestions] {dict(counts)}; reviewer={reviewer}; backup: {bak}")
    else:
        say(f"[apply-suggestions] nothing to change ({dict(counts) or 'no matching suggestions'})")
    return dict(counts)


def provenance(rows: list[dict]) -> dict[str, int]:
    """Who reviewed the questions an eval used: {reviewer: n}, 'unreviewed' for the rest."""
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        m = r.get("meta", {})
        out[m.get("reviewer") or ("unreviewed" if m.get("status") == "unreviewed" else "human")] += 1
    return dict(out)


def lint_drop(path: Path, undo: bool = False, say: Callable[[str], None] = print) -> int:
    """Mark stitched questions dropped (status=dropped, reason=lint; previous status kept in meta.lint_prev),
    after a timestamped backup. --undo restores exactly those. Returns how many rows changed."""
    rows = read_rows(path)
    if not rows:
        say(f"no questions in {path}")
        return 0
    if undo:
        hit = [r for r in rows if r["meta"].get("reason") == "lint" and r["meta"].get("status") == "dropped"]
        for r in hit:
            r["meta"]["status"] = r["meta"].pop("lint_prev", "unreviewed")
            r["meta"].pop("reason", None)
    else:
        hit = [r for r in rows if not is_dropped(r) and stitched(r["query"])]
        for r in hit:
            r["meta"]["lint_prev"] = r["meta"].get("status", "unreviewed")
            r["meta"]["status"], r["meta"]["reason"] = "dropped", "lint"
    if hit:
        import shutil
        from datetime import datetime

        bak = path.with_name(f"{path.stem}.bak-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
        shutil.copy2(path, bak)
        write_rows(path, rows)
        say(f"[lint] {'restored' if undo else 'dropped'} {len(hit)} stitched questions; backup: {bak}")
    else:
        say(f"[lint] nothing to {'restore' if undo else 'drop'}")
    return len(hit)


def load_for_eval(path: Path, reviewed_only: bool = False) -> list[dict]:
    rows = [r for r in read_rows(path) if not is_dropped(r)]
    if reviewed_only:
        rows = [r for r in rows if r["meta"].get("status") in ("keep", "edited")]
    return rows


# -- paraphrase pass (questions_custom_para.jsonl): same qid, answer and gold; wording in the user's own words

PARAPHRASE = """Rewrite this search question the way a real user would type it, in your own words.
Keep exactly the same meaning and the same specific entities, years and numbers, so the answer stays: {answer}
Do NOT reuse phrases from the source text below: use synonyms and a different sentence structure, and do not
copy any run of {n} or more words from it. One sentence, at most 30 words.{retry}
Return JSON only: {{"question": "..."}}

Question: {q}

Source text (do not copy from it):
{source}
"""

_TOK = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def _toks(t: str) -> list[str]:
    return _TOK.findall(t.lower())


def ngram_overlap(question: str, sources: list[str], n: int = 3) -> dict:
    """longest: longest run of question words that appears verbatim in a source; share: fraction of the question's
    word n-grams found in the sources. Numbers and years count as words (they must stay, so short runs are fine)."""
    q = _toks(question)
    src = [_toks(s) for s in sources]
    grams = {k: set() for k in range(1, len(q) + 1)}
    for t in src:
        for k in range(1, min(len(q), 12) + 1):
            grams.setdefault(k, set()).update(tuple(t[i : i + k]) for i in range(len(t) - k + 1))
    longest = 0
    for i in range(len(q)):
        for k in range(longest + 1, min(len(q) - i, 12) + 1):
            if tuple(q[i : i + k]) in grams[k]:
                longest = k
            else:
                break
    qn = [tuple(q[i : i + n]) for i in range(len(q) - n + 1)]
    share = sum(g in grams.get(n, set()) for g in qn) / len(qn) if qn else 0.0
    return {"longest": longest, f"share{n}": round(share, 4)}


def para_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "questions_custom_para.jsonl"


def paraphrase(cfg: Config, client: LLMClient, max_ngram: int = 4, max_overlap: float = 0.25, attempts: int = 3,
               chunks: list[dict] | None = None, src: Path | None = None, out: Path | None = None,
               say: Callable[[str], None] = print) -> Path:  # fmt: skip
    """Rewrite every reviewed (keep/edited) question. Kept only if the longest copied n-gram < max_ngram and the
    3-gram share <= max_overlap (against the gold/near chunks); otherwise retried, then skipped (reported).
    Resumable: qids already in the output are skipped. The original file is never modified."""
    check_available(client.s, client.env, client.which)
    src, out = src or custom_path(cfg), out or para_path(cfg)
    rows = load_for_eval(src, reviewed_only=True)
    by_id = {c["chunk_id"]: c for c in (chunks if chunks is not None else load_chunks(cfg))}
    done = {r["qid"] for r in read_rows(out)}
    if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
        with out.open("a") as f:
            f.write("\n")
    stats: dict[str, int] = defaultdict(int)
    todo = [r for r in rows if r["qid"] not in done]
    say(f"[paraphrase] {len(todo)} to rewrite ({len(done)} already done), max n-gram < {max_ngram}, "
        f"3-gram share <= {max_overlap}")  # fmt: skip
    pb = ui.progress(len(todo), "paraphrase", "q ").start()
    try:
        for r in todo:
            m = r["meta"]
            sources = [by_id[c]["text"] for c in (m.get("citations") or m.get("near") or []) if c in by_id]
            src_text = "\n---\n".join(" ".join(t.split())[:1200] for t in sources) or "(none)"
            before = ngram_overlap(r["query"], sources)
            best, retry = None, ""
            for _ in range(attempts):
                try:
                    res = extract_json(client.complete(PARAPHRASE.format(q=r["query"], answer=m.get("answer") or
                                       "(not answerable from the documents)", n=max_ngram, retry=retry,
                                       source=src_text)))  # fmt: skip
                except LLMTimeout:
                    stats["timeout"] += 1
                    break
                q2 = (res or {}).get("question", "").strip()
                if not q2:
                    stats["parse error"] += 1
                    continue
                ov = ngram_overlap(q2, sources)
                if ov["longest"] < max_ngram and ov["share3"] <= max_overlap and not stitched(q2):
                    best = (q2, ov)
                    break
                retry = (f"\nYour previous rewrite copied {ov['longest']} words in a row from the source; "
                         "use different words.")  # fmt: skip
                stats["too close, retried"] += 1
            pb.advance()
            if not best:
                stats["skipped"] += 1
                ui.logger.info("paraphrase skipped %s", r["qid"])
                continue
            new = json.loads(json.dumps(r))
            new["query"] = best[0]
            new["meta"].update({"paraphrase_of": r["query"], "overlap_before": before, "overlap_after": best[1],
                                "paraphrase_model": f"{client.s.provider}:{client.s.model or 'default'}"})  # fmt: skip
            with out.open("a") as f:
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
            stats["kept"] += 1
    except KeyboardInterrupt:
        say(f"[paraphrase] interrupted; {stats['kept']} new rewrites saved in {out}; re-run to resume")
        raise
    finally:
        pb.close()
    say(f"[paraphrase] {dict(stats)}; {len(read_rows(out))} questions in {out}")
    return out
