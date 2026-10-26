"""Hard-page question set (`questions draft --hard` -> data/questions_hard.jsonl).

Custom questions came out easy because their pages are easy to retrieve (M8). This samples pages search struggles
with, and asks for a question that tells the target apart from look-alikes; gold is exactly the target page:

  hard_neardup  near-duplicate slides/pages (same deck or source, high word overlap), e.g. the IBM slides where the
                reranker hurt. The question must hinge on the detail that differs.
  hard_table    dense tables and spreadsheets: a specific cell/row that needs the right sheet and row.
  hard_missed   pages that ranked low (or were missed) in the M3 public-set eval (per-question results file); the
                pages that wrongly outranked them are shown as distractors.
  hard_multi    two docs from the same source that share vocabulary but give different answers; gold is both pages.

All types keep the stitched filter, the closed-book filter and the paraphrase no-copy rule (no run of 4+ words
from the passages, 3-gram share <= 0.25).
"""

from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from . import ui
from .config import Config
from .manifest import read_questions
from .questions import RULES, Drafter, _fmt, _numbers, _words, chunk_units, ngram_overlap, stitched

HARD_TYPES = ("hard_neardup", "hard_table", "hard_missed", "hard_multi")
HARD_MIX = {"hard_neardup": 0.3, "hard_table": 0.25, "hard_missed": 0.25, "hard_multi": 0.2}
TABLE_FORMATS = {"xlsx", "csv"}
VISUAL_FORMATS = {"slide_image", "pptx", "pdf", "image"}
REFERENCE_RUN = "m3_skip_slide_rerank"  # the adopted M3 default on the public set

DISTINGUISH = """You write HARD evaluation questions for a document search engine.
Below is ONE TARGET passage and {k} DISTRACTOR passages from the same collection; they look alike. Write ONE
question (one sentence, at most 30 words, exactly one question mark) whose answer is in the TARGET only. The
distractors must NOT answer it: ask about the detail that differs ({focus}). Name enough context (topic, period,
entity) that a user could mean only the target. Use your own words: never copy a run of 4 or more words from the
passages; use synonyms and natural phrasing. Numbers and names may stay.
{rules}
{{"question": "...", "answer": "...", "citations": ["<TARGET chunk_id>"]}}

TARGET:
{target}

DISTRACTORS:
{distractors}
"""

PAIR = """You write HARD evaluation questions for a document search engine.
Below are passages from two different documents of the same collection. They use similar words but give
different facts. Write ONE natural question (one sentence, at most 30 words, exactly one question mark) that
needs a fact from EACH document to answer, such as a comparison or difference of their values. Do not join
sub-questions with "and". Use your own words: never copy a run of 4 or more words from the passages.
{rules}
{{"question": "...", "answer": "...", "citations": ["<chunk_id>", "<chunk_id>"]}}

{passages}
"""

FOCUS = {"hard_neardup": "a value, period, label or name that differs between the slides/pages",
         "hard_table": "one specific cell: the value for a given row and column, or a row-to-row difference",
         "hard_missed": "the specific fact only the target contains"}  # fmt: skip


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def hard_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "questions_hard.jsonl"


def missed_pages(cfg: Config, run: str = REFERENCE_RUN, max_rank: int = 3) -> list[dict]:
    """Page-level public questions whose first gold page ranked worse than max_rank (or not in the top 10), from the
    latest per-question file of `run`. Returns [{qid, gold, wrong}] with wrong = non-gold pages ranked above it."""
    files = sorted(glob.glob(str(cfg.path("results_dir") / "runs" / f"*-{run}.jsonl")))
    qpath = cfg.path("data_dir") / "questions.jsonl"
    if not files or not qpath.exists():
        ui.logger.warning("hard_missed: no per-question results for run %s (or no questions.jsonl)", run)
        return []
    gold = {q.qid: q for q in read_questions(qpath) if q.gold and q.level == "page"}
    out = []
    with open(files[-1]) as f:
        for line in f:
            r = json.loads(line)
            q = gold.get(r.get("qid"))
            if not q:
                continue
            top = r.get("top") or []
            ranks = [top.index(u) for u in q.gold if u in top]
            best = min(ranks) if ranks else len(top)
            if best >= max_rank:
                tgt = [u for u in q.gold if u not in top[:best]]
                out.append({"qid": q.qid, "gold": sorted(tgt), "wrong": [u for u in top[:best] if u not in q.gold]})
    return out


class HardDrafter(Drafter):
    PREFIX = "hard"

    def __init__(self, *args, missed: list[dict] | None = None, max_ngram: int = 4, max_overlap: float = 0.25, **kw):
        super().__init__(*args, **kw)
        self.max_ngram, self.max_overlap = max_ngram, max_overlap
        self.missed = missed or []
        self.words = {c["chunk_id"]: _words(c["text"]) for c in self.chunks}
        self.by_unit: dict[str, list[dict]] = defaultdict(list)
        self.by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in self.chunks:
            self.by_doc[c["doc_id"]].append(c)
            for u in chunk_units(c["doc_id"], c.get("pages"), "page"):
                self.by_unit[u].append(c)

    def generators(self) -> dict:
        return {"hard_neardup": self.neardup, "hard_table": self.table, "hard_missed": self.missed_page,
                "hard_multi": self.pair}  # fmt: skip

    def _row(self, kind, i, q, gold, fmt, source, meta):
        row = super()._row(kind, i, q, gold, fmt, source, meta)
        row["qid"] = f"{self.PREFIX}:{kind}:{i}"
        return row

    # -- helpers
    def _one_page(self, c: dict) -> list[str]:
        u = chunk_units(c["doc_id"], c.get("pages"), "page")
        return u if len(u) == 1 else []

    def _similar(self, c: dict, rng: random.Random, k: int = 3, min_j: float = 0.3, same_doc_only: bool = False):
        """Most look-alike chunks on other pages: the whole doc plus a sample of the same source."""
        own = set(self._one_page(c))
        cand = [x for x in self.by_doc[c["doc_id"]] if x is not c]
        if not same_doc_only:
            others = [x for d, cs in self.by_source[c["source"]].items() if d != c["doc_id"] for x in cs]
            cand += rng.sample(others, min(300, len(others)))
        w = self.words[c["chunk_id"]]
        scored = sorted(((jaccard(w, self.words[x["chunk_id"]]), x["chunk_id"], x) for x in cand), reverse=True)
        out, pages = [], set(own)
        for j, _, x in scored:
            p = tuple(chunk_units(x["doc_id"], x.get("pages"), "page"))
            if j < min_j or len(out) == k:
                break
            if not set(p) & pages:
                out.append(x)
                pages.update(p)
        return out

    def _no_copy(self, q: str, shown: list[dict]) -> bool:
        ov = ngram_overlap(q, [c["text"] for c in shown])
        if ov["longest"] >= self.max_ngram or ov["share3"] > self.max_overlap:
            self.dropped["copied"] += 1
            ui.logger.info("copied wording dropped (%s): %s", ov, q)
            return False
        return True

    def _checks(self, res: dict | None, shown: list[dict]) -> bool:
        if not res or not res.get("question"):
            self.dropped["bad_json"] += 1
            return False
        if stitched(res["question"]):
            self.dropped["stitched"] += 1
            return False
        return self._no_copy(res["question"], shown) and self._closed_book_ok(res["question"])

    def _distinguish(self, kind: str, i: int, target: dict, distractors: list[dict], meta: dict) -> dict | None:
        res = self._ask(DISTINGUISH.format(k=len(distractors), focus=FOCUS[kind], rules=RULES,
                                           target=_fmt([target], self.max_chars),
                                           distractors=_fmt(distractors, self.max_chars)))  # fmt: skip
        if res and res.get("question"):
            cited = set(res.get("citations") or [])
            if target["chunk_id"] not in cited or cited & {d["chunk_id"] for d in distractors}:
                self.dropped["citations"] += 1  # must cite the target and only the target
                return None
        if not self._checks(res, [target, *distractors]):
            return None
        return self._row(kind, i, res["question"], {u: 1 for u in self._one_page(target)}, target["format"],
                         target["source"], {"answer": res.get("answer", ""), "citations": [target["chunk_id"]],
                                            "distractors": [d["chunk_id"] for d in distractors], **meta})  # fmt: skip

    # -- generators
    def neardup(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:neardup:{i}")
        pool = sorted((c for c in self.chunks if c["format"] in VISUAL_FORMATS and self._one_page(c)),
                      key=lambda c: c["chunk_id"])  # fmt: skip
        slides = [c for c in pool if c["format"] in ("slide_image", "pptx")]
        if slides and i % 2 == 0:  # PDFs dominate look-alikes; keep half the slots on decks (where rerank hurt)
            pool = slides
        for _ in range(6):  # a few targets until one has real look-alikes
            if not pool:
                return None
            t = rng.choice(pool)
            sim = self._similar(t, rng, k=3, min_j=0.4)
            if sim:
                return self._distinguish("hard_neardup", i, t, sim, {"max_jaccard": round(
                    max(jaccard(self.words[t["chunk_id"]], self.words[s["chunk_id"]]) for s in sim), 3)})  # fmt: skip
        self.dropped["no_lookalike"] += 1
        return None

    def table(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:table:{i}")
        pool = sorted((c for c in self.chunks if self._one_page(c) and (c["format"] in TABLE_FORMATS
                       or _numbers(c["text"]) >= 15)), key=lambda c: c["chunk_id"])  # fmt: skip
        if not pool:
            return None
        for _ in range(6):  # need at least one look-alike table to tell apart
            t = rng.choice(pool)
            sim = self._similar(t, rng, k=2, min_j=0.2, same_doc_only=True) or self._similar(t, rng, k=2, min_j=0.2)
            if sim:
                return self._distinguish("hard_table", i, t, sim, {"numbers": _numbers(t["text"])})
        self.dropped["no_lookalike"] += 1
        return None

    def missed_page(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:missed:{i}")
        if not self.missed:
            self.dropped["no_missed_pages"] += 1
            return None
        m = rng.choice(self.missed)
        tgts = [c for u in m["gold"] for c in self.by_unit.get(u, []) if self._one_page(c)]
        if not tgts:
            self.dropped["no_chunks"] += 1
            return None
        t = max(tgts, key=lambda c: (len(c["text"]), c["chunk_id"]))
        wrong = [max(self.by_unit[u], key=lambda c: len(c["text"])) for u in m["wrong"][:3] if self.by_unit.get(u)]
        wrong = wrong or self._similar(t, rng, k=2, min_j=0.2)
        return self._distinguish("hard_missed", i, t, wrong, {"from_qid": m["qid"], "wrong_pages": m["wrong"][:3]})

    def pair(self, i: int) -> dict | None:
        rng = random.Random(f"{self.seed}:pair:{i}")
        sources = sorted(s for s, docs in self.by_source.items() if len(docs) >= 2)
        if not sources:
            return None
        for _ in range(6):  # a pair that really shares vocabulary (jaccard >= 0.2)
            got = self._draw_pair(rng, sources)
            if got:
                break
        else:
            self.dropped["no_lookalike"] += 1
            return None
        a, b = got
        return self._pair_question(i, a, b)

    def _draw_pair(self, rng, sources, min_j: float = 0.2):
        src = rng.choice(sources)
        cs = sorted(
            (c for d in self.by_source[src].values() for c in d if self._one_page(c)), key=lambda c: c["chunk_id"]
        )
        if len(cs) < 2:
            return None
        a = rng.choice(cs)
        others = [c for c in cs if c["doc_id"] != a["doc_id"]]
        b = max(rng.sample(others, min(300, len(others))) or [None],
                key=lambda c: (jaccard(self.words[a["chunk_id"]], self.words[c["chunk_id"]]), c["chunk_id"]) if c else 0)  # fmt: skip
        if b is None or jaccard(self.words[a["chunk_id"]], self.words[b["chunk_id"]]) < min_j:
            return None
        return a, b

    def _pair_question(self, i, a, b):
        src = a["source"]
        shown = [a, b]
        res = self._ask(PAIR.format(rules=RULES, passages=_fmt(shown, self.max_chars)))
        if res and res.get("question") and set(res.get("citations") or []) != {a["chunk_id"], b["chunk_id"]}:
            self.dropped["citations"] += 1
            return None
        if not self._checks(res, shown):
            return None
        fmts = sorted({a["format"], b["format"]})
        gold = {u: 1 for c in shown for u in self._one_page(c)}
        return self._row("hard_multi", i, res["question"], gold, fmts[0] if len(fmts) == 1 else "mixed", src,
                         {"answer": res.get("answer", ""), "citations": [a["chunk_id"], b["chunk_id"]], "formats": fmts,
                          "jaccard": round(jaccard(self.words[a["chunk_id"]], self.words[b["chunk_id"]]), 3)})  # fmt: skip
