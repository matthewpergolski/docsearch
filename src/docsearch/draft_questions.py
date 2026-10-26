"""Draft extra eval questions with an LLM (default role 'draft' = Haiku): multi-document questions
and no-answer questions. Output is marked unreviewed; hand-check before adding to questions.jsonl.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .config import Config
from .llm import LLMClient, check_available, settings
from .manifest import Doc, page_unit, read_manifest

MULTI = """You write evaluation questions for a document search engine.
Below are excerpts from two different documents. Write ONE question that can only be answered by
combining facts from BOTH excerpts (e.g. compare a figure, or connect two facts). Do not mention
"excerpt", "document A/B", or file names; phrase it as a real user would.
Return JSON only: {{"question": "...", "answer": "...", "uses": ["A", "B"]}}

Excerpt A ({a_label}):
{a_text}

Excerpt B ({b_label}):
{b_text}
"""

NO_ANSWER = """You write evaluation questions for a document search engine.
Below is an excerpt from a document. Write ONE realistic question on the SAME topic that this
excerpt does NOT answer (a plausible detail that is absent: a different year, metric, or entity).
It should look answerable from a collection like this one. Do not mention the excerpt.
Return JSON only: {{"question": "...", "why_unanswerable": "..."}}

Excerpt ({label}):
{text}
"""

CLOSED_BOOK = """Answer from general knowledge only, in one short sentence. If you cannot answer
reliably without a specific document, reply exactly UNKNOWN.
Question: {q}"""


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def page_excerpts(cfg: Config, d: Doc, max_chars: int = 1800) -> list[tuple[int, str]]:
    """(page_no, text) excerpts from the parsed document; page 0 = whole file (unpaginated)."""
    from .chunk import doc_text_by_page
    from .ingest import load_parsed, parsed_path

    if not parsed_path(cfg, d.doc_id).exists():
        return []
    text, bounds = doc_text_by_page(load_parsed(cfg, d.doc_id))
    out = [(p, text[s:e].strip()[:max_chars]) for s, e, p in bounds]
    return [(p, t) for p, t in out if len(t) > 300]


def draft(
    cfg: Config,
    n_multi: int = 20,
    n_none: int = 10,
    closed_book_filter: bool = True,
    client: LLMClient | None = None,
    excerpts=page_excerpts,
    seed: int = 7,
    out: Path | None = None,
) -> Path:
    client = client or LLMClient(settings(cfg, "draft"))
    check_available(client.s, client.env, client.which)
    rng = random.Random(seed)
    docs = read_manifest(cfg.path("corpus_dir") / "manifest.csv")
    by_source: dict[str, list[Doc]] = defaultdict(list)
    for d in docs:
        by_source[d.source].append(d)
    out = out or cfg.path("data_dir") / "drafted_questions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, dropped = [], 0

    def label(d: Doc, p: int) -> str:
        return f"{Path(d.path).name}" + (f" p.{p}" if p else "")

    def unit(d: Doc, p: int) -> str:
        return page_unit(d.doc_id, p) if p else d.doc_id

    def closed_book_ok(q: str) -> bool:
        if not closed_book_filter:
            return True
        return "UNKNOWN" in client.complete(CLOSED_BOOK.format(q=q)).upper()

    sources = [s for s, ds in by_source.items() if len(ds) >= 2]
    for i in range(n_multi):
        if not sources:
            break
        pool = by_source[rng.choice(sources)]
        a, b = rng.sample(pool, 2)
        ea, eb = excerpts(cfg, a), excerpts(cfg, b)
        if not ea or not eb:
            continue
        (pa, ta), (pb, tb) = rng.choice(ea), rng.choice(eb)
        res = extract_json(
            client.complete(MULTI.format(a_label=label(a, pa), a_text=ta, b_label=label(b, pb), b_text=tb))
        )
        if not res or not res.get("question"):
            continue
        if not closed_book_ok(res["question"]):
            dropped += 1
            continue
        level = "page" if pa and pb else "file"
        rows.append({
            "qid": f"draft_multi:{i}", "query": res["question"], "source": "drafted", "format": "multi",
            "level": level, "gold": {unit(a, pa if level == "page" else 0): 1, unit(b, pb if level == "page" else 0): 1},
            "meta": {"type": "multi_doc", "answer": res.get("answer", ""), "status": "unreviewed", "model": client.s.model},
        })  # fmt: skip
    for i in range(n_none):
        d = rng.choice(docs)
        ex = excerpts(cfg, d)
        if not ex:
            continue
        p, t = rng.choice(ex)
        res = extract_json(client.complete(NO_ANSWER.format(label=label(d, p), text=t)))
        if not res or not res.get("question"):
            continue
        rows.append({
            "qid": f"draft_none:{i}", "query": res["question"], "source": "drafted", "format": d.format,
            "level": "file", "gold": {},
            "meta": {"type": "no_answer", "near_doc": d.doc_id, "why": res.get("why_unanswerable", ""),
                     "status": "unreviewed", "model": client.s.model},
        })  # fmt: skip
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[draft] wrote {len(rows)} questions to {out} (closed-book filter dropped {dropped})")
    return out
