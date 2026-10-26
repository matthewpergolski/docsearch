"""C4b (Phase C4): a plain RAG baseline on the C3 held-out questions, against the M34h agent, on the same pages.

  uv run python scripts/rag_baseline.py --dry-run     rebuild the pages, check them, make no LLM call
  uv run python scripts/rag_baseline.py               ask the LLM (cached), score, print the table

Pages. For a gated question the baseline sees the 10 pages the agent saw in m34h_agent_tables, in its order, with
the text agent.py builds: the page's retrieved chunks first, then its other chunks. The run file keeps the pages but
not the retrieved chunks, so their order is read back from the agent's cached prompt; the prompt is then rebuilt with
agent.Agent.prompt and must hash to the run's prompt_hash. A question that passed no gate had no agent pages: the
baseline gets the single-page files of the plain top 10 (a file with more pages is left out, its page is unknown),
their chunks in index order. Those records say pages_exact false.

Prompt. answer.py's build_prompt and SYSTEM (numbered sources, cite [n], 'not found'), not the agent's JSON pick.
The LLM is the configured agent LLM (agent.backend / provider / model). Replies are cached by prompt hash in
work/cache/rag_baseline/; a usage limit (LLMUnavailable) stops the run and keeps the cache.

Scoring, as scripts/agent_report.py reads it: value right when the gold number is one of the numbers in the reply
(citation marks like [3] removed first), and as a stricter look when it is the first number; cited page right when the first cited source is a gold unit (file-level
gold: any page of the gold file), None when the reply cites nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_report import gold_number, page_right, run_rows, table_questions, value_right  # noqa: E402
from draft_table_questions import NUM, norm, to_dec  # noqa: E402

from docsearch import agent as A  # noqa: E402
from docsearch.answer import SYSTEM, build_prompt, parse_citations  # noqa: E402
from docsearch.config import load_config  # noqa: E402
from docsearch.index import connect, table_name  # noqa: E402
from docsearch.llm import LLMError, LLMUnavailable, check_available, get_client  # noqa: E402
from docsearch.search import COLS  # noqa: E402

# Input list of held-out qids, one per line. Not published: write it with scripts/agent_report.py --qids-out.
HELDOUT = Path("logs/m34/c3_heldout.txt")
AGENT_RUN, PLAIN_RUN = "m34h_agent_tables", "m28_default_tables"


@dataclass
class Record:
    qid: str
    gated: bool
    pages: list[A.Page]
    texts: list[str]
    pages_exact: bool
    prompt_hash: str = ""
    reply: str | None = None
    seconds: float = 0.0
    cached: bool = False
    error: str | None = None
    cited: str | None = None
    value_right: bool = False
    cited_page_right: bool | None = None


def page_from_unit(unit: str, texts: A.PageTexts) -> A.Page:
    doc, p = unit.split("#p")
    return (doc, int(p)) if (doc, int(p)) in texts.by_page else (doc, None)


def parses(block: str, pos: int, left: list[tuple[str, str]]):
    """Every way to read block[pos:] as whole chunk texts joined by newlines, the last one possibly cut."""
    if pos >= len(block):
        yield []
        return
    fits = [c for c in left if c[1].startswith(block[pos:]) or (block.startswith(c[1], pos)
            and block[pos + len(c[1]) : pos + len(c[1]) + 1] in ("", "\n"))]  # fmt: skip
    for c in sorted(fits, key=lambda c: -len(c[1])):
        for rest in parses(block, pos + len(c[1]) + 1, [x for x in left if x != c]):
            yield [c, *rest]


def hits_from_prompt(prompt: str, shown: list[A.Page], texts: A.PageTexts, cap: int) -> list[dict]:
    """Chunk dicts in the order their text appears in the agent's prompt: the retrieved chunks the run did not keep."""
    heads = [f"[{A.LETTERS[i]}] {p[0].split('/')[-1]}" + (f", page {p[1]}" if p[1] is not None else "") + "\n"
             for i, p in enumerate(shown)]  # fmt: skip
    starts, at = [], prompt.find("\n\n", prompt.find("Question: "))
    for h in heads:
        at = prompt.find(h, at)
        starts.append(at)
    hits = []
    for i, p in enumerate(shown):
        end = starts[i + 1] - 2 if i + 1 < len(shown) else len(prompt) - 1
        block = prompt[starts[i] + len(heads[i]) : end]
        chunks = [(cid, body) for cid, body in texts.by_page.get(p, []) if body]
        pages = [p[1]] if p[1] is not None else []
        page_hits = []
        for found in parses(block, 0, chunks):
            page_hits = [{"chunk_id": cid, "doc_id": p[0], "pages": pages, "text": body} for cid, body in found]
            if texts.excerpt(p, page_hits, cap) == block:
                break
        hits += page_hits
    return hits


def build_records(cfg, qs: dict[str, dict]) -> tuple[list[Record], int]:
    """One record per held-out question with its pages and texts; and how many rebuilt agent prompts hash-matched."""
    held = HELDOUT.read_text().split()
    agent_rows = {r["qid"]: r for r in run_rows(AGENT_RUN)}
    plain_rows = {r["qid"]: r for r in run_rows(PLAIN_RUN)}
    tbl = connect(cfg).open_table(table_name("hybrid"))
    texts = A.PageTexts(tbl.search().select(COLS).limit(tbl.count_rows()).to_list())
    a = cfg["agent"]
    cache = cfg.path("cache_dir") / "agent"
    records, matched = [], 0
    for qid in held:
        step = agent_rows[qid]["agent"]
        if step["gate"]:
            shown = [page_from_unit(u, texts) for u in step["candidates"]]
            hit = json.loads((cache / f"{step['prompt_hash']}.json").read_text())
            hits = hits_from_prompt(hit["prompt"], shown, texts, int(a["page_chars"]))
            ag = A.Agent(None, step["model"], texts, cache, int(a["gate_k"]), int(a["pages"]), int(a["page_chars"]))
            prompt = ag.prompt(qs[qid]["query"], hits, shown)
            ok = hashlib.sha256(f"{step['model']}\n{prompt}".encode()).hexdigest() == step["prompt_hash"]
            matched += ok
            if not ok:
                raise SystemExit(f"{qid}: the rebuilt agent prompt does not hash to {step['prompt_hash'][:12]}")
            body = [ag.texts.excerpt(p, hits, ag.page_chars) for p in shown]
            records.append(Record(qid, True, shown, body, True))
        else:
            shown = [(u, None) if (u, None) in texts.by_page else (u, 1) for u in plain_rows[qid]["top"]]
            shown = [p for p in shown if sum(q[0] == p[0] for q in texts.by_page) == 1 and p in texts.by_page]
            body = [texts.excerpt(p, [], int(a["page_chars"])) for p in shown]
            records.append(Record(qid, False, shown, body, False))
    return records, matched


def page_hits(rec: Record) -> list[dict]:
    return [{"doc_id": d, "pages": [p] if p is not None else [], "text": t}
            for (d, p), t in zip(rec.pages, rec.texts, strict=True)]  # fmt: skip


def rag_prompt(query: str, rec: Record) -> str:
    hits = page_hits(rec)
    return build_prompt(query, hits, {}, max_chars=10**9)


def ask(client, model: str, prompt: str, cache: Path, rec: Record) -> None:
    rec.prompt_hash = hashlib.sha256(f"{model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()
    path = cache / f"{rec.prompt_hash}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        rec.reply, rec.seconds, rec.cached = hit["reply"], hit["seconds"], True
        return
    t0 = time.time()
    try:
        rec.reply = client.complete(prompt, system=SYSTEM)
    except LLMError as e:
        rec.seconds, rec.error = round(time.time() - t0, 2), str(e)
        return
    rec.seconds = round(time.time() - t0, 2)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps({"model": model, "system": SYSTEM, "prompt": prompt, "reply": rec.reply,
                               "seconds": rec.seconds}))  # fmt: skip
    tmp.replace(path)


def score(rec: Record, q: dict) -> None:
    if rec.reply is None:
        return
    rec.value_right = value_right(re.sub(r"\[\d+\]", " ", rec.reply), q)
    cites = parse_citations(rec.reply, page_hits(rec), {})
    rec.cited = A.unit_of(rec.pages[cites[0]["n"] - 1]) if cites else None
    rec.cited_page_right = page_right(rec.cited, q) if rec.cited else None


def first_number_right(text: str | None, q: dict) -> bool:
    """A stricter look than value_right: only the first number in the text counts (citation marks removed)."""
    nums = [d for d in map(to_dec, NUM.findall(re.sub(r"\[\d+\]", " ", text or ""))) if d is not None]
    return bool(nums) and norm(abs(nums[0])) == gold_number(q)


def table(records: list[Record], qs: dict[str, dict]) -> None:
    agent_rows = {r["qid"]: r["agent"] for r in run_rows(AGENT_RUN)}
    plain_rows = {r["qid"]: r for r in run_rows(PLAIN_RUN)}
    for label, rs in (("gated", [r for r in records if r.gated]), ("all", records)):
        n = len(rs)
        rows = []
        for r in rs:
            q, s = qs[r.qid], agent_rows[r.qid]
            plain_first = page_right(plain_rows[r.qid]["top"][0], q)
            agent_first = page_right(s["chosen"] or plain_rows[r.qid]["top"][0], q)
            rows.append((r.value_right, value_right(s["value"], q), bool(r.cited_page_right),
                         page_right(s["chosen"], q), plain_first, agent_first, first_number_right(r.reply, q),
                         first_number_right(s["value"], q)))  # fmt: skip
        col = [sum(x[i] for x in rows) for i in range(8)]

        def paired(i, j, rows=rows):
            return (sum(x[i] and x[j] for x in rows), sum(x[i] and not x[j] for x in rows),
                    sum(x[j] and not x[i] for x in rows), sum(not x[i] and not x[j] for x in rows))  # fmt: skip

        print(f"\n{label} held-out questions: n={n}")
        print(f"  {'':40s} {'RAG':>8s} {'agent':>8s}   both / only RAG / only agent / neither")
        print(f"  {'value right':40s} {col[0]:>8d} {col[1]:>8d}   {' / '.join(map(str, paired(0, 1)))}")
        print(f"  {'value right, first number only (look)':40s} {col[6]:>8d} {col[7]:>8d}   "
              f"{' / '.join(map(str, paired(6, 7)))}")  # fmt: skip
        print(f"  {'page right (RAG first cite, agent pick)':40s} {col[2]:>8d} {col[3]:>8d}   "
              f"{' / '.join(map(str, paired(2, 3)))}")  # fmt: skip
        print(f"  {'plain search gold first':40s} {col[4]:>8d}")
        print(f"  {'agent list gold first (pick, else plain)':40s} {'':>8s} {col[5]:>8d}")
        print(f"  RAG: {sum(r.cited is None and r.reply is not None for r in rs)} replies cite nothing, "
              f"{sum('not found in the documents' in (r.reply or '').lower() for r in rs)} say not found, "
              f"{sum(r.reply is None for r in rs)} have no reply; agent: "
              f"{sum(agent_rows[r.qid]['gate'] and agent_rows[r.qid]['chosen'] is None for r in rs)} gated with no "
              f"page, {sum(not agent_rows[r.qid]['gate'] for r in rs)} not gated (no value, no pick)")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="rebuild and check the pages; no LLM call")
    ap.add_argument("--max-median", type=float, default=20.0, help="stop when the first 10 calls' median is over")
    ap.add_argument("--out", type=Path, default=Path("logs/c4b/rag_baseline.jsonl"))
    a = ap.parse_args()
    cfg = load_config()
    qs = table_questions()
    records, matched = build_records(cfg, qs)
    gated = sum(r.gated for r in records)
    print(f"{len(records)} held-out questions, {gated} gated; rebuilt agent prompts that hash to the run's "
          f"prompt_hash: {matched}/{gated}; not gated (approximate plain pages): "
          f"{[(r.qid, len(r.pages)) for r in records if not r.gated]}")  # fmt: skip
    if a.dry_run:
        return 0
    ac = cfg["agent"]
    client = get_client(cfg, "agent", backend=ac["backend"], provider=ac["provider"], model=ac["model"])
    check_available(client.s)
    model = f"{ac['backend']}:{ac['provider']}:{ac['model'] or 'default'}"
    cache = cfg.path("cache_dir") / "rag_baseline"
    fresh: list[float] = []
    stopped = None
    for i, rec in enumerate(records):
        try:
            ask(client, model, rag_prompt(qs[rec.qid]["query"], rec), cache, rec)
        except LLMUnavailable as e:
            stopped = f"stopped at {rec.qid} ({i} of {len(records)} done): {e}"
            break
        if not rec.cached:
            fresh.append(rec.seconds)
            print(f"[rag] {rec.qid}: {rec.seconds:.1f}s{' ERROR ' + rec.error[:100] if rec.error else ''}", flush=True)
            if len(fresh) == 10 and statistics.median(fresh) > a.max_median:
                stopped = f"stopped: median of the first 10 calls {statistics.median(fresh):.1f}s > {a.max_median}s"
                break
        score(rec, qs[rec.qid])
    done = [r for r in records if r.reply is not None or r.error]
    if fresh:
        s = sorted(fresh)
        print(f"\n{len(fresh)} new calls: first 10 median {statistics.median(fresh[:10]):.1f}s; all: median "
              f"{statistics.median(s):.1f}s, p95 {s[min(len(s) - 1, int(0.95 * len(s)))]:.1f}s, max {s[-1]:.1f}s, "
              f"total {sum(s):.0f}s; {sum(r.cached for r in records)} cached")  # fmt: skip
    print(f"errors: {[(r.qid, r.error[:100]) for r in records if r.error]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(asdict(r)) + "\n" for r in done))
    print(f"{len(done)} records -> {a.out}")
    if stopped:
        print(stopped)
        return 1
    table(records, qs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
