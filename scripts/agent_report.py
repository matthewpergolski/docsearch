"""M34 (+agent, Phase C4): what the agent step did on the C3 table questions, from a run's per-question file.

  uv run python scripts/agent_report.py --qids-out DIR               writes judged.txt (hard table-cell + C3 held-out),
                                                                      c3_heldout.txt, c3_dev.txt into DIR
  uv run python scripts/agent_report.py m34_agent_tables [m34_agent_grok_tables]
  uv run python scripts/agent_report.py --gold-check                 other pages that also hold a held-out answer

A value is right when the gold number (meta.value_norm, sign dropped) is one of the numbers in the agent's value,
read as scripts/draft_table_questions.py reads numbers. A chosen page is right when it is a gold unit (for a
file-level question: a page of the gold file). Latency counts every step that made a call, cached or not (a cached
step keeps the seconds of its first call).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from draft_table_questions import norm, numbers  # noqa: E402

QUESTIONS = Path("data/questions_tables.jsonl")
# Input list of qids, one per line. Not published: write it with scripts/table_reach.py --qids-out.
HARD_TABLE_CELL = Path("logs/m33/hard_table_cell.txt")
STOP = {"and", "the", "for", "of", "in", "to", "total", "all", "other", "with", "from", "per", "by", "on", "at", "or"}


def table_questions() -> dict[str, dict]:
    rows = (json.loads(x) for x in QUESTIONS.read_text().splitlines() if x.strip())
    return {r["qid"]: r for r in rows if r["meta"].get("status") == "keep"}


def run_rows(name: str) -> list[dict]:
    files = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))
    if not files:
        raise SystemExit(f"no run file for {name} in results/runs/")
    return [json.loads(x) for x in Path(files[-1]).read_text().splitlines() if x.strip()]


def gold_number(q: dict) -> str:
    return norm(abs(Decimal(q["meta"]["value_norm"])))


def value_right(value: str | None, q: dict) -> bool:
    return value is not None and gold_number(q) in {norm(d) for d in numbers(value)}


def page_right(unit: str | None, q: dict) -> bool:
    if unit is None:
        return False
    return unit in q["gold"] if q["level"] == "page" else unit.split("#p")[0] in q["gold"]


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def report(name: str, qs: dict[str, dict]) -> None:
    rows = [r for r in run_rows(name) if r["qid"] in qs]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        q = qs[r["qid"]]
        groups[q["meta"]["split"]].append(r)
        groups[f"{q['meta']['split']} {'look-alike' if q['meta']['lookalike'] else 'single'}"].append(r)
    print(f"{name}: {len(rows)} C3 questions with an agent record")
    print(f"  {'group':22s} {'gate passed':>13s} {'page chosen':>13s} {'page right':>13s} {'value right':>13s} "
          f"{'value right (all q)':>20s} {'plain first right':>18s} {'agent first right':>18s} "
          f"{'value right, page not gold':>27s}")  # fmt: skip
    for g in sorted(groups):
        rs = groups[g]
        st = [r["agent"] for r in rs]
        passed = [s for s in st if s["gate"]]
        chosen = [(s, qs[r["qid"]]) for r, s in zip(rs, st, strict=True) if s["chosen"]]
        vright = sum(value_right(s["value"], qs[r["qid"]]) for r, s in zip(rs, st, strict=True))
        plain_first = sum(page_right(s["candidates"][0] if s["candidates"] else None, qs[r["qid"]])
                          for r, s in zip(rs, st, strict=True) if s["gate"])  # fmt: skip
        agent_first = sum(page_right(s["chosen"] or (s["candidates"] or [None])[0], qs[r["qid"]])
                          for r, s in zip(rs, st, strict=True) if s["gate"])  # fmt: skip
        off_gold = sum(value_right(s["value"], q) and not page_right(s["chosen"], q) for s, q in chosen)
        print(f"  {g:22s} {pct(len(passed), len(st)):>13s} {pct(len(chosen), len(passed)):>13s} "
              f"{pct(sum(page_right(s['chosen'], q) for s, q in chosen), len(chosen)):>13s} "
              f"{pct(vright, len(passed)):>13s} {pct(vright, len(st)):>20s} {pct(plain_first, len(passed)):>18s} "
              f"{pct(agent_first, len(passed)):>18s} {off_gold:>27d}")  # fmt: skip
    called = [r["agent"] for r in rows if r["agent"]["calls"]]
    secs = sorted(s["seconds"] for s in called)
    if secs:
        p95 = secs[min(len(secs) - 1, int(0.95 * len(secs)))]
        print(f"  latency: {len(secs)} steps with a call, median {statistics.median(secs):.1f}s, p95 {p95:.1f}s, "
              f"max {secs[-1]:.1f}s, total {sum(secs):.0f}s; {sum(s['calls'] for s in called)} calls, "
              f"{sum(s['error'] is not None for s in called)} errors or malformed")  # fmt: skip
    for r in rows:
        if r["agent"]["error"]:
            print(f"  {r['qid']}: {r['agent']['error'][:120]}")


def qids_out(out: Path, qs: dict[str, dict]) -> None:
    ran = {r["qid"] for r in run_rows("m28_default_tables")}
    held = [q for q, r in qs.items() if r["meta"]["split"] == "heldout" and q in ran]
    dev = [q for q, r in qs.items() if r["meta"]["split"] == "dev" and q in ran]
    hard = HARD_TABLE_CELL.read_text().split()
    out.mkdir(parents=True, exist_ok=True)
    for fname, ids in (("judged.txt", hard + held), ("c3_heldout.txt", held), ("c3_dev.txt", dev)):
        (out / fname).write_text("".join(f"{q}\n" for q in ids))
        print(f"{len(ids)} qids -> {out / fname}")


def words(label: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z ]", " ", label.lower()).split() if len(w) > 1 and w not in STOP}


def gold_check(qs: dict[str, dict]) -> None:
    """For each held-out PDF question, other pages (index text) that hold the gold number and every word of the gold
    row label: anywhere on the page, and in one table row (a 'label ..., column = value.' segment of the text)."""
    pages: dict[str, list[str]] = defaultdict(list)
    for line in Path("work/chunks/hybrid.jsonl").read_text().splitlines():
        c = json.loads(line)
        unit = f"{c['doc_id']}#p{min(c['pages'])}" if c.get("pages") else c["doc_id"]
        pages[unit].append(c["text"])
    held = [q for q in qs.values() if q["meta"]["split"] == "heldout" and q["format"] == "pdf"]
    want = {q["qid"]: (gold_number(q), words(q["meta"]["gold_cell"].get("row_label", ""))) for q in held}
    nolabel = [q["qid"] for q in held if not want[q["qid"]][1]]
    held = [q for q in held if want[q["qid"]][1]]
    on_page: dict[str, list[str]] = defaultdict(list)
    in_row: dict[str, list[str]] = defaultdict(list)
    for unit, texts in pages.items():
        text = "\n".join(texts)
        nums = {norm(d) for d in numbers(text)}
        low = set(re.sub(r"[^a-z ]", " ", text.lower()).split())
        rows = None
        for q in held:
            num, ws = want[q["qid"]]
            if unit in q["gold"] or num not in nums or not ws <= low:
                continue
            on_page[q["qid"]].append(unit)
            if rows is None:
                rows = [(set(re.sub(r"[^a-z ]", " ", seg.lower()).split()), {norm(d) for d in numbers(seg)})
                        for seg in re.split(r"\.\s|\n", text)]  # fmt: skip
            if any(ws <= toks and num in ns for toks, ns in rows):
                in_row[q["qid"]].append(unit)
    gold_doc = {q["qid"]: list(q["gold"])[0].split("#p")[0] for q in held}
    same_file = {k: [u for u in v if u.split("#p")[0] == gold_doc[k]] for k, v in in_row.items()}
    print(f"held-out PDF questions: {len(held) + len(nolabel)}; {len(nolabel)} with no row label ({nolabel}) not "
          f"checked. Of the {len(held)} others, with another page that holds the gold number and the row-label words: "
          f"on the page {len(on_page)}, in one table row {len(in_row)}, in one table row of the gold file "
          f"{sum(bool(v) for v in same_file.values())}")  # fmt: skip
    for q in held:
        num, ws = want[q["qid"]]
        if in_row[q["qid"]]:
            print(f"  {q['qid']:11s} value {num:>6s} label {sorted(ws)} gold p{list(q['gold'])[0].split('#p')[1]}: "
                  f"other pages on page {len(on_page[q['qid']])}, in a row {len(in_row[q['qid']])}, in a row of the "
                  f"gold file {[u.split('#')[1] for u in same_file[q['qid']]]}")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--qids-out", type=Path)
    ap.add_argument("--gold-check", action="store_true")
    a = ap.parse_args()
    qs = table_questions()
    if a.qids_out:
        qids_out(a.qids_out, qs)
    if a.gold_check:
        gold_check(qs)
    for name in a.runs:
        report(name, qs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
