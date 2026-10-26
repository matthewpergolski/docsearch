"""Phase C0: for each wrong-page miss of a run, say where the answer's numbers sit on the gold page, from Docling's
parse: in a table cell, only in body text, or not on the page. Answers with no number are counted apart. Read-only.

  uv run python scripts/table_reach.py m28_default_hard m28_default --formats pdf docx pptx xlsx
  uv run python scripts/table_reach.py m28_default_hard --qids-out hard_table_cell.txt --keyed-out hard

A number matches when its digits (sign, separators and % stripped) equal a number token in the cell or text.
--qids-out writes the qids whose answer is in a table cell. --keyed-out PREFIX splits every question of the runs into
PREFIX.value_keyed (the query holds a cell value: a decimal, or an integer of 100 or more that is not a year;
docsearch.cells.query_values) and PREFIX.entity_named (the rest).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

QUESTION_FILES = ("data/questions.jsonl", "data/questions_hard.jsonl")
NUM = re.compile(r"\d[\d,.   ]*\d|\d")


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text or ""):
        n = re.sub(r"[,   ]", "", m).rstrip(".")
        if n and not (len(n) == 4 and n.startswith(("19", "20"))):
            out.add(n)
    return out


def page_parts(parsed: dict, page: int) -> tuple[set[str], set[str]]:
    def on_page(item: dict) -> bool:
        return any(p.get("page_no") == page for p in item.get("prov") or [])

    cells = {n for t in parsed.get("tables") or [] if on_page(t)
             for c in (t.get("data") or {}).get("table_cells") or [] for n in numbers(c.get("text", ""))}  # fmt: skip
    text = {n for t in parsed.get("texts") or [] if on_page(t) for n in numbers(t.get("text", ""))}
    return cells, text


def latest_run(name: str) -> Path:
    files = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))
    if not files:
        raise SystemExit(f"no run file for {name} in results/runs/")
    return Path(files[-1])


def reach(run: Path, questions: dict[str, dict], formats: set[str]) -> list[tuple[str, str, str]]:
    """(format, where the answer sits, qid) for each right-file, wrong-page miss."""
    out: list[tuple[str, str, str]] = []
    for r in (json.loads(x) for x in run.read_text().splitlines() if x.strip()):
        q = questions.get(r["qid"])
        if q is None or r["metrics"]["mrr@10"] >= 1 or r["format"] not in formats or not r["top"]:
            continue
        gold_docs = {u.split("#")[0] for u in q["gold"]}
        if r["top"][0].split("#")[0] not in gold_docs:
            continue
        want = numbers(str(q["meta"].get("answer", "")))
        if not want:
            out.append((r["format"], "answer has no number", r["qid"]))
            continue
        where = "not on the gold page"
        for u in q["gold"]:
            doc, _, page = u.partition("#p")
            path = Path("work/parsed") / f"{doc}.json"
            if not page or not path.exists():
                continue
            cells, text = page_parts(json.loads(path.read_text()), int(page))
            if want & cells:
                where = "in a table cell"
                break
            if want & text:
                where = "in body text only"
        out.append((r["format"], where, r["qid"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--formats", nargs="+", default=["pdf", "docx", "pptx", "xlsx"])
    ap.add_argument("--qids-out", type=Path, help="write the qids whose answer is in a table cell, one per line")
    ap.add_argument("--keyed-out", help="PREFIX: write PREFIX.value_keyed and PREFIX.entity_named (all run qids)")
    a = ap.parse_args()
    questions = {q["qid"]: q for f in QUESTION_FILES if Path(f).exists()
                 for q in (json.loads(x) for x in Path(f).read_text().splitlines() if x.strip())}  # fmt: skip
    in_cell, run_qids = [], []
    for name in a.runs:
        run = latest_run(name)
        run_qids += [json.loads(x)["qid"] for x in run.read_text().splitlines() if x.strip()]
        rows = reach(run, questions, set(a.formats))
        in_cell += [qid for _, where, qid in rows if where == "in a table cell"]
        out = Counter((fmt, where) for fmt, where, _ in rows)
        print(f"{name}: {sum(out.values())} right-file, wrong-page misses in {', '.join(a.formats)}")
        for (fmt, where), n in sorted(out.items()):
            print(f"  {fmt:6s} {where}: {n}")
    for p in (a.qids_out, Path(a.keyed_out) if a.keyed_out else None):
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
    if a.qids_out:
        a.qids_out.write_text("".join(f"{q}\n" for q in in_cell))
        print(f"{len(in_cell)} qids in a table cell -> {a.qids_out}")
    if a.keyed_out:
        from docsearch.cells import query_values

        keyed = {q for q in run_qids if q in questions and query_values(questions[q]["query"])}
        for kind, qs in (("value_keyed", [q for q in run_qids if q in keyed]),
                         ("entity_named", [q for q in run_qids if q not in keyed])):  # fmt: skip
            Path(f"{a.keyed_out}.{kind}").write_text("".join(f"{q}\n" for q in qs))
            print(f"{len(qs)} {kind} -> {a.keyed_out}.{kind}")


if __name__ == "__main__":
    main()
