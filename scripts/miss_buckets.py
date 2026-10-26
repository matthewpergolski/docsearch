"""Sort a run's misses (gold not first) by where the first hit went: the wrong file, or the right file or slide
deck at the wrong page. A file-level change (metadata, filters, file-first search) can only fix the first kind.

  uv run python scripts/miss_buckets.py m28_default m28_default_hard [--period]

--period keeps only questions that name a year (1900-2099), a quarter (Q1-Q4, 1Q18) or the word "quarter".
Slide images (format slide_image in the corpus manifest) are one file per slide; their deck is the file name
before "_page_". Read-only.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from docsearch.config import load_config
from docsearch.manifest import file_of, read_manifest, unit_doc

PERIOD = re.compile(r"\b(19|20)\d{2}\b|\bQ[1-4]\b|\b[1-4]Q\d{2}\b|quarter", re.I)
QUESTION_FILES = ("data/questions.jsonl", "data/questions_hard.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def latest_run(name: str) -> Path:
    files = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))
    if not files:
        raise SystemExit(f"no run file for {name} in results/runs/")
    return Path(files[-1])


def doc_formats() -> dict[str, str]:
    """doc_id -> format, from the corpus manifest."""
    return {d.doc_id: d.format for d in read_manifest(load_config().path("corpus_dir") / "manifest.csv")}


def unit_file(unit: str, formats: dict[str, str]) -> str:
    return file_of(unit, formats.get(unit_doc(unit), ""))


def buckets(rows: list[dict], questions: dict[str, dict], period_only: bool, formats: dict[str, str]) -> Counter:
    out: Counter = Counter()
    for r in rows:
        q = questions.get(r["qid"], {})
        if r["metrics"]["mrr@10"] >= 1 or (period_only and not PERIOD.search(q.get("query", ""))):
            continue
        gold = {unit_file(u, formats) for u in q.get("gold") or {}}
        top = unit_file(r["top"][0], formats) if r["top"] else ""
        out[(r["format"], "right file, wrong page" if top in gold else "wrong file first")] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--period", action="store_true")
    a = ap.parse_args()
    questions = {q["qid"]: q for f in QUESTION_FILES if Path(f).exists() for q in read_jsonl(Path(f))}
    formats = doc_formats()
    for name in a.runs:
        run = latest_run(name)
        rows = read_jsonl(run)
        out = buckets(rows, questions, a.period, formats)
        what = " naming a period" if a.period else ""
        print(f"{name} ({run.name}): {len(rows)} questions, {sum(out.values())} misses{what}")
        for kind in ("wrong file first", "right file, wrong page"):
            by = {fmt: c for (fmt, k), c in sorted(out.items()) if k == kind}
            print(f"  {kind}: {sum(by.values())} {by}")


if __name__ == "__main__":
    main()
