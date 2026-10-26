"""Box subset: a smaller, fixed-seed slice of the pooled corpus for CPU-only machines.

Keeps a sample of each source's labeled questions, every page/file they reference (gold), plus
`distractor_ratio` x as many distractor pages/files drawn with a fixed seed from the same sources.
Spreadsheets/Office docs are capped at `office_max_chunks` chunks. `--full` disables all of it.
The selection is written to data/subset.json and recorded in the README and report.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from . import ui
from .config import Config
from .manifest import page_unit, read_manifest, read_questions, unit_doc


def subset_path(cfg: Config) -> Path:
    return cfg.path("data_dir") / "subset.json"


def build_subset(cfg: Config) -> dict:
    sc = cfg["subset"]
    rng = random.Random(int(sc["seed"]))
    docs = read_manifest(cfg.path("corpus_dir") / "manifest.csv")
    qs = [q for q in read_questions(cfg.path("data_dir") / "questions.jsonl") if q.gold]
    by_src: dict[str, list] = defaultdict(list)
    for q in qs:
        by_src[q.source].append(q)
    n_per = {k: int(v) for k, v in sc["questions_per_source"].items()}
    keep_q: list[str] = []
    gold_units: dict[str, set[str]] = defaultdict(set)
    for src, lst in sorted(by_src.items()):
        lst = sorted(lst, key=lambda q: q.qid)
        rng.shuffle(lst)
        n = n_per.get(src, len(lst))
        for q in lst[:n]:
            keep_q.append(q.qid)
            gold_units[src] |= set(q.gold)
    ratio = float(sc["distractor_ratio"])
    keep_docs: set[str] = set()
    keep_pages: dict[str, list[int]] = {}  # doc_id -> kept pages (paginated sources only)
    docs_by_src: dict[str, list] = defaultdict(list)
    for d in docs:
        docs_by_src[d.source].append(d)
    summary = {}
    for src, ds in sorted(docs_by_src.items()):
        gold = gold_units.get(src, set())
        page_level = any("#p" in u for u in gold) and src not in ("finslides", "pd_images")
        if page_level:  # e.g. ViDoRe: sample pages across all PDFs
            pool = [page_unit(d.doc_id, p) for d in ds for p in range(1, d.pages + 1)]
        else:  # one unit per file (OCB files; FinSlides slides and images are single-page files)
            pool = [d.doc_id if src == "ocb" else page_unit(d.doc_id, 1) for d in ds]
        others = sorted(set(pool) - gold)
        rng.shuffle(others)
        chosen = gold | set(others[: int(len(gold) * ratio)])
        for u in chosen:
            keep_docs.add(unit_doc(u))
            if page_level:
                keep_pages.setdefault(unit_doc(u), []).append(int(u.rsplit("#p", 1)[1]))
        summary[src] = {"gold_units": len(gold), "distractor_units": len(chosen) - len(gold),
                        "docs": len({unit_doc(u) for u in chosen}), "unit": "page" if page_level else "file"}  # fmt: skip
    for k in keep_pages:
        keep_pages[k] = sorted(set(keep_pages[k]))
    out = {
        "config": sc,
        "questions": sorted(keep_q),
        "docs": sorted(keep_docs),
        "pages": keep_pages,
        "summary": summary,
        "question_counts": {s: sum(1 for q in qs if q.qid in set(keep_q) and q.source == s) for s in by_src},
    }
    subset_path(cfg).write_text(json.dumps(out, indent=1))
    ui.info(f"[subset] {json.dumps(summary)}", flush=True)
    return out


def load_subset(cfg: Config, full: bool) -> dict | None:
    if full or not cfg["subset"].get("enabled", False):
        return None
    if (
        not (cfg.path("corpus_dir") / "manifest.csv").exists()
        or not (cfg.path("data_dir") / "questions.jsonl").exists()
    ):
        return None  # no benchmark corpus or questions here (a fresh clone), so there is nothing to slice
    p = subset_path(cfg)
    return json.loads(p.read_text()) if p.exists() else build_subset(cfg)
