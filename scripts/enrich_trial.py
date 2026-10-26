"""Phase H trial: pick the files, time the LLM, write the enriched chunk variant.

Usage:
  uv run python scripts/enrich_trial.py pick [--ref m28_default] [--heldout 0.3] [--seed 2026]
  uv run python scripts/enrich_trial.py probe [--n 10] [--batch 8]
  uv run python scripts/enrich_trial.py build [--batch 8] [--max-calls 1000] [--workers 4] [--slides]
pick reads data/questions.jsonl, the reference run in results/runs and work/chunks/hybrid.jsonl, and writes
work/enrich/trial.json: the split (qid -> dev | heldout), the weak and control files, and the held-out
questions on each. No LLM, no question text in the output.
probe sends --n batches of trial chunks to the LLM (Claude Haiku through the claude CLI) and prints the time.
build gives every chunk of the weak and control files a context line (cached in work/cache/enrich) and writes
work/chunks/hybrid_ctx.jsonl: all hybrid chunks, the trial files' with their line. Then index it with
`uv run docsearch index --variant hybrid_ctx` (unchanged chunks reuse the hybrid vectors). build reads only
trial.json's file lists and the chunks, never the questions. build --slides (H2) gives every slide image a
line instead and writes hybrid_ctx_slides.jsonl.

- Split: within each gold file, about `--heldout` of its questions go to held-out (seeded), so a weak file
  keeps held-out questions to judge it on.
- Weak: a file that holds the gold of a dev question whose gold is not first in the reference run.
- Control: files whose dev questions all have the gold first, sampled to the weak files' count per format.
"""

import argparse
import collections
import glob
import json
import random
import time
from pathlib import Path

from docsearch.chunk import chunks_path, read_chunks
from docsearch.config import load_config
from docsearch.enrich import Enricher, apply

TRIAL = Path("work/enrich/trial.json")


def gold_file(unit: str) -> str:
    return unit.split("#", 1)[0]


def pick(a) -> None:
    rnd = random.Random(a.seed)
    with open("data/questions.jsonl") as fh:
        qs = [json.loads(line) for line in fh]
    with open(sorted(glob.glob(f"results/runs/*-{a.ref}.jsonl"))[-1]) as fh:
        ref = {d["qid"]: d for d in map(json.loads, fh) if d.get("metrics")}
    qs = [q for q in qs if q["qid"] in ref]
    fmt_of, chunks = {}, collections.Counter()
    with open("work/chunks/hybrid.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            fmt_of[r["doc_id"]] = r["format"]
            chunks[r["doc_id"]] += 1

    by_file = collections.defaultdict(list)
    for q in qs:
        by_file[gold_file(sorted(q["gold"])[0])].append(q["qid"])
    split = {}
    for f in sorted(by_file):
        ids = sorted(by_file[f])
        rnd.shuffle(ids)
        n = round(a.heldout * len(ids)) if len(ids) > 1 else int(rnd.random() < a.heldout)
        for i, qid in enumerate(ids):
            split[qid] = "heldout" if i < n else "dev"

    first = {q["qid"]: ref[q["qid"]]["metrics"]["mrr@10"] >= 1 for q in qs}
    files_of = {q["qid"]: {gold_file(u) for u in q["gold"]} for q in qs}
    dev = [q["qid"] for q in qs if split[q["qid"]] == "dev"]
    weak = sorted({f for qid in dev if not first[qid] for f in files_of[qid] if f in chunks})
    seen = {f for qid in dev for f in files_of[qid]}
    good = sorted(seen - set(weak))
    control = []
    for fmt, n in sorted(collections.Counter(fmt_of.get(f, "?") for f in weak).items()):
        pool = [f for f in good if fmt_of.get(f) == fmt]
        control += sorted(rnd.sample(pool, min(n, len(pool))))

    def held(files: list[str]) -> list[str]:
        s = set(files)
        return sorted(q for q in split if split[q] == "heldout" and files_of[q] & s)

    out = {"ref": a.ref, "seed": a.seed, "heldout_share": a.heldout, "split": split, "weak": weak,
           "control": control, "heldout_weak": held(weak), "heldout_control": held(control)}  # fmt: skip
    TRIAL.parent.mkdir(parents=True, exist_ok=True)
    TRIAL.write_text(json.dumps(out, indent=1))

    def line(name: str, files: list[str]) -> None:
        by = collections.Counter(fmt_of.get(f, "?") for f in files)
        n = sum(chunks[f] for f in files)
        hq = held(files)
        print(f"{name}: {len(files)} files, {n} chunks, {len(hq)} held-out questions; files by format {dict(by)}")
        print(f"  held-out first in {a.ref}: {sum(first[q] for q in hq)}/{len(hq)}")

    print(
        f"split: {sum(v == 'dev' for v in split.values())} dev, {sum(v == 'heldout' for v in split.values())} held-out"
    )
    line("weak", weak)
    line("control", control)
    print(f"all files: {len(chunks)}, {sum(chunks.values())} chunks")


def enricher(batch: int) -> Enricher:
    from docsearch.llm import check_available, get_client

    cfg = load_config()
    client = get_client(cfg, "enrich", backend="cli", provider="claude", model="haiku")
    check_available(client.s)
    return Enricher(client, "claude:haiku", cfg.path("cache_dir") / "enrich", batch)


def trial_chunks(slides: bool = False):
    """All hybrid chunks, and those of the trial files (or, with slides, of every slide image: H2)."""
    allc = read_chunks(chunks_path(load_config(), "hybrid"))
    if slides:
        return allc, [c for c in allc if c.format == "slide_image"]
    t = json.loads(TRIAL.read_text())
    files = set(t["weak"]) | set(t["control"])
    return allc, [c for c in allc if c.doc_id in files]


def probe(a) -> None:
    _, mine = trial_chunks()
    by_doc = collections.defaultdict(list)
    for c in mine:
        by_doc[c.doc_id].append(c)
    rnd = random.Random(a.seed)
    docs = rnd.sample(sorted(by_doc), a.n)
    e = enricher(a.batch)
    for d in docs:
        part = by_doc[d][: a.batch]
        t0 = time.time()
        got = e.lines(part)
        print(
            f"{d}: {len(part)} chunks, {len(got)} lines, {time.time() - t0:.1f}s; e.g. {next(iter(got.values()), '-')}"
        )
    s = e.stats
    n = sum(min(len(by_doc[d]), a.batch) for d in docs)
    print(
        f"probe: {s['calls']} calls, {s['cached_batches']} cached, {s['seconds']:.1f}s, {n} chunks, missing {s['missing']}"
    )
    if s["calls"]:
        per = s["seconds"] / max(1, s["batches"] - s["cached_batches"])
        need = sum(-(-len(v) // a.batch) for v in by_doc.values())
        print(f"projection: {need} batches for the trial files at {per:.1f}s each = {need * per / 3600:.1f} h serial")


def build(a) -> None:
    allc, mine = trial_chunks(a.slides)
    batches = sum(-(-n // a.batch) for n in collections.Counter(c.doc_id for c in mine).values())
    if batches > a.max_calls:
        raise SystemExit(f"{batches} batches is over --max-calls {a.max_calls}")
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    by_doc = collections.defaultdict(list)
    for c in mine:
        by_doc[c.doc_id].append(c)
    shares = [[] for _ in range(a.workers)]
    for i, d in enumerate(sorted(by_doc, key=lambda d: -len(by_doc[d]))):
        shares[i % a.workers] += by_doc[d]
    workers = [enricher(a.batch) for _ in shares]
    t0, done, lock = time.time(), [0], Lock()

    def tick(n: int) -> None:
        with lock:
            done[0] += n
            if done[0] % 200 < n:
                calls = sum(w.stats["calls"] for w in workers)
                print(f"[enrich] {done[0]}/{len(mine)} chunks, {calls} calls, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        parts = list(pool.map(lambda wc: wc[0].lines(wc[1], on_batch=tick), zip(workers, shares, strict=True)))
    lines = {k: v for part in parts for k, v in part.items()}
    e = workers[0]
    for w in workers[1:]:
        for k, v in w.stats.items():
            e.stats[k] += v
    variant = "hybrid_ctx_slides" if a.slides else "hybrid_ctx"
    out = chunks_path(load_config(), variant)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for c in apply(allc, lines):
            fh.write(json.dumps(c.__dict__) + "\n")
    tmp.replace(out)
    log = {"chunks": len(mine), "lines": len(lines), **e.stats, "wall_seconds": round(time.time() - t0, 1),
           "batch": a.batch, "workers": a.workers, "seconds_per_chunk": round(e.stats["seconds"] / max(1, len(mine)), 2)}  # fmt: skip
    Path(f"work/enrich/build_{variant}.json").write_text(json.dumps(log, indent=1))
    print(f"build: {json.dumps(log)} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--ref", default="m28_default")
    p.add_argument("--heldout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("build")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-calls", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--slides", action="store_true", help="H2: every slide image, not the trial files")
    a = ap.parse_args()
    {"pick": pick, "probe": probe, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
