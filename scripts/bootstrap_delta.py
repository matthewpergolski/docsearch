"""Paired bootstrap CI for metric deltas between two eval runs (by run name; latest file wins).

Usage: uv run python scripts/bootstrap_delta.py <baseline_run_name> <challenger_run_name> [--by format]
       [--qids FILE]  (only the qids listed in FILE, one per line)
A name may be a comma list (m28_default_hard,m28_default_tables): the runs' questions are pooled.
"""

import argparse
import glob
import json
import random
from collections import defaultdict


def load(names: str) -> dict:
    out = {}
    for name in names.split(","):
        f = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))[-1]
        with open(f) as fh:  # no-answer rows carry no ranking metrics
            out.update({d["qid"]: d for d in map(json.loads, fh) if d.get("metrics")})
    return out


def ci(d: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("chal")
    ap.add_argument("--by", choices=["format", "source", "type"])
    ap.add_argument("--qids", help="a file of qids, one per line: only those questions")
    a = ap.parse_args()
    b, c = load(a.base), load(a.chal)
    qs = sorted(set(b) & set(c))
    if a.qids:
        with open(a.qids) as fh:
            keep = set(fh.read().split())
        qs = [q for q in qs if q in keep]
    print(f"paired n={len(qs)} (base {len(b)}, challenger {len(c)})")
    groups = defaultdict(list)
    for q in qs:
        groups["all"].append(q)
        if a.by:
            groups[b[q][a.by]].append(q)
    for g, ids in groups.items():
        for m in ["recall@10", "ndcg@10", "mrr@10"]:
            d = [c[q]["metrics"][m] - b[q]["metrics"][m] for q in ids]
            mean, lo, hi = ci(d)
            print(f"{g:12s} n={len(ids):3d} {m:9s} delta {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
