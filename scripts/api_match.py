"""Phase G proof: does models.backend api give what the local backend gives?

uv run python scripts/api_match.py cosine --embed-url http://127.0.0.1:8088/v1 [--n 200]
    embeds the first N public questions with the local embedder and through the api client (a server on
    --embed-url), and prints the cosine of each pair: min, median, max, and the share >= 0.999.
uv run python scripts/api_match.py top10 <run_a> <run_b>
    compares the stored top-10 lists of two eval runs (results/runs, latest file per name), question by question:
    the share of equal lists, and each question that differs (by format).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from collections import Counter

import numpy as np


def cosine(a: argparse.Namespace) -> None:
    from docsearch.config import load_config
    from docsearch.manifest import read_questions
    from docsearch.models import load_embedder

    cfg = load_config()
    qs = [q for q in read_questions(cfg.path("data_dir") / "questions.jsonl") if q.gold][: a.n]
    texts = [q.query for q in qs]
    loc = load_embedder(cfg)
    local = loc.queries(texts)
    api_cfg = copy.deepcopy(cfg)
    api_cfg.raw["models"].update(backend="api", embed_api_url=a.embed_url)
    emb = load_embedder(api_cfg)
    api = emb.queries(texts)
    cos = np.sum(local * api, axis=1) / (np.linalg.norm(local, axis=1) * np.linalg.norm(api, axis=1))
    print(f"n={len(cos)} questions; local {loc.name} on {loc.device} vs api {emb.ep.url} ({emb.requests} requests)")
    print(f"cosine min {cos.min():.8f}  median {np.median(cos):.8f}  max {cos.max():.8f}  "
          f"share >= 0.999: {np.mean(cos >= 0.999):.3f}  dims local {local.shape[1]} api {api.shape[1]}")  # fmt: skip
    worst = np.argsort(cos)[:3]
    for i in worst:
        print(f"  lowest: {cos[i]:.6f}  {qs[i].qid}  {texts[i][:80]!r}")


def load(name: str) -> dict:
    f = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))[-1]
    with open(f) as fh:
        return {d["qid"]: d for d in map(json.loads, fh)}


def top10(a: argparse.Namespace) -> None:
    x, y = load(a.run_a), load(a.run_b)
    qs = sorted(set(x) & set(y))
    diff = [q for q in qs if x[q]["top"] != y[q]["top"]]
    print(f"paired n={len(qs)} ({a.run_a} {len(x)}, {a.run_b} {len(y)}); equal top-10 lists: "
          f"{len(qs) - len(diff)}/{len(qs)} = {(len(qs) - len(diff)) / len(qs):.3f}")  # fmt: skip
    print("differ by format:", dict(Counter(x[q]["format"] for q in diff)))
    same_set = [q for q in diff if set(x[q]["top"]) == set(y[q]["top"])]
    print(f"of those, same 10 units in another order: {len(same_set)}")
    for q in diff:
        ga, gb = x[q]["top"], y[q]["top"]
        first = next((i for i, (u, v) in enumerate(zip(ga, gb, strict=False)) if u != v), min(len(ga), len(gb)))
        nd = (x[q]["metrics"].get("ndcg@10", 0), y[q]["metrics"].get("ndcg@10", 0))
        print(f"  {q} [{x[q]['format']}] first change at rank {first + 1}; nDCG@10 {nd[0]:.4f} -> {nd[1]:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cosine")
    c.add_argument("--embed-url", required=True)
    c.add_argument("--n", type=int, default=200)
    t = sub.add_parser("top10")
    t.add_argument("run_a")
    t.add_argument("run_b")
    a = ap.parse_args()
    {"cosine": cosine, "top10": top10}[a.cmd](a)


if __name__ == "__main__":
    main()
