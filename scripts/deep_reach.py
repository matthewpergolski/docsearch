"""Phase D0 (a look, no LLM): for the questions whose gold is not in the plain top 10, how deep does plain search
find it, and what is the most a deeper +agent step could add?

  uv run python scripts/deep_reach.py m28_default:m34h_agent m28_default_hard:m34h_agent_hard [--out FILE]

Each argument is PLAIN:AGENT, two run names (latest file wins). The plain default pipeline runs again in-process
for the plain run's misses, on the plain run's questions in its order (so the cached query vectors are the same),
and keeps its ranked units to depth K=100 over every candidate (eval stops at the first 200 chunks, so a
file-level list can be longer here). The top 10, cut as eval cuts it, must equal the run file's top 10; the match count is printed.
Bound: every gold unit found in the top 50 goes first, on top of the agent run's top 10; nDCG@10 delta against the
agent run, over all questions of the set (paired bootstrap, the same as scripts/bootstrap_delta.py). The D1 bound
counts only the misses the agent gated and answered with no page, the ones D1 (decisions.tsv) would act on.
Rerank and page-rerank scores come from their caches; the script prints how many pairs it had to score.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEPTH = 100
REACH = 50
BUCKETS = (("1-10", 10), ("11-20", 20), ("21-50", 50), ("51-100", 100))
ABSENT = "absent"


@dataclass
class Reach:
    qid: str
    format: str
    gold: dict[str, int]
    rank: int | None  # best rank of any gold unit in the deep list (1-based), None past DEPTH
    deep: list[str]
    top_match: bool
    gated: bool
    no_page: bool
    agent_ndcg: float

    @property
    def bucket(self) -> str:
        return next((label for label, hi in BUCKETS if self.rank is not None and self.rank <= hi), ABSENT)

    def bound_ndcg(self, agent_top: list[str], d1_only: bool) -> float:
        """nDCG@10 when every gold unit in the top REACH goes first (or the agent's, when this one is out of reach)."""
        from docsearch.metrics import ndcg_at_k

        found = [u for u in self.deep[:REACH] if u in self.gold]
        if not found or (d1_only and not self.no_page):
            return self.agent_ndcg
        first = sorted(found, key=lambda u: -self.gold[u])
        return ndcg_at_k(first + [u for u in agent_top if u not in found], self.gold, 10)


def run_file(name: str) -> Path:
    files = sorted(glob.glob(f"results/runs/*-{name}.jsonl"))
    if not files:
        raise SystemExit(f"no run file for {name} in results/runs/")
    return Path(files[-1])


def read_run(name: str) -> list[dict]:
    return [json.loads(x) for x in run_file(name).read_text().splitlines() if x.strip()]


def ci(d: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rnd = random.Random(seed)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def questions_of(cfg, rows: list[dict]):
    """The run's questions in its order: the eval embeds them in one batch, so the same list hits the same cache."""
    from docsearch.eval import load_custom
    from docsearch.manifest import read_questions

    by_qid = {q.qid: q for q in read_questions(cfg.path("data_dir") / "questions.jsonl")}
    if any(r["qid"] not in by_qid for r in rows):
        by_qid.update({q.qid: q for q in load_custom(cfg, True, "hard")})
    return [by_qid[r["qid"]] for r in rows]


def deep_lists(cfg, qs, want: set[str], visual: str) -> tuple[dict[str, tuple[list[str], list[str]]], int, int]:
    """Per qid: the ranked units to DEPTH over every candidate, and the top 10 as eval cuts it (first 200 chunks)."""
    from docsearch.altembed import query_embeddings
    from docsearch.eval import query_vectors
    from docsearch.manifest import ranked_units
    from docsearch.models import PageReranker, load_reranker
    from docsearch.search import Retriever, rerank_pipeline

    qvecs = query_vectors(cfg, qs)
    vqs = query_embeddings(cfg, visual, [q.query for q in qs])
    r = Retriever(cfg, "hybrid", None, visual, None)
    reranker = load_reranker(cfg)
    page_rr = PageReranker(cfg, r.sc.page_reranker) if r.sc.page_reranker else None
    out = {}
    for i, q in enumerate(qs):
        if q.qid in want:
            hits = rerank_pipeline(q.query, r.candidates(q.query, qvecs[i], "hybrid_visual_fa", vqs[i]), reranker,
                                   r.sc, page_rr, None)  # fmt: skip
            out[q.qid] = ranked_units(hits, q.level)[:DEPTH], ranked_units(hits[:200], q.level)[:10]
    return out, reranker.calls, page_rr.calls if page_rr else 0


def reach_set(cfg, plain: str, agent: str, visual: str) -> tuple[list[Reach], dict[str, dict], int]:
    rows, ag = read_run(plain), {r["qid"]: r for r in read_run(agent)}
    qs = questions_of(cfg, rows)
    miss = {r["qid"]: r for r in rows if not set(r["gold"]) & set(r["top"])}
    deep, rr_calls, prr_calls = deep_lists(cfg, qs, set(miss), visual)
    print(f"{plain}: {len(rows)} questions, {len(miss)} misses; pairs scored outside the caches: rerank {rr_calls}, "
          f"page rerank {prr_calls}")  # fmt: skip
    gold = {q.qid: q.gold for q in qs}
    out = []
    for qid, r in miss.items():
        (units, top10), step = deep[qid], ag[qid].get("agent") or {}
        rank = next((i + 1 for i, u in enumerate(units) if u in gold[qid]), None)
        out.append(Reach(qid, r["format"], gold[qid], rank, units, top10 == r["top"], bool(step.get("gate")),
                         bool(step.get("gate")) and step.get("chosen") is None and step.get("error") is None,
                         ag[qid]["metrics"]["ndcg@10"]))  # fmt: skip
    return out, ag, len(rows)


def report(name: str, reach: list[Reach], ag: dict[str, dict], n: int) -> dict:
    labels = [b for b, _ in BUCKETS] + [ABSENT]
    fmts = sorted({x.format for x in reach})
    print(f"\n{name}: top 10 equal to the run file for {sum(x.top_match for x in reach)}/{len(reach)} misses")
    print(f"{'format':12s} {'n':>3s} " + " ".join(f"{b:>7s}" for b in labels) + "   gated no_page")
    for f in [*fmts, "all"]:
        xs = [x for x in reach if f in ("all", x.format)]
        c = Counter(x.bucket for x in xs)
        print(f"{f:12s} {len(xs):3d} " + " ".join(f"{c[b]:7d}" for b in labels)
              + f"   {sum(x.gated for x in xs):5d} {sum(x.no_page for x in xs):7d}")  # fmt: skip
    np_c = Counter(x.bucket for x in reach if x.no_page)
    print("no_page misses by bucket: " + ", ".join(f"{b} {np_c[b]}" for b in labels))
    res = {}
    for label, d1 in (("bound_top50_all", False), ("bound_top50_d1", True)):
        by_q = {x.qid: x.bound_ndcg(ag[x.qid]["top"], d1) - x.agent_ndcg for x in reach}
        d = [by_q.get(q, 0.0) for q in ag if ag[q].get("metrics")]
        mean, lo, hi = ci(d)
        res[label] = {
            "delta": round(mean, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "moved": sum(v > 0 for v in by_q.values()),
        }
        print(f"{label}: {res[label]['moved']} questions gain; nDCG@10 delta vs agent run {mean:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] over n={len(d)}")  # fmt: skip
    res["buckets"] = dict(Counter(x.bucket for x in reach))
    res["top_match"] = [sum(x.top_match for x in reach), len(reach)]
    res["no_page_buckets"] = dict(np_c)
    res["per_question"] = [{"qid": x.qid, "format": x.format, "rank": x.rank, "gated": x.gated, "no_page": x.no_page,
                            "top_match": x.top_match} for x in reach]  # fmt: skip
    return res


def bar(plain: str, agent: str) -> None:
    """The spread of the agent run's per-question deltas against plain."""
    b = {r["qid"]: r for r in read_run(plain) if r.get("metrics")}
    a = {r["qid"]: r for r in read_run(agent) if r.get("metrics")}
    d = [a[q]["metrics"]["ndcg@10"] - b[q]["metrics"]["ndcg@10"] for q in b]
    mean, lo, hi = ci(d)
    print(f"spread {agent} vs {plain}: delta {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], half width {(hi - lo) / 2:.4f}, "
          f"{sum(x != 0 for x in d)} of {len(d)} questions moved")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="PLAIN:AGENT run names")
    ap.add_argument("--visual", default="pplx-v2-late")
    ap.add_argument("--out", help="write the per-set results as JSON here")
    a = ap.parse_args()
    sys.path.insert(0, "src")
    from docsearch.config import load_config

    cfg = load_config(None)
    out = {}
    for pair in a.pairs:
        plain, agent = pair.split(":")
        reach, ag, n = reach_set(cfg, plain, agent, a.visual)
        out[plain] = report(plain, reach, ag, n)
        bar(plain, agent)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
