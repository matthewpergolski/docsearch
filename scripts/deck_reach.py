"""Phase D2-0 (a look, no LLM): how many questions would a deck-aware agent gate newly send to the reader, and
where is their gold page in the pages the reader would see?

  uv run python scripts/deck_reach.py m28_default:m34h_agent m28_default_hard:m34h_agent_hard [--out FILE]

Each argument is PLAIN:AGENT, two run names (latest file wins). The plain default pipeline runs again in-process
for every question of the plain run, in its order (the cached query vectors and score caches are the same), as
scripts/deep_reach.py does. Per question: the M34h gate (the top 5 pages hold 2+ pages of one doc_id), checked
against the gate the agent run recorded, and the deck gate (the same, but a slide counts as its deck,
manifest.file_of). New = the deck gate passes and the M34h gate does not: one reader call each, so the new count is
the call projection. For the new ones: the gold page's rank in the page order the reader sees (1, 2-10, 11-20,
past 20 or absent). Bounds against the agent run: every new question with its gold page in the top 10 (or top 20)
gets it first, the ranked list reordered exactly as the agent reorders it.

--agent-check (no LLM): the agent itself runs on every question with a client that refuses every call, once with
agent.deck_gate off and once on. Off: gate, prompt hash and pick must equal the agent run's on every question, all
from the cache. On: the questions the M34h gate passes must still equal it; the ones that need a call are the
deck-gated ones, and their count is the call projection from the code itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

GATE_K = 5
BUCKETS = (("1", 1), ("2-10", 10), ("11-20", 20))
PAST = "past 20"


def bucket(rank: int | None) -> str:
    return next((label for label, hi in BUCKETS if rank is not None and rank <= hi), PAST)


def deck_gate(hits: list[dict], k: int) -> bool:
    from docsearch.agent import page_of
    from docsearch.manifest import file_of

    first: dict = {}
    for c in hits:
        first.setdefault(page_of(c), c)
    files = Counter(file_of(c["doc_id"], c.get("format", "")) for c in list(first.values())[:k])
    return bool(files) and max(files.values()) >= 2


class Refuse:
    def complete(self, prompt: str) -> str:
        from docsearch.llm import LLMError

        raise LLMError("no call in the check")


def agent_check(cfg, r, queries: list[str], hit_lists: list[list[dict]], ag_rows: list[dict]) -> dict:
    from docsearch import agent

    rows = r.tbl.search().limit(r.tbl.count_rows()).to_list()
    out = {}
    for deck in ("off", "on"):
        cfg["agent"]["deck_gate"] = deck
        ag = agent.from_cfg(cfg, rows, who="deck_reach --agent-check")
        ag.client = Refuse()
        steps = [s for _, s in ag.run_all(queries, hit_lists, 1)]
        old = [(s, a) for s, a in zip(steps, ag_rows, strict=True) if (a.get("agent") or {}).get("gate")]
        same = sum(s.gate and s.prompt_hash == a["agent"]["prompt_hash"] and s.chosen == a["agent"]["chosen"]
                   and s.cached for s, a in old)  # fmt: skip
        gate_same = sum(s.gate == bool((a.get("agent") or {}).get("gate")) for s, a in zip(steps, ag_rows, strict=True))
        need = [s for s in steps if s.error == "no call in the check"]
        out[deck] = {"m34h_gated": len(old), "same_hash_pick_cached": same, "gate_same": gate_same,
                     "need_call": len(need), "need_call_deck": sum(s.deck for s in need),
                     "deck_gated": sum(s.deck for s in steps)}  # fmt: skip
        print(f"agent check, deck_gate {deck}: M34h-gated {len(old)}, same prompt hash + pick + from cache {same}; "
              f"gate equal to the agent run {gate_same}/{len(steps)}; deck-gated {out[deck]['deck_gated']}; "
              f"need a new call {len(need)} (deck-gated {out[deck]['need_call_deck']})")  # fmt: skip
    cfg["agent"]["deck_gate"] = "off"
    return out


def look(cfg, plain: str, agent_run: str, visual: str, check: bool = False) -> dict:
    from deep_reach import ci, questions_of, read_run

    from docsearch.agent import gate, page_of, pages_in_order, reorder, unit_of
    from docsearch.altembed import query_embeddings
    from docsearch.eval import query_vectors
    from docsearch.manifest import ranked_units
    from docsearch.metrics import ndcg_at_k
    from docsearch.models import PageReranker, load_reranker
    from docsearch.search import Retriever, rerank_pipeline

    rows, ag = read_run(plain), {r["qid"]: r for r in read_run(agent_run)}
    qs = questions_of(cfg, rows)
    qvecs = query_vectors(cfg, qs)
    vqs = query_embeddings(cfg, visual, [q.query for q in qs])
    r = Retriever(cfg, "hybrid", None, visual, None)
    reranker = load_reranker(cfg)
    page_rr = PageReranker(cfg, r.sc.page_reranker) if r.sc.page_reranker else None
    per, top_match, gate_match, hit_lists = [], 0, 0, []
    for i, (q, row) in enumerate(zip(qs, rows, strict=True)):
        hits = rerank_pipeline(q.query, r.candidates(q.query, qvecs[i], "hybrid_visual_fa", vqs[i]), reranker,
                               r.sc, page_rr, None)  # fmt: skip
        hit_lists.append(hits)
        top_match += ranked_units(hits[:200], q.level)[:10] == row["top"]
        old, deck = gate(hits, GATE_K), deck_gate(hits, GATE_K)
        gate_match += old == bool((ag[q.qid].get("agent") or {}).get("gate"))
        pages = pages_in_order(hits)
        golds = [p for p in pages if (unit_of(p) if q.level == "page" else p[0]) in q.gold]
        rank = pages.index(golds[0]) + 1 if golds else None
        base = ag[q.qid]["metrics"]["ndcg@10"]
        first = ndcg_at_k(ranked_units(reorder(hits, golds[0])[:200], q.level), q.gold, 10) if golds else base
        per.append({"qid": q.qid, "format": q.format, "type": q.meta.get("type", "labeled"), "old_gate": old,
                    "deck_gate": deck, "new": deck and not old, "rank": rank, "bucket": bucket(rank),
                    "agent_ndcg": base, "gold_first_ndcg": first,
                    "top5_decks": sorted({page_of(c)[0].split("/")[-1] for c in hits if page_of(c) in pages[:GATE_K]})})  # fmt: skip
    new = [x for x in per if x["new"]]
    print(f"\n{plain} vs {agent_run}: {len(per)} questions; top 10 equal to the run file {top_match}/{len(per)}; "
          f"M34h gate equal to the agent run's {gate_match}/{len(per)}; pairs scored outside the caches: rerank "
          f"{reranker.calls}, page rerank {page_rr.calls if page_rr else 0}")  # fmt: skip
    print(f"gate passed: M34h {sum(x['old_gate'] for x in per)}, deck {sum(x['deck_gate'] for x in per)}, "
          f"new {len(new)} (= projected new reader calls)")  # fmt: skip
    labels = [b for b, _ in BUCKETS] + [PAST]
    print(f"{'format':12s} {'new':>4s} " + " ".join(f"{b:>8s}" for b in labels) + "  (gold page rank, new only)")
    for f in [*sorted({x["format"] for x in new}), "all"]:
        xs = [x for x in new if f in ("all", x["format"])]
        c = Counter(x["bucket"] for x in xs)
        print(f"{f:12s} {len(xs):4d} " + " ".join(f"{c[b]:8d}" for b in labels))
    out = {"n": len(per), "top_match": top_match, "gate_match": gate_match, "new": len(new),
           "buckets": dict(Counter(x["bucket"] for x in new)), "per_question": per}  # fmt: skip
    for label, hi in (("bound_top10", 10), ("bound_top20", 20)):
        d = [(x["gold_first_ndcg"] if x["new"] and x["rank"] and x["rank"] <= hi else x["agent_ndcg"]) - x["agent_ndcg"]
             for x in per]  # fmt: skip
        mean, lo, hi_ = ci(d)
        out[label] = {"delta": round(mean, 4), "ci": [round(lo, 4), round(hi_, 4)], "gain": sum(v > 0 for v in d)}
        print(f"{label}: {out[label]['gain']} questions gain if their gold page goes first; nDCG@10 vs {agent_run} "
              f"{mean:+.4f} [{lo:+.4f}, {hi_:+.4f}] over n={len(d)}")  # fmt: skip
    if check:
        out["agent_check"] = agent_check(cfg, r, [q.query for q in qs], hit_lists, [ag[q.qid] for q in qs])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="PLAIN:AGENT run names")
    ap.add_argument("--visual", default="pplx-v2-late")
    ap.add_argument("--out", help="write the per-set results as JSON here")
    ap.add_argument("--agent-check", action="store_true", help="run the agent with a refusing client (see above)")
    a = ap.parse_args()
    sys.path[:0] = ["src", "scripts"]
    from docsearch.config import load_config

    cfg = load_config(None)
    out = {}
    for pair in a.pairs:
        plain, agent_run = pair.split(":")
        out[plain] = look(cfg, plain, agent_run, a.visual, a.agent_check)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
