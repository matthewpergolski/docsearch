"""Offline sweep of the gated-abstention threshold from a judge eval run (no LLM calls).

Uses the per-query file of an `eval --abstain judge` run (results/runs/*-<name>.jsonl), whose verdicts came from
work/cache/judge. Questions without a verdict are reported, never judged. To re-check a chosen gate end to end from
the cache alone: eval ... --abstain gated --gate G --judge-cache-only (missing verdicts are reported, not fetched).

Usage: uv run python scripts/abstain_sweep.py m3_questions_custom_judge [--max-false 0.107] [--gates 0.9 0.95 ...]
Without --gates every distinct top score is tried and the Pareto frontier (abstain vs false-abstain) is printed.
"""

import argparse
import glob
import json
import sys

from docsearch.judge import sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="judge run name, e.g. m3_questions_custom_judge")
    ap.add_argument("--max-false", type=float, default=0.107, help="false-abstain ceiling (threshold method's value)")
    ap.add_argument("--gates", type=float, nargs="*", help="gate values (default: a grid over observed scores)")
    a = ap.parse_args()
    files = sorted(glob.glob(f"results/runs/*-{a.run}.jsonl"))
    if not files:
        print(f"no per-query file for run {a.run!r} in results/runs/", file=sys.stderr)
        return 1
    with open(files[-1]) as f:
        pq = [json.loads(line) for line in f]
    have = [r for r in pq if "judge" in r]
    if not have:
        print(f"{files[-1]} has no judge verdicts; run eval --abstain judge first", file=sys.stderr)
        return 1
    missing = [r["qid"] for r in pq if r.get("judge") is None]
    gates = a.gates  # default: every distinct top score (exact optimum), printed as the Pareto frontier
    rows, best = sweep(pq, gates, a.max_false)
    print(
        f"run {a.run}: {len(pq)} questions ({sum(not r['gold'] for r in pq)} no-answer), "
        f"verdicts missing for {len(missing)}{': ' + ', '.join(missing[:5]) if missing else ''}"
    )
    print(f"{'gate (top score >=)':>20} {'abstain':>8} {'false-abstain':>14} {'judge calls':>12} {'missing':>8}")
    judge_only = rows[-1]
    if a.gates is None:  # keep points not dominated (higher abstain at equal-or-lower false-abstain)
        rows = [r for r in rows if r["abstain_rate"] is not None and not any(
            o["abstain_rate"] > r["abstain_rate"] and o["false_abstain_rate"] <= r["false_abstain_rate"]
            or o["abstain_rate"] == r["abstain_rate"] and o["false_abstain_rate"] < r["false_abstain_rate"]
            for o in rows if o["abstain_rate"] is not None)]  # fmt: skip
        rows = [r for i, r in enumerate(rows) if i == 0 or (r["abstain_rate"], r["false_abstain_rate"])
                != (rows[i - 1]["abstain_rate"], rows[i - 1]["false_abstain_rate"]) or r is best]  # fmt: skip
        if rows[-1] is not judge_only:
            rows.append(judge_only)  # reference: judge on every question
    for r in rows:
        g = "judge only" if r["gate"] == float("inf") else f"{r['gate']:.4f}"
        mark = "  <- best" if best is r else ""
        print(
            f"{g:>20} {r['abstain_rate']:>8.3f} {r['false_abstain_rate']:>14.3f} "
            f"{r['judge_calls']:>5d} ({r['judge_call_share']:.0%}) {r['missing_verdicts']:>8d}{mark}"
        )
    if best:
        print(
            f"best: gate {best['gate']} -> abstain {best['abstain_rate']}, false-abstain {best['false_abstain_rate']}"
            f" (<= {a.max_false}), judge on {best['judge_call_share']:.0%} of questions"
        )
    else:
        print(f"no gate keeps false-abstain <= {a.max_false} with full verdict coverage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
