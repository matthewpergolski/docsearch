"""LLM answerability judge for abstention (eval --abstain judge).

The judge sees the question and the top-5 reranked chunks and answers ANSWERABLE or NOT, with the chunk_ids it
relied on. Judgments are cached in work/cache/judge/<provider>.jsonl keyed by (question, chunk ids, model), so a
re-run only calls the LLM for new pairs (resumable). Eval never needs this module unless --abstain judge is given.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import ui
from .config import Config
from .draft_questions import extract_json
from .llm import LLMClient, LLMError

PROMPT = """You check whether a question can be answered from search results.
Below are the top search results (each with a chunk_id). Decide ONLY from these passages, not from general
knowledge: can a careful reader answer the question fully and specifically from them?
Reply with JSON only: {{"verdict": "ANSWERABLE" or "NOT", "chunk_ids": ["<chunk_id the answer comes from>", ...]}}
Use "NOT" if the passages are only on the same topic but lack the specific fact, number, year or entity asked for.

Question: {q}

{passages}
"""


class Judge:
    def __init__(self, cfg: Config, client: LLMClient, k: int = 5, max_chars: int = 1200, cache_only: bool = False,
                 gate: float | None = None):  # fmt: skip
        """cache_only: never call the LLM; a pair not in the cache comes back as missing.
        gate: used by eval --abstain gated (questions whose top rerank score is >= gate skip the judge)."""
        self.client, self.k, self.max_chars, self.cache_only, self.gate = client, k, max_chars, cache_only, gate
        self.missing = 0
        d = cfg.path("cache_dir") / "judge"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{client.s.provider}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        self.calls = self.errors = 0

    def key(self, q: str, ids: list[str]) -> str:
        raw = json.dumps([q, ids, self.client.s.backend, self.client.s.model])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def __call__(self, q: str, hits: list[dict]) -> dict:
        """{"answerable": bool | None, "chunk_ids": [...], "cached": bool}; None = the call failed (not counted)."""
        top = hits[: self.k]
        ids = [h["chunk_id"] for h in top]
        k = self.key(q, ids)
        if k in self.cache:
            return {**self.cache[k], "cached": True}
        if self.cache_only:
            self.missing += 1
            return {"answerable": None, "chunk_ids": [], "cached": False, "missing": True}
        passages = "\n\n".join(f"[chunk_id: {h['chunk_id']}]\n{h['text'][: self.max_chars]}" for h in top)
        try:
            text = self.client.complete(PROMPT.format(q=q, passages=passages))
        except LLMError as e:
            self.errors += 1
            ui.logger.error("judge call failed for %r: %s", q[:80], e)
            return {"answerable": None, "chunk_ids": [], "cached": False}
        self.calls += 1
        res = extract_json(text) or {}
        verdict = str(res.get("verdict", "")).upper()
        if verdict not in ("ANSWERABLE", "NOT"):
            ui.logger.warning("judge reply without a verdict: %s", text[:300])
            return {"answerable": None, "chunk_ids": [], "cached": False}
        row = {"key": k, "answerable": verdict == "ANSWERABLE",
               "chunk_ids": [c for c in res.get("chunk_ids") or [] if c in ids], "model": self.client.s.model}  # fmt: skip
        self.cache[k] = row
        with Path(self.path).open("a") as f:
            f.write(json.dumps(row) + "\n")
        return {**row, "cached": False}


def judge_abstention(per_query: list[dict]) -> dict:
    """Abstain = judge says NOT. Rates over questions the judge answered (failed calls are reported, not scored)."""
    na = [r for r in per_query if not r["gold"] and r.get("judge") is not None]
    ans = [r for r in per_query if r["gold"] and r.get("judge") is not None]
    if not na and not ans:
        return {}
    failed = sum(1 for r in per_query if "judge" in r and r["judge"] is None)
    supported = [r for r in ans if r["judge"] is True]
    cites_gold = [r for r in supported if set(r.get("judge_units") or []) & set(r["gold"])]
    return {"n_no_answer": len(na), "n_answerable": len(ans), "failed_calls": failed,
            "abstain_rate": round(sum(r["judge"] is False for r in na) / len(na), 4) if na else None,
            "false_abstain_rate": round(sum(r["judge"] is False for r in ans) / len(ans), 4) if ans else None,
            "answerable_cites_gold": round(len(cites_gold) / len(supported), 4) if supported else None}  # fmt: skip


def gated(per_query: list[dict], gate: float) -> dict:
    """Gated abstention: answer when the best rerank score >= gate (no judge call); otherwise follow the judge
    (abstain when it says NOT). Needs a judge verdict for every question below the gate; those without one are
    counted in missing_verdicts and left out of the rates, never guessed."""

    def decide(r):
        s = r.get("top_score")
        if s is not None and s >= gate:
            return False, False  # (abstain, judge needed)
        return (r.get("judge") is False) if r.get("judge") is not None else None, True

    na = ans = 0
    abst = fabst = calls = missing = 0
    for r in per_query:
        a, used = decide(r)
        calls += used
        if a is None:
            missing += 1
            continue
        if r["gold"]:
            ans += 1
            fabst += a
        else:
            na += 1
            abst += a
    return {"gate": gate, "abstain_rate": round(abst / na, 4) if na else None,
            "false_abstain_rate": round(fabst / ans, 4) if ans else None, "judge_calls": calls,
            "judge_call_share": round(calls / len(per_query), 4) if per_query else None,
            "missing_verdicts": missing, "n_no_answer": na, "n_answerable": ans}  # fmt: skip


def sweep(
    per_query: list[dict], gates: list[float] | None = None, max_false: float = 0.107
) -> tuple[list[dict], dict | None]:
    """Gate sweep. gates default: every distinct top score (plus +inf = judge only). Best point: highest abstain rate
    with false-abstain <= max_false, ties broken by fewer judge calls."""
    if gates is None:
        sc = sorted({r["top_score"] for r in per_query if r.get("top_score") is not None})
        gates = sc + [float("inf")]
    rows = [gated(per_query, g) for g in gates]
    ok = [r for r in rows if r["false_abstain_rate"] is not None and r["false_abstain_rate"] <= max_false
          and r["abstain_rate"] is not None and r["missing_verdicts"] == 0]  # fmt: skip
    best = max(ok, key=lambda r: (r["abstain_rate"], -r["judge_calls"]), default=None)
    return rows, best
