"""Ranking metrics over de-duplicated unit lists (pages or files)."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], gold: dict[str, int], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def ndcg_at_k(ranked: list[str], gold: dict[str, int], k: int = 10) -> float:
    """Graded nDCG with gain 2^rel - 1."""
    dcg = sum((2 ** gold.get(u, 0) - 1) / math.log2(i + 2) for i, u in enumerate(ranked[:k]))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], gold: dict[str, int], k: int = 10) -> float:
    for i, u in enumerate(ranked[:k]):
        if gold.get(u, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked: list[str], gold: dict[str, int], k: int = 10) -> float:
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def score_query(ranked: list[str], gold: dict[str, int], k: int = 10) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(ranked, gold, k),
        f"ndcg@{k}": ndcg_at_k(ranked, gold, k),
        f"mrr@{k}": mrr_at_k(ranked, gold, k),
        f"hit@{k}": hit_at_k(ranked, gold, k),
        f"all@{k}": 1.0 if gold and set(gold) <= set(ranked[:k]) else 0.0,  # every gold unit in top k (multi-doc)
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
