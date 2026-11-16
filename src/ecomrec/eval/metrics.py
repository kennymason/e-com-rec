"""Ranking metrics for a single customer, plus a bootstrap confidence interval.

All metrics take a ranked recommendation list and the set of articles the customer
actually bought. A repeated recommendation only counts the first time it appears.
"""

import math
from collections.abc import Collection, Sequence

import numpy as np


def _hits(recs: Sequence[int], relevant: Collection[int], k: int) -> list[bool]:
    seen: set[int] = set()
    hits = []
    for item in recs[:k]:
        hits.append(item in relevant and item not in seen)
        seen.add(item)
    return hits


def recall_at_k(recs: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Share of the relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    return sum(_hits(recs, relevant, k)) / len(relevant)


def ap_at_k(recs: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Average precision at k, as defined for the Kaggle H&M competition (MAP@12):
    the sum of precision at each hit position, divided by ``min(len(relevant), k)``."""
    if not relevant:
        return 0.0
    n_hits = 0
    total = 0.0
    for i, hit in enumerate(_hits(recs, relevant, k)):
        if hit:
            n_hits += 1
            total += n_hits / (i + 1)
    return total / min(len(relevant), k)


def ndcg_at_k(recs: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Binary-relevance NDCG at k."""
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(_hits(recs, relevant, k)) if hit)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-customer metric values."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, values.size, values.size)].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))
