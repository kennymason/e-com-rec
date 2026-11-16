"""Evaluation harness: score one model's recommendations against a split."""

import math

import numpy as np
import polars as pl

from ecomrec.config import RETRIEVAL_K, TOP_K
from ecomrec.data.split import Split
from ecomrec.eval.metrics import ap_at_k, bootstrap_ci, ndcg_at_k, recall_at_k


def evaluate(
    recs: pl.DataFrame,
    split: Split,
    n_catalog: int,
    k: int = TOP_K,
    retrieval_k: int = RETRIEVAL_K,
    n_boot: int = 1000,
) -> dict[str, float | tuple[float, float]]:
    """Score ``recs`` (``customer_idx``, ``article_id`` as a ranked list) on ``split``.

    Customers are those with at least one purchase in the target window; one without a
    recommendation row scores zero. Returned keys (for the default k values):

    - ``recall@100``            retrieval-stage recall
    - ``ndcg@12``, ``map@12``   final-list quality, each with a ``*_ci`` 95% bootstrap CI
    - ``new_item_recall@100``   recall of purchased new items, over customers who bought one
    - ``coverage@12``           share of the catalog that appears in any top-k list
    - ``top1pct_share@12``      share of top-k slots filled by the 1% most-purchased items
    """
    joined = split.targets.join(
        recs.select("customer_idx", pl.col("article_id").alias("recs")),
        on="customer_idx",
        how="left",
    )
    new_items = set(split.new_items.to_list())
    popularity = split.history.group_by("article_id").len().sort("len", descending=True)
    head = set(popularity["article_id"].head(math.ceil(popularity.height / 100)).to_list())

    recall, ndcg, ap, new_recall = [], [], [], []
    recommended: set[int] = set()
    slots = head_slots = 0
    for bought, rec in zip(joined["article_id"].to_list(), joined["recs"].to_list(), strict=True):
        rec = rec or []
        relevant = set(bought)
        recall.append(recall_at_k(rec, relevant, retrieval_k))
        ndcg.append(ndcg_at_k(rec, relevant, k))
        ap.append(ap_at_k(rec, relevant, k))
        relevant_new = relevant & new_items
        if relevant_new:
            new_recall.append(recall_at_k(rec, relevant_new, retrieval_k))
        top = rec[:k]
        recommended.update(top)
        slots += len(top)
        head_slots += sum(item in head for item in top)

    return {
        "n_customers": joined.height,
        f"recall@{retrieval_k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"ndcg@{k}_ci": bootstrap_ci(ndcg, n_boot),
        f"map@{k}": float(np.mean(ap)),
        f"map@{k}_ci": bootstrap_ci(ap, n_boot),
        "n_new_item_customers": len(new_recall),
        f"new_item_recall@{retrieval_k}": float(np.mean(new_recall)) if new_recall else math.nan,
        f"coverage@{k}": len(recommended) / n_catalog,
        f"top1pct_share@{k}": head_slots / slots if slots else math.nan,
    }
