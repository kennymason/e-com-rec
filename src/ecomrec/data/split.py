"""Time-based, leakage-safe splits.

A split is defined by a single cutoff date. Everything strictly before the cutoff is
history (the only data models and features may see); the ``horizon_days`` starting at the
cutoff are the targets. There are no random splits anywhere in this project.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from ecomrec.config import HORIZON_DAYS


@dataclass(frozen=True)
class Split:
    cutoff: date
    horizon_days: int
    history: pl.DataFrame
    """Transactions with ``t_dat < cutoff``."""
    targets: pl.DataFrame
    """One row per customer who bought in the target window:
    ``customer_idx``, ``article_id`` (list of distinct articles bought)."""
    new_items: pl.Series
    """Articles sold in the target window that have no sale in ``history``."""


def window_cutoffs(max_date: date, horizon_days: int = HORIZON_DAYS) -> dict[str, date]:
    """Cutoffs for the standard windows: test is the final week, val the week before."""
    test = max_date - timedelta(days=horizon_days - 1)
    return {"val": test - timedelta(days=horizon_days), "test": test}


def make_split(transactions: pl.DataFrame, cutoff: date, horizon_days: int = HORIZON_DAYS) -> Split:
    end = cutoff + timedelta(days=horizon_days)
    history = transactions.filter(pl.col("t_dat") < cutoff)
    window = transactions.filter((pl.col("t_dat") >= cutoff) & (pl.col("t_dat") < end))
    if window.is_empty():
        raise ValueError(f"no transactions in target window [{cutoff}, {end})")

    targets = (
        window.group_by("customer_idx")
        .agg(pl.col("article_id").unique().sort())
        .sort("customer_idx")
    )
    seen = history["article_id"].unique()
    new_items = window["article_id"].unique().sort()
    new_items = new_items.filter(~new_items.is_in(seen.implode()))
    return Split(cutoff, horizon_days, history, targets, new_items)
