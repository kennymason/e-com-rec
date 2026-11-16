"""Rung 1: global popularity over the last few days before the cutoff."""

from datetime import date, timedelta

import polars as pl


class Popularity:
    def __init__(self, window_days: int = 7) -> None:
        self.window_days = window_days
        self.ranked: list[int] = []

    def fit(self, history: pl.DataFrame, cutoff: date) -> "Popularity":
        start = cutoff - timedelta(days=self.window_days)
        self.ranked = (
            history.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") < cutoff))
            .group_by("article_id")
            .len()
            .sort(["len", "article_id"], descending=[True, False])["article_id"]
            .to_list()
        )
        return self

    def recommend(self, customers: pl.Series, n: int) -> pl.DataFrame:
        top = self.ranked[:n]
        return pl.DataFrame(
            {"customer_idx": customers, "article_id": [top] * len(customers)},
            schema_overrides={"article_id": pl.List(pl.Int32)},
        )
